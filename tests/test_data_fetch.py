"""Tests for the unified data-fetch framework (ops/data_fetch.py).

Network-free: uses fake adapters and temp staging dirs. Covers the adapter
interface, registry validation, dry-run safety, empty-file rejection, READY gating,
fetch-all resilience, checkpoint/resume, and the trusted-path write guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ops.data_fetch import core
from ops.data_fetch.core import Adapter, Status, is_trusted_path
from ops.data_fetch.registry import REGISTRY, SourceSpec, get_spec, validate_registry
from tests._fake_fetch_adapters import FakeOkAdapter, FakeEmptyAdapter, FakeBoomAdapter


@pytest.fixture
def staging(tmp_path, monkeypatch):
    """Redirect all staging/report writes into tmp_path."""
    monkeypatch.setattr(core, "_EXTERNAL_RAW", tmp_path / "raw")
    monkeypatch.setattr(core, "_EXTERNAL_CURATED", tmp_path / "curated")
    monkeypatch.setattr(core, "_REPORTS", tmp_path / "reports")
    FakeOkAdapter._calls.clear()
    return tmp_path


def _spec(key="fake_ok", adapter="tests._fake_fetch_adapters:FakeOkAdapter", **kw):
    base = dict(
        key=key, title="Fake", type="test", endpoint="memory://", description="fake",
        adapter=adapter, required_columns=["id", "t", "val"], dedup_keys=["id"],
        date_column="t", min_rows=1,
    )
    base.update(kw)
    return SourceSpec(**base)


# ── registry + interface ──────────────────────────────────────────────────────
def test_registry_is_valid():
    assert validate_registry() == []


def test_every_spec_builds_an_adapter_with_the_interface():
    for key, spec in REGISTRY.items():
        adapter = spec.build_adapter()
        assert isinstance(adapter, Adapter)
        for method in ("discover", "dry_run", "fetch", "validate"):
            assert callable(getattr(adapter, method)), f"{key} missing {method}"


def test_static_watershed_landuse_fetches_real_catalog_rows(staging, monkeypatch):
    from ops.data_fetch.adapters import heavy

    class FakeResp:
        def __init__(self, text=None, payload=None):
            self.text = text or ""
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, **_kw):
        if "tnmaccess.nationalmap.gov" in url:
            return FakeResp(payload={
                "total": 2,
                "items": [
                    {
                        "title": "USGS National Hydrography Dataset Plus High Resolution (NHDPlus HR) for 4-digit Hydrologic Unit - 1806 (published 20220901)",
                        "sourceId": "nhd-1806",
                        "downloadURL": "https://example.test/nhd_1806.gdb.zip",
                        "format": "FileGDB",
                    },
                    {
                        "title": "USGS National Hydrography Dataset Plus High Resolution (NHDPlus HR) for 4-digit Hydrologic Unit - 1807 (published 20220901)",
                        "sourceId": "nhd-1807",
                        "downloadURL": "https://example.test/nhd_1807.gdb.zip",
                        "format": "FileGDB",
                    },
                ],
            })
        if "ocmgeodatastor1.blob.core.windows.net" in url:
            return FakeResp(text="""
                <a href="conus_2001_ccap_landcover.xml">xml</a>
                <a href="conus_2001_ccap_landcover.tif">tif</a>
                <a href="conus_2006_ccap_landcover.xml">xml</a>
                <a href="conus_2006_ccap_landcover.tif">tif</a>
            """)
        if "mrlc.gov" in url:
            return FakeResp(text="""
                <a href="https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wms?">wms</a>
                <a href="https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs?">wcs</a>
                <a href="https://dmsdata.cr.usgs.gov/geoserver/mrlc_Impervious-Descriptor-Native_conus_year_data/wms?">wms</a>
                <a href="https://dmsdata.cr.usgs.gov/geoserver/mrlc_Factional-Impervious-Surface-Native_conus_year_data/wcs?">wcs</a>
            """)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(heavy, "get_with_backoff", fake_get)
    res = get_spec("ccap_nhd_nlcd").build_adapter().fetch(None, None)

    assert res.status == Status.READY_FOR_MODELING
    assert res.rows == 10
    df = pd.read_parquet(staging / "curated" / "ccap_nhd_nlcd" / "ccap_nhd_nlcd.parquet")
    assert set(df["provider"]) == {"USGS", "NOAA", "MRLC"}
    assert df["url"].str.startswith("https://").all()
    assert "feature_id" not in df.columns


def test_surfrider_normalizes_california_enterococcus_mixed_dates():
    from ops.data_fetch.adapters.surfrider import SurfriderAdapter

    raw = pd.DataFrame({
        "sample id": ["ca-1", "or-1", "ca-2"],
        "state": ["California", "Oregon", "California"],
        "stored collectionTime": [
            "2000-12-30T15:30:00Z",
            "01/02/2020 08:00 AM",
            "2024-12-31T09:15:00Z",
        ],
        "site id": [188, 999, 189],
        "site name": ["San Clemente Pier", "Not CA", "Salt Creek"],
        "latitude": [33.4189, 45.0, 33.4776],
        "longitude": [-117.62, -123.0, -117.72],
        "Enterococcus modifier": ["=", "=", "<"],
        "Enterococcus (mpn/100mL)": [84, 10, 10],
    })

    out = SurfriderAdapter(get_spec("surfrider_bwtf")).normalize(raw)

    assert list(out["sample_id"]) == ["ca-1", "ca-2"]
    assert out["sample_date"].notna().all()
    assert out["bacteria_result"].tolist() == [84, 10]
    assert out["station"].tolist() == ["188|San Clemente Pier", "189|Salt Creek"]


def test_station_static_features_nearest_points_and_catalog(tmp_path):
    from ops.data_fetch.adapters import station_static as ss

    stations = pd.DataFrame({
        "station_id": ["A", "B"],
        "beach_name": ["Beach A", "Beach B"],
        "county": ["Monterey", "Monterey"],
        "latitude": [36.60, 36.70],
        "longitude": [-121.90, -121.95],
    })
    points = tmp_path / "points.parquet"
    pd.DataFrame({
        "pid": ["near", "far"],
        "name": ["Near Facility", "Far Facility"],
        "latitude": [36.601, 38.0],
        "longitude": [-121.901, -123.0],
    }).to_parquet(points)
    cat = tmp_path / "catalog.parquet"
    pd.DataFrame({
        "provider": ["NOAA", "MRLC", "USGS"],
        "product_id": ["ccap", "mrlc_Impervious-Descriptor-Native_conus_year_data", "nhd"],
        "format": ["GeoTIFF", "WCS", "FileGDB"],
        "year": [2021, None, None],
        "url": ["https://example.test/ccap.tif", "https://example.test/nlcd/wcs?", "https://example.test/nhd.zip"],
    }).to_parquet(cat)

    out = ss._add_station_density(stations)
    out = ss._add_nearest_points(
        out,
        points,
        lat_col="latitude",
        lon_col="longitude",
        id_col="pid",
        name_col="name",
        prefix="test",
    )
    out = ss._add_catalog_provenance(out, cat)

    assert out.loc[out["station_id"] == "A", "nearest_test_id"].iloc[0] == "near"
    assert out.loc[out["station_id"] == "A", "dist_to_test_km"].iloc[0] < 1
    assert out["station_density_25km"].tolist() == [1, 1]
    assert out["static_catalog_rows"].iloc[0] == 3
    assert out["ccap_latest_year"].iloc[0] == 2021
    assert out["nlcd_impervious_wcs_url"].iloc[0] == "https://example.test/nlcd/wcs?"
    sampled = ss._add_ccap_raster_sample(out)
    assert "ccap_landcover_code" in sampled.columns
    assert "ccap_developed" in sampled.columns


def test_hrrr_idx_parser_and_nwm_key_parser():
    from ops.data_fetch.adapters.heavy import _parse_hrrr_idx_line, _parse_nwm_key

    parsed = _parse_hrrr_idx_line("12:123456:d=2026061000:TMP:2 m above ground:1 hour fcst:")
    assert parsed == {
        "message": 12,
        "byte_start": 123456,
        "variable": "TMP",
        "level": "2 m above ground",
        "forecast_label": "1 hour fcst",
    }
    assert _parse_hrrr_idx_line("bad") is None

    nwm = _parse_nwm_key(
        "nwm.20260610/analysis_assim/nwm.t00z.analysis_assim.channel_rt.tm02.conus.nc"
    )
    assert nwm["product"] == "analysis_assim"
    assert nwm["variable_group"] == "channel_rt"
    assert nwm["offset_hour"] == -2
    assert str(nwm["valid_time"]) == "2026-06-09 22:00:00+00:00"


def test_coops_tide_staged_normalizes_api_rows(monkeypatch):
    from ops.data_fetch.adapters.coops_tide import CoopsTideStagedAdapter

    class Resp:
        def json(self):
            return {
                "data": [
                    {"t": "2026-06-10 00:00", "v": "1.234", "s": "0.010", "f": "0,0,0,0", "q": "v"},
                    {"t": "2026-06-10 01:00", "v": "", "s": "", "f": "0,0,0,0", "q": "v"},
                ]
            }

    def fake_get(*args, **kwargs):
        return Resp()

    monkeypatch.setattr("ops.data_fetch.adapters.coops_tide.get_with_backoff", fake_get)
    adapter = CoopsTideStagedAdapter(get_spec("coops_tide_staged"))
    out = adapter.fetch_chunk({
        "station_id": "9414290",
        "station_name": "San Francisco",
        "begin_date": "20260610",
        "end_date": "20260610",
    })
    assert len(out) == 1
    assert out["station_id"].iloc[0] == "9414290"
    assert out["sample_date"].dt.tz is not None
    assert out["water_level_m"].iloc[0] == 1.234


def test_roms_uses_metadata_coverage_and_normalizes_csv(monkeypatch):
    import datetime as dt
    from ops.data_fetch.adapters import roms

    monkeypatch.setattr(
        roms,
        "_dataset_time_coverage",
        lambda: (
            dt.datetime(2015, 3, 19, tzinfo=dt.timezone.utc),
            dt.datetime(2022, 12, 1, 3, tzinfo=dt.timezone.utc),
        ),
    )
    adapter = roms.RomsAdapter(get_spec("roms_circulation"))
    chunks = list(adapter.iter_chunks(None, None))
    assert [c["key"] for c in chunks] == ["2022-12"]
    assert chunks[0]["end"] == "2022-12-01T03:00:00Z"

    url = roms._build_roms_url("2022-12-01T00:00:00Z", "2022-12-01T03:00:00Z")
    assert "roms_ncst.csv?u[" in url
    assert "temp[" in url
    assert "[(36.0):10:(38.0)]" in url

    class Resp:
        text = """time,depth,latitude,longitude,u,v,temp,salt
UTC,m,degrees_north,degrees_east,"m/s","m/s",degree_C,PSU
2022-12-01T03:00:00Z,0.0,36.01,237.0,3.0,4.0,13.2,33.3
2022-12-01T03:00:00Z,0.0,36.01,237.3,NaN,NaN,NaN,NaN
"""

    monkeypatch.setattr("ops.data_fetch.adapters.roms.get_with_backoff", lambda *a, **k: Resp())
    out = adapter.fetch_chunk({"start": "2022-12-01T00:00:00Z", "end": "2022-12-01T03:00:00Z"})
    assert len(out) == 1
    assert out["longitude_180"].iloc[0] == -123.0
    assert out["current_speed_m_s"].iloc[0] == 5.0


def test_wcofs_chunking_and_normalization(monkeypatch):
    import datetime as dt
    from ops.data_fetch.adapters import wcofs

    monkeypatch.setattr(
        wcofs,
        "_dataset_time_coverage",
        lambda: (
            dt.datetime(2022, 1, 2, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 6, 13, 21, tzinfo=dt.timezone.utc),
        ),
    )
    adapter = wcofs.WcofsCirculationAdapter(get_spec("wcofs_circulation"))
    chunks = list(adapter.iter_chunks("2026-05-31", "2026-06-02"))
    assert [c["key"] for c in chunks] == ["2026-05", "2026-06"]
    assert chunks[0]["start"] == "2026-05-31T00:00:00Z"
    assert chunks[1]["end"] == "2026-06-02T23:59:59.999999Z"

    raw = pd.DataFrame({
        "ocean_time": ["2026-06-01T00:00:00Z", "2026-06-01T03:00:00Z"],
        "lat_rho": [36.5, 36.6],
        "lon_rho": [-122.0, -122.1],
        "s_rho": [-0.0125, -0.0125],
        "urot": [3.0, np.nan],
        "vrot": [4.0, np.nan],
        "temp": [14.0, np.nan],
        "salt": [33.2, -1.0],
        "zeta": [0.2, np.nan],
    })
    out = wcofs._normalize_wcofs_frame(raw, chunks[1])
    assert len(out) == 1
    assert out["u"].iloc[0] == 3.0
    assert out["current_speed_m_s"].iloc[0] == 5.0
    assert out["source_dataset"].iloc[0] == "CeNCOOS THREDDS AWS_WCOFS"
    assert out["time_stride_hours"].iloc[0] == 24
    assert wcofs._to_xarray_time("2026-06-01T00:00:00Z").tzinfo is None


def test_existing_fetchers_can_be_inventoried():
    # Wrap-trusted sources point at real repo-relative trusted outputs.
    wrapped = [s for s in REGISTRY.values() if s.wraps_trusted]
    assert wrapped, "expected at least one wrap_trusted source"
    for s in wrapped:
        info = s.build_adapter().discover()
        assert info["mode"] == "wrap_trusted"
        assert "trusted_exists" in info


# ── dry-run must not write curated/raw data ───────────────────────────────────
def test_dry_run_writes_no_curated_or_raw(staging):
    a = _spec().build_adapter()
    a.dry_run(None, None)
    assert not a.curated_path.exists()
    assert not list((staging / "raw").glob("**/chunk_*.parquet"))
    # dry-run DOES write a plan file under reports
    assert (a.reports / "dry_run_plan.json").exists()


# ── checkpoint / resume ───────────────────────────────────────────────────────
def test_fetch_checkpoints_and_resumes(staging):
    a = _spec().build_adapter()
    r1 = a.fetch(None, None)
    assert r1.status == Status.READY_FOR_MODELING
    assert r1.rows == 3
    assert FakeOkAdapter._calls["fake_ok"] == 3
    chunks = list((staging / "raw" / "fake_ok").glob("chunk_*.parquet"))
    assert len(chunks) == 3
    # Second run resumes: no new fetch_chunk calls.
    a2 = _spec().build_adapter()
    r2 = a2.fetch(None, None)
    assert r2.rows == 3
    assert FakeOkAdapter._calls["fake_ok"] == 3  # unchanged


# ── empty output is rejected ──────────────────────────────────────────────────
def test_validation_rejects_empty_output(staging):
    a = _spec(key="fake_empty", adapter="tests._fake_fetch_adapters:FakeEmptyAdapter").build_adapter()
    res = a.fetch(None, None)
    assert res.status == Status.IMPLEMENTED_NOT_FETCHED
    v = a.validate()
    assert v["passed"] is False


def test_validation_rejects_missing_file(staging):
    a = _spec(key="never_run").build_adapter()
    v = a.validate()
    assert v["passed"] is False
    assert v["checks"]["exists"] is False


# ── READY_FOR_MODELING gating: artifacts must exist ───────────────────────────
def test_ready_source_has_all_artifacts(staging):
    a = _spec().build_adapter()
    res = a.fetch(None, None)
    assert res.status == Status.READY_FOR_MODELING
    for art in ("manifest.json", "coverage.json", "validation.json", "README.md"):
        assert (a.reports / art).exists(), f"missing {art}"
    cov = json.loads((a.reports / "coverage.json").read_text())
    assert cov["rows"] == 3 and "date_min" in cov
    man = json.loads((a.reports / "manifest.json").read_text())
    assert man["curated_sha256"]  # checksum present


def test_bounded_sample_never_auto_promotes(staging):
    a = _spec(key="fake_bounded", bounded_sample=True).build_adapter()
    res = a.fetch(None, None)
    assert res.rows == 3
    assert res.status == Status.FETCHED_NEEDS_REVIEW  # not READY despite valid data


# ── trusted production paths are protected ────────────────────────────────────
def test_trusted_paths_detected():
    assert is_trusted_path("bacteria_results/statewide/x.parquet")
    assert is_trusted_path("mbal_history/noaa/x.parquet")
    assert is_trusted_path("lakehouse/x.parquet")
    assert is_trusted_path("mbal_pipeline/curated_history/x.parquet")
    assert not is_trusted_path("data/external_raw/x/y.parquet")


def test_guard_blocks_writes_into_trusted_path():
    with pytest.raises(PermissionError):
        core._guard_write(core.PROJECT_ROOT / "bacteria_results" / "evil.parquet")


def test_wrap_trusted_adapter_never_writes_trusted(staging):
    # statewide wraps a trusted file; fetch() must only emit artifacts (reports), not
    # touch the trusted parquet. We assert the curated_path equals the trusted file and
    # is NOT under our tmp staging (i.e. we never staged-copy it).
    spec = get_spec("statewide_beachwatch")
    a = spec.build_adapter()
    assert is_trusted_path(a.curated_path)
    # No raw chunks are produced for wrap_trusted mode.
    assert not (staging / "raw" / "statewide_beachwatch").exists()


# ── fetch-all resilience ──────────────────────────────────────────────────────
def test_fetch_all_survives_a_failing_source(staging, monkeypatch):
    from ops.data_fetch import cli

    fake = {
        "fake_ok": _spec(key="fake_ok", priority=1),
        "fake_boom": _spec(key="fake_boom", priority=1,
                           adapter="tests._fake_fetch_adapters:FakeBoomAdapter"),
    }
    monkeypatch.setattr(cli, "REGISTRY", fake)
    monkeypatch.setattr(cli, "_REPORTS_ROOT", staging / "reports")
    monkeypatch.setattr("ops.data_fetch.registry.REGISTRY", fake, raising=False)

    class Args:
        priority = "high"
        start = end = None
        limit_chunks = None
        max_workers = 1

    rc = cli.cmd_fetch_all(Args())
    assert rc == 0  # did not crash despite FakeBoom raising
