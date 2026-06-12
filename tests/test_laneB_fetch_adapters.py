"""Offline tests for lane-B data-fetch adapters (agent: claude-fetch-2).

Network-free: they exercise the pure transforms (NDBC fixed-width parse, CEDEN CKAN
normalize) and assert registry wiring / spec validity. The actual fetched row counts
are validated separately by the framework's validation.json (>=50k gate).
"""
from __future__ import annotations

import pandas as pd

from ops.data_fetch.registry import REGISTRY, get_spec, validate_registry
from ops.data_fetch.registry_laneB import LANE_B_SPECS
from ops.data_fetch.adapters.ndbc import NdbcStdmetAdapter
from ops.data_fetch.adapters.ceden import CedenWaterChemAdapter

LANE_B_KEYS = ["ceden_water_chem", "ndbc_stdmet", "usgs_iv_turbidity",
               "nasa_power_daily", "ceden_toxicity", "gridmet_daily",
               "open_meteo_archive",
               # round 2
               "usgs_dv_statewide", "cdip_wave_network", "cdip_sst_network",
               "ndbc_cwind", "ncei_isd_hourly"]


def test_lane_b_specs_registered_and_valid():
    assert not validate_registry(), validate_registry()
    for key in LANE_B_KEYS:
        spec = get_spec(key)
        assert spec.min_rows >= 50_000, f"{key} must target >=50k rows"
        assert spec.required_columns and spec.date_column
        assert ":" in spec.adapter


def test_lane_b_adapters_build_and_plan_chunks():
    for key in LANE_B_KEYS:
        adapter = get_spec(key).build_adapter()
        # iter_chunks must yield filesystem-safe keyed descriptors without network
        # (NDBC / NASA are static plans; USGS hits the site service so skip its plan).
        if key in ("ndbc_stdmet", "nasa_power_daily"):
            chunks = list(adapter.iter_chunks(None, None))
            assert chunks and all("key" in c for c in chunks)


def test_ndbc_parse_fixedwidth_and_sentinels():
    text = (
        "#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE\n"
        "#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft\n"
        "2015 06 01 00 00 280  5.0  6.0  1.20   9.0   7.0 290 1015.0  13.0  14.5 999.0 99.0 99.00\n"
        "2015 06 01 01 00 270 99.0 99.0 99.00  99.0  99.0 999 9999.0 999.0 999.0 999.0 99.0 99.00\n"
    )
    df = NdbcStdmetAdapter._parse(text, "46042")
    assert len(df) == 2
    assert {"station_id", "time", "WTMP", "WVHT"}.issubset(df.columns)
    # row 0 has real values, row 1 is all-sentinel -> NaN
    assert df.iloc[0]["WTMP"] == 14.5
    assert df.iloc[0]["WSPD"] == 5.0
    assert pd.isna(df.iloc[1]["WSPD"])
    assert pd.isna(df.iloc[1]["WTMP"])
    assert pd.isna(df.iloc[0]["DEWP"])  # 999.0 sentinel on the otherwise-good row
    assert str(df.iloc[0]["time"]) == "2015-06-01 00:00:00"


def test_ndbc_parse_empty_or_html_is_safe():
    assert NdbcStdmetAdapter._parse("", "46042").empty
    assert NdbcStdmetAdapter._parse("<html>not found</html>", "46042").empty


def test_ceden_normalize_types_and_columns():
    raw = pd.DataFrame.from_records([
        {"StationCode": "HGA", "StationName": "HGA", "SampleDate": "2017-04-18T00:00:00",
         "CollectionTime": "00:00", "Analyte": "Nitrate", "AnalyteCode": "NO3",
         "Unit": "mg/L", "Result": "1.25", "Latitude": "36.6", "Longitude": "-121.9",
         "MethodName": "EPA 353.2", "CollectionReplicate": 1, "ResultsReplicate": 1},
    ])
    out = CedenWaterChemAdapter(get_spec("ceden_water_chem")).normalize(raw)
    for col in ("StationCode", "SampleDate", "Analyte", "Result"):
        assert col in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["SampleDate"])
    assert float(out.iloc[0]["Result"]) == 1.25
    assert float(out.iloc[0]["Latitude"]) == 36.6


def test_ceden_normalize_empty_returns_schema():
    out = CedenWaterChemAdapter(get_spec("ceden_water_chem")).normalize(pd.DataFrame())
    assert out.empty
    assert "StationCode" in out.columns


def test_ndbc_consolidate_clean_helper_nans_impossible():
    # the physical-range cleaner must NaN sensor-failure values (e.g. PRES==0.0)
    from ops.data_fetch.adapters.ndbc import _clean_physical
    df = pd.DataFrame({"PRES": [1013.2, 0.0, 950.0], "WTMP": [14.0, -999.0, 12.0]})
    out = _clean_physical(df.copy())
    assert pd.isna(out["PRES"].iloc[1])          # 0.0 hPa -> NaN
    assert out["PRES"].iloc[0] == 1013.2          # valid kept
    assert pd.isna(out["WTMP"].iloc[1])          # -999 -> NaN


def test_open_meteo_archive_plans_grid_chunks():
    from ops.data_fetch.registry import SourceSpec
    from ops.data_fetch.adapters.open_meteo_archive import OpenMeteoArchiveAdapter
    spec = SourceSpec(key="open_meteo_archive", title="t", type="tabular_api", endpoint="e",
                      description="d",
                      adapter="ops.data_fetch.adapters.open_meteo_archive:OpenMeteoArchiveAdapter",
                      required_columns=["grid_lat", "grid_lon", "time", "parameter", "value"],
                      date_column="time")
    chunks = list(OpenMeteoArchiveAdapter(spec).iter_chunks(None, None))
    assert len(chunks) >= 10 and all({"key", "lat", "lon"} <= set(c) for c in chunks)


def test_ndbc_cwind_parse():
    from ops.data_fetch.adapters.ndbc_cwind import NdbcCwindAdapter
    text = (
        "#YY  MM DD hh mm WDIR WSPD GDR GST GTIME\n"
        "#yr  mo dy hr mn degT m/s degT m/s hhmm\n"
        "2015 06 01 00 00 280  5.0 290  7.0 0010\n"
        "2015 06 01 00 10 999 99.0 999 99.0 9999\n"
    )
    df = NdbcCwindAdapter._parse(text, "46042")
    assert len(df) == 2
    assert {"station_id", "time", "WSPD", "WDIR"}.issubset(df.columns)
    assert df.iloc[0]["WSPD"] == 5.0 and df.iloc[0]["WDIR"] == 280
    assert pd.isna(df.iloc[1]["WSPD"])          # 99.0 sentinel -> NaN
    assert str(df.iloc[0]["time"]) == "2015-06-01 00:00:00"


def test_ncei_isd_coded_parser():
    from ops.data_fetch.adapters.ncei_isd import _coded
    assert _coded("+0061,1") == 6.1            # temp 6.1 C
    assert _coded("10132,1") == 1013.2          # SLP
    assert _coded("+9999,9") is None            # missing
    assert _coded("99999") is None
    assert _coded("") is None


def test_run_buildout_select_and_import():
    from ops.data_fetch import run_buildout
    assert run_buildout.BUILDOUT_SOURCES, "buildout source list must be non-empty"

    class _A:  # minimal args
        sources = "ndbc_stdmet,nasa_power_daily"
        priority = None
    assert run_buildout._select(_A()) == ["ndbc_stdmet", "nasa_power_daily"]
