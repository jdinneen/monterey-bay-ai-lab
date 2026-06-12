#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import fetch_chlorophyll as fch  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, timeout: int) -> FakeResponse:
        self.urls.append(url)
        return self.responses.pop(0)


CSV_TEXT = """time,altitude,latitude,longitude,chla
UTC,m,degrees_north,degrees_east,mg m^-3
2024-01-01T12:00:00Z,0.0,36.75375,-122.02875,1.7
2024-01-02T12:00:00Z,0.0,36.75375,-122.02875,NaN
"""


def test_date_chunks_respects_window_size():
    chunks = fch.date_chunks(dt.date(2024, 1, 1), dt.date(2024, 1, 5), chunk_days=2)

    assert chunks == [
        (dt.date(2024, 1, 1), dt.date(2024, 1, 2)),
        (dt.date(2024, 1, 3), dt.date(2024, 1, 4)),
        (dt.date(2024, 1, 5), dt.date(2024, 1, 5)),
    ]


def test_build_chla_url_encodes_single_validation_point():
    cfg = fch.FetchConfig(erddap_base_url="https://example.test/erddap", lat=36.7511, lon=-122.0292)

    url = fch.build_chla_url(dt.date(2024, 1, 1), dt.date(2024, 1, 3), cfg)

    assert url.startswith("https://example.test/erddap/griddap/erdVHNchla1day.csv?chla")
    assert "2024-01-01T12:00:00Z" in url
    assert "2024-01-03T12:00:00Z" in url
    assert "36.7511" in url
    assert "-122.0292" in url


def test_normalize_chla_csv_keeps_grid_coordinates_and_numeric_chla():
    df = fch.normalize_chla_csv(CSV_TEXT)

    assert list(df.columns) == ["time", "latitude", "longitude", "chla_m1_mg_m3"]
    assert len(df) == 2
    assert df["time"].dt.tz is not None
    assert df.loc[0, "chla_m1_mg_m3"] == 1.7
    assert pd.isna(df.loc[1, "chla_m1_mg_m3"])
    assert df.loc[0, "latitude"] == 36.75375


def test_fetch_year_writes_resumable_chunks_and_year_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(fch, "year_bounds", lambda year: ("2024-01-01", "2024-01-03"))
    client = FakeClient([FakeResponse(200, CSV_TEXT), FakeResponse(200, CSV_TEXT)])
    cfg = fch.FetchConfig(out_dir=tmp_path, chunk_days=2, delay_seconds=0)

    done, retry_after = fch.fetch_year(2024, cfg, client=client)

    assert done is True
    assert retry_after is None
    assert len(client.urls) == 2
    chunk_files = sorted((tmp_path / "_chunks").glob("*.parquet"))
    assert [p.name for p in chunk_files] == [
        "chla_2024_20240101_20240102.parquet",
        "chla_2024_20240103_20240103.parquet",
    ]
    yearly = pd.read_parquet(tmp_path / "chla_2024.parquet")
    assert len(yearly) == 2
    assert yearly["time"].is_monotonic_increasing
    metadata = json.loads((tmp_path / "chla_2024.metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["dataset_id"] == "erdVHNchla1day"
    assert metadata["details"]["chunk_count"] == 2


def _run_without_pytest() -> int:
    import tempfile

    tests = [
        test_date_chunks_respects_window_size,
        test_build_chla_url_encodes_single_validation_point,
        test_normalize_chla_csv_keeps_grid_coordinates_and_numeric_chla,
    ]
    for test in tests:
        test()
    with tempfile.TemporaryDirectory() as tmp:
        class MonkeyPatch:
            def setattr(self, obj, name, value):
                setattr(obj, name, value)

        test_fetch_year_writes_resumable_chunks_and_year_metadata(Path(tmp), MonkeyPatch())
    print("4 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
