#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import autonomous_rate_limit_fetcher as fetcher  # noqa: E402


def test_mur_sst_url_clamps_first_year_to_dataset_start(tmp_path):
    cfg = fetcher.FetchConfig(out_dir=tmp_path)

    url = fetcher.build_mur_sst_url(2002, cfg, today=dt.date(2026, 6, 8))

    assert "2002-06-01T09:00:00Z" in url
    assert "2002-12-31T09:00:00Z" in url
    assert "36.7511" in url
    assert "-122.0292" in url


def test_mur_sst_url_uses_yesterday_for_current_year(tmp_path):
    cfg = fetcher.FetchConfig(out_dir=tmp_path)

    url = fetcher.build_mur_sst_url(2026, cfg, today=dt.date(2026, 6, 8))

    assert "2026-01-01T09:00:00Z" in url
    assert "2026-06-07T09:00:00Z" in url


def test_mur_sst_url_current_year_january_first_does_not_cross_year(tmp_path):
    cfg = fetcher.FetchConfig(out_dir=tmp_path)

    url = fetcher.build_mur_sst_url(2026, cfg, today=dt.date(2026, 1, 1))

    assert "2026-01-01T09:00:00Z" in url
    assert "2025-12-31T09:00:00Z" not in url


def test_retry_after_parser_handles_seconds_http_date_and_missing():
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=120)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )

    assert fetcher.parse_retry_after_seconds("30", 900) == 30
    assert 0 <= fetcher.parse_retry_after_seconds(future, 900) <= 120
    assert fetcher.parse_retry_after_seconds(None, 900) == 900


def test_fetch_year_honors_retry_after_without_writing(tmp_path):
    class Response:
        status_code = 429
        text = ""
        headers = {"Retry-After": "5"}

    class Client:
        def get(self, url: str, timeout: int) -> Response:
            assert "jplMURSST41.csv" in url
            assert timeout == 30
            return Response()

    done, retry_after = fetcher.fetch_year(2020, fetcher.FetchConfig(out_dir=tmp_path), client=Client())

    assert done is False
    assert retry_after == 5
    assert not list(tmp_path.glob("*.parquet"))
    assert not list(tmp_path.glob("*.complete.json"))


def test_fetch_year_writes_resume_marker_for_permanent_response(tmp_path):
    class Response:
        status_code = 404
        text = ""
        headers: dict[str, str] = {}

    class Client:
        def get(self, url: str, timeout: int) -> Response:
            return Response()

    cfg = fetcher.FetchConfig(out_dir=tmp_path)

    done, retry_after = fetcher.fetch_year(1999, cfg, client=Client())
    done_again, retry_again = fetcher.fetch_year(1999, cfg, client=Client())

    assert done is True
    assert retry_after is None
    assert done_again is True
    assert retry_again is None
    marker = tmp_path / "mur_sst_1999.complete.json"
    assert marker.exists()
    assert "permanent_http_404" in marker.read_text(encoding="utf-8")


def test_fetch_year_writes_resume_marker_for_empty_success(tmp_path):
    class Response:
        status_code = 200
        text = "time,analysed_sst\nUTC,degree_C\n"
        headers: dict[str, str] = {}

    class Client:
        def get(self, url: str, timeout: int) -> Response:
            return Response()

    done, retry_after = fetcher.fetch_year(2020, fetcher.FetchConfig(out_dir=tmp_path), client=Client())

    assert done is True
    assert retry_after is None
    assert not (tmp_path / "mur_sst_2020.parquet").exists()
    marker = tmp_path / "mur_sst_2020.complete.json"
    assert marker.exists()
    assert "empty_200" in marker.read_text(encoding="utf-8")
