#!/usr/bin/env python3
"""Polite resumable fetcher for NOAA CoastWatch ERDDAP MUR SST data.

The fetcher writes one Parquet file per year, skips files that already exist,
and backs off when the server asks for slower traffic. It is intended for
operator-launched cache refreshes, not for bypassing provider limits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


DEFAULT_OUT_DIR = Path("mbari_history/noaa/mur_sst_cache")
COASTWATCH_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"
M1_LAT = 36.7511
M1_LON = -122.0292
FIRST_MUR_DATE = dt.date(2002, 6, 1)
YEAR_FILE_RE = re.compile(r"^mur_sst_\d{4}\.parquet$")


class HttpClient(Protocol):
    def get(self, url: str, timeout: int) -> Any:
        ...


@dataclass(frozen=True)
class FetchConfig:
    out_dir: Path = DEFAULT_OUT_DIR
    start_year: int = FIRST_MUR_DATE.year
    end_year: int = dt.date.today().year
    cooldown_seconds: int = 15 * 60
    delay_seconds: float = 2.0
    max_retries_per_year: int = 3
    timeout_seconds: int = 30
    erddap_base_url: str = COASTWATCH_ERDDAP
    lat: float = M1_LAT
    lon: float = M1_LON


def year_bounds(year: int, today: dt.date | None = None) -> tuple[str, str]:
    today = today or dt.date.today()
    start = FIRST_MUR_DATE if year == FIRST_MUR_DATE.year else dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    if year == today.year:
        end = max(start, today - dt.timedelta(days=1))
    return start.isoformat(), end.isoformat()


def build_mur_sst_url(year: int, cfg: FetchConfig, today: dt.date | None = None) -> str:
    start, end = year_bounds(year, today=today)
    return (
        f"{cfg.erddap_base_url}/griddap/jplMURSST41.csv?analysed_sst"
        f"%5B({start}T09:00:00Z):({end}T09:00:00Z)%5D"
        f"%5B({cfg.lat}):({cfg.lat})%5D"
        f"%5B({cfg.lon}):({cfg.lon})%5D"
    )


def parse_retry_after_seconds(value: str | None, default_seconds: int) -> int:
    if not value:
        return default_seconds
    try:
        return max(0, int(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
        now = dt.datetime.now(dt.timezone.utc)
        return max(0, int((retry_at - now).total_seconds()))
    except (TypeError, ValueError, IndexError, OverflowError):
        return default_seconds


def completion_marker_path(path: Path) -> Path:
    return path.with_suffix(".complete.json")


def write_completion_marker(path: Path, year: int, status: str, details: dict[str, Any] | None = None) -> None:
    marker = {
        "year": year,
        "status": status,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "details": details or {},
    }
    completion_marker_path(path).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def fetch_year(year: int, cfg: FetchConfig, client: HttpClient | None = None) -> tuple[bool, int | None]:
    """Fetch one year.

    Returns ``(done, retry_after_seconds)``. ``done`` means no more work is
    needed for that year, including existing files and permanent out-of-bounds
    responses.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"mur_sst_{year}.parquet"
    marker = completion_marker_path(path)
    if path.exists() or marker.exists():
        print(f"[{year}] Already exists. Skipping.")
        return True, None

    if client is None:
        import requests

        client = requests.Session()
    url = build_mur_sst_url(year, cfg)
    print(f"[{year}] Fetching from ERDDAP...")
    response = client.get(url, timeout=cfg.timeout_seconds)

    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["sst_mur_m1_c"] = pd.to_numeric(df["analysed_sst"], errors="coerce")
            df = df[["time", "sst_mur_m1_c"]].set_index("time")
            df.to_parquet(path)
            print(f"[{year}] Success. Wrote {len(df)} rows.")
        else:
            write_completion_marker(path, year, "empty_200")
            print(f"[{year}] Success, but no data returned.")
        return True, None

    if response.status_code in {400, 404}:
        write_completion_marker(path, year, f"permanent_http_{response.status_code}")
        print(f"[{year}] Permanent HTTP {response.status_code}. Treating as complete.")
        return True, None

    retry_after = parse_retry_after_seconds(
        response.headers.get("Retry-After"),
        cfg.cooldown_seconds,
    )
    print(f"[{year}] Transient HTTP {response.status_code}. Will retry after {retry_after}s.")
    return False, retry_after


def resumable_fetch(cfg: FetchConfig, client: HttpClient | None = None) -> int:
    years_to_fetch = list(range(cfg.start_year, cfg.end_year + 1))
    retries = {year: 0 for year in years_to_fetch}

    while years_to_fetch:
        year = years_to_fetch[0]
        if retries[year] >= cfg.max_retries_per_year:
            print(f"[fetcher] Max retries reached for {year}. Skipping.")
            years_to_fetch.pop(0)
            continue

        try:
            done, retry_after = fetch_year(year, cfg, client=client)
        except Exception as exc:
            retries[year] += 1
            print(f"[fetcher] Error for {year}: {exc}")
            retry_after = cfg.cooldown_seconds
            done = False

        if done:
            years_to_fetch.pop(0)
            time.sleep(cfg.delay_seconds)
        else:
            retries[year] += 1
            time.sleep(retry_after if retry_after is not None else cfg.cooldown_seconds)

    files = sorted(path.name for path in cfg.out_dir.iterdir() if YEAR_FILE_RE.match(path.name))
    print(f"[fetcher] Complete. Found {len(files)} yearly Parquet cache files.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MUR SST yearly cache files politely.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-year", type=int, default=FIRST_MUR_DATE.year)
    parser.add_argument("--end-year", type=int, default=dt.date.today().year)
    parser.add_argument("--cooldown-seconds", type=int, default=15 * 60)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries-per-year", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = FetchConfig(
        out_dir=args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        cooldown_seconds=args.cooldown_seconds,
        delay_seconds=args.delay_seconds,
        max_retries_per_year=args.max_retries_per_year,
        timeout_seconds=args.timeout_seconds,
    )
    return resumable_fetch(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

