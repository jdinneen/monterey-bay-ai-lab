#!/usr/bin/env python3
"""Polite resumable fetcher for NOAA CoastWatch ERDDAP Chlorophyll-a data.

Fetches a fixed validation point from the erdVHNchla1day 750m North Pacific
dataset. This is intentionally a narrow research cache, not production
lakehouse wiring: downstream model integration should wait until this physical
driver proves lift against the San-Diego-excluded bacteria baseline.
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
from urllib.parse import quote

import pandas as pd


DEFAULT_OUT_DIR = Path("mbal_history/noaa/chlorophyll_cache")
COASTWATCH_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET_ID = "erdVHNchla1day"
M1_LAT = 36.7511
M1_LON = -122.0292
FIRST_CHLA_DATE = dt.date(2015, 2, 25)  # erdVHNchla1day start
YEAR_FILE_RE = re.compile(r"^chla_\d{4}\.parquet$")
CHUNK_FILE_RE = re.compile(r"^chla_\d{4}_\d{8}_\d{8}\.parquet$")


class HttpClient(Protocol):
    def get(self, url: str, timeout: int) -> Any:
        ...


@dataclass(frozen=True)
class FetchConfig:
    out_dir: Path = DEFAULT_OUT_DIR
    start_year: int = FIRST_CHLA_DATE.year
    end_year: int = dt.date.today().year
    cooldown_seconds: int = 15 * 60
    delay_seconds: float = 2.0
    max_retries_per_year: int = 3
    timeout_seconds: int = 60
    chunk_days: int = 31
    erddap_base_url: str = COASTWATCH_ERDDAP
    lat: float = M1_LAT
    lon: float = M1_LON


def year_bounds(year: int, today: dt.date | None = None) -> tuple[str, str]:
    today = today or dt.date.today()
    start = FIRST_CHLA_DATE if year == FIRST_CHLA_DATE.year else dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    if year == today.year:
        end = max(start, today - dt.timedelta(days=1))
    return start.isoformat(), end.isoformat()


def date_chunks(start: dt.date, end: dt.date, chunk_days: int) -> list[tuple[dt.date, dt.date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if end < start:
        return []

    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def build_chla_url(start: dt.date, end: dt.date, cfg: FetchConfig) -> str:
    query = (
        "chla"
        f"[({start.isoformat()}T12:00:00Z):({end.isoformat()}T12:00:00Z)]"
        "[({altitude}):({altitude})]"
        f"[({cfg.lat}):({cfg.lat})]"
        f"[({cfg.lon}):({cfg.lon})]"
    ).format(altitude=0.0)
    return (
        f"{cfg.erddap_base_url.rstrip('/')}/griddap/{DATASET_ID}.csv?"
        f"{quote(query, safe='(),:')}"
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


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def write_metadata(path: Path, year: int, status: str, details: dict[str, Any] | None = None) -> None:
    metadata = {
        "year": year,
        "status": status,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": DATASET_ID,
        "source": COASTWATCH_ERDDAP,
        "details": details or {},
    }
    metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_client(client: HttpClient | None = None) -> HttpClient:
    if client is not None:
        return client
    import requests

    return requests.Session()


def normalize_chla_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    if df.empty:
        return pd.DataFrame(columns=["time", "latitude", "longitude", "chla_m1_mg_m3"])

    required = {"time", "latitude", "longitude", "chla"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ERDDAP response missing columns: {missing}")

    out = df[["time", "latitude", "longitude", "chla"]].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["chla_m1_mg_m3"] = pd.to_numeric(out["chla"], errors="coerce")
    return out[["time", "latitude", "longitude", "chla_m1_mg_m3"]]


def chunk_path(cfg: FetchConfig, year: int, start: dt.date, end: dt.date) -> Path:
    chunk_dir = cfg.out_dir / "_chunks"
    return chunk_dir / f"chla_{year}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"


def fetch_chunk(
    year: int,
    start: dt.date,
    end: dt.date,
    cfg: FetchConfig,
    client: HttpClient | None = None,
) -> tuple[bool, int | None]:
    path = chunk_path(cfg, year, start, end)
    if path.exists():
        print(f"[{year}] Chunk {start}..{end} already exists.")
        return True, None

    path.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = get_client()

    url = build_chla_url(start, end, cfg)
    print(f"[{year}] Fetching {start}..{end} from ERDDAP...")
    response = client.get(url, timeout=cfg.timeout_seconds)

    if response.status_code == 200:
        df = normalize_chla_csv(response.text)
        df.to_parquet(path, index=False)
        print(f"[{year}] Wrote {len(df)} rows for {start}..{end}.")
        return True, None

    if response.status_code in {400, 404}:
        write_metadata(
            path,
            year,
            f"permanent_http_{response.status_code}",
            {"start": start.isoformat(), "end": end.isoformat(), "url": url},
        )
        print(f"[{year}] Permanent HTTP {response.status_code} for {start}..{end}.")
        return True, None

    retry_after = parse_retry_after_seconds(
        response.headers.get("Retry-After"),
        cfg.cooldown_seconds,
    )
    print(f"[{year}] Transient HTTP {response.status_code}. Will retry after {retry_after}s.")
    return False, retry_after


def merge_year_chunks(year: int, cfg: FetchConfig, chunks: list[tuple[dt.date, dt.date]]) -> Path:
    path = cfg.out_dir / f"chla_{year}.parquet"
    frames = []
    for start, end in chunks:
        cpath = chunk_path(cfg, year, start, end)
        if cpath.exists():
            frames.append(pd.read_parquet(cpath))

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
    else:
        df = pd.DataFrame(columns=["time", "latitude", "longitude", "chla_m1_mg_m3"])

    df.to_parquet(path, index=False)
    write_metadata(
        path,
        year,
        "complete",
        {
            "rows": int(len(df)),
            "chunk_count": len(chunks),
            "chunk_days": cfg.chunk_days,
            "lat": cfg.lat,
            "lon": cfg.lon,
        },
    )
    return path


def fetch_year(year: int, cfg: FetchConfig, client: HttpClient | None = None) -> tuple[bool, int | None]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"chla_{year}.parquet"
    if path.exists() and metadata_path(path).exists():
        print(f"[{year}] Year cache already exists. Skipping.")
        return True, None

    start_s, end_s = year_bounds(year)
    chunks = date_chunks(dt.date.fromisoformat(start_s), dt.date.fromisoformat(end_s), cfg.chunk_days)
    for start, end in chunks:
        done, retry_after = fetch_chunk(year, start, end, cfg, client=client)
        if not done:
            return False, retry_after
        time.sleep(cfg.delay_seconds)

    merge_year_chunks(year, cfg, chunks)
    print(f"[{year}] Complete. Wrote {path}.")
    return True, None


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
    parser = argparse.ArgumentParser(description="Fetch Chlorophyll-a yearly cache files politely.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-year", type=int, default=FIRST_CHLA_DATE.year)
    parser.add_argument("--end-year", type=int, default=dt.date.today().year)
    parser.add_argument("--cooldown-seconds", type=int, default=15 * 60)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries-per-year", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--lat", type=float, default=M1_LAT)
    parser.add_argument("--lon", type=float, default=M1_LON)
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
        chunk_days=args.chunk_days,
        lat=args.lat,
        lon=args.lon,
    )
    return resumable_fetch(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
