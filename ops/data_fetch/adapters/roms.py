"""SCCOOS ROMS circulation adapter.

Fetches bounded surface-grid samples from SCCOOS ERDDAP `roms_ncst`. The
provider dataset currently exposes a historical nowcast archive, so the adapter
uses ERDDAP metadata to stay inside the actual coverage instead of assuming the
feed is current.
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..core import Adapter, _guard_write, _to_parquet_safe, get_with_backoff

SCCOOS = "https://erddap.sccoos.org/erddap"
DATASET_ID = "roms_ncst"

# Surface depth
DEPTH = 0.0

VARIABLES = ["u", "v", "temp", "salt"]
LAT_MIN = 36.0
LAT_MAX = 38.0
LON_MIN = 237.0
LON_MAX = 239.0
SPACE_STRIDE = 10
TIME_STRIDE = 4

class RomsAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        cov_start, cov_end = _dataset_time_coverage()
        if start:
            start_date = dt.date.fromisoformat(start[:10]).replace(day=1)
        else:
            start_date = cov_end.date().replace(day=1)
        if end:
            end_date = dt.date.fromisoformat(end[:10]).replace(day=1)
        else:
            end_date = cov_end.date().replace(day=1)

        coverage_month_start = cov_start.date().replace(day=1)
        coverage_month_end = cov_end.date().replace(day=1)
        start_date = max(start_date, coverage_month_start)
        end_date = min(end_date, coverage_month_end)
        if start_date > end_date:
            return

        curr = start_date
        while curr <= end_date:
            month_end = _month_end(curr)
            s = max(dt.datetime.combine(curr, dt.time.min, tzinfo=dt.timezone.utc), cov_start)
            e = min(dt.datetime.combine(month_end, dt.time.max, tzinfo=dt.timezone.utc), cov_end)
            yield {
                "key": curr.strftime("%Y-%m"),
                "start": s.isoformat().replace("+00:00", "Z"),
                "end": e.isoformat().replace("+00:00", "Z"),
            }
            if curr.month == 12:
                curr = dt.date(curr.year + 1, 1, 1)
            else:
                curr = dt.date(curr.year, curr.month + 1, 1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        url = _build_roms_url(chunk["start"], chunk["end"])
        
        resp = get_with_backoff(url, timeout=900, retries=3, base_sleep=5, headers={"User-Agent": "mbal-datafetch/0.1"})
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        
        if not text.strip() or "Error" in text[:200]:
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])
        except Exception:
            return pd.DataFrame()
            
        if "time" not in df.columns:
            return pd.DataFrame()
            
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for col in ("depth", "latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for v in VARIABLES:
            if v in df.columns:
                df[v] = pd.to_numeric(df[v], errors="coerce")
                df.loc[df[v] <= -9990, v] = np.nan
        if "salt" in df.columns:
            df.loc[df["salt"] < 0, "salt"] = np.nan
        if "longitude" in df.columns:
            df["longitude_180"] = ((df["longitude"] + 180.0) % 360.0) - 180.0
        if {"u", "v"}.issubset(df.columns):
            df["current_speed_m_s"] = (df["u"] ** 2 + df["v"] ** 2) ** 0.5

        # Drop rows where all variable columns are NaN
        df = df.dropna(subset=VARIABLES, how="all")
        df["source"] = "SCCOOS ERDDAP roms_ncst"
        return df.dropna(subset=["time", "latitude", "longitude"])

    def _consolidate(self):
        frames = []
        required = set(self.spec.required_columns + ["longitude_180", "current_speed_m_s", "source"])
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[roms_circulation] skipping unreadable chunk {p.name}: {exc}")
                continue
            if len(d) and required.issubset(d.columns):
                frames.append(d)
        out_path = self.curated_path
        _guard_write(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            _to_parquet_safe(pd.DataFrame(columns=self.spec.required_columns), out_path)
            return out_path
        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        for col in ["depth", "latitude", "longitude", "u", "v", "temp", "salt", "longitude_180", "current_speed_m_s"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in VARIABLES:
            if col in df.columns:
                df.loc[df[col] <= -9990, col] = np.nan
        if "salt" in df.columns:
            df.loc[df["salt"] < 0, "salt"] = np.nan
        df = df.dropna(subset=VARIABLES, how="all")
        df = df.dropna(subset=["time", "depth", "latitude", "longitude"])
        if {"u", "v"}.issubset(df.columns):
            df["current_speed_m_s"] = (df["u"] ** 2 + df["v"] ** 2) ** 0.5
        if "longitude" in df.columns:
            df["longitude_180"] = ((df["longitude"] + 180.0) % 360.0) - 180.0
        keys = [k for k in self.spec.dedup_keys if k in df.columns]
        if keys:
            df = df.drop_duplicates(keys)
        df = df.sort_values(["time", "depth", "latitude", "longitude"])
        _to_parquet_safe(df, out_path)
        return out_path


def _build_roms_url(start: str, end: str) -> str:
    time_bound = f"[({start}):{TIME_STRIDE}:({end})]"
    depth_bound = f"[({DEPTH}):1:({DEPTH})]"
    lat_bound = f"[({LAT_MIN}):{SPACE_STRIDE}:({LAT_MAX})]"
    lon_bound = f"[({LON_MIN}):{SPACE_STRIDE}:({LON_MAX})]"
    bounds = f"{time_bound}{depth_bound}{lat_bound}{lon_bound}"
    var_str = ",".join([f"{v}{bounds}" for v in VARIABLES])
    return f"{SCCOOS}/griddap/{DATASET_ID}.csv?{var_str}"


def _dataset_time_coverage() -> tuple[dt.datetime, dt.datetime]:
    url = f"{SCCOOS}/info/{DATASET_ID}/index.csv"
    text = get_with_backoff(url, timeout=60, retries=3).text
    meta = pd.read_csv(io.StringIO(text))
    rows = meta[
        (meta["Row Type"] == "attribute")
        & (meta["Variable Name"] == "time")
        & (meta["Attribute Name"] == "actual_range")
    ]
    if rows.empty:
        raise RuntimeError(f"{DATASET_ID}: missing time actual_range metadata")
    raw = str(rows.iloc[0]["Value"])
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) != 2:
        raise RuntimeError(f"{DATASET_ID}: invalid time actual_range: {raw}")
    return (
        dt.datetime.fromtimestamp(parts[0], tz=dt.timezone.utc),
        dt.datetime.fromtimestamp(parts[1], tz=dt.timezone.utc),
    )


def _month_end(month_start: dt.date) -> dt.date:
    if month_start.month == 12:
        return dt.date(month_start.year, 12, 31)
    return dt.date(month_start.year, month_start.month + 1, 1) - dt.timedelta(days=1)
