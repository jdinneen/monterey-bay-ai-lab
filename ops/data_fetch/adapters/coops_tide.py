"""Staged NOAA CO-OPS tide water-level fetcher.

This adapter is intentionally additive: it writes only under data/external_* and
does not mutate the trusted bacteria tide-stage outputs.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_STATIONS = {
    "9410230": "La Jolla",
    "9410660": "Los Angeles",
    "9410840": "Santa Monica",
    "9411340": "Santa Barbara",
    "9414290": "San Francisco",
}


class CoopsTideStagedAdapter(Adapter):
    """Fetch hourly CO-OPS water levels in API-safe chunks."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = dt.date.fromisoformat(start) if start else dt.date.today() - dt.timedelta(days=365)
        e = dt.date.fromisoformat(end) if end else dt.date.today() - dt.timedelta(days=1)
        for station_id, station_name in COOPS_STATIONS.items():
            d0 = s
            while d0 <= e:
                d1 = min(d0 + dt.timedelta(days=29), e)
                yield {
                    "key": f"{station_id}_{d0.isoformat()}_{d1.isoformat()}",
                    "station_id": station_id,
                    "station_name": station_name,
                    "begin_date": d0.strftime("%Y%m%d"),
                    "end_date": d1.strftime("%Y%m%d"),
                }
                d0 = d1 + dt.timedelta(days=1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        params = {
            "product": "water_level",
            "application": "Monterey_Bay_AI_Lab_data_fetcher",
            "begin_date": chunk["begin_date"],
            "end_date": chunk["end_date"],
            "datum": "MLLW",
            "station": chunk["station_id"],
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        }
        url = f"{COOPS_API}?{urlencode(params)}"
        payload = get_with_backoff(
            url,
            timeout=60,
            retries=4,
            headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
        ).json()
        rows = payload.get("data") or []
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        if "t" not in out.columns or "v" not in out.columns:
            return pd.DataFrame()
        out = out.rename(columns={
            "t": "sample_date",
            "v": "water_level_m",
            "s": "sigma_m",
            "f": "quality_flags",
            "q": "quality_code",
        })
        out["station_id"] = chunk["station_id"]
        out["station_name"] = chunk["station_name"]
        out["sample_date"] = pd.to_datetime(out["sample_date"], errors="coerce", utc=True)
        out["water_level_m"] = pd.to_numeric(out["water_level_m"], errors="coerce")
        if "sigma_m" in out.columns:
            out["sigma_m"] = pd.to_numeric(out["sigma_m"], errors="coerce")
        out["datum"] = "MLLW"
        out["units"] = "metric"
        out["source"] = "NOAA CO-OPS water_level API"
        keep = [
            "station_id",
            "station_name",
            "sample_date",
            "water_level_m",
            "sigma_m",
            "quality_flags",
            "quality_code",
            "datum",
            "units",
            "source",
        ]
        keep = [c for c in keep if c in out.columns]
        return out[keep].dropna(subset=["sample_date", "water_level_m"])
