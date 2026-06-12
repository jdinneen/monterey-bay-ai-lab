"""CCIEA Forage Biomass index, California Current Central (staged).

NOAA's California Current Integrated Ecosystem Assessment (CCIEA) publishes an annual
forage-biomass anomaly index from the RREAS midwater-trawl survey. The Central region
(`cciea_EI_FBC`) covers the core central-California foraging grounds (incl. Monterey
Bay). Species groups include Total Krill, Adult/YOY Anchovy, YOY Rockfish, Market Squid,
Myctophids, Sardine, Salps — i.e. the prey base.

This is the mechanistic *malnutrition* driver for the marine-mammal-mortality work:
NOAA attributed the 2019-2023 Eastern N. Pacific gray-whale Unusual Mortality Event to
feeding-ground ecosystem change -> reduced prey -> malnutrition. A forage-biomass
collapse is the testable upstream signal.

Long format, one row per (year, species_group): time, species_group, mean_cpue (log
anomaly), SEup, SElo. The full index is small (~500 rows, 1990->present) — this is the
COMPLETE published index, not a sample. ERDDAP tabledap; one chunk. Writes only to
data/external_*.
"""
from __future__ import annotations

from typing import Iterator, Optional
from urllib.parse import quote

import pandas as pd

from ..core import Adapter, get_with_backoff

ERDDAP = "https://oceanview.pfeg.noaa.gov/erddap"
DATASET_ID = "cciea_EI_FBC"
VARS = "time,mean_cpue,species_group,SEup,SElo"


class CcieaForageAdapter(Adapter):
    """Single-chunk ERDDAP tabledap pull of the complete forage-biomass index."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        # Small, complete table — one chunk. start/end optionally narrow the time range.
        yield {"key": "all", "start": start, "end": end}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        q = VARS
        if chunk.get("start"):
            q += f"&time>={chunk['start']}"
        if chunk.get("end"):
            q += f"&time<={chunk['end']}"
        url = f"{ERDDAP}/tabledap/{DATASET_ID}.json?{quote(q, safe='&<>=,')}"
        payload = get_with_backoff(
            url, timeout=90, retries=4, base_sleep=2,
            headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
        ).json()
        table = payload.get("table") or {}
        cols = table.get("columnNames") or []
        rows = table.get("rows") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=cols)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        for c in ("mean_cpue", "SEup", "SElo"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["region"] = "central_california"
        df = df.dropna(subset=["time", "species_group"])
        return df.reset_index(drop=True)
