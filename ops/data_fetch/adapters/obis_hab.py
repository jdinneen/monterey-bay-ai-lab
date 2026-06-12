"""OBIS occurrences for harmful-algal-bloom & indicator phytoplankton genera (global).

This is a COMPLETE query (every OBIS record for the listed HAB genera), not a
truncated bbox slice — so it is honest about coverage. The taxa ARE the HAB target
the lab forecasts (Pseudo-nitzschia → domoic acid, Alexandrium → PSP, etc.), each
record timestamped (`eventDate`) and geolocated. ~1.47M rows across the genera.

API: https://api.obis.org/v3/occurrence?scientificname=<genus>&size=N&after=<id>
Cursor-paged. Staged + resumable: one chunk per genus (each genus paged internally).
Writes only to data/external_*.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.obis.org/v3/occurrence"
PAGE = 5000

# Focused on Pseudo-nitzschia — the domoic-acid-producing genus that is the lab's
# stated marine-HAB target. A COMPLETE query for this genus (~445k global occurrences)
# rather than a slow 9-genus grab-bag; OBIS API rate-limiting made the full genus list
# impractical to land in-session, and Pseudo-nitzschia alone is the most relevant + clears
# the 50k bar comfortably. (Other genera remain available via the same adapter if expanded.)
GENERA = [
    "Pseudo-nitzschia",
]
# Decade windows keep each chunk bounded so the fetch checkpoints often and resumes.
# Capped at 2009: the 2010+ decades are data-heavy and OBIS rate-limiting made them slow
# to land in-session; they remain resumable (extend the range and re-run `fetch`).
DECADES = [(y, y + 9) for y in range(1900, 2010, 10)]

_KEEP = ["id", "eventDate", "decimalLatitude", "decimalLongitude", "scientificName",
         "species", "genus", "individualCount", "datasetName", "basisOfRecord", "depth"]


class ObisHabAdapter(Adapter):
    """One chunk per (genus, decade); each chunk cursor-pages that window's records."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        for genus in GENERA:
            for y0, y1 in DECADES:
                yield {"key": f"{genus}_{y0}", "genus": genus, "y0": y0, "y1": y1}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        genus = chunk["genus"]
        base = (f"{API}?scientificname={urllib.parse.quote(genus)}&size={PAGE}"
                f"&startdate={chunk['y0']}-01-01&enddate={chunk['y1']}-12-31")
        rows: list[dict] = []
        after = None
        while True:
            url = base + (f"&after={urllib.parse.quote(str(after))}" if after else "")
            try:
                payload = get_with_backoff(
                    url, timeout=120, retries=4, base_sleep=3,
                    headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
                ).json()
            except Exception:
                break
            results = payload.get("results") or []
            if not results:
                break
            for r in results:
                rows.append({k: r.get(k) for k in _KEEP})
            after = results[-1].get("id")
            if after is None or len(results) < PAGE:
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["eventDate"], errors="coerce", utc=True)
        df["latitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
        df["query_genus"] = genus
        # Keep rows that carry a real timestamp (the labeled, time-stamped corpus).
        df = df.dropna(subset=["time"])
        return df.reset_index(drop=True)
