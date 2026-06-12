"""Marine-mammal mortality labels from iNaturalist, California coast (staged).

The authoritative per-animal cause-of-death record — NOAA's National Stranding
Database — is NOT openly downloadable (request-only via MMHSRP.NationalDB@noaa.gov).
This adapter builds the best OPEN mortality-label set available: research- and
needs-id-grade iNaturalist cetacean observations in the California bbox, carrying the
community "Alive or Dead" annotation (controlled term 17; value 19 == Dead).

Crucially it fetches BOTH dead and not-dead cetacean observations, so the curated
table carries its own EFFORT DENOMINATOR: every downstream mortality rate can be
normalized by total cetacean observations in the same cell/period, which is what makes
the signal robust to iNaturalist's year-over-year user growth (the raw dead-count trend
is otherwise confounded by platform growth — see the source note).

This is a DELIBERATE bounded slice (capped pages/year), not a census — flagged
bounded_sample so it never auto-promotes to READY. Writes only to data/external_*.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
PER_YEAR_PAGE_CAP = 60  # generous: captures ~all CA cetacean obs/yr (peak ~4k) + headroom
BBOX = dict(nelat=42.0, nelng=-117.0, swlat=32.0, swlng=-124.0)
CETACEA_TAXON_ID = 152871           # iNaturalist taxon: Cetacea
TERM_ALIVE_OR_DEAD = 17             # controlled attribute "Alive or Dead"
VALUE_DEAD = 19                     # controlled value "Dead"
VALUE_ALIVE = 18                    # controlled value "Alive"


class MarineMammalMortalityAdapter(Adapter):
    """One chunk per year; id_above deep-paging within each year (capped).

    Pulls all CA cetacean observations and derives `dead_flag` from the
    Alive-or-Dead annotation so the same fetch yields labels + effort.
    """

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(str(start)[:4]) if start else 2013
        y1 = int(str(end)[:4]) if end else dt.date.today().year
        for year in range(y0, y1 + 1):
            yield {"key": str(year), "year": year}

    @staticmethod
    def _dead_flag(obs: dict):
        """1 = annotated Dead, 0 = annotated Alive, <NA> = no alive/dead annotation."""
        flag = pd.NA
        for a in obs.get("annotations") or []:
            if a.get("controlled_attribute_id") == TERM_ALIVE_OR_DEAD:
                cv = a.get("controlled_value_id")
                if cv == VALUE_DEAD:
                    return 1
                if cv == VALUE_ALIVE:
                    flag = 0
        return flag

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        year = chunk["year"]
        rows: list[dict] = []
        id_above = 0
        for _ in range(PER_YEAR_PAGE_CAP):
            params = {
                **BBOX, "taxon_id": CETACEA_TAXON_ID, "geo": "true",
                "d1": f"{year}-01-01", "d2": f"{year}-12-31",
                "per_page": PER_PAGE, "order_by": "id", "order": "asc",
                "id_above": id_above,
            }
            try:
                payload = get_with_backoff(
                    f"{API}?{urlencode(params)}", timeout=60, retries=3, base_sleep=2,
                    headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
                ).json()
            except Exception:
                break
            results = payload.get("results") or []
            if not results:
                break
            for r in results:
                taxon = r.get("taxon") or {}
                geo = r.get("geojson") or {}
                coords = geo.get("coordinates") or [None, None]
                rows.append({
                    "observation_id": r.get("id"),
                    "observed_on": r.get("observed_on"),
                    "time_observed_at": r.get("time_observed_at"),
                    "taxon_name": taxon.get("name"),
                    "common_name": taxon.get("preferred_common_name"),
                    "taxon_rank": taxon.get("rank"),
                    "quality_grade": r.get("quality_grade"),
                    "dead_flag": self._dead_flag(r),
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "place_guess": r.get("place_guess"),
                })
            id_above = results[-1].get("id", id_above)
            if len(results) < PER_PAGE:
                break
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(
            df["time_observed_at"].fillna(df["observed_on"]), errors="coerce", utc=True)
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        # dead_flag: 1=dead, 0=alive, <NA>=unannotated. Keep as nullable Int.
        df["dead_flag"] = df["dead_flag"].astype("Int64")
        df = df.dropna(subset=["time", "taxon_name"])
        return df.reset_index(drop=True)
