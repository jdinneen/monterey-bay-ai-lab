"""iNaturalist research-grade observations, California coast (staged).

Citizen-science species occurrences (research-grade = community-verified ID) in the
California coastal bounding box — a labeled, timestamped, geolocated biodiversity
corpus (incl. marine wildlife, bloom/mortality-relevant taxa). Distinct modality from
the agency monitoring sources.

API: https://api.inaturalist.org/v1/observations (deep-paged by id_above, which has no
offset cap). Chunked by year; each year paged up to PER_YEAR_PAGE_CAP pages. This is a
DELIBERATE representative slice of a 13.5M-record universe, not a full census — the cap
is documented here and in the source note so the fetch never masquerades as complete.

Writes only to data/external_*.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
PER_YEAR_PAGE_CAP = 40  # <= 8k rows/year; ~80k over the year span (documented slice)
BBOX = dict(nelat=42.0, nelng=-117.0, swlat=32.0, swlng=-124.0)


class INaturalistAdapter(Adapter):
    """One chunk per year; id_above deep-paging within each year (capped)."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(str(start)[:4]) if start else 2015
        y1 = int(str(end)[:4]) if end else (dt.date.today().year - 1)
        for year in range(y0, y1 + 1):
            yield {"key": str(year), "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        year = chunk["year"]
        rows: list[dict] = []
        id_above = 0
        for _ in range(PER_YEAR_PAGE_CAP):
            params = {
                **BBOX, "quality_grade": "research", "geo": "true",
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
                    "taxon_rank": taxon.get("rank"),
                    "iconic_taxon": taxon.get("iconic_taxon_name"),
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
        df = df.dropna(subset=["time", "taxon_name"])
        return df.reset_index(drop=True)
