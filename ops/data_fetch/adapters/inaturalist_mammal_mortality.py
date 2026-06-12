"""iNaturalist marine-mammal MORTALITY signal — California coast (Project Leviathan labels).

The open mortality label whale-2's recon validated: iNaturalist's "Alive or Dead" annotation
(controlled_attribute_id=17, controlled_value_id=19 = Dead) on community-verified marine-mammal
observations. Each DEAD record is dated, geolocated, and species-tagged — and the per-year
observation EFFORT (total verifiable obs of the same taxa) is attached to every row, so the
effort-normalized dead-FRACTION (dead / effort) is computable. That fraction reconstructs NOAA's
2019-2023 gray-whale UME from open data (whale-2: peaks ~9% in 2020-21 vs ~2-3% baseline).

Honest caveats (carry these downstream): voluntary annotation; observer-effort + COVID-2020/21
behavior confounds; small early denominators. It is a citizen-science PROXY for mortality, not the
NOAA per-animal cause-of-death record (that DB is request-walled).

Taxa (verified iNat ids): Cetacea 152871 (whales/dolphins/porpoises), Otariidae 41736 (sea lions/
fur seals), Phocidae 41687 (true seals). bbox = CA coast (32-42N, -124..-117).
API: https://api.inaturalist.org/v1/observations  (deep-paged by id_above). Writes only to data/external_*.
"""
from __future__ import annotations

from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
DEAD_TERM_ID = 17        # iNat "Alive or Dead" controlled attribute
DEAD_VALUE_ID = 19       # value = Dead
BBOX = dict(nelat=42.0, nelng=-117.0, swlat=32.0, swlng=-124.0)
TAXA = {"Cetacea": 152871, "Otariidae": 41736, "Phocidae": 41687}


def _total(params: dict) -> int:
    """Count-only query (per_page=0) -> total_results, for the effort denominator."""
    try:
        payload = get_with_backoff(
            f"{API}?{urlencode({**params, 'per_page': 0})}", timeout=60, retries=3, base_sleep=2,
            headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"}).json()
        return int(payload.get("total_results") or 0)
    except Exception:
        return 0


class INatMammalMortalityAdapter(Adapter):
    """One chunk per (taxon, year): fetch that year's DEAD records + attach the year's effort total."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(str(start)[:4]) if start else 2010
        y1 = int(str(end)[:4]) if end else 2026
        for taxon, tid in TAXA.items():
            for year in range(y0, y1 + 1):
                yield {"key": f"{taxon}_{year}", "taxon": taxon, "tid": tid, "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        tid, year = chunk["tid"], chunk["year"]
        window = {**BBOX, "taxon_id": tid, "verifiable": "true",
                  "d1": f"{year}-01-01", "d2": f"{year}-12-31"}
        effort_total = _total(window)            # all verifiable obs of this taxon this year
        rows: list[dict] = []
        id_above = 0
        while True:                              # dead records are sparse -> page them fully
            params = {**window, "term_id": DEAD_TERM_ID, "term_value_id": DEAD_VALUE_ID,
                      "per_page": PER_PAGE, "order_by": "id", "order": "asc", "id_above": id_above}
            try:
                payload = get_with_backoff(
                    f"{API}?{urlencode(params)}", timeout=60, retries=3, base_sleep=2,
                    headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"}).json()
            except Exception:
                break
            results = payload.get("results") or []
            if not results:
                break
            for r in results:
                taxon = r.get("taxon") or {}
                coords = (r.get("geojson") or {}).get("coordinates") or [None, None]
                rows.append({
                    "observation_id": r.get("id"), "observed_on": r.get("observed_on"),
                    "time_observed_at": r.get("time_observed_at"),
                    "taxon_name": taxon.get("name"), "taxon_rank": taxon.get("rank"),
                    "query_taxon": chunk["taxon"], "longitude": coords[0], "latitude": coords[1],
                    "place_guess": r.get("place_guess"), "quality_grade": r.get("quality_grade"),
                    "year": year, "year_effort_total": effort_total, "is_dead": True,
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
