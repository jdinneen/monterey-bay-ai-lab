"""OBIS occurrences for MARINE MAMMALS on the US West Coast — the mortality/stranding labels
for Project Leviathan (why are the whales dying?).

This is the missing LABEL side of the whale-mortality work: the driver side (domoic acid via
habmap_cdph + c_harm, ocean conditions via mur_sst/roms/wcofs) already exists. OBIS (Ocean
Biodiversity Information System; includes the OBIS-SEAMAP marine-mammal node) carries cetacean
and pinniped records — both live SIGHTINGS and dead/stranded specimens — each timestamped
(`eventDate`) and geolocated. We keep `basisOfRecord` / `datasetName` / `occurrenceStatus` so the
modeling lane can separate STRANDINGS (mortality signal) from sightings downstream.

Scope: US West Coast (CA Current) polygon lon -130..-116, lat 32..49 — spans the CA/OR/WA
Unusual Mortality Events. Taxa: Cetacea (whales/dolphins/porpoises) + Otariidae (sea lions/fur
seals) + Phocidae (true seals). Probe (2026-06): ~159k Cetacea, ~13k Otariidae, ~18k Phocidae.

API: https://api.obis.org/v3/occurrence?scientificname=<taxon>&geometry=<polygon>&size=N&after=<id>
Cursor-paged, decade-windowed, staged + resumable. Writes only to data/external_*.
"""
from __future__ import annotations

import urllib.parse
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.obis.org/v3/occurrence"
PAGE = 5000

# Higher taxa that cover the CA-coast marine-mammal fauna without double-counting: Cetacea already
# contains the gray whale / rorqual / dolphin / porpoise families, so we only add the two pinniped
# families alongside it. Records are deduped on `id` at consolidation, so any overlap is harmless.
TAXA = ["Cetacea", "Otariidae", "Phocidae"]

# US West Coast / California Current — the region of the declared marine-mammal mortality events.
WEST_COAST = "POLYGON((-130 32,-116 32,-116 49,-130 49,-130 32))"

# Decade windows so each chunk checkpoints and resumes; marine-mammal records are post-1950.
DECADES = [(y, y + 9) for y in range(1950, 2030, 10)]

_KEEP = ["id", "eventDate", "decimalLatitude", "decimalLongitude", "scientificName", "species",
         "genus", "family", "individualCount", "datasetName", "basisOfRecord", "occurrenceStatus",
         "lifeStage"]


class ObisMarineMammalsAdapter(Adapter):
    """One chunk per (taxon, decade); each chunk cursor-pages that window's West-Coast records."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        for taxon in TAXA:
            for y0, y1 in DECADES:
                yield {"key": f"{taxon}_{y0}", "taxon": taxon, "y0": y0, "y1": y1}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        taxon = chunk["taxon"]
        base = (f"{API}?scientificname={urllib.parse.quote(taxon)}"
                f"&geometry={urllib.parse.quote(WEST_COAST)}&size={PAGE}"
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
        df["query_taxon"] = taxon
        # a stranding/mortality flag the modeling lane can use (heuristic, refined downstream):
        # dead/preserved specimens or datasets named for strandings.
        ds = df["datasetName"].astype(str).str.lower()
        bor = df["basisOfRecord"].astype(str).str.lower()
        df["is_stranding_candidate"] = (
            ds.str.contains("strand|mortal|dead|necrop", regex=True, na=False)
            | bor.str.contains("preservedspecimen|materialsample", regex=True, na=False))
        df = df.dropna(subset=["time"])   # keep only timestamped records (the labeled corpus)
        return df.reset_index(drop=True)
