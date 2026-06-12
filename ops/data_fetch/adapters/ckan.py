"""Shared base for CKAN datastore_search sources (data.ca.gov).

CKAN paginates via offset+limit; we treat each page as one resumable chunk, so a
re-run only fetches outstanding pages. Subclasses set RESOURCE_ID, optional FILTERS
(exact-match dict) / Q (full-text), PAGE_SIZE, MAX_PAGES, and override `normalize()`
to map raw CKAN fields to the spec's required columns.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

CKAN_BASE = "https://data.ca.gov/api/3/action/datastore_search"


class CkanAdapter(Adapter):
    RESOURCE_ID: str = ""
    FILTERS: Optional[dict] = None
    Q: Optional[str] = None
    PAGE_SIZE: int = 5000
    MAX_PAGES: int = 50  # safety cap; raise via env or subclass for full pulls

    def _count(self) -> int:
        url = f"{CKAN_BASE}?resource_id={self.RESOURCE_ID}&limit=0"
        if self.FILTERS:
            url += "&filters=" + urllib.parse.quote(json.dumps(self.FILTERS))
        if self.Q:
            url += "&q=" + urllib.parse.quote(self.Q)
        resp = get_with_backoff(url, timeout=60, headers={"User-Agent": "mbal-datafetch/0.1"})
        return int(resp.json()["result"]["total"])

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        total = self._count()
        pages = (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        pages = min(pages, self.MAX_PAGES)
        for i in range(pages):
            yield {"key": f"p{i:04d}", "offset": i * self.PAGE_SIZE, "total": total}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        url = (
            f"{CKAN_BASE}?resource_id={self.RESOURCE_ID}"
            f"&limit={self.PAGE_SIZE}&offset={chunk['offset']}"
        )
        if self.FILTERS:
            url += "&filters=" + urllib.parse.quote(json.dumps(self.FILTERS))
        if self.Q:
            url += "&q=" + urllib.parse.quote(self.Q)
        resp = get_with_backoff(url, timeout=120, headers={"User-Agent": "mbal-datafetch/0.1"})
        records = resp.json()["result"]["records"]
        df = pd.DataFrame.from_records(records)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return self.normalize(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw CKAN columns to the spec schema. Override in subclasses."""
        return df
