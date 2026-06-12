"""Cal OES HazMat spill-release archive fetcher.

Cal OES publishes year-specific Excel archive files from its official spill
release reporting page. This adapter discovers those links, fetches bounded
years, and stages normalized rows without touching trusted production data.
"""
from __future__ import annotations

import io
import re
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

ARCHIVE_PAGE = (
    "https://www.caloes.ca.gov/office-of-the-director/operations/response-operations/"
    "fire-rescue/hazardous-materials/spill-release-reporting/"
)


def _normalize_col(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unnamed"


class CalOesSpillArchiveAdapter(Adapter):
    FIRST_YEAR = 2020

    def _archive_links(self) -> dict[int, str]:
        resp = get_with_backoff(
            ARCHIVE_PAGE,
            timeout=60,
            retries=3,
            base_sleep=2,
            headers={"User-Agent": "mbal-datafetch/0.1"},
        )
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        links: dict[int, str] = {}
        for match in re.finditer(r"""href=["']([^"']+\.xls[x]?)["']""", text, re.I):
            url = match.group(1).replace("&amp;", "&")
            year_match = re.search(r"(19|20)\d{2}", url)
            if not year_match:
                continue
            year = int(year_match.group(0))
            if url.startswith("/"):
                url = "https://www.caloes.ca.gov" + url
            links[year] = url
        return links

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        links = self._archive_links()
        y0 = int(start[:4]) if start else self.FIRST_YEAR
        y1 = int(end[:4]) if end else max(links) if links else self.FIRST_YEAR
        for year in sorted(y for y in links if y0 <= y <= y1):
            yield {"key": str(year), "year": year, "url": links[year]}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        resp = get_with_backoff(
            chunk["url"],
            timeout=120,
            retries=3,
            base_sleep=2,
            headers={"User-Agent": "mbal-datafetch/0.1"},
        )
        content = resp.content if hasattr(resp, "content") else resp.text.encode("utf-8")
        try:
            df = pd.read_excel(io.BytesIO(content), dtype=object)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.source}] {chunk['year']} Excel parse failed: {exc}")
            return pd.DataFrame(columns=["source_year", "source_url"])
        df = df.dropna(how="all").copy()
        df.columns = [_normalize_col(c) for c in df.columns]
        df.insert(0, "source_year", int(chunk["year"]))
        df.insert(1, "source_url", chunk["url"])

        date_candidates = [c for c in df.columns if c in {"date", "date_time", "spill_date", "received_date"}]
        if date_candidates:
            df["event_time"] = pd.to_datetime(df[date_candidates[0]], errors="coerce", utc=True)
        else:
            possible = [c for c in df.columns if "date" in c or "time" in c]
            if possible:
                df["event_time"] = pd.to_datetime(df[possible[0]], errors="coerce", utc=True)
            else:
                df["event_time"] = pd.NaT

        id_candidates = [c for c in df.columns if c in {"control_number", "spill_number", "incident_number", "report_number"}]
        if id_candidates:
            df["source_record_id"] = df[id_candidates[0]].astype("string")
        else:
            df["source_record_id"] = df["source_year"].astype(str) + "-" + df.index.astype(str)
        return df
