"""Water Quality Portal (waterqualitydata.us) — CA agricultural nutrient loading.

Pulls nutrient characteristic results (Nitrate, Phosphate, etc.) for the
whole state, chunked by YEAR so each request is bounded and the run is resumable.
CSV "narrowResult" profile.
"""
from __future__ import annotations

import datetime as dt
import io
import urllib.parse
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

BASE = "https://www.waterqualitydata.us/data/Result/search"
CHARACTERISTICS = [
    "Nitrate",
    "Phosphorus",
    "Phosphate-phosphorus",
    "Nitrogen",
    "Total Nitrogen, mixed forms",
    "Ammonia and ammonium",
    "Orthophosphate"
]
DEFAULT_START_YEAR = 2018


class WqpNutrientsAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else DEFAULT_START_YEAR
        y1 = int(end[:4]) if end else dt.date.today().year
        for yr in range(y0, y1 + 1):
            for char in CHARACTERISTICS:
                # safe key generation without spaces or commas
                safe_char = char.replace(' ', '').replace(',', '').replace('(', '').replace(')', '')
                yield {"key": f"{yr}_{safe_char}", "year": yr, "characteristic": char}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        yr = chunk["year"]
        params = {
            "statecode": "US:06",
            "characteristicName": chunk["characteristic"],
            "startDateLo": f"01-01-{yr}",
            "startDateHi": f"12-31-{yr}",
            "mimeType": "csv",
            "zip": "no",
            "dataProfile": "narrowResult",
            "providers": "STORET",
        }
        url = BASE + "?" + urllib.parse.urlencode(params)
        resp = get_with_backoff(url, timeout=300, headers={"User-Agent": "mbal-datafetch/0.1"})
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        if not text.strip() or text.lstrip().startswith("<"):
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception:
            return pd.DataFrame()
        # Keep a focused, modeling-relevant column subset when present.
        keep = [c for c in [
            "OrganizationIdentifier", "MonitoringLocationIdentifier", "ActivityStartDate",
            "CharacteristicName", "ResultMeasureValue", "ResultMeasure/MeasureUnitCode",
            "ResultIdentifier", "ActivityMediaName",
        ] if c in df.columns]
        if keep:
            df = df[keep].copy()
        if "ActivityStartDate" in df.columns:
            df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
        # ResultMeasureValue mixes numerics + non-detect flags; keep as string for parquet.
        for c in df.columns:
            if c != "ActivityStartDate" and df[c].dtype == object:
                df[c] = df[c].astype("string")
        return df
