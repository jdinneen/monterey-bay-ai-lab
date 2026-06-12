"""CUTI + BEUTI coastal upwelling indices (Jacox et al.) — the canonical California
upwelling drivers and the *strongest published domoic-acid (HAB) driver*.

Source: ``https://mjacox.com/wp-content/uploads/{CUTI,BEUTI}_daily.csv`` (open). Daily
1988→present at 1° latitude bands. CUTI = Coastal Upwelling Transport Index (volume);
BEUTI = Biologically Effective Upwelling Transport Index (nitrate flux — the mechanistic
driver of Pseudo-nitzschia blooms → domoic acid). Melted to long
(date × latitude × index → value); CA bands 32N–42N.

Lane B (claude-fetch-2). One chunk per index CSV.
"""
from __future__ import annotations

import io
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

_BASE = "https://mjacox.com/wp-content/uploads/{idx}_daily.csv"
_INDICES = ["CUTI", "BEUTI"]
_LAT_MIN, _LAT_MAX = 32, 42  # California-relevant latitude bands


class UpwellingIndicesAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        for idx in _INDICES:
            yield {"key": idx, "index": idx}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        idx = chunk["index"]
        try:
            resp = get_with_backoff(_BASE.format(idx=idx), timeout=120, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            return pd.DataFrame()
        lat_cols = [c for c in df.columns if c.endswith("N") and c[:-1].isdigit()
                    and _LAT_MIN <= int(c[:-1]) <= _LAT_MAX]
        if not lat_cols or not {"year", "month", "day"}.issubset(df.columns):
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
        long = df.melt(id_vars=["date"], value_vars=lat_cols,
                       var_name="latitude", value_name="value")
        long["latitude"] = long["latitude"].str[:-1].astype(int)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long["index_name"] = idx
        long = long[long["date"].notna() & long["value"].notna()]
        return long[["date", "latitude", "index_name", "value"]].reset_index(drop=True)
