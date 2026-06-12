"""Shared helpers for ERDDAP griddap point extractions (MUR SST, VIIRS chl-a).

These reuse the same CoastWatch ERDDAP endpoints + M1 point the existing
ops/fetch_chlorophyll.py and ops/autonomous_rate_limit_fetcher.py target, but fetch
into the framework's STAGED area (data/external_*) instead of the trusted
mbal_history cache. Chunked by year, resumable, Retry-After backoff.
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

COASTWATCH = "https://coastwatch.pfeg.noaa.gov/erddap"
M1_LAT = 36.7511
M1_LON = -122.0292


class ErddapPointAdapter(Adapter):
    """Yearly griddap point extraction. Subclasses set DATASET_ID, VARIABLE,
    OUT_TIME (output time col name), OUT_VALUE (output value col name), FIRST_YEAR.
    """

    DATASET_ID: str = ""
    VARIABLE: str = ""
    OUT_TIME: str = "time"
    OUT_VALUE: str = "value"
    FIRST_YEAR: int = 2002
    LAT: float = M1_LAT
    LON: float = M1_LON
    # Some griddap datasets (e.g. VIIRS ocean-color) carry an altitude dimension
    # between time and lat; SST (jplMURSST41) does not. Subclasses set HAS_ALTITUDE.
    HAS_ALTITUDE: bool = False
    ALTITUDE: float = 0.0

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.FIRST_YEAR
        y0 = max(y0, self.FIRST_YEAR)
        y1 = int(end[:4]) if end else dt.date.today().year
        for yr in range(y0, y1 + 1):
            yield {"key": str(yr), "year": yr}

    def _bounds(self, chunk: dict) -> tuple[str, str]:
        yr = chunk["year"]
        s = f"{yr}-01-01"
        if yr == self.FIRST_YEAR and self.DATASET_ID.lower().startswith("jplmur"):
            s = "2002-06-01T09:00:00Z"
        e = f"{yr}-12-31"
        today = dt.date.today()
        if yr == today.year:
            e = (today - dt.timedelta(days=1)).isoformat()
        return s, e

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        s, e = self._bounds(chunk)
        alt = f"%5B({self.ALTITUDE}):({self.ALTITUDE})%5D" if self.HAS_ALTITUDE else ""
        url = (
            f"{COASTWATCH}/griddap/{self.DATASET_ID}.csv?{self.VARIABLE}"
            f"%5B({s}):({e})%5D{alt}%5B({self.LAT}):({self.LAT})%5D%5B({self.LON}):({self.LON})%5D"
        )
        resp = get_with_backoff(url, timeout=600, retries=3, base_sleep=5, headers={"User-Agent": "mbal-datafetch/0.1"})
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        if not text.strip() or "Error" in text[:200] and self.VARIABLE not in text[:200]:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])  # row 1 = units
        except Exception:
            return pd.DataFrame()
        if self.VARIABLE not in df.columns or "time" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame({
            self.OUT_TIME: pd.to_datetime(df["time"], utc=True, errors="coerce"),
            self.OUT_VALUE: pd.to_numeric(df[self.VARIABLE], errors="coerce"),
        }).dropna()
        return out
