"""California Harmful Algae Risk Mapping (C-HARM) adapter.

Fetches the operational C-HARM Nowcast V3 dataset from NOAA CoastWatch ERDDAP.
Provides statewide 3km gridded probabilities for:
- pseudo_nitzschia
- particulate_domoic
- cellular_domoic

Chunked by month to avoid ERDDAP proxy timeouts. Drops NaNs (land/open ocean)
to keep the staged parquet files extremely compact.
"""
from __future__ import annotations

import datetime as dt
import io
import urllib.parse
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

COASTWATCH = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET_ID = "wvcharmV3_0day_LonPM180"

# CA Bounding Box
LAT_MIN, LAT_MAX = 32.5, 42.0
LON_MIN, LON_MAX = -124.5, -117.0

# V3 variables
VARIABLES = ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic"]

class CharmAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        # C-HARM V3 starts around late 2018; fallback to user bounds
        y0 = int(start[:4]) if start else 2018
        m0 = int(start[5:7]) if start and len(start) >= 7 else 1
        
        y1 = int(end[:4]) if end else dt.date.today().year
        m1 = int(end[5:7]) if end and len(end) >= 7 else dt.date.today().month

        start_date = dt.date(y0, m0, 1)
        end_date = dt.date(y1, m1, 1)

        curr = start_date
        while curr <= end_date:
            yield {"key": curr.strftime("%Y-%m"), "year": curr.year, "month": curr.month}
            # Move to next month
            if curr.month == 12:
                curr = dt.date(curr.year + 1, 1, 1)
            else:
                curr = dt.date(curr.year, curr.month + 1, 1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        year = chunk["year"]
        month = chunk["month"]
        
        s = f"{year}-{month:02d}-01"
        # Find the last day of the month
        if month == 12:
            next_m = dt.date(year + 1, 1, 1)
        else:
            next_m = dt.date(year, month + 1, 1)
        e = (next_m - dt.timedelta(days=1)).isoformat()
        
        # Ensure we do not query into the future
        today = dt.date.today()
        if dt.date.fromisoformat(e) >= today:
            e = (today - dt.timedelta(days=1)).isoformat()
            if dt.date.fromisoformat(s) >= today:
                return pd.DataFrame()

        lat_bound = f"%5B({LAT_MIN}):1:({LAT_MAX})%5D"
        lon_bound = f"%5B({LON_MIN}):1:({LON_MAX})%5D"
        time_bound = f"%5B({s}T12:00:00Z):1:({e}T12:00:00Z)%5D"
        bounds = f"{time_bound}{lat_bound}{lon_bound}"
        
        var_str = ",".join([f"{v}{bounds}" for v in VARIABLES])
        
        url = f"{COASTWATCH}/griddap/{DATASET_ID}.csv?{var_str}"
        
        resp = get_with_backoff(url, timeout=900, retries=3, base_sleep=5, headers={"User-Agent": "mbal-datafetch/0.1"})
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        
        if not text.strip() or "Error" in text[:200]:
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])  # row 1 = units
        except Exception:
            return pd.DataFrame()
            
        if "time" not in df.columns:
            return pd.DataFrame()
            
        # Clean up output
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for v in VARIABLES:
            if v in df.columns:
                df[v] = pd.to_numeric(df[v], errors="coerce")
        
        # Drop rows where all HAB probabilities are NaN (land / open ocean)
        df = df.dropna(subset=VARIABLES, how="all")
        
        return df.dropna(subset=["time"])
