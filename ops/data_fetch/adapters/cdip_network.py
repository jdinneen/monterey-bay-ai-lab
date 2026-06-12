"""CDIP wave-buoy NETWORK (all California buoys) via CDIP ERDDAP tabledap.

Source: ``https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg`` (open). Aggregated
wave measurements across the full CDIP buoy network — significant wave height, peak/
average period, peak direction — for every CA-coast station, hourly. Distinct from the
existing `cdip_waves` source, which wraps a single trusted buoy (≈NDBC 46042). One
ERDDAP call per year (CA bbox), resumable. ~9.8M rows/yr across ~44 CA buoys.

Lane B (claude-fetch-2).
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

_BASE = "https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.csv"
_VARS = "station_id,time,latitude,longitude,waveHs,waveTp,waveDp,waveTa,waveTz"
_LAT_MIN, _LAT_MAX = 32.0, 42.5


class CdipWaveNetworkAdapter(Adapter):
    # Sub-hourly × ~44 buoys ≈ 9.8M rows/yr. Default to a recent multi-year window to
    # keep the pull tractable; the full 2000→ archive is reachable via --start.
    FIRST_YEAR = 2021

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.FIRST_YEAR
        y1 = int(end[:4]) if end else dt.date.today().year
        for yr in range(y0, y1 + 1):
            yield {"key": str(yr), "year": yr}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        yr = chunk["year"]
        url = (f"{_BASE}?{_VARS}"
               f"&time%3E={yr}-01-01T00:00:00Z&time%3C={yr + 1}-01-01T00:00:00Z"
               f"&latitude%3E={_LAT_MIN}&latitude%3C={_LAT_MAX}")
        try:
            resp = get_with_backoff(url, timeout=600, retries=3, base_sleep=5,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        except Exception:
            return pd.DataFrame()
        if not text.strip() or text[:60].lstrip().startswith("Error") or "station_id" not in text[:200]:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])  # row 1 = ERDDAP units
        except Exception:
            return pd.DataFrame()
        if "time" not in df.columns:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("waveHs", "waveTp", "waveDp", "waveTa", "waveTz", "latitude", "longitude"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["station_id"] = df["station_id"].astype("string")
        return df[df["time"].notna()].reset_index(drop=True)


class CdipSstNetworkAdapter(Adapter):
    """In-situ sea-surface temperature across the CDIP buoy network (sst_agg).

    Distinct from satellite `mur_sst` (model/satellite, single M1 point): this is
    measured buoy SST at every CA CDIP station — ground-truth temperature for the
    HAB / marine-heatwave driver set. ~324k rows/week across CA buoys.
    """

    FIRST_YEAR = 2021  # ~17M rows/yr; recent default, full archive via --start
    _SST = "https://erddap.cdip.ucsd.edu/erddap/tabledap/sst_agg.csv"
    _VARS = "station_id,time,latitude,longitude,sstSeaSurfaceTemperature"

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.FIRST_YEAR
        y1 = int(end[:4]) if end else dt.date.today().year
        for yr in range(y0, y1 + 1):
            yield {"key": str(yr), "year": yr}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        yr = chunk["year"]
        url = (f"{self._SST}?{self._VARS}"
               f"&time%3E={yr}-01-01T00:00:00Z&time%3C={yr + 1}-01-01T00:00:00Z"
               f"&latitude%3E={_LAT_MIN}&latitude%3C={_LAT_MAX}")
        try:
            resp = get_with_backoff(url, timeout=600, retries=3, base_sleep=5,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        except Exception:
            return pd.DataFrame()
        if not text.strip() or text[:60].lstrip().startswith("Error") or "station_id" not in text[:200]:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])
        except Exception:
            return pd.DataFrame()
        if "time" not in df.columns:
            return pd.DataFrame()
        df = df.rename(columns={"sstSeaSurfaceTemperature": "sst_c"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        for c in ("sst_c", "latitude", "longitude"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["station_id"] = df["station_id"].astype("string")
        df = df[df["time"].notna() & df["sst_c"].notna()]
        return df.reset_index(drop=True)
