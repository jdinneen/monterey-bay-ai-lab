"""HF radar surface currents (Monterey Bay), CoastWatch ERDDAP griddap.

CoastWatch serves the HFRNet US West Coast 6km hourly surface-current grid
(`ucsdHfrW6`) as a ROLLING near-real-time dataset: only the most recent ~92 days
are retained (the full multi-year archive lives on the HFRNet THREDDS server, a
separate integration). This adapter discovers the live time coverage, clamps any
requested range to it, and fetches one day per chunk over the Monterey Bay bbox.
Variables/dimensions: water_u/water_v[time][latitude][longitude] (no altitude).
"""
from __future__ import annotations

import datetime as dt
import io
import re
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

# CoastWatch West-Coast HF radar 6km hourly (public ERDDAP griddap).
ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET = "ucsdHfrW6"  # HF Radar, US West Coast, 6km, hourly (rolling ~92-day window)
# Monterey Bay bbox
LAT0, LAT1 = 36.5, 37.0
LON0, LON1 = -122.2, -121.8


class HfRadarAdapter(Adapter):
    def _live_window(self) -> tuple[Optional[str], Optional[str]]:
        """Discover the dataset's live time window (rolling NRT) from the .das."""
        try:
            resp = get_with_backoff(f"{ERDDAP}/griddap/{DATASET}.das", timeout=60,
                                    retries=2, base_sleep=3, headers={"User-Agent": "mbal-datafetch/0.1"})
            das = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        except Exception:
            return None, None
        s = re.search(r'time_coverage_start\s+"(.*?)"', das)
        e = re.search(r'time_coverage_end\s+"(.*?)"', das)
        return (s.group(1)[:10] if s else None, e.group(1)[:10] if e else None)

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        cov_s, cov_e = self._live_window()
        if not cov_s or not cov_e:
            return  # dataset unreachable -> no chunks; source stays empty/honest
        lo = max(start[:10], cov_s) if start else cov_s
        hi = min(end[:10], cov_e) if end else cov_e
        try:
            d = dt.date.fromisoformat(lo)
            last = dt.date.fromisoformat(hi)
        except ValueError:
            return
        while d <= last:
            yield {"key": d.isoformat(), "date": d.isoformat()}
            d += dt.timedelta(days=1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        day = chunk["date"]
        s, e = f"{day}T00:00:00Z", f"{day}T23:00:00Z"
        box = f"%5B({LAT0}):({LAT1})%5D%5B({LON0}):({LON1})%5D"
        url = (
            f"{ERDDAP}/griddap/{DATASET}.csv?"
            f"water_u%5B({s}):({e})%5D{box},water_v%5B({s}):({e})%5D{box}"
        )
        try:
            resp = get_with_backoff(url, timeout=180, retries=2, base_sleep=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
        except Exception:
            return pd.DataFrame()
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        if not text.strip() or text.lstrip().startswith("<") or "water_u" not in text[:300]:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])  # row 1 = units
        except Exception:
            return pd.DataFrame()
        out = pd.DataFrame({
            "time": pd.to_datetime(df.get("time"), utc=True, errors="coerce"),
            "lat": pd.to_numeric(df.get("latitude"), errors="coerce"),
            "lon": pd.to_numeric(df.get("longitude"), errors="coerce"),
            "u": pd.to_numeric(df.get("water_u"), errors="coerce"),
            "v": pd.to_numeric(df.get("water_v"), errors="coerce"),
        }).dropna(subset=["time", "u", "v"])  # keep only real current cells
        return out
