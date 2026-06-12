"""NASA POWER daily agro-meteorology over a California coastal grid.

Source: ``https://power.larc.nasa.gov/api/temporal/daily/point`` (keyless). One API
call returns the full daily series for a point; we tile a coastal CA grid and chunk
one point per resumable chunk. Long-format rows (point × date × parameter):
precipitation, 2 m air temperature (mean/min/max), relative humidity, wind speed,
and surface shortwave — gridded met drivers for beach-bacteria / runoff modeling,
complementary to the Open-Meteo rainfall grid (independent reanalysis).

Lane B (claude-fetch-2). ~grid-points × ~20 yr × params >> 50k labeled rows.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import pandas as pd

from ..core import Adapter, get_with_backoff

_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
_PARAMS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "ALLSKY_SFC_SW_DWN"]


def _ca_coastal_grid() -> List[Tuple[float, float]]:
    """Coastal CA grid points (lat 32.5–41.5 by 0.5°, near-shore longitudes)."""
    band = {
        32.5: -117.1, 33.0: -117.3, 33.5: -118.0, 34.0: -118.5, 34.5: -120.5,
        35.0: -120.7, 35.5: -121.1, 36.0: -121.6, 36.5: -121.9, 37.0: -122.4,
        37.5: -122.5, 38.0: -123.0, 38.5: -123.3, 39.0: -123.7, 39.5: -123.8,
        40.0: -124.1, 40.5: -124.3, 41.0: -124.2, 41.5: -124.2,
    }
    return [(lat, lon) for lat, lon in band.items()]


class NasaPowerDailyAdapter(Adapter):
    START = "20050101"
    END = "20241231"

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = (start or self.START).replace("-", "")[:8]
        e = (end or self.END).replace("-", "")[:8]
        for lat, lon in _ca_coastal_grid():
            yield {"key": f"{lat}_{lon}", "lat": lat, "lon": lon, "start": s, "end": e}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        url = (f"{_BASE}?parameters={','.join(_PARAMS)}&community=AG"
               f"&longitude={chunk['lon']}&latitude={chunk['lat']}"
               f"&start={chunk['start']}&end={chunk['end']}&format=JSON")
        try:
            resp = get_with_backoff(url, timeout=120, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            params = resp.json().get("properties", {}).get("parameter", {})
        except Exception:
            return pd.DataFrame()
        frames = []
        for pname, series in params.items():
            if not isinstance(series, dict):
                continue
            d = pd.DataFrame({"date": list(series.keys()),
                              "value": list(series.values())})
            d["parameter"] = pname
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out[out["value"] > -900]  # POWER fill value is -999
        out["grid_lat"] = chunk["lat"]
        out["grid_lon"] = chunk["lon"]
        return out[["grid_lat", "grid_lon", "date", "parameter", "value"]].reset_index(drop=True)
