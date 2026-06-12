"""Open-Meteo ERA5 archive — hourly atmospheric reanalysis over the CA coastal grid.

Source: ``https://archive-api.open-meteo.com/v1/archive`` (open, keyless). Unlike the
Open-Meteo *marine forecast* API (which has no historical wave labels), the ERA5
**archive** endpoint serves full, non-null hourly reanalysis history. One call returns
a point's full hourly series; we tile the CA coastal grid and chunk one point per
resumable chunk. Long-format rows (point × hour × variable): 2 m temperature, relative
humidity, dewpoint, precipitation, 10 m wind speed/direction, surface pressure,
shortwave radiation, cloud cover — hourly atmospheric drivers distinct from the
existing DAILY rainfall wrapper.

Lane B (claude-fetch-2).
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff
from .nasa_power import _ca_coastal_grid

_BASE = "https://archive-api.open-meteo.com/v1/archive"
_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
    "shortwave_radiation", "cloud_cover",
]


class OpenMeteoArchiveAdapter(Adapter):
    START = "2010-01-01"
    END = "2024-12-31"

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = start or self.START
        e = end or self.END
        for lat, lon in _ca_coastal_grid():
            base_key = f"{lat}_{lon}"
            key = base_key if (s == self.START and e == self.END) else f"{base_key}_{s}_{e}"
            yield {"key": key, "lat": lat, "lon": lon, "start": s, "end": e}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        url = (f"{_BASE}?latitude={chunk['lat']}&longitude={chunk['lon']}"
               f"&start_date={chunk['start']}&end_date={chunk['end']}"
               f"&hourly={','.join(_HOURLY)}&timezone=UTC")
        try:
            resp = get_with_backoff(url, timeout=180, retries=4,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            hourly = resp.json().get("hourly", {})
        except Exception:
            return pd.DataFrame()
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()
        base = pd.to_datetime(pd.Series(times), errors="coerce", utc=True)
        frames = []
        for p in _HOURLY:
            vals = hourly.get(p)
            if not vals:
                continue
            d = pd.DataFrame({"time": base, "value": pd.to_numeric(pd.Series(vals), errors="coerce")})
            d["parameter"] = p
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["grid_lat"] = chunk["lat"]
        out["grid_lon"] = chunk["lon"]
        out = out[out["time"].notna() & out["value"].notna()]
        return out[["grid_lat", "grid_lon", "time", "parameter", "value"]].reset_index(drop=True)
