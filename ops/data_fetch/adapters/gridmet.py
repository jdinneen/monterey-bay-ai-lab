"""gridMET daily 4km gridded surface meteorology over the CA coastal grid.

Source: University of Idaho gridMET aggregated OPeNDAP
``http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_<var>_1979_CurrentYear_CONUS.nc``
(open data, no credentials). Daily precip, max/min air temperature, max/min relative
humidity, wind speed, vapor-pressure deficit, and shortwave radiation at ~4km — a
gridded driver complementary to NASA POWER (independent product/resolution) and the
Open-Meteo rainfall grid. Points × ~20 yr × 8 variables >> 50k labeled rows.

Lane B (claude-fetch-2). Resumable: one chunk per variable (each opens one OPeNDAP
dataset and extracts all grid points).
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter
from .nasa_power import _ca_coastal_grid

_BASE = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{var}_1979_CurrentYear_CONUS.nc"
# gridMET variable code -> output parameter label
_VARS = {
    "pr": "precip_mm", "tmmx": "tmax_k", "tmmn": "tmin_k",
    "rmax": "rh_max_pct", "rmin": "rh_min_pct", "vs": "wind_ms",
    "vpd": "vpd_kpa", "srad": "srad_wm2",
}


class GridmetDailyAdapter(Adapter):
    START = "2005-01-01"
    END = "2024-12-31"

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = start or self.START
        e = end or self.END
        for var, label in _VARS.items():
            key = var if (s == self.START and e == self.END) else f"{var}_{s}_{e}"
            yield {"key": key, "var": var, "label": label, "start": s, "end": e}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        import xarray as xr  # lazy; heavy dep

        url = _BASE.format(var=chunk["var"])
        try:
            ds = xr.open_dataset(url, decode_times=True)
        except Exception:
            return pd.DataFrame()
        try:
            dvar = list(ds.data_vars)[0]
            latn = [c for c in ds.coords if "lat" in c.lower()][0]
            lonn = [c for c in ds.coords if "lon" in c.lower()][0]
            timen = "day" if "day" in ds.dims else [d for d in ds.dims if "time" in d.lower() or d == "day"][0]
            da = ds[dvar].sel({timen: slice(chunk["start"], chunk["end"])})
            frames = []
            for lat, lon in _ca_coastal_grid():
                series = da.sel({latn: lat, lonn: lon}, method="nearest")
                vals = series.values
                times = pd.to_datetime(series[timen].values)
                d = pd.DataFrame({"date": times, "value": vals})
                d["grid_lat"] = lat
                d["grid_lon"] = lon
                d["parameter"] = chunk["label"]
                frames.append(d)
        finally:
            ds.close()
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out[out["value"].notna()]
        return out[["grid_lat", "grid_lon", "date", "parameter", "value"]].reset_index(drop=True)
