"""NOAA CO-OPS meteorology / oceanography at California stations (staged).

Hourly water temperature, air temperature, and wind at CA CO-OPS stations — coastal
mixing / temperature drivers that pair with the existing tide source. Distinct from
the tide water-level product. One station-product-year is ~8.8k hourly rows, so a
handful of station-years clears 50k.

API: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter (interval=h, yearly).
Staged + resumable: one chunk per (station, product, year). Writes only to data/external_*.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

STATIONS = {
    "9414290": "San Francisco", "9410660": "Los Angeles", "9410170": "San Diego Bay",
    "9411340": "Santa Barbara", "9413450": "Monterey", "9415020": "Point Reyes",
    "9418767": "North Spit (Humboldt)", "9419750": "Crescent City",
    "9412110": "Port San Luis", "9410230": "La Jolla", "9415144": "Port Chicago",
    "9416841": "Arena Cove",
}
PRODUCTS = {
    "water_temperature": "water_temp_c",
    "air_temperature": "air_temp_c",
    "wind": "wind_speed_ms",
}


class CoopsMetAdapter(Adapter):
    """One chunk per (station, product, year); hourly verified met series."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(str(start)[:4]) if start else 2010
        y1 = int(str(end)[:4]) if end else (dt.date.today().year - 1)
        for station_id, name in STATIONS.items():
            for product, label in PRODUCTS.items():
                for year in range(y0, y1 + 1):
                    yield {
                        "key": f"{station_id}_{product}_{year}",
                        "station_id": station_id, "station_name": name,
                        "product": product, "label": label, "year": year,
                    }

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        params = {
            "product": chunk["product"], "application": "Monterey_Bay_AI_Lab_data_fetcher",
            "begin_date": f"{chunk['year']}0101", "end_date": f"{chunk['year']}1231",
            "station": chunk["station_id"], "time_zone": "gmt", "units": "metric",
            "interval": "h", "format": "json",
        }
        try:
            payload = get_with_backoff(
                f"{API}?{urlencode(params)}", timeout=90, retries=3,
                headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
            ).json()
        except Exception:
            return pd.DataFrame()
        data = payload.get("data") or []
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "t" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame({
            "station_id": chunk["station_id"],
            "station_name": chunk["station_name"],
            "time": pd.to_datetime(df["t"], errors="coerce", utc=True),
            "product": chunk["product"],
            "value": pd.to_numeric(df.get("v"), errors="coerce"),
        })
        # wind payloads use 's' for speed; prefer it when present.
        if chunk["product"] == "wind" and "s" in df.columns:
            out["value"] = pd.to_numeric(df["s"], errors="coerce")
        out["units"] = "metric"
        return out.dropna(subset=["time", "value"]).reset_index(drop=True)
