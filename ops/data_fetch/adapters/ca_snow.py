"""California snow water content — monthly snow-course measurements (statewide CA).

data.ca.gov CKAN (CA DWR Snow Water Content): ~191k monthly snow-water-equivalent
measurements at mountain snow courses since 1930. A genuinely new modality — snowpack —
relevant to snowmelt-driven spring streamflow timing that conditions coastal water
quality. Single complete CKAN resource (not a truncated slice). Writes only to
data/external_*.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter


class CaSnowAdapter(CkanAdapter):
    RESOURCE_ID = "8569ccdc-aab8-4d41-a026-7126f0bf841d"  # Monthly Snow Water Content
    PAGE_SIZE = 32000
    MAX_PAGES = 12  # 12*32000 >> 191k total (full coverage)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        keep = ["STATION_ID", "SENSOR_NUM", "SENSOR_TYPE", "OBS_DATE", "VALUE", "UNITS"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["date"] = pd.to_datetime(df["OBS_DATE"], format="%Y%m%d %H%M", errors="coerce", utc=True)
        df["value"] = pd.to_numeric(df["VALUE"], errors="coerce")
        df = df.rename(columns={"STATION_ID": "station_id", "SENSOR_TYPE": "sensor_type",
                                "SENSOR_NUM": "sensor_num", "UNITS": "units"})
        df = df.dropna(subset=["date", "value"])
        return df.reset_index(drop=True)
