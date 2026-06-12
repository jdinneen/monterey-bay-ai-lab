"""DWR continuous groundwater-level daily measurements (statewide California).

data.ca.gov CKAN (DWR Continuous Groundwater Level Measurements — Daily): ~3.13M daily
water-surface-elevation records from continuous sensor loggers. Distinct from the
periodic *manual* groundwater measurements (`dwr_groundwater`): a different network and
cadence (automated daily loggers with QC flags vs periodic hand readings). Label is
groundwater surface elevation (WSE) / depth-to-water (GSE_WSE). Writes only to
data/external_*.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter


class DwrGwContinuousAdapter(CkanAdapter):
    RESOURCE_ID = "93b82a80-1c94-4ac3-aa16-d461d4ffc6c0"  # Continuous GWL — Daily
    PAGE_SIZE = 32000
    MAX_PAGES = 110  # 110 * 32000 >= the ~3.13M total (full coverage)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        keep = ["STATION", "MSMT_DATE", "WSE", "GSE_WSE", "RPE_WSE", "WSE_QC"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.rename(columns={"STATION": "station_id", "MSMT_DATE": "date",
                                "WSE": "wse", "GSE_WSE": "depth_to_water", "RPE_WSE": "rpe_wse",
                                "WSE_QC": "wse_qc"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        for col in ("wse", "depth_to_water", "rpe_wse"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date"])
        label_cols = [c for c in ("wse", "depth_to_water") if c in df.columns]
        if label_cols:
            df = df[df[label_cols].notna().any(axis=1)]
        return df.reset_index(drop=True)
