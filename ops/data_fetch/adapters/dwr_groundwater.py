"""DWR periodic groundwater-level measurements (statewide California).

data.ca.gov CKAN resource (Department of Water Resources Periodic Groundwater Level
Measurements): ~6.25M timestamped well measurements statewide. Groundwater elevation
(`gwe`) and depth-to-water (`gse_gwe`) are a genuinely NEW modality for the lab — none
of the existing sources cover groundwater — relevant as a baseflow / seawater-intrusion
context driver for coastal water quality.

Subclasses the shared CKAN datastore_search base (offset-paged, resumable). Large page
size + high page cap so the FULL native range is fetched, not a truncated slice. Writes
only to data/external_*.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter


class DwrGroundwaterAdapter(CkanAdapter):
    RESOURCE_ID = "70f3b403-f1c3-4bc1-acf5-b6c43ba8171d"  # Periodic GWL Measurements
    PAGE_SIZE = 32000          # CKAN datastore max page
    MAX_PAGES = 260            # 260 * 32000 = 8.32M >= the ~6.25M total (full coverage)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        keep = ["site_code", "msmt_date", "gwe", "gse_gwe", "wlm_gse", "rpe_wse",
                "basin_code", "county_name", "well_use", "monitoring_program", "source"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["msmt_date"] = pd.to_datetime(df["msmt_date"], errors="coerce", utc=True)
        for col in ("gwe", "gse_gwe", "wlm_gse", "rpe_wse"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep rows with a timestamp AND a real water-level label (elevation or depth).
        label_cols = [c for c in ("gwe", "gse_gwe") if c in df.columns]
        df = df.dropna(subset=["msmt_date"])
        if label_cols:
            df = df[df[label_cols].notna().any(axis=1)]
        return df.reset_index(drop=True)
