"""CIWQS Sanitary Sewer Overflow (SSO) spill events.

Source: data.ca.gov CKAN, "Surface Water - Water Quality Regulatory Information"
wastewater-violations resource, filtered to PROGRAM CATEGORY == 'SSO'. Each SSO row
is a reported sanitary-sewer-overflow violation with an OCCURRED ON date, agency,
facility, WDID, lat/lon (where present), and a free-text description that frequently
states the spilled volume — a direct land-based contamination driver for the bacteria
model.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter


class CiwqsSsoAdapter(CkanAdapter):
    RESOURCE_ID = "e397598d-6a92-4769-a135-76fa000cb5c7"
    FILTERS = {"PROGRAM CATEGORY": "SSO"}
    PAGE_SIZE = 5000
    MAX_PAGES = 40  # ~51k SSO rows / 5k ≈ 11 pages; headroom for growth

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["spill_id", "spill_date", "agency_name",
                                         "facility_name", "wdid", "region",
                                         "latitude", "longitude", "violation_subtype",
                                         "status", "description"])
        ren = {
            "VIOLATION ID (VID)": "spill_id",
            "OCCURRED ON": "spill_date",
            "AGENCY NAME": "agency_name",
            "FACILITY NAME": "facility_name",
            "WDID": "wdid",
            "VIOLATED FACILITY REGION": "region",
            "PLACE LATITUDE": "latitude",
            "PLACE LONGITUDE": "longitude",
            "VIOLATION SUBTYPE": "violation_subtype",
            "STATUS": "status",
            "VIOLATION DESCRIPTION": "description",
        }
        out = df.rename(columns=ren)
        keep = [c for c in ren.values() if c in out.columns]
        out = out[keep].copy()
        out["spill_date"] = pd.to_datetime(out["spill_date"], errors="coerce")
        for c in ("latitude", "longitude"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out["spill_id"] = out["spill_id"].astype("string")
        return out
