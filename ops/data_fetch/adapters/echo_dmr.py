"""Permitted wastewater effluent monitoring.

The federal EPA ECHO `dmr_rest_services` endpoint was unreachable (HTTP 500) at build
time, so this adapter targets the authoritative California equivalent: the State Water
Board electronic Self-Monitoring Report (eSMR) analytical data on data.ca.gov, which
holds permitted-discharger effluent results (the CA slice of what ECHO DMR aggregates
federally). Chunked by CKAN page; the default cap is high enough for the complete
2024 annual resource and can still be lowered with MBAL_ESMR_MAX_PAGES for smoke tests.
"""
from __future__ import annotations

import os

import pandas as pd

from .ckan import CkanAdapter


class EchoDmrAdapter(CkanAdapter):
    RESOURCE_ID = "7adb8aea-62fb-412f-9e67-d13b0729222f"  # 2024 eSMR analytical data
    PAGE_SIZE = 5000
    MAX_PAGES = int(os.environ.get("MBAL_ESMR_MAX_PAGES", "1000"))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["npdes_id", "monitoring_period_end_date",
                                         "facility_name", "parameter", "result", "units",
                                         "latitude", "longitude", "receiving_water_body"])
        ren = {
            "location_place_id": "npdes_id",
            "sampling_date": "monitoring_period_end_date",
            "facility_name": "facility_name",
            "parameter": "parameter",
            "result": "result",
            "units": "units",
            "latitude": "latitude",
            "longitude": "longitude",
            "receiving_water_body": "receiving_water_body",
        }
        out = df.rename(columns=ren)
        keep = [c for c in ren.values() if c in out.columns]
        out = out[keep].copy()
        out["monitoring_period_end_date"] = pd.to_datetime(out["monitoring_period_end_date"], errors="coerce")
        for c in ("latitude", "longitude", "result"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        # synthesize a parameter_code for dedup (eSMR has no separate code column)
        out["parameter_code"] = out.get("parameter", pd.Series(dtype="string")).astype("string")
        out["npdes_id"] = out["npdes_id"].astype("string")
        return out
