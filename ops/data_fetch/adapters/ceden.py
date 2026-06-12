"""CEDEN surface-water chemistry results (California Environmental Data Exchange Network).

Source: data.ca.gov CKAN, "Surface Water - Chemistry Results - CEDEN Augmentation
(Field & Lab Chemistry)" datastore resource. ~1.84M timestamped analyte results
statewide (StationCode + SampleDate + Analyte + Result), a broad water-chemistry
driver/label corpus complementary to the bacteria (WQP) and nutrient sources.

Lane B (claude-fetch-2). Trivial CKAN pagination via the shared CkanAdapter base.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter

_KEEP = [
    "Program", "Project", "StationName", "StationCode", "SampleDate", "CollectionTime",
    "MatrixName", "MethodName", "Analyte", "AnalyteCode", "Unit", "Result", "ResultQualCode",
    "MDL", "RL", "QACode", "Latitude", "Longitude", "SampleTypeCode",
    "CollectionReplicate", "ResultsReplicate", "DataQuality",
]


class CedenWaterChemAdapter(CkanAdapter):
    RESOURCE_ID = "e07c5e0b-cace-4b70-9f13-b3e696cd5a99"
    PAGE_SIZE = 5000
    MAX_PAGES = 400  # ~1.84M / 5000 ≈ 368 pages; headroom for growth (full pull)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=_KEEP)
        out = df[[c for c in _KEEP if c in df.columns]].copy()
        # SampleDate is ISO ("2017-04-18T00:00:00"); fold in CollectionTime when present.
        out["SampleDate"] = pd.to_datetime(out.get("SampleDate"), errors="coerce")
        for c in ("Result", "Latitude", "Longitude", "MDL", "RL"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in ("StationCode", "Analyte", "AnalyteCode", "Unit", "MethodName"):
            if c in out.columns:
                out[c] = out[c].astype("string")
        return out
