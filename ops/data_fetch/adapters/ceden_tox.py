"""CEDEN surface-water toxicity bioassay results (data.ca.gov).

Source: data.ca.gov CKAN, "Surface Water - Toxicity Results" datastore resource
(~1.7M rows). A *distinct modality* from CEDEN chemistry: lab toxicity bioassay
endpoints (organism survival/growth, % control, % effect) against test species,
timestamped by SampleDate and labeled by StationCode + OrganismName + Analyte.
A direct biological water-quality / contamination signal.

Lane B (claude-fetch-2). Paginated via the shared CkanAdapter base.
"""
from __future__ import annotations

import pandas as pd

from .ckan import CkanAdapter

_KEEP = [
    "Program", "Project", "StationName", "StationCode", "SampleDate", "CollectionTime",
    "MatrixName", "MethodName", "OrganismName", "ToxTestDurCode", "Analyte", "AnalyteCode",
    "Unit", "Result", "ResultQualCode", "Mean", "StdDev", "PctControl", "PercentEffect",
    "SigEffectCode", "Latitude", "Longitude", "ToxBatch", "LabReplicate", "DataQuality",
]


class CedenToxicityAdapter(CkanAdapter):
    RESOURCE_ID = "bd484e9b-426a-4ba6-ba4d-f5f8ce095836"
    PAGE_SIZE = 5000
    MAX_PAGES = 400  # ~1.71M / 5000 ≈ 342 pages; headroom (full pull)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=_KEEP)
        out = df[[c for c in _KEEP if c in df.columns]].copy()
        out["SampleDate"] = pd.to_datetime(out.get("SampleDate"), errors="coerce")
        for c in ("Result", "Mean", "StdDev", "PctControl", "PercentEffect",
                  "Latitude", "Longitude"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        for c in ("StationCode", "OrganismName", "Analyte", "AnalyteCode", "Unit"):
            if c in out.columns:
                out[c] = out[c].astype("string")
        return out
