"""BEACON beach advisories/closures normalization.

Reads the TRUSTED statewide advisories (read-only) and normalizes them into a clean
BEACON-style open/close event table written to the STAGED area. Never overwrites trusted.
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import PROJECT_ROOT, Adapter

TRUSTED_ADV = PROJECT_ROOT / "bacteria_results" / "statewide" / "statewide_advisories.parquet"


class BeaconAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        yield {"key": "all"}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        if not TRUSTED_ADV.exists():
            return pd.DataFrame(columns=self.spec.required_columns)
        df = pd.read_parquet(TRUSTED_ADV)  # read-only
        ren = {
            "advisory_date": "advisory_date",
            "opened_date": "opened_date",
            "county": "county",
            "beach_name": "beach_name",
            "station_name": "station_name",
            "advisory_type": "advisory_type",
            "advisory_cause": "advisory_cause",
        }
        out = df.rename(columns=ren)
        for c in ("advisory_date", "opened_date"):
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
        for c in ("county", "beach_name", "station_name", "advisory_type", "advisory_cause"):
            if c in out.columns:
                out[c] = out[c].astype("string")
        # Normalize advisory_type into a coarse open/close status for BEACON semantics.
        if "advisory_type" in out.columns:
            t = out["advisory_type"].str.lower().fillna("")
            out["is_closure"] = t.str.contains("clos")
            out["is_advisory"] = t.str.contains("advis|warn|post", regex=True)
        return out
