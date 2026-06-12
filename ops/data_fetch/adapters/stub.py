"""Stub adapter for sources with no usable public bulk API located at build time
(Surfrider BWTF, C-CAP/NHDPlus/NLCD bulk rasters). These declare needs_credentials so
the base Adapter classifies them IMPLEMENTED_NOT_FETCHED with the documented missing
inputs, rather than pretending to fetch. If the operator provides the documented local
export path, fetch_chunk reads it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter


class StubAdapter(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        yield {"key": "local"}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        # Only reached if MBAL_* credential env points at a local export.
        for env in self.spec.credentials_env:
            val = os.environ.get(env)
            if val and Path(val).exists() and val.lower().endswith((".csv", ".parquet")):
                if val.lower().endswith(".csv"):
                    return pd.read_csv(val)
                return pd.read_parquet(val)
        return pd.DataFrame(columns=self.spec.required_columns)
