"""MUR SST backfill (M1 point), staged. Reuses the CoastWatch ERDDAP MUR dataset
the existing ops/autonomous_rate_limit_fetcher.py / fetch_mur_sst_resumable.py target,
but writes into data/external_* and is chunked/resumable by year for a 2020-2026 backfill.
"""
from __future__ import annotations

from .erddap import ErddapPointAdapter


class MurAdapter(ErddapPointAdapter):
    DATASET_ID = "jplMURSST41"
    VARIABLE = "analysed_sst"
    OUT_TIME = "time"
    OUT_VALUE = "sst_c"
    FIRST_YEAR = 2002
