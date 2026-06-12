#!/usr/bin/env python3
"""
Generate 'Golden Era' (2004-2011) split contracts for multi-sensor modeling.

This ensures models can be trained and evaluated strictly on the timeframe where
both M1 and M2 moorings have valid overlapping data.
"""
import os
from pathlib import Path

import pandas as pd

from mbal_split_contracts import build_split_contract, write_split_contract

PROJECT_ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
LAKEHOUSE = PROJECT_ROOT / "lakehouse"
SOURCE_PARQUET = PROJECT_ROOT / "mbal_pipeline" / "curated_history"

def generate_golden_splits():
    # We define the Golden Era explicitly as 2004-10-01 to 2011-12-31.
    train_start = pd.Timestamp("2004-10-01T00:00:00Z")
    era_end = pd.Timestamp("2011-12-31T23:59:59Z")
    
    print(f"Golden Era bounds: {train_start} to {era_end}")

    # Generate 5 expanding window cutoffs inside this era.
    # Start the first fold halfway through the era to ensure enough training data.
    cv_start = train_start + (era_end - train_start) * 0.5
    
    cutoffs = pd.date_range(cv_start, era_end, periods=5).round("1h")
    
    print("Cutoffs:", [str(c) for c in cutoffs])
    
    # We'll use the neural horizons by default
    horizons = [1, 6, 24, 72, 168]
    
    contract = build_split_contract(
        source_path=SOURCE_PARQUET,
        cutoffs=cutoffs,
        train_start=train_start,
        horizons=horizons,
        cache_version="golden_era_v1",
        config={"period": "2004-2011", "description": "Golden Era Multi-Sensor Overlap"},
        family="shared",
        alias="golden-era-2004-2011",
    )
    
    print(f"Generated Split ID: {contract.split_id}")
    split_dir = write_split_contract(contract, LAKEHOUSE)
    print(f"Wrote split to: {split_dir}")

if __name__ == "__main__":
    generate_golden_splits()
