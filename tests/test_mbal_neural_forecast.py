#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbal_neural_forecast import target_missingness_features, merge_drivers  # noqa: E402


def test_target_missingness_features_are_causal_gap_state():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")
    observed = pd.Series([True, False, False, True, False], index=idx)

    features = target_missingness_features(observed)

    assert features["target_was_observed"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert features["target_was_filled"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert features["target_hours_since_observed"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_merge_drivers_zero_fills_without_extending_stale_values_or_global_mean():
    long_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b", "b", "b"],
            "ds": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                ]
            ),
            "y": [1, 2, 3, 4, 5, 6],
        }
    )
    drivers = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"]),
            "wind": [10.0, 20.0],
        }
    )

    merged = merge_drivers(long_df, drivers, logfn=lambda *_: None)

    assert merged.loc[(merged["unique_id"] == "a") & (merged["ds"].dt.hour == 1), "wind"].item() == 0.0
    assert merged.loc[(merged["unique_id"] == "b") & (merged["ds"].dt.hour == 1), "wind"].item() == 0.0
    assert 15.0 not in set(merged["wind"])
