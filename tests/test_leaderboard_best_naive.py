#!/usr/bin/env python3
"""Tests for ops/leaderboard_best_naive.augment_leaderboard.

Guards the fix that closes the persistence-only trap: a leaderboard must gain an
honest best-naive skill column, and a model that beats both persistence and
same-hour-yesterday must score positive skill_vs_best_naive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.leaderboard_best_naive import augment_leaderboard  # noqa: E402


def _obs_panel():
    # One series, 48 hourly observed points with distinct values so persistence
    # and seasonal-naive both have nonzero error.
    base = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")  # match seasonal_naive._to_dt (utc=True)
    ds = [base + pd.Timedelta(hours=i) for i in range(48)]
    y = np.sin(np.arange(48) / 3.0) + np.arange(48) * 0.01  # smooth, monotone-ish, no repeats
    df = pd.DataFrame({"unique_id": "x", "ds": ds, "y": y})
    return df.set_index(["unique_id", "ds"])["y"], base, y


def test_augment_adds_best_naive_and_scores_perfect_model():
    obs, base, y = _obs_panel()
    # h=1 predictions at t=24..28; cutoff=ds-1h; seasonal ref = ds-24h (all in range).
    rows = []
    for i in range(24, 29):
        ds = base + pd.Timedelta(hours=i)
        rows.append({"unique_id": "x", "ds": ds, "cutoff": ds - pd.Timedelta(hours=1), "NHITS": y[i]})
    preds = pd.DataFrame(rows)  # NHITS == truth -> model_rmse 0 -> skill 100%

    leaderboard = pd.DataFrame(
        {"unique_id": ["x"], "horizon_h": [1], "model_rmse": [0.0], "skill_vs_persistence_pct": [1.0]}
    )

    out = augment_leaderboard(leaderboard, preds, obs)

    # The honest-baseline column now exists and the original rows are preserved.
    assert "skill_vs_best_naive_pct" in out.columns
    assert len(out) == len(leaderboard)
    cell = out[(out["unique_id"] == "x") & (out["horizon_h"] == 1)].iloc[0]
    # Perfect predictions beat any naive baseline -> ~100% skill, and a real baseline RMSE exists.
    assert cell["skill_vs_best_naive_pct"] > 99.0
    assert cell["best_naive_rmse"] > 0


def test_augment_is_left_join_keeps_unscored_rows():
    obs, base, y = _obs_panel()
    preds = pd.DataFrame(
        [{"unique_id": "x", "ds": base + pd.Timedelta(hours=25),
          "cutoff": base + pd.Timedelta(hours=24), "NHITS": y[25]}]
    )  # only 1 obs < min_observations(3) -> no scored cell
    leaderboard = pd.DataFrame({"unique_id": ["x"], "horizon_h": [1], "skill_vs_persistence_pct": [5.0]})

    out = augment_leaderboard(leaderboard, preds, obs)

    assert len(out) == 1  # row kept
    assert "skill_vs_best_naive_pct" in out.columns
    assert pd.isna(out.iloc[0]["skill_vs_best_naive_pct"])  # left NaN, not dropped
