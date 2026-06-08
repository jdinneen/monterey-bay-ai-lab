#!/usr/bin/env python3
"""Unit tests for the shared best-naive (persistence vs seasonal-naive) scorer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.seasonal_naive import (  # noqa: E402
    load_observed_panel,
    score_predictions,
    seasonal_naive_table,
)


def _write_panel(root: Path, uid: str, hours: int) -> pd.Series:
    cache = root / "nn_cache"
    cache.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=hours, freq="1h")
    y = np.arange(hours, dtype=float)
    long_df = pd.DataFrame({"unique_id": uid, "ds": idx, "y": y})
    mask = pd.DataFrame({"unique_id": uid, "ds": idx, "observed": True})
    long_df.to_parquet(cache / "long_v2_past_only_fill_origin_observed_full.parquet", index=False)
    mask.to_parquet(cache / "mask_v2_past_only_fill_origin_observed_full.parquet", index=False)
    return pd.Series(y, index=pd.MultiIndex.from_arrays([[uid] * hours, idx], names=["unique_id", "ds"]))


def _write_predictions(root: Path, run_id: str, split_id: str, uid: str, ds, cutoff, model_pred) -> None:
    part = root / "lakehouse" / "gold" / "forecast_predictions" / f"run_id={run_id}"
    part.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_id": run_id,
            "split_id": split_id,
            "unique_id": uid,
            "ds": ds,
            "cutoff": cutoff,
            "y": np.nan,  # truth is derived from the observed panel, not this column
            "ModelPred": model_pred,
            "model_prediction_col": "ModelPred",
        }
    ).to_parquet(part / "predictions.parquet", index=False)


def test_seasonal_naive_table_scores_horizon_one_against_both_baselines(tmp_path):
    # Truth y = 0..47 hourly. A perfect +1h forecast should read 100% skill vs
    # the best naive: persistence (y-1, rmse 1) beats seasonal-naive (y-24, rmse 24).
    _write_panel(tmp_path, "a", 48)
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="1h")
    ds = idx[24:]  # ds-24h must exist
    cutoff = ds - pd.Timedelta(hours=1)
    model_pred = np.arange(24, 48, dtype=float)  # perfect

    _write_predictions(tmp_path, "run-a", "split-a", "a", ds, cutoff, model_pred)

    table = seasonal_naive_table(tmp_path)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["horizon_h"] == 1
    assert row["n_best_naive"] == 24
    assert row["model_rmse_common"] == 0.0
    assert np.isclose(row["persistence_rmse_common"], 1.0)
    assert np.isclose(row["seasonal_naive_rmse"], 24.0)
    assert np.isclose(row["best_naive_rmse"], 1.0)
    assert np.isclose(row["skill_vs_best_naive_pct"], 100.0)


def test_seasonal_naive_flags_model_worse_than_seasonal(tmp_path):
    # A model that just echoes persistence at +72h loses to seasonal-naive when
    # the series is perfectly diurnal — this is the diurnal free-lunch failure.
    cache = tmp_path / "nn_cache"
    cache.mkdir(parents=True)
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=24 * 6, freq="1h")
    # Diurnal shape plus a linear trend. At +72h the same-hour value recurs, so
    # seasonal-naive (ds-24h) is closer in time than persistence (ds-72h) and
    # therefore strictly better — the classic diurnal free-lunch gap.
    y = np.sin(2 * np.pi * (idx.hour.to_numpy()) / 24.0) + 0.5 * np.arange(len(idx))
    pd.DataFrame({"unique_id": "a", "ds": idx, "y": y}).to_parquet(
        cache / "long_v2_past_only_fill_origin_observed_full.parquet", index=False
    )
    pd.DataFrame({"unique_id": "a", "ds": idx, "observed": True}).to_parquet(
        cache / "mask_v2_past_only_fill_origin_observed_full.parquet", index=False
    )
    obs = pd.Series(y, index=pd.MultiIndex.from_arrays([["a"] * len(idx), idx]))
    ds = idx[72:]
    cutoff = ds - pd.Timedelta(hours=72)
    model_pred = obs.reindex(pd.MultiIndex.from_arrays([["a"] * len(ds), cutoff])).to_numpy()  # echo persistence

    _write_predictions(tmp_path, "run-a", "split-a", "a", ds, cutoff, model_pred)

    table = seasonal_naive_table(tmp_path)
    row = table[table["horizon_h"] == 72].iloc[0]
    # Seasonal-naive is ~exact (diurnal) so it crushes persistence; echoing
    # persistence must therefore score negative vs best-naive.
    assert row["seasonal_naive_rmse"] < row["persistence_rmse_common"]
    assert row["skill_vs_best_naive_pct"] < 0


def test_load_observed_panel_returns_none_without_cache(tmp_path):
    assert load_observed_panel(tmp_path) is None


def test_seasonal_naive_table_empty_without_predictions(tmp_path):
    _write_panel(tmp_path, "a", 48)
    assert seasonal_naive_table(tmp_path).empty


def test_score_predictions_masks_to_observed_origin(tmp_path):
    obs = _write_panel(tmp_path, "a", 48)
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="1h")
    ds = idx[24:]
    cutoff = ds - pd.Timedelta(hours=1)
    preds = pd.DataFrame(
        {
            "run_id": "r",
            "split_id": "s",
            "unique_id": "a",
            "ds": ds,
            "cutoff": cutoff,
            "ModelPred": np.arange(24, 48, dtype=float),
            "model_prediction_col": "ModelPred",
        }
    )
    scored = score_predictions(preds, obs)
    assert scored["skill_vs_best_naive_pct"].iloc[0] == 100.0
