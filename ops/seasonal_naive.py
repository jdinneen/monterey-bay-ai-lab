#!/usr/bin/env python3
"""Shared best-naive (persistence vs seasonal-naive) scoring for the promotion path.

Skill vs PERSISTENCE alone is unfairly easy at diurnal horizons (+24/72/168h),
where seasonal-naive (same-hour-yesterday) is a strong free baseline. ``AGENTS.md``
makes the honest baseline binding: a candidate must beat the BETTER of
{persistence, seasonal-naive} before it is promotable.

This module is the single source of truth for that computation. It re-scores every
gold prediction partition against persistence and seasonal-naive on one common,
observed + origin-observed subset (matching ``mbal_neural_forecast.evaluate`` and
``research/model_lab/honest_baseline``) and returns a per-cell table keyed by
``run_id, split_id, unique_id, horizon_h``.

Read-only over the lakehouse. No training, no mutation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


BEST_NAIVE_COLUMNS = [
    "run_id",
    "split_id",
    "unique_id",
    "horizon_h",
    "n_best_naive",
    "model_rmse_common",
    "persistence_rmse_common",
    "seasonal_naive_rmse",
    "best_naive_rmse",
    "skill_vs_seasonal_naive_pct",
    "skill_vs_best_naive_pct",
]

# Columns that are never the model prediction column in a prediction partition.
_NON_PRED_COLS = {
    "run_id",
    "split_id",
    "unique_id",
    "ds",
    "cutoff",
    "y",
    "observed",
    "cache_version",
    "horizon_h",
    "model_prediction_col",
}


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=BEST_NAIVE_COLUMNS)


def load_observed_panel(project_root: Path) -> pd.Series | None:
    """Observed-only target panel indexed by ``(unique_id, ds)``.

    Returns ``None`` when the cached panel/mask are unavailable so callers can
    fail closed rather than silently scoring against a partial baseline.
    """
    cache = Path(os.environ.get("MBAL_CACHE_DIR", project_root / "nn_cache"))
    long_path = cache / "long_v2_past_only_fill_origin_observed_full.parquet"
    mask_path = cache / "mask_v2_past_only_fill_origin_observed_full.parquet"
    if not long_path.exists() or not mask_path.exists():
        return None
    long_df = pd.read_parquet(long_path)
    mask = pd.read_parquet(mask_path)
    if not {"unique_id", "ds", "y"}.issubset(long_df.columns) or "observed" not in mask.columns:
        return None
    long_df["ds"] = _to_dt(long_df["ds"])
    mask["ds"] = _to_dt(mask["ds"])
    obs = long_df.merge(mask[["unique_id", "ds", "observed"]], on=["unique_id", "ds"], how="left")
    obs = obs[obs["observed"].fillna(False)][["unique_id", "ds", "y"]]
    obs = obs.dropna(subset=["y"]).drop_duplicates(["unique_id", "ds"])
    return obs.set_index(["unique_id", "ds"])["y"]


def _prediction_column(preds: pd.DataFrame) -> str | None:
    if "model_prediction_col" in preds.columns and preds["model_prediction_col"].notna().any():
        col = str(preds["model_prediction_col"].dropna().iloc[0])
        if col in preds.columns:
            return col
    candidates = [c for c in preds.columns if c not in _NON_PRED_COLS]
    return candidates[0] if candidates else None


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def score_predictions(
    preds: pd.DataFrame,
    obs_val: pd.Series,
    *,
    min_observations: int = 3,
) -> pd.DataFrame:
    """Score one prediction frame against persistence and seasonal-naive.

    Truth, persistence (value at the forecast origin), and seasonal-naive
    (value at ``ds - 24h``) are all derived from the observed panel, so a cell
    is scored only where the model prediction, the target, the origin, and
    yesterday-same-hour are all observed. This is the common subset the honest
    baseline demands.
    """
    pcol = _prediction_column(preds)
    if pcol is None or preds.empty:
        return _empty_table()
    df = preds.copy()
    df["ds"] = _to_dt(df["ds"])
    df["cutoff"] = _to_dt(df["cutoff"])
    df["horizon_h"] = ((df["ds"] - df["cutoff"]) / pd.Timedelta(hours=1)).round().astype("Int64")
    df = df[df["horizon_h"].notna()].copy()
    df["horizon_h"] = df["horizon_h"].astype(int)

    df["_y"] = obs_val.reindex(pd.MultiIndex.from_arrays([df["unique_id"], df["ds"]])).to_numpy()
    df["_y_persist"] = obs_val.reindex(pd.MultiIndex.from_arrays([df["unique_id"], df["cutoff"]])).to_numpy()
    df["_y_seas"] = obs_val.reindex(
        pd.MultiIndex.from_arrays([df["unique_id"], df["ds"] - pd.Timedelta(hours=24)])
    ).to_numpy()
    df = df.dropna(subset=[pcol, "_y", "_y_persist", "_y_seas"])
    if df.empty:
        return _empty_table()

    for key in ("run_id", "split_id"):
        if key not in df.columns:
            df[key] = ""

    rows: list[dict] = []
    for (run_id, split_id, uid, hh), g in df.groupby(["run_id", "split_id", "unique_id", "horizon_h"]):
        if len(g) < min_observations:
            continue
        y = g["_y"].to_numpy()
        model_rmse = _rmse(y, g[pcol].to_numpy())
        persistence_rmse = _rmse(y, g["_y_persist"].to_numpy())
        seasonal_rmse = _rmse(y, g["_y_seas"].to_numpy())
        best = min(persistence_rmse, seasonal_rmse)
        rows.append(
            {
                "run_id": run_id,
                "split_id": split_id,
                "unique_id": uid,
                "horizon_h": int(hh),
                "n_best_naive": int(len(g)),
                "model_rmse_common": model_rmse,
                "persistence_rmse_common": persistence_rmse,
                "seasonal_naive_rmse": seasonal_rmse,
                "best_naive_rmse": best,
                "skill_vs_seasonal_naive_pct": 100.0 * (1 - model_rmse / seasonal_rmse) if seasonal_rmse else np.nan,
                "skill_vs_best_naive_pct": 100.0 * (1 - model_rmse / best) if best else np.nan,
            }
        )
    if not rows:
        return _empty_table()
    return pd.DataFrame(rows, columns=BEST_NAIVE_COLUMNS)


def seasonal_naive_table(project_root: Path, *, min_observations: int = 3) -> pd.DataFrame:
    """Best-naive table across every gold prediction partition.

    Returns an empty (well-typed) frame when the panel or the prediction
    partitions are unavailable, which forces downstream gates to fail closed at
    diurnal horizons rather than promote on persistence alone.
    """
    obs_val = load_observed_panel(project_root)
    if obs_val is None:
        return _empty_table()
    pred_root = Path(
        os.environ.get("MBAL_LAKEHOUSE_DIR", project_root / "lakehouse")
    ) / "gold" / "forecast_predictions"
    if not pred_root.exists():
        return _empty_table()
    frames: list[pd.DataFrame] = []
    for partition in sorted(pred_root.glob("run_id=*")):
        for parquet in sorted(partition.glob("*.parquet")):
            preds = pd.read_parquet(parquet)
            scored = score_predictions(preds, obs_val, min_observations=min_observations)
            if not scored.empty:
                frames.append(scored)
    if not frames:
        return _empty_table()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["run_id", "split_id", "unique_id", "horizon_h"], keep="last").reset_index(drop=True)


def attach_best_naive(metrics: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    """Left-join the best-naive columns onto a metrics-like frame.

    Keyed on ``(run_id, split_id, unique_id, horizon_h)``. Missing matches stay
    NaN so callers can distinguish "verified and failed" from "not verified".
    """
    join_cols = ["run_id", "split_id", "unique_id", "horizon_h"]
    add_cols = [c for c in BEST_NAIVE_COLUMNS if c not in join_cols]
    out = metrics.copy()
    if not set(join_cols).issubset(out.columns):
        for col in add_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out
    if seasonal is None or seasonal.empty:
        for col in add_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out
    out["horizon_h"] = pd.to_numeric(out["horizon_h"], errors="coerce").astype("Int64")
    seasonal = seasonal.copy()
    seasonal["horizon_h"] = pd.to_numeric(seasonal["horizon_h"], errors="coerce").astype("Int64")
    merged = out.merge(seasonal[BEST_NAIVE_COLUMNS], on=join_cols, how="left")
    merged["horizon_h"] = merged["horizon_h"].astype(int)
    return merged
