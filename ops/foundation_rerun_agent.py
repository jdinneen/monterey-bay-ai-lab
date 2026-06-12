#!/usr/bin/env python3
"""Rerun Chronos foundation candidates on shared split contracts."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbal_core import apply_physical_quality_filters
from mbal_deep_models import chronos_forecast
from mbal_forecast_v2 import build_hourly_matrix, load_source, variable_of
from ops.seasonal_naive import attach_best_naive, seasonal_naive_table


@dataclass(frozen=True)
class Job:
    target: str
    horizon_h: int
    split_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_matrix(project_root: Path) -> pd.DataFrame:
    raw = load_source("parquet", str(project_root / "mbal_history" / "opendap" / "m1_history.parquet"), None)
    filtered = apply_physical_quality_filters(raw)
    clean = filtered[0] if isinstance(filtered, tuple) else filtered
    matrix, _coverage = build_hourly_matrix(clean)
    return matrix.sort_index()


def load_split_rows(project_root: Path, split_id: str) -> pd.DataFrame:
    path = project_root / "lakehouse" / "silver" / "forecast_splits" / f"split_id={split_id}" / "split_rows.parquet"
    if not path.exists():
        raise FileNotFoundError(f"split rows not found for {split_id}: {path}")
    rows = pd.read_parquet(path)
    rows["cutoff"] = pd.to_datetime(rows["cutoff"], utc=True)
    return rows


def select_jobs(project_root: Path, max_jobs: int | None = None) -> list[Job]:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    promo = pd.read_parquet(matrix_path)
    rows = promo[
        promo["status"].eq("candidate_split_mismatch")
        & promo["candidate_split_id"].astype(str).eq("foundation_zero_shot_benchmark")
        & promo["candidate_model"].astype(str).str.lower().eq("chronos")
        & promo["xgb_split_id"].notna()
    ].copy()
    if rows.empty:
        return []
    rows = rows.sort_values("xgb_delta_skill_pct", ascending=False)
    jobs: list[Job] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows.itertuples(index=False):
        split_id = str(row.xgb_split_id)
        if split_id == "xgb_forecast_v2_internal":
            continue
        key = (str(row.target), int(row.horizon_h), split_id)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(Job(target=key[0], horizon_h=key[1], split_id=key[2]))
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
    return jobs


def evaluate_job(project_root: Path, matrix: pd.DataFrame, job: Job) -> tuple[dict[str, Any], pd.DataFrame]:
    split_rows = load_split_rows(project_root, job.split_id)
    rows = split_rows[split_rows["horizon_h"].astype(int).eq(job.horizon_h)].copy()
    y_hist = matrix[job.target].sort_index()
    cutoffs = pd.DatetimeIndex(rows["cutoff"].drop_duplicates().sort_values())
    preds = chronos_forecast(y_hist, cutoffs, job.horizon_h)
    pred_df = pd.DataFrame({"cutoff": cutoffs, "chronos": preds})
    pred_df["ds"] = pred_df["cutoff"] + pd.to_timedelta(job.horizon_h, unit="h")
    pred_df["unique_id"] = job.target
    pred_df["horizon_h"] = int(job.horizon_h)
    pred_df["split_id"] = job.split_id
    pred_df["y"] = y_hist.reindex(pred_df["ds"]).to_numpy()
    pred_df["persistence"] = y_hist.reindex(pred_df["cutoff"]).to_numpy()
    pred_df = pred_df.dropna(subset=["y", "persistence", "chronos"]).copy()
    if pred_df.empty:
        raise ValueError(f"no scored Chronos rows for {job}")
    y = pred_df["y"].to_numpy()
    model_pred = pred_df["chronos"].to_numpy()
    persistence = pred_df["persistence"].to_numpy()
    model_rmse = float(root_mean_squared_error(y, model_pred))
    persistence_rmse = float(root_mean_squared_error(y, persistence))
    variable, depth_m = variable_of(job.target)
    metric = {
        "split_id": job.split_id,
        "unique_id": job.target,
        "horizon_h": int(job.horizon_h),
        "model_rmse": model_rmse,
        "model_mae": float(mean_absolute_error(y, model_pred)),
        "persistence_rmse": persistence_rmse,
        "persistence_mae": float(mean_absolute_error(y, persistence)),
        "skill_vs_persistence_pct": float((1.0 - model_rmse / persistence_rmse) * 100.0) if persistence_rmse > 0 else np.nan,
        "n": int(len(pred_df)),
        "model": "chronos",
        "loss": "zero_shot_shared",
        "drivers_enabled": False,
        "variable": variable,
        "depth_m": depth_m,
        "cache_version": "foundation_shared_split_chronos_v1",
        "created_at_utc": utc_now(),
    }
    return metric, pred_df


def refresh_aggregate(path: Path, run_metrics: pd.DataFrame, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        aggregate = pd.read_parquet(path)
        if "run_id" in aggregate.columns:
            aggregate = aggregate[aggregate["run_id"] != run_id]
        aggregate = pd.concat([aggregate, run_metrics], ignore_index=True)
    else:
        aggregate = run_metrics
    aggregate.to_parquet(path, index=False)


def run(project_root: Path, max_jobs: int | None, apply: bool) -> dict[str, Any]:
    project_root = project_root.resolve()
    jobs = select_jobs(project_root, max_jobs=max_jobs)
    if not jobs:
        return {"applied": apply, "jobs": 0, "metrics_rows": 0}
    run_id = f"foundation_chronos_shared_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not apply:
        return {"applied": False, "jobs": len(jobs), "metrics_rows": 0, "planned": [job.__dict__ for job in jobs]}
    matrix = load_matrix(project_root)
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for job in jobs:
        metric, pred = evaluate_job(project_root, matrix, job)
        metrics.append(metric)
        predictions.append(pred)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "run_id", run_id)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.insert(0, "run_id", run_id)
    pred_df["model_prediction_col"] = "chronos"
    pred_df["cache_version"] = "foundation_shared_split_chronos_v1"

    run_dir = project_root / "lakehouse" / "gold" / "forecast_runs" / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": utc_now(),
                "model": "chronos",
                "loss": "zero_shot_shared",
                "status": "completed",
                "purpose": "foundation_chronos_shared_split_rerun",
                "jobs": [job.__dict__ for job in jobs],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    metrics_path = project_root / "lakehouse" / "gold" / "forecast_metrics" / f"run_id={run_id}" / "metrics.parquet"
    pred_path = project_root / "lakehouse" / "gold" / "forecast_predictions" / f"run_id={run_id}" / "predictions.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(pred_path, index=False)
    seasonal = seasonal_naive_table(project_root)
    metrics_df = attach_best_naive(metrics_df, seasonal)
    metrics_df.to_parquet(metrics_path, index=False)
    refresh_aggregate(project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet", metrics_df, run_id)
    return {"applied": True, "run_id": run_id, "jobs": len(jobs), "metrics_rows": int(len(metrics_df)), "planned": [job.__dict__ for job in jobs]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.max_jobs, args.apply), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
