#!/usr/bin/env python
"""Run XGBoost baselines on existing candidate split contracts.

This closes the promotion loop for neural candidates that already have split
rows under lakehouse/silver/forecast_splits. It trains a causal XGBoost delta
model at each forecast origin using only labels known by that origin, scores the
same target/horizon cells, and writes lakehouse metrics with the same split_id.
"""

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
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mbal_forecast_v2 import (
    RANDOM_STATE,
    build_causal_features,
    build_hourly_matrix,
    load_source,
    variable_of,
)
from mbal_core import apply_physical_quality_filters


DEFAULT_RUN_PREFIX = "xgb_shared"


@dataclass(frozen=True)
class Job:
    split_id: str
    target: str
    horizon_h: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def xgb_params(device: str, n_estimators: int) -> dict[str, Any]:
    return {
        "n_estimators": int(n_estimators),
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "random_state": RANDOM_STATE,
        "n_jobs": 0,
    }


def load_matrix(project_root: Path) -> pd.DataFrame:
    source = project_root / "mbal_history" / "opendap" / "m1_history.parquet"
    raw = load_source("parquet", str(source), None)
    filtered = apply_physical_quality_filters(raw)
    clean = filtered[0] if isinstance(filtered, tuple) else filtered
    matrix, _coverage = build_hourly_matrix(clean)
    return matrix


def load_split_rows(project_root: Path, split_id: str) -> pd.DataFrame:
    path = project_root / "lakehouse" / "silver" / "forecast_splits" / f"split_id={split_id}" / "split_rows.parquet"
    if not path.exists():
        raise FileNotFoundError(f"split rows not found for {split_id}: {path}")
    rows = pd.read_parquet(path)
    rows["cutoff"] = pd.to_datetime(rows["cutoff"], utc=True)
    return rows


def select_jobs(project_root: Path, max_jobs: int | None = None) -> list[Job]:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError("promotion_matrix.parquet missing; run release_gate/mbal_promotion_matrix.py first")
    promo = pd.read_parquet(matrix_path)
    candidates = promo[promo["status"].eq("candidate_split_mismatch")].copy()
    if candidates.empty:
        return []
    # Only jobs with real split rows can be closed without rerunning the candidate.
    existing_split_ids = {
        p.name.split("split_id=", 1)[1]
        for p in (project_root / "lakehouse" / "silver" / "forecast_splits").glob("split_id=*")
        if (p / "split_rows.parquet").exists()
    }
    candidates = candidates[candidates["candidate_split_id"].astype(str).isin(existing_split_ids)]
    candidates = candidates.sort_values("xgb_delta_skill_pct", ascending=False)
    jobs: list[Job] = []
    seen: set[tuple[str, str, int]] = set()
    for row in candidates.itertuples(index=False):
        key = (str(row.candidate_split_id), str(row.target), int(row.horizon_h))
        if key in seen:
            continue
        seen.add(key)
        jobs.append(Job(split_id=key[0], target=key[1], horizon_h=key[2]))
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
    return jobs


def evaluate_job(matrix: pd.DataFrame, split_rows: pd.DataFrame, job: Job, device: str, n_estimators: int) -> tuple[dict, pd.DataFrame]:
    features = build_causal_features(matrix, job.target)
    y_future = matrix[job.target].shift(-job.horizon_h).rename("y_future")
    y_now = matrix[job.target].rename("y_now")
    frame = features.copy()
    frame["__y_future"] = y_future
    frame["__y_now"] = y_now
    frame = frame.dropna(subset=["__y_future", "__y_now"])
    feat_cols = [c for c in features.columns if frame[c].notna().sum() >= max(50, int(len(frame) * 0.1))]
    frame = frame.loc[frame[feat_cols].notna().sum(axis=1) >= max(3, len(feat_cols) // 3)]
    if len(feat_cols) < 3 or frame.empty:
        raise ValueError(f"not enough usable features for {job.target}+{job.horizon_h}h")

    rows = split_rows[split_rows["horizon_h"].astype(int).eq(job.horizon_h)].copy()
    preds = []
    for split_index, row in enumerate(rows.itertuples(index=False)):
        cutoff = pd.Timestamp(row.cutoff)
        train_end = cutoff - pd.Timedelta(hours=job.horizon_h)
        target_ds = cutoff + pd.Timedelta(hours=job.horizon_h)
        if cutoff not in frame.index or target_ds not in matrix.index:
            continue
        train = frame.loc[frame.index <= train_end].copy()
        test = frame.loc[[cutoff]].copy()
        if len(train) < 150 or test.empty:
            continue
        x_train = train[feat_cols]
        d_train = train["__y_future"] - train["__y_now"]
        model = xgb.XGBRegressor(**xgb_params(device, n_estimators))
        model.fit(x_train, d_train, verbose=False)
        now = float(test["__y_now"].iloc[0])
        y_true = float(test["__y_future"].iloc[0])
        y_hat = now + float(model.predict(test[feat_cols])[0])
        preds.append(
            {
                "split_id": job.split_id,
                "split_index": int(getattr(row, "split_index", split_index)),
                "unique_id": job.target,
                "horizon_h": int(job.horizon_h),
                "cutoff": cutoff,
                "ds": target_ds,
                "y": y_true,
                "xgb": y_hat,
                "persistence": now,
            }
        )
    pred_df = pd.DataFrame(preds)
    if pred_df.empty:
        raise ValueError(f"no scored rows for {job}")
    y = pred_df["y"].to_numpy()
    model_pred = pred_df["xgb"].to_numpy()
    persistence = pred_df["persistence"].to_numpy()
    model_mae = float(mean_absolute_error(y, model_pred))
    model_rmse = float(root_mean_squared_error(y, model_pred))
    persistence_rmse = float(root_mean_squared_error(y, persistence))
    persistence_mae = float(mean_absolute_error(y, persistence))
    skill_pct = float((1.0 - model_rmse / persistence_rmse) * 100.0) if persistence_rmse > 0 else np.nan
    variable, depth_m = variable_of(job.target)
    metric = {
        "split_id": job.split_id,
        "unique_id": job.target,
        "horizon_h": int(job.horizon_h),
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "persistence_rmse": persistence_rmse,
        "persistence_mae": persistence_mae,
        "skill_vs_persistence_pct": skill_pct,
        "n": int(len(pred_df)),
        "model": "xgboost",
        "loss": "squared_error",
        "drivers_enabled": False,
        "variable": variable,
        "depth_m": depth_m,
        "cache_version": "xgb_on_candidate_split_v1",
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


def run(project_root: Path, max_jobs: int | None, device: str, n_estimators: int, apply: bool) -> dict:
    project_root = project_root.resolve()
    jobs = select_jobs(project_root, max_jobs=max_jobs)
    if not jobs:
        return {"jobs": 0, "metrics_rows": 0, "applied": apply}
    matrix = load_matrix(project_root)
    metrics = []
    predictions = []
    for job in jobs:
        split_rows = load_split_rows(project_root, job.split_id)
        metric, pred = evaluate_job(matrix, split_rows, job, device=device, n_estimators=n_estimators)
        metrics.append(metric)
        predictions.append(pred)
    run_id = f"{DEFAULT_RUN_PREFIX}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "run_id", run_id)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.insert(0, "run_id", run_id)
    if apply:
        run_dir = project_root / "lakehouse" / "gold" / "forecast_runs" / f"run_id={run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": "xgboost",
            "loss": "squared_error",
            "status": "completed",
            "purpose": "xgb_baseline_on_candidate_splits",
            "jobs": [job.__dict__ for job in jobs],
            "n_estimators": int(n_estimators),
            "device": device,
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        metrics_path = project_root / "lakehouse" / "gold" / "forecast_metrics" / f"run_id={run_id}" / "metrics.parquet"
        pred_path = project_root / "lakehouse" / "gold" / "forecast_predictions" / f"run_id={run_id}" / "predictions.parquet"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_parquet(metrics_path, index=False)
        pred_df.to_parquet(pred_path, index=False)
        refresh_aggregate(project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet", metrics_df, run_id)
    return {
        "jobs": len(jobs),
        "metrics_rows": int(len(metrics_df)),
        "applied": apply,
        "run_id": run_id,
        "jobs_detail": [job.__dict__ for job in jobs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XGBoost on existing candidate split contracts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=96)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.max_jobs, args.device, args.n_estimators, args.apply), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
