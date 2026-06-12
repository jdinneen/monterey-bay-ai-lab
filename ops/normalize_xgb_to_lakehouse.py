#!/usr/bin/env python
"""Normalize MBAL forecast-v2 XGBoost results into lakehouse contracts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "xgb_forecast_v2"
DEFAULT_SPLIT_ID = "xgb_forecast_v2_internal"


@dataclass(frozen=True)
class NormalizeConfig:
    project_root: Path
    run_id: str = DEFAULT_RUN_ID
    apply: bool = False

    @property
    def xgb_results_dir(self) -> Path:
        return self.project_root / "mbal_forecast_v2_results"

    @property
    def leaderboard_path(self) -> Path:
        return self.xgb_results_dir / "leaderboard.csv"

    @property
    def model_results_path(self) -> Path:
        return self.xgb_results_dir / "model_results.json"

    @property
    def forecast_runs_dir(self) -> Path:
        return self.project_root / "lakehouse" / "gold" / "forecast_runs" / f"run_id={self.run_id}"

    @property
    def forecast_metrics_dir(self) -> Path:
        return self.project_root / "lakehouse" / "gold" / "forecast_metrics"

    @property
    def run_metrics_dir(self) -> Path:
        return self.forecast_metrics_dir / f"run_id={self.run_id}"

    @property
    def run_manifest_path(self) -> Path:
        return self.forecast_runs_dir / "run_manifest.json"

    @property
    def run_metrics_path(self) -> Path:
        return self.run_metrics_dir / "metrics.parquet"

    @property
    def aggregate_metrics_path(self) -> Path:
        return self.forecast_metrics_dir / "metrics.parquet"


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return None
    return result


def _target_key(target: Any, horizon_h: Any) -> tuple[str, int]:
    horizon = _to_int(horizon_h)
    if horizon is None:
        raise ValueError(f"invalid horizon_h for target {target!r}: {horizon_h!r}")
    return str(target), horizon


def load_model_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "target" not in row or "horizon_h" not in row:
            continue
        indexed[_target_key(row["target"], row["horizon_h"])] = row
    return indexed


def _persistence_rmse_from_folds(result: dict[str, Any]) -> float | None:
    folds = result.get("folds")
    if not isinstance(folds, list):
        return None
    values = [_to_float(fold.get("persistence_rmse")) for fold in folds if isinstance(fold, dict)]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _test_rows_from_folds(result: dict[str, Any]) -> int | None:
    folds = result.get("folds")
    if not isinstance(folds, list):
        return None
    values = [_to_int(fold.get("test_rows")) for fold in folds if isinstance(fold, dict)]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return int(sum(values))


def _infer_persistence_rmse(model_rmse: float | None, skill_fraction: float | None) -> float | None:
    if model_rmse is None or skill_fraction is None:
        return None
    denominator = 1.0 - skill_fraction
    if denominator <= 0:
        return None
    return model_rmse / denominator


def build_metrics(config: NormalizeConfig) -> pd.DataFrame:
    if not config.leaderboard_path.exists():
        raise FileNotFoundError(config.leaderboard_path)
    leaderboard = pd.read_csv(config.leaderboard_path)
    model_results = load_model_results(config.model_results_path)

    rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for _, lb_row in leaderboard.iterrows():
        target = str(lb_row["target"])
        horizon_h = _to_int(lb_row["horizon_h"])
        if horizon_h is None:
            continue
        result = model_results.get((target, horizon_h), {})

        model_rmse = _to_float(lb_row.get("rmse"))
        if model_rmse is None:
            model_rmse = _to_float(result.get("mean_model_rmse"))
        model_mae = _to_float(lb_row.get("mae"))
        if model_mae is None:
            model_mae = _to_float(result.get("mean_model_mae"))

        skill_fraction = _to_float(lb_row.get("skill_rmse_vs_persistence"))
        if skill_fraction is None:
            skill_fraction = _to_float(result.get("mean_skill_rmse_vs_persistence"))

        persistence_rmse = _persistence_rmse_from_folds(result)
        if persistence_rmse is None:
            persistence_rmse = _infer_persistence_rmse(model_rmse, skill_fraction)

        n = _to_int(lb_row.get("usable_rows"))
        if n is None:
            n = _to_int(result.get("usable_rows"))
        if n is None:
            n = _test_rows_from_folds(result)

        rows.append(
            {
                "run_id": config.run_id,
                "split_id": DEFAULT_SPLIT_ID,
                "unique_id": target,
                "horizon_h": horizon_h,
                "model_rmse": model_rmse,
                "model_mae": model_mae,
                "persistence_rmse": persistence_rmse,
                "skill_vs_persistence_pct": None if skill_fraction is None else skill_fraction * 100.0,
                "n": n,
                "model": "xgboost_forecast_v2",
                "loss": "rmse",
                "drivers_enabled": True,
                "cache_version": None,
                "created_at_utc": created_at,
            }
        )

    return pd.DataFrame(rows)


def build_manifest(config: NormalizeConfig, metrics: pd.DataFrame) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "model": "xgboost_forecast_v2",
        "split_id": DEFAULT_SPLIT_ID,
        "source": "mbal_forecast_v2_results",
        "leaderboard_csv": str(config.leaderboard_path.relative_to(config.project_root)),
        "model_results_json": str(config.model_results_path.relative_to(config.project_root)),
        "metrics_rows": int(len(metrics)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def refresh_aggregate_metrics(config: NormalizeConfig, run_metrics: pd.DataFrame) -> pd.DataFrame:
    if config.aggregate_metrics_path.exists():
        aggregate = pd.read_parquet(config.aggregate_metrics_path)
        if "run_id" in aggregate.columns:
            aggregate = aggregate[aggregate["run_id"] != config.run_id]
        refreshed = pd.concat([aggregate, run_metrics], ignore_index=True)
    else:
        refreshed = run_metrics.copy()
    return refreshed


def write_outputs(config: NormalizeConfig, run_metrics: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    aggregate = refresh_aggregate_metrics(config, run_metrics)
    config.forecast_runs_dir.mkdir(parents=True, exist_ok=True)
    config.run_metrics_dir.mkdir(parents=True, exist_ok=True)
    config.forecast_metrics_dir.mkdir(parents=True, exist_ok=True)
    with config.run_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    run_metrics.to_parquet(config.run_metrics_path, index=False)
    aggregate.to_parquet(config.aggregate_metrics_path, index=False)
    try:  # Phase-1 shadow dual-write — opt-in via MBAL_GOLD_SHADOW_DELTA=1, non-fatal
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from ops.migrate_gold_to_delta import shadow_dual_write
        shadow_dual_write(
            "forecast_metrics", aggregate, gold_dir=config.aggregate_metrics_path.parent.parent
        )
    except Exception:
        pass
    return aggregate


def normalize(config: NormalizeConfig) -> dict[str, Any]:
    metrics = build_metrics(config)
    manifest = build_manifest(config, metrics)
    aggregate_rows = None
    if config.apply:
        aggregate = write_outputs(config, metrics, manifest)
        aggregate_rows = int(len(aggregate))
    return {
        "run_id": config.run_id,
        "mode": "apply" if config.apply else "dry-run",
        "metrics_rows": int(len(metrics)),
        "positive_skill_rows": int((metrics["skill_vs_persistence_pct"] > 0).sum()),
        "run_manifest": str(config.run_manifest_path),
        "run_metrics": str(config.run_metrics_path),
        "aggregate_metrics": str(config.aggregate_metrics_path),
        "aggregate_rows": aggregate_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", type=Path, help="project root path")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="lakehouse run id")
    parser.add_argument("--apply", action="store_true", help="write lakehouse outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NormalizeConfig(project_root=args.project_root.resolve(), run_id=args.run_id, apply=args.apply)
    summary = normalize(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
