#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ID = "foundation_benchmark"
DEFAULT_SPLIT_ID = "foundation_zero_shot_benchmark"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_foundation_summary(summary_path: Path) -> dict[str, Any]:
    text = read_text_or_empty(summary_path)
    if not text:
        return {"path": str(summary_path), "exists": False, "signals": []}

    signals: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "foundation" in stripped.lower() or "sweet spot" in stripped.lower():
            signals.append(stripped)

    best_match = re.search(r"best foundation model \(\*\*(?P<model>[^*)]+)\*\*\)", text, re.IGNORECASE)
    skill_match = re.search(r"median skill\s+(?P<skill>[+-]?\d+(?:\.\d+)?)%", text, re.IGNORECASE)

    return {
        "path": str(summary_path),
        "exists": True,
        "signals": signals[:10],
        "best_model": best_match.group("model") if best_match else None,
        "median_skill_pct": float(skill_match.group("skill")) if skill_match else None,
    }


def find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def numeric_column(frame: pd.DataFrame, source_col: str | None) -> pd.Series:
    if source_col is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[source_col], errors="coerce")


def normalize_metrics(leaderboard_path: Path, run_id: str, split_id: str = DEFAULT_SPLIT_ID) -> pd.DataFrame:
    raw = pd.read_csv(leaderboard_path)
    columns = list(raw.columns)

    model_col = find_first_column(columns, ["model", "foundation_model", "candidate_model"])
    target_col = find_first_column(columns, ["unique_id", "target", "series", "variable"])
    horizon_col = find_first_column(columns, ["horizon_h", "horizon", "h", "lead_h", "lead_time_h"])
    skill_col = find_first_column(
        columns,
        [
            "skill_vs_persistence_pct",
            "skill_rmse_vs_persistence_pct",
            "skill_pct",
            "skill",
            "median_skill_pct",
        ],
    )
    model_rmse_col = find_first_column(columns, ["model_rmse", "rmse", "candidate_rmse", "forecast_rmse"])
    model_mae_col = find_first_column(columns, ["model_mae", "mae", "candidate_mae", "forecast_mae"])
    persistence_rmse_col = find_first_column(
        columns,
        ["persistence_rmse", "baseline_rmse", "persistence_baseline_rmse"],
    )
    n_col = find_first_column(columns, ["n", "count", "observations", "n_obs", "num_obs"])

    required = {
        "model": model_col,
        "unique_id/target": target_col,
        "horizon_h": horizon_col,
        "skill": skill_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"foundation leaderboard missing required column(s): {', '.join(missing)}")

    metrics = pd.DataFrame(
        {
            "run_id": run_id,
            "split_id": split_id,
            "unique_id": raw[target_col].astype(str),
            "horizon_h": pd.to_numeric(raw[horizon_col], errors="coerce").astype("Int64"),
            "model_rmse": numeric_column(raw, model_rmse_col),
            "model_mae": numeric_column(raw, model_mae_col),
            "persistence_rmse": numeric_column(raw, persistence_rmse_col),
            "skill_vs_persistence_pct": numeric_column(raw, skill_col),
            "n": numeric_column(raw, n_col).astype("Int64") if n_col is not None else pd.Series([pd.NA] * len(raw), dtype="Int64"),
            "model": raw[model_col].astype(str),
            "loss": "zero_shot",
            "drivers_enabled": False,
            "benchmark_type": "foundation_zero_shot",
            "created_at_utc": utc_now(),
        }
    )
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@dataclass(frozen=True)
class NormalizeResult:
    run_id: str
    status: str
    leaderboard_path: Path
    summary_path: Path
    manifest_path: Path
    metrics_path: Path
    metrics_rows: int
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "leaderboard_path": str(self.leaderboard_path),
            "summary_path": str(self.summary_path),
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "metrics_rows": self.metrics_rows,
            "applied": self.applied,
        }


def build_manifest(
    project_root: Path,
    run_id: str,
    status: str,
    leaderboard_path: Path,
    summary_path: Path,
    metrics_path: Path,
    metrics_rows: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "split_id": DEFAULT_SPLIT_ID,
        "created_at_utc": utc_now(),
        "model": "foundation",
        "loss": "zero_shot",
        "status": status,
        "benchmark_type": "foundation_zero_shot",
        "drivers_enabled": False,
        "project_root": str(project_root),
        "source": {
            "leaderboard_csv": str(leaderboard_path),
            "leaderboard_exists": leaderboard_path.exists(),
            "foundation_summary_md": str(summary_path),
            "foundation_summary": parse_foundation_summary(summary_path),
        },
        "artifacts": {
            "metrics_parquet": str(metrics_path) if metrics_rows else None,
            "metrics_rows": metrics_rows,
        },
    }


def normalize_foundation_to_lakehouse(project_root: Path, run_id: str, apply: bool = False) -> NormalizeResult:
    project_root = project_root.resolve()
    leaderboard_path = project_root / "nn_results" / "foundation_leaderboard.csv"
    summary_path = project_root / "nn_results" / "foundation_summary.md"
    manifest_path = project_root / "lakehouse" / "gold" / "forecast_runs" / f"run_id={run_id}" / "run_manifest.json"
    metrics_root = project_root / "lakehouse" / "gold" / "forecast_metrics"
    metrics_path = metrics_root / f"run_id={run_id}" / "metrics.parquet"
    aggregate_metrics_path = metrics_root / "metrics.parquet"

    metrics = pd.DataFrame()
    status = "skipped"
    if leaderboard_path.exists():
        metrics = normalize_metrics(leaderboard_path, run_id)
        status = "completed"

    manifest = build_manifest(
        project_root=project_root,
        run_id=run_id,
        status=status,
        leaderboard_path=leaderboard_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        metrics_rows=int(len(metrics)),
    )

    if apply:
        write_json(manifest_path, manifest)
        if not metrics.empty:
            write_parquet(metrics_path, metrics)
            if aggregate_metrics_path.exists():
                aggregate = pd.read_parquet(aggregate_metrics_path)
                if "run_id" in aggregate.columns:
                    aggregate = aggregate[aggregate["run_id"] != run_id]
                aggregate = pd.concat([aggregate, metrics], ignore_index=True)
            else:
                aggregate = metrics
            write_parquet(aggregate_metrics_path, aggregate)

    return NormalizeResult(
        run_id=run_id,
        status=status,
        leaderboard_path=leaderboard_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        metrics_rows=int(len(metrics)),
        applied=apply,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize foundation benchmark outputs into lakehouse run partitions.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--apply", action="store_true", help="Write lakehouse artifacts. Default is dry-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = normalize_foundation_to_lakehouse(args.project_root, args.run_id, apply=args.apply)
    mode = "applied" if result.applied else "dry-run"
    print(json.dumps({"mode": mode, **result.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
