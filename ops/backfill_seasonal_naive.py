#!/usr/bin/env python3
"""Backfill best-naive evidence into the gold forecast-metrics table.

Enriches ``lakehouse/gold/forecast_metrics/metrics.parquet`` with
``seasonal_naive_rmse`` and ``skill_vs_best_naive_pct`` (plus supporting common
columns) recomputed from the gold prediction partitions. The honest baseline
(skill vs the better of persistence and seasonal-naive) becomes a first-class
gold column so the promotion matrix and evidence gate enforce it.

This step is deterministic and idempotent: it recomputes the best-naive table
from predictions on every run and re-attaches it, so re-running reproduces the
same enriched table. It never mutates the model/persistence metrics themselves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from ops.seasonal_naive import BEST_NAIVE_COLUMNS, attach_best_naive, seasonal_naive_table  # noqa: E402


def enrich_metrics(project_root: Path) -> dict:
    metrics_path = project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet"
    if not metrics_path.exists():
        return {"status": "FAIL", "reason": "aggregate metrics table is missing", "path": str(metrics_path)}
    metrics = pd.read_parquet(metrics_path)
    # Drop any stale best-naive columns so the recompute is authoritative.
    derived = [c for c in BEST_NAIVE_COLUMNS if c not in ("run_id", "split_id", "unique_id", "horizon_h")]
    metrics = metrics.drop(columns=[c for c in derived if c in metrics.columns])

    seasonal = seasonal_naive_table(project_root)
    enriched = attach_best_naive(metrics, seasonal)
    enriched.to_parquet(metrics_path, index=False)

    verified = int(enriched["skill_vs_best_naive_pct"].notna().sum()) if "skill_vs_best_naive_pct" in enriched else 0
    return {
        "status": "PASS",
        "path": str(metrics_path),
        "metric_rows": int(len(enriched)),
        "best_naive_cells_scored": int(len(seasonal)),
        "metric_rows_with_best_naive": verified,
        "metric_rows_without_best_naive": int(len(enriched) - verified),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = enrich_metrics(args.project_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
