#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbal_lakehouse import read_forecast_metrics  # noqa: E402


def test_read_forecast_metrics_deduplicates_aggregate_and_partitions(tmp_path):
    metrics_root = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    run_dir = metrics_root / "run_id=r1"
    run_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "split_id": "s1",
                "unique_id": "temp_d1p0",
                "horizon_h": 6,
                "model": "nhits",
                "loss": "mae",
                "drivers_enabled": False,
                "model_rmse": 1.0,
                "persistence_rmse": 2.0,
                "skill_vs_persistence_pct": 50.0,
            }
        ]
    )
    rows.to_parquet(metrics_root / "metrics.parquet", index=False)
    rows.to_parquet(run_dir / "metrics.parquet", index=False)

    metrics = read_forecast_metrics(tmp_path, include_partitions=True)

    assert len(metrics) == 1
    assert metrics["run_id"].item() == "r1"
