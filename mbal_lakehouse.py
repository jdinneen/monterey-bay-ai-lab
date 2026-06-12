#!/usr/bin/env python3
"""Small lakehouse read helpers shared by gates and reports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


FORECAST_METRIC_KEY_COLS = [
    "run_id",
    "split_id",
    "unique_id",
    "horizon_h",
    "model",
    "loss",
    "drivers_enabled",
]


def read_forecast_metrics(project_root: Path, *, include_partitions: bool = False) -> pd.DataFrame:
    """Read forecast metrics without double-counting aggregate + partition files.

    The canonical fast path is the aggregate table at
    ``lakehouse/gold/forecast_metrics/metrics.parquet``. ``include_partitions``
    is available for audits and repair tools, but rows are deduplicated on the
    stable metric key so a recursive read cannot inflate evidence counts.
    """
    metrics_root = project_root / "lakehouse" / "gold" / "forecast_metrics"
    aggregate_path = metrics_root / "metrics.parquet"
    frames: list[pd.DataFrame] = []
    if aggregate_path.exists():
        frames.append(pd.read_parquet(aggregate_path))
    if include_partitions and metrics_root.exists():
        for path in sorted(metrics_root.glob("run_id=*/metrics.parquet")):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    key_cols = [c for c in FORECAST_METRIC_KEY_COLS if c in out.columns]
    if key_cols:
        out = out.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return out
