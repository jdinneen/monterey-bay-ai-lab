#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.foundation_rerun_agent import select_jobs  # noqa: E402


def test_select_jobs_keeps_exported_chronos_foundation_splits_only(tmp_path):
    promo_dir = tmp_path / "lakehouse" / "gold" / "promotion_matrix"
    promo_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "candidate_split_mismatch",
                "candidate_split_id": "foundation_zero_shot_benchmark",
                "candidate_model": "chronos",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "xgb_split_id": "shared",
                "xgb_delta_skill_pct": 5.0,
            },
            {
                "status": "candidate_split_mismatch",
                "candidate_split_id": "foundation_zero_shot_benchmark",
                "candidate_model": "timesfm",
                "target": "sal_d10p0",
                "horizon_h": 6,
                "xgb_split_id": "shared",
                "xgb_delta_skill_pct": 9.0,
            },
            {
                "status": "candidate_split_mismatch",
                "candidate_split_id": "foundation_zero_shot_benchmark",
                "candidate_model": "chronos",
                "target": "air_pressure",
                "horizon_h": 1,
                "xgb_split_id": "xgb_forecast_v2_internal",
                "xgb_delta_skill_pct": 10.0,
            },
        ]
    ).to_parquet(promo_dir / "promotion_matrix.parquet", index=False)

    jobs = select_jobs(tmp_path)

    assert len(jobs) == 1
    assert jobs[0].target == "sal_d1p0"
    assert jobs[0].split_id == "shared"
