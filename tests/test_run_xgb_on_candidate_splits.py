#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.run_xgb_on_candidate_splits import select_jobs  # noqa: E402


def test_select_jobs_only_uses_existing_split_rows(tmp_path):
    promo_dir = tmp_path / "lakehouse" / "gold" / "promotion_matrix"
    promo_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "candidate_split_mismatch",
                "candidate_split_id": "has_rows",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "xgb_delta_skill_pct": 10.0,
            },
            {
                "status": "candidate_split_mismatch",
                "candidate_split_id": "missing_rows",
                "target": "sal_d20p0",
                "horizon_h": 72,
                "xgb_delta_skill_pct": 20.0,
            },
        ]
    ).to_parquet(promo_dir / "promotion_matrix.parquet", index=False)
    split_dir = tmp_path / "lakehouse" / "silver" / "forecast_splits" / "split_id=has_rows"
    split_dir.mkdir(parents=True)
    pd.DataFrame([{"split_id": "has_rows", "horizon_h": 6, "cutoff": "2026-01-01T00:00:00Z"}]).to_parquet(
        split_dir / "split_rows.parquet", index=False
    )

    jobs = select_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].split_id == "has_rows"
    assert jobs[0].target == "sal_d1p0"
    assert jobs[0].horizon_h == 6
