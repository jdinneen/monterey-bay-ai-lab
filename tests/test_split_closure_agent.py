#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.split_closure_agent import split_candidates  # noqa: E402


def test_split_candidates_separates_closable_foundation_and_missing(tmp_path):
    split_dir = tmp_path / "lakehouse" / "silver" / "forecast_splits" / "split_id=has_rows"
    split_dir.mkdir(parents=True)
    (split_dir / "split_rows.parquet").write_bytes(b"placeholder")
    matrix = pd.DataFrame(
        [
            {"status": "candidate_split_mismatch", "target": "a", "horizon_h": 6, "candidate_split_id": "has_rows", "xgb_delta_skill_pct": 3.0, "candidate_n": 50},
            {"status": "candidate_split_mismatch", "target": "b", "horizon_h": 6, "candidate_split_id": "foundation_zero_shot_benchmark", "xgb_delta_skill_pct": 5.0, "candidate_n": 52},
            {"status": "candidate_split_mismatch", "target": "c", "horizon_h": 6, "candidate_split_id": "missing", "xgb_delta_skill_pct": 1.0, "candidate_n": 30},
        ]
    )

    groups = split_candidates(tmp_path, matrix)

    assert len(groups["closable_now"]) == 1
    assert len(groups["foundation_needs_rerun"]) == 1
    assert len(groups["unclosable_missing_split"]) == 1
