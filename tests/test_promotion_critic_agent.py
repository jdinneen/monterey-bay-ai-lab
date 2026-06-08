#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.promotion_critic_agent import grade_row  # noqa: E402


def test_grade_row_blocks_invalid_promotion_claim():
    row = pd.Series(
        {
            "shared_split": "False",
            "candidate_n": 10,
            "candidate_skill_vs_persistence_pct": -1.0,
            "xgb_delta_skill_pct": -0.1,
            "xgb_skill_vs_persistence_pct": 5.0,
        }
    )

    grade, reasons = grade_row(row)

    assert grade == "do_not_claim"
    assert "not on shared split" in reasons
    assert "does not beat persistence" in reasons


def test_grade_row_labels_weak_xgb_baseline_as_weak_not_strong():
    row = pd.Series(
        {
            "shared_split": True,
            "candidate_n": 52,
            "candidate_skill_vs_persistence_pct": 6.0,
            "xgb_delta_skill_pct": 10.0,
            "xgb_skill_vs_persistence_pct": -4.0,
            "candidate_drivers_enabled": False,
        }
    )

    grade, reasons = grade_row(row)

    assert grade == "weak"
    assert "beats weak XGBoost baseline" in reasons
