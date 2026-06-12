#!/usr/bin/env python3
"""Tests for release_gate/driver_cell_demotion_gate.assess_driver_cell.

DEMOTE if the cell fails the promotion critic OR is not reproducible; otherwise
KEEP_WITH_DRIVER_POLICY. Uses the real grade_row, synthetic rows only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_gate.driver_cell_demotion_gate import assess_driver_cell  # noqa: E402


def _row(**kw):
    base = {
        "shared_split": True,
        "candidate_n": 52,
        "candidate_skill_vs_persistence_pct": 5.0,
        "candidate_skill_vs_best_naive_pct": 5.0,
        "xgb_delta_skill_pct": 7.78,
        "xgb_skill_vs_persistence_pct": -3.38,
        "candidate_drivers_enabled": True,
    }
    base.update(kw)
    return pd.Series(base)


def test_weak_lift_vs_xgb_demotes_on_skill():
    # xgb_delta 0.14% < 1.0% bar -> grade_row returns do_not_claim -> DEMOTE.
    out = assess_driver_cell(_row(xgb_delta_skill_pct=0.14), "PASS")
    assert out["grade"] == "do_not_claim"
    assert out["recommendation"] == "DEMOTE"


def test_beats_baselines_and_reproducible_is_kept():
    out = assess_driver_cell(_row(), "PASS")
    assert out["grade"] != "do_not_claim"
    assert out["recommendation"] == "KEEP_WITH_DRIVER_POLICY"


def test_orphaned_model_demotes_even_if_skill_ok():
    # Same strong row, but the model cannot be rebuilt -> DEMOTE on reproducibility.
    out = assess_driver_cell(_row(), "FAIL")
    assert out["recommendation"] == "DEMOTE"
    assert any("not reproducible" in r for r in out["demote_reasons"])
