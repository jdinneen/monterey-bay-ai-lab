#!/usr/bin/env python3
"""Tests for release_gate/champion_reproducibility_gate.

Verifies the gate: orphaned models FAIL, zero-shot models WARN, missing
artifacts FAIL, and a constructible model with a resolving artifact PASSes.
The model builder is injected so the test never imports neuralforecast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_gate.champion_reproducibility_gate import (  # noqa: E402
    classify_constructibility,
    grade_champions,
    overall_verdict,
)


def _fake_builder(model: str):
    # Pretend only nhits/patchtst/mbal_moe are constructible.
    if model in {"nhits", "patchtst", "mbal_moe"}:
        return True, "constructible via make_model"
    if model == "mbal_recursive":
        return False, "constructor error: ImportError: cannot import MBAL_Recursive_Transformer"
    return False, f"orphaned: make_model cannot build (unknown model {model})"


def test_classify_zeroshot_is_warn():
    status, reason = classify_constructibility("chronos", _fake_builder)
    assert status == "WARN"
    assert "zero-shot" in reason


def test_classify_orphaned_is_fail():
    assert classify_constructibility("nbeatsx", _fake_builder)[0] == "FAIL"
    assert classify_constructibility("tft", _fake_builder)[0] == "FAIL"


def test_classify_recursive_missing_class_is_fail():
    status, reason = classify_constructibility("mbal_recursive", _fake_builder)
    assert status == "FAIL"
    assert "Recursive" in reason


def test_grade_and_overall(tmp_path):
    # One existing artifact dir; one missing.
    good = tmp_path / "runs" / "good"
    good.mkdir(parents=True)

    champions = pd.DataFrame(
        [
            {"target": "t1", "horizon_h": 6, "candidate_model": "patchtst",
             "candidate_drivers_enabled": False, "artifact_path": "runs/good"},      # PASS
            {"target": "t2", "horizon_h": 24, "candidate_model": "nbeatsx",
             "candidate_drivers_enabled": False, "artifact_path": "runs/good"},      # FAIL (orphan)
            {"target": "t3", "horizon_h": 6, "candidate_model": "chronos",
             "candidate_drivers_enabled": False, "artifact_path": "runs/good"},      # WARN
            {"target": "t4", "horizon_h": 1, "candidate_model": "nhits",
             "candidate_drivers_enabled": False, "artifact_path": "runs/missing"},   # FAIL (artifact)
        ]
    )

    graded = grade_champions(champions, tmp_path, _fake_builder)
    by_target = {r["target"]: r["verdict"] for _, r in graded.iterrows()}
    assert by_target == {"t1": "PASS", "t2": "FAIL", "t3": "WARN", "t4": "FAIL"}
    assert overall_verdict(graded) == "FAIL"
    # nhits is constructible but its artifact is missing -> the FAIL is on the artifact axis.
    t4 = graded[graded["target"] == "t4"].iloc[0]
    assert t4["constructible"] == "PASS"
    assert t4["artifact_ok"] == "FAIL"
