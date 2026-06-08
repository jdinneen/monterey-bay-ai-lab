#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.evidence_gate_agent import check_metric_duplicates, check_promotion_evidence  # noqa: E402


def _promotion_row(**overrides):
    row = {
        "status": "promote",
        "target": "sal_d20p0",
        "horizon_h": 6,
        "candidate_model": "patchtst",
        "candidate_loss": "mae",
        "candidate_drivers_enabled": False,
        "candidate_run_id": "neural",
        "candidate_split_id": "split-a",
        "candidate_n": 52,
        "candidate_skill_vs_persistence_pct": 20.0,
        "candidate_skill_vs_best_naive_pct": 18.0,
        "xgb_delta_skill_pct": 10.0,
        "shared_split": True,
    }
    row.update(overrides)
    return row


def test_promotion_evidence_passes_strict_promoted_row():
    matrix = pd.DataFrame([_promotion_row()])
    summary = {
        "promoted_row_count": 1,
        "unique_promoted_target_horizon_count": 1,
        "unique_promoted_model_cell_count": 1,
    }

    check = check_promotion_evidence(matrix, summary, min_n=20)

    assert check.status == "PASS"


def test_promotion_evidence_fails_unsupported_promoted_row():
    matrix = pd.DataFrame([_promotion_row(shared_split=False, xgb_delta_skill_pct=-1.0)])
    summary = {
        "promoted_row_count": 1,
        "unique_promoted_target_horizon_count": 1,
        "unique_promoted_model_cell_count": 1,
    }

    check = check_promotion_evidence(matrix, summary, min_n=20)

    assert check.status == "FAIL"
    assert "not on shared split" in check.summary
    assert "do not beat XGBoost" in check.summary


def test_promotion_evidence_fails_null_metrics_and_string_false_split():
    matrix = pd.DataFrame(
        [
            _promotion_row(
                shared_split="False",
                candidate_n=None,
                candidate_skill_vs_persistence_pct=None,
                xgb_delta_skill_pct=None,
            )
        ]
    )
    summary = {
        "promoted_row_count": 1,
        "unique_promoted_target_horizon_count": 1,
        "unique_promoted_model_cell_count": 1,
    }

    check = check_promotion_evidence(matrix, summary, min_n=20)

    assert check.status == "FAIL"
    assert "not on shared split" in check.summary
    assert "candidate_n < 20" in check.summary
    assert "do not beat persistence" in check.summary
    assert "do not beat XGBoost" in check.summary


def test_metric_duplicate_check_fails_duplicate_identity(tmp_path):
    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"run_id": "r", "split_id": "s", "unique_id": "temp", "horizon_h": 6, "model": "x", "loss": "mae"},
            {"run_id": "r", "split_id": "s", "unique_id": "temp", "horizon_h": 6, "model": "x", "loss": "mae"},
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    check = check_metric_duplicates(tmp_path)

    assert check.status == "FAIL"
    assert check.details["duplicate_metric_identities"] == 1


def test_metric_duplicate_check_allows_matching_aggregate_and_partition(tmp_path):
    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    run_dir = metrics_dir / "run_id=r"
    run_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {"run_id": "r", "split_id": "s", "unique_id": "temp", "horizon_h": 6, "model": "x", "loss": "mae"},
        ]
    )
    rows.to_parquet(metrics_dir / "metrics.parquet", index=False)
    rows.to_parquet(run_dir / "metrics.parquet", index=False)

    check = check_metric_duplicates(tmp_path)

    assert check.status == "PASS"
    assert check.details["duplicate_metric_identities"] == 0
