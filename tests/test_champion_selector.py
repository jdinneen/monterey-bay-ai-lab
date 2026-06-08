#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.build_champion_selector import build_champions, summarize_champions, write_outputs  # noqa: E402


def test_build_champions_selects_best_promoted_target_horizon_cell():
    matrix = pd.DataFrame(
        [
            {
                "status": "promote",
                "target": "temp_d1p0",
                "horizon_h": 6,
                "candidate_model": "nhits",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "r1",
                "candidate_split_id": "s1",
                "candidate_rmse": 0.8,
                "candidate_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 5.0,
                "xgb_run_id": "xgb1",
                "xgb_split_id": "s1",
                "xgb_rmse": 0.9,
                "reason": "candidate beats persistence and XGBoost on shared split",
            },
            {
                "status": "promote",
                "target": "temp_d1p0",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "r2",
                "candidate_split_id": "s1",
                "candidate_rmse": 0.7,
                "candidate_skill_vs_persistence_pct": 22.0,
                "xgb_delta_skill_pct": 8.0,
                "xgb_run_id": "xgb1",
                "xgb_split_id": "s1",
                "xgb_rmse": 0.9,
                "reason": "candidate beats persistence and XGBoost on shared split",
            },
            {
                "status": "candidate_split_mismatch",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "candidate_model": "chronos",
                "candidate_loss": "zero_shot",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "r3",
                "candidate_split_id": "foundation",
                "candidate_rmse": 0.4,
                "candidate_skill_vs_persistence_pct": 40.0,
                "xgb_delta_skill_pct": 15.0,
                "xgb_run_id": "xgb2",
                "xgb_split_id": "s2",
                "xgb_rmse": 0.5,
                "reason": "candidate beats persistence and XGBoost; shared split required",
            },
        ]
    )

    champions = build_champions(matrix)
    summary = summarize_champions(champions)

    assert len(champions) == 1
    assert champions["candidate_model"].item() == "patchtst"
    assert champions["fallback_model"].item() == "xgboost"
    assert champions["fallback_run_id"].item() == "xgb1"
    assert champions["fallback_split_id"].item() == "s1"
    assert champions["serving_status"].item() == "needs_artifact"
    assert champions["requires_driver_table"].item() == False
    assert champions["driver_policy_status"].item() == "no_driver_table_required"
    assert champions["promoted_candidate_count"].item() == 2
    assert champions["duplicate_promoted_candidate_count"].item() == 2
    assert "nhits:mae" in champions["competing_promoted_models"].item()
    assert "patchtst:mae" in champions["competing_promoted_models"].item()
    assert champions["reason"].item() == "candidate beats persistence and XGBoost on shared split"
    assert summary["champion_cells"] == 1
    assert summary["models"] == {"patchtst": 1}
    assert summary["fallback_models"] == {"xgboost": 1}
    assert summary["driver_policy_required_cells"] == 0
    assert summary["needs_artifact_cells"] == 1


def test_build_champions_marks_driver_cells_for_review():
    matrix = pd.DataFrame(
        [
            {
                "status": "promote",
                "target": "temp_d1p0",
                "horizon_h": 24,
                "candidate_model": "nhits",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": True,
                "candidate_run_id": "r1",
                "candidate_split_id": "s1",
                "candidate_rmse": 0.8,
                "candidate_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 5.0,
                "xgb_run_id": "",
                "xgb_split_id": "",
                "xgb_rmse": None,
            },
        ]
    )

    champions = build_champions(matrix)
    summary = summarize_champions(champions)

    assert champions["fallback_model"].item() == "persistence"
    assert champions["fallback_run_id"].item() == ""
    assert champions["fallback_split_id"].item() == "s1"
    assert champions["serving_status"].item() == "promotion_gated_driver_policy_required"
    assert champions["requires_driver_table"].item() == True
    assert champions["driver_policy_status"].item() == "requires_driver_policy_approval"
    assert summary["driver_policy_required_cells"] == 1


def test_write_outputs_includes_reason_and_deployment_columns(tmp_path):
    matrix = pd.DataFrame(
        [
            {
                "status": "promote",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "r2",
                "candidate_split_id": "s1",
                "candidate_rmse": 0.7,
                "candidate_skill_vs_persistence_pct": 22.0,
                "xgb_delta_skill_pct": 8.0,
                "xgb_run_id": "xgb1",
                "xgb_split_id": "s1",
                "xgb_rmse": 0.9,
                "reason": "candidate beats persistence and XGBoost on shared split",
            },
        ]
    )
    champions = build_champions(matrix)

    paths = write_outputs(champions, tmp_path)
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert "fallback_model" in markdown
    assert "fallback_run_id" in markdown
    assert "fallback_reason" in markdown
    assert "serving_status" in markdown
    assert "artifact_path" in markdown
    assert "promoted_candidate_count" in markdown
    assert "duplicate_promoted_candidate_count" in markdown
    assert "competing_promoted_models" in markdown
    assert "candidate beats persistence and XGBoost on shared split" in markdown


def test_build_champions_resolves_artifact_path_against_project_root(tmp_path):
    run_id = "r2"
    artifact_dir = tmp_path / "lakehouse" / "gold" / "forecast_runs" / f"run_id={run_id}"
    artifact_dir.mkdir(parents=True)
    matrix = pd.DataFrame(
        [
            {
                "status": "promote",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": run_id,
                "candidate_split_id": "s1",
                "candidate_rmse": 0.7,
                "candidate_skill_vs_persistence_pct": 22.0,
                "xgb_delta_skill_pct": 8.0,
                "xgb_run_id": "xgb1",
                "xgb_split_id": "s1",
                "xgb_rmse": 0.9,
            },
        ]
    )

    champions = build_champions(matrix, project_root=tmp_path)

    assert champions["artifact_path"].item() == f"lakehouse/gold/forecast_runs/run_id={run_id}"
    assert champions["serving_status"].item() == "promotion_gated_candidate"


def test_build_champions_no_promotes_keeps_extended_schema():
    matrix = pd.DataFrame(
        [
            {
                "status": "reject",
                "target": "sal_d1p0",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "r1",
                "candidate_split_id": "s1",
                "candidate_rmse": 0.7,
                "candidate_skill_vs_persistence_pct": -2.0,
                "xgb_delta_skill_pct": -1.0,
            },
        ]
    )

    champions = build_champions(matrix)

    assert champions.empty
    for col in [
        "fallback_model",
        "fallback_reason",
        "fallback_run_id",
        "fallback_split_id",
        "serving_status",
        "artifact_path",
        "requires_driver_table",
        "driver_policy_status",
        "promoted_candidate_count",
        "competing_promoted_models",
        "reason",
    ]:
        assert col in champions.columns
