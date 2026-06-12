#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_gate.mbal_promotion_matrix import (  # noqa: E402
    PromotionConfig,
    build_promotion_matrix,
    make_markdown,
    promotion_uniqueness_counts,
    summarize_matrix,
    write_outputs,
)


def test_promotion_matrix_marks_split_mismatch_candidate(tmp_path):
    xgb_dir = tmp_path / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "rmse": 1.0,
                "skill_rmse_vs_persistence": 0.20,
            }
        ]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "candidate-run",
                "split_id": "neural-split",
                "unique_id": "air_pressure",
                "horizon_h": 6,
                "model_rmse": 0.8,
                "model_mae": 0.6,
                "persistence_rmse": 1.5,
                "skill_vs_persistence_pct": 30.0,
                "n": 52,
                "model": "patchtst",
                "loss": "mae",
                "drivers_enabled": False,
            }
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    matrix = build_promotion_matrix(PromotionConfig(project_root=tmp_path))
    assert len(matrix) == 1
    row = matrix.iloc[0]
    assert row["xgb_delta_skill_pct"] == 10.0
    assert row["status"] == "candidate_split_mismatch"
    assert "shared split required" in row["reason"]


def test_promotion_matrix_can_allow_split_mismatch(tmp_path):
    xgb_dir = tmp_path / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [{"target": "temp_d10p0", "horizon_h": 24, "rmse": 1.0, "skill_rmse_vs_persistence": 0.05}]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "r",
                "split_id": "s",
                "unique_id": "temp_d10p0",
                "horizon_h": 24,
                "model_rmse": 0.7,
                "model_mae": 0.5,
                "persistence_rmse": 1.2,
                "seasonal_naive_rmse": 1.1,
                "skill_vs_persistence_pct": 15.0,
                "skill_vs_best_naive_pct": 12.0,
                "n": 30,
                "model": "chronos",
                "loss": "mae",
                "drivers_enabled": False,
            }
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    cfg = PromotionConfig(project_root=tmp_path, allow_split_mismatch=True)
    matrix = build_promotion_matrix(cfg)
    assert matrix.iloc[0]["status"] == "promote"


def test_promotion_matrix_promotes_matching_split(tmp_path):
    xgb_dir = tmp_path / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "target": "temp_d10p0",
                "horizon_h": 24,
                "rmse": 1.0,
                "skill_rmse_vs_persistence": 0.05,
                "run_id": "xgb-r",
                "split_id": "shared-split",
            }
        ]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "candidate-r",
                "split_id": "shared-split",
                "unique_id": "temp_d10p0",
                "horizon_h": 24,
                "model_rmse": 0.7,
                "model_mae": 0.5,
                "persistence_rmse": 1.2,
                "seasonal_naive_rmse": 1.1,
                "skill_vs_persistence_pct": 15.0,
                "skill_vs_best_naive_pct": 12.0,
                "n": 30,
                "model": "patchtst",
                "loss": "mae",
                "drivers_enabled": False,
            }
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    matrix = build_promotion_matrix(PromotionConfig(project_root=tmp_path))
    row = matrix.iloc[0]
    assert row["xgb_run_id"] == "xgb-r"
    assert row["xgb_split_id"] == "shared-split"
    assert row["status"] == "promote"
    assert "shared split" in row["reason"]


def test_promotion_matrix_blocks_different_exported_split(tmp_path):
    xgb_dir = tmp_path / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "target": "temp_d10p0",
                "horizon_h": 24,
                "rmse": 1.0,
                "skill_rmse_vs_persistence": 0.05,
                "run_id": "xgb-r",
                "split_id": "xgb-split",
            }
        ]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "candidate-r",
                "split_id": "neural-split",
                "unique_id": "temp_d10p0",
                "horizon_h": 24,
                "model_rmse": 0.7,
                "model_mae": 0.5,
                "persistence_rmse": 1.2,
                "seasonal_naive_rmse": 1.1,
                "skill_vs_persistence_pct": 15.0,
                "skill_vs_best_naive_pct": 12.0,
                "n": 30,
                "model": "patchtst",
                "loss": "mae",
                "drivers_enabled": False,
            }
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    matrix = build_promotion_matrix(PromotionConfig(project_root=tmp_path))
    assert matrix.iloc[0]["status"] == "candidate_split_mismatch"


def test_promotion_matrix_suppresses_superseded_foundation_screening_row(tmp_path):
    xgb_dir = tmp_path / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "target": "sal_d1p0",
                "horizon_h": 6,
                "rmse": 0.9,
                "skill_rmse_vs_persistence": 0.05,
                "run_id": "xgb-r",
                "split_id": "shared-split",
            }
        ]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "foundation_benchmark",
                "split_id": "foundation_zero_shot_benchmark",
                "unique_id": "sal_d1p0",
                "horizon_h": 6,
                "model_rmse": 0.8,
                "model_mae": 0.5,
                "persistence_rmse": 1.0,
                "skill_vs_persistence_pct": 20.0,
                "skill_vs_best_naive_pct": 20.0,
                "n": 30,
                "model": "chronos",
                "loss": "zero_shot",
                "drivers_enabled": False,
            },
            {
                "run_id": "foundation_shared",
                "split_id": "shared-split",
                "unique_id": "sal_d1p0",
                "horizon_h": 6,
                "model_rmse": 0.7,
                "model_mae": 0.4,
                "persistence_rmse": 1.0,
                "skill_vs_persistence_pct": 30.0,
                "skill_vs_best_naive_pct": 30.0,
                "n": 30,
                "model": "chronos",
                "loss": "zero_shot_shared",
                "drivers_enabled": False,
            },
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    matrix = build_promotion_matrix(PromotionConfig(project_root=tmp_path))

    assert len(matrix) == 1
    assert matrix.iloc[0]["candidate_run_id"] == "foundation_shared"
    assert matrix.iloc[0]["status"] == "promote"


def test_promotion_summary_distinguishes_raw_rows_from_unique_promoted_cells():
    matrix = pd.DataFrame(
        [
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_skill_vs_persistence_pct": 30.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 10.0,
                "status": "promote",
                "reason": "candidate beats persistence and XGBoost on shared split",
            },
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_skill_vs_persistence_pct": 31.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 11.0,
                "status": "promote",
                "reason": "duplicate run for same model cell",
            },
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "nhits",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_skill_vs_persistence_pct": 28.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 8.0,
                "status": "promote",
                "reason": "same target/horizon with different model",
            },
            {
                "target": "temp_d100p0",
                "horizon_h": 24,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": True,
                "candidate_skill_vs_persistence_pct": 18.0,
                "xgb_skill_vs_persistence_pct": 10.0,
                "xgb_delta_skill_pct": 8.0,
                "status": "promote",
                "reason": "second target/horizon",
            },
            {
                "target": "sal_d100p0",
                "horizon_h": 24,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_skill_vs_persistence_pct": 6.0,
                "xgb_skill_vs_persistence_pct": 8.0,
                "xgb_delta_skill_pct": -2.0,
                "status": "reject",
                "reason": "candidate does not match or beat XGBoost skill",
            },
        ]
    )

    counts = promotion_uniqueness_counts(matrix)
    assert counts == {
        "promoted_row_count": 4,
        "unique_promoted_target_horizon_count": 2,
        "unique_promoted_model_cell_count": 3,
    }

    summary = summarize_matrix(matrix)
    assert summary["status_counts"]["promote"] == 4
    assert summary["promoted_row_count"] == 4
    assert summary["unique_promoted_target_horizon_count"] == 2
    assert summary["unique_promoted_model_cell_count"] == 3

    markdown = make_markdown(matrix, summary)
    assert "Raw promoted rows: `4`" in markdown
    assert "Unique promoted target/horizon cells: `2`" in markdown
    assert "Unique promoted model cells: `3`" in markdown


def test_write_outputs_mirrors_current_promotion_artifacts_to_reports(tmp_path):
    matrix = pd.DataFrame(
        [
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_skill_vs_persistence_pct": 30.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 10.0,
                "status": "promote",
                "reason": "candidate beats persistence and XGBoost on shared split",
            }
        ]
    )

    paths = write_outputs(matrix, tmp_path, tmp_path / "lakehouse" / "gold" / "promotion_matrix")

    assert Path(paths["reports_parquet"]).exists()
    assert Path(paths["reports_json"]).exists()
    assert Path(paths["markdown"]).exists()
    mirrored = pd.read_parquet(paths["reports_parquet"])
    assert len(mirrored) == 1
    assert mirrored.iloc[0]["target"] == "air_pressure"
