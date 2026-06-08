#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_gate import mbari_release_gate as rg  # noqa: E402


def _check(name: str, status: str = "PASS") -> rg.Check:
    return rg.Check(name, status, f"{name} {status}")


def _complete_checks(**overrides: str) -> list[rg.Check]:
    names = [
        "gpu_stack",
        "package_versions",
        "historical_parquet",
        "curated_dataset",
        "model_outputs",
        "neural_lakehouse_outputs",
        "promotion_matrix",
    ]
    return [_check(name, overrides.get(name, "PASS")) for name in names]


def test_overall_status_passes_with_enforceable_promotions_and_advisory_warnings():
    checks = _complete_checks(
        model_outputs="WARN",
        neural_lakehouse_outputs="WARN",
        promotion_matrix="PASS",
    )

    assert rg.compute_overall_status(checks) == "PASS"


def test_overall_status_keeps_structural_warnings_as_warn():
    checks = _complete_checks(
        curated_dataset="WARN",
        model_outputs="WARN",
        neural_lakehouse_outputs="WARN",
        promotion_matrix="PASS",
    )

    assert rg.compute_overall_status(checks) == "WARN"


def test_overall_status_preserves_fail_for_invalid_artifacts():
    checks = _complete_checks(
        historical_parquet="FAIL",
        model_outputs="WARN",
        neural_lakehouse_outputs="WARN",
        promotion_matrix="PASS",
    )

    assert rg.compute_overall_status(checks) == "FAIL"


def test_overall_status_warns_without_enforceable_promotions():
    checks = _complete_checks(
        model_outputs="WARN",
        neural_lakehouse_outputs="WARN",
        promotion_matrix="WARN",
    )

    assert rg.compute_overall_status(checks) == "WARN"


def test_neural_lakehouse_outputs_reports_best_and_warns_on_drivers(tmp_path):
    phase2_dir = tmp_path / "nn_results"
    phase2_dir.mkdir()
    pd.DataFrame(
        [
            {
                "outdir": "patchtst",
                "model_name": "patchtst",
                "loss": "mae",
                "drivers_enabled": False,
                "rows": 120,
                "mean_skill": 7.1,
                "median_skill": 4.0,
                "mean_rmse": 0.35,
            },
            {
                "outdir": "tsmixerx_drv_q",
                "model_name": "tsmixerx",
                "loss": "quantile",
                "drivers_enabled": True,
                "rows": 120,
                "mean_skill": -17.2,
                "median_skill": -2.6,
                "mean_rmse": 0.40,
            },
        ]
    ).to_csv(phase2_dir / "phase2_run_summary.csv", index=False)

    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "r1",
                "split_id": "s1",
                "unique_id": "air_pressure",
                "horizon_h": 6,
                "model_rmse": 1.0,
                "persistence_rmse": 2.0,
                "skill_vs_persistence_pct": 50.0,
            },
            {
                "run_id": "r2",
                "split_id": "s2",
                "unique_id": "temp_d10p0",
                "horizon_h": 24,
                "model_rmse": 2.0,
                "persistence_rmse": 1.0,
                "skill_vs_persistence_pct": -100.0,
            },
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    manifest_dir = tmp_path / "lakehouse" / "gold" / "forecast_runs"
    manifest_dir.mkdir(parents=True)
    pd.DataFrame([{"run_id": "r1"}, {"run_id": "r2"}]).to_parquet(
        manifest_dir / "run_manifest.parquet", index=False
    )

    check = rg.check_neural_lakehouse_outputs(tmp_path)
    assert check.status == "WARN"
    assert check.details["phase2_best_run"]["outdir"] == "patchtst"
    assert check.details["phase2_positive_mean_skill_runs"] == 1
    assert check.details["lakehouse_positive_skill_cells"] == 1
    assert "driver-enabled neural runs are negative" in check.summary


def test_promotion_matrix_check_reports_unique_promotion_counts(tmp_path):
    promotion_dir = tmp_path / "lakehouse" / "gold" / "promotion_matrix"
    promotion_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "schema_version": 1,
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "run-a",
                "candidate_split_id": "shared",
                "xgb_run_id": "xgb",
                "xgb_split_id": "shared",
                "candidate_skill_vs_persistence_pct": 30.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 10.0,
                "status": "promote",
                "reason": "candidate beats persistence and XGBoost on shared split",
            },
            {
                "schema_version": 1,
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "patchtst",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "run-b",
                "candidate_split_id": "shared",
                "xgb_run_id": "xgb",
                "xgb_split_id": "shared",
                "candidate_skill_vs_persistence_pct": 31.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 11.0,
                "status": "promote",
                "reason": "duplicate run for same model cell",
            },
            {
                "schema_version": 1,
                "target": "air_pressure",
                "horizon_h": 6,
                "candidate_model": "nhits",
                "candidate_loss": "mae",
                "candidate_drivers_enabled": False,
                "candidate_run_id": "run-c",
                "candidate_split_id": "shared",
                "xgb_run_id": "xgb",
                "xgb_split_id": "shared",
                "candidate_skill_vs_persistence_pct": 28.0,
                "xgb_skill_vs_persistence_pct": 20.0,
                "xgb_delta_skill_pct": 8.0,
                "status": "promote",
                "reason": "same target/horizon with different model",
            },
        ]
    ).to_parquet(promotion_dir / "promotion_matrix.parquet", index=False)
    (promotion_dir / "promotion_summary.json").write_text('{"rows": 3}', encoding="utf-8")

    check = rg.check_promotion_matrix(tmp_path)

    assert check.status == "PASS"
    assert check.details["promote_count"] == 3
    assert check.details["promoted_row_count"] == 3
    assert check.details["unique_promoted_target_horizon_count"] == 1
    assert check.details["unique_promoted_model_cell_count"] == 2


def test_historical_parquet_warns_when_auxiliary_check_fails(tmp_path, monkeypatch):
    production = tmp_path / "mbari_history" / "opendap" / "m1_history.parquet"
    auxiliary = tmp_path / "mbari_history" / "noaa" / "bad_aux.parquet"
    production.parent.mkdir(parents=True)
    auxiliary.parent.mkdir(parents=True)
    production.touch()
    auxiliary.touch()

    def fake_single(path: Path) -> rg.Check:
        assert path == production
        return rg.Check(f"parquet::{path.name}", "PASS", "ok")

    def fake_aux(path: Path) -> rg.Check:
        assert path == auxiliary
        return rg.Check(f"aux_parquet::{path.name}", "FAIL", "bad aux")

    monkeypatch.setattr(rg, "check_single_parquet", fake_single)
    monkeypatch.setattr(rg, "check_auxiliary_parquet", fake_aux)

    check = rg.check_historical_parquet(tmp_path)

    assert check.status == "WARN"
    assert "0 PASS, 0 WARN, 1 FAIL across 1 auxiliary" in check.summary
