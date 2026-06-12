#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import generate_daily_learnings as gdl  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    write_csv(
        project / "nn_results" / "phase2_run_summary.csv",
        [
            {
                "outdir": "nhits",
                "model_name": "nhits",
                "loss": "mae",
                "drivers_enabled": False,
                "rows": 120,
                "mean_skill": 3.9796,
                "median_skill": 2.075,
                "mean_rmse": 0.3587,
            },
            {
                "outdir": "patchtst",
                "model_name": "patchtst",
                "loss": "mae",
                "drivers_enabled": False,
                "rows": 120,
                "mean_skill": 7.0736,
                "median_skill": 3.955,
                "mean_rmse": 0.3547,
            },
            {
                "outdir": "tsmixerx_drv",
                "model_name": "tsmixerx",
                "loss": "mae",
                "drivers_enabled": True,
                "rows": 120,
                "mean_skill": -19.2883,
                "median_skill": -4.15,
                "mean_rmse": 0.4089,
            },
        ],
    )
    write_csv(
        project / "nn_results" / "phase2_driver_delta.csv",
        [
            {
                "model_name": "tsmixerx",
                "rmse_delta_drv_minus_base": -0.4,
                "skill_delta_drv_minus_base": 10.0,
            },
            {
                "model_name": "tsmixerx",
                "rmse_delta_drv_minus_base": 0.2,
                "skill_delta_drv_minus_base": -3.0,
            },
        ],
    )
    report_path = project / "release_gate" / "reports" / "release_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": "2026-06-07T22:56:31+00:00",
                "overall_status": "WARN",
                "checks": [
                    {
                        "name": "neural_lakehouse_outputs",
                        "status": "WARN",
                        "summary": "driver-enabled neural runs are negative on median mean-skill",
                        "details": {
                        "lakehouse_positive_skill_cells": 477,
                        "lakehouse_total_skill_cells": 1090,
                        },
                    },
                    {
                        "name": "promotion_matrix",
                        "status": "WARN",
                        "summary": "promotion matrix has candidates but no enforceable promotions",
                        "details": {
                            "rows": 1090,
                            "promote_count": 0,
                            "candidate_count": 51,
                            "status_counts": {
                                "candidate_split_mismatch": 51,
                                "insufficient_data": 802,
                                "reject": 237,
                            },
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (project / "nn_results" / "foundation_summary.md").write_text(
        "\n".join(
            [
                "# Foundation",
                "",
                "At the 6h sweet spot, the best foundation model (**chronos**) has median skill +27.2% vs persistence.",
                "",
                "=> Foundation models **BEAT** our deep models at 6h (delta +4.6 pts, median).",
            ]
        ),
        encoding="utf-8",
    )
    promotion_dir = project / "lakehouse" / "gold" / "promotion_matrix"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    (promotion_dir / "promotion_summary.json").write_text(
        json.dumps(
            {
                "rows": 1090,
                "status_counts": {
                    "candidate_split_mismatch": 51,
                    "insufficient_data": 802,
                    "reject": 237,
                },
                "best_candidates": [
                    {
                        "target": "sal_d1p0",
                        "horizon_h": 6,
                        "candidate_model": "nhits",
                        "xgb_delta_skill_pct": 28.1438,
                        "status": "candidate_split_mismatch",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return project


def test_generate_daily_learnings_deterministic_report(tmp_path):
    project = make_project(tmp_path)
    output = tmp_path / "daily.md"

    first = gdl.generate_daily_learnings(project, output)
    second = gdl.generate_daily_learnings(project, output)

    assert first == second
    assert output.read_text(encoding="utf-8") == first
    assert "## Executive Summary" in first
    assert "## Best Completed Neural Run" in first
    assert "## Driver Lessons" in first
    assert "## Promotion Matrix" in first
    assert "## Release Gate Status" in first
    assert "## Next Experiments" in first
    assert "Best completed neural run is patchtst (patchtst, mean skill 7.0736%)." in first
    assert "Drivers improved RMSE in 1/2 cells." in first
    assert "Overall status: WARN." in first
    assert "Lakehouse positive skill cells: 477/1090." in first
    assert "Promotion matrix: 0 promote, 51 candidate, 237 reject." in first
    assert "Top review candidate: sal_d1p0 +6h nhits delta vs XGBoost 28.1438 pts" in first
    assert "Promotion matrix check: WARN - promotion matrix has candidates but no enforceable promotions." in first
    assert "Foundation models **BEAT** our deep models at 6h" in first


def test_missing_optional_artifacts_are_reported(tmp_path):
    project = tmp_path / "project"
    write_csv(
        project / "nn_results" / "phase2_run_summary.csv",
        [
            {
                "outdir": "a",
                "model_name": "model_a",
                "loss": "mae",
                "drivers_enabled": False,
                "rows": 1,
                "mean_skill": 1.0,
                "median_skill": 0.5,
                "mean_rmse": 2.0,
            }
        ],
    )

    report = gdl.generate_daily_learnings(project, tmp_path / "out.md")

    assert "Release gate status: MISSING." in report
    assert "No driver delta artifact was found." in report
    assert "Promotion matrix artifact was not found." in report
    assert "Run or attach foundation_summary.md" in report


def test_cli_writes_requested_output(tmp_path):
    project = make_project(tmp_path)
    output = tmp_path / "requested.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "generate_daily_learnings.py"),
            "--project-root",
            str(project),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(output) in completed.stdout
    assert output.exists()
    assert "Monterey Bay AI Lab Daily Learnings" in output.read_text(encoding="utf-8")


def _run_without_pytest() -> int:
    tests = [
        test_generate_daily_learnings_deterministic_report,
        test_missing_optional_artifacts_are_reported,
        test_cli_writes_requested_output,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for index, test in enumerate(tests):
            test(base / f"case_{index}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
