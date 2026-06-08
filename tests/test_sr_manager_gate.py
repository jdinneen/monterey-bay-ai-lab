#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_gate.mbari_sr_manager_gate import SrManagerGateConfig, evaluate_sr_manager_gate  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gate_artifacts(
    root: Path,
    *,
    release_status: str = "PASS",
    status_counts: dict[str, int] | None = None,
    daily: bool = True,
) -> None:
    _write_json(
        root / "release_gate" / "reports" / "release_gate_report.json",
        {"overall_status": release_status, "checks": []},
    )
    _write_json(
        root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json",
        {"rows": 10, "status_counts": status_counts or {"promote": 1}},
    )
    if daily:
        daily_path = root / "archive" / "reports" / "MBARI_AI_DAILY_LEARNINGS.md"
        daily_path.parent.mkdir(parents=True, exist_ok=True)
        daily_path.write_text("# Daily Learnings\n", encoding="utf-8")


def test_sr_manager_gate_passes_when_artifacts_are_present_with_promotions(tmp_path):
    _write_gate_artifacts(tmp_path)

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    assert result["overall_status"] == "PASS"
    assert {check["name"]: check["status"] for check in result["checks"]} == {
        "release_gate_report": "PASS",
        "promotion_summary": "PASS",
        "daily_learnings": "PASS",
    }


def test_sr_manager_gate_reports_promotion_breadth_counts(tmp_path):
    _write_gate_artifacts(tmp_path)
    _write_json(
        tmp_path / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json",
        {
            "rows": 10,
            "status_counts": {"promote": 4},
            "promoted_row_count": 4,
            "unique_promoted_target_horizon_count": 2,
            "unique_promoted_model_cell_count": 3,
        },
    )

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    promotion_check = next(check for check in result["checks"] if check["name"] == "promotion_summary")
    assert promotion_check["status"] == "PASS"
    assert "4 promoted rows across 2 target/horizon cells and 3 model cells" in promotion_check["summary"]
    assert promotion_check["details"]["promoted_row_count"] == 4
    assert promotion_check["details"]["unique_promoted_target_horizon_count"] == 2
    assert promotion_check["details"]["unique_promoted_model_cell_count"] == 3


def test_sr_manager_gate_warns_when_no_enforceable_promotions(tmp_path):
    _write_gate_artifacts(tmp_path, status_counts={"candidate_split_mismatch": 2, "reject": 3})

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    assert result["overall_status"] == "WARN"
    promotion_check = next(check for check in result["checks"] if check["name"] == "promotion_summary")
    assert promotion_check["status"] == "WARN"
    assert "no enforceable promotions" in promotion_check["summary"]


def test_sr_manager_gate_warns_when_daily_learnings_are_missing(tmp_path):
    _write_gate_artifacts(tmp_path, daily=False)

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    assert result["overall_status"] == "WARN"
    daily_check = next(check for check in result["checks"] if check["name"] == "daily_learnings")
    assert daily_check["status"] == "WARN"


def test_sr_manager_gate_fails_when_release_gate_fails(tmp_path):
    _write_gate_artifacts(tmp_path, release_status="FAIL")

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    assert result["overall_status"] == "FAIL"
    release_check = next(check for check in result["checks"] if check["name"] == "release_gate_report")
    assert release_check["status"] == "FAIL"


def test_sr_manager_gate_fails_when_promotion_summary_is_missing(tmp_path):
    _write_json(
        tmp_path / "release_gate" / "reports" / "release_gate_report.json",
        {"overall_status": "PASS", "checks": []},
    )
    daily_path = tmp_path / "archive" / "reports" / "MBARI_AI_DAILY_LEARNINGS.md"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.write_text("# Daily Learnings\n", encoding="utf-8")

    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=tmp_path))

    assert result["overall_status"] == "FAIL"
    promotion_check = next(check for check in result["checks"] if check["name"] == "promotion_summary")
    assert promotion_check["status"] == "FAIL"


def test_sr_manager_gate_cli_prints_json(tmp_path):
    _write_gate_artifacts(tmp_path, status_counts={"candidate_split_mismatch": 1})

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "release_gate" / "mbari_sr_manager_gate.py"),
            "--project-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["overall_status"] == "WARN"
    assert payload["checks"][1]["name"] == "promotion_summary"


def test_sr_manager_gate_cli_writes_output_json(tmp_path):
    _write_gate_artifacts(tmp_path)
    output_path = tmp_path / "release_gate" / "reports" / "sr_manager_gate_report.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "release_gate" / "mbari_sr_manager_gate.py"),
            "--project-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["overall_status"] == "PASS"
