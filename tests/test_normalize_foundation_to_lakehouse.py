#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.normalize_foundation_to_lakehouse import normalize_foundation_to_lakehouse  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_dry_run_plans_outputs_without_writing(tmp_path):
    project = tmp_path / "project"
    write_csv(
        project / "nn_results" / "foundation_leaderboard.csv",
        [
            {
                "target": "air_pressure",
                "h": 6,
                "rmse": 0.9,
                "baseline_rmse": 1.2,
                "skill_pct": 25.0,
                "count": 52,
                "foundation_model": "chronos",
            }
        ],
    )

    result = normalize_foundation_to_lakehouse(project, "foundation_test", apply=False)

    assert result.status == "completed"
    assert result.metrics_rows == 1
    assert not result.manifest_path.exists()
    assert not result.metrics_path.exists()


def test_apply_writes_manifest_and_normalized_metrics(tmp_path):
    project = tmp_path / "project"
    write_csv(
        project / "nn_results" / "foundation_leaderboard.csv",
        [
            {
                "unique_id": "sal_d1p0",
                "horizon_h": 24,
                "model_rmse": 0.4,
                "model_mae": 0.3,
                "persistence_rmse": 0.5,
                "skill_vs_persistence_pct": 20.0,
                "n": 50,
                "model": "chronos",
            },
            {
                "unique_id": "temp_d10p0",
                "horizon_h": 72,
                "model_rmse": 0.7,
                "model_mae": 0.5,
                "persistence_rmse": 0.8,
                "skill_vs_persistence_pct": 12.5,
                "n": 49,
                "model": "timesfm",
            },
        ],
    )

    result = normalize_foundation_to_lakehouse(project, "foundation_apply", apply=True)

    assert result.status == "completed"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "foundation_apply"
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]["metrics_rows"] == 2

    metrics = pd.read_parquet(result.metrics_path)
    assert list(metrics["run_id"].unique()) == ["foundation_apply"]
    assert set(metrics["model"]) == {"chronos", "timesfm"}
    assert set(metrics["unique_id"]) == {"sal_d1p0", "temp_d10p0"}
    assert metrics.loc[metrics["unique_id"] == "sal_d1p0", "skill_vs_persistence_pct"].iloc[0] == 20.0
    assert bool(metrics["drivers_enabled"].eq(False).all())
    assert set(["split_id", "loss", "benchmark_type"]).issubset(metrics.columns)


def test_missing_leaderboard_apply_writes_skipped_manifest_only(tmp_path):
    project = tmp_path / "project"
    summary = project / "nn_results" / "foundation_summary.md"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "\n".join(
            [
                "# Foundation",
                "At the 6h sweet spot, the best foundation model (**chronos**) has median skill +27.2% vs persistence.",
            ]
        ),
        encoding="utf-8",
    )

    result = normalize_foundation_to_lakehouse(project, "foundation_skip", apply=True)

    assert result.status == "skipped"
    assert result.metrics_rows == 0
    assert result.manifest_path.exists()
    assert not result.metrics_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "skipped"
    assert manifest["source"]["leaderboard_exists"] is False
    assert manifest["source"]["foundation_summary"]["best_model"] == "chronos"
    assert manifest["source"]["foundation_summary"]["median_skill_pct"] == 27.2


def test_cli_defaults_to_dry_run(tmp_path):
    project = tmp_path / "project"
    write_csv(
        project / "nn_results" / "foundation_leaderboard.csv",
        [
            {
                "unique_id": "air_pressure",
                "horizon_h": 1,
                "skill_vs_persistence_pct": 38.9,
                "model": "chronos",
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "normalize_foundation_to_lakehouse.py"),
            "--project-root",
            str(project),
            "--run-id",
            "foundation_cli",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["metrics_rows"] == 1
    assert not (project / "lakehouse").exists()


def _run_without_pytest() -> int:
    tests = [
        test_dry_run_plans_outputs_without_writing,
        test_apply_writes_manifest_and_normalized_metrics,
        test_missing_leaderboard_apply_writes_skipped_manifest_only,
        test_cli_defaults_to_dry_run,
    ]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for index, test in enumerate(tests):
            test(base / f"case_{index}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
