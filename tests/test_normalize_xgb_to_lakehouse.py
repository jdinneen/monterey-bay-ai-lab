#!/usr/bin/env python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.normalize_xgb_to_lakehouse import NormalizeConfig, build_metrics, normalize  # noqa: E402


def write_xgb_fixture(root: Path) -> None:
    xgb_dir = root / "mbal_forecast_v2_results"
    xgb_dir.mkdir()
    pd.DataFrame(
        [
            {
                "target": "air_pressure",
                "horizon_h": 6,
                "rmse": 2.0,
                "mae": 1.5,
                "skill_rmse_vs_persistence": 0.20,
                "usable_rows": 100,
            },
            {
                "target": "temp_d10p0",
                "horizon_h": 24,
                "rmse": 3.0,
                "mae": 2.5,
                "skill_rmse_vs_persistence": -0.50,
                "usable_rows": "",
            },
        ]
    ).to_csv(xgb_dir / "leaderboard.csv", index=False)
    (xgb_dir / "model_results.json").write_text(
        json.dumps(
            [
                {
                    "target": "air_pressure",
                    "horizon_h": 6,
                    "usable_rows": 90,
                    "folds": [
                        {"test_rows": 10, "persistence_rmse": 2.4},
                        {"test_rows": 15, "persistence_rmse": 2.6},
                    ],
                },
                {
                    "target": "temp_d10p0",
                    "horizon_h": 24,
                    "folds": [
                        {"test_rows": 20},
                        {"test_rows": 30},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_build_metrics_maps_xgb_fields_and_infers_values(tmp_path):
    write_xgb_fixture(tmp_path)

    metrics = build_metrics(NormalizeConfig(project_root=tmp_path))

    assert list(metrics["unique_id"]) == ["air_pressure", "temp_d10p0"]
    first = metrics.iloc[0]
    assert first["run_id"] == "xgb_forecast_v2"
    assert first["horizon_h"] == 6
    assert first["model_rmse"] == 2.0
    assert first["model_mae"] == 1.5
    assert first["skill_vs_persistence_pct"] == 20.0
    assert first["persistence_rmse"] == 2.5
    assert first["n"] == 100

    second = metrics.iloc[1]
    assert second["skill_vs_persistence_pct"] == -50.0
    assert second["persistence_rmse"] == 2.0
    assert second["n"] == 50


def test_dry_run_writes_nothing(tmp_path):
    write_xgb_fixture(tmp_path)

    summary = normalize(NormalizeConfig(project_root=tmp_path, apply=False))

    assert summary["mode"] == "dry-run"
    assert summary["metrics_rows"] == 2
    assert not (tmp_path / "lakehouse").exists()


def test_apply_writes_run_manifest_metrics_and_dedupes_aggregate(tmp_path):
    write_xgb_fixture(tmp_path)
    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "xgb_forecast_v2",
                "split_id": "old",
                "unique_id": "old",
                "horizon_h": 1,
                "model_rmse": 99.0,
            },
            {
                "run_id": "neural",
                "split_id": "s",
                "unique_id": "air_pressure",
                "horizon_h": 6,
                "model_rmse": 1.0,
            },
        ]
    ).to_parquet(metrics_dir / "metrics.parquet", index=False)

    summary = normalize(NormalizeConfig(project_root=tmp_path, apply=True))

    assert summary["mode"] == "apply"
    manifest_path = tmp_path / "lakehouse" / "gold" / "forecast_runs" / "run_id=xgb_forecast_v2" / "run_manifest.json"
    run_metrics_path = metrics_dir / "run_id=xgb_forecast_v2" / "metrics.parquet"
    aggregate_path = metrics_dir / "metrics.parquet"
    assert manifest_path.exists()
    assert run_metrics_path.exists()
    assert aggregate_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "xgb_forecast_v2"
    assert manifest["metrics_rows"] == 2

    aggregate = pd.read_parquet(aggregate_path)
    assert len(aggregate[aggregate["run_id"] == "xgb_forecast_v2"]) == 2
    assert len(aggregate[aggregate["run_id"] == "neural"]) == 1
    assert "old" not in set(aggregate["unique_id"])


def test_cli_dry_run_prints_summary_and_does_not_write(tmp_path):
    write_xgb_fixture(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "normalize_xgb_to_lakehouse.py"),
            "--project-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["mode"] == "dry-run"
    assert summary["metrics_rows"] == 2
    assert not (tmp_path / "lakehouse").exists()
