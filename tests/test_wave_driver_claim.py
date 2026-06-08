#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.report_wave_driver_claim import WAVE_HIST_COLS, build_report  # noqa: E402


def _write_run(root: Path, name: str, model: str, run_id: str, rmse: list[float]) -> None:
    out = root / "nn_results" / name
    out.mkdir(parents=True)
    summary = {
        "run_id": run_id,
        "split_id": "same-keys",
        "scored_rows": 2,
        "tail_weeks": 104.0,
        "max_steps": 500,
        "n_windows": 16,
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "unique_id": "temp_d1p0",
                "horizon_h": 1,
                "model_rmse": rmse[0],
                "model_mae": rmse[0],
                "persistence_rmse": 1.0,
                "skill_vs_persistence_pct": 100.0 * (1.0 - rmse[0]),
                "n": 2,
                "model": model,
            },
            {
                "unique_id": "temp_d1p0",
                "horizon_h": 6,
                "model_rmse": rmse[1],
                "model_mae": rmse[1],
                "persistence_rmse": 1.0,
                "skill_vs_persistence_pct": 100.0 * (1.0 - rmse[1]),
                "n": 2,
                "model": model,
            },
        ]
    ).to_csv(out / "leaderboard.csv", index=False)
    pd.DataFrame(
        {
            "unique_id": ["temp_d1p0", "temp_d1p0"],
            "ds": pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1h"),
            "cutoff": pd.date_range("2025-12-31T00:00:00Z", periods=2, freq="1h"),
            model.upper(): [0.1, 0.2],
            "y": [0.1, 0.2],
        }
    ).to_parquet(out / "cv_predictions.parquet", index=False)


def test_wave_driver_claim_report_rejects_mixed_or_worse_general_claim(tmp_path):
    manifest_dir = tmp_path / "nn_cache" / "driver_ablation"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "ndbc_wave.json").write_text(
        json.dumps(
            {
                "futr": [],
                "hist": WAVE_HIST_COLS,
                "coverage": {col: 0.7 for col in WAVE_HIST_COLS},
                "hist_raw_coverage": {col: 0.64 for col in WAVE_HIST_COLS},
                "observed_driver_availability_lag_hours": 1,
                "max_hist_ffill_hours": 168,
            }
        ),
        encoding="utf-8",
    )

    _write_run(tmp_path, "ablate_nhits_baseline", "nhits", "nhits-base", [1.0, 1.0])
    _write_run(tmp_path, "ablate_nhits_ndbc_wave", "nhits", "nhits-wave", [0.9, 1.2])
    _write_run(tmp_path, "ablate_tsmixerx_baseline", "tsmixerx", "tsmixerx-base", [1.0, 1.0])
    _write_run(tmp_path, "ablate_tsmixerx_ndbc_wave", "tsmixerx", "tsmixerx-wave", [1.1, 1.2])

    report = build_report(tmp_path)

    assert report["verdict"] == "not_supported_as_general_claim"
    assert report["cells"] == 4
    assert report["wave_rmse_wins"] == 1
    assert Path(report["paths"]["markdown"]).exists()
    assert Path(report["paths"]["delta_csv"]).exists()
