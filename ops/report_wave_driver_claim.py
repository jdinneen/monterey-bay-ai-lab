#!/usr/bin/env python3
"""Report whether NDBC wave-driver ablations support a general driver claim."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


WAVE_HIST_COLS = [
    "ndbc46042_wave_height_m",
    "ndbc46042_dom_wave_period_s",
    "ndbc46042_avg_wave_period_s",
    "ndbc46042_mean_wave_dir_sin",
    "ndbc46042_mean_wave_dir_cos",
]

PAIRS = [
    ("nhits", "ablate_nhits_baseline", "ablate_nhits_ndbc_wave"),
    ("tsmixerx", "ablate_tsmixerx_baseline", "ablate_tsmixerx_ndbc_wave"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_keys_match(base_dir: Path, wave_dir: Path) -> bool:
    base = pd.read_parquet(base_dir / "cv_predictions.parquet")
    wave = pd.read_parquet(wave_dir / "cv_predictions.parquet")
    keys = ["unique_id", "ds", "cutoff"]
    if not all(k in base.columns and k in wave.columns for k in keys):
        return False
    base_keys = set(map(tuple, base[keys].astype(str).to_records(index=False)))
    wave_keys = set(map(tuple, wave[keys].astype(str).to_records(index=False)))
    return base_keys == wave_keys


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def build_report(project_root: Path) -> dict[str, Any]:
    results = project_root / "nn_results"
    manifest_path = project_root / "nn_cache" / "driver_ablation" / "ndbc_wave.json"
    manifest = _read_json(manifest_path)
    if manifest.get("hist") != WAVE_HIST_COLS:
        raise ValueError(f"{manifest_path} does not contain the expected wave-only hist driver set")

    rows: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for model, baseline_name, wave_name in PAIRS:
        baseline_dir = results / baseline_name
        wave_dir = results / wave_name
        baseline_summary = _read_json(baseline_dir / "summary.json")
        wave_summary = _read_json(wave_dir / "summary.json")
        baseline = pd.read_csv(baseline_dir / "leaderboard.csv")
        wave = pd.read_csv(wave_dir / "leaderboard.csv")
        matched = baseline.merge(wave, on=["unique_id", "horizon_h"], suffixes=("_baseline", "_wave"))
        matched["model"] = model
        matched["rmse_delta_wave_minus_baseline"] = (
            matched["model_rmse_wave"] - matched["model_rmse_baseline"]
        )
        matched["rmse_delta_pct"] = (
            100.0 * matched["rmse_delta_wave_minus_baseline"] / matched["model_rmse_baseline"]
        )
        matched["skill_delta_pp"] = (
            matched["skill_vs_persistence_pct_wave"] - matched["skill_vs_persistence_pct_baseline"]
        )
        matched["wave_wins_rmse"] = matched["rmse_delta_wave_minus_baseline"] < 0
        rows.append(matched)
        run_rows.append(
            {
                "model": model,
                "baseline_run_id": baseline_summary.get("run_id"),
                "wave_run_id": wave_summary.get("run_id"),
                "baseline_split_id": baseline_summary.get("split_id"),
                "wave_split_id": wave_summary.get("split_id"),
                "identical_prediction_keys": _prediction_keys_match(baseline_dir, wave_dir),
                "scored_rows": wave_summary.get("scored_rows"),
                "tail_weeks": wave_summary.get("tail_weeks"),
                "max_steps": wave_summary.get("max_steps"),
                "n_windows": wave_summary.get("n_windows"),
            }
        )

    all_rows = pd.concat(rows, ignore_index=True)
    summary = all_rows.groupby("model").agg(
        cells=("unique_id", "size"),
        wave_rmse_wins=("wave_wins_rmse", "sum"),
        mean_rmse_delta=("rmse_delta_wave_minus_baseline", "mean"),
        median_rmse_delta=("rmse_delta_wave_minus_baseline", "median"),
        mean_rmse_delta_pct=("rmse_delta_pct", "mean"),
        mean_skill_delta_pp=("skill_delta_pp", "mean"),
    ).reset_index()
    summary["win_rate"] = summary["wave_rmse_wins"] / summary["cells"]
    summary = summary[
        [
            "model",
            "cells",
            "wave_rmse_wins",
            "win_rate",
            "mean_rmse_delta",
            "median_rmse_delta",
            "mean_rmse_delta_pct",
            "mean_skill_delta_pp",
        ]
    ]

    wins = int(all_rows["wave_wins_rmse"].sum())
    cells = int(len(all_rows))
    win_rate = float(wins / cells) if cells else 0.0
    mean_rmse_delta_pct = float(all_rows["rmse_delta_pct"].mean())
    mean_skill_delta_pp = float(all_rows["skill_delta_pp"].mean())
    verdict = (
        "supported_as_screening_only"
        if mean_rmse_delta_pct < 0.0 and win_rate >= 0.55
        else "not_supported_as_general_claim"
    )

    delta_cols = [
        "model",
        "unique_id",
        "horizon_h",
        "model_rmse_baseline",
        "model_rmse_wave",
        "rmse_delta_wave_minus_baseline",
        "rmse_delta_pct",
        "skill_vs_persistence_pct_baseline",
        "skill_vs_persistence_pct_wave",
        "skill_delta_pp",
        "wave_wins_rmse",
        "n_baseline",
        "n_wave",
    ]
    delta_path = results / "wave_driver_delta.csv"
    all_rows[delta_cols].sort_values(["model", "unique_id", "horizon_h"]).to_csv(delta_path, index=False)

    by_horizon = all_rows.groupby(["model", "horizon_h"]).agg(
        cells=("unique_id", "size"),
        wins=("wave_wins_rmse", "sum"),
        mean_rmse_delta=("rmse_delta_wave_minus_baseline", "mean"),
        mean_rmse_delta_pct=("rmse_delta_pct", "mean"),
        mean_skill_delta_pp=("skill_delta_pp", "mean"),
    ).reset_index().sort_values(["model", "horizon_h"])

    report_path = results / "WAVE_DRIVER_CLAIM_REPORT.md"
    lines = [
        "# Wave Driver Claim Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "Bounded screening of observed-only NDBC 46042 wave drivers against matching no-driver baselines for NHITS and TSMixerx. This is not a repeated-seed production claim.",
        "",
        "## Driver Policy",
        "",
        f"- hist columns: {', '.join(manifest['hist'])}",
        f"- observed lag hours: {manifest.get('observed_driver_availability_lag_hours')}",
        f"- max historical ffill/staleness hours: {manifest.get('max_hist_ffill_hours')}",
        "- coverage: " + ", ".join(f"{k}={v}" for k, v in manifest.get("coverage", {}).items()),
        "- raw coverage: " + ", ".join(f"{k}={v}" for k, v in manifest.get("hist_raw_coverage", {}).items()),
        "",
        "## Run Comparability",
        "",
    ]
    for row in run_rows:
        lines.append(
            f"- {row['model']}: baseline_run_id={row['baseline_run_id']}, "
            f"wave_run_id={row['wave_run_id']}, baseline_split_id={row['baseline_split_id']}, "
            f"wave_split_id={row['wave_split_id']}, identical prediction keys/cutoffs="
            f"{'yes' if row['identical_prediction_keys'] else 'no'}, cells={row['scored_rows']}, "
            f"tail_weeks={row['tail_weeks']}, max_steps={row['max_steps']}, n_windows={row['n_windows']}"
        )
    lines += [
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## By Horizon",
        "",
        _markdown_table(by_horizon),
        "",
        "## Critic Verdict",
        "",
        (
            f"{verdict}: wave drivers win {wins}/{cells} cells ({win_rate:.1%}), "
            f"with mean RMSE delta {mean_rmse_delta_pct:.3f}% and mean skill delta "
            f"{mean_skill_delta_pp:.3f} pp. Treat as bounded screening evidence only; "
            "production promotion still requires the release-gate promotion matrix and no split mismatch."
        ),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    machine_report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "cells": cells,
        "wave_rmse_wins": wins,
        "win_rate": win_rate,
        "mean_rmse_delta_pct": mean_rmse_delta_pct,
        "mean_skill_delta_pp": mean_skill_delta_pp,
        "runs": run_rows,
        "summary": summary.to_dict(orient="records"),
        "paths": {
            "markdown": str(report_path),
            "delta_csv": str(delta_path),
            "json": str(results / "wave_driver_claim_summary.json"),
        },
    }
    json_path = Path(machine_report["paths"]["json"])
    json_path.write_text(json.dumps(machine_report, indent=2), encoding="utf-8")
    return machine_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."), help="project root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(Path(args.project_root).resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
