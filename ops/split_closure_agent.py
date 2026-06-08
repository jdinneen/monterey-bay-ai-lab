#!/usr/bin/env python3
"""Plan or run shared-split closure for promotion split-mismatch rows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def split_candidates(project_root: Path, matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if matrix.empty:
        return {"closable_now": pd.DataFrame(), "foundation_needs_rerun": pd.DataFrame(), "unclosable_missing_split": pd.DataFrame()}
    required = {"status", "target", "horizon_h", "candidate_split_id", "xgb_delta_skill_pct", "candidate_n"}
    missing = required - set(matrix.columns)
    if missing:
        raise ValueError(f"promotion matrix missing columns: {sorted(missing)}")
    candidates = matrix[matrix["status"].eq("candidate_split_mismatch")].copy()
    if candidates.empty:
        return {"closable_now": candidates, "foundation_needs_rerun": candidates, "unclosable_missing_split": candidates}
    candidates["xgb_delta_skill_pct"] = pd.to_numeric(candidates["xgb_delta_skill_pct"], errors="coerce")
    candidates["candidate_n"] = pd.to_numeric(candidates["candidate_n"], errors="coerce")
    candidates = candidates.sort_values(
        ["xgb_delta_skill_pct", "candidate_n"], ascending=[False, False]
    ).drop_duplicates(["target", "horizon_h", "candidate_split_id"], keep="first")
    split_root = project_root / "lakehouse" / "silver" / "forecast_splits"
    has_split = candidates["candidate_split_id"].apply(
        lambda split_id: (split_root / f"split_id={split_id}" / "split_rows.parquet").exists()
    )
    is_foundation = candidates["candidate_split_id"].astype(str).eq("foundation_zero_shot_benchmark")
    return {
        "closable_now": candidates[has_split].reset_index(drop=True),
        "foundation_needs_rerun": candidates[~has_split & is_foundation].reset_index(drop=True),
        "unclosable_missing_split": candidates[~has_split & ~is_foundation].reset_index(drop=True),
    }


def candidate_plan(project_root: Path, matrix: pd.DataFrame, max_jobs: int) -> pd.DataFrame:
    return split_candidates(project_root, matrix)["closable_now"].head(max_jobs).reset_index(drop=True)


def _status_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    matrix = pd.read_parquet(path)
    return {str(k): int(v) for k, v in matrix["status"].value_counts().sort_index().items()}


def build_report(project_root: Path, max_jobs: int, apply: bool, device: str, n_estimators: int) -> dict[str, Any]:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"promotion matrix missing: {matrix_path}")
    before_counts = _status_counts(matrix_path)
    matrix = pd.read_parquet(matrix_path)
    groups = split_candidates(project_root, matrix)
    plan = groups["closable_now"].head(max_jobs).reset_index(drop=True)
    commands: list[list[str]] = []
    if not plan.empty:
        commands.append(
            [
                sys.executable,
                str(project_root / "ops" / "run_xgb_on_candidate_splits.py"),
                "--project-root",
                str(project_root),
                "--max-jobs",
                str(max_jobs),
                "--device",
                device,
                "--n-estimators",
                str(n_estimators),
                "--apply",
            ]
        )
        commands.append(
            [
                sys.executable,
                str(project_root / "release_gate" / "mbari_promotion_matrix.py"),
                "--project-root",
                str(project_root),
                "--output-dir",
                str(project_root / "lakehouse" / "gold" / "promotion_matrix"),
            ]
        )
    executed: list[dict[str, Any]] = []
    if apply:
        for cmd in commands:
            proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True, check=False)
            executed.append({"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]})
            if proc.returncode:
                break
    after_counts = _status_counts(matrix_path)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": "WARN" if len(plan) else "PASS",
        "apply": apply,
        "before_status_counts": before_counts,
        "after_status_counts": after_counts,
        "planned_jobs": int(len(plan)),
        "closable_now": int(len(groups["closable_now"])),
        "foundation_needs_rerun": int(len(groups["foundation_needs_rerun"])),
        "unclosable_missing_split": int(len(groups["unclosable_missing_split"])),
        "plan": plan.to_dict(orient="records"),
        "commands": commands,
        "executed": executed,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_closure_report.json"
    md_path = output_dir / "SPLIT_CLOSURE_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        "# MBARI Split Closure Agent",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Apply mode: `{report['apply']}`",
        f"- Planned jobs: `{report['planned_jobs']}`",
        f"- Closable now: `{report['closable_now']}`",
        f"- Foundation rows needing rerun: `{report['foundation_needs_rerun']}`",
        f"- Missing split rows: `{report['unclosable_missing_split']}`",
        "",
        "## Plan",
        "",
    ]
    for row in report["plan"]:
        lines.append(
            f"- `{row['target']}` h={row['horizon_h']} split=`{row['candidate_split_id']}` "
            f"delta={row.get('xgb_delta_skill_pct')}"
        )
    if not report["plan"]:
        lines.append("No split-mismatch rows need closure.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-jobs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = build_report(root, args.max_jobs, args.apply, args.device, args.n_estimators)
    paths = write_outputs(report, args.output_dir or root / "reports" / "split_closure")
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, indent=2, sort_keys=True))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
