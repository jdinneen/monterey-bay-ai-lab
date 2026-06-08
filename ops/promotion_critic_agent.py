#!/usr/bin/env python3
"""Adversarial critic for promoted MBARI forecast claims."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def grade_row(row: pd.Series) -> tuple[str, list[str]]:
    reasons: list[str] = []
    shared = _as_bool(row.get("shared_split"))
    n = pd.to_numeric(pd.Series([row.get("candidate_n")]), errors="coerce").iloc[0]
    skill = pd.to_numeric(pd.Series([row.get("candidate_skill_vs_persistence_pct")]), errors="coerce").iloc[0]
    delta = pd.to_numeric(pd.Series([row.get("xgb_delta_skill_pct")]), errors="coerce").iloc[0]
    xgb_skill = pd.to_numeric(pd.Series([row.get("xgb_skill_vs_persistence_pct")]), errors="coerce").iloc[0]
    if not shared:
        reasons.append("not on shared split")
    if pd.isna(n) or n < 20:
        reasons.append("too few scored observations")
    if pd.isna(skill) or skill <= 0:
        reasons.append("does not beat persistence")
    if pd.isna(delta) or delta <= 0:
        reasons.append("does not beat XGBoost")
    if reasons:
        return "do_not_claim", reasons
    weak = []
    if n < 50:
        weak.append("small sample size")
    if delta < 2:
        weak.append("small XGBoost margin")
    if xgb_skill <= 0:
        weak.append("beats weak XGBoost baseline")
    if _as_bool(row.get("candidate_drivers_enabled")):
        weak.append("driver-enabled caveat")
    if weak:
        return "weak", weak
    return "strong", ["shared-split promotion with adequate n and positive margins"]


def build_report(project_root: Path) -> dict[str, Any]:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    champion_path = project_root / "reports" / "champion_selector" / "champion_selector.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"promotion matrix missing: {matrix_path}")
    matrix = pd.read_parquet(matrix_path)
    promoted = matrix[matrix["status"].eq("promote")].copy()
    rows = []
    for _, row in promoted.iterrows():
        grade, reasons = grade_row(row)
        rows.append(
            {
                "target": row.get("target"),
                "horizon_h": int(row.get("horizon_h")),
                "candidate_model": row.get("candidate_model"),
                "candidate_loss": row.get("candidate_loss"),
                "candidate_drivers_enabled": bool(_as_bool(row.get("candidate_drivers_enabled"))),
                "candidate_run_id": row.get("candidate_run_id"),
                "candidate_split_id": row.get("candidate_split_id"),
                "grade": grade,
                "reasons": reasons,
                "candidate_n": row.get("candidate_n"),
                "candidate_skill_vs_persistence_pct": row.get("candidate_skill_vs_persistence_pct"),
                "xgb_skill_vs_persistence_pct": row.get("xgb_skill_vs_persistence_pct"),
                "xgb_delta_skill_pct": row.get("xgb_delta_skill_pct"),
            }
        )
    graded = pd.DataFrame(rows)
    champion_rows = 0
    if champion_path.exists() and not graded.empty:
        champions = pd.read_parquet(champion_path)
        champion_keys = set(zip(champions["target"].astype(str), champions["horizon_h"].astype(int)))
        graded["is_champion_cell"] = [
            (str(row["target"]), int(row["horizon_h"])) in champion_keys for _, row in graded.iterrows()
        ]
        champion_rows = int(graded["is_champion_cell"].sum())
    elif not graded.empty:
        graded["is_champion_cell"] = False
    counts = {str(k): int(v) for k, v in graded["grade"].value_counts().sort_index().items()} if not graded.empty else {}
    overall = "FAIL" if counts.get("do_not_claim", 0) else "WARN" if counts.get("weak", 0) else "PASS"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": overall,
        "promoted_rows": int(len(promoted)),
        "champion_promoted_rows": champion_rows,
        "grade_counts": counts,
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "promotion_critic_report.json"
    md_path = output_dir / "PROMOTION_CRITIC_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        "# MBARI Promotion Critic",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Promoted rows: `{report['promoted_rows']}`",
        f"- Grade counts: `{report['grade_counts']}`",
        "",
    ]
    for row in report["rows"][:60]:
        lines.append(
            f"- **{row['grade']}** `{row['target']}` h={row['horizon_h']} "
            f"{row['candidate_model']}/{row['candidate_loss']}: {', '.join(row['reasons'])}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = build_report(root)
    paths = write_outputs(report, args.output_dir or root / "reports" / "promotion_critic")
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, indent=2, sort_keys=True))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
