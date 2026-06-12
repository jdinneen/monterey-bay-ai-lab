#!/usr/bin/env python3
"""Driver-required champion-cell demotion gate.

WHY THIS EXISTS
---------------
Two champion cells require an exogenous driver table at serving time
(``candidate_drivers_enabled == True``). The inventory flags these for explicit
signoff. Today's driver A/B showed drivers usually HURT against best-naive, so
each driver cell must justify itself on two axes:

  1. Skill: does it still clear the promotion critic (>0.5% vs best-naive,
     >1.0% vs XGBoost, n>=20)? -> ops/promotion_critic_agent.grade_row
  2. Reproducibility: can the current code rebuild the model at all?
     -> release_gate/champion_reproducibility_gate.classify_constructibility

Recommend DEMOTE if it fails EITHER axis (grade == do_not_claim, OR the model is
not reproducible). Otherwise KEEP_WITH_DRIVER_POLICY (beats the baselines but
still needs operational driver-table approval). Read-only: recommends, does not
mutate the champion selector.

It also independently recomputes best-naive skill from the gold prediction
partition (ops/seasonal_naive.score_predictions) as an integrity cross-check
against the value stored in the promotion matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.promotion_critic_agent import grade_row  # noqa: E402
from ops.seasonal_naive import load_observed_panel, score_predictions  # noqa: E402
from release_gate.champion_reproducibility_gate import classify_constructibility  # noqa: E402


def recompute_best_naive_skill(project_root: Path, run_id, target, horizon_h, obs_val) -> float | None:
    """Recompute skill_vs_best_naive_pct for one (target, horizon) from the gold
    prediction partition, as an integrity cross-check. Returns None if unavailable."""
    if obs_val is None or run_id is None:
        return None
    part = project_root / "lakehouse" / "gold" / "forecast_predictions" / f"run_id={run_id}"
    files = list(part.glob("*.parquet"))
    if not files:
        return None
    try:
        preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        scored = score_predictions(preds, obs_val)
    except Exception:
        return None
    hit = scored[(scored["unique_id"] == target) & (scored["horizon_h"] == int(horizon_h))]
    if hit.empty:
        return None
    return float(hit["skill_vs_best_naive_pct"].iloc[0])


def assess_driver_cell(promo_row: pd.Series, reproducible_status: str) -> dict:
    """Pure decision: grade the cell and combine with reproducibility.

    reproducible_status in {PASS, WARN, FAIL} from classify_constructibility.
    """
    grade, reasons = grade_row(promo_row)
    demote_reasons = []
    if grade == "do_not_claim":
        demote_reasons.append(f"fails promotion critic: {', '.join(reasons)}")
    if reproducible_status == "FAIL":
        demote_reasons.append("model not reproducible from current code (orphaned/broken constructor)")
    recommendation = "DEMOTE" if demote_reasons else "KEEP_WITH_DRIVER_POLICY"
    return {
        "grade": grade,
        "grade_reasons": reasons,
        "reproducible": reproducible_status,
        "recommendation": recommendation,
        "demote_reasons": demote_reasons,
    }


def _promoted_row_for(matrix: pd.DataFrame, target, horizon_h, model) -> pd.Series | None:
    sel = matrix[
        (matrix["target"] == target)
        & (matrix["horizon_h"] == int(horizon_h))
        & (matrix["candidate_model"] == model)
        & (matrix["candidate_drivers_enabled"] == True)  # noqa: E712
        & (matrix["status"] == "promote")
    ]
    if sel.empty:
        return None
    return sel.iloc[0]


def build_report(project_root: Path) -> dict:
    champ = pd.read_parquet(project_root / "reports" / "champion_selector" / "champion_selector.parquet")
    matrix = pd.read_parquet(project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet")
    driver_cells = champ[champ["candidate_drivers_enabled"] == True]  # noqa: E712
    obs = load_observed_panel(project_root)

    cells = []
    for _, c in driver_cells.iterrows():
        target, h, model = c["target"], int(c["horizon_h"]), c["candidate_model"]
        promo = _promoted_row_for(matrix, target, h, model)
        repro_status, _ = classify_constructibility(model)
        if promo is None:
            cells.append({"target": target, "horizon_h": h, "candidate_model": model,
                          "recommendation": "DEMOTE", "demote_reasons": ["no matching promoted row in matrix"],
                          "reproducible": repro_status})
            continue
        assessment = assess_driver_cell(promo, repro_status)
        recomputed = recompute_best_naive_skill(project_root, c.get("candidate_run_id"), target, h, obs)
        cells.append({
            "target": target,
            "horizon_h": h,
            "candidate_model": model,
            "candidate_n": int(promo.get("candidate_n", 0)),
            "skill_vs_best_naive_pct_stored": float(promo.get("candidate_skill_vs_best_naive_pct")),
            "skill_vs_best_naive_pct_recomputed": recomputed,
            "xgb_delta_skill_pct": float(promo.get("xgb_delta_skill_pct")),
            **assessment,
        })
    n_demote = sum(1 for c in cells if c["recommendation"] == "DEMOTE")
    return {"n_driver_cells": len(cells), "n_demote": n_demote, "cells": cells}


def _to_markdown(report: dict) -> str:
    lines = [
        "# Driver-Required Champion Cell Demotion Gate",
        "",
        f"Driver cells: {report['n_driver_cells']}; recommend DEMOTE: {report['n_demote']}.",
        "",
        "DEMOTE if the cell fails the promotion critic (vs best-naive / XGBoost / n) OR",
        "its model cannot be rebuilt from current code. Otherwise KEEP_WITH_DRIVER_POLICY",
        "(beats baselines but still needs operational driver-table approval).",
        "",
        "| target | h | model | n | skill vs best-naive (stored/recomp) | xgb delta | grade | reproducible | recommendation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cells"]:
        recomp = c.get("skill_vs_best_naive_pct_recomputed")
        recomp_s = f"{recomp:.2f}" if isinstance(recomp, (int, float)) else "n/a"
        stored = c.get("skill_vs_best_naive_pct_stored")
        stored_s = f"{stored:.2f}" if isinstance(stored, (int, float)) else "n/a"
        xgbd = c.get("xgb_delta_skill_pct")
        xgbd_s = f"{xgbd:.2f}" if isinstance(xgbd, (int, float)) else "?"
        lines.append(
            f"| {c['target']} | {c['horizon_h']} | {c['candidate_model']} | {c.get('candidate_n','?')} | "
            f"{stored_s} / {recomp_s} | {xgbd_s} | {c.get('grade','?')} | "
            f"{c.get('reproducible','?')} | {c['recommendation']} |"
        )
    lines.append("")
    for c in report["cells"]:
        if c["recommendation"] == "DEMOTE":
            lines.append(f"- DEMOTE {c['target']}@{c['horizon_h']}h ({c['candidate_model']}): "
                         f"{'; '.join(c.get('demote_reasons', []))}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Driver-cell demotion gate.")
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    root = Path(a.project_root).resolve()
    out_dir = Path(a.out_dir) if a.out_dir else root / "reports" / "driver_cell_demotion"

    report = build_report(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "driver_cell_demotion.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "DRIVER_CELL_DEMOTION.md").write_text(_to_markdown(report), encoding="utf-8")
    print(f"{report['n_demote']}/{report['n_driver_cells']} driver cells recommend DEMOTE -> {out_dir}")
    return 1 if report["n_demote"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
