#!/usr/bin/env python3
"""Champion-cell reproducibility + artifact-existence gate.

WHY THIS EXISTS
---------------
The production-hardening inventory makes a hard rule: a champion row is NOT
production-ready if the current code cannot rebuild its model, or if its
artifact does not resolve on disk. The champion selector advertises 12 cells,
but the live ``make_model`` factory in mbal_neural_forecast.py can only build
``nhits``/``patchtst`` (and the wrapper-backed ``mbal_moe``); ``nbeatsx``/``tft``
champion rows are orphaned artifact evidence, ``mbal_recursive`` has no backing
class, and ``chronos``/``timesfm`` are zero-shot foundation models with no local
constructor. This gate turns that into a pass/fail check.

It is additive and read-only: it reads the champion selector + tries to
construct each cell's model + checks each artifact_path, then writes a report.
It does not mutate the selector or the lakehouse.

Binding inventory rule: "Do not use a champion row as production-ready if the
current code cannot rebuild its model."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Zero-shot foundation models have no local make_model constructor; they are
# servable only by re-running the foundation harness, so we WARN (not FAIL).
ZEROSHOT_MODELS = {"chronos", "timesfm"}

# Verdict ordering for "worst-of" aggregation.
_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _default_builder(model: str) -> tuple[bool, str]:
    """Attempt to construct ``model`` via the live make_model factory.

    Returns (constructible, detail). Imports are lazy so importing this module
    (e.g. in tests) never pulls in neuralforecast/torch.
    """
    try:
        from mbal_neural_forecast import make_model  # lazy: avoids NF import at module load
    except Exception as e:  # pragma: no cover - environment issue, not a model issue
        return False, f"make_model import failed: {type(e).__name__}: {e}"
    try:
        is_local_wrapper = model.startswith("mbal_")
        mdl = make_model(model, h=24, input_size=168, max_steps=1,
                         semantic_info=[("f", 1.0)] * 3 if is_local_wrapper else None)
        # Local wrapper models (mbal_*) can CONSTRUCT while their forward/semantic_info wiring
        # is broken — that would be a false PASS. Run a tiny forward smoke so "constructible"
        # actually means "servable" for these (P2 hardening). Library models (NHITS/PatchTST)
        # have standard NeuralForecast forwards and need no extra smoke.
        if is_local_wrapper:
            import torch

            mdl.forward({"insample_y": torch.randn(2, 168, 3)})
            return True, "constructible + forward smoke passed"
        return True, "constructible via make_model"
    except ValueError as e:
        # make_model raises ValueError("unknown model ...") for names it cannot build.
        return False, f"orphaned: make_model cannot build ({e})"
    except Exception as e:
        # construct OR forward-smoke failure (e.g. a wrapper whose forward wiring is invalid).
        return False, f"constructor/forward error: {type(e).__name__}: {e}"


def classify_constructibility(
    model: str,
    builder: Callable[[str], tuple[bool, str]] = _default_builder,
) -> tuple[str, str]:
    """Return (status, reason) for whether the model can be rebuilt from code.

    status in {PASS, WARN, FAIL}. ``builder`` is injectable for testing so the
    test never needs neuralforecast.
    """
    m = (model or "").strip().lower()
    if m in ZEROSHOT_MODELS:
        return "WARN", "zero-shot foundation model; no local make_model constructor (re-run harness to serve)"
    ok, detail = builder(m)
    return ("PASS", detail) if ok else ("FAIL", detail)


def check_artifact(artifact_path: str, project_root: Path) -> tuple[str, str]:
    """Resolve artifact_path (abs or relative to project_root) and check it exists."""
    artifact = str(artifact_path or "").strip()
    if not artifact:
        return "FAIL", "no artifact_path"
    p = Path(artifact)
    if not p.is_absolute():
        p = project_root / p
    if p.exists():
        return "PASS", f"artifact resolves: {p}"
    return "FAIL", f"artifact missing: {p}"


def grade_champions(
    champions: pd.DataFrame,
    project_root: Path,
    builder: Callable[[str], tuple[bool, str]] = _default_builder,
) -> pd.DataFrame:
    """Return a per-cell verdict table (pure; no IO besides artifact existence)."""
    rows = []
    for _, r in champions.iterrows():
        model = r.get("candidate_model")
        c_status, c_reason = classify_constructibility(model, builder)
        a_status, a_reason = check_artifact(r.get("artifact_path", ""), project_root)
        verdict = max([c_status, a_status], key=lambda s: _RANK[s])
        rows.append(
            {
                "target": r.get("target"),
                "horizon_h": r.get("horizon_h"),
                "candidate_model": model,
                "drivers_enabled": bool(r.get("candidate_drivers_enabled", False)),
                "constructible": c_status,
                "constructible_reason": c_reason,
                "artifact_ok": a_status,
                "artifact_reason": a_reason,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def overall_verdict(graded: pd.DataFrame) -> str:
    if graded.empty:
        return "FAIL"
    return max(graded["verdict"], key=lambda s: _RANK[s])


def _to_markdown(graded: pd.DataFrame, overall: str) -> str:
    # ASCII-only (cp1252-safe).
    lines = [
        "# Champion Reproducibility + Artifact Gate",
        "",
        f"Overall verdict: **{overall}**",
        "",
        "A champion cell is production-serviceable only if the current code can",
        "rebuild its model AND its artifact_path resolves on disk.",
        "",
        "| target | h | model | drivers | constructible | artifact | verdict | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in graded.iterrows():
        reason = r["constructible_reason"] if r["verdict"] == r["constructible"] and r["constructible"] != "PASS" else r["artifact_reason"] if r["artifact_ok"] != "PASS" else r["constructible_reason"]
        lines.append(
            f"| {r['target']} | {r['horizon_h']} | {r['candidate_model']} | "
            f"{'yes' if r['drivers_enabled'] else 'no'} | {r['constructible']} | "
            f"{r['artifact_ok']} | {r['verdict']} | {reason} |"
        )
    n_fail = int((graded["verdict"] == "FAIL").sum())
    n_warn = int((graded["verdict"] == "WARN").sum())
    n_pass = int((graded["verdict"] == "PASS").sum())
    lines += ["", f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL of {len(graded)} cells."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Champion reproducibility + artifact gate.")
    ap.add_argument("--project-root", default=str(ROOT))
    ap.add_argument("--champion", default=None, help="champion_selector.parquet (default: reports/champion_selector/)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: reports/champion_reproducibility/)")
    a = ap.parse_args()

    root = Path(a.project_root).resolve()
    champ_path = Path(a.champion) if a.champion else root / "reports" / "champion_selector" / "champion_selector.parquet"
    out_dir = Path(a.out_dir) if a.out_dir else root / "reports" / "champion_reproducibility"
    if not champ_path.exists():
        print(f"ERROR: champion selector not found: {champ_path}", file=sys.stderr)
        return 2

    champions = pd.read_parquet(champ_path)
    graded = grade_champions(champions, root)
    overall = overall_verdict(graded)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_verdict": overall,
        "n_cells": int(len(graded)),
        "n_fail": int((graded["verdict"] == "FAIL").sum()),
        "n_warn": int((graded["verdict"] == "WARN").sum()),
        "n_pass": int((graded["verdict"] == "PASS").sum()),
        "cells": graded.to_dict(orient="records"),
    }
    (out_dir / "champion_reproducibility.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "CHAMPION_REPRODUCIBILITY.md").write_text(_to_markdown(graded, overall), encoding="utf-8")
    print(f"{overall}: {payload['n_pass']} PASS, {payload['n_warn']} WARN, {payload['n_fail']} FAIL "
          f"of {payload['n_cells']} cells -> {out_dir}")
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
