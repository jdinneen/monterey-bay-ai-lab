#!/usr/bin/env python3
"""Generate a concise neural model comparison report from summary.json files."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "nn_results"
OUT = RESULTS / "MODEL_COMPARISON.md"


def load_summaries() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS.glob("*/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append({
            "path": path,
            "label": path.parent.name,
            "model": summary.get("model"),
            "loss": summary.get("loss"),
            "tail_weeks": summary.get("tail_weeks"),
            "drivers": bool(summary.get("drivers")),
            "minutes": summary.get("minutes"),
            "scored_rows": summary.get("scored_rows"),
            "rmse": summary.get("mean_rmse_by_h") or {},
            "coverage": summary.get("interval_coverage_0.8"),
            "width": summary.get("mean_interval_width_0.8"),
        })
    return rows


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(rows: list[dict]) -> str:
    headers = [
        "label", "model", "loss", "drivers", "tail_weeks", "minutes",
        "rmse_h1", "rmse_h24", "rmse_h168", "coverage_0.8", "width_0.8",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        rmse = r["rmse"]
        vals = [
            r["label"],
            r["model"],
            r["loss"],
            r["drivers"],
            r["tail_weeks"],
            r["minutes"],
            rmse.get("1"),
            rmse.get("24"),
            rmse.get("168"),
            r["coverage"],
            r["width"],
        ]
        lines.append("| " + " | ".join(fmt(v) for v in vals) + " |")
    return "\n".join(lines)


def main() -> int:
    rows = load_summaries()
    full = [r for r in rows if not r["tail_weeks"]]
    bounded = [r for r in rows if r["tail_weeks"]]

    doc = [
        "# MBARI Neural Model Comparison",
        "",
        "Generated from `nn_results/*/summary.json`.",
        "",
        "## Full-History Runs",
        "",
        table(full) if full else "_No full-history summaries found._",
        "",
        "## Bounded-History Runs",
        "",
        "These runs intentionally use `tail_weeks` and must not be compared as full-history equivalents.",
        "",
        table(bounded) if bounded else "_No bounded-history summaries found._",
        "",
        "## Promotion Note",
        "",
        "Candidate promotion should compare models only within the same history class unless a senior manager explicitly accepts the bounded-history tradeoff.",
        "",
    ]
    OUT.write_text("\n".join(doc), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
