#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


SECTION_ORDER = (
    "Executive Summary",
    "Best Completed Neural Run",
    "Driver Lessons",
    "Promotion Matrix",
    "Release Gate Status",
    "Next Experiments",
)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def fmt_num(value: Any, digits: int = 4) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "n/a"
    text = f"{parsed:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_release_gate(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_foundation_summary(project_root: Path) -> Path | None:
    candidates = [
        project_root / "foundation_summary.md",
        project_root / "nn_results" / "foundation_summary.md",
    ]
    return next((path for path in candidates if path.exists()), None)


def extract_foundation_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    useful: list[str] = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("At the 6h sweet spot"):
            useful.append(clean)
        elif clean.startswith("=>"):
            useful.append(clean.lstrip("=> "))
    return useful[:2]


def best_completed_run(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = []
    for row in rows:
        mean_skill = parse_float(row.get("mean_skill"))
        if mean_skill is None:
            continue
        scored.append((mean_skill, str(row.get("outdir", "")), row))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][2]


def summarize_driver_delta(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "present": False,
            "rows": 0,
            "rmse_improved": 0,
            "skill_improved": 0,
            "models": [],
        }

    rmse_improved = 0
    skill_improved = 0
    by_model: dict[str, dict[str, int]] = {}
    for row in rows:
        model = str(row.get("model_name") or row.get("model_name_drv") or "unknown")
        bucket = by_model.setdefault(model, {"rows": 0, "rmse_improved": 0, "skill_improved": 0})
        bucket["rows"] += 1

        rmse_delta = parse_float(row.get("rmse_delta_drv_minus_base"))
        if rmse_delta is not None and rmse_delta < 0:
            rmse_improved += 1
            bucket["rmse_improved"] += 1

        skill_delta = parse_float(row.get("skill_delta_drv_minus_base"))
        if skill_delta is not None and skill_delta > 0:
            skill_improved += 1
            bucket["skill_improved"] += 1

    models = [
        {"model": model, **counts}
        for model, counts in sorted(by_model.items(), key=lambda item: item[0])
    ]
    return {
        "present": True,
        "rows": len(rows),
        "rmse_improved": rmse_improved,
        "skill_improved": skill_improved,
        "models": models,
    }


def release_gate_neural_check(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    for check in report.get("checks", []):
        if check.get("name") == "neural_lakehouse_outputs":
            return check
    return None


def release_gate_promotion_check(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    for check in report.get("checks", []):
        if check.get("name") == "promotion_matrix":
            return check
    return None


def read_promotion_summary(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def line_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_report(
    phase2_rows: list[dict[str, str]],
    driver_rows: list[dict[str, str]],
    release_report: dict[str, Any] | None,
    foundation_lines: list[str],
    promotion_summary: dict[str, Any] | None = None,
) -> str:
    best = best_completed_run(phase2_rows)
    driver = summarize_driver_delta(driver_rows)
    gate_status = str(release_report.get("overall_status", "UNKNOWN")) if release_report else "MISSING"
    neural_check = release_gate_neural_check(release_report)
    promotion_check = release_gate_promotion_check(release_report)

    positive_runs = sum(
        1 for row in phase2_rows if (parse_float(row.get("mean_skill")) or 0.0) > 0
    )
    driver_runs = sum(1 for row in phase2_rows if parse_bool(row.get("drivers_enabled")))

    executive_items = [
        f"Phase 2 completed neural runs summarized: {len(phase2_rows)}.",
        f"Positive mean-skill neural runs: {positive_runs}/{len(phase2_rows)}.",
        f"Driver-enabled neural runs summarized: {driver_runs}.",
        f"Release gate status: {gate_status}.",
    ]
    if best:
        executive_items.insert(
            1,
            (
                "Best completed neural run is "
                f"{best.get('outdir', 'unknown')} "
                f"({best.get('model_name', 'unknown')}, mean skill {fmt_num(best.get('mean_skill'))}%)."
            ),
        )
    if foundation_lines:
        executive_items.append(re.sub(r"\s+", " ", foundation_lines[-1]))
    if promotion_summary:
        counts = promotion_summary.get("status_counts", {})
        executive_items.append(
            "Promotion matrix: "
            f"{counts.get('promote', 0)} promote, "
            f"{counts.get('candidate', 0) + counts.get('candidate_split_mismatch', 0)} candidate, "
            f"{counts.get('reject', 0)} reject."
        )

    if best:
        best_items = [
            f"Outdir: {best.get('outdir', 'unknown')}.",
            f"Model: {best.get('model_name', 'unknown')}.",
            f"Loss: {best.get('loss', 'unknown')}.",
            f"Drivers enabled: {str(parse_bool(best.get('drivers_enabled'))).lower()}.",
            f"Rows: {best.get('rows', 'n/a')}.",
            f"Mean skill vs persistence: {fmt_num(best.get('mean_skill'))}%.",
            f"Median skill vs persistence: {fmt_num(best.get('median_skill'))}%.",
            f"Mean RMSE: {fmt_num(best.get('mean_rmse'))}.",
        ]
    else:
        best_items = ["No completed neural run summary rows were found."]

    if driver["present"]:
        driver_items = [
            f"Driver delta rows: {driver['rows']}.",
            f"Drivers improved RMSE in {driver['rmse_improved']}/{driver['rows']} cells.",
            f"Drivers improved skill in {driver['skill_improved']}/{driver['rows']} cells.",
        ]
        for model in driver["models"]:
            driver_items.append(
                f"{model['model']}: RMSE improved {model['rmse_improved']}/{model['rows']}; "
                f"skill improved {model['skill_improved']}/{model['rows']}."
            )
    else:
        driver_items = ["No driver delta artifact was found."]

    if promotion_summary:
        counts = promotion_summary.get("status_counts", {})
        promotion_items = [
            f"Rows: {promotion_summary.get('rows', 'n/a')}.",
            f"Promote: {counts.get('promote', 0)}.",
            f"Candidate: {counts.get('candidate', 0)}.",
            f"Candidate split mismatch: {counts.get('candidate_split_mismatch', 0)}.",
            f"Reject: {counts.get('reject', 0)}.",
            f"Insufficient data: {counts.get('insufficient_data', 0)}.",
        ]
        best_candidates = promotion_summary.get("best_candidates") or []
        if best_candidates:
            top = best_candidates[0]
            promotion_items.append(
                "Top review candidate: "
                f"{top.get('target', 'unknown')} +{top.get('horizon_h', '?')}h "
                f"{top.get('candidate_model', 'unknown')} "
                f"delta vs XGBoost {fmt_num(top.get('xgb_delta_skill_pct'))} pts "
                f"({top.get('status', 'unknown')})."
            )
    else:
        promotion_items = ["Promotion matrix artifact was not found."]

    gate_items = [f"Overall status: {gate_status}."]
    if release_report:
        gate_items.append(f"Schema version: {release_report.get('schema_version', 'n/a')}.")
        if release_report.get("generated_at_utc"):
            gate_items.append(f"Gate generated at UTC: {release_report['generated_at_utc']}.")
    if neural_check:
        gate_items.append(
            f"Neural/lakehouse check: {neural_check.get('status', 'UNKNOWN')} - "
            f"{neural_check.get('summary', 'no summary')}."
        )
        details = neural_check.get("details", {})
        if "lakehouse_positive_skill_cells" in details and "lakehouse_total_skill_cells" in details:
            gate_items.append(
                "Lakehouse positive skill cells: "
                f"{details['lakehouse_positive_skill_cells']}/{details['lakehouse_total_skill_cells']}."
            )
    elif release_report:
        gate_items.append("Neural/lakehouse check was not present in the release gate report.")
    if promotion_check:
        gate_items.append(
            f"Promotion matrix check: {promotion_check.get('status', 'UNKNOWN')} - "
            f"{promotion_check.get('summary', 'no summary')}."
        )

    next_items = [
        "Promote only target/horizon cells that beat persistence and the production XGBoost baseline on the same split.",
        "Unify XGBoost and neural split artifacts before making airtight model comparisons.",
        "Keep driver-enabled experiments bounded unless a run explicitly documents why full history is required.",
    ]
    if foundation_lines:
        next_items.append("Integrate foundation-model benchmark results into the automated promotion matrix.")
    else:
        next_items.append("Run or attach foundation_summary.md so foundation models are represented in daily learnings.")

    sections = {
        "Executive Summary": line_list(executive_items),
        "Best Completed Neural Run": line_list(best_items),
        "Driver Lessons": line_list(driver_items),
        "Promotion Matrix": line_list(promotion_items),
        "Release Gate Status": line_list(gate_items),
        "Next Experiments": line_list(next_items),
    }

    output = ["# Monterey Bay AI Lab Daily Learnings"]
    for section in SECTION_ORDER:
        output.extend(["", f"## {section}", "", sections[section]])
    return "\n".join(output).rstrip() + "\n"


def generate_daily_learnings(project_root: Path, output: Path) -> str:
    project_root = project_root.resolve()
    phase2_rows = read_csv_rows(project_root / "nn_results" / "phase2_run_summary.csv")
    driver_rows = read_csv_rows(project_root / "nn_results" / "phase2_driver_delta.csv")
    release_report = read_release_gate(project_root / "release_gate" / "reports" / "release_gate_report.json")
    foundation_lines = extract_foundation_lines(find_foundation_summary(project_root))
    promotion_summary = read_promotion_summary(project_root)

    report = render_report(phase2_rows, driver_rows, release_report, foundation_lines, promotion_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Monterey Bay AI Lab daily learnings report.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to <project-root>/archive/reports/MBAL_AI_DAILY_LEARNINGS.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or (args.project_root / "archive" / "reports" / "MBAL_AI_DAILY_LEARNINGS.md")
    generate_daily_learnings(args.project_root, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
