#!/usr/bin/env python3
"""Check tracked reports/docs against current MBAL gate truth."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOCS = ["PRODUCTION_READINESS.md", "MBAL_PRODUCTION_LAKEHOUSE_CONTRACTS.md", "README.md"]
FORBIDDEN = [
    r"\bis (?:a )?global SOTA\b",
    r"\bestablished global SOTA\b",
    r"\bwave drivers (?:are )?validated\b",
    r"\bwave drivers help\b",
    r"\bdrivers improve globally\b",
    r"\bfoundation models are production-promoted\b",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _scan_forbidden(project_root: Path) -> list[dict[str, Any]]:
    hits = []
    patterns = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN]
    for rel in DOCS:
        path = project_root / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            lowered = line.lower()
            if "no " in lowered or "not " in lowered or "not approved" in lowered:
                continue
            for pattern in patterns:
                if pattern.search(line):
                    hits.append({"path": rel, "line": lineno, "pattern": pattern.pattern, "text": line.strip()})
    return hits


def _missing_doc_counts(project_root: Path, truth: dict[str, Any]) -> list[str]:
    readiness = project_root / "PRODUCTION_READINESS.md"
    if not readiness.exists():
        return ["PRODUCTION_READINESS.md missing"]
    text = readiness.read_text(encoding="utf-8", errors="replace")
    normalized_text = text.replace(",", "")
    required = [
        str(truth.get("promoted_row_count")),
        str(truth.get("unique_promoted_target_horizon_count")),
    ]
    status_counts = truth.get("status_counts", {})
    required.extend(str(v) for v in status_counts.values())
    missing = [value for value in required if value and value not in normalized_text]
    return missing


def build_report(project_root: Path) -> dict[str, Any]:
    promotion = _read_json(project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json")
    release = _read_json(project_root / "release_gate" / "reports" / "release_gate_report.json")
    sr = _read_json(project_root / "release_gate" / "reports" / "sr_manager_gate_report.json")
    champion = _read_json(project_root / "reports" / "champion_selector" / "champion_selector_summary.json")
    failures = []
    warnings = []
    forbidden_hits = _scan_forbidden(project_root)
    if forbidden_hits:
        failures.append(f"{len(forbidden_hits)} forbidden overclaim phrases found")
    missing_counts = _missing_doc_counts(project_root, promotion)
    if missing_counts:
        failures.append(f"PRODUCTION_READINESS.md missing current promotion counts: {missing_counts}")
    if release.get("overall_status") != "PASS":
        failures.append(f"release gate is not PASS: {release.get('overall_status')}")
    if sr.get("overall_status") != "PASS":
        failures.append(f"SR manager gate is not PASS: {sr.get('overall_status')}")
    if not champion:
        warnings.append("champion selector summary missing")
    overall = "FAIL" if failures else "WARN" if warnings else "PASS"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": overall,
        "failures": failures,
        "warnings": warnings,
        "truth": {
            "promotion": promotion,
            "release_status": release.get("overall_status"),
            "sr_status": sr.get("overall_status"),
            "champion": champion,
        },
        "forbidden_hits": forbidden_hits,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report_consistency_report.json"
    md_path = output_dir / "REPORT_CONSISTENCY_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        "# Monterey Bay AI Lab Report Consistency Agent",
        "",
        f"- Overall status: **{report['overall_status']}**",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {item}" for item in report["failures"]) if report["failures"] else lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"]) if report["warnings"] else lines.append("None.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = build_report(root)
    paths = write_outputs(report, args.output_dir or root / "reports" / "report_consistency")
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, indent=2, sort_keys=True))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

