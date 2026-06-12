#!/usr/bin/env python3
"""Evidence gate for Monterey Bay AI Lab model claims.

This script does not promote models. It audits existing promotion/lakehouse artifacts
and fails only on evidence contract violations: unsupported promoted rows, duplicate
metric identities, or stale/missing summary counts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbal_lakehouse import read_forecast_metrics


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
DOCS_TO_SCAN = [
    "README.md",
    "PRODUCTION_READINESS.md",
    "MBAL_PRODUCTION_LAKEHOUSE_CONTRACTS.md",
    "release_gate/reports/promotion_matrix.md",
    "release_gate/reports/release_gate_report.md",
]
UNSUPPORTED_CLAIM_PATTERNS = [
    r"\bdriver-enabled models are better\b",
    r"\bdrivers improve globally\b",
    r"\bdrivers improve forecasts\b",
    r"\bdriver features improve forecasts\b",
    r"\bglobal driver win\b",
    r"\bdriver model is validated\b",
    r"\bwave drivers improve globally\b",
    r"\bwave features help\b",
    r"\bwave drivers help\b",
    r"\bdriver value is proven\b",
    r"\bwave drivers are validated\b",
]


@dataclass
class Check:
    name: str
    status: str
    summary: str
    details: dict[str, Any]


def combine_status(checks: list[Check]) -> str:
    if not checks:
        return "PASS"
    return max((check.status for check in checks), key=lambda s: STATUS_ORDER[s])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _unique_promoted_counts(matrix: pd.DataFrame) -> dict[str, int]:
    required = {"status", "target", "horizon_h", "candidate_model", "candidate_loss", "candidate_drivers_enabled"}
    if matrix.empty or not required.issubset(matrix.columns):
        return {
            "promoted_row_count": 0,
            "unique_promoted_target_horizon_count": 0,
            "unique_promoted_model_cell_count": 0,
        }
    promoted = matrix[matrix["status"].eq("promote")].copy()
    return {
        "promoted_row_count": int(len(promoted)),
        "unique_promoted_target_horizon_count": int(promoted.drop_duplicates(["target", "horizon_h"]).shape[0]),
        "unique_promoted_model_cell_count": int(
            promoted.drop_duplicates(
                ["target", "horizon_h", "candidate_model", "candidate_loss", "candidate_drivers_enabled"]
            ).shape[0]
        ),
    }


def check_promotion_evidence(matrix: pd.DataFrame, summary: dict[str, Any], min_n: int) -> Check:
    details: dict[str, Any] = {"min_n": int(min_n)}
    failures: list[str] = []
    warnings: list[str] = []
    required = {
        "status",
        "target",
        "horizon_h",
        "candidate_n",
        "candidate_skill_vs_persistence_pct",
        "candidate_skill_vs_best_naive_pct",
        "xgb_delta_skill_pct",
        "shared_split",
    }
    missing = sorted(required - set(matrix.columns))
    if missing:
        return Check("promotion_evidence", "FAIL", f"promotion matrix missing columns: {', '.join(missing)}", {"missing": missing})

    promoted = matrix[matrix["status"].eq("promote")].copy()
    candidates = matrix[matrix["status"].eq("candidate_split_mismatch")].copy()
    details["promoted_rows"] = int(len(promoted))
    details["screening_rows"] = int(len(candidates))
    if promoted.empty:
        warnings.append("no promoted rows are available")

    shared_split = _as_bool(promoted["shared_split"])
    candidate_n = pd.to_numeric(promoted["candidate_n"], errors="coerce")
    persistence_skill = pd.to_numeric(promoted["candidate_skill_vs_persistence_pct"], errors="coerce")
    best_naive_skill = pd.to_numeric(promoted["candidate_skill_vs_best_naive_pct"], errors="coerce")
    horizon = pd.to_numeric(promoted["horizon_h"], errors="coerce")
    xgb_delta = pd.to_numeric(promoted["xgb_delta_skill_pct"], errors="coerce")
    bad_shared = promoted[~shared_split]
    bad_n = promoted[candidate_n.isna() | (candidate_n < int(min_n))]
    bad_persistence = promoted[persistence_skill.isna() | (persistence_skill <= 0)]
    bad_best_naive = promoted[best_naive_skill.isna() | (best_naive_skill < 0.5)]
    # Persistence is a free lunch at diurnal horizons (>=24h), so a diurnal
    # promotion MUST carry seasonal-naive evidence. Missing evidence there is a
    # hard contract violation the matrix should never emit.
    diurnal_unverified = promoted[(horizon >= 24) & best_naive_skill.isna()]
    subdiurnal_unverified = promoted[(horizon < 24) & best_naive_skill.isna()]
    bad_xgb = promoted[xgb_delta.isna() | (xgb_delta < 1.0)]
    details["bad_shared_split_rows"] = int(len(bad_shared))
    details["low_n_rows"] = int(len(bad_n))
    details["not_beating_persistence_rows"] = int(len(bad_persistence))
    details["not_beating_best_naive_rows"] = int(len(bad_best_naive))
    details["diurnal_unverified_best_naive_rows"] = int(len(diurnal_unverified))
    details["subdiurnal_unverified_best_naive_rows"] = int(len(subdiurnal_unverified))
    details["not_beating_xgb_rows"] = int(len(bad_xgb))
    if len(bad_shared):
        failures.append(f"{len(bad_shared)} promoted rows are not on shared split")
    if len(bad_n):
        failures.append(f"{len(bad_n)} promoted rows have candidate_n < {min_n}")
    if len(bad_best_naive):
        failures.append(f"{len(bad_best_naive)} promoted rows do not beat best-naive (persistence/seasonal-naive)")
    if len(diurnal_unverified):
        failures.append(f"{len(diurnal_unverified)} diurnal promoted rows lack seasonal-naive verification")
    if len(bad_persistence):
        failures.append(f"{len(bad_persistence)} promoted rows do not beat persistence")
    if len(bad_xgb):
        failures.append(f"{len(bad_xgb)} promoted rows do not beat XGBoost")
    if len(subdiurnal_unverified):
        warnings.append(f"{len(subdiurnal_unverified)} sub-diurnal promoted rows lack seasonal-naive verification")
    if len(candidates):
        warnings.append(f"{len(candidates)} rows remain screening-only split-mismatch evidence")

    counts = _unique_promoted_counts(matrix)
    details.update(counts)
    mismatched = {
        key: {"summary": summary.get(key), "actual": value}
        for key, value in counts.items()
        if summary.get(key) != value
    }
    details["summary_count_mismatches"] = mismatched
    if mismatched:
        failures.append("promotion summary counts do not match promotion matrix")

    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    summary_text = "; ".join(failures or warnings or ["promotion evidence contracts pass"])
    return Check("promotion_evidence", status, summary_text, details)


def check_metric_duplicates(project_root: Path) -> Check:
    metrics_path = project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet"
    details: dict[str, Any] = {"path": str(metrics_path)}
    if not metrics_path.exists():
        return Check("metric_identity", "WARN", "aggregate forecast metrics table is missing", details)
    metrics = pd.read_parquet(metrics_path)
    deduped_recursive = read_forecast_metrics(project_root, include_partitions=True)
    keys = [c for c in ["run_id", "split_id", "unique_id", "horizon_h", "model", "loss"] if c in metrics.columns]
    details["rows"] = int(len(metrics))
    details["recursive_deduped_rows"] = int(len(deduped_recursive))
    details["identity_keys"] = keys
    if len(keys) < 4:
        return Check("metric_identity", "FAIL", "metrics table lacks enough identity columns to audit duplicates", details)
    duplicate_count = int(metrics.duplicated(keys).sum())
    details["duplicate_metric_identities"] = duplicate_count
    if duplicate_count:
        examples = metrics[metrics.duplicated(keys, keep=False)][keys].head(20).to_dict(orient="records")
        details["duplicate_examples"] = examples
        return Check("metric_identity", "FAIL", f"{duplicate_count} duplicate metric identities found", details)
    return Check("metric_identity", "PASS", "metric identities are unique", details)


def check_claim_language(project_root: Path) -> Check:
    hits: list[dict[str, Any]] = []
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in UNSUPPORTED_CLAIM_PATTERNS]
    for rel in DOCS_TO_SCAN:
        path = project_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in compiled:
                if pattern.search(line):
                    hits.append({"path": rel, "line": lineno, "pattern": pattern.pattern, "text": line.strip()})
    details = {"scanned": DOCS_TO_SCAN, "hits": hits}
    if hits:
        return Check("claim_language", "FAIL", f"{len(hits)} unsupported global driver/wave claims found", details)
    return Check("claim_language", "PASS", "no forbidden global driver/wave claim language found", details)


def build_report(project_root: Path, min_n: int) -> dict[str, Any]:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    summary_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json"
    checks: list[Check] = []
    if not matrix_path.exists() or not summary_path.exists():
        missing = [str(p) for p in [matrix_path, summary_path] if not p.exists()]
        checks.append(Check("promotion_artifacts", "FAIL", "promotion artifacts are missing", {"missing": missing}))
    else:
        matrix = pd.read_parquet(matrix_path)
        summary = _read_json(summary_path)
        checks.append(check_promotion_evidence(matrix, summary, min_n=min_n))
    checks.append(check_metric_duplicates(project_root))
    checks.append(check_claim_language(project_root))
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "overall_status": combine_status(checks),
        "checks": [asdict(check) for check in checks],
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evidence_gate_report.json"
    md_path = output_dir / "EVIDENCE_GATE_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        "# Monterey Bay AI Lab Evidence Gate",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Generated: `{report['generated_at_utc']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.extend(
            [
                f"### {check['status']} - {check['name']}",
                "",
                check["summary"],
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-n", type=int, default=20)
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = build_report(root, min_n=args.min_n)
    paths = write_outputs(report, args.output_dir or root / "reports" / "evidence_gate")
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, indent=2, sort_keys=True))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

