#!/usr/bin/env python3
"""Machine-checkable senior-manager gate for Monterey Bay AI Lab promotion artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class SrManagerGateConfig:
    project_root: Path

    @property
    def release_report_path(self) -> Path:
        return self.project_root / "release_gate" / "reports" / "release_gate_report.json"

    @property
    def promotion_summary_path(self) -> Path:
        return self.project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json"

    @property
    def daily_learnings_path(self) -> Path:
        return self.project_root / "archive" / "reports" / "MBAL_AI_DAILY_LEARNINGS.md"


def _check(status: str, name: str, summary: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "details": details or {},
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(data, dict):
        return None, "json_root_not_object"
    return data, None


def _promotion_counts(summary: dict[str, Any]) -> dict[str, int]:
    raw_counts = summary.get("status_counts", {})
    if not isinstance(raw_counts, dict):
        return {}

    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def evaluate_sr_manager_gate(config: SrManagerGateConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    release_report, release_error = _read_json(config.release_report_path)
    if release_error:
        checks.append(
            _check(
                "FAIL",
                "release_gate_report",
                "release gate report JSON is required",
                {"path": str(config.release_report_path), "error": release_error},
            )
        )
    else:
        release_status = str(release_report.get("overall_status", "WARN")).upper()
        status = "FAIL" if release_status == "FAIL" else ("WARN" if release_status == "WARN" else "PASS")
        checks.append(
            _check(
                status,
                "release_gate_report",
                f"release gate overall status is {release_status}",
                {"path": str(config.release_report_path), "overall_status": release_status},
            )
        )

    promotion_summary, promotion_error = _read_json(config.promotion_summary_path)
    if promotion_error:
        checks.append(
            _check(
                "FAIL",
                "promotion_summary",
                "promotion summary JSON is required",
                {"path": str(config.promotion_summary_path), "error": promotion_error},
            )
        )
    else:
        counts = _promotion_counts(promotion_summary)
        promote_count = int(promotion_summary.get("promoted_row_count", counts.get("promote", 0)) or 0)
        unique_target_horizon_count = int(
            promotion_summary.get("unique_promoted_target_horizon_count", promote_count) or 0
        )
        unique_model_cell_count = int(
            promotion_summary.get("unique_promoted_model_cell_count", promote_count) or 0
        )
        status = "PASS" if promote_count > 0 else "WARN"
        summary = (
            "promotion matrix has "
            f"{promote_count} promoted rows across "
            f"{unique_target_horizon_count} target/horizon cells and "
            f"{unique_model_cell_count} model cells"
            if promote_count > 0
            else "promotion matrix exists but has no enforceable promotions"
        )
        checks.append(
            _check(
                status,
                "promotion_summary",
                summary,
                {
                    "path": str(config.promotion_summary_path),
                    "rows": promotion_summary.get("rows"),
                    "status_counts": counts,
                    "promoted_row_count": promote_count,
                    "unique_promoted_target_horizon_count": unique_target_horizon_count,
                    "unique_promoted_model_cell_count": unique_model_cell_count,
                },
            )
        )

    if config.daily_learnings_path.exists():
        checks.append(
            _check(
                "PASS",
                "daily_learnings",
                "daily learnings report exists",
                {"path": str(config.daily_learnings_path)},
            )
        )
    else:
        checks.append(
            _check(
                "WARN",
                "daily_learnings",
                "daily learnings report is missing",
                {"path": str(config.daily_learnings_path)},
            )
        )

    overall_status = max((check["status"] for check in checks), key=lambda status: STATUS_ORDER[status])
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(config.project_root),
        "overall_status": overall_status,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MBAL senior-manager promotion gate.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root containing release gate, promotion, and daily learning artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path to write the evaluated gate JSON report",
    )
    parser.add_argument(
        "--no-exit-code",
        action="store_true",
        help="always exit 0 after printing JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_sr_manager_gate(SrManagerGateConfig(project_root=args.project_root.resolve()))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.no_exit_code:
        return 0
    return 1 if result["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

