#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.report_consistency_agent import build_report  # noqa: E402


def test_report_consistency_fails_stale_counts_and_overclaim(tmp_path):
    promo_dir = tmp_path / "lakehouse" / "gold" / "promotion_matrix"
    promo_dir.mkdir(parents=True)
    (promo_dir / "promotion_summary.json").write_text(
        json.dumps(
            {
                "promoted_row_count": 57,
                "unique_promoted_target_horizon_count": 14,
                "status_counts": {"promote": 57, "candidate_split_mismatch": 33},
            }
        ),
        encoding="utf-8",
    )
    reports = tmp_path / "release_gate" / "reports"
    reports.mkdir(parents=True)
    (reports / "release_gate_report.json").write_text(json.dumps({"overall_status": "PASS"}), encoding="utf-8")
    (reports / "sr_manager_gate_report.json").write_text(json.dumps({"overall_status": "PASS"}), encoding="utf-8")
    (tmp_path / "PRODUCTION_READINESS.md").write_text("Promotion matrix: 12 promote. wave drivers help.", encoding="utf-8")
    (tmp_path / "MBARI_PRODUCTION_LAKEHOUSE_CONTRACTS.md").write_text("contracts", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    report = build_report(tmp_path)

    assert report["overall_status"] == "FAIL"
    assert report["forbidden_hits"]
    assert any("missing current promotion counts" in failure for failure in report["failures"])
