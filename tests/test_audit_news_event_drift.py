#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import audit_news_event_drift as audit  # noqa: E402


def taxonomy() -> dict:
    return {
        "taxonomy_version": "test",
        "geofence": {"primary_places": ["Monterey Bay", "Pacific Grove"]},
        "event_classes": [
            {
                "event_type": "spill_release",
                "positive_terms": ["sewage spill"],
                "negative_terms": ["spill the beans"],
            }
        ],
    }


def evidence_record(**updates) -> dict:
    row = {
        "event_type": "spill_release",
        "title": "Pacific Grove sewage spill reaches Monterey Bay",
        "canonical_url": "https://example.com/story",
        "published_at_utc": "2022-01-01T00:00:00Z",
        "event_time_utc": "2022-01-01T00:00:00Z",
        "available_at_utc": "2022-01-01T01:00:00Z",
        "place_name": "Pacific Grove",
        "evidence_text": "A sewage spill affected Monterey Bay water quality.",
        "agent_context_assessment": "Strong local match.",
    }
    row.update(updates)
    return row


def run_audit(record: dict) -> tuple[list[audit.Finding], dict]:
    return audit.audit(
        {
            "evidence_set_id": "test",
            "window_start_utc": "2021-06-08T00:00:00Z",
            "window_end_utc": "2026-06-08T00:00:00Z",
            "records": [record],
        },
        taxonomy(),
    )


def test_drift_audit_passes_valid_record():
    findings, summary = run_audit(evidence_record())

    assert findings == []
    assert summary["overall_status"] == "PASS"


def test_drift_audit_fails_outside_window():
    findings, summary = run_audit(evidence_record(event_time_utc="2020-01-01T00:00:00Z"))

    assert summary["overall_status"] == "FAIL"
    assert any(f.check == "window" for f in findings)


def test_drift_audit_fails_negative_and_missing_geofence():
    findings, summary = run_audit(
        evidence_record(
            title="Spill the beans at Los Angeles beach",
            place_name="Los Angeles",
            evidence_text="This says spill the beans and is not local.",
        )
    )

    assert summary["overall_status"] == "FAIL"
    assert any(f.check == "negative_terms" for f in findings)
    assert any(f.check == "geofence" for f in findings)

