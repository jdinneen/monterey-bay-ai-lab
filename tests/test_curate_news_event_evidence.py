#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import curate_news_event_evidence as curate  # noqa: E402


def taxonomy() -> dict:
    return {
        "taxonomy_version": "test",
        "event_classes": [
            {
                "event_type": "spill_release",
                "positive_terms": ["sewage spill", "wastewater spill"],
                "negative_terms": ["spill the beans"],
            },
            {
                "event_type": "beach_closure",
                "positive_terms": ["beach closure", "closed beaches"],
                "negative_terms": ["road closure"],
            },
        ],
    }


def base_record(event_type: str = "spill_release") -> dict:
    return {
        "event_type": event_type,
        "event_subtype": "sewage",
        "title": "Monterey sewage spill closes beaches",
        "source_name": "Local News",
        "source_type": "local_news",
        "publisher_domain": "example.com",
        "canonical_url": "https://example.com/story?utm_source=x&id=1",
        "published_at_utc": "2026-01-02T00:00:00Z",
        "event_time_utc": "2026-01-01T00:00:00Z",
        "available_at_utc": "2026-01-02T01:00:00Z",
        "place_name": "Monterey Bay",
        "lat": 36.75,
        "lon": -122.02,
        "severity_score": 0.8,
        "relevance_score": 0.9,
        "confidence_score": 0.9,
        "query_terms": ["Monterey sewage spill"],
        "evidence_text": "A sewage spill caused closed beaches along Monterey Bay.",
        "agent_context_assessment": "Strong local beach closure and wastewater spill match.",
    }


def test_canonicalize_url_strips_tracking_params():
    url = curate.canonicalize_url("HTTPS://Example.COM/story/?utm_source=x&id=1&fbclid=abc")

    assert url == "https://example.com/story?id=1"


def test_build_tables_keeps_distinct_event_types_for_same_article():
    record_a = base_record("spill_release")
    record_b = base_record("beach_closure")
    evidence = {
        "evidence_set_id": "test",
        "generated_at_utc": "2026-01-03T00:00:00Z",
        "records": [record_a, record_b],
    }

    articles, events, decisions = curate.build_tables(evidence, taxonomy())

    assert len(articles) == 1
    assert len(events) == 2
    assert {row["event_type"] for _, row in events.iterrows()} == {"spill_release", "beach_closure"}
    assert all(decision["accepted"] for decision in decisions)


def test_negative_taxonomy_match_rejects_record():
    record = base_record("spill_release")
    record["evidence_text"] = "This story says spill the beans, not a water pollution event."
    evidence = {
        "evidence_set_id": "test",
        "generated_at_utc": "2026-01-03T00:00:00Z",
        "records": [record],
    }

    articles, events, decisions = curate.build_tables(evidence, taxonomy())

    assert articles.empty
    assert events.empty
    assert decisions[0]["accepted"] is False
    assert decisions[0]["matched_negative_terms"] == ["spill the beans"]


def test_unknown_event_type_raises():
    record = base_record("other")
    evidence = {
        "evidence_set_id": "test",
        "generated_at_utc": "2026-01-03T00:00:00Z",
        "records": [record],
    }

    try:
        curate.build_tables(evidence, taxonomy())
    except ValueError as exc:
        assert "unknown event_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")
