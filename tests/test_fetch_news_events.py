#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import fetch_news_events as fetcher  # noqa: E402


def sample_taxonomy() -> dict:
    return {
        "taxonomy_version": "test-taxonomy",
        "geofence": {"primary_places": ["Monterey Bay", "Moss Landing"]},
        "event_classes": [
            {
                "event_type": "spill_release",
                "positive_terms": ["oil spill", "sewage spill"],
                "negative_terms": ["spill the beans"],
            },
            {
                "event_type": "hab_biotoxin",
                "positive_terms": ["domoic acid", "harmful algal bloom"],
                "negative_terms": ["Lake Erie"],
            },
        ],
    }


def sample_sources() -> dict:
    return {
        "registry_version": "test-sources",
        "sources": [
            {
                "source_id": "gdelt_doc_gkg_event",
                "source_type": "news_search",
                "event_types": ["spill_release", "hab_biotoxin"],
            },
            {
                "source_id": "caloes_spill_release_reporting",
                "source_type": "state",
                "event_types": ["spill_release"],
            },
        ],
    }


def test_build_boolean_query_includes_geo_event_and_negative_terms():
    query = fetcher.build_boolean_query(
        ["Monterey Bay", "Moss Landing"],
        ["oil spill", "hazmat"],
        ["spill the beans"],
    )

    assert '("Monterey Bay" OR "Moss Landing")' in query
    assert '("oil spill" OR hazmat)' in query
    assert '-"spill the beans"' in query


def test_plan_queries_builds_stable_gdelt_url_and_authoritative_review_rows():
    planned = fetcher.plan_queries(
        sample_taxonomy(),
        sample_sources(),
        "2026-06-01T00:00:00Z",
        "2026-06-08T00:00:00Z",
        max_records=25,
    )

    assert len(planned) == 3
    planned_again = fetcher.plan_queries(
        sample_taxonomy(),
        sample_sources(),
        "2026-06-01T00:00:00Z",
        "2026-06-08T00:00:00Z",
        max_records=25,
    )
    assert [q.query_id for q in planned] == [q.query_id for q in planned_again]

    gdelt_spill = next(q for q in planned if q.source_id == "gdelt_doc_gkg_event" and q.event_type == "spill_release")
    assert gdelt_spill.request_url is not None
    assert "api.gdeltproject.org/api/v2/doc/doc" in gdelt_spill.request_url
    assert "startdatetime=20260601000000" in gdelt_spill.request_url
    assert "enddatetime=20260608000000" in gdelt_spill.request_url
    assert "maxrecords=25" in gdelt_spill.request_url

    caloes = next(q for q in planned if q.source_id == "caloes_spill_release_reporting")
    assert caloes.request_url is None


def test_write_outputs_creates_manifest_query_log_and_markdown(tmp_path):
    planned = fetcher.plan_queries(
        sample_taxonomy(),
        sample_sources(),
        "2026-06-01T00:00:00Z",
        "2026-06-08T00:00:00Z",
        max_records=25,
    )

    paths = fetcher.write_outputs(
        planned,
        taxonomy=sample_taxonomy(),
        sources=sample_sources(),
        output_dir=tmp_path,
        dry_run=True,
    )

    for path in paths.values():
        assert Path(path).exists()

    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    queries = json.loads(Path(paths["queries_json"]).read_text(encoding="utf-8"))
    log_text = Path(paths["log"]).read_text(encoding="utf-8")
    summary = Path(paths["summary_markdown"]).read_text(encoding="utf-8")

    assert manifest["dry_run"] is True
    assert manifest["planned_query_count"] == 3
    assert manifest["event_type_counts"]["spill_release"] == 2
    assert len(queries) == 3
    assert "planned_query_count=3" in log_text
    assert "This dry run did not call external services" in summary

