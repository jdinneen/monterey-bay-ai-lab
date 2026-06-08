#!/usr/bin/env python3
"""Audit news/event evidence for taxonomy, time-window, and geography drift."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = DEFAULT_PROJECT_ROOT / "research" / "news_events_last5_evidence.json"
DEFAULT_TAXONOMY = DEFAULT_PROJECT_ROOT / "research" / "news_events_taxonomy.json"
DEFAULT_REPORT_DIR = DEFAULT_PROJECT_ROOT / "reports" / "news_events_drift"


@dataclass
class Finding:
    status: str
    check: str
    record_index: int | None
    message: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalized(value: str) -> str:
    return " ".join(str(value).lower().split())


def matched_terms(text: str, terms: list[str]) -> list[str]:
    haystack = normalized(text)
    return [term for term in terms if normalized(term) in haystack]


def taxonomy_maps(taxonomy: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    classes = {str(item["event_type"]): item for item in taxonomy.get("event_classes", [])}
    places = [str(place) for place in taxonomy.get("geofence", {}).get("primary_places", [])]
    places.extend(["Asilomar", "Lovers Point", "Garrapata", "McAbee", "San Carlos", "Monterey County"])
    return classes, places


def audit(evidence: dict[str, Any], taxonomy: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    event_classes, place_terms = taxonomy_maps(taxonomy)
    start = parse_utc(str(evidence.get("window_start_utc", "2021-06-08T00:00:00Z")))
    end = parse_utc(str(evidence.get("window_end_utc", "2026-06-08T23:59:59Z")))
    findings: list[Finding] = []
    records = list(evidence.get("records", []))

    seen_keys: set[tuple[str, str, str]] = set()
    for idx, record in enumerate(records):
        event_type = str(record.get("event_type", ""))
        event_class = event_classes.get(event_type)
        if event_class is None:
            findings.append(Finding("FAIL", "taxonomy", idx, f"unknown event_type={event_type!r}"))
            continue

        try:
            event_time = parse_utc(str(record["event_time_utc"]))
            available_at = parse_utc(str(record["available_at_utc"]))
            published_at = parse_utc(str(record["published_at_utc"]))
        except Exception as exc:
            findings.append(Finding("FAIL", "timestamp_parse", idx, str(exc)))
            continue

        if not (start <= event_time <= end):
            findings.append(
                Finding("FAIL", "window", idx, f"event_time_utc {event_time.isoformat()} outside {start.isoformat()}..{end.isoformat()}")
            )
        if available_at < event_time:
            findings.append(Finding("FAIL", "availability", idx, "available_at_utc precedes event_time_utc"))
        if available_at < published_at:
            findings.append(Finding("FAIL", "availability", idx, "available_at_utc precedes published_at_utc"))

        evidence_blob = " ".join(
            str(record.get(key, ""))
            for key in ["title", "place_name", "event_subtype", "evidence_text", "agent_context_assessment"]
        )
        positives = matched_terms(evidence_blob, list(event_class.get("positive_terms", [])))
        negatives = matched_terms(evidence_blob, list(event_class.get("negative_terms", [])))
        places = matched_terms(evidence_blob, place_terms)
        if not positives:
            findings.append(Finding("FAIL", "taxonomy_terms", idx, f"no positive taxonomy terms for {event_type}"))
        if negatives:
            findings.append(Finding("FAIL", "negative_terms", idx, f"negative taxonomy terms matched: {negatives}"))
        if not places:
            findings.append(Finding("FAIL", "geofence", idx, "no Monterey-area geofence/place term matched"))

        key = (event_type, str(record.get("canonical_url", "")), str(event_time.floor("D").date()))
        if key in seen_keys:
            findings.append(Finding("WARN", "duplicate", idx, f"duplicate event/url/day key: {key}"))
        seen_keys.add(key)

    fail_count = sum(1 for finding in findings if finding.status == "FAIL")
    warn_count = sum(1 for finding in findings if finding.status == "WARN")
    summary = {
        "schema_version": 1,
        "evidence_set_id": evidence.get("evidence_set_id"),
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "window_start_utc": str(start),
        "window_end_utc": str(end),
        "record_count": len(records),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "overall_status": "FAIL" if fail_count else "WARN" if warn_count else "PASS",
        "event_type_counts": pd.Series([r.get("event_type") for r in records]).value_counts().sort_index().to_dict() if records else {},
    }
    return findings, summary


def write_report(findings: list[Finding], summary: dict[str, Any], report_dir: Path) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "drift_audit.json"
    md_path = report_dir / "DRIFT_AUDIT.md"
    payload = {**summary, "findings": [asdict(finding) for finding in findings]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    lines = [
        "# News/Event Drift Audit",
        "",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Evidence set: `{summary['evidence_set_id']}`",
        f"- Records: `{summary['record_count']}`",
        f"- Failures: `{summary['fail_count']}`",
        f"- Warnings: `{summary['warn_count']}`",
        "",
        "## Event Counts",
        "",
        "| event type | records |",
        "|---|---:|",
    ]
    for event_type, count in sorted(summary["event_type_counts"].items()):
        lines.append(f"| `{event_type}` | {int(count)} |")
    if findings:
        lines.extend(["", "## Findings", "", "| status | check | record | message |", "|---|---|---:|---|"])
        for finding in findings:
            record = "" if finding.record_index is None else str(finding.record_index)
            lines.append(f"| `{finding.status}` | `{finding.check}` | {record} | {finding.message} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=None,
        help="evidence JSON path; may be repeated to audit a combined dataset",
    )
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    evidence_paths = args.evidence or [root / "research" / "news_events_last5_evidence.json"]
    evidence_docs = [load_json(path) for path in evidence_paths]
    if len(evidence_docs) == 1:
        evidence = evidence_docs[0]
    else:
        starts = [parse_utc(str(doc.get("window_start_utc"))) for doc in evidence_docs if doc.get("window_start_utc")]
        ends = [parse_utc(str(doc.get("window_end_utc"))) for doc in evidence_docs if doc.get("window_end_utc")]
        evidence = {
            "schema_version": 1,
            "evidence_set_id": "+".join(str(doc.get("evidence_set_id", path.stem)) for doc, path in zip(evidence_docs, evidence_paths)),
            "window_start_utc": min(starts).isoformat() if starts else None,
            "window_end_utc": max(ends).isoformat() if ends else None,
            "records": [record for doc in evidence_docs for record in doc.get("records", [])],
        }
    taxonomy = load_json(args.taxonomy or root / "research" / "news_events_taxonomy.json")
    findings, summary = audit(evidence, taxonomy)
    paths = write_report(findings, summary, args.report_dir or root / "reports" / "news_events_drift" / "last5")
    print(json.dumps({**summary, "paths": paths}, indent=2, sort_keys=True, default=str))
    return 1 if summary["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
