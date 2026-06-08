#!/usr/bin/env python3
"""Curate source-backed news/event evidence into silver lakehouse tables.

This is the deterministic curation sidecar for the MBARI news modality. It
does not search the web itself; it converts source-backed candidate evidence
into auditable article and event tables that downstream feature builders can
consume. LLM or human agents can provide candidate context, but the persisted
outputs are deterministic for a fixed evidence file and taxonomy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = DEFAULT_PROJECT_ROOT / "research" / "news_events_seed_evidence.json"
DEFAULT_TAXONOMY = DEFAULT_PROJECT_ROOT / "research" / "news_events_taxonomy.json"
DEFAULT_ARTICLES = DEFAULT_PROJECT_ROOT / "lakehouse" / "silver" / "news_articles" / "articles.parquet"
DEFAULT_EVENTS = DEFAULT_PROJECT_ROOT / "lakehouse" / "silver" / "news_events" / "events.parquet"
DEFAULT_REPORT_DIR = DEFAULT_PROJECT_ROOT / "reports" / "news_events_curation"

TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}

SOURCE_AUTHORITY = {
    "federal": 1.0,
    "state": 0.95,
    "county": 0.9,
    "research": 0.85,
    "local_news": 0.72,
    "news_search": 0.55,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = re.sub(r"/+$", "", parts.path or "/")
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def stable_id(*parts: str, length: int = 20) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def event_types_from_taxonomy(taxonomy: dict[str, Any]) -> set[str]:
    return {str(item["event_type"]) for item in taxonomy.get("event_classes", [])}


def taxonomy_terms(taxonomy: dict[str, Any], event_type: str) -> tuple[list[str], list[str]]:
    for item in taxonomy.get("event_classes", []):
        if item.get("event_type") == event_type:
            return list(item.get("positive_terms", [])), list(item.get("negative_terms", []))
    return [], []


def matched_terms(text: str, terms: list[str]) -> list[str]:
    haystack = normalized_text(text)
    out: list[str] = []
    for term in terms:
        clean = normalized_text(str(term))
        if clean and clean in haystack:
            out.append(str(term))
    return out


def classify_record(record: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    positive, negative = taxonomy_terms(taxonomy, str(record["event_type"]))
    evidence_blob = " ".join(
        str(record.get(key, ""))
        for key in ["title", "place_name", "event_subtype", "evidence_text", "agent_context_assessment"]
    )
    positives = matched_terms(evidence_blob, positive)
    negatives = matched_terms(evidence_blob, negative)
    source_type = str(record.get("source_type", "unknown"))
    source_authority = SOURCE_AUTHORITY.get(source_type, 0.5)
    confidence = float(record.get("confidence_score", 0.75))
    relevance = float(record.get("relevance_score", 0.75))
    context_score = min(1.0, 0.45 + 0.08 * len(positives) + 0.25 * source_authority)
    if negatives:
        context_score *= 0.6
    curation_score = round(float((confidence + relevance + context_score) / 3.0), 4)
    return {
        "matched_positive_terms": positives,
        "matched_negative_terms": negatives,
        "source_authority_score": round(source_authority, 4),
        "context_score": round(context_score, 4),
        "curation_score": curation_score,
        "accepted": curation_score >= 0.65 and len(negatives) == 0,
    }


def build_tables(evidence: dict[str, Any], taxonomy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    valid_event_types = event_types_from_taxonomy(taxonomy)
    article_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for idx, record in enumerate(evidence.get("records", [])):
        event_type = str(record.get("event_type", ""))
        if event_type not in valid_event_types:
            raise ValueError(f"record {idx} has unknown event_type={event_type!r}")
        url = canonicalize_url(str(record["canonical_url"]))
        article_id = stable_id(url)
        published_at = parse_utc(str(record["published_at_utc"]))
        fetched_at = parse_utc(str(record.get("fetched_at_utc") or evidence.get("generated_at_utc") or utc_now()))
        available_at = parse_utc(str(record["available_at_utc"]))
        event_time = parse_utc(str(record["event_time_utc"]))
        decision = classify_record(record, taxonomy)
        decision.update({"record_index": idx, "title": record.get("title"), "event_type": event_type})
        decisions.append(decision)
        if not decision["accepted"]:
            continue

        content_hash = stable_id(
            normalized_text(str(record.get("title", ""))),
            normalized_text(str(record.get("evidence_text", ""))),
            length=32,
        )
        article_rows.append(
            {
                "article_id": article_id,
                "source_name": record.get("source_name"),
                "source_type": record.get("source_type"),
                "canonical_url": url,
                "url_hash": stable_id(url, length=32),
                "title": record.get("title"),
                "snippet": record.get("evidence_text"),
                "body_text_path": None,
                "published_at_utc": published_at,
                "fetched_at_utc": fetched_at,
                "available_at_utc": available_at,
                "language": "en",
                "publisher_domain": record.get("publisher_domain"),
                "query_id": stable_id(evidence.get("evidence_set_id", "evidence"), event_type, url),
                "raw_ref_path": None,
                "content_hash": content_hash,
            }
        )

        place = str(record.get("place_name", "Monterey Bay"))
        bucket = event_time.floor("D").isoformat()
        cluster_key = "|".join([event_type, str(record.get("event_subtype", "")), bucket, slug(place)])
        event_id = stable_id(cluster_key, article_id)
        event_rows.append(
            {
                "event_id": event_id,
                "article_id": article_id,
                "event_cluster_id": stable_id(cluster_key),
                "event_type": event_type,
                "event_subtype": record.get("event_subtype"),
                "event_time_utc": event_time,
                "event_time_confidence": record.get("event_time_confidence", "inferred_from_publication"),
                "available_at_utc": available_at,
                "lat": float(record.get("lat")) if record.get("lat") is not None else None,
                "lon": float(record.get("lon")) if record.get("lon") is not None else None,
                "place_name": place,
                "geo_confidence": record.get("geo_confidence", "place_match"),
                "distance_km_to_m1": record.get("distance_km_to_m1"),
                "distance_km_to_monterey_bay": record.get("distance_km_to_monterey_bay"),
                "severity_score": float(record.get("severity_score", 1.0)),
                "relevance_score": float(record.get("relevance_score", 1.0)),
                "confidence_score": float(record.get("confidence_score", 1.0)),
                "source_authority_score": decision["source_authority_score"],
                "curation_score": decision["curation_score"],
                "is_authoritative": bool(record.get("is_authoritative", False)),
                "is_duplicate_cluster_representative": True,
                "evidence_json": json.dumps(
                    {
                        "title": record.get("title"),
                        "url": url,
                        "query_terms": record.get("query_terms", []),
                        "evidence_text": record.get("evidence_text"),
                        "agent_context_assessment": record.get("agent_context_assessment"),
                        "matched_positive_terms": decision["matched_positive_terms"],
                        "matched_negative_terms": decision["matched_negative_terms"],
                    },
                    sort_keys=True,
                ),
            }
        )

    articles = pd.DataFrame(article_rows).drop_duplicates("article_id") if article_rows else pd.DataFrame()
    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = events.sort_values(
            ["event_cluster_id", "is_authoritative", "curation_score", "available_at_utc"],
            ascending=[True, False, False, True],
        )
        events["is_duplicate_cluster_representative"] = ~events.duplicated("event_cluster_id")
        events = events.sort_values("available_at_utc").reset_index(drop=True)
    return articles, events, decisions


def write_outputs(
    articles: pd.DataFrame,
    events: pd.DataFrame,
    decisions: list[dict[str, Any]],
    evidence: dict[str, Any],
    taxonomy: dict[str, Any],
    articles_path: Path,
    events_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    articles_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(articles_path, index=False)
    events.to_parquet(events_path, index=False)

    event_counts = events["event_type"].value_counts().sort_index().to_dict() if not events.empty else {}
    accepted = sum(1 for d in decisions if d["accepted"])
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "evidence_set_id": evidence.get("evidence_set_id"),
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "records_in": int(len(evidence.get("records", []))),
        "records_accepted": int(accepted),
        "articles_out": int(len(articles)),
        "events_out": int(len(events)),
        "event_type_counts": {str(k): int(v) for k, v in event_counts.items()},
        "articles_path": str(articles_path),
        "events_path": str(events_path),
        "leakage_rule": "curated events expose features only through available_at_utc in downstream hist-only builders",
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (report_dir / "curation_decisions.json").write_text(json.dumps(decisions, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# News/Event Curation Proof",
        "",
        f"- Evidence set: `{manifest['evidence_set_id']}`",
        f"- Taxonomy version: `{manifest['taxonomy_version']}`",
        f"- Records accepted: `{manifest['records_accepted']}` / `{manifest['records_in']}`",
        f"- Articles written: `{manifest['articles_out']}`",
        f"- Events written: `{manifest['events_out']}`",
        "",
        "## Event Counts",
        "",
        "| event type | events |",
        "|---|---:|",
    ]
    for event_type, count in sorted(manifest["event_type_counts"].items()):
        lines.append(f"| `{event_type}` | {count} |")
    lines.extend(["", "## Accepted Evidence", "", "| date | event type | place | title |", "|---|---|---|---|"])
    if not events.empty:
        event_articles = events.merge(articles[["article_id", "title"]], on="article_id", how="left")
        for _, row in event_articles.sort_values("available_at_utc").iterrows():
            date = pd.Timestamp(row["event_time_utc"]).date().isoformat()
            lines.append(f"| {date} | `{row['event_type']}` | {row['place_name']} | {row['title']} |")
    lines.extend(
        [
            "",
            "## Proof Rules",
            "",
            "- All accepted rows have source URLs and `available_at_utc`.",
            "- Event type must exist in the versioned taxonomy.",
            "- Negative taxonomy matches reject a record.",
            "- Downstream feature visibility is controlled by `available_at_utc`, not event time.",
        ]
    )
    (report_dir / "CURATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=None,
        help="evidence JSON path; may be repeated to curate a combined dataset",
    )
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--articles-out", type=Path, default=None)
    parser.add_argument("--events-out", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    evidence_paths = args.evidence or [root / "research" / "news_events_seed_evidence.json"]
    taxonomy_path = args.taxonomy or root / "research" / "news_events_taxonomy.json"
    articles_path = args.articles_out or root / "lakehouse" / "silver" / "news_articles" / "articles.parquet"
    events_path = args.events_out or root / "lakehouse" / "silver" / "news_events" / "events.parquet"
    report_dir = args.report_dir or root / "reports" / "news_events_curation" / f"curation_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    evidence_docs = [load_json(path) for path in evidence_paths]
    if len(evidence_docs) == 1:
        evidence = evidence_docs[0]
    else:
        evidence = {
            "schema_version": 1,
            "evidence_set_id": "+".join(str(doc.get("evidence_set_id", path.stem)) for doc, path in zip(evidence_docs, evidence_paths)),
            "generated_at_utc": utc_now(),
            "records": [record for doc in evidence_docs for record in doc.get("records", [])],
        }
    taxonomy = load_json(taxonomy_path)
    articles, events, decisions = build_tables(evidence, taxonomy)
    manifest = write_outputs(articles, events, decisions, evidence, taxonomy, articles_path, events_path, report_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
