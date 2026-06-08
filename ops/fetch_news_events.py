#!/usr/bin/env python3
"""Dry-run-first news/event fetch planner for the MBARI lakehouse.

The default mode does not touch the network. It reads the versioned news event
taxonomy and source registry, builds auditable search plans, and writes proof
artifacts that a later fetch implementation can execute.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import time
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = DEFAULT_PROJECT_ROOT / "research" / "news_events_taxonomy.json"
DEFAULT_SOURCES = DEFAULT_PROJECT_ROOT / "research" / "news_events_sources.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "reports" / "news_events_fetch"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class PlannedQuery:
    query_id: str
    source_id: str
    source_type: str
    event_type: str
    start_utc: str
    end_utc: str
    query_text: str
    request_url: str | None
    dry_run: bool = True


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def stable_id(*parts: str, length: int = 16) -> str:
    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def quote_terms(terms: list[str]) -> list[str]:
    quoted: list[str] = []
    for term in terms:
        clean = term.strip()
        if not clean:
            continue
        if " " in clean or "-" in clean:
            quoted.append(f'"{clean}"')
        else:
            quoted.append(clean)
    return quoted


def build_boolean_query(geo_terms: list[str], event_terms: list[str], negative_terms: list[str]) -> str:
    geo = " OR ".join(quote_terms(geo_terms))
    events = " OR ".join(quote_terms(event_terms))
    negatives = " ".join(f"-{term}" for term in quote_terms(negative_terms))
    base = f"({geo}) ({events})"
    return f"{base} {negatives}".strip()


def gdelt_datetime(value: str) -> str:
    parsed = parse_utc(value)
    return parsed.strftime("%Y%m%d%H%M%S")


def build_gdelt_url(query_text: str, start_utc: str, end_utc: str, max_records: int) -> str:
    return (
        f"{GDELT_DOC_API}?query={quote_plus(query_text)}"
        f"&mode=artlist&format=json&maxrecords={int(max_records)}"
        f"&startdatetime={gdelt_datetime(start_utc)}&enddatetime={gdelt_datetime(end_utc)}"
    )


def source_supports_event(source: dict[str, Any], event_type: str) -> bool:
    return event_type in set(source.get("event_types", []))


def plan_queries(
    taxonomy: dict[str, Any],
    sources: dict[str, Any],
    start_utc: str,
    end_utc: str,
    *,
    max_records: int,
) -> list[PlannedQuery]:
    geo_terms = list(taxonomy.get("geofence", {}).get("primary_places", []))
    event_classes = list(taxonomy.get("event_classes", []))
    registry = list(sources.get("sources", []))
    planned: list[PlannedQuery] = []
    for event_class in event_classes:
        event_type = str(event_class["event_type"])
        query_text = build_boolean_query(
            geo_terms,
            list(event_class.get("positive_terms", [])),
            list(event_class.get("negative_terms", [])),
        )
        for source in registry:
            if not source_supports_event(source, event_type):
                continue
            source_id = str(source["source_id"])
            source_type = str(source.get("source_type", "unknown"))
            request_url = None
            if source_type == "news_search" and source_id == "gdelt_doc_gkg_event":
                request_url = build_gdelt_url(query_text, start_utc, end_utc, max_records)
            query_id = stable_id(source_id, event_type, start_utc, end_utc, query_text)
            planned.append(
                PlannedQuery(
                    query_id=query_id,
                    source_id=source_id,
                    source_type=source_type,
                    event_type=event_type,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    query_text=query_text,
                    request_url=request_url,
                )
            )
    return planned


def default_window(days: int) -> tuple[str, str]:
    end = utc_now()
    start = end - dt.timedelta(days=int(days))
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def write_outputs(
    planned: list[PlannedQuery],
    *,
    taxonomy: dict[str, Any],
    sources: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(query) for query in planned]
    event_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for query in planned:
        event_counts[query.event_type] = event_counts.get(query.event_type, 0) + 1
        source_counts[query.source_id] = source_counts.get(query.source_id, 0) + 1

    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now().isoformat().replace("+00:00", "Z"),
        "dry_run": dry_run,
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "source_registry_version": sources.get("registry_version"),
        "planned_query_count": len(planned),
        "event_type_counts": event_counts,
        "source_counts": source_counts,
        "outputs": {
            "queries_json": "planned_queries.json",
            "summary_markdown": "DRY_RUN_SUMMARY.md",
            "log": "dry_run.log",
        },
    }
    manifest_path = output_dir / "manifest.json"
    queries_path = output_dir / "planned_queries.json"
    log_path = output_dir / "dry_run.log"
    md_path = output_dir / "DRY_RUN_SUMMARY.md"

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    queries_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log_lines = [
        f"generated_at_utc={manifest['generated_at_utc']}",
        f"dry_run={dry_run}",
        f"taxonomy_version={manifest['taxonomy_version']}",
        f"source_registry_version={manifest['source_registry_version']}",
        f"planned_query_count={len(planned)}",
    ]
    for query in planned:
        target = query.request_url or "source_registry_review"
        log_lines.append(
            f"query_id={query.query_id} source_id={query.source_id} "
            f"event_type={query.event_type} target={target}"
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# News/Event Fetch Dry Run",
        "",
        f"- Dry run: `{dry_run}`",
        f"- Planned queries: `{len(planned)}`",
        f"- Taxonomy version: `{manifest['taxonomy_version']}`",
        f"- Source registry version: `{manifest['source_registry_version']}`",
        "",
        "## Event Types",
        "",
        "| event type | planned queries |",
        "|---|---:|",
    ]
    for event_type, count in sorted(event_counts.items()):
        md_lines.append(f"| `{event_type}` | {count} |")
    md_lines.extend(["", "## Sources", "", "| source | planned queries |", "|---|---:|"])
    for source_id, count in sorted(source_counts.items()):
        md_lines.append(f"| `{source_id}` | {count} |")
    md_lines.extend(
        [
            "",
            "## Proof",
            "",
            "This dry run did not call external services. It proves the current taxonomy and source registry can generate auditable query plans with stable IDs, time windows, and target URLs where applicable.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "manifest": str(manifest_path),
        "queries_json": str(queries_path),
        "summary_markdown": str(md_path),
        "log": str(log_path),
    }


def fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "mbari-ai-forecasting-news-events/0.1"})
    with urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def normalize_gdelt_article(article: dict[str, Any], query: PlannedQuery, fetched_at_utc: str) -> dict[str, Any]:
    url = str(article.get("url") or "")
    title = str(article.get("title") or "")
    seendate = str(article.get("seendate") or article.get("date") or "")
    doc_id = stable_id(url, title, seendate, length=24)
    return {
        "doc_id": doc_id,
        "query_id": query.query_id,
        "source_id": query.source_id,
        "source_type": query.source_type,
        "event_type_query": query.event_type,
        "query_text": query.query_text,
        "url": url,
        "title": title,
        "snippet": str(article.get("snippet") or ""),
        "domain": str(article.get("domain") or ""),
        "language": str(article.get("language") or ""),
        "published_at": seendate,
        "retrieved_at": fetched_at_utc,
        "raw": article,
    }


def execute_gdelt_queries(
    planned: list[PlannedQuery],
    *,
    output_dir: Path,
    timeout_seconds: int,
    delay_seconds: float,
) -> dict[str, Any]:
    bronze_dir = output_dir / "bronze" / "gdelt_doc"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    raw_path = bronze_dir / "documents.jsonl"
    response_path = bronze_dir / "responses.jsonl"
    fetched_docs = 0
    query_statuses: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    with raw_path.open("w", encoding="utf-8") as docs_fh, response_path.open("w", encoding="utf-8") as resp_fh:
        for query in planned:
            if not query.request_url:
                query_statuses.append({
                    "query_id": query.query_id,
                    "source_id": query.source_id,
                    "event_type": query.event_type,
                    "status": "source_review_only",
                    "article_count": 0,
                })
                continue
            fetched_at = utc_now().isoformat().replace("+00:00", "Z")
            try:
                payload = fetch_json(query.request_url, timeout_seconds)
                articles = list(payload.get("articles", []))
                resp_fh.write(json.dumps({
                    "query": asdict(query),
                    "fetched_at_utc": fetched_at,
                    "article_count": len(articles),
                    "payload_keys": sorted(payload.keys()),
                }, sort_keys=True) + "\n")
                new_count = 0
                for article in articles:
                    row = normalize_gdelt_article(article, query, fetched_at)
                    if row["doc_id"] in seen_doc_ids:
                        continue
                    seen_doc_ids.add(row["doc_id"])
                    docs_fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                    new_count += 1
                fetched_docs += new_count
                query_statuses.append({
                    "query_id": query.query_id,
                    "source_id": query.source_id,
                    "event_type": query.event_type,
                    "status": "ok",
                    "article_count": len(articles),
                    "new_document_count": new_count,
                })
            except Exception as exc:
                query_statuses.append({
                    "query_id": query.query_id,
                    "source_id": query.source_id,
                    "event_type": query.event_type,
                    "status": "error",
                    "error": str(exc),
                    "article_count": 0,
                })
            time.sleep(max(0.0, float(delay_seconds)))
    return {
        "raw_documents_jsonl": str(raw_path),
        "raw_responses_jsonl": str(response_path),
        "fetched_document_count": fetched_docs,
        "query_statuses": query_statuses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--days", type=int, default=14, help="rolling lookback window when start/end are omitted")
    parser.add_argument("--start-utc", default=None, help="inclusive ISO UTC start, e.g. 2026-06-01T00:00:00Z")
    parser.add_argument("--end-utc", default=None, help="exclusive ISO UTC end, e.g. 2026-06-08T00:00:00Z")
    parser.add_argument("--max-records", type=int, default=75, help="planned max records per news-search query")
    parser.add_argument("--execute", action="store_true", help="execute supported API-backed news-search queries")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    taxonomy_path = args.taxonomy or project_root / "research" / "news_events_taxonomy.json"
    sources_path = args.sources or project_root / "research" / "news_events_sources.json"
    if args.start_utc and args.end_utc:
        start_utc = parse_utc(args.start_utc).isoformat().replace("+00:00", "Z")
        end_utc = parse_utc(args.end_utc).isoformat().replace("+00:00", "Z")
    else:
        start_utc, end_utc = default_window(args.days)

    taxonomy = load_json(taxonomy_path)
    sources = load_json(sources_path)
    planned = plan_queries(taxonomy, sources, start_utc, end_utc, max_records=args.max_records)

    run_id = f"{'fetch' if args.execute else 'dry_run'}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = args.output_dir or project_root / "reports" / "news_events_fetch" / run_id
    paths = write_outputs(planned, taxonomy=taxonomy, sources=sources, output_dir=output_dir, dry_run=not args.execute)
    fetch_result: dict[str, Any] | None = None
    if args.execute:
        fetch_result = execute_gdelt_queries(
            planned,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
            delay_seconds=args.delay_seconds,
        )
        fetch_manifest_path = output_dir / "fetch_manifest.json"
        fetch_manifest_path.write_text(json.dumps(fetch_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["fetch_manifest"] = str(fetch_manifest_path)
        paths["raw_documents_jsonl"] = fetch_result["raw_documents_jsonl"]
    print(json.dumps({
        "dry_run": not args.execute,
        "planned_query_count": len(planned),
        "fetched_document_count": fetch_result["fetched_document_count"] if fetch_result else 0,
        "paths": paths,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
