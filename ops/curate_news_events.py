#!/usr/bin/env python3
"""Curate bronze news documents into silver event mentions/events.

This first-pass classifier is deliberately inspectable: it uses event taxonomy
terms, negative context terms, Monterey geofence terms, source trust, and short
article context. The output schema is compatible with a future LLM classifier
that can replace the scoring internals without changing downstream features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = DEFAULT_PROJECT_ROOT / "research" / "news_events_taxonomy.json"
DEFAULT_SOURCES = DEFAULT_PROJECT_ROOT / "research" / "news_events_sources.json"
DEFAULT_OUT_DIR = DEFAULT_PROJECT_ROOT / "lakehouse" / "silver" / "news_events"
M1_LAT = 36.7511
M1_LON = -122.0292

PLACE_COORDS = {
    "monterey bay": (36.8, -121.95),
    "monterey": (36.6002, -121.8947),
    "pacific grove": (36.6177, -121.9166),
    "lovers point": (36.6263, -121.9167),
    "carmel": (36.5552, -121.9233),
    "carmel beach": (36.5533, -121.9308),
    "moss landing": (36.8044, -121.7869),
    "elkhorn slough": (36.8177, -121.7380),
    "santa cruz": (36.9741, -122.0308),
    "marina": (36.6844, -121.8022),
    "seaside": (36.6111, -121.8516),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(*parts: str, length: int = 24) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def term_hits(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    hits = []
    for term in terms:
        clean = str(term).lower().strip()
        if clean and clean in lower:
            hits.append(str(term))
    return hits


def parse_gdelt_datetime(value: str, fallback: str | None = None) -> pd.Timestamp:
    text = str(value or "")
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed) and re.match(r"^\d{8}T\d{6}Z$", text):
        parsed = pd.to_datetime(text, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    if pd.isna(parsed) and fallback:
        parsed = pd.to_datetime(fallback, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.now(tz="UTC").floor("s")
    return pd.Timestamp(parsed).floor("s")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def best_place(text: str, geo_terms: list[str]) -> tuple[str, float, float, float]:
    lower = text.lower()
    for place, coords in PLACE_COORDS.items():
        if place in lower:
            lat, lon = coords
            return place, lat, lon, haversine_km(lat, lon, M1_LAT, M1_LON)
    for term in geo_terms:
        clean = str(term).lower()
        if clean in lower:
            lat, lon = PLACE_COORDS.get(clean, PLACE_COORDS["monterey bay"])
            return clean, lat, lon, haversine_km(lat, lon, M1_LAT, M1_LON)
    lat, lon = PLACE_COORDS["monterey bay"]
    return "Monterey Bay", lat, lon, haversine_km(lat, lon, M1_LAT, M1_LON)


def score_doc(
    doc: dict[str, Any],
    event_class: dict[str, Any],
    geo_terms: list[str],
    source_trust: int,
) -> dict[str, Any]:
    title = str(doc.get("title") or "")
    snippet = str(doc.get("snippet") or "")
    domain = str(doc.get("domain") or "")
    text = f"{title}\n{snippet}\n{domain}"
    positive = term_hits(text, list(event_class.get("positive_terms", [])))
    negative = term_hits(text, list(event_class.get("negative_terms", [])))
    geo = term_hits(text, geo_terms)
    marine = term_hits(
        text,
        ["beach", "bay", "ocean", "harbor", "water", "coast", "marine", "shellfish", "whale", "shark", "sewage"],
    )
    title_hits = term_hits(title, list(event_class.get("positive_terms", [])))
    event_term_score = min(1.0, len(positive) / 2.0)
    geo_score = 1.0 if geo else 0.0
    source_score = {1: 1.0, 2: 0.75, 3: 0.45}.get(int(source_trust or 3), 0.35)
    marine_score = 1.0 if marine else 0.0
    title_score = 1.0 if title_hits else 0.0
    confidence = (
        0.30 * geo_score
        + 0.30 * event_term_score
        + 0.15 * source_score
        + 0.15 * marine_score
        + 0.10 * title_score
    )
    if negative:
        confidence *= 0.45
    return {
        "confidence_score": round(float(confidence), 4),
        "matched_positive_terms": positive,
        "matched_negative_terms": negative,
        "matched_geo_terms": geo,
        "matched_marine_terms": marine,
        "evidence_text": " ".join(text.split())[:1000],
    }


def curate(
    docs: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    sources: dict[str, Any],
    min_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo_terms = list(taxonomy.get("geofence", {}).get("primary_places", []))
    event_by_type = {str(item["event_type"]): item for item in taxonomy.get("event_classes", [])}
    source_by_id = {str(item["source_id"]): item for item in sources.get("sources", [])}
    mentions: list[dict[str, Any]] = []
    for doc in docs:
        event_type = str(doc.get("event_type_query") or "")
        event_class = event_by_type.get(event_type)
        if not event_class:
            continue
        source = source_by_id.get(str(doc.get("source_id")), {})
        trust = int(source.get("trust_tier", 3))
        score = score_doc(doc, event_class, geo_terms, source_trust=trust)
        if score["confidence_score"] < min_confidence:
            continue
        available_at = parse_gdelt_datetime(str(doc.get("published_at") or ""), fallback=str(doc.get("retrieved_at") or ""))
        location_name, lat, lon, dist = best_place(score["evidence_text"], geo_terms)
        mention_id = stable_id(str(doc.get("doc_id", "")), event_type, str(available_at))
        mentions.append({
            "mention_id": mention_id,
            "doc_id": doc.get("doc_id"),
            "source_id": doc.get("source_id"),
            "source_type": doc.get("source_type"),
            "event_type": event_type,
            "event_subtype": "",
            "event_time_start": available_at,
            "event_time_end": pd.NaT,
            "published_at": available_at,
            "available_at_utc": available_at,
            "location_name": location_name,
            "lat": lat,
            "lon": lon,
            "distance_to_m1_km": round(float(dist), 3),
            "monterey_relevance": 1.0 if score["matched_geo_terms"] else 0.5,
            "source_trust": trust,
            "classifier_confidence": score["confidence_score"],
            "severity_score": max(0.25, score["confidence_score"]),
            "confidence_score": score["confidence_score"],
            "relevance_score": score["confidence_score"],
            "evidence_text": score["evidence_text"],
            "evidence_url": doc.get("url"),
            "title": doc.get("title"),
            "domain": doc.get("domain"),
            "matched_positive_terms": score["matched_positive_terms"],
            "matched_negative_terms": score["matched_negative_terms"],
            "matched_geo_terms": score["matched_geo_terms"],
            "classifier_version": "context_rules_2026_06_08_v1",
        })
    mentions_df = pd.DataFrame(mentions)
    if mentions_df.empty:
        return mentions_df, pd.DataFrame()
    mentions_df = mentions_df.drop_duplicates(subset=["event_type", "evidence_url"], keep="first").reset_index(drop=True)
    events_rows: list[dict[str, Any]] = []
    for _, row in mentions_df.iterrows():
        event_id = stable_id(str(row["event_type"]), str(row["location_name"]), str(pd.Timestamp(row["available_at_utc"]).date()), str(row["evidence_url"]))
        events_rows.append({
            "event_id": event_id,
            "event_type": row["event_type"],
            "event_subtype": row["event_subtype"],
            "event_time_start": row["event_time_start"],
            "event_time_end": row["event_time_end"],
            "first_published_at": row["published_at"],
            "first_available_at": row["available_at_utc"],
            "available_at_utc": row["available_at_utc"],
            "best_location_name": row["location_name"],
            "lat": row["lat"],
            "lon": row["lon"],
            "distance_to_m1_km": row["distance_to_m1_km"],
            "severity_score": row["severity_score"],
            "confidence_score": row["confidence_score"],
            "relevance_score": row["relevance_score"],
            "supporting_mention_count": 1,
            "source_count": 1,
            "event_key_hash": event_id,
            "evidence_url": row["evidence_url"],
            "title": row["title"],
        })
    events_df = pd.DataFrame(events_rows).drop_duplicates("event_id").reset_index(drop=True)
    return mentions_df, events_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--bronze-jsonl", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--sources", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    taxonomy = load_json(args.taxonomy or root / "research" / "news_events_taxonomy.json")
    sources = load_json(args.sources or root / "research" / "news_events_sources.json")
    docs = read_jsonl(args.bronze_jsonl)
    mentions, events = curate(docs, taxonomy, sources, min_confidence=args.min_confidence)
    out_dir = args.output_dir or root / "lakehouse" / "silver" / "news_events"
    out_dir.mkdir(parents=True, exist_ok=True)
    mentions_path = out_dir / "event_mentions.parquet"
    events_path = out_dir / "events.parquet"
    mentions.to_parquet(mentions_path, index=False)
    events.to_parquet(events_path, index=False)
    summary = {
        "bronze_jsonl": str(args.bronze_jsonl),
        "document_rows": len(docs),
        "mention_rows": int(len(mentions)),
        "event_rows": int(len(events)),
        "event_type_counts": events["event_type"].value_counts().to_dict() if not events.empty else {},
        "mentions": str(mentions_path),
        "events": str(events_path),
    }
    (out_dir / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
