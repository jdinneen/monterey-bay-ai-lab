#!/usr/bin/env python3
"""Fetch known news/agency URLs into the bronze news document schema.

Use this when search has identified high-value incident URLs and we need real
page context without relying on a rate-limited search API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta_description = ""
        self.meta_published = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True
        if tag.lower() == "meta":
            data = {k.lower(): v or "" for k, v in attrs}
            if data.get("name", "").lower() == "description" or data.get("property", "").lower() == "og:description":
                self.meta_description = data.get("content", "")
            if data.get("property", "").lower() == "og:title" and data.get("content"):
                self.title_parts = [data["content"]]
            if data.get("property", "").lower() in {"article:published_time", "og:published_time"}:
                self.meta_published = data.get("content", "")
            if data.get("name", "").lower() in {"date", "pubdate", "publishdate", "publish-date"}:
                self.meta_published = data.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        clean = " ".join(html.unescape(data).split())
        if not clean:
            return
        if self.in_title:
            self.title_parts.append(clean)
        elif len(clean) > 20:
            self.text_parts.append(clean)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(*parts: str, length: int = 24) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:length]


def parse_visible_date(text: str) -> str | None:
    month = (
        r"January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )
    match = re.search(rf"\b({month})\.?\s+([0-3]?\d),\s+(20\d{{2}})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    mon_raw, day, year = match.groups()
    mon = mon_raw[:3].title()
    if mon == "Sep":
        mon = "Sep"
    try:
        parsed = dt.datetime.strptime(f"{mon} {int(day)} {year}", "%b %d %Y")
        return parsed.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def fetch_page(url: str, timeout_seconds: int) -> tuple[str, dict[str, str]]:
    req = Request(url, headers={"User-Agent": "mbari-ai-forecasting-news-events/0.1"})
    with urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in response.headers.items()}
    return body, headers


def extract_page(
    url: str,
    html_text: str,
    event_type: str,
    fetched_at: str,
    published_override: str | None = None,
) -> dict[str, Any]:
    parser = PageTextParser()
    parser.feed(html_text)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    snippet = parser.meta_description or text[:700]
    domain = urlparse(url).netloc.lower()
    doc_id = stable_id(url, title)
    published_at = published_override or parser.meta_published or parse_visible_date(f"{title} {snippet} {text[:3000]}") or fetched_at
    return {
        "doc_id": doc_id,
        "query_id": "seed_url",
        "source_id": f"seed_{domain}",
        "source_type": "seed_url",
        "event_type_query": event_type,
        "query_text": f"seed url {event_type}",
        "url": url,
        "title": title or url,
        "snippet": snippet[:1200],
        "body_text": text[:12000],
        "domain": domain,
        "language": "",
        "published_at": published_at,
        "retrieved_at": fetched_at,
    }


def parse_seed(value: str) -> tuple[str, str, str | None]:
    if "=" not in value:
        raise ValueError("--seed expects event_type=url or event_type@published_utc=url")
    event_spec, url = value.split("=", 1)
    published = None
    if "@" in event_spec:
        event_type, published = event_spec.split("@", 1)
    else:
        event_type = event_spec
    return event_type.strip(), url.strip(), published.strip() if published else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "reports" / "news_events_fetch" / "seed_urls")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--seed", action="append", required=True, help="event_type=url")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bronze_dir = args.output_dir / "bronze" / "seed_url"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    docs_path = bronze_dir / "documents.jsonl"
    manifest: dict[str, Any] = {"generated_at_utc": utc_now(), "rows": [], "document_count": 0}
    with docs_path.open("w", encoding="utf-8") as fh:
        for seed in args.seed:
            event_type, url, published_override = parse_seed(seed)
            fetched_at = utc_now()
            try:
                body, _headers = fetch_page(url, args.timeout_seconds)
                row = extract_page(url, body, event_type, fetched_at, published_override=published_override)
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                manifest["rows"].append({
                    "url": url,
                    "event_type": event_type,
                    "published_at": row["published_at"],
                    "status": "ok",
                    "doc_id": row["doc_id"],
                })
                manifest["document_count"] += 1
            except Exception as exc:
                manifest["rows"].append({"url": url, "event_type": event_type, "status": "error", "error": str(exc)})
    manifest["raw_documents_jsonl"] = str(docs_path)
    manifest_path = args.output_dir / "seed_fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"documents": manifest["document_count"], "manifest": str(manifest_path), "raw_documents_jsonl": str(docs_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
