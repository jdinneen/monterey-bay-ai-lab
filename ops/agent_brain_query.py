#!/usr/bin/env python3
"""Search the local Agent Brain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_brain_core import BRAIN_DIR, query_brain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hits = query_brain(args.query, brain_dir=args.brain_dir, limit=args.limit)
    if args.json:
        print(json.dumps(hits, indent=2, sort_keys=True))
    else:
        for hit in hits:
            print(f"{hit['path']}:{hit['line']} score={hit['score']} {hit['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
