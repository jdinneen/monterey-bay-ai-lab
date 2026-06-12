#!/usr/bin/env python3
"""Record Agent Brain reflections or rejected ideas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_brain_core import BRAIN_DIR, append_jsonl, reflection_payload, rejection_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    sub = parser.add_subparsers(dest="kind", required=True)

    learn = sub.add_parser("learning", help="append a completed-task lesson")
    learn.add_argument("--agent", required=True)
    learn.add_argument("--task", required=True)
    learn.add_argument("--lesson", required=True)
    learn.add_argument("--evidence", required=True)
    learn.add_argument("--reuse-when", required=True)

    reject = sub.add_parser("rejection", help="append a rejected idea")
    reject.add_argument("--idea", required=True)
    reject.add_argument("--category", required=True)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--source", default="docs/VALUE_GATE.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.kind == "learning":
        payload = reflection_payload(
            agent=args.agent,
            task=args.task,
            lesson=args.lesson,
            evidence=args.evidence,
            reuse_when=args.reuse_when,
        )
        path = args.brain_dir / "learnings.jsonl"
    else:
        payload = rejection_payload(
            idea=args.idea,
            verdict="REJECT",
            category=args.category,
            reason=args.reason,
            source=args.source,
        )
        path = args.brain_dir / "rejected_ideas.jsonl"
    append_jsonl(path, payload)
    print(json.dumps({"path": str(path), "payload": payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
