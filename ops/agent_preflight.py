#!/usr/bin/env python3
"""Run the Agent Brain value preflight for proposed work."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from agent_brain_core import GateInput, parse_bool, query_brain, run_value_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--stakeholder-question", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--reversible", default="false")
    parser.add_argument("--valuable-now", default="false")
    parser.add_argument("--genuinely-new", default="false")
    parser.add_argument("--correctness-fix", default="false")
    parser.add_argument("--kill-category", default="bloat")
    parser.add_argument("--brain-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = GateInput(
        idea=args.idea,
        agent=args.agent,
        task=args.task,
        baseline=args.baseline,
        stakeholder_question=args.stakeholder_question,
        test=args.test,
        reversible=parse_bool(args.reversible),
        valuable_now=parse_bool(args.valuable_now),
        genuinely_new=parse_bool(args.genuinely_new),
        correctness_fix=parse_bool(args.correctness_fix),
        kill_category=args.kill_category,
    )
    result = run_value_gate(gate)
    memory_hits = query_brain(args.idea, brain_dir=args.brain_dir) if args.brain_dir else query_brain(args.idea)
    output = {**asdict(result), "memory_hits": memory_hits}
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(result.summary)
        if memory_hits:
            print("Relevant brain hits:")
            for hit in memory_hits[:5]:
                print(f"- {hit['path']}:{hit['line']} {hit['text'][:180]}")
    return 0 if result.verdict == "DO_NOW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
