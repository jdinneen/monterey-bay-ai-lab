#!/usr/bin/env python3
"""Small wrapper that rejects proposals that fail the Agent Brain value gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_brain_core import GateInput, parse_bool, run_value_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proposal = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    gate = GateInput(
        idea=str(proposal.get("idea", "")),
        agent=str(proposal.get("agent", "")),
        task=str(proposal.get("task", "")),
        baseline=str(proposal.get("baseline", "")),
        test=str(proposal.get("test", "")),
        reversible=parse_bool(proposal.get("reversible", False)),
        valuable_now=parse_bool(proposal.get("valuable_now", False)),
        genuinely_new=parse_bool(proposal.get("genuinely_new", False)),
        stakeholder_question=str(proposal.get("stakeholder_question", "")),
        correctness_fix=parse_bool(proposal.get("correctness_fix", False)),
        kill_category=str(proposal.get("kill_category", "bloat")),
    )
    result = run_value_gate(gate)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.verdict == "DO_NOW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
