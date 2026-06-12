#!/usr/bin/env python3
"""Shared utilities for the local Agent Brain.

The brain is a deterministic project-memory layer. It does not call an LLM and it
does not enforce deployment hooks by itself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = ROOT / "docs" / "agent_brain"


@dataclass(frozen=True)
class GateInput:
    idea: str
    agent: str
    task: str
    baseline: str = ""
    test: str = ""
    reversible: bool = False
    valuable_now: bool = False
    genuinely_new: bool = False
    stakeholder_question: str = ""
    correctness_fix: bool = False
    kill_category: str = "bloat"


@dataclass(frozen=True)
class GateResult:
    verdict: str
    category: str
    summary: str
    missing: list[str]
    payload: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "do_now"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def brain_snapshot(brain_dir: Path = BRAIN_DIR) -> dict[str, Any]:
    return {
        "project_state": read_json(brain_dir / "project_state.json"),
        "invariants": read_json(brain_dir / "invariants.json"),
        "agent_roles": read_json(brain_dir / "agent_roles.json"),
        "learnings": iter_jsonl(brain_dir / "learnings.jsonl"),
        "rejected_ideas": iter_jsonl(brain_dir / "rejected_ideas.jsonl"),
        "decisions": iter_jsonl(brain_dir / "decision_log.jsonl"),
    }


def query_brain(query: str, brain_dir: Path = BRAIN_DIR, limit: int = 10) -> list[dict[str, Any]]:
    terms = [term.casefold() for term in query.split() if term.strip()]
    if not terms:
        return []

    hits: list[dict[str, Any]] = []
    files = [
        brain_dir / "project_state.json",
        brain_dir / "science_mission.md",
        brain_dir / "invariants.json",
        brain_dir / "agent_roles.json",
        brain_dir / "learnings.jsonl",
        brain_dir / "rejected_ideas.jsonl",
        brain_dir / "decision_log.jsonl",
    ]
    for path in files:
        if not path.exists():
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            folded = line.casefold()
            score = sum(1 for term in terms if term in folded)
            if score:
                hits.append({"path": str(path), "line": idx, "score": score, "text": line.strip()})
    return sorted(hits, key=lambda item: (-int(item["score"]), item["path"], int(item["line"])))[:limit]


def run_value_gate(gate: GateInput) -> GateResult:
    missing: list[str] = []
    if not gate.idea.strip():
        missing.append("idea")
    if not gate.agent.strip():
        missing.append("agent")
    if not gate.task.strip():
        missing.append("task")

    answers_real_question = bool(gate.baseline.strip() or gate.stakeholder_question.strip())
    if not answers_real_question:
        missing.append("baseline_or_stakeholder_question")
    if not (gate.reversible or gate.correctness_fix):
        missing.append("reversible_or_correctness_fix")
    if not gate.test.strip():
        missing.append("test")
    if not gate.valuable_now:
        missing.append("valuable_now")
    if not gate.genuinely_new:
        missing.append("genuinely_new")

    payload = asdict(gate)
    if missing:
        category = gate.kill_category or "bloat"
        return GateResult(
            verdict="REJECT",
            category=category,
            summary=f"rejected: {category} - missing {', '.join(missing)}",
            missing=missing,
            payload=payload,
        )

    reason = gate.baseline.strip() or gate.stakeholder_question.strip()
    return GateResult(
        verdict="DO_NOW",
        category="value_gate_pass",
        summary=f"Value gate: DO_NOW because {reason}; test {gate.test}",
        missing=[],
        payload=payload,
    )


def reflection_payload(
    *,
    agent: str,
    task: str,
    lesson: str,
    evidence: str,
    reuse_when: str,
    date: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "date": date or utc_now(),
        "agent": agent,
        "task": task,
        "lesson": lesson,
        "evidence": evidence,
        "reuse_when": reuse_when,
    }


def rejection_payload(
    *,
    idea: str,
    verdict: str,
    category: str,
    reason: str,
    source: str,
    date: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "date": date or utc_now(),
        "idea": idea,
        "verdict": verdict,
        "category": category,
        "reason": reason,
        "source": source,
    }
