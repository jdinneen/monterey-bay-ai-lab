#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))

from agent_brain_core import (  # noqa: E402
    GateInput,
    append_jsonl,
    iter_jsonl,
    query_brain,
    reflection_payload,
    run_value_gate,
)


def test_value_gate_rejects_missing_evidence():
    result = run_value_gate(
        GateInput(
            idea="Add a general framework",
            agent="test",
            task="bloat-check",
            reversible=True,
            valuable_now=True,
            genuinely_new=True,
        )
    )

    assert result.verdict == "REJECT"
    assert "baseline_or_stakeholder_question" in result.missing


def test_value_gate_accepts_evidenced_reversible_tested_work():
    result = run_value_gate(
        GateInput(
            idea="Add agent preflight",
            agent="test",
            task="agent-brain",
            stakeholder_question="User asked for all agents to push back on bloat",
            test="tests/test_agent_brain.py",
            reversible=True,
            valuable_now=True,
            genuinely_new=True,
        )
    )

    assert result.verdict == "DO_NOW"
    assert result.missing == []
    assert "tests/test_agent_brain.py" in result.summary


def test_brain_query_finds_rejected_idea(tmp_path):
    brain = tmp_path / "brain"
    brain.mkdir()
    for name in ["project_state.json", "invariants.json", "agent_roles.json"]:
        (brain / name).write_text("{}", encoding="utf-8")
    for name in ["learnings.jsonl", "decision_log.jsonl"]:
        (brain / name).write_text("", encoding="utf-8")
    append_jsonl(
        brain / "rejected_ideas.jsonl",
        {"idea": "Restore Databricks files", "category": "forbidden"},
    )

    hits = query_brain("Databricks", brain_dir=brain)

    assert hits
    assert "Databricks" in hits[0]["text"]


def test_project_brain_finds_science_mission_lessons():
    queries = {
        "Monterey Bay Moat": "Monterey Bay Moat",
        "memory_reserved": "memory_allocated",
        "32 expert MoE": "8 experts",
        "normalization mismatch": "MAPE",
        "NOAA MUR SST quarter": "chunk satellite fetches by quarter",
    }

    for query, expected in queries.items():
        hits = query_brain(query)
        joined = "\n".join(str(hit["text"]) for hit in hits)
        assert expected in joined


def test_reflection_payload_round_trips_jsonl(tmp_path):
    path = tmp_path / "learnings.jsonl"
    payload = reflection_payload(
        agent="test",
        task="brain",
        lesson="Keep memories small",
        evidence="unit test",
        reuse_when="agent work",
        date="2026-06-09",
    )
    append_jsonl(path, payload)

    rows = iter_jsonl(path)

    assert rows == [payload]


def test_bloat_gate_cli_rejects_bad_proposal(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"idea": "Add unlabeled SOTA phase", "agent": "test", "task": "x"}), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "ops" / "bloat_gate.py"), "--proposal-json", str(proposal)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "REJECT" in proc.stdout


def test_run_safe_preflight_rejects_before_traffic_admission(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps({"idea": "Add vague framework", "agent": "test", "task": "unsafe"}),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "run_safe.py"),
            "--task",
            "agent-brain-preflight-test",
            "--agent",
            "test",
            "--preflight",
            str(proposal),
            "--",
            sys.executable,
            "-c",
            "print('should not run')",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "Agent Brain preflight rejected" in proc.stderr
    assert "should not run" not in proc.stdout


def test_run_safe_preflight_passes_then_runs_command(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "idea": "Run a tiny verified safe command",
                "agent": "test",
                "task": "run-safe-preflight",
                "stakeholder_question": "Test optional preflight integration",
                "test": "tests/test_agent_brain.py",
                "reversible": True,
                "valuable_now": True,
                "genuinely_new": True,
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "run_safe.py"),
            "--task",
            "agent-brain-preflight-test-pass",
            "--agent",
            "test",
            "--preflight",
            str(proposal),
            "--",
            sys.executable,
            "-c",
            "print('preflight command ran')",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Value gate: DO_NOW" in proc.stdout
    assert "preflight command ran" in proc.stdout


def test_run_safe_warn_mode_rejects_but_runs_command():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "run_safe.py"),
            "--task",
            "agent-brain-preflight-warn",
            "--agent",
            "test",
            "--brain-preflight",
            "warn",
            "--preflight-idea",
            "Add vague untested framework",
            "--",
            sys.executable,
            "-c",
            "print('warn mode ran')",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "rejected:" in proc.stdout
    assert "mode=warn" in proc.stderr
    assert "warn mode ran" in proc.stdout


def test_run_safe_enforce_mode_requires_preflight_idea():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops" / "run_safe.py"),
            "--task",
            "agent-brain-preflight-missing",
            "--agent",
            "test",
            "--brain-preflight",
            "enforce",
            "--",
            sys.executable,
            "-c",
            "print('should not run')",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "no preflight idea" in proc.stderr
    assert "should not run" not in proc.stdout
