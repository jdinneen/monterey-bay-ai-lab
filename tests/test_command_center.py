#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.command_center import _gate_from_payload, command_center_state, stack_coverage  # noqa: E402


def test_command_center_state_has_core_sections():
    state = command_center_state()

    assert state["project"] == "Monterey Bay AI Lab"
    assert "invariants" in state
    assert "learnings" in state
    assert "gpu" in state
    assert "locks" in state
    assert "next_safe_action" in state
    assert "operating_loop" in state
    assert "stack_coverage" in state
    assert "in_flight" in state
    assert "by_layer" in state["in_flight"]
    assert any(section["title"] == "Daily Loop" for section in state["operating_loop"])
    assert state["next_safe_action"]["command"]
    assert "preflight" in state["commands"]


def test_gate_payload_matches_preflight_contract():
    gate = _gate_from_payload(
        {
            "idea": "Run quarterly NOAA MUR fetch",
            "agent": "codex",
            "task": "fetch",
            "stakeholder_question": "Avoid HTTP 408 annual fetch failure",
            "test": "tests/test_agent_brain.py",
            "reversible": True,
            "valuable_now": True,
            "genuinely_new": True,
        }
    )

    assert gate.idea == "Run quarterly NOAA MUR fetch"
    assert gate.agent == "codex"
    assert gate.reversible is True


def test_static_assets_exist_and_are_not_empty():
    static = ROOT / "ops" / "stack_viz" / "static"
    for name in ["index.html", "styles.css", "app.js"]:
        path = static / name
        assert path.exists()
        assert path.stat().st_size > 500


def test_state_is_json_serializable():
    json.dumps(command_center_state())


def test_stack_coverage_has_required_layers():
    coverage = stack_coverage()

    ids = {row["id"] for row in coverage["rows"]}
    assert {"sources", "bronze", "silver", "training", "evaluation", "gold", "claims"}.issubset(ids)
    assert {"agents", "brain", "traffic", "sentinel", "telemetry"}.issubset(ids)
    assert coverage["required_count"] >= 12


def test_in_flight_payload_groups_items():
    state = command_center_state()

    assert isinstance(state["in_flight"]["items"], list)
    assert isinstance(state["in_flight"]["by_layer"], dict)
