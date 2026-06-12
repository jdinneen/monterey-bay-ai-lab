"""Tests for the model registry + suite (research/model_lab/model_suite.py).

The registry is the project's fail-closed source of truth for model claims, so these tests guard
the invariants that keep claims honest: valid metadata, evidence-backed claimability, and the rule
that failed/negative/baseline statuses can never be marked claimable.
"""
from __future__ import annotations

import textwrap

import pytest

from research.model_lab import model_suite as ms


def test_shipped_registry_validates_clean():
    reg = ms.load_registry()
    errors = ms.validate(reg)
    assert errors == [], "shipped model_registry.yaml must validate clean:\n" + "\n".join(errors)


def test_every_model_has_required_fields_and_valid_enums():
    reg = ms.load_registry()
    for m in reg["models"]:
        for f in ms.REQUIRED_FIELDS:
            assert f in m, f"{m.get('model_id')} missing {f}"
        assert m["status"] in ms.STATUS_ENUM
        assert m["family"] in ms.FAMILY_ENUM


def test_claimable_models_have_existing_evidence():
    reg = ms.load_registry()
    claimable = [m for m in reg["models"] if ms.is_claimable(m)]
    assert claimable, "at least the incumbent bacteria model should be claimable"
    for m in claimable:
        assert m["status"] not in ms.NEVER_CLAIMABLE
        assert (ms._REPO_ROOT / m["evidence"]).exists()


def test_failed_and_negative_models_are_not_claimable():
    reg = ms.load_registry()
    by_id = {m["model_id"]: m for m in reg["models"]}
    assert by_id["early_warning_cusum"]["status"] == "negative_result"
    assert ms.is_claimable(by_id["early_warning_cusum"]) is False


def _write_reg(tmp_path, body: str):
    p = tmp_path / "reg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_validate_flags_unknown_status(tmp_path):
    reg = ms.load_registry(_write_reg(tmp_path, """
        models:
          - model_id: x
            family: tree
            architecture: a
            task: bacteria_classification
            status: totally_made_up
            claimable: false
            supports_calibration: false
            dependencies: []
            entrypoint: e.py
    """))
    errors = ms.validate(reg)
    assert any("status" in e for e in errors)


def test_validate_fails_closed_on_claimable_without_evidence(tmp_path):
    reg = ms.load_registry(_write_reg(tmp_path, """
        models:
          - model_id: x
            family: tree
            architecture: a
            task: bacteria_classification
            status: production_candidate
            claimable: true
            supports_calibration: true
            dependencies: []
            entrypoint: e.py
            evidence: ""
    """))
    errors = ms.validate(reg)
    assert any("evidence" in e for e in errors)
    # and is_claimable returns False regardless
    assert ms.is_claimable(reg["models"][0]) is False


def test_validate_rejects_claimable_failed_gate(tmp_path):
    reg = ms.load_registry(_write_reg(tmp_path, """
        models:
          - model_id: x
            family: continual
            architecture: a
            task: m1_forecasting
            status: research_only_failed_gate
            claimable: true
            supports_calibration: false
            dependencies: []
            entrypoint: e.py
            evidence: README.md
    """))
    errors = ms.validate(reg)
    assert any("may not be claimable" in e for e in errors)


def test_render_report_is_ascii_safe_and_mentions_claimable_headline():
    reg = ms.load_registry()
    md = ms.render_report(reg)
    md.encode("cp1252")  # CLI prints to a Windows console
    assert "claimable" in md.lower()
    assert "bacteria_hgbt_isotonic" in md
    assert "deps available?" in md
    assert "runs?" not in md
