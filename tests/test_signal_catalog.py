"""Tests for the signal catalog (signals/catalog.yaml + catalog.py).

Guards the fail-closed validator and the renderer, and asserts the seeded ledger encodes the
honest facts we established (PRIMARY/KEEP/WASH/REJECT verdicts + the DA coverage caveat).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "signals"))
import catalog as C  # noqa: E402


def test_real_catalog_is_valid():
    doc = C.load()
    assert C.validate(doc) == []  # the committed catalog must pass fail-closed validation


def test_seeded_verdicts_match_what_we_established():
    doc = C.load()
    by = {s["name"]: s["predictive_value"] for s in doc["signals"]}
    assert by["rainfall"]["bacteria_exceedance"]["status"] == "KEEP"
    assert by["tide_water_level"]["bacteria_exceedance"]["status"] == "WASH"
    assert by["domoic_acid_pda"]["domoic_acid"]["status"] == "PRIMARY"
    # DA must NOT be claimed as a bacteria signal — only UNTESTED (coverage-limited)
    assert by["domoic_acid_pda"]["bacteria_exceedance"]["status"] == "UNTESTED"
    # the driver-null discipline is recorded
    assert by["hab_physical_nutrient_drivers"]["domoic_acid"]["status"] == "REJECT"


def _entry(**pv):
    """A schema-complete signal entry with the given predictive_value, for validator tests."""
    return {"name": "x", "modality": "derived", "captures": "c", "coverage": "cov",
            "access": "a", "does": "d", "does_not": "dn", "why": "w",
            "predictive_value": pv}


def test_validator_requires_clarity_fields(tmp_path):
    bare = {"name": "x", "modality": "derived", "coverage": "c", "access": "a",
            "predictive_value": {"t": {"status": "UNTESTED"}}}  # missing captures/does/does_not/why
    assert any("missing fields" in p for p in C.validate({"signals": [bare]}, root=tmp_path))


def test_validator_rejects_bad_status(tmp_path):
    doc = {"signals": [_entry(t={"status": "MAGIC"})]}
    assert any("bad status" in p for p in C.validate(doc, root=tmp_path))


def test_validator_requires_evidence_for_skill_claims(tmp_path):
    doc = {"signals": [_entry(t={"status": "KEEP", "metric": "m"})]}
    assert any("requires an evidence path" in p for p in C.validate(doc, root=tmp_path))


def test_validator_flags_missing_evidence_file(tmp_path):
    doc = {"signals": [_entry(t={"status": "PRIMARY", "metric": "m", "evidence": "does/not/exist.md"})]}
    assert any("evidence not found" in p for p in C.validate(doc, root=tmp_path))


def test_render_is_nonempty_and_ascii_table():
    doc = C.load()
    md = C.render(doc)
    assert "# Signal Catalog" in md and "domoic_acid_pda" in md
    assert "| signal |" in md  # the by-target matrix header
