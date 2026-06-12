"""Focused tests for ops/project_status.py (the read-only control tower)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops import project_status as ps

REQUIRED_JSON_KEYS = [
    "generated_at", "git_status_summary", "untracked_important_files",
    "data_assets", "fetchers", "models", "gates", "tests",
    "active_processes", "unsafe_artifacts", "public_claims", "next_actions",
]


def _build_empty(tmp_path: Path) -> dict:
    return ps.build_status(tmp_path, include_processes=False)


def test_runs_with_missing_optional_artifacts(tmp_path):
    """An empty root (no git, no registry, no reports) must not raise."""
    status = _build_empty(tmp_path)
    assert status["overall_status"] in ("GREEN", "YELLOW", "RED")
    assert status["models"].get("error")  # registry missing is reported, not fatal
    assert all(g["overall_status"] == "missing" for g in status["gates"])


def test_json_has_required_keys(tmp_path):
    status = _build_empty(tmp_path)
    for key in REQUIRED_JSON_KEYS:
        assert key in status, f"missing key: {key}"
    # and it must serialize cleanly
    json.dumps(status, default=str)


def test_markdown_renders_cp1252_safe(tmp_path):
    status = _build_empty(tmp_path)
    md = ps.render_markdown(status)
    md.encode("cp1252")  # must not raise
    md.encode("ascii")   # report is forced to ASCII


def test_html_is_self_contained_and_valid(tmp_path):
    """HTML dashboard must render, be UTF-8 clean, and need no network."""
    status = _build_empty(tmp_path)
    html = ps.render_html(status)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    html.encode("utf-8")  # must not raise
    # no external resources: no network access allowed
    low = html.lower()
    assert "http://" not in low and "https://" not in low
    assert "<script" not in low  # no JS dependency
    # the overall status badge is present
    assert status["overall_status"] in html


def test_html_escapes_untrusted_strings(tmp_path):
    yaml = pytest.importorskip("yaml")
    reg = tmp_path / "research" / "model_lab"
    reg.mkdir(parents=True)
    (reg / "model_registry.yaml").write_text(
        yaml.safe_dump({"models": [{
            "model_id": "x<script>y", "family": "tree",
            "task": "bacteria_classification", "status": "benchmark",
            "claimable": False, "evidence": "",
        }]}),
        encoding="utf-8")
    html = ps.render_html(_build_empty(tmp_path))
    assert "x&lt;script&gt;y" in html
    assert "x<script>y" not in html


def test_writes_only_inside_report_dir(tmp_path):
    (tmp_path / "some_data.txt").write_text("x", encoding="utf-8")
    before = {p for p in tmp_path.rglob("*")}
    rc = ps.main(["--root", str(tmp_path), "--no-processes"])
    assert rc == 0
    new = {p for p in tmp_path.rglob("*")} - before
    allowed = tmp_path / "reports" / "project_status"
    for p in new:
        assert p == allowed or p == allowed.parent or str(p).startswith(str(allowed)), (
            f"wrote outside report dir: {p}")
    assert (allowed / "PROJECT_STATUS.md").exists()
    assert (allowed / "project_status.json").exists()


def test_smoke_and_quarantine_detection(tmp_path):
    nn = tmp_path / "nn_results"
    (nn / "dbx_smoke_dlinear").mkdir(parents=True)
    quarantine = nn / "_quarantine"
    quarantine.mkdir()
    (quarantine / "bad_run").mkdir()
    status = _build_empty(tmp_path)
    kinds = {(u["kind"], u["path"]) for u in status["unsafe_artifacts"]}
    assert ("smoke_output", "nn_results/dbx_smoke_dlinear") in kinds
    assert ("quarantine", "nn_results/_quarantine") in kinds


def test_stale_artifact_detection(tmp_path):
    results = tmp_path / "mbal_forecast_v2_results"
    results.mkdir()
    target = results / "model_results.json"
    target.write_text("{}", encoding="utf-8")
    old = time.time() - 90 * 86400
    os.utime(target, (old, old))
    status = ps.build_status(tmp_path, stale_days=14, include_processes=False)
    stale = [u for u in status["unsafe_artifacts"] if u["kind"] == "stale_output"]
    assert any("model_results.json" in u["path"] for u in stale)


def test_broken_claim_detection(tmp_path):
    yaml = pytest.importorskip("yaml")
    reg = tmp_path / "research" / "model_lab"
    reg.mkdir(parents=True)
    (reg / "model_registry.yaml").write_text(
        yaml.safe_dump({"models": [{
            "model_id": "ghost_model", "family": "tree", "task": "bacteria_classification",
            "status": "production_candidate", "claimable": True,
            "evidence": "does/not/exist.json",
        }]}),
        encoding="utf-8")
    status = _build_empty(tmp_path)
    broken = [u for u in status["unsafe_artifacts"] if u["kind"] == "broken_claim"]
    assert broken and broken[0]["path"] == "ghost_model"
    # a broken claim must never surface as a public claim
    assert status["public_claims"]["claimable_models"] == []


def test_smoke_output_never_claimable(tmp_path):
    """Smoke artifacts are flagged unsafe and contribute nothing to claims."""
    (tmp_path / "reports" / "smoke_run_output").mkdir(parents=True)
    status = _build_empty(tmp_path)
    assert any(u["kind"] == "smoke_output" for u in status["unsafe_artifacts"])
    assert status["public_claims"]["claimable_models"] == []
