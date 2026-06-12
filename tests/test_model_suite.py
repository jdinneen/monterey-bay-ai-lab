"""Tests for the live bacteria model suite — reuses the synthetic-obs + clf seam from
operational_benchmark so it runs without the 1.3M-row real dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import operational_benchmark as ob
from research.model_lab import model_suite as ms


def _synthetic_obs(tmp_path):
    """Two regions, dirty/clean stations, all four analytes, weekly 2017-2023 — same shape the
    operational_benchmark tests use, so run() exercises the real leakage-safe pipeline."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", "2023-06-01", freq="7D")
    stations = [
        ("sd_dirty", "San Diego", 0.6), ("sd_clean", "San Diego", 0.05),
        ("mb_dirty", "Monterey", 0.4), ("mb_clean", "Monterey", 0.05),
        ("la_dirty", "Los Angeles", 0.5), ("la_clean", "Los Angeles", 0.06),
    ]
    rows = []
    for sid, county, p in stations:
        for d in dates:
            exceed = rng.random() < p
            vals = {
                "Enterococcus": 300.0 if exceed else float(rng.integers(1, 90)),
                "Fecal Coliforms": float(rng.integers(1, 350)),
                "Total Coliforms": float(rng.integers(10, 9000)),
                "E. Coli": float(rng.integers(1, 200)),
            }
            for param, val in vals.items():
                rows.append({
                    "sample_date": d, "county": county, "beach_name": sid, "station_name": sid,
                    "station_id": sid, "source_parameter": param, "property_id": "p",
                    "result_comparator": "=", "result_value_numeric": val,
                })
    path = tmp_path / "statewide_beach_observations.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_bacteria_suite_runs_hgbt_and_xgboost(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = _synthetic_obs(tmp_path)
    # no rain dir in the synthetic run; label "any" so the tiny set has both classes
    suite = ms.run_bacteria_suite(obs_path=obs, rain_dir=None, reveal_lag_days=0,
                                  label="any", stratum="ALL")
    assert "error" not in suite
    for mid in ("bacteria_hgbt_isotonic", "bacteria_xgboost"):
        r = suite["models"][mid]
        assert r["status"] == "ok", r
        assert 0.0 <= r["ap_calibrated"] <= 1.0
        assert 0.0 <= r["roc_auc"] <= 1.0
        assert isinstance(r["passes_suite_gate"], bool)
        assert "suite_claimable" not in r


def test_missing_optional_dependency_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = _synthetic_obs(tmp_path)
    # lightgbm / catboost are OPTIONAL: whether or not they are installed in this env, a
    # *missing* one must be reported (fail-closed), not crash the suite. Mock them absent
    # so the test is deterministic regardless of the local environment.
    import importlib.util as _ilu
    _real = _ilu.find_spec
    monkeypatch.setattr(_ilu, "find_spec",
                        lambda name, *a, **k: None if name in ("lightgbm", "catboost")
                        else _real(name, *a, **k))
    suite = ms.run_bacteria_suite(obs_path=obs, rain_dir=None, reveal_lag_days=0,
                                  label="any", stratum="ALL")
    for mid in ("bacteria_lightgbm", "bacteria_catboost"):
        assert suite["models"][mid]["status"] == "unavailable_dependency"


def test_suite_reports_missing_data_cleanly(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    suite = ms.run_bacteria_suite(obs_path=tmp_path / "nope.parquet", rain_dir=None)
    assert "error" in suite


def test_suite_markdown_is_ascii_safe(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = _synthetic_obs(tmp_path)
    suite = ms.run_bacteria_suite(obs_path=obs, rain_dir=None, reveal_lag_days=0,
                                  label="any", stratum="ALL")
    ms.suite_to_markdown(suite).encode("cp1252")


# ---------------------------------------------------------------- model-hardening audit
def test_inventory_covers_every_model_with_required_audit_keys():
    reg = ms.load_registry()
    inv = ms.build_inventory(reg)
    assert len(inv) == len(reg["models"])
    required = {
        "model_id", "entrypoint", "task", "family", "architecture", "status",
        "classification", "source_exists", "deps_available", "evidence_exists",
        "runnable", "claimable", "baseline", "output_dir_policy", "recommended_action",
    }
    for r in inv:
        assert required <= set(r), f"{r.get('model_id')} missing {required - set(r)}"
        assert r["classification"] in set(ms.CLASSIFICATION.values())
        assert r["recommended_action"] in {
            "KEEP_AND_HARDEN", "WRAP_AS_BENCHMARK", "DEMOTE_TO_RESEARCH",
            "MARK_ORPHANED", "RETIRE_CLAIM", "FIX_NOW",
        }


def test_shipped_registry_entrypoints_and_evidence_exist_on_disk():
    """Every registered entrypoint must be a real file; every declared evidence path must resolve.
    This is what stops a model from being 'in the registry' but un-runnable/un-evidenced."""
    reg = ms.load_registry()
    for m in reg["models"]:
        assert ms.source_exists(m), f"{m['model_id']}: entrypoint missing: {m.get('entrypoint')}"
        if (m.get("evidence") or "").strip():
            assert ms.evidence_exists(m), f"{m['model_id']}: evidence missing: {m['evidence']}"


def test_runnable_requires_source_deps_and_smoke_seam():
    reg = ms.load_registry()
    for m in reg["models"]:
        if ms.runnable(m):
            assert ms.source_exists(m)
            assert ms.deps_available(m)
            assert ms.has_smoke_seam(m)


def test_orphaned_models_are_not_runnable_and_not_claimable():
    reg = ms.load_registry()
    for m in reg["models"]:
        if m["status"] == "orphaned":
            assert ms.runnable(m) is False, f"{m['model_id']} orphaned but reported runnable"
            assert ms.is_claimable(m) is False


def test_only_production_candidates_are_claimable_in_status_matrix():
    reg = ms.load_registry()
    matrix = ms.build_status_matrix(reg)
    for mid, row in matrix["models"].items():
        if row["claimable"]:
            assert row["classification"] == "PRODUCTION_CANDIDATE", mid
            assert row["evidence_exists"], mid


def test_status_matrix_counts_are_self_consistent():
    reg = ms.load_registry()
    matrix = ms.build_status_matrix(reg)
    assert matrix["registry_valid"] is True
    assert matrix["total_models"] == len(reg["models"])
    assert sum(matrix["by_classification"].values()) == matrix["total_models"]
    assert matrix["claimable"] == sum(1 for r in matrix["models"].values() if r["claimable"])
    assert matrix["runnable"] == sum(1 for r in matrix["models"].values() if r["runnable"])


def test_inventory_md_is_ascii_safe():
    reg = ms.load_registry()
    ms.render_inventory_md(reg).encode("cp1252")  # CLI prints/writes on a Windows console
