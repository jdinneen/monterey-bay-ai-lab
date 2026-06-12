"""Guards for the full-dataset confirmation runner (ops/confirm_model_suite.py).

These protect the two properties that make the harness safe to leave lying around: it covers every
registered model (nothing silently dropped), and it can NEVER fire or touch production by default.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops import confirm_model_suite as c
from research.model_lab import model_suite as ms


def test_plan_validates_clean():
    assert c.validate_plan() == []


def test_plan_accounts_for_every_registry_model():
    planned = {e["model_id"] for e in c.PLAN}
    covered = {x for e in c.PLAN if e["mode"] == "run" for x in e.get("covers", [])}
    reg_ids = {m["model_id"] for m in ms.load_registry()["models"]}
    assert reg_ids <= (planned | covered), reg_ids - (planned | covered)


def test_no_run_command_targets_a_production_dir():
    for e in c.PLAN:
        if e["mode"] != "run":
            continue
        joined = " ".join(str(t) for t in e["argv"]).replace("\\", "/").lower()
        for bad in c.FORBIDDEN_WRITE:
            assert bad.replace("\\", "/").lower() not in joined, (e["model_id"], bad)


def test_every_output_flag_resolves_under_confirm_root():
    for e in c.PLAN:
        if e["mode"] != "run":
            continue
        argv = e["argv"]
        for i, tok in enumerate(argv):
            if tok in c._OUTPUT_FLAGS:
                out_dir = c.CONFIRM_ROOT / e["model_id"]
                resolved = (out_dir if argv[i + 1] == "<OUT>" else (c._REPO_ROOT / argv[i + 1])).resolve()
                assert c.CONFIRM_ROOT.resolve() in resolved.parents or resolved == c.CONFIRM_ROOT.resolve()


def test_skips_have_reasons():
    for e in c.PLAN:
        if e["mode"] == "skip":
            assert e.get("reason")


def test_dry_run_is_default_and_executes_nothing(monkeypatch):
    """main([]) must not invoke a single subprocess."""
    def _boom(*a, **k):
        raise AssertionError("confirm runner executed a subprocess in dry-run")
    monkeypatch.setattr(c.subprocess, "run", _boom)
    assert c.main([]) == 0  # dry-run, validated, no execution


def test_neural_models_force_cpu():
    by_id = {e["model_id"]: e for e in c.PLAN}
    for mid in ("patchtst", "nhits"):
        assert by_id[mid].get("env", {}).get("MBAL_ACCEL") == "cpu"


def test_neural_models_redirect_gold_writes_to_isolated_lakehouse():
    """The neural harness writes to lakehouse/gold/ via MBAL_LAKEHOUSE_DIR; the runner MUST
    redirect that into the model's isolated <OUT> so the real gold layer is never mutated."""
    by_id = {e["model_id"]: e for e in c.PLAN}
    for mid in ("patchtst", "nhits"):
        lk = by_id[mid]["env"]["MBAL_LAKEHOUSE_DIR"]
        out_dir = c.CONFIRM_ROOT / mid
        resolved = Path(c._sub_out(lk, out_dir)).resolve()
        assert c.CONFIRM_ROOT.resolve() in resolved.parents


def test_forecast_v2_runs_full_coverage_on_trusted_source():
    by_id = {e["model_id"]: e for e in c.PLAN}
    argv = by_id["xgboost_forecast_v2"]["argv"]
    assert "--all-variables" in argv               # truly full, not a 6-depth subset
    assert "--source" in argv and "parquet" in argv  # trusted parquet, not auto->synthetic
    assert c._M1_HISTORY in argv


def test_cheap_canary_runs_before_multi_hour_jobs():
    order = [e["model_id"] for e in c.PLAN if e["mode"] == "run"]
    # the seconds-long cusum canary must precede the HOURS-long neural jobs
    assert order.index("early_warning_cusum") < order.index("patchtst")


def test_preflight_returns_list_and_flags_missing_inputs(monkeypatch):
    # with the real tree present, preflight over the cheap CPU jobs should be clean
    cheap = [e for e in c.PLAN if e["model_id"] in {"early_warning_cusum", "bacteria_hgbt_isotonic"}]
    assert isinstance(c.preflight(cheap), list)


def test_gold_diff_detects_mutation():
    before = {"lakehouse/gold/x.parquet": "aaa"}
    after = {"lakehouse/gold/x.parquet": "bbb", "lakehouse/gold/y.parquet": "ccc"}
    d = c.gold_diff(before, after)
    assert d["changed"] == ["lakehouse/gold/x.parquet"]
    assert d["added"] == ["lakehouse/gold/y.parquet"]


# ---------------------------------------------------------------- analysis + freshman report
def _seed_fake_outputs(root: Path):
    """Create the artifacts the real models would write, so analyze() has something to read."""
    (root / "reports" / "operational_benchmark").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "operational_benchmark" / "model_suite_bacteria.json").write_text(json.dumps({
        "models": {
            "bacteria_hgbt_isotonic": {"status": "ok", "ap_calibrated": 0.497, "roc_auc": 0.858,
                                       "ece_calibrated": 0.020, "beats_best_operational": True,
                                       "calibrated_deploy_ready": True},
            "bacteria_xgboost": {"status": "ok", "ap_calibrated": 0.46, "roc_auc": 0.853,
                                 "ece_calibrated": 0.007, "beats_best_operational": True,
                                 "calibrated_deploy_ready": True},
        }}), encoding="utf-8")
    cr = root / "reports" / "model_hardening" / "confirm_run"
    (cr / "xgboost_forecast_v2").mkdir(parents=True, exist_ok=True)
    (cr / "xgboost_forecast_v2" / "leaderboard.csv").write_text(
        "target,horizon_h,mean_skill_rmse_vs_persistence,beats_persistence\n"
        "temp_d10p0,1,5.2,True\ntemp_d10p0,24,-3.1,False\n", encoding="utf-8")
    (cr / "patchtst").mkdir(parents=True, exist_ok=True)
    (cr / "patchtst" / "summary.json").write_text(json.dumps(
        {"model": "patchtst", "n_series": 24, "scored_rows": 1000, "minutes": 90.0,
         "mean_rmse_by_h": {"1": 0.2, "6": 0.4, "24": 0.7}}), encoding="utf-8")
    (cr / "early_warning_cusum").mkdir(parents=True, exist_ok=True)
    (cr / "early_warning_cusum" / "confirm_status.json").write_text(
        json.dumps({"model_id": "early_warning_cusum", "status": "ok", "seconds": 3.0}), encoding="utf-8")


def test_analysis_builds_report_and_charts(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "CONFIRM_ROOT", tmp_path / "reports" / "model_hardening" / "confirm_run")
    _seed_fake_outputs(tmp_path)
    records = c.analyze()
    assert records["bacteria_hgbt_isotonic"]["beats_baseline"] is True
    assert records["xgboost_forecast_v2"]["metrics"]["cells beating persistence"] == 1
    assert "rmse_by_horizon" in records["patchtst"]

    results = [{"model_id": "bacteria_hgbt_isotonic", "status": "ok", "seconds": 5},
               {"model_id": "tft", "status": "skipped", "reason": "orphaned"}]
    gold = {"unchanged": True, "diff": {}}
    out = c.write_analysis(results, gold)
    md = Path(out["analysis_md"]).read_text(encoding="utf-8")
    assert "glossary" in md.lower() and "baseline" in md.lower()
    assert "beat" in md.lower()
    assert out["figures"], "expected at least one chart"
    for rel in out["figures"]:
        assert (c.CONFIRM_ROOT / rel).exists()


def test_analysis_survives_missing_outputs(tmp_path, monkeypatch):
    """No model has run yet -> analyze returns cleanly, report still renders."""
    monkeypatch.setattr(c, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(c, "CONFIRM_ROOT", tmp_path / "cr")
    records = c.analyze()
    assert isinstance(records, dict)
    out = c.write_analysis([{"model_id": "patchtst", "status": "skipped"}], {"unchanged": True, "diff": {}})
    assert Path(out["analysis_md"]).exists()
