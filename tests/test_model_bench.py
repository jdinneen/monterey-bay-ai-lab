"""Offline tests for the comprehensive model bench (ops/model_bench.py).

Network-free / data-free: exercises the model zoo construction and the aggregation +
rendering logic on synthetic cell results. The real per-model metrics come from the
leakage-safe seam and are validated by tests/test_operational_benchmark.py.
"""
from __future__ import annotations

from ops import model_bench as mb


def test_model_zoo_has_all_families_and_callable_factories():
    zoo = mb.model_zoo()
    for fam in ("hgbt", "xgboost", "lightgbm", "catboost",
                "random_forest", "extra_trees", "logistic"):
        assert fam in zoo, f"missing model family {fam}"
    # the extra sklearn families must construct (deps present in this env)
    for fam in ("random_forest", "extra_trees", "logistic"):
        factory, state = zoo[fam]
        assert state == "ok" and callable(factory)
        est = factory()
        assert hasattr(est, "fit") and hasattr(est, "predict_proba")


def test_driver_sets_cover_every_seam_dir():
    # 'all' must reference every driver dir the seam supports
    assert set(mb.DRIVER_SETS["all"]) == set(mb.DRIVER_DIRS)
    assert mb.DRIVER_SETS["rain"] == ["rain_dir"]


def _cell(model, drivers, ap, auc=0.85):
    return {"model": model, "drivers": drivers, "status": "ok",
            "headline": {"ap_calibrated": ap, "roc_auc": auc, "ece": 0.02,
                         "best_baseline_ap": 0.37, "beats_ab411": True,
                         "beats_vb_mlr": True, "deploy_ready": True}}


def test_aggregate_ranks_and_computes_driver_lift():
    cells = [
        _cell("hgbt", "rain", 0.49), _cell("hgbt", "all", 0.50),
        _cell("logistic", "rain", 0.40), _cell("logistic", "all", 0.41),
        {"model": "catboost", "drivers": "all", "status": "errored", "error": "Boom"},
        {"model": "lightgbm", "drivers": "rain", "status": "unavailable_dependency"},
    ]
    agg = mb.aggregate(cells, hab=None)
    assert agg["n_ok"] == 4
    # ranked by AP desc -> hgbt/all first
    assert agg["leaderboard"][0]["model"] == "hgbt" and agg["leaderboard"][0]["drivers"] == "all"
    # driver lift = all - rain
    assert agg["driver_lift_all_minus_rain"]["hgbt"] == 0.01
    assert any(e["model"] == "catboost" for e in agg["errored"])
    assert "lightgbm" in agg["unavailable"]


def test_render_md_is_stable():
    cells = [_cell("hgbt", "all", 0.50), _cell("hgbt", "rain", 0.49)]
    agg = mb.aggregate(cells, hab={"status": "ok", "headline": {"ap": 0.23, "roc_auc": 0.83,
                                                                "beats_baselines": True}})
    md = mb.render_md(agg)
    assert "Model Bench" in md and "hgbt" in md and "HAB task" in md
    assert "KEEP" in md or "WASH" in md  # lift section rendered
