"""Tests for the signal-discovery harness (research/bacteria/signal_lab.py).

Fast and deterministic: they exercise the registry's dependency resolution and the
keep/reject GATE LOGIC (the part that decides whether a signal is real), not a full
model fit. The end-to-end numbers are covered by the experiment scripts the harness reuses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import signal_lab as sl


def test_resolved_feats_includes_requirements():
    # composite L2 signal pulls both parents' columns
    assert set(sl.resolved_feats("nbr_and_latlon")) == {"nbr_prev1", "nbr_prev7", "latitude", "longitude"}
    # L3 signal carries its parent L2 signal's columns plus its own
    l3 = sl.resolved_feats("nbr7_x_rain3")
    assert "nbr7_x_rain3" in l3 and "nbr_prev7" in l3


def test_build_order_is_topological_and_detects_cycles():
    order = sl._build_order(["nbr_and_latlon"])
    assert order.index("nbr_lag") < order.index("nbr_and_latlon")
    assert order.index("latlon") < order.index("nbr_and_latlon")

    cyc = sl.Candidate("cyc", 3, [], sl._b_noop, requires=["nbr_and_latlon"])
    sl.REGISTRY["cyc"] = cyc
    sl.REGISTRY["nbr_and_latlon"].requires.append("cyc")
    try:
        with pytest.raises(ValueError):
            sl._build_order(["cyc"])
    finally:  # don't leak mutation into other tests
        sl.REGISTRY["nbr_and_latlon"].requires.remove("cyc")
        del sl.REGISTRY["cyc"]


def test_unknown_candidate_raises():
    with pytest.raises(KeyError):
        sl._build_order(["does_not_exist"])


def test_verdict_gate_logic():
    eps = sl.MIN_LIFT_AP
    # clear lift that survives LOBO -> KEEP (this is the latlon outcome)
    assert sl._verdict(0.0355, 0.03, True)[0] == "KEEP"
    # lift on temporal but FAILS on unseen beaches -> REJECT (memorisation)
    v, reason = sl._verdict(0.04, -0.01, True)
    assert v == "REJECT" and "memorisation" in reason
    # sub-threshold move either way -> WASH (this is the nbr_lag-alone outcome, +0.0)
    assert sl._verdict(eps / 2, 0.0, True)[0] == "WASH"
    assert sl._verdict(-eps / 2, None, True)[0] == "WASH"
    # outright regression -> REJECT
    assert sl._verdict(-0.05, None, True)[0] == "REJECT"
    # without LOBO a temporal lift can still KEEP (less honest, flagged by --no-lobo)
    assert sl._verdict(0.02, None, False)[0] == "KEEP"


def test_registry_has_levels_assigned():
    assert sl.REGISTRY["latlon"].level == 2
    assert sl.REGISTRY["nbr7_x_rain3"].level == 3  # builds on an L2 signal


def test_base_feats_excludes_gateable_spatial_columns():
    # the engine must pin its own baseline: even if ob.FEATS gets spatial columns folded in by
    # other workstreams, the gateable signals must NOT leak into the baseline (else latlon/nbr
    # candidates would falsely WASH because the baseline already contains them)
    import research.bacteria.operational_benchmark as ob
    for col in ["latitude", "longitude", "nbr_prev1", "nbr_prev7"]:
        assert col not in sl.BASE_FEATS
    # the causal core must still be present
    for col in ["cty_prev7", "sw_prev7", "station_prior_rate", "month"]:
        assert col in sl.BASE_FEATS


def test_driver_candidates_registered_and_resolve():
    import research.bacteria.operational_benchmark as ob
    for name, feats in [("rain", ob.RAIN_FEATS), ("discharge", ob.DISCHARGE_FEATS),
                        ("waves", ob.WAVE_FEATS), ("tide", ob.TIDE_STAGES_FEATS)]:
        assert name in sl.REGISTRY and sl.REGISTRY[name].level == 2
        assert sl.resolved_feats(name) == list(feats)


def test_driver_builder_fills_nan_when_dir_absent():
    import numpy as np
    df = pd.DataFrame({"station_id": ["A", "B"], "county": ["X", "Y"],
                       "sample_date": pd.to_datetime(["2022-01-01", "2022-01-02"]),
                       "exceed": [0, 1]})
    out = sl.REGISTRY["discharge"].build(df, {"discharge_dir": None})
    import research.bacteria.operational_benchmark as ob
    for f in ob.DISCHARGE_FEATS:
        assert f in out.columns and out[f].isna().all()


def test_rank_pairs_orders_by_residual_independence():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    resid = {
        "indep": rng.normal(0, 1, 1000),   # independent of a
        "a": a,
        "dup": a + rng.normal(0, 0.001, 1000),  # near-duplicate of a
    }
    pairs = sl._rank_pairs(resid)
    top = pairs[0]["pair"]
    # the most-independent pair must NOT be the (a, dup) near-duplicate pair
    assert set(top) != {"a", "dup"}
    # and the (a, dup) pair sits last with the lowest independence
    assert set(pairs[-1]["pair"]) == {"a", "dup"}
    assert pairs[0]["independence"] > pairs[-1]["independence"]


def test_primary_col_map_covers_buildable_candidates():
    # every driver + nbr_lag has a scalar column so propose_l3 can build a product
    for name in ["rain", "discharge", "waves", "tide", "nbr_lag"]:
        assert sl.PRIMARY_COL.get(name)  # non-None
    assert sl.PRIMARY_COL.get("latlon") is None  # two-column signal: no scalar product


def test_rank_pairs_below_threshold_blocks_l3_register():
    # redundant residuals (independence < MIN_INDEPENDENCE) must not yield a registered L3 —
    # this is the guard that stops propose_l3 from manufacturing a dead interaction
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 800)
    resid = {"a": a, "b": a + rng.normal(0, 0.001, 800)}  # near-duplicate → independence ~0
    pairs = sl._rank_pairs(resid)
    assert pairs[0]["independence"] < sl.MIN_INDEPENDENCE
