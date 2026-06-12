"""Tests for the leave-one-beach-out + Moran's I spatial-autocorrelation harness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import operational_benchmark as ob
from research.bacteria import spatial_autocorr as sa


def _synthetic_obs(n_stations: int = 12, seed: int = 0) -> pd.DataFrame:
    """A post-load_station_days frame: one row per station-day with analyte columns and a
    learnable label, spanning the train (<=2019) / cal (2020-21) / test (2022+) eras."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2023-06-30", freq="7D")
    rows = []
    for s in range(n_stations):
        # each station has a baseline dirtiness; label depends on it + recent rain proxy
        level = rng.uniform(0.5, 3.0)
        for d in dates:
            ent = float(np.exp(level + 0.4 * rng.standard_normal()))
            rows.append({
                "station_id": f"st{s:03d}", "county": f"C{s % 3}",
                "sample_date": d, "ent": ent, "fec": ent * 2,
                "tot": ent * 5, "ecoli": ent,
                "exceed": int(ent > np.exp(2.0)),
            })
    return pd.DataFrame(rows)


def test_agg_prev7_excludes_holdout_labels():
    """The recomputed statewide aggregate must not move when a held-out station's label
    changes — i.e. held-out beaches are genuinely excluded from the prior-rate features."""
    df = _synthetic_obs(n_stations=6)
    df = ob.add_causal_features(df)
    hold = {"st000"}

    _, sw_a = sa._agg_prev7_excluding(df, hold)
    df2 = df.copy()
    # flip the held-out station's labels; excluded aggregate must be identical
    mask = df2["station_id"] == "st000"
    df2.loc[mask, "exceed"] = 1 - df2.loc[mask, "exceed"]
    _, sw_b = sa._agg_prev7_excluding(df2, hold)

    np.testing.assert_allclose(np.nan_to_num(sw_a), np.nan_to_num(sw_b))


def test_agg_prev7_changes_when_kept_station_flips():
    """Sanity counterpart: flipping a NON-held-out station DOES change the aggregate."""
    df = ob.add_causal_features(_synthetic_obs(n_stations=6))
    _, sw_a = sa._agg_prev7_excluding(df, {"st000"})
    df2 = df.copy()
    mask = df2["station_id"] == "st001"
    df2.loc[mask, "exceed"] = 1 - df2.loc[mask, "exceed"]
    _, sw_b = sa._agg_prev7_excluding(df2, {"st000"})
    assert not np.allclose(np.nan_to_num(sw_a), np.nan_to_num(sw_b))


def test_morans_i_detects_clustering_and_randomness():
    pytest.importorskip("sklearn")
    # 12x12 lattice of coordinates
    gx, gy = np.meshgrid(np.linspace(36.0, 37.0, 12), np.linspace(-122.0, -121.0, 12))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    smooth = gx.ravel()  # value varies smoothly in space -> strong positive autocorrelation

    clustered = sa.morans_i(smooth, coords, k=6, n_perm=199, seed=1)
    assert clustered["morans_i"] > 0.3
    assert clustered["p_value_positive_autocorr"] < 0.05

    rng = np.random.default_rng(2)
    rand = sa.morans_i(rng.permutation(smooth), coords, k=6, n_perm=199, seed=1)
    assert abs(rand["morans_i"]) < 0.2  # near the expected ~0
    assert rand["p_value_positive_autocorr"] > 0.05


def test_leave_one_beach_out_smoke():
    pytest.importorskip("sklearn")
    df = ob.add_causal_features(_synthetic_obs(n_stations=12, seed=3))
    feats = list(ob.FEATS)
    res = sa.leave_one_beach_out(df, feats, n_splits=3)
    assert "error" not in res
    # every beach with 2022+ rows is held out exactly once -> all scored on unseen folds
    assert res["n_test_beaches"] >= 1
    assert res["n_test_rows"] > 0
    ap = res["models"]["model_calibrated"].get("ap")
    assert ap is None or 0.0 <= ap <= 1.0
    assert set(res["models"]) >= {"model_calibrated", "baseline_station_memory", "baseline_prior_lab"}
