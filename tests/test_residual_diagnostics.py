"""Tests for the residual-diagnostics discovery front-end (research/bacteria/residual_diagnostics.py).

Fast and deterministic: they exercise the per-axis effect-size estimators and the
irreducible-error floor on SYNTHETIC residual frames, not a full model fit — we assert that
injected structure is detected and that pure noise is not (the false-positive guard that keeps
the backlog honest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import residual_diagnostics as rd


def test_eta_squared_detects_group_structure_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 1500
    groups = rng.integers(0, 5, n)
    # residual driven entirely by group identity -> eta^2 near 1
    structured = groups.astype(float) + rng.normal(0, 0.01, n)
    assert rd._eta_squared(structured, groups) > 0.9
    # residual independent of group -> eta^2 near 0 (and below the flag threshold)
    noise = rng.normal(0, 1, n)
    assert rd._eta_squared(noise, groups) < rd.GROUP_ICC_MIN


def test_driver_decile_profile_detects_monotone_bias():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    # residual increases monotonically with the driver -> large spread across deciles
    resid = x + rng.normal(0, 0.05, 3000)
    rng_val, prof = rd._driver_decile_profile(resid, x)
    assert rng_val >= rd.DRIVER_RANGE_MIN
    assert len(prof) == 10 and prof[0] < prof[-1]
    # flat residual vs driver -> negligible spread
    flat_range, _ = rd._driver_decile_profile(rng.normal(0, 0.01, 3000), x)
    assert flat_range < rd.DRIVER_RANGE_MIN


def test_driver_decile_profile_handles_sentinel_and_sparse():
    # -999.0 tide sentinels are dropped, not treated as a real low value; with only 40 real
    # points left (< q*5) the profile declines to bin rather than fabricate deciles
    x = np.concatenate([np.full(60, -999.0), np.linspace(0, 1, 40)])
    resid = np.concatenate([np.full(60, 5.0), np.zeros(40)])
    rng_val, prof = rd._driver_decile_profile(resid, x)
    assert rng_val == 0.0 and prof == []  # too few non-sentinel points to bin


def test_irreducible_floor_reflects_label_noise():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    n = 2000
    # one informative covariate; label is a noisy threshold on it
    feat = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-3 * feat))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"feat": feat, "exceed": y})
    out = rd._irreducible_floor(df, ["feat"], k=15)
    assert 0.0 <= out["bayes_brier_floor"] <= 0.25  # mean p(1-p) is bounded by 0.25
    # a label that is a DETERMINISTIC function of the feature has ~no irreducible Brier floor
    df_det = pd.DataFrame({"feat": feat, "exceed": (feat > 0).astype(int)})
    det = rd._irreducible_floor(df_det, ["feat"], k=15)
    assert det["bayes_brier_floor"] < out["bayes_brier_floor"]


def test_present_drivers_only_flags_populated_columns():
    df = pd.DataFrame({
        "rain_3d": [1.0, 2.0, np.nan],
        "discharge_log": [np.nan, np.nan, np.nan],   # present but empty -> not a usable driver
        "water_level_m": [-999.0, -999.0, -999.0],   # all sentinel -> not usable
        "Hs": [0.5, 1.0, 1.5],
    })
    present = rd._present_drivers(df)
    assert present.get("rain_3d") == "rain"
    assert present.get("Hs") == "waves"
    assert "discharge_log" not in present
    assert "water_level_m" not in present
