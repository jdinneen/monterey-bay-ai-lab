"""Tests for the leakage-safe driver→mortality model (Whale 2 lane).

Validates the harness on SYNTHETIC panels before Whale 1's real panel exists:
- leakage discipline: period-t features use only pre-t pDA;
- a panel whose mortality is driven by LAGGED pDA → model beats seasonal climatology;
- a panel whose mortality is pure seasonal noise → model does NOT claim lift (honest null).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.whale.mortality_model import (
    add_lagged_pda, seasonal_baseline_predict, time_split, evaluate,
)


def _pda_series(n_months=72, seed=0):
    idx = pd.period_range("2015-01", periods=n_months, freq="M")
    rng = np.random.default_rng(seed)
    # bloom-like: mostly low with occasional spikes
    vals = np.abs(rng.normal(0.1, 0.05, n_months)) + (rng.random(n_months) < 0.15) * rng.random(n_months) * 2
    return pd.Series(vals, index=idx)


def _panel_from_pda(pda, k_pda=0.0, seasonal_amp=0.0, region="central", seed=1):
    """Build a panel where mortality_count = base + k_pda*pda_{t-1} + seasonal + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in pda.index[3:]:  # leave room for lags
        prev = pda.get(m - 1, 0.0)
        seasonal = seasonal_amp * (1 + np.sin(2 * np.pi * m.month / 12))
        mu = max(0.0, 1.0 + k_pda * prev + seasonal + rng.normal(0, 0.2))
        rows.append({"region": region, "period_start": m.to_timestamp(),
                     "species_group": "all_cetacea", "mortality_count": float(rng.poisson(mu))})
    return pd.DataFrame(rows)


def test_add_lagged_pda_is_leakage_safe():
    pda = _pda_series(36)
    panel = _panel_from_pda(pda)
    feat = add_lagged_pda(panel, pda)
    # for a known row, pda_lag1 must equal the PREVIOUS month's pDA, never the current month's
    row = feat.iloc[10]
    m = pd.Period(pd.Timestamp(row["period_start"]).to_period("M"), "M")
    assert row["pda_lag1"] == pytest_approx(pda.get(m - 1))
    assert row["pda_lag2"] == pytest_approx(pda.get(m - 2))
    # current-month pDA must NOT appear as any lag feature (no leakage)
    cur = pda.get(m)
    if cur is not None and not np.isnan(cur):
        assert not np.isclose([row["pda_lag1"], row["pda_lag2"], row["pda_lag3"]], cur).all()


def test_time_split_is_chronological():
    pda = _pda_series(48)
    panel = _panel_from_pda(pda)
    tr, te = time_split(panel)
    assert tr["period_start"].max() <= te["period_start"].min()


def test_seasonal_baseline_predicts_group_means():
    df = pd.DataFrame({"region": ["a"] * 4, "month_of_year": [1, 1, 2, 2],
                       "mortality_count": [10.0, 20.0, 5.0, 7.0]})
    pred = seasonal_baseline_predict(df, df)
    # month 1 rows -> mean 15, month 2 rows -> mean 6
    assert list(pred) == [15.0, 15.0, 6.0, 6.0]


def test_model_beats_climatology_when_pda_drives_mortality():
    pda = _pda_series(96, seed=2)
    panel = _panel_from_pda(pda, k_pda=8.0, seasonal_amp=0.0, seed=3)  # strong DA→death link
    res = evaluate(panel, pda)
    assert res["status"] == "ok"
    assert res["beats_baseline"], f"expected DA signal to beat climatology, got {res}"


def test_model_reports_honest_null_when_no_pda_link():
    pda = _pda_series(96, seed=4)
    # mortality purely seasonal, independent of pDA -> drivers should NOT add lift
    panel = _panel_from_pda(pda, k_pda=0.0, seasonal_amp=1.5, seed=5)
    res = evaluate(panel, pda)
    assert res["status"] == "ok"
    # honest: skill should be near/below zero (climatology already captures the seasonality)
    assert res["skill_vs_climatology"] < 0.25


# tiny local approx helper (avoid importing pytest.approx at module top for clarity)
def pytest_approx(x, rel=1e-6):
    import pytest
    return pytest.approx(x, rel=rel)
