"""Tests for the DA↔whale-mortality evidence core (Whale 2 / modeling-evidence lane).

Covers the pure, data-independent functions so the evidence verdict is trustworthy:
monthly collapsing, and the leakage-safe seasonal window-anomaly (in-window vs a
same-calendar-month baseline built from OUTSIDE the window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.whale.da_mortality_evidence import monthly_series, window_anomaly


def _ts(periods, start="2010-01-01", freq="ME"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def test_monthly_series_collapses_to_one_value_per_month():
    # two readings in the same month -> averaged to one monthly value
    df = pd.DataFrame({
        "time": pd.to_datetime(["2020-03-01", "2020-03-20", "2020-04-05"], utc=True),
        "pDA": [10.0, 20.0, 5.0],
    })
    s = monthly_series(df, "time", "pDA", agg="mean")
    assert s.loc[pd.Period("2020-03", "M")] == pytest.approx(15.0)
    assert s.loc[pd.Period("2020-04", "M")] == pytest.approx(5.0)


def test_monthly_series_drops_non_numeric_and_nan():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2020-03-01", "2020-03-02", "2020-03-03"], utc=True),
        "pDA": ["bad", None, 12.0],
    })
    s = monthly_series(df, "time", "pDA", agg="mean")
    assert s.loc[pd.Period("2020-03", "M")] == pytest.approx(12.0)


def test_window_anomaly_detects_injected_elevation():
    # 10 years of monthly data; flat baseline ~1.0, but a 12-month window is spiked to ~6.0
    n = 120
    idx = _ts(n)
    # realistic noisy baseline (~1.0 ± 0.1) so the robust scale is well-defined
    vals = 1.0 + np.random.default_rng(1).normal(0, 0.1, n)
    monthly = pd.Series(vals, index=idx.to_period("M"))
    # spike calendar year 2018 (months 96..107)
    spike = (monthly.index >= pd.Period("2018-01", "M")) & (monthly.index <= pd.Period("2018-12", "M"))
    monthly[spike] = 6.0
    anom = window_anomaly(monthly, "2018-01", "2018-12")
    assert anom["ratio_window_over_baseline"] == pytest.approx(6.0, rel=0.05)
    assert anom["robust_z"] > 1.0
    assert anom["n_window_months"] == 12


def test_window_anomaly_flat_series_is_quiet():
    # negative control: no elevation anywhere -> window ~ baseline, low/zero z
    n = 120
    monthly = pd.Series(np.ones(n) + np.random.default_rng(0).normal(0, 0.01, n),
                        index=_ts(n).to_period("M"))
    anom = window_anomaly(monthly, "2018-01", "2018-12")
    assert anom["ratio_window_over_baseline"] == pytest.approx(1.0, abs=0.05)
    assert abs(anom["robust_z"]) < 3.0  # not a strong signal


def test_window_anomaly_baseline_excludes_window():
    # the window's own elevated values must NOT leak into its baseline:
    # a permanently-elevated series should look ~flat (ratio ~1), because baseline = same months elsewhere
    n = 120
    monthly = pd.Series(np.ones(n), index=_ts(n).to_period("M"))
    anom = window_anomaly(monthly, "2015-01", "2015-12")
    assert anom["ratio_window_over_baseline"] == pytest.approx(1.0, rel=0.01)


def test_window_anomaly_handles_missing_window():
    monthly = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2010-01", "2010-02"], freq="M"))
    anom = window_anomaly(monthly, "2099-01", "2099-12")
    assert anom["n_window_months"] == 0
    assert anom["reason"] != "ok"
