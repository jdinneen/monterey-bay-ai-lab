"""Smoke test for the pre-registered drift-detector core (research/bacteria/early_warning).

Tests the CUSUM primitive in isolation (no dependency on the gitignored statewide data):
it must stay quiet under no-shift data and accumulate past threshold after a step change.
The full pre-registered run() is exercised against the real data by the analysis script.
"""
from __future__ import annotations

import pandas as pd

from research.bacteria import early_warning as ew


def test_cusum_quiet_before_step_fires_after():
    months = pd.period_range("2022-01", "2022-12", freq="M")
    rates = [0.10] * 6 + [0.60] * 6  # baseline ~0.10, then a regime step from 2022-07
    cm_c = pd.DataFrame({"county": "X", "reveal_month": months, "p": rates, "n": 50})
    path = ew.cusum_path(cm_c, mu=0.10, sigma=0.05, lo="2022-01-01", hi="2022-12-31")
    pre = path[path["reveal_month"] < pd.Period("2022-07", "M")]["S"].max()
    post = path[path["reveal_month"] >= pd.Period("2022-10", "M")]["S"].max()
    assert pre < 1.0            # quiet under no-shift data (CUSUM hovers near 0)
    assert post > pre + 3.0     # accumulates clearly after the step


def test_cusum_flat_series_stays_near_zero():
    months = pd.period_range("2022-01", "2022-12", freq="M")
    cm_c = pd.DataFrame({"county": "X", "reveal_month": months, "p": [0.10] * 12, "n": 50})
    path = ew.cusum_path(cm_c, mu=0.10, sigma=0.05, lo="2022-01-01", hi="2022-12-31")
    assert path["S"].max() == 0.0  # exactly-at-reference data never accumulates
