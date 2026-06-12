#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.model_lab.mbal_big_analysis import _m1_m2_overlap  # noqa: E402


def test_m1_m2_overlap_returns_schema_when_no_series_pass_threshold():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h")
    m1 = pd.DataFrame({"temp_d1p0": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    m2 = pd.DataFrame({"temp_d1p0": [1.1, 2.1, 3.1, 4.1]}, index=idx)

    out = _m1_m2_overlap(m1, m2)

    assert out.empty
    assert list(out.columns) == ["series", "overlap_hours", "m1_mean", "m2_mean", "m2_minus_m1_mean", "corr"]
