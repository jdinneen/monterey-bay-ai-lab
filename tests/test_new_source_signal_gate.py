"""Tests for the new-source BACTERIA signal gate (research/bacteria/new_source_signal_gate.py).

The critical property is LEAKAGE SAFETY: a feature for a sample on date D must use only
source data available strictly before D - reveal_lag. We verify that with a synthetic
source whose value spikes AFTER the sample date — the spike must never enter the feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.bacteria import new_source_signal_gate as g


def _geo():
    return pd.DataFrame({"station_id": ["B1"], "latitude": [36.0], "longitude": [-122.0]})


def test_register_candidates_into_signal_lab():
    from research.bacteria import signal_lab as sl
    names = g.register_new_source_candidates()
    assert "usgs_turbidity" in names and "sst_buoy" in names
    for n in names:
        assert n in sl.REGISTRY and sl.REGISTRY[n].feats == [f"{n}_prev", f"{n}_roll7"]


def test_causal_join_is_leakage_safe(monkeypatch):
    # sample on 2023-06-10; reveal_lag=2 -> as-of 2023-06-08.
    df = pd.DataFrame({"station_id": ["B1"], "sample_date": [pd.Timestamp("2023-06-10")]})
    # source at the SAME location: a benign value BEFORE the sample, a huge spike AFTER it.
    src = pd.DataFrame({
        "lat": [36.0, 36.0, 36.0],
        "lon": [-122.0, -122.0, -122.0],
        "date": pd.to_datetime(["2023-06-01", "2023-06-05", "2023-06-20"]),  # last is AFTER
        "value": [1.0, 2.0, 999.0],  # 999 spike is in the future
    })
    monkeypatch.setattr(g, "_read_src", lambda key: src)
    cfg = dict(src="fake", time="date", value="value", loc=("lat", "lon"))
    out = g._causal_join(df, {"geo": _geo(), "reveal_lag_days": 2}, cfg, "x")
    # the future 999 spike must NOT appear; the most-recent past value (<= 2023-06-08) is 2.0
    assert out["x_prev"].iloc[0] == 2.0
    assert out["x_roll7"].iloc[0] not in (999.0,)
    assert out["x_prev"].iloc[0] != 999.0


def test_future_only_source_yields_nan(monkeypatch):
    df = pd.DataFrame({"station_id": ["B1"], "sample_date": [pd.Timestamp("2023-06-10")]})
    src = pd.DataFrame({"lat": [36.0], "lon": [-122.0],
                        "date": pd.to_datetime(["2024-01-01"]), "value": [5.0]})  # all future
    monkeypatch.setattr(g, "_read_src", lambda key: src)
    cfg = dict(src="fake", time="date", value="value", loc=("lat", "lon"))
    out = g._causal_join(df, {"geo": _geo(), "reveal_lag_days": 2}, cfg, "x")
    assert np.isnan(out["x_prev"].iloc[0])


def test_missing_source_is_nan_not_crash(monkeypatch):
    df = pd.DataFrame({"station_id": ["B1"], "sample_date": [pd.Timestamp("2023-06-10")]})
    monkeypatch.setattr(g, "_read_src", lambda key: None)
    cfg = dict(src="absent", time="date", value="value", loc=("lat", "lon"))
    out = g._causal_join(df, {"geo": _geo(), "reveal_lag_days": 2}, cfg, "x")
    assert out["x_prev"].isna().all() and "x_roll7" in out.columns
