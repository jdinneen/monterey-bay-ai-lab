"""Tests for the k-nearest-beach spatial-lag driver (research/bacteria/spatial_drivers_experiment.py).

The key property is causality: a neighbour's exceedance state may only enter a target beach's
feature once it has been REVEALED (sample date <= target date - reveal_lag_days)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import spatial_drivers_experiment as sde


def _two_station_frame(flip_date: str = "2021-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Station B is clean (exceed=0) before flip_date and dirty (exceed=1) after; station A is
    its only neighbour. Lets us check exactly when B's state reaches A's neighbour feature."""
    dates = pd.date_range("2020-06-01", "2021-06-01", freq="7D")
    rows = []
    for d in dates:
        rows.append({"station_id": "A", "sample_date": d, "exceed": 0})
        rows.append({"station_id": "B", "sample_date": d,
                     "exceed": int(pd.Timestamp(d) >= pd.Timestamp(flip_date))})
    df = pd.DataFrame(rows)
    geo = pd.DataFrame({
        "station_id": ["A", "B"],
        "latitude": [36.60, 36.61],   # ~1 km apart -> mutual nearest neighbour
        "longitude": [-121.90, -121.90],
    })
    return df, geo


def test_knn_spatial_lag_is_causal_with_reveal_lag():
    pytest.importorskip("sklearn")
    flip = pd.Timestamp("2021-01-01")
    lag = 2
    df, geo = _two_station_frame(str(flip.date()))
    out = sde.add_knn_spatial_lag(df, geo, k=1, reveal_lag_days=lag)

    a = out[out["station_id"] == "A"].sort_values("sample_date")
    # well before the flip, A's neighbour (B) state must be 0
    before = a[a["sample_date"] <= flip - pd.Timedelta(days=10)]
    assert np.nanmax(before["nbr_prev1"].to_numpy()) == 0.0
    # B's first dirty sample is on `flip`; it must not be visible to A until flip + lag days
    not_yet = a[(a["sample_date"] >= flip) & (a["sample_date"] < flip + pd.Timedelta(days=lag))]
    if len(not_yet):
        assert np.nanmax(np.nan_to_num(not_yet["nbr_prev1"].to_numpy())) == 0.0
    # well after the flip + lag, B's dirty state has been revealed -> reaches A
    after = a[a["sample_date"] >= flip + pd.Timedelta(days=lag + 14)]
    assert np.nanmax(after["nbr_prev1"].to_numpy()) == 1.0


def test_knn_spatial_lag_columns_present_and_bounded():
    pytest.importorskip("sklearn")
    df, geo = _two_station_frame()
    out = sde.add_knn_spatial_lag(df, geo, k=1, reveal_lag_days=0)
    for col in sde.SPATIAL_FEATS:
        assert col in out.columns
        v = out[col].to_numpy()
        v = v[~np.isnan(v)]
        assert ((v >= 0.0) & (v <= 1.0)).all()  # neighbour exceedance rate is a probability


def test_coordinate_merge_suffixes_are_normalized():
    df, geo = _two_station_frame()
    with_coords = df.merge(geo, on="station_id", how="left")
    merged_again = with_coords.merge(geo, on="station_id", how="left")

    out = sde.ob._ensure_optional_feature_columns(merged_again)

    assert "latitude" in out.columns
    assert "longitude" in out.columns
    assert "latitude_x" not in out.columns
    assert "longitude_y" not in out.columns
    assert out["latitude"].notna().all()


def test_spatial_feature_lists_are_unique_when_base_already_has_optional_features():
    feats = ["station_prior_rate", "nbr_prev1", "nbr_prev7", "latitude", "longitude"]

    out = sde._unique_features(feats + sde.SPATIAL_FEATS + ["latitude", "longitude"])

    assert out == ["station_prior_rate", "nbr_prev1", "nbr_prev7", "latitude", "longitude"]
    assert sde._unique_features(feats + ["latitude", "longitude"]) == feats
