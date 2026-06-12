"""Tests for the static physical spatial covariates (gauge distance, monitoring density)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import physical_spatial_covariates as pc


def _frame():
    # three beaches: A,B ~1 km apart; C far away (~100 km)
    df = pd.DataFrame({
        "station_id": ["A", "A", "B", "C"],
        "sample_date": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-01", "2022-01-01"]),
        "exceed": [0, 1, 0, 1],
    })
    geo = pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "latitude": [36.60, 36.61, 37.50],
        "longitude": [-121.90, -121.90, -122.50],
    })
    return df, geo


def test_physical_covariate_columns_and_density(tmp_path):
    pytest.importorskip("sklearn")
    df, geo = _frame()
    # no gauge map supplied -> dist_to_gauge_km is all-NaN, density still computed
    out = pc.add_physical_covariates(
        df,
        geo,
        gauge_map_path=tmp_path / "missing.parquet",
        station_static_path=tmp_path / "missing_static.parquet",
    )
    for col in pc.PHYS_FEATS:
        assert col in out.columns
    # A and B are ~1 km apart, C is far: within 10 km each of A,B sees exactly the other (1)
    a = out[out["station_id"] == "A"]["stn_density_10km"].iloc[0]
    c = out[out["station_id"] == "C"]["stn_density_10km"].iloc[0]
    assert a == 1
    assert c == 0
    assert out["dist_to_gauge_km"].isna().all()


def test_gauge_distance_merged(tmp_path):
    pytest.importorskip("sklearn")
    df, geo = _frame()
    gm = tmp_path / "station_gauge_map.parquet"
    pd.DataFrame({"station_id": ["A", "B"], "gauge_id": ["g1", "g2"],
                  "distance_km": [5.0, 22.0]}).to_parquet(gm)
    out = pc.add_physical_covariates(
        df,
        geo,
        gauge_map_path=gm,
        station_static_path=tmp_path / "missing_static.parquet",
    )
    assert out[out["station_id"] == "A"]["dist_to_gauge_km"].iloc[0] == 5.0
    assert out[out["station_id"] == "C"]["dist_to_gauge_km"].isna().all()  # unmapped -> NaN


def test_station_static_table_takes_precedence(tmp_path):
    df, geo = _frame()
    static = tmp_path / "station_static.parquet"
    pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "latitude": [36.60, 36.61, 37.50],
        "longitude": [-121.90, -121.90, -122.50],
        "station_density_10km": [7, 8, 9],
        "station_density_25km": [10, 11, 12],
        "dist_to_usgs_gauge_km": [1.0, 2.0, 3.0],
        "dist_to_npdes_km": [4.0, 5.0, 6.0],
        "npdes_count_5km": [1, 0, 2],
        "npdes_count_25km": [3, 4, 5],
        "dist_to_sso_km": [7.0, 8.0, 9.0],
        "sso_count_5km": [0, 1, 2],
        "sso_count_25km": [2, 3, 4],
        "ccap_developed": [True, False, True],
    }).to_parquet(static)

    out = pc.add_physical_covariates(df, geo, station_static_path=static)

    assert out[out["station_id"] == "A"]["dist_to_gauge_km"].iloc[0] == 1.0
    assert out[out["station_id"] == "A"]["stn_density_10km"].iloc[0] == 7
    assert out[out["station_id"] == "C"]["sso_count_25km"].iloc[0] == 4
    assert out[out["station_id"] == "B"]["ccap_developed"].iloc[0] == 0.0
