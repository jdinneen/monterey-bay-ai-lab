#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbal_forecast_v2 import build_hourly_matrix  # noqa: E402


def test_build_hourly_matrix_preserves_m1_names_and_prefixes_other_stations():
    df = pd.DataFrame(
        [
            {
                "station": "M1",
                "time": "2026-01-01T00:00:00Z",
                "z": -10.0,
                "sea_water_temperature": 10.0,
                "sea_water_practical_salinity": 33.0,
                "air_pressure": 1010.0,
            },
            {
                "station": "M2",
                "time": "2026-01-01T00:00:00Z",
                "z": -10.0,
                "sea_water_temperature": 11.0,
                "sea_water_practical_salinity": 34.0,
                "air_pressure": 1011.0,
            },
        ]
    )

    df["time"] = pd.to_datetime(df["time"], utc=True)

    matrix, coverage = build_hourly_matrix(df)

    assert matrix.loc[pd.Timestamp("2026-01-01T00:00:00Z"), "temp_d10p0"] == 10.0
    assert matrix.loc[pd.Timestamp("2026-01-01T00:00:00Z"), "M2_temp_d10p0"] == 11.0
    assert "air_pressure" in matrix
    assert "M2_air_pressure" in matrix
    assert set(coverage["series"]) >= {"temp_d10p0", "M2_temp_d10p0"}


def test_build_hourly_matrix_defaults_missing_station_to_m1_names():
    df = pd.DataFrame(
        [
            {
                "time": "2026-01-01T00:00:00Z",
                "z": -1.0,
                "sea_water_temperature": 10.0,
            }
        ]
    )

    df["time"] = pd.to_datetime(df["time"], utc=True)

    matrix, _ = build_hourly_matrix(df)

    assert "temp_d1p0" in matrix
    assert "M1_temp_d1p0" not in matrix
