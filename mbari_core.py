"""Shared lightweight MBARI forecasting utilities.

This module is intentionally free of BigQuery, SciPy, and neural-model imports so
forecasting, normalization, and tests can import core constants and filters in a
clean environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# GCP project is environment-configured (set MBARI_GCP_PROJECT) so no specific
# cloud project is baked into the published source.
PROJECT_ID = os.environ.get("MBARI_GCP_PROJECT", "your-gcp-project")
DATASET_ID = "blue_current_raw"
TABLE_ID = "mbari_m1_realtime"

RAW_COLUMNS = [
    "source",
    "dataset_id",
    "station",
    "time",
    "latitude",
    "longitude",
    "z",
    "air_pressure",
    "relative_humidity",
    "air_temperature",
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "wind_speed_sonic",
    "wind_from_direction_sonic",
    "wind_speed_windbird",
    "wind_from_direction_windbird",
    "air_pressure_qc_agg",
    "relative_humidity_qc_agg",
    "air_temperature_qc_agg",
    "sea_water_temperature_qc_agg",
    "sea_water_practical_salinity_qc_agg",
    "eastward_sea_water_velocity_qc_agg",
    "northward_sea_water_velocity_qc_agg",
    "wind_speed_sonic_qc_agg",
    "wind_from_direction_sonic_qc_agg",
    "ingested_at",
]

MEASURE_COLUMNS = [
    "latitude",
    "longitude",
    "z",
    "air_pressure",
    "relative_humidity",
    "air_temperature",
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "wind_speed_sonic",
    "wind_from_direction_sonic",
    "wind_speed_windbird",
    "wind_from_direction_windbird",
]

TARGETS = [
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "air_temperature",
    "relative_humidity",
    "air_pressure",
]

PHYSICAL_RANGES = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "z": (-11000.0, 100.0),
    "air_pressure": (870.0, 1085.0),
    "relative_humidity": (0.0, 100.0),
    "air_temperature": (-10.0, 45.0),
    "sea_water_temperature": (-3.0, 35.0),
    "sea_water_practical_salinity": (20.0, 40.0),
    "eastward_sea_water_velocity": (-5.0, 5.0),
    "northward_sea_water_velocity": (-5.0, 5.0),
    "wind_speed_sonic": (0.0, 60.0),
    "wind_from_direction_sonic": (0.0, 360.0),
    "wind_speed_windbird": (0.0, 60.0),
    "wind_from_direction_windbird": (0.0, 360.0),
}


@dataclass
class ComputeStatus:
    backend: str
    gpu_name: str | None
    cuda_available: bool
    cuda_kernel_ok: bool
    note: str


def detect_compute() -> ComputeStatus:
    try:
        import torch
    except Exception:
        return ComputeStatus("numpy/sklearn-cpu", None, False, False, "PyTorch is not importable.")

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return ComputeStatus("numpy/sklearn-cpu", None, False, False, "CUDA is not available to PyTorch.")

    gpu_name = torch.cuda.get_device_name(0)
    try:
        x = torch.randn(512, 512, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        _ = float(y[0, 0].detach().cpu())
        return ComputeStatus("torch-cuda+sklearn-cpu", gpu_name, True, True, "CUDA tensor kernels are usable.")
    except Exception as exc:
        return ComputeStatus(
            "numpy/sklearn-cpu",
            gpu_name,
            True,
            False,
            f"CUDA is visible but unusable for PyTorch kernels: {type(exc).__name__}: {exc}",
        )


def read_gbq(query: str) -> pd.DataFrame:
    try:
        import pandas_gbq
    except Exception as exc:
        raise RuntimeError("BigQuery source requires the optional 'geo' extra: pandas-gbq.") from exc
    return pandas_gbq.read_gbq(query, project_id=PROJECT_ID, progress_bar_type=None)


def load_mbari_data(limit: int | None = None) -> pd.DataFrame:
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        WITH latest AS (
          SELECT
            {", ".join(RAW_COLUMNS)},
            ROW_NUMBER() OVER (
              PARTITION BY time, FORMAT("%0.6f", z)
              ORDER BY ingested_at DESC
            ) AS rn
          FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
          WHERE time IS NOT NULL
        )
        SELECT {", ".join(RAW_COLUMNS)}
        FROM latest
        WHERE rn = 1
        ORDER BY time, z
        {limit_clause}
    """
    df = read_gbq(query)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    if "z" in out:
        out["depth_m"] = out["z"].abs()
    out["hour"] = out["time"].dt.hour.astype(float)
    out["day_of_year"] = out["time"].dt.dayofyear.astype(float)
    out["days_since_start"] = (out["time"] - out["time"].min()).dt.total_seconds() / 86400.0
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)
    if {"eastward_sea_water_velocity", "northward_sea_water_velocity"}.issubset(out.columns):
        out["current_speed"] = np.sqrt(
            out["eastward_sea_water_velocity"].pow(2) + out["northward_sea_water_velocity"].pow(2)
        )
    if {"wind_speed_sonic", "wind_from_direction_sonic"}.issubset(out.columns):
        out["wind_u_sonic"] = out["wind_speed_sonic"] * np.sin(np.deg2rad(out["wind_from_direction_sonic"]))
        out["wind_v_sonic"] = out["wind_speed_sonic"] * np.cos(np.deg2rad(out["wind_from_direction_sonic"]))
    return out


def apply_physical_quality_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows: list[dict[str, Any]] = []
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        present = values.notna()
        invalid = present & ((values < lo) | (values > hi) | ~np.isfinite(values))
        rows.append(
            {
                "variable": col,
                "non_null_before": int(present.sum()),
                "invalidated": int(invalid.sum()),
                "valid_after": int((present & ~invalid).sum()),
                "min_allowed": lo,
                "max_allowed": hi,
            }
        )
        out.loc[invalid, col] = np.nan
    return out, pd.DataFrame(rows)
