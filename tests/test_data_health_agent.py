#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.data_health_agent import (  # noqa: E402
    check_data_inventory,
    check_driver_manifest,
    check_metric_duplicates,
    check_target_coverage,
    current_missing_run_hours,
    max_missing_run_hours,
)


def test_max_missing_run_hours_counts_longest_nan_streak():
    series = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan])

    assert max_missing_run_hours(series) == 3
    assert current_missing_run_hours(series) == 3


def test_driver_manifest_fails_when_staleness_exceeds_cap(tmp_path):
    cache = tmp_path / "nn_cache"
    cache.mkdir()
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h")
    pd.DataFrame({"wind": [1.0, 1.0, 1.0]}, index=idx).to_parquet(cache / "drivers_hourly.parquet")
    (cache / "drivers_manifest.json").write_text(
        json.dumps(
            {
                "hist": ["wind"],
                "coverage": {"wind": 1.0},
                "hist_raw_coverage": {"wind": 0.5},
                "hist_max_staleness_hours": {"wind": 200.0},
                "max_hist_ffill_hours": 168,
            }
        ),
        encoding="utf-8",
    )

    check = check_driver_manifest(tmp_path)

    assert check.status == "FAIL"
    assert "exceed max staleness cap" in check.summary


def test_driver_manifest_fails_when_manifest_column_missing_from_parquet(tmp_path):
    cache = tmp_path / "nn_cache"
    cache.mkdir()
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h")
    pd.DataFrame({"other": [1.0, 1.0, 1.0]}, index=idx).to_parquet(cache / "drivers_hourly.parquet")
    (cache / "drivers_manifest.json").write_text(
        json.dumps(
            {
                "hist": ["wind"],
                "coverage": {"wind": 1.0},
                "hist_raw_coverage": {"wind": 1.0},
                "hist_max_staleness_hours": {"wind": 0.0},
                "max_hist_ffill_hours": 168,
            }
        ),
        encoding="utf-8",
    )

    check = check_driver_manifest(tmp_path)

    assert check.status == "FAIL"
    assert "manifest driver columns are missing" in check.summary


def test_target_coverage_warns_on_current_stale_gaps(tmp_path):
    matrix_dir = tmp_path / "mbari_big_analysis_results"
    matrix_dir.mkdir()
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")
    data = {target: [1.0, 2.0, np.nan, np.nan, np.nan] for target in [
        "temp_d1p0",
        "temp_d10p0",
        "temp_d20p0",
        "temp_d100p0",
        "sal_d1p0",
        "sal_d10p0",
        "sal_d20p0",
        "sal_d100p0",
        "air_temperature",
        "air_pressure",
    ]}
    pd.DataFrame(data, index=idx).to_parquet(matrix_dir / "m1_hourly_matrix.parquet")

    check = check_target_coverage(tmp_path, warn_coverage=0.1, warn_gap_hours=2, warn_current_gap_hours=2)

    assert check.status == "WARN"
    assert "currently stale" in check.summary


def test_target_coverage_passes_when_only_historical_gaps_are_long(tmp_path):
    matrix_dir = tmp_path / "mbari_big_analysis_results"
    matrix_dir.mkdir()
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")
    data = {target: [1.0, np.nan, np.nan, np.nan, 2.0, 3.0] for target in [
        "temp_d1p0",
        "temp_d10p0",
        "temp_d20p0",
        "temp_d100p0",
        "sal_d1p0",
        "sal_d10p0",
        "sal_d20p0",
        "sal_d100p0",
        "air_temperature",
        "air_pressure",
    ]}
    pd.DataFrame(data, index=idx).to_parquet(matrix_dir / "m1_hourly_matrix.parquet")

    check = check_target_coverage(tmp_path, warn_coverage=0.1, warn_gap_hours=2, warn_current_gap_hours=2)

    assert check.status == "PASS"
    assert len(check.details["historical_long_gap_targets"]) == 10


def test_metric_duplicate_check_allows_matching_aggregate_and_partition(tmp_path):
    metrics_dir = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    run_dir = metrics_dir / "run_id=r"
    run_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {"run_id": "r", "split_id": "s", "unique_id": "temp", "horizon_h": 6, "model": "x", "loss": "mae"},
        ]
    )
    rows.to_parquet(metrics_dir / "metrics.parquet", index=False)
    rows.to_parquet(run_dir / "metrics.parquet", index=False)

    check = check_metric_duplicates(tmp_path)

    assert check.status == "PASS"
    assert check.details["duplicate_metric_identities"] == 0


def test_data_inventory_warns_when_optional_mur_cache_missing(tmp_path):
    for rel in [
        "mbari_history/opendap/m1_history.parquet",
        "mbari_history/opendap/m2_history.parquet",
        "mbari_history/noaa/noaa_ndbc46042.parquet",
        "mbari_history/noaa/noaa_coops.parquet",
        "mbari_history/noaa/noaa_upwelling.parquet",
        "mbari_history/noaa/noaa_drivers_daily.parquet",
        "nn_cache/drivers_hourly.parquet",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    check = check_data_inventory(tmp_path)

    assert check.status == "WARN"
    assert "MUR SST cache" in check.summary


def test_data_inventory_accepts_legacy_mur_year_cache_names(tmp_path):
    for rel in [
        "mbari_history/opendap/m1_history.parquet",
        "mbari_history/opendap/m2_history.parquet",
        "mbari_history/noaa/noaa_ndbc46042.parquet",
        "mbari_history/noaa/noaa_coops.parquet",
        "mbari_history/noaa/noaa_upwelling.parquet",
        "mbari_history/noaa/noaa_drivers_daily.parquet",
        "nn_cache/drivers_hourly.parquet",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    mur_cache = tmp_path / "mbari_history" / "noaa" / "mur_sst_cache"
    mur_cache.mkdir(parents=True)
    (mur_cache / "mur_2002.parquet").write_bytes(b"placeholder")

    check = check_data_inventory(tmp_path)

    assert check.status == "PASS"
    assert check.details["mur_sst_cache_files"] == 1
