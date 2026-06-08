import numpy as np
import pandas as pd

from mbari_drivers_build import (
    _cap_hist_staleness,
    _clean_wave_series,
    _hist_quality,
    _load_daily_available,
    _load_hourly,
)


def test_load_hourly_delays_subhour_observations_until_available(tmp_path):
    path = tmp_path / "hourly.parquet"
    source = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.to_datetime(["2026-01-01T00:30:00Z", "2026-01-01T02:00:00Z"]),
    )
    source.to_parquet(path)
    grid = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")

    loaded = _load_hourly(path, grid, availability_lag_hours=1)

    assert pd.isna(loaded.loc[pd.Timestamp("2026-01-01T01:00:00Z"), "x"])
    assert loaded.loc[pd.Timestamp("2026-01-01T02:00:00Z"), "x"] == 1.0
    assert loaded.loc[pd.Timestamp("2026-01-01T03:00:00Z"), "x"] == 2.0


def test_clean_wave_series_bounds_periods_and_encodes_direction():
    ndbc = pd.DataFrame(
        {
            "ndbc46042_wave_height_m": [-1.0, 2.5, 31.0],
            "ndbc46042_dom_wave_period_s": [0.0, 12.0, 41.0],
            "ndbc46042_avg_wave_period_s": [8.0, 0.0, 20.0],
            "ndbc46042_mean_wave_dir_deg": [0.0, 90.0, 361.0],
        }
    )

    out = _clean_wave_series(ndbc)

    assert pd.isna(out["ndbc46042_wave_height_m"].iloc[0])
    assert out["ndbc46042_wave_height_m"].iloc[1] == 2.5
    assert pd.isna(out["ndbc46042_wave_height_m"].iloc[2])
    assert pd.isna(out["ndbc46042_dom_wave_period_s"].iloc[0])
    assert out["ndbc46042_dom_wave_period_s"].iloc[1] == 12.0
    assert pd.isna(out["ndbc46042_dom_wave_period_s"].iloc[2])
    assert np.isclose(out["ndbc46042_mean_wave_dir_sin"].iloc[1], 1.0)
    assert np.isclose(out["ndbc46042_mean_wave_dir_cos"].iloc[0], 1.0)
    assert pd.isna(out["ndbc46042_mean_wave_dir_sin"].iloc[2])


def test_hist_quality_separates_raw_coverage_from_forward_filled_input():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h")
    raw = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0]}, index=idx)
    filled = raw.ffill()

    quality = _hist_quality(raw, filled)

    assert quality["raw_coverage"]["x"] == 0.5
    assert quality["max_staleness_hours"]["x"] == 2.0


def test_daily_observed_product_is_delayed_until_next_day():
    daily = pd.DataFrame(
        {"cuti_37n": [5.0, 6.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
    )
    grid = pd.date_range("2026-01-01T00:00:00Z", periods=49, freq="1h")

    loaded = _load_daily_available(daily, grid, availability_lag_days=1)
    filled = loaded.ffill()

    assert pd.isna(filled.loc[pd.Timestamp("2026-01-01T23:00:00Z"), "cuti_37n"])
    assert filled.loc[pd.Timestamp("2026-01-02T00:00:00Z"), "cuti_37n"] == 5.0
    assert filled.loc[pd.Timestamp("2026-01-03T00:00:00Z"), "cuti_37n"] == 6.0


def test_hist_staleness_cap_masks_overaged_forward_fills():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")
    raw = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, np.nan]}, index=idx)

    capped = _cap_hist_staleness(raw, raw.ffill(), max_age_hours=2)
    quality = _hist_quality(raw, capped)

    assert capped.loc[pd.Timestamp("2026-01-01T02:00:00Z"), "x"] == 1.0
    assert pd.isna(capped.loc[pd.Timestamp("2026-01-01T03:00:00Z"), "x"])
    assert quality["max_staleness_hours"]["x"] == 2.0
