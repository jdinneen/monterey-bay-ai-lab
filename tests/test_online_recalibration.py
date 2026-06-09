"""Tests for era-local (prequential) recalibration — including the causality guardrail
the reviewer insisted on: the calibrator at a block must use no future/unrevealed label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import online_recalibration as orc


def _tiny_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=40, early_stopping=False, random_state=0)


def test_revealed_mask_excludes_unrevealed_and_lagged():
    dates = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-10", "2022-01-31", "2022-02-05"]))
    mask = orc._revealed_mask(dates, pd.Timestamp("2022-02-01"), lag_days=2)  # cutoff 2022-01-30
    assert list(mask) == [True, True, False, False]  # 01-31 (within lag) and 02-05 (future) excluded


def test_prequential_cannot_use_future_labels():
    """Flipping the LAST month's labels must not change the FIRST month's recalibrated
    probabilities — proves no future label leaks into a past block's calibrator."""
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    dates = pd.to_datetime(pd.date_range("2022-01-01", "2022-04-28", freq="D"))
    raw = rng.random(len(dates))
    y = (rng.random(len(dates)) < raw).astype(int)
    static = IsotonicRegression(out_of_bounds="clip").fit(raw, y)

    p1 = orc.prequential_recalibrate(dates, raw, y, static, lag_days=2, block_freq="M", min_calib=5)
    y2 = y.copy()
    last_month = np.asarray(dates >= "2022-04-01")
    y2[last_month] = 1 - y2[last_month]  # corrupt only the most recent (still-arriving) labels
    p2 = orc.prequential_recalibrate(dates, raw, y2, static, lag_days=2, block_freq="M", min_calib=5)

    first_month = np.asarray(dates < "2022-02-01")
    assert np.allclose(p1[first_month], p2[first_month])  # past predictions unchanged


def test_prequential_cold_start_falls_back_to_static():
    from sklearn.isotonic import IsotonicRegression

    dates = pd.to_datetime(pd.date_range("2022-01-01", "2022-01-31", freq="D"))
    raw = np.linspace(0, 1, len(dates))
    y = (raw > 0.5).astype(int)
    static = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    # min_calib huge -> never enough revealed labels -> output is exactly the static map
    out = orc.prequential_recalibrate(dates, raw, y, static, lag_days=2, min_calib=10_000)
    assert np.allclose(out, static.predict(raw))


def _multi_county_obs(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", "2023-06-01", freq="7D")
    stations = [("sf_d", "San Francisco", 0.4), ("sf_c", "San Francisco", 0.05),
                ("a_d", "Alpha", 0.4), ("a_c", "Alpha", 0.05),
                ("b_d", "Beta", 0.4), ("b_c", "Beta", 0.05)]
    rows = []
    for sid, county, p in stations:
        for d in dates:
            ex = rng.random() < p
            vals = {"Enterococcus": 300.0 if ex else float(rng.integers(1, 90)),
                    "Fecal Coliforms": float(rng.integers(1, 350)),
                    "Total Coliforms": float(rng.integers(10, 9000)),
                    "E. Coli": float(rng.integers(1, 200))}
            for param, val in vals.items():
                rows.append({"sample_date": d, "county": county, "beach_name": sid,
                             "station_name": sid, "station_id": sid, "source_parameter": param,
                             "property_id": "p", "result_comparator": "=", "result_value_numeric": val})
    path = tmp_path / "statewide_beach_observations.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_run_online_recal_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("MBARI_BACTERIA_OBS", raising=False)
    res = orc.run_online_recal(_multi_county_obs(tmp_path), clf=_tiny_clf(), min_calib=20)
    assert "ALL" in res["strata"]
    al = res["strata"]["ALL"]
    if "static" in al:  # populated stratum
        assert 0.0 <= al["online"]["ece"] <= 1.0
        assert isinstance(al["online_deploy_ready"], bool)
        assert isinstance(al["online_fixes_calibration"], bool)
