"""Tests for leave-one-county-out spatial generalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import spatial_holdout as sh


def _tiny_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=40, early_stopping=False, random_state=0)


def _multi_county_obs(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", "2023-06-01", freq="7D")
    # three counties, each with a dirty and a clean station -> events in every era
    stations = [
        ("a_dirty", "Alpha", 0.5), ("a_clean", "Alpha", 0.05),
        ("b_dirty", "Beta", 0.4), ("b_clean", "Beta", 0.05),
        ("c_dirty", "Gamma", 0.45), ("c_clean", "Gamma", 0.05),
    ]
    rows = []
    for sid, county, p in stations:
        for d in dates:
            exceed = rng.random() < p
            vals = {"Enterococcus": 300.0 if exceed else float(rng.integers(1, 90)),
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


def test_loco_runs_and_holds_out_each_county(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = sh.run_loco(_multi_county_obs(tmp_path), clf_factory=_tiny_clf, min_test_events=5)
    counties = {r["county"] for r in res["per_county"]}
    assert counties <= {"Alpha", "Beta", "Gamma"} and len(counties) >= 2
    a = res["aggregate"]
    assert a["n_counties"] == len(res["per_county"])
    assert a["median_model_ap"] is None or 0.0 <= a["median_model_ap"] <= 1.0
    # each held-out county's verdict fields exist and are typed
    for r in res["per_county"]:
        assert isinstance(r["deploy_ready"], bool)
        assert r["model_beats_memory"] in (True, False, None)
        assert "model_calibrated" in r["models"]


def test_sw_prev7_excludes_held_out_county():
    # County C always exceeds, D never. The statewide-prior-7d feature excluding C must
    # be driven only by D (-> low), and excluding D only by C (-> high). If C's labels
    # leaked into the excluded aggregate, the two would not separate.
    dates = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"])
    rows = ([{"sample_date": d, "county": "C", "exceed": 1} for d in dates] +
            [{"sample_date": d, "county": "D", "exceed": 0} for d in dates])
    df = pd.DataFrame(rows)
    sw_excl_C = sh._sw_prev7_excluding(df, "C")
    sw_excl_D = sh._sw_prev7_excluding(df, "D")
    # day 3 carries a defined prior-7d value; excluding C (dirty) must be < excluding D.
    last = df["sample_date"] == "2022-01-03"
    assert sw_excl_C[last].iloc[0] < sw_excl_D[last].iloc[0]
    assert sw_excl_C[last].iloc[0] == 0.0  # only D (clean) remains
    assert sw_excl_D[last].iloc[0] == 1.0  # only C (dirty) remains


def test_loco_markdown_ascii_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = sh.run_loco(_multi_county_obs(tmp_path), clf_factory=_tiny_clf, min_test_events=5)
    sh.to_markdown(res).encode("cp1252")  # must not raise on a Windows console
