"""Tests for the operational nowcast scorer (research/bacteria/predict_current.py).

Exercises the tiering logic and a synthetic end-to-end run that checks the leakage-safe
train/calibrate/score split, the San-Diego deploy-ready exclusion, valid probabilities, and
ASCII-safe artifacts — without the 1.3M-row real dataset.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.bacteria import predict_current as pc


def _synthetic_obs(tmp_path):
    """Weekly samples 2017-2023, dirty/clean stations across San Diego + Monterey Bay,
    all four analytes so no feature column is all-NaN (the HGBT binner rejects that)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", "2023-06-01", freq="7D")
    stations = [
        ("sd_dirty", "San Diego", 0.6), ("sd_clean", "San Diego", 0.05),
        ("mb_dirty", "Monterey", 0.4), ("mb_clean", "Monterey", 0.05),
        ("sc_mid", "Santa Cruz", 0.2), ("or_clean", "Orange", 0.08),
    ]
    rows = []
    for sid, county, p in stations:
        for d in dates:
            exceed = rng.random() < p
            vals = {
                "Enterococcus": 300.0 if exceed else float(rng.integers(1, 90)),
                "Fecal Coliforms": float(rng.integers(1, 350)),
                "Total Coliforms": float(rng.integers(10, 9000)),
                "E. Coli": float(rng.integers(1, 200)),
            }
            for param, val in vals.items():
                rows.append({
                    "sample_date": d, "county": county, "beach_name": sid, "station_name": sid,
                    "station_id": sid, "source_parameter": param,
                    "property_id": "prop", "result_comparator": "=", "result_value_numeric": val,
                })
    path = tmp_path / "statewide_beach_observations.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_tier_thresholds():
    assert pc._tier(0.9) == "HIGH"
    assert pc._tier(0.50) == "HIGH"
    assert pc._tier(0.49) == "ELEVATED"
    assert pc._tier(0.20) == "ELEVATED"
    assert pc._tier(0.05) == "WATCH"
    assert pc._tier(0.04) == "LOW"
    assert pc._tier(0.0) == "LOW"


def test_build_scored_frame_split_and_probs(tmp_path):
    obs = _synthetic_obs(tmp_path)
    sc, meta = pc.build_scored_frame(obs, rain_dir=None)
    # one scored row per station (the latest read in the score window)
    assert meta["n_scored_beaches"] == len(sc) == sc["station_id"].nunique()
    # leakage-safe ordering of the split boundaries
    assert meta["calib_start"] < meta["score_cut"] <= meta["max_sample_date"]
    assert meta["n_train"] > 0 and meta["n_calib"] > 0
    # calibrated probabilities are valid
    assert sc["p_cal"].between(0.0, 1.0).all()
    assert sc["p_raw"].between(0.0, 1.0).all()
    # dirty stations should out-rank clean ones
    by_station = sc.set_index("station_id")["p_cal"]
    assert by_station["mb_dirty"] > by_station["mb_clean"]


def test_main_excludes_san_diego_and_writes_artifacts(tmp_path):
    obs = _synthetic_obs(tmp_path)
    out = tmp_path / "nowcast"
    rc = pc.main(["--obs", str(obs), "--out-dir", str(out)])
    assert rc == 0

    risk = pd.read_csv(out / "current_risk.csv")
    assert (risk["county"] != "San Diego").all()          # deploy-ready set excludes San Diego
    assert risk["p_cal"].between(0.0, 1.0).all()
    assert list(risk["p_cal"]) == sorted(risk["p_cal"], reverse=True)  # ranked desc

    meta = json.loads((out / "nowcast_meta.json").read_text(encoding="utf-8"))
    assert meta["n_san_diego_ranking_only"] >= 1           # SD scored but held separate
    assert meta["n_deploy_ready_beaches"] == len(risk)

    digest = (out / "digest.md").read_text(encoding="utf-8")
    digest.encode("ascii")                                  # artifact must be ASCII-safe on Windows
    assert "current exceedance risk" in digest
