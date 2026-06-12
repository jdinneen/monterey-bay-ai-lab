"""Tests for the operational beach-bacteria benchmark.

Covers the novel, bug-prone scoring helpers (ECE, recall@FPR, score) on small
deterministic arrays, plus a synthetic end-to-end run() that exercises the
leakage-safe features, region stratification, isotonic calibration, and the
operational/deploy-readiness verdict — without the 1.3M-row real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.bacteria import operational_benchmark as ob


def test_portable_obs_path_is_relative_and_leak_free():
    # The emitted obs_path must never be an absolute/personal path: an absolute
    # path leaks the local filesystem AND breaks byte-identical reproduction of
    # expected/ across machines. It must be a repo-relative POSIX string.
    from pathlib import Path

    portable = ob.portable_obs_path(ob.default_obs_path())
    assert portable == "bacteria_results/statewide/statewide_beach_observations.parquet"
    assert ":" not in portable and "\\" not in portable
    assert not portable.startswith("/")
    # A path outside the project root degrades to just the basename (still leak-free).
    assert ob.portable_obs_path(Path("/some/other/root/x.parquet")) == "x.parquet"


def test_recall_at_fpr_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # cleanly separable
    assert ob._recall_at_fpr(y, p, fpr_budget=0.20) == pytest.approx(1.0)


def test_ece_zero_when_perfectly_calibrated():
    # 100 points at p=0.0 all negative, 100 at p=1.0 all positive -> ECE 0.
    y = np.array([0] * 100 + [1] * 100)
    p = np.array([0.0] * 100 + [1.0] * 100)
    assert ob._expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_high_when_overconfident():
    # Predict 0.9 for everything but only 10% are positive -> large gap.
    y = np.array([1] * 10 + [0] * 90)
    p = np.full(100, 0.9)
    assert ob._expected_calibration_error(y, p) > 0.5


def test_score_perfect_ranker_has_auc_one_and_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = ob.score(y, p, base=0.5)
    assert r["roc_auc"] == pytest.approx(1.0)
    for k in ("ap", "recall_at_20pct_fpr", "brier", "ece", "precision_at_10pct"):
        assert k in r


def test_score_degenerate_single_class_is_flagged_not_crash():
    y = np.zeros(20, dtype=int)
    r = ob.score(y, np.linspace(0, 1, 20), base=0.1)
    assert "note" in r and "ap" not in r  # ranking metrics undefined, no crash


def _synthetic_obs(tmp_path):
    """Weekly samples 2017-2023, dirty vs clean stations in two regions, all four
    analytes (so no feature column is all-NaN, which the HGBT binner rejects)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", "2023-06-01", freq="7D")
    stations = [
        ("sd_dirty", "San Diego", 0.6), ("sd_mid", "San Diego", 0.3), ("sd_clean", "San Diego", 0.05),
        ("mb_dirty", "Monterey", 0.4), ("mb_clean", "Monterey", 0.05), ("sc_clean", "Santa Cruz", 0.08),
    ]
    rows = []
    for sid, county, p in stations:
        for d in dates:
            exceed = rng.random() < p  # enterococcus drives the exceedance label (>104)
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
    obs = pd.DataFrame(rows)
    path = tmp_path / "statewide_beach_observations.parquet"
    obs.to_parquet(path, index=False)
    return path


def _tiny_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=40, early_stopping=False, random_state=0)


def test_run_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = _synthetic_obs(tmp_path)
    res = ob.run(obs, clf=_tiny_clf())
    assert set(res["strata"]) == {"ALL", "EXCLUDE_SAN_DIEGO", "SAN_DIEGO_ONLY", "MONTEREY"}
    assert "NOT EVALUATED" in res["ab411_rain_rule"]  # rain gap reported honestly, never faked
    all_s = res["strata"]["ALL"]
    assert {"baseline_prior_lab", "baseline_station_memory", "model_hgbt",
            "model_hgbt_calibrated"} <= set(all_s["models"])
    v = all_s["operational_verdict"]
    assert v is not None
    assert isinstance(v["model_beats_operational_ranking"], bool)
    assert isinstance(v["calibrated_deploy_ready"], bool)
    # ECE is a valid [0,1] quantity for the calibrated model in a populated stratum.
    assert 0.0 <= all_s["models"]["model_hgbt_calibrated"]["ece"] <= 1.0


def test_markdown_is_ascii_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = ob.run(_synthetic_obs(tmp_path), clf=_tiny_clf())
    md = ob.to_markdown(res)
    md.encode("cp1252")  # must not raise - CLI prints to a Windows console


def test_reveal_lag_skips_unreturned_label_and_lag0_is_legacy():
    df = pd.DataFrame({
        "station_id": ["s"] * 4, "county": ["A"] * 4,
        "sample_date": pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-09", "2022-01-16"]),
        "ent": [300.0, 10, 300, 10], "fec": [10.0] * 4, "tot": [10.0] * 4, "ecoli": [10.0] * 4,
        "exceed": [1, 0, 1, 0]})
    f0 = ob.add_causal_features(df, reveal_lag_days=0)
    f2 = ob.add_causal_features(df, reveal_lag_days=2)
    at = lambda f, d: f.loc[f["sample_date"] == d, "exc_prev"].item()
    # 2022-01-09 is 1 day after 01-08: lag0 uses 01-08 (exceed 0); lag2 must skip the
    # not-yet-returned label and use 01-01 (exceed 1).
    assert at(f0, "2022-01-09") == 0
    assert at(f2, "2022-01-09") == 1
    # lag0 reproduces legacy shift(1) exactly (first row NaN, then 1,0,1).
    assert f0.sort_values("sample_date")["exc_prev"].fillna(-1).tolist() == [-1, 1, 0, 1]


def test_available_at_features_skip_unreturned_station_and_county_labels():
    base = pd.Timestamp("2022-01-01")
    df = pd.DataFrame([
        {"station_id": "s", "county": "A", "sample_date": base,
         "ent": 300.0, "fec": 10.0, "tot": 10.0, "ecoli": 10.0, "exceed": 1,
         "available_at": base + pd.Timedelta(days=2)},
        {"station_id": "s", "county": "A", "sample_date": base + pd.Timedelta(days=7),
         "ent": 10.0, "fec": 10.0, "tot": 10.0, "ecoli": 10.0, "exceed": 0,
         "available_at": base + pd.Timedelta(days=27)},
        {"station_id": "s", "county": "A", "sample_date": base + pd.Timedelta(days=14),
         "ent": 10.0, "fec": 10.0, "tot": 10.0, "ecoli": 10.0, "exceed": 0,
         "available_at": base + pd.Timedelta(days=16)},
    ])
    feat = ob.add_causal_features_available(df)
    row = feat[feat["sample_date"] == base + pd.Timedelta(days=14)].iloc[0]
    assert row["exc_prev"] == 1.0
    assert row["cty_prev7"] != 0.0


def test_run_timed_available_at_synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = pd.read_parquet(_synthetic_obs(tmp_path))
    rows = []
    for i, r in obs.iterrows():
        analyte = {
            "Enterococcus": "prop_enterococcus",
            "Fecal Coliforms": "prop_fecal_coliform",
            "Total Coliforms": "prop_total_coliform",
            "E. Coli": "prop_e_coli",
        }[r["source_parameter"]]
        rows.append({
            "record_status": "accepted",
            "analyte": analyte,
            "value": r["result_value_numeric"],
            "sampledate": pd.Timestamp(r["sample_date"]).strftime("%Y-%m-%d"),
            "available_at": pd.Timestamp(r["sample_date"]) + pd.Timedelta(days=2),
            "station_id": r["station_id"],
            "county": r["county"],
        })
    timed = tmp_path / "timed.parquet"
    pd.DataFrame(rows).to_parquet(timed, index=False)
    res = ob.run(timed, clf=_tiny_clf(), label="enterococcus", availability_mode="available_at")
    assert res["availability_mode"] == "available_at"
    assert res["strata"]["ALL"]["operational_verdict"] is not None


def test_timed_loader_bridges_raw_station_id_to_research_station_id(tmp_path, monkeypatch):
    obs = pd.DataFrame(
        [{
            "sample_date": "2022-01-01",
            "county": "Monterey",
            "beach_name": "Beach",
            "station_name": "Station",
            "station_id": "hashed_station",
            "source_parameter": "Enterococcus",
            "property_id": "prop_enterococcus",
            "result_comparator": "=",
            "result_value_numeric": 120.0,
        }]
    )
    obs_path = tmp_path / "obs.parquet"
    obs.to_parquet(obs_path, index=False)
    monkeypatch.setenv("MBAL_BACTERIA_OBS", str(obs_path))
    timed = pd.DataFrame(
        [{
            "record_status": "accepted",
            "analyte": "prop_enterococcus",
            "value": 120.0,
            "sampledate": "2022-01-01",
            "available_at": "2022-01-03",
            "station_id": "raw_1",
            "county": "Monterey",
            "beach_name": "Beach",
            "station_name": "Station",
        }]
    )
    timed_path = tmp_path / "timed.parquet"
    timed.to_parquet(timed_path, index=False)

    out = ob.load_timed_station_days(timed_path, label="enterococcus")

    assert out.loc[0, "station_id"] == "hashed_station"


def test_days_since_wet_first_flush_logic():
    out = ob._days_since_wet(np.array([0.0, 5.0, 0.0, 0.0, 3.0, 0.0]), thr=2.54)
    assert np.isnan(out[0])  # before the first wet day -> undefined
    assert list(out[1:]) == [0.0, 1.0, 2.0, 0.0, 1.0]


def _synthetic_rain(tmp_path):
    rain_dir = tmp_path / "rainfall"
    rain_dir.mkdir()
    dates = pd.date_range("2016-12-01", "2023-06-10", freq="D")
    cells = [(34.0, -118.5), (36.6, -121.9)]
    rng = np.random.default_rng(1)
    grid = pd.concat([
        pd.DataFrame({"grid_lat": la, "grid_lon": lo, "date": dates,
                      "precip_mm": rng.gamma(0.3, 4.0, len(dates)) * (rng.random(len(dates)) < 0.25)})
        for la, lo in cells
    ], ignore_index=True)
    grid.to_parquet(rain_dir / "rainfall_grid.parquet", index=False)
    stations = ["sd_dirty", "sd_mid", "sd_clean", "mb_dirty", "mb_clean", "sc_clean"]
    pd.DataFrame({
        "station_id": stations,
        "grid_lat": [cells[i % 2][0] for i in range(len(stations))],
        "grid_lon": [cells[i % 2][1] for i in range(len(stations))],
    }).to_parquet(rain_dir / "station_grid_map.parquet", index=False)
    return rain_dir


def test_run_with_rainfall_adds_ab411_baseline(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = ob.run(_synthetic_obs(tmp_path), clf=_tiny_clf(), rain_dir=_synthetic_rain(tmp_path))
    assert "EVALUATED" in res["ab411_rain_rule"]  # rain path engaged
    all_s = res["strata"]["ALL"]
    assert "baseline_ab411_rain" in all_s["models"]
    v = all_s["operational_verdict"]
    assert isinstance(v["model_beats_ab411"], bool)  # AB411 present -> a real comparison
    assert 0.0 <= all_s["models"]["baseline_ab411_rain"]["ece"] <= 1.0


def _synthetic_discharge(tmp_path):
    ddir = tmp_path / "discharge"
    ddir.mkdir()
    dates = pd.date_range("2016-12-01", "2023-06-10", freq="D")
    rng = np.random.default_rng(2)
    gauges = ["g1", "g2"]
    pd.concat([
        pd.DataFrame({"gauge_id": g, "date": dates, "discharge_cfs": rng.gamma(2.0, 20.0, len(dates))})
        for g in gauges
    ], ignore_index=True).to_parquet(ddir / "discharge_gauge.parquet", index=False)
    stations = ["sd_dirty", "sd_mid", "sd_clean", "mb_dirty", "mb_clean", "sc_clean"]
    pd.DataFrame({"station_id": stations,
                  "gauge_id": [gauges[i % 2] for i in range(len(stations))],
                  "distance_km": [5.0] * len(stations)}).to_parquet(ddir / "station_gauge_map.parquet", index=False)
    return ddir


def test_enterococcus_label_is_marine_standard_only(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    obs = _synthetic_obs(tmp_path)
    sd_ent = ob.load_station_days(obs, label="enterococcus")
    assert sd_ent["ent"].notna().all()  # only station-days with a marine enterococcus reading
    # exceed is exactly the single-sample marine standard ent > 104, not a multi-analyte OR.
    assert (sd_ent["exceed"] == (sd_ent["ent"] > 104).astype(int)).all()


def test_vb_mlr_baseline_present_and_judged(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = ob.run(_synthetic_obs(tmp_path), clf=_tiny_clf(), rain_dir=_synthetic_rain(tmp_path))
    all_s = res["strata"]["ALL"]
    assert "baseline_vb_mlr" in all_s["models"]  # Virtual-Beach-class baseline scored
    assert isinstance(all_s["operational_verdict"]["model_beats_vb_mlr"], bool)


def test_run_with_discharge_adds_first_flush_features(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_BACTERIA_OBS", raising=False)
    res = ob.run(_synthetic_obs(tmp_path), clf=_tiny_clf(), discharge_dir=_synthetic_discharge(tmp_path))
    assert "EVALUATED" in res["discharge_first_flush"]   # discharge path engaged
    assert "NOT EVALUATED" in res["ab411_rain_rule"]     # independent of rain
    assert res["strata"]["ALL"]["operational_verdict"] is not None
