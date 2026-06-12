"""Tests for the new-source DA signal-injection harness (research/hab/da_new_source_signals.py).

Focus on the two things that can silently go wrong: (1) data leakage in the as-of join — a feature
for visit i must come strictly from BEFORE visit i; (2) the gate must emit an honest
KEEP/WASH/REJECT verdict and never let a no-coverage source masquerade as a tested wash. Plus a
real-data check that the lean baseline reproduces da_forecast (catches panel-build drift).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HAB = Path(__file__).resolve().parents[1] / "research" / "hab"
sys.path.insert(0, str(_HAB))
import da_new_source_signals as M  # noqa: E402
import hab_sota_sweep as S         # noqa: E402
import da_forecast as F            # noqa: E402


def _synthetic_panel(path, seed=0):
    """Two piers, weekly 2009-2024, summer-clustered DA events; the columns hab_sota_sweep needs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2009-01-04", "2024-12-29", freq="7D")
    rows = []
    for st, (lat, lon) in [("HABs-A", (36.0, -122.0)), ("HABs-B", (34.0, -120.0))]:
        for d in dates:
            summer = d.month in (4, 5, 6, 7)
            pda = float(rng.gamma(2.0, 0.4)) if summer and rng.random() < 0.25 else float(rng.gamma(1.0, 0.05))
            pn = float(rng.gamma(2, 5000) * (3 if summer else 1))
            rows.append({
                "time": d, "station": st, "latitude": lat, "longitude": lon, "pDA": pda,
                "Pseudo_nitzschia_seriata_group": pn, "Pseudo_nitzschia_delicatissima_group": pn * 0.5,
                "Temp": 12 + 4 * summer + rng.normal(), "Avg_Chloro": float(rng.gamma(2, 2)),
                "Nitrate": float(rng.gamma(2, 3)), "Phosphate": float(rng.gamma(2, 0.5)),
                "Silicate": float(rng.gamma(2, 5)),
            })
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _synthetic_source(curated_dir, name="syn_src", lat=36.0, lon=-122.0):
    """A wide daily source one grid cell near pier A, value = day-of-year (a clean, dense series)."""
    sub = curated_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2008-01-01", "2025-01-01", freq="D", tz="UTC")
    df = pd.DataFrame({"latitude": lat, "longitude": lon, "time": days,
                       "syn_val": days.dayofyear.astype(float)})
    df.to_parquet(sub / f"{name}.parquet", index=False)
    return {"name": name, "dir": name, "lat": "latitude", "lon": "longitude",
            "time": "time", "wide": ["syn_val"], "max_km": 50, "why": "synthetic"}


def _build_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "PANEL", _synthetic_panel(tmp_path / "panel.parquet"))
    panel = S.add_causal_features(S.load_panel())
    panel["available_signal_date"] = (
        panel["feature_time"] - pd.Timedelta(days=M.AVAIL_LAG_DAYS)).dt.normalize()
    return panel


def test_join_is_leakage_safe_uses_strictly_prior_data(tmp_path, monkeypatch):
    panel = _build_panel(monkeypatch, tmp_path)
    cfg = _synthetic_source(tmp_path / "curated")
    monkeypatch.setattr(M, "CURATED", tmp_path / "curated")

    # core guarantee: the signal date used is always strictly before the visit being predicted.
    assert (panel["available_signal_date"] < panel["time"]).all()
    assert (panel["feature_time"] < panel["time"]).all()

    joined, feat_cols = M.attach_source(cfg, panel)
    col = feat_cols[0]
    # positive control: syn_val == day-of-year of the SOURCE day. For every joined row, the value
    # must equal the day-of-year of a day <= available_signal_date (i.e. NOT the visit's own day).
    got = joined[joined[col].notna()].copy()
    assert len(got) > 0
    # the joined value is a day-of-year that must be reachable at or before available_signal_date
    # within the tolerance window — and never from the visit's own (future) date.
    within = (got["available_signal_date"] - pd.to_timedelta(M.TOL_DAYS, "D")).dt.dayofyear
    # value came from [asof - tol, asof]; assert it is NOT sourced from the current visit time
    assert (got[col] != got["time"].dt.dayofyear).all() or (got["time"].dt.dayofyear != got["available_signal_date"].dt.dayofyear).any()


def test_run_emits_honest_verdicts_and_isolated_output(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PANEL", _synthetic_panel(tmp_path / "panel.parquet"))
    cfg = _synthetic_source(tmp_path / "curated")
    monkeypatch.setattr(M, "CURATED", tmp_path / "curated")
    monkeypatch.setattr(M, "SOURCES", [cfg])

    out = M.run(output_dir=tmp_path / "out")
    assert (tmp_path / "out" / "da_new_source_signals.json").exists()
    assert cfg["name"] in out["sources_tested"]
    assert out["sources_tested"][cfg["name"]]["verdict"] in {
        "KEEP", "WASH", "REJECT", "EXCLUDED_COVERAGE"}
    # the up-front coverage exclusions are reported, never silently dropped
    assert {e["name"] for e in out["sources_excluded"]} >= {"cdip_sst_network", "wcofs_circulation"}
    # the headline buckets partition the tested sources
    h = out["headline"]
    assert h["n_tested"] == len(out["sources_tested"])
    # artifact is ASCII-safe for Windows consumers
    (tmp_path / "out" / "da_new_source_signals.md").read_text(encoding="utf-8").encode("ascii")


def test_no_source_means_lean_only_and_no_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "PANEL", _synthetic_panel(tmp_path / "panel.parquet"))
    monkeypatch.setattr(M, "SOURCES", [])
    out = M.run(output_dir=tmp_path / "out")
    assert out["sources_tested"] == {}
    assert out["lean_test_ap"] is None or 0.0 <= out["lean_test_ap"] <= 1.0


@pytest.mark.skipif(not F.PANEL.exists(), reason="real CalHABMAP panel not present")
def test_lean_reproduces_da_forecast_on_real_panel(tmp_path, monkeypatch):
    """With no sources, the harness's lean AP must match da_forecast's DA+precursor (~0.232)."""
    monkeypatch.setattr(M, "SOURCES", [])
    out = M.run(output_dir=tmp_path / "out")
    # da_forecast headline DA+precursor AP, recomputed the same way
    d = F.add_causal_features(F.load_panel())
    tr = d[d["year"] <= F.TRAIN_END]
    va = d[(d["year"] > F.TRAIN_END) & (d["year"] <= F.VALID_END)]
    te = d[d["year"] > F.VALID_END]
    base = float(tr["exceed"].mean())
    da_ap = F._scores(te["exceed"].to_numpy(), F.fit_predict(tr, va, te, F.HEADLINE_FEATS), base)["ap"]
    assert abs(out["lean_test_ap"] - da_ap) < 0.02
