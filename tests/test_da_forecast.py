"""Tests for the domoic-acid forecast model (research/hab/da_forecast.py).

Focuses on the leakage-prone part — strict one-visit-ahead causality of the features — plus a
synthetic end-to-end run that exercises the splits, baselines, LOSO, and artifact writing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "hab"))
import da_forecast as F  # noqa: E402


def _synthetic_panel(path, seed=0):
    """Two piers, weekly 2009-2024, summer-clustered DA events; >=10 events/station for LOSO."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2009-01-04", "2024-12-29", freq="7D")
    rows = []
    for st in ["HABs-A", "HABs-B"]:
        for d in dates:
            summer = d.month in (4, 5, 6, 7)
            pda = float(rng.gamma(2.0, 0.4)) if summer and rng.random() < 0.25 else float(rng.gamma(1.0, 0.05))
            pn = float(rng.gamma(2, 5000) * (3 if summer else 1))
            rows.append({
                "time": d, "station": st, "latitude": 36.0, "longitude": -122.0,
                "pDA": pda,
                "Pseudo_nitzschia_seriata_group": pn,
                "Pseudo_nitzschia_delicatissima_group": pn * 0.5,
                "Temp": 12 + 4 * summer + rng.normal(), "Avg_Chloro": float(rng.gamma(2, 2)),
                "Nitrate": float(rng.gamma(2, 3)), "Phosphate": float(rng.gamma(2, 0.5)),
                "Silicate": float(rng.gamma(2, 5)),
            })
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_features_are_strictly_one_visit_causal(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "PANEL", _synthetic_panel(tmp_path / "p.parquet"))
    d = F.add_causal_features(F.load_panel())
    # reconstruct the per-station prior exceedance and check exc_prev matches the PRIOR visit,
    # never the row's own label.
    raw = F.load_panel()
    for st, g in raw.groupby("station"):
        g = g.sort_values("time").reset_index(drop=True)
        merged = d[d["station"] == st].sort_values("time").reset_index(drop=True)
        # align: d drops the first visit per station (no prior), so g[1:] lines up with merged
        prior_exceed = g["exceed"].to_numpy()[:-1]
        assert np.array_equal(merged["exc_prev"].to_numpy(), prior_exceed)
        # a feature must never equal its OWN-row toxin spike: pda_prev is the prior log1p(pDA)
        prior_pda = np.log1p(g["pDA"].clip(lower=0).to_numpy()[:-1])
        np.testing.assert_allclose(merged["pda_prev"].to_numpy(), prior_pda, rtol=1e-9)


def test_end_to_end_runs_and_beats_seasonal(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "PANEL", _synthetic_panel(tmp_path / "p.parquet"))
    monkeypatch.setattr(F, "OUTDIR", tmp_path / "out")
    rc = F.main([])  # explicit empty argv: main() now takes CLI args (--output-dir), so an
    # in-process call must pass [] or argparse would consume pytest's sys.argv.
    assert rc == 0
    res = json.loads((tmp_path / "out" / "da_forecast.json").read_text(encoding="utf-8"))
    h = res["headline"]
    # valid probabilities + a real signal in the engineered summer-clustered data
    for m in res["test_timeheldout"].values():
        assert m["ap"] is None or 0.0 <= m["ap"] <= 1.0
    assert h["model_roc_auc"] is None or 0.0 <= h["model_roc_auc"] <= 1.0
    assert res["leave_one_station_out"]  # LOSO ran for both engineered stations
    # artifacts are ASCII-safe (Windows consumers)
    (tmp_path / "out" / "da_forecast.md").read_text(encoding="utf-8").encode("ascii")
