#!/usr/bin/env python3
"""Monterey Bay Predictive Algae Bloom Experiment.

Pivots the modeling architecture to predict Marine Harmful Algal Blooms (HABs) 
instead of beach bacteria. 
Target: `Pseudo_nitzschia_seriata_group > 10000` (cells/L) at Monterey Bay piers.
Features: 
- Lakehouse Drivers (`drivers_hourly.parquet`) containing Nutrients, Upwelling, SST, Winds, Tides.

Evaluates if we can predict localized toxic algae blooms using macro physics and river nutrient chemistry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def load_algae_targets() -> pd.DataFrame:
    habmap = pd.read_parquet(_REPO_ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet")
    # Filter to Monterey Bay region piers
    bay_piers = ["HABs-SantaCruzWharf", "HABs-MontereyWharf"]
    df = habmap[habmap["station"].isin(bay_piers)].copy()
    
    # Target definition: Pseudo-nitzschia seriata > 10,000 cells/L (common bloom threshold)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["sample_date"] = df["time"].dt.normalize().dt.tz_localize(None)
    
    df["pn_seriata"] = pd.to_numeric(df["Pseudo_nitzschia_seriata_group"], errors="coerce").fillna(0)
    df["exceed"] = (df["pn_seriata"] > 10000).astype(int)
    
    # We only care about date, station, and exceed
    return df[["sample_date", "station", "exceed", "pn_seriata"]]

def build_features(df: pd.DataFrame, reveal_lag_days: int = 1) -> tuple[pd.DataFrame, list[str]]:
    drivers_path = _REPO_ROOT / "lakehouse" / "silver" / "external_drivers" / "drivers_hourly.parquet"
    if not drivers_path.exists():
        raise FileNotFoundError(f"Drivers not found at {drivers_path}")
        
    drivers = pd.read_parquet(drivers_path)
    
    # Shift index by reveal_lag_days to prevent leakage
    drivers.index = drivers.index + pd.Timedelta(days=reveal_lag_days)
    daily_drivers = drivers.resample("D").mean()
    daily_drivers.index = daily_drivers.index.tz_localize(None).normalize()
    daily_drivers.index.name = "sample_date"
    
    ignore = ["tide_Mf_cos", "solar_elev_pos"]
    driver_cols = [c for c in daily_drivers.columns if c not in ignore and daily_drivers[c].dtype in (np.float64, np.float32)]
    
    df = df.merge(daily_drivers[driver_cols].reset_index(), on="sample_date", how="left")
    for c in driver_cols:
        df[c] = df[c].fillna(0)
        
    return df, driver_cols

def _fit_eval(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, feats: list[str], base: float) -> np.ndarray:
    x_tr = tr[feats].fillna(0).to_numpy()
    y_tr = tr["exceed"].to_numpy()
    x_te = te[feats].fillna(0).to_numpy()
    
    if len(np.unique(y_tr)) < 2:
        return np.full(len(x_te), base)
        
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42
    )
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_te)[:, 1]

def run_experiment(reveal_lag_days: int = 1) -> dict:
    df = load_algae_targets()
    df, driver_feats = build_features(df, reveal_lag_days=reveal_lag_days)
    
    # Train / Test split
    tr = df[df["sample_date"] <= "2020-12-31"]
    va = df[(df["sample_date"] >= "2021-01-01") & (df["sample_date"] <= "2022-12-31")]
    te = df[df["sample_date"] >= "2023-01-01"].copy()
    
    if len(te) == 0:
        return {"error": "No holdout data available (2023+)."}
        
    base = float(tr["exceed"].mean())
    y = te["exceed"].to_numpy()
    
    # We test two models: 
    # 1. Baseline: Just predict the mean
    # 2. Physics+Chemistry: Lakehouse drivers
    
    p_base = np.full(len(te), base)
    p_model = _fit_eval(tr, va, te, driver_feats, base)
    
    # Score
    def _score(y_true, y_pred):
        if len(np.unique(y_true)) < 2:
            return {"ap": base, "roc_auc": 0.5}
        return {
            "ap": round(float(average_precision_score(y_true, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_pred)), 4)
        }
        
    scores = {
        "baseline": _score(y, p_base),
        "lakehouse_physics": _score(y, p_model)
    }
    
    ns_ap = scores["baseline"]["ap"]
    mod_ap = scores["lakehouse_physics"]["ap"]
    
    verdict = {
        "baseline_ap": ns_ap, 
        "model_ap": mod_ap,
        "ap_lift": round(mod_ap - ns_ap, 4),
        "improves": bool(mod_ap > ns_ap)
    }
    
    return {
        "label": "Pseudo_nitzschia_seriata > 10,000", 
        "region": ["Santa Cruz Wharf", "Monterey Wharf"],
        "n_test": int(len(te)), 
        "events": int(y.sum()), 
        "num_features": len(driver_feats),
        "scores": scores, 
        "verdict": verdict
    }

def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"Algae prediction experiment error: {res['error']}"
    s, v = res["scores"], res["verdict"]
    lines = [
        "# Monterey Bay HABs Prediction Experiment",
        "",
        f"- Target: `{res['label']}`",
        f"- Region: {res['region']}",
        f"- Test Set (2023+): {res['n_test']} observations | {res['events']} bloom events",
        f"- Features: {res['num_features']} Lakehouse physics & chemistry drivers",
        "",
        "| model | AP | ROC-AUC |",
        "|---|--:|--:|",
    ]
    for name in ["baseline", "lakehouse_physics"]:
        sc = s[name]
        lines.append(f"| {name} | {sc['ap']} | {sc['roc_auc']} |")
    lines += [
        "",
        f"- **AP Lift:** {v['ap_lift']:+}",
        f"- **Verdict:** {'SUCCESS' if v['improves'] else 'WASH'}",
    ]
    return "\n".join(lines)

def main():
    res = run_experiment(reveal_lag_days=1)
    if "error" in res:
        print(f"Error: {res['error']}")
        return 1
        
    out_dir = _REPO_ROOT / "reports" / "algae_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "predict_bloom_experiment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "predict_bloom_experiment.md").write_text(md, encoding="utf-8")
    
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
