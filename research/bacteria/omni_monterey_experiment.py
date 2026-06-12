#!/usr/bin/env python3
"""Monterey Bay Omni-Model Experiment.

Builds a 'kitchen-sink' model for Monterey Bay beaches by injecting ALL available 
lakehouse drivers:
- Site-memory (previous exceedances)
- Rainfall
- Tides
- Ocean currents, winds, and waves (NDBC)
- Upwelling indices (CUTI/BEUTI)
- Sewage spills (CalOES)
- Marine HABs (CDPH HABMAP cell counts)
- Satellite SST and Chlorophyll

Evaluates if the omni-model beats the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob
from research.bacteria.spatial_autocorr import _load_geo
from research.bacteria.habmap_experiment import add_habmap_features

def _fit_eval_omni(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, feats: list[str], base: float) -> np.ndarray:
    x_tr = tr[feats].fillna(0).to_numpy()
    y_tr = tr["exceed"].to_numpy()
    x_te = te[feats].fillna(0).to_numpy()
    
    if len(np.unique(y_tr)) < 2:
        return np.full(len(x_te), base)
        
    # Using HistGradientBoosting because it handles many features gracefully
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )
    clf.fit(x_tr, y_tr)
    return clf.predict_proba(x_te)[:, 1]

def run_omni(obs_path: Path, reveal_lag_days: int = 2, label: str = "enterococcus") -> dict:
    df = ob.load_station_days(obs_path, label=label)
    
    # Filter to Monterey Bay region (Monterey, Santa Cruz)
    bay_counties = {"Monterey", "Santa Cruz"}
    df = df[df["county"].isin(bay_counties)].copy()
    
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    
    # 1. Base features
    feats_base = list(ob.FEATS)
    
    # 2. HABMAP features
    geo = _load_geo()
    df = add_habmap_features(df, geo, reveal_lag_days=reveal_lag_days)
    habmap_feats = [
        "Pseudo_nitzschia_seriata_group", 
        "Pseudo_nitzschia_delicatissima_group",
        "Alexandrium_spp", 
        "Dinophysis_spp", 
        "Akashiwo_sanguinea", 
        "Total_Phytoplankton",
        "pDA", "tDA"
    ]
    
    # 3. Hourly Drivers (Winds, Waves, Spills, Upwelling, SST)
    drivers_path = _REPO_ROOT / "lakehouse" / "silver" / "external_drivers" / "drivers_hourly.parquet"
    driver_feats = []
    if drivers_path.exists():
        drivers = pd.read_parquet(drivers_path)
        # Shift index by reveal_lag_days to prevent leakage
        drivers.index = drivers.index + pd.Timedelta(days=reveal_lag_days)
        # Resample to daily average
        daily_drivers = drivers.resample("D").mean()
        daily_drivers.index = daily_drivers.index.tz_localize(None).normalize()
        daily_drivers.index.name = "sample_date"
        
        # Select important numeric columns
        ignore = ["tide_Mf_cos", "solar_elev_pos"] # some trivial ones
        driver_cols = [c for c in daily_drivers.columns if c not in ignore and daily_drivers[c].dtype in (np.float64, np.float32)]
        
        df = df.merge(daily_drivers[driver_cols].reset_index(), on="sample_date", how="left")
        
        for c in driver_cols:
            df[c] = df[c].fillna(0)
            
        driver_feats = driver_cols

    all_feats = feats_base + [f for f in habmap_feats if f in df.columns] + driver_feats

    arms = {
        "baseline": feats_base,
        "omni_kitchen_sink": all_feats,
    }

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    
    if len(te) == 0:
        return {"error": "No holdout data available."}
        
    base = float(tr["exceed"].mean())
    y = te["exceed"].to_numpy()

    scores = {}
    for name, feats in arms.items():
        te[f"p_{name}"] = _fit_eval_omni(tr, va, te, feats, base)
        scores[name] = ob.score(y, te[f"p_{name}"].to_numpy(), base)

    ns_ap = scores["baseline"]["ap"]
    omni_ap = scores["omni_kitchen_sink"]["ap"]
    
    verdict = {
        "baseline_ap": round(ns_ap, 4), 
        "omni_ap": round(omni_ap, 4),
        "ap_lift": round(omni_ap - ns_ap, 4),
        "omni_improves_model": bool(omni_ap > ns_ap)
    }

    return {
        "label": label, 
        "region": list(bay_counties),
        "n_test": int(len(te)), 
        "events": int(y.sum()), 
        "num_features": len(all_feats),
        "scores": scores, 
        "verdict": verdict
    }

def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"Omni-model experiment error: {res['error']}"
    s, v = res["scores"], res["verdict"]
    lines = [
        "# Monterey Bay Omni-Model Experiment",
        "",
        f"- label={res['label']} | region={res['region']} ",
        f"- test rows={res['n_test']} | events={res['events']} | features={res['num_features']}",
        "",
        "| model | AP | ROC-AUC | ECE |",
        "|---|--:|--:|--:|",
    ]
    for name in ["baseline", "omni_kitchen_sink"]:
        sc = s[name]
        lines.append(f"| {name} | {sc['ap']} | {sc['roc_auc']} | {sc['ece']} |")
    lines += [
        "",
        f"- AP Lift: **{v['ap_lift']:+}**",
        f"- Verdict: **{'SUCCESS' if v['omni_improves_model'] else 'WASH'}**",
    ]
    return "\n".join(lines)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    res = run_omni(obs, reveal_lag_days=args.reveal_lag_days)
    
    if "error" in res:
        print(f"Error: {res['error']}")
        return 1
        
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "omni_monterey_experiment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "omni_monterey_experiment.md").write_text(md, encoding="utf-8")
    
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
