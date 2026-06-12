#!/usr/bin/env python3
"""HABMAP Cell Count Integration Experiment.

Evaluates whether adding CDPH HABMAP microscopic cell counts 
(Pseudo-nitzschia, Alexandrium, Total Phytoplankton, etc.)
improves the MBAL bacteria-exceedance prediction baseline.

Uses kNN to map the nearest HABMAP pier to each beach station, 
applying a safe reveal lag to prevent data leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob
from research.bacteria.spatial_autocorr import _load_geo
from research.bacteria.spatial_drivers_experiment import _fit_eval

HABMAP_PATH = _REPO_ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet"

def add_habmap_features(df: pd.DataFrame, geo: pd.DataFrame, reveal_lag_days: int = 2) -> pd.DataFrame:
    if not HABMAP_PATH.exists():
        print("HABMAP data not found in lakehouse.")
        return df

    from sklearn.neighbors import BallTree
    
    habmap = pd.read_parquet(HABMAP_PATH)
    if habmap.empty:
        return df
        
    # Standardize time to local date
    habmap["time"] = pd.to_datetime(habmap["time"]).dt.tz_localize(None).dt.normalize()
    
    # Get unique HABMAP pier locations
    piers = habmap[["latitude", "longitude", "station"]].drop_duplicates(subset=["station"]).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    coords_piers = np.radians(piers[["latitude", "longitude"]].to_numpy())
    
    tree = BallTree(coords_piers, metric="haversine")
    
    # Map each beach to its nearest HABMAP pier
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    coords_beach = np.radians(geo_t[["latitude", "longitude"]].to_numpy())
    
    dist, ind = tree.query(coords_beach, k=1)
    
    # Assign the nearest HABMAP station name to the beach
    geo_mapped = geo_t.copy()
    geo_mapped["habmap_station"] = piers.iloc[ind[:, 0]]["station"].values
    
    # Merge mapped pier name onto main dataframe
    df = df.merge(geo_mapped[["station_id", "habmap_station"]], on="station_id", how="left")
    
    # Because HABMAP is sampled weekly, we should forward-fill the recent observation
    # First, shift the sample date forward by reveal_lag_days to prevent leakage
    habmap_lagged = habmap.copy()
    habmap_lagged["sample_date"] = habmap_lagged["time"] + pd.Timedelta(days=reveal_lag_days)
    
    features = [
        "Pseudo_nitzschia_seriata_group", 
        "Pseudo_nitzschia_delicatissima_group",
        "Alexandrium_spp", 
        "Dinophysis_spp", 
        "Akashiwo_sanguinea", 
        "Total_Phytoplankton",
        "pDA", "tDA"
    ]
    
    # Create a daily calendar per station to forward-fill weekly counts
    daily_idx = pd.date_range(habmap_lagged["sample_date"].min(), df["sample_date"].max() + pd.Timedelta(days=7), freq="D")
    
    ffill_dfs = []
    for station, group in habmap_lagged.groupby("station"):
        # Take max per day in case of multiple samples
        daily = group.groupby("sample_date")[features].max()
        # Reindex to daily calendar and forward fill up to 14 days
        daily_filled = daily.reindex(daily_idx).ffill(limit=14)
        daily_filled["habmap_station"] = station
        daily_filled = daily_filled.reset_index().rename(columns={"index": "sample_date"})
        ffill_dfs.append(daily_filled)
        
    habmap_daily = pd.concat(ffill_dfs, ignore_index=True)
    
    # Join HABMAP features to beach observations
    df = df.merge(habmap_daily, on=["sample_date", "habmap_station"], how="left")
    
    # Fill missing with 0 since missing HABMAP usually implies below detection, 
    # or just fill with mean. For now, 0.
    for f in features:
        df[f] = df[f].fillna(0)
        
    return df

def run(obs_path: Path, reveal_lag_days: int = 2, label: str = "enterococcus",
        exclude_counties=("San Diego",)) -> dict:
    
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    feats_base = list(ob.FEATS)

    geo = _load_geo()
    if geo is None:
        return {"error": "station_geo.parquet not found"}
        
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
    
    if habmap_feats[0] not in df.columns:
        return {"error": "HABMAP features not added successfully."}

    arms = {
        "baseline": feats_base,
        "baseline_plus_habmap": feats_base + habmap_feats,
    }

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    
    if len(te) == 0:
        return {"error": "No holdout data available. Check dataset range."}
        
    base = float(tr["exceed"].mean())
    y = te["exceed"].to_numpy()

    scores = {}
    for name, feats in arms.items():
        te[f"p_{name}"] = _fit_eval(tr, va, te, feats, base)
        scores[name] = ob.score(y, te[f"p_{name}"].to_numpy(), base)

    ns_ap = scores["baseline"]["ap"]
    hab_ap = scores["baseline_plus_habmap"]["ap"]
    
    verdict = {
        "baseline_ap": round(ns_ap, 4), 
        "habmap_ap": round(hab_ap, 4),
        "ap_lift": round(hab_ap - ns_ap, 4),
        "habmap_improves_model": bool(hab_ap > ns_ap)
    }

    return {
        "label": label, 
        "excluded_counties": list(exclude_counties or []),
        "n_test": int(len(te)), 
        "events": int(y.sum()), 
        "scores": scores, 
        "verdict": verdict
    }

def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"HABMAP experiment error: {res['error']}"
    s, v = res["scores"], res["verdict"]
    lines = [
        "# HABMAP Cell Counts Integration Experiment",
        "",
        f"- label={res['label']} | excluded={res['excluded_counties']} "
        f"| test rows={res['n_test']} | events={res['events']}",
        "",
        "| model | AP | ROC-AUC | ECE |",
        "|---|--:|--:|--:|",
    ]
    for name in ["baseline", "baseline_plus_habmap"]:
        sc = s[name]
        lines.append(f"| {name} | {sc['ap']} | {sc['roc_auc']} | {sc['ece']} |")
    lines += [
        "",
        f"- AP Lift: **{v['ap_lift']:+}**",
        f"- Verdict: **{'SUCCESS' if v['habmap_improves_model'] else 'WASH'}**",
    ]
    return "\n".join(lines)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    res = run(obs, reveal_lag_days=args.reveal_lag_days)
    
    if "error" in res:
        print(f"Error: {res['error']}")
        return 1
        
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "habmap_experiment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "habmap_experiment.md").write_text(md, encoding="utf-8")
    
    print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
