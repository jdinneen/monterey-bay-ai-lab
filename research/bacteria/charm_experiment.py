#!/usr/bin/env python3
"""C-HARM Integration Experiment.

Evaluates whether adding the NOAA C-HARM Nowcast V3 probabilities
(pseudo_nitzschia, particulate_domoic, cellular_domoic) as input features 
improves the MBAL bacteria-exceedance prediction baseline.

Uses kNN to map the 3km C-HARM grid to beach stations, applying a safe
reveal lag to prevent data leakage.
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

CHARM_PATH = _REPO_ROOT / "data" / "external_curated" / "c_harm" / "c_harm.parquet"

def add_charm_features(df: pd.DataFrame, geo: pd.DataFrame, reveal_lag_days: int = 2) -> pd.DataFrame:
    if not CHARM_PATH.exists():
        print("C-HARM data not found. Waiting for fetch to complete.")
        return df

    from sklearn.neighbors import BallTree
    
    charm = pd.read_parquet(CHARM_PATH)
    if charm.empty:
        return df
        
    charm["time"] = pd.to_datetime(charm["time"]).dt.tz_localize(None).dt.normalize()
    
    # We only need the unique C-HARM grid points
    grid_points = charm[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    coords_charm = np.radians(grid_points[["latitude", "longitude"]].to_numpy())
    
    tree = BallTree(coords_charm, metric="haversine")
    
    # Map each beach to its nearest C-HARM grid point
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    coords_beach = np.radians(geo_t[["latitude", "longitude"]].to_numpy())
    
    dist, ind = tree.query(coords_beach, k=1)
    
    # Assign the C-HARM grid point (lat/lon) to the beach station
    geo_mapped = geo_t.copy()
    geo_mapped["charm_lat"] = grid_points.iloc[ind[:, 0]]["latitude"].values
    geo_mapped["charm_lon"] = grid_points.iloc[ind[:, 0]]["longitude"].values
    
    # Merge mapped grid lat/lon onto main dataframe
    df = df.merge(geo_mapped[["station_id", "charm_lat", "charm_lon"]], on="station_id", how="left")
    
    # Shift C-HARM time forward by reveal_lag_days to prevent leakage
    charm_lagged = charm.copy()
    charm_lagged["sample_date"] = charm_lagged["time"] + pd.Timedelta(days=reveal_lag_days)
    
    # Keep only the features we want to join
    features = ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic"]
    
    charm_subset = charm_lagged[["sample_date", "latitude", "longitude"] + features].rename(
        columns={"latitude": "charm_lat", "longitude": "charm_lon"}
    )
    
    # Join C-HARM features to beach observations
    df = df.merge(charm_subset, on=["sample_date", "charm_lat", "charm_lon"], how="left")
    
    # Optional: Fill missing with 0 or mean
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
        
    df = add_charm_features(df, geo, reveal_lag_days=reveal_lag_days)
    
    charm_feats = ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic"]
    
    # If fetch hasn't completed or join failed
    if charm_feats[0] not in df.columns:
        return {"error": "C-HARM features not added successfully."}

    arms = {
        "baseline_no_charm": feats_base,
        "baseline_plus_charm": feats_base + charm_feats,
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

    ns_ap = scores["baseline_no_charm"]["ap"]
    charm_ap = scores["baseline_plus_charm"]["ap"]
    
    verdict = {
        "baseline_ap": round(ns_ap, 4), 
        "charm_ap": round(charm_ap, 4),
        "ap_lift": round(charm_ap - ns_ap, 4),
        "charm_improves_model": bool(charm_ap > ns_ap)
    }

    return {
        "label": label, 
        "excluded_counties": list(exclude_counties or []),
        "n_test": int(len(te)), 
        "events": int(y.sum()), 
        "scores": scores, 
        "verdict": verdict
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    args = ap.parse_args(argv)
    
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    res = run(obs, reveal_lag_days=args.reveal_lag_days)
    
    if "error" in res:
        print(f"Error: {res['error']}")
        return 1
        
    v = res["verdict"]
    print("=== C-HARM Integration Experiment ===")
    print(f"Baseline AP: {v['baseline_ap']}")
    print(f"With C-HARM: {v['charm_ap']}")
    print(f"AP Lift:     {v['ap_lift']:+}")
    print(f"Verdict:     {'SUCCESS - C-HARM provides lift' if v['charm_improves_model'] else 'WASH - No lift from C-HARM'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
