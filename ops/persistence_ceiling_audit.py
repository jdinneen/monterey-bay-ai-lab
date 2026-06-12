#!/usr/bin/env python3
"""Persistence Ceiling Audit for MBAL Forecasts (Robust Version).
Determines if models are limited by sensor noise or lack of signal.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def estimate_noise_floor(y: np.ndarray) -> float:
    if len(y) < 2: return 0.0
    diffs = np.diff(y)
    return float(np.std(diffs) / np.sqrt(2))

def main():
    metrics_path = Path("lakehouse/gold/forecast_metrics/metrics.parquet")
    pred_root = Path("lakehouse/gold/forecast_predictions")
    
    if not metrics_path.exists() or not pred_root.exists():
        print("Gold layer artifacts not found.")
        return

    metrics = pd.read_parquet(metrics_path)
    print(f"Auditing metrics for {len(metrics)} target-horizon pairs...")

    all_stats = []
    
    # Process by run_id to avoid loading everything at once
    for run_id, run_metrics in metrics.groupby("run_id"):
        pred_path = pred_root / f"run_id={run_id}" / "predictions.parquet"
        if not pred_path.exists(): continue
        
        preds = pd.read_parquet(pred_path)
        
        # Join metrics and preds on unique_id and ds if needed, 
        # but here we just need the residuals from the pred file if available.
        # The pred file usually has 'y' (actual) and the model col.
        
        for _, row in run_metrics.iterrows():
            uid = row["unique_id"]
            h = row["horizon_h"]
            model_name = row["model"]
            
            # Find the prediction column in the parquet
            # It might be exactly the model name or 'model' or 'prediction'
            cand_cols = [model_name, "model", "prediction", "forecast"]
            pred_col = next((c for c in cand_cols if c in preds.columns), None)
            
            if not pred_col:
                # Try to find any column that isn't metadata
                fixed = ["run_id", "split_id", "unique_id", "ds", "cutoff", "y", "observed", "origin_observed", "model_prediction_col", "cache_version"]
                others = [c for c in preds.columns if c not in fixed]
                if others: pred_col = others[0]
            
            if not pred_col or "y" not in preds.columns: continue
            
            # Filter preds for this unique_id
            g = preds[preds["unique_id"] == uid]
            if g.empty: continue
            
            actual = g["y"].to_numpy()
            model_pred = g[pred_col].to_numpy()
            
            # Need persistence. If not in file, we estimate it from y and horizon if ds is continuous.
            # But the Gold layer is supposed to have 'persistence' if it's a v2 run.
            if "persistence" in g.columns:
                persist = g["persistence"].to_numpy()
            else:
                # Fallback: RMSE persistence from metrics if available
                rmse_persist = row.get("persistence_rmse", 0.0)
                persist = None # Can't do point-wise but can do aggregate
            
            rmse_model = np.sqrt(np.mean((actual - model_pred)**2))
            noise_floor = estimate_noise_floor(actual)
            
            if persist is not None:
                rmse_persist = np.sqrt(np.mean((actual - persist)**2))
            
            skill = 1.0 - (rmse_model / rmse_persist) if rmse_persist > 0 else 0.0
            headroom = (rmse_model - noise_floor) / rmse_model if rmse_model > 0 else 0.0
            
            all_stats.append({
                "run_id": run_id,
                "target": uid,
                "horizon_h": int(h),
                "model": model_name,
                "rmse_persist": float(rmse_persist),
                "rmse_model": float(rmse_model),
                "skill_pct": float(skill * 100),
                "noise_floor": float(noise_floor),
                "headroom_pct": float(headroom * 100)
            })

    if not all_stats:
        print("No valid stats could be computed.")
        return
        
    audit_df = pd.DataFrame(all_stats)
    pivot = audit_df.groupby("horizon_h")[["skill_pct", "headroom_pct"]].mean()
    
    print("\n# Persistence Ceiling Audit Results\n")
    print(pivot)
    
    print("\n## The 'Ceiling' Verdict")
    for h, row in pivot.iterrows():
        if row["headroom_pct"] < 10:
            print(f"- {h}h Horizon: CRITICAL CEILING ({row['headroom_pct']:.1f}% headroom). Further neural modeling is likely a waste of VRAM. Error is indistinguishable from sensor jitter.")
        elif row["headroom_pct"] < 25:
            print(f"- {h}h Horizon: NEAR CEILING ({row['headroom_pct']:.1f}% headroom). Models are hitting the 'Predictability Limit' of the physical system.")
        else:
            print(f"- {h}h Horizon: ROOM TO GROW ({row['headroom_pct']:.1f}% headroom). Optimization is still justifiable.")

if __name__ == "__main__":
    main()
