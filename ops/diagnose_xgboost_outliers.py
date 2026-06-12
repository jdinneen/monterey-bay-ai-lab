#!/usr/bin/env python3
"""Diagnostic for XGBoost metric explosions.
Checks if predictions in the gold layer are physically impossible (out of range).
"""
import pandas as pd
from pathlib import Path
import json

PHYSICAL_RANGES = {
    "temp": (-3.0, 35.0),
    "sal": (20.0, 40.0),
    "air_temp": (-10.0, 45.0),
    "air_press": (870.0, 1085.0),
}

def get_range(uid: str):
    if "temp" in uid: return PHYSICAL_RANGES["temp"]
    if "sal" in uid: return PHYSICAL_RANGES["sal"]
    if "air_temp" in uid: return PHYSICAL_RANGES["air_temp"]
    if "air_press" in uid: return PHYSICAL_RANGES["air_press"]
    return None

def main():
    lakehouse = Path("lakehouse/gold/forecast_predictions")
    if not lakehouse.exists():
        print("Lakehouse predictions not found.")
        return

    # Find XGBoost runs
    metrics_path = Path("lakehouse/gold/forecast_metrics/metrics.parquet")
    if not metrics_path.exists():
        print("Metrics table not found.")
        return
    
    metrics = pd.read_parquet(metrics_path)
    xgb_runs = metrics[metrics["model"].str.contains("xgboost", case=False)]["run_id"].unique()
    
    if len(xgb_runs) == 0:
        print("No XGBoost runs found in metrics.")
        return

    print(f"Auditing {len(xgb_runs)} XGBoost runs for outlier explosions...")
    
    for run_id in xgb_runs:
        pred_path = lakehouse / f"run_id={run_id}" / "predictions.parquet"
        if not pred_path.exists():
            continue
        
        df = pd.read_parquet(pred_path)
        # Prediction col is often in 'model_prediction_col' metadata or just the unique one
        pred_col = df.get("model_prediction_col", pd.Series(["prediction"])).iloc[0]
        if pred_col not in df.columns:
            # Try to find it
            cols = [c for c in df.columns if c not in ["run_id", "split_id", "unique_id", "ds", "cutoff", "y", "observed", "origin_observed"]]
            if not cols: continue
            pred_col = cols[0]

        for uid, g in df.groupby("unique_id"):
            r = get_range(uid)
            if not r: continue
            
            p = g[pred_col]
            outliers = g[(p < r[0]) | (p > r[1])]
            if not outliers.empty:
                print(f"!!! FAIL: Run {run_id} | Series {uid} | {len(outliers)} outliers found.")
                print(f"    Range: {r} | Min/Max pred: {p.min():.2f} / {p.max():.2f}")
                print(f"    Sample outliers:\n{outliers[['ds', 'y', pred_col]].head(3)}")
            else:
                pass
                # print(f"PASS: Run {run_id} | Series {uid}")

if __name__ == "__main__":
    main()
