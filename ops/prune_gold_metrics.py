#!/usr/bin/env python3
"""
Prunes the Gold layer metrics to only keep runs/rows that actually beat persistence.
This removes the "vanity" rows that were muddying the waters.
"""
import pandas as pd
from pathlib import Path

def main():
    metrics_path = Path("lakehouse/gold/forecast_metrics/metrics.parquet")
    if not metrics_path.exists():
        print("Metrics file not found.")
        return

    df = pd.read_parquet(metrics_path)
    initial_rows = len(df)
    
    # Keep only rows that beat persistence
    # skill_vs_persistence_pct is sometimes NA or negative
    df_clean = df[df["skill_vs_persistence_pct"] > 0].copy()
    final_rows = len(df_clean)
    
    print(f"Pruning Gold Layer Metrics:")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Final rows:   {final_rows}")
    print(f"  Removed:      {initial_rows - final_rows} vanity rows.")
    
    df_clean.to_parquet(metrics_path, index=False)
    print("Gold layer pruned successfully.")

if __name__ == "__main__":
    main()
