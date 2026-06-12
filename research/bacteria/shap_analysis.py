#!/usr/bin/env python3
"""
SHAP Analysis for the Spatial Bacteria Model.
Computes feature importances for the best-performing spatial arm to explain predictions.
"""

import os
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob
from research.bacteria.spatial_drivers_experiment import add_knn_spatial_lag
from research.bacteria.spatial_autocorr import _load_geo

def main():
    obs_path = ob.default_obs_path()
    rain_dir = _REPO_ROOT / "bacteria_results" / "rainfall"
    
    print("Loading data...")
    df = ob.load_station_days(obs_path, label="enterococcus")
    # Honest evaluation: exclude San Diego due to 2022 lab breakpoint
    df = df[df["county"] != "San Diego"].copy()
    df = ob.add_causal_features(df, reveal_lag_days=2)
    df = ob.add_rain_features(df, rain_dir)
    
    geo = _load_geo()
    df = add_knn_spatial_lag(df, geo, k=8, reveal_lag_days=2)
    
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    df = df.merge(geo_t[["station_id", "latitude", "longitude"]], on="station_id", how="left")
    
    # Target features
    feats = list(ob.FEATS)
    if "rain_3d" in df.columns:
        feats += ob.RAIN_FEATS
    feats += ["nbr_prev1", "nbr_prev7", "latitude", "longitude"]
    
    # Temporal split exactly matching spatial_drivers_experiment.py
    tr = df[df["sample_date"] <= "2019-12-31"].copy()
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    
    # Drop rows where all features are NaN just in case, though HGBT handles NaNs
    X_train = tr[feats].astype(float)
    y_train = tr["exceed"].to_numpy()
    X_test = te[feats].astype(float)
    
    print(f"Training on {len(X_train)} rows, explaining {len(X_test)} rows...")
    
    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # SHAP Explainer
    print("Computing SHAP values...")
    try:
        # TreeExplainer is fast but sometimes has issues with sklearn HGBT depending on version
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"TreeExplainer failed ({e}). Falling back to general Explainer/Permutation...")
        # Since Explainer on tabular is slow for many rows, we take a random sample
        sample_idx = X_test.sample(min(1000, len(X_test)), random_state=42).index
        X_test_sample = X_test.loc[sample_idx]
        explainer = shap.Explainer(clf.predict, X_train.sample(min(2000, len(X_train)), random_state=42))
        shap_values = explainer(X_test_sample)
        X_test = X_test_sample

    out_dir = _REPO_ROOT / "reports" / "model_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Summary Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_summary.png")
    
    # Plot 2: Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Directional Impact of Features")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_beeswarm.png")
    
    # Compute mean absolute SHAP values for JSON output
    vals = shap_values.values
    if len(vals.shape) == 3: # If multi-class output, take class 1 (positive)
        vals = vals[:, :, 1]
    mean_shap = dict(zip(feats, [float(x) for x in abs(vals).mean(axis=0)]))
    mean_shap_sorted = dict(sorted(mean_shap.items(), key=lambda item: item[1], reverse=True))
    
    (out_dir / "shap_importances.json").write_text(json.dumps(mean_shap_sorted, indent=2), encoding="utf-8")
    print("Saved shap_importances.json")

if __name__ == "__main__":
    main()
