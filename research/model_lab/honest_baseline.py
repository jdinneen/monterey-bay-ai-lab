#!/usr/bin/env python3
"""Honest skill baseline — persistence AND seasonal-naive (same-hour-yesterday).

Tracked, committable companion to mbari_neural_merge.py. The published leaderboard scores
skill vs PERSISTENCE only, which is unfairly weak at diurnal horizons (+24/72/168h) where
"yesterday's value" is a strong free baseline. This module re-scores every full-history model
against the BETTER of {persistence, seasonal-naive} — the baseline a reviewer actually demands —
and writes HONEST_SKILL_BASELINE.md so the real margin is version-controlled, not buried in
untracked scratch scripts.

Read-only over nn_cache (panel+mask) and nn_results/<model>/cv_predictions.parquet. No training.
Scoring matches mbari_neural_forecast.evaluate(): observed target points only; persistence = obs
at the forecast origin; seasonal-naive = obs at ds-24h; all baselines + model on a common subset.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
NNC = PROJECT_ROOT / "nn_cache"
NNR = PROJECT_ROOT / "nn_results"
OUT = Path(__file__).resolve().parent / "HONEST_SKILL_BASELINE.md"
HORIZONS = [1, 6, 24, 72, 168]
# Full-history production models only (exclude smoke/ablate/bounded dirs). chronos = foundation.
MODELS = {
    "itransformer": NNR / "itransformer", "nbeatsx": NNR / "nbeatsx", "nhits": NNR / "nhits",
    "patchtst": NNR / "patchtst", "tft": NNR / "tft", "tsmixerx": NNR / "tsmixerx",
    "chronos": NNR / "chronos_full",
}


def load_panel():
    long_df = pd.read_parquet(NNC / "long_v2_past_only_fill_origin_observed_full.parquet")
    mask = pd.read_parquet(NNC / "mask_v2_past_only_fill_origin_observed_full.parquet")
    long_df["ds"] = pd.to_datetime(long_df["ds"]); mask["ds"] = pd.to_datetime(mask["ds"])
    obs = long_df.merge(mask, on=["unique_id", "ds"], how="left")
    obs = obs[obs["observed"].fillna(False)][["unique_id", "ds", "y"]]
    return mask, obs.set_index(["unique_id", "ds"])["y"]


def load_preds(model_dir: Path):
    df = pd.read_parquet(model_dir / "cv_predictions.parquet")
    df["ds"] = pd.to_datetime(df["ds"]); df["cutoff"] = pd.to_datetime(df["cutoff"])
    pcol = [c for c in df.columns if c not in ("unique_id", "ds", "cutoff", "y", "observed", "cache_version")][0]
    return df, pcol


def score(cv, model_col, mask, obs_val):
    cv = cv.merge(mask, on=["unique_id", "ds"], how="left")
    cv = cv[cv["observed"].fillna(False)].copy()
    cv["horizon_h"] = ((cv["ds"] - cv["cutoff"]) / pd.Timedelta(hours=1)).round().astype(int)
    cv = cv[cv["horizon_h"].isin(HORIZONS)].copy()
    # Derive the truth from the observed panel (some pred files, e.g. chronos, omit 'y').
    cv["y"] = obs_val.reindex(pd.MultiIndex.from_arrays([cv["unique_id"], cv["ds"]])).to_numpy()
    cv["y_persist"] = obs_val.reindex(pd.MultiIndex.from_arrays([cv["unique_id"], cv["cutoff"]])).to_numpy()
    cv["y_seas"] = obs_val.reindex(pd.MultiIndex.from_arrays([cv["unique_id"], cv["ds"] - pd.Timedelta(hours=24)])).to_numpy()
    cv = cv.dropna(subset=[model_col, "y", "y_persist", "y_seas"])
    rows = []
    for (uid, hh), g in cv.groupby(["unique_id", "horizon_h"]):
        if len(g) < 3:
            continue
        rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        mr = rmse(g[model_col].to_numpy(), g["y"].to_numpy())
        pr = rmse(g["y_persist"].to_numpy(), g["y"].to_numpy())
        sr = rmse(g["y_seas"].to_numpy(), g["y"].to_numpy())
        best = min(pr, sr)
        rows.append(dict(unique_id=uid, horizon_h=hh,
                         skill_vs_persist=100 * (1 - mr / pr) if pr else np.nan,
                         skill_vs_seasonal=100 * (1 - mr / sr) if sr else np.nan,
                         skill_vs_bestnaive=100 * (1 - mr / best) if best else np.nan))
    return pd.DataFrame(rows)


def main():
    mask, obs_val = load_panel()
    scored = {}
    for name, d in MODELS.items():
        if (d / "cv_predictions.parquet").exists():
            cv, pcol = load_preds(d)
            scored[name] = score(cv, pcol, mask, obs_val)
    big = pd.concat([s.assign(model=n) for n, s in scored.items()], ignore_index=True)

    def med(df, col):
        return df.groupby("horizon_h")[col].median().reindex(HORIZONS).round(1)

    # best model per (series,horizon) by skill vs best-naive — the honest leaderboard
    best = big.sort_values("skill_vs_bestnaive").groupby(["unique_id", "horizon_h"]).tail(1)

    L = ["# MBARI — Honest Skill Baseline (persistence vs seasonal-naive)", "",
         "Skill vs PERSISTENCE alone is unfairly easy at diurnal horizons (+24/72/168h), where",
         "seasonal-naive (same-hour-yesterday) is a strong free baseline. The honest metric is skill",
         "vs the BETTER of the two. Median across the 24 series; full-history runs only.", "",
         "## Best-model-per-cell median skill (%) by horizon", "",
         "| baseline | +1h | +6h | +24h | +72h | +168h |",
         "|---|---|---|---|---|---|"]
    for col, lab in [("skill_vs_persist", "vs persistence"), ("skill_vs_seasonal", "vs seasonal-naive"),
                     ("skill_vs_bestnaive", "vs BEST-naive (honest)")]:
        m = med(best, col)
        L.append(f"| {lab} | " + " | ".join(f"{m.get(h):.1f}" if pd.notna(m.get(h)) else "—" for h in HORIZONS) + " |")
    L += ["", "**Read:** genuine skill exists only at +1h/+6h. At +24h it is a wash; at +72h/+168h the",
          "models are worse than seasonal-naive (negative vs best-naive) and should fall back to it.", "",
          "## Per-model median skill vs BEST-naive (%) by horizon", "",
          "| model | +1h | +6h | +24h | +72h | +168h |", "|---|---|---|---|---|---|"]
    for name in scored:
        m = med(big[big.model == name], "skill_vs_bestnaive")
        L.append(f"| {name} | " + " | ".join(f"{m.get(h):.1f}" if pd.notna(m.get(h)) else "—" for h in HORIZONS) + " |")
    L += ["", f"Models scored: {sorted(scored)}. Generated by research/model_lab/honest_baseline.py.",
          "Note: numbers are in-sample expanding-window CV (52 weekly cutoffs), not a held-out test split."]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[out] wrote {OUT}")


if __name__ == "__main__":
    main()
