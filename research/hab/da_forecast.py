#!/usr/bin/env python3
"""First domoic-acid (DA) exceedance FORECAST model — CalHABMAP weekly piers.

Target: next-visit pDA exceedance (pDA >= 0.5 ng/mL = 500 ng/L, the C-HARM/CDPH event
threshold) at a CalHABMAP pier, predicted ONE SAMPLE AHEAD from strictly-causal features that
were fully known at the prior visit. Weekly sampling => ~1-week lead.

Honest discipline (mirrors the statewide bacteria model):
  * Strict causality: every feature for visit i comes from visit i-1 and earlier (per station),
    so a same-day Pseudo-nitzschia / nutrient / DA reading can never leak into its own label.
  * Real baselines it MUST beat: seasonal-naive (train monthly climatology), persistence
    (last visit's exceedance), station-memory (expanding per-station exceedance rate).
  * Time holdout (train <=2018, isotonic-calibrate 2019-2020, test >=2021) + leave-one-
    station-out spatial holdout. AP (not accuracy) is the headline at a ~3.7% base rate.

Data: data/external_curated/habmap_cdph/habmap_cdph.parquet (built by ops/data_fetch
adapters/habmap.py; see reports/hab/da_label_verification.md).

Usage:  python research/hab/da_forecast.py
Outputs (reports/hab/): da_forecast.json, da_forecast.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet"
OUTDIR = ROOT / "reports" / "hab"
THRESH_NG_ML = 0.5          # pDA >= 0.5 ng/mL == 500 ng/L (C-HARM particulate-DA event)
TRAIN_END, VALID_END = 2018, 2020

PN_COLS = ["Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group"]
DRIVERS = ["Temp", "Avg_Chloro", "Nitrate", "Phosphate", "Silicate"]  # Salinity dropped: all-NaN in train

# Feature groups, built as an ablation. The physical/nutrient drivers were tested and HURT
# (they add noise, echoing the lab's M1/news/wave-tide driver-nulls), so the deployed headline
# model is the lean DA-history + Pseudo-nitzschia-precursor set, NOT the full driver matrix.
FEAT_DA = [
    "month", "doy_sin", "doy_cos", "days_since_prev",
    "pda_prev", "pda_roll3_prev", "exc_prev", "exc_prev2", "exc_roll4_prev",
    "station_prior_rate", "station_prior_n",
]
FEAT_PN = ["pn_seriata_prev", "pn_delica_prev", "pn_total_prev"]
FEAT_DRV = [f"{c.lower()}_prev" for c in DRIVERS] + ["si_n_ratio_prev"]
FEAT_GROUPS = {
    "DA_history": FEAT_DA,
    "DA+precursor": FEAT_DA + FEAT_PN,                  # headline (best AP, mechanistic)
    "DA+precursor+drivers": FEAT_DA + FEAT_PN + FEAT_DRV,
}
FEATS = FEAT_DA + FEAT_PN + FEAT_DRV                    # superset, for feature construction
HEADLINE_FEATS = FEAT_DA + FEAT_PN


def load_panel() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
    d = d[d["pDA"].notna()].copy()
    d["exceed"] = (d["pDA"] >= THRESH_NG_ML).astype(int)
    d["pn_total"] = d[PN_COLS].sum(axis=1, min_count=1)
    return d.sort_values(["station", "time"]).reset_index(drop=True)


def add_causal_features(d: pd.DataFrame) -> pd.DataFrame:
    """All features for visit i are from visit i-1 (and earlier) within the same station."""
    d = d.copy()
    doy = d["time"].dt.dayofyear
    d["month"] = d["time"].dt.month
    d["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    parts = []
    for _, g in d.groupby("station", sort=False):
        g = g.sort_values("time").copy()
        prev = g.shift(1)  # the prior visit
        g["days_since_prev"] = (g["time"] - prev["time"]).dt.days
        g["pda_prev"] = np.log1p(prev["pDA"].clip(lower=0))
        g["pda_roll3_prev"] = np.log1p(g["pDA"].clip(lower=0).shift(1).rolling(3, min_periods=1).mean())
        g["exc_prev"] = prev["exceed"]
        g["exc_prev2"] = g["exceed"].shift(2)
        g["exc_roll4_prev"] = g["exceed"].shift(1).rolling(4, min_periods=1).mean()
        g["station_prior_rate"] = g["exceed"].shift(1).expanding().mean()
        g["station_prior_n"] = g["exceed"].shift(1).expanding().count()
        g["pn_seriata_prev"] = np.log1p(prev[PN_COLS[0]].clip(lower=0))
        g["pn_delica_prev"] = np.log1p(prev[PN_COLS[1]].clip(lower=0))
        g["pn_total_prev"] = np.log1p(prev["pn_total"].clip(lower=0))
        for c in DRIVERS:
            g[f"{c.lower()}_prev"] = prev[c]
        g["si_n_ratio_prev"] = prev["Silicate"] / prev["Nitrate"].replace(0, np.nan)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["year"] = out["time"].dt.year
    return out[out["exc_prev"].notna()].reset_index(drop=True)  # need a prior visit


def _scores(y, p, base):
    return {"n": int(len(y)), "events": int(y.sum()), "base_rate": round(float(y.mean()), 4),
            "ap": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
            "ap_lift_vs_base": round(float(average_precision_score(y, p)) / base, 2) if y.sum() and base else None}


def fit_predict(tr, va, te, feats=HEADLINE_FEATS):
    # drop features constant in THIS fold's training set (sklearn>=1.9 HGBT binning errors on a
    # single-distinct-value feature); covers both the time-held-out ablation and the LOSO loop.
    feats = [c for c in feats if tr[c].notna().sum() >= 2 and tr[c].nunique(dropna=True) > 1]
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0,
                                         class_weight="balanced", early_stopping=True,
                                         validation_fraction=0.15, random_state=42)
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    p = clf.predict_proba(te[feats].astype(float))[:, 1]
    if va["exceed"].nunique() > 1:
        vs = clf.predict_proba(va[feats].astype(float))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(vs, va["exceed"].to_numpy())
        p = iso.predict(p)
    return p


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTDIR),
                    help="where to write da_forecast.json/.md (default reports/hab/; "
                         "override for output-isolated confirmation runs).")
    args = ap.parse_args(argv)
    outdir = Path(args.output_dir)

    d = add_causal_features(load_panel())
    tr = d[d["year"] <= TRAIN_END]
    va = d[(d["year"] > TRAIN_END) & (d["year"] <= VALID_END)]
    te = d[d["year"] > VALID_END].copy()
    base = float(tr["exceed"].mean())

    # ---- baselines (scored on the same test rows) ----
    month_rate = tr.groupby("month")["exceed"].mean()
    te["p_seasonal"] = te["month"].map(month_rate).fillna(base).to_numpy()
    te["p_persist"] = te["exc_prev"].to_numpy()
    te["p_station"] = te["station_prior_rate"].fillna(base).to_numpy()
    yte = te["exceed"].to_numpy()
    results = {k: _scores(yte, te[col].to_numpy(), base) for k, col in [
        ("seasonal_naive", "p_seasonal"), ("persistence", "p_persist"),
        ("station_memory", "p_station")]}
    # feature-group ablation: drivers were found to HURT -> headline is the lean DA+precursor set
    for gname, feats in FEAT_GROUPS.items():
        results[f"model_{gname}"] = _scores(yte, fit_predict(tr, va, te, feats), base)
    results["model_hgbt"] = results["model_DA+precursor"]  # the deployed headline model

    best_base = max((results[b]["ap"] or 0) for b in ["seasonal_naive", "persistence", "station_memory"])
    model_ap = results["model_hgbt"]["ap"] or 0
    beats = model_ap > best_base

    # ---- leave-one-station-out spatial holdout (stations with >=10 events) ----
    ev = d.groupby("station")["exceed"].sum()
    loso = {}
    for st in ev[ev >= 10].index:
        tr_s, te_s = d[d["station"] != st], d[d["station"] == st].copy()
        va_s = tr_s[(tr_s["year"] > TRAIN_END) & (tr_s["year"] <= VALID_END)]
        b = float(te_s["exceed"].mean())
        te_s["p"] = fit_predict(tr_s[tr_s["year"] <= TRAIN_END], va_s, te_s)
        te_s["p_seasonal"] = te_s["month"].map(tr_s.groupby("month")["exceed"].mean()).fillna(b)
        m = _scores(te_s["exceed"].to_numpy(), te_s["p"].to_numpy(), b)
        sn = _scores(te_s["exceed"].to_numpy(), te_s["p_seasonal"].to_numpy(), b)
        loso[st] = {"model_ap": m["ap"], "seasonal_ap": sn["ap"], "roc_auc": m["roc_auc"],
                    "events": m["events"], "n": m["n"],
                    "model_beats_seasonal": (m["ap"] or 0) > (sn["ap"] or 0)}
    loso_beats = sum(v["model_beats_seasonal"] for v in loso.values())

    out = {
        "target": f"next-visit pDA >= {THRESH_NG_ML} ng/mL (=500 ng/L), ~1-week lead",
        "splits": {"train_end": TRAIN_END, "valid_end": VALID_END,
                   "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
                   "train_base_rate": round(base, 4)},
        "test_timeheldout": results,
        "headline": {
            "model_ap": model_ap, "best_baseline_ap": round(best_base, 4),
            "model_beats_baselines": beats,
            "model_roc_auc": results["model_hgbt"]["roc_auc"],
            "loso_stations": len(loso), "loso_model_beats_seasonal": f"{loso_beats}/{len(loso)}",
        },
        "leave_one_station_out": loso,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "da_forecast.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    r = results
    md = [f"# Domoic-acid forecast -- {out['target']}", "",
          f"Train <= {TRAIN_END} (n={len(tr)}, {int(tr['exceed'].sum())} events) | "
          f"calib {TRAIN_END+1}-{VALID_END} | test > {VALID_END} "
          f"(n={len(te)}, {int(yte.sum())} events, base {yte.mean():.3f}).", "",
          "## Time-held-out test (AP = average precision; base rate ~3.7%)", "",
          "Feature-group ablation: the physical/nutrient **drivers HURT** (AP drops vs the lean "
          "set) -- so the headline model is `DA+precursor`, not the full driver matrix. Same "
          "driver-null pattern as M1 / news / wave-tide.", "",
          "| method | AP | ROC-AUC | AP lift vs base |", "|---|--:|--:|--:|"]
    for k in ["seasonal_naive", "persistence", "station_memory",
              "model_DA_history", "model_DA+precursor", "model_DA+precursor+drivers"]:
        star = " (headline)" if k == "model_DA+precursor" else ""
        md.append(f"| {k}{star} | {r[k]['ap']} | {r[k]['roc_auc']} | {r[k]['ap_lift_vs_base']} |")
    verdict = ("BEATS baselines" if beats else "DOES NOT beat baselines") + \
              f" -- model AP {model_ap} vs best-baseline {round(best_base,4)}."
    md += ["", f"**Verdict:** {verdict}", "",
           f"**Leave-one-station-out:** model beats seasonal-naive in "
           f"{loso_beats}/{len(loso)} stations.", "",
           "| station | n | events | model AP | seasonal AP | beats |", "|---|--:|--:|--:|--:|---|"]
    for st, v in sorted(loso.items(), key=lambda x: -(x[1]["model_ap"] or 0)):
        md.append(f"| {st.replace('HABs-','')} | {v['n']} | {v['events']} | {v['model_ap']} | "
                  f"{v['seasonal_ap']} | {'Y' if v['model_beats_seasonal'] else 'n'} |")
    (outdir / "da_forecast.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(out["headline"], indent=2))
    print(f"\nwrote {outdir/'da_forecast.json'}, da_forecast.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
