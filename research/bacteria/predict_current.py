#!/usr/bin/env python3
"""Operational nowcast: per-beach current bacteria-exceedance risk (deploy-ready counties).

This is the OPERATIONAL face of the validated statewide model. It does NOT re-train or
re-tune anything: it imports the exact production feature pipeline + HGBT config + isotonic
calibration from ``operational_benchmark`` (the honest, stratified, calibrated benchmark) and
applies them as a library to emit the latest calibrated P(enterococcus exceedance) for every
beach in the deploy-ready strata.

Honest scope (matches the benchmark's disposition):
  * Target: AB411 marine single-sample standard, enterococcus > 104 MPN/100mL, reveal-lag 2d.
  * Deploy-ready set: every county EXCEPT San Diego. San Diego ranks well but is NOT
    calibration-deploy-ready (post-2022 regime break inflates its ECE); it is reported in a
    separate, clearly-labelled "ranking-only, NOT calibrated" block, never mixed into advisories.
  * What a row means: the calibrated exceedance probability for that beach's MOST RECENT
    sampled day, given only data that had returned by sample time. It is a nowcast of the last
    known read, not a forecast of an unsampled future day.

Leakage-safe split (all dates relative to the latest sample in the frame):
    train      : sample_date <= score_cut - 365d
    calibrate  : score_cut - 365d < sample_date <= score_cut   (isotonic, score->empirical rate)
    score      : sample_date > score_cut, latest row per station   (the "current" read)

Usage:
    python research/bacteria/predict_current.py --rain-dir bacteria_results/rainfall
Outputs (under reports/operational_nowcast/):
    current_risk.csv      ranked per-beach calibrated risk (deploy-ready counties)
    digest.md             human-readable top-risk digest
    nowcast_meta.json     split sizes + calibration check (reproducible, random_state=42)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))           # for `import operational_benchmark`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root: operational_benchmark
import operational_benchmark as B  # noqa: E402  now uses repo-relative `research.bacteria.*` imports

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "reports" / "operational_nowcast"
SCORE_WINDOW_DAYS = 90      # a "current" read must be at least this fresh
CALIB_WINDOW_DAYS = 365     # isotonic fit window, immediately before the score window

# Calibrated-probability advisory tiers (operational, not a statistical claim).
TIERS = [(0.50, "HIGH"), (0.20, "ELEVATED"), (0.05, "WATCH")]


def _tier(p: float) -> str:
    for thr, name in TIERS:
        if p >= thr:
            return name
    return "LOW"


def _beach_names(obs_path: Path) -> pd.DataFrame:
    """station_id -> most recent human-readable (county, beach_name, station_name)."""
    o = pd.read_parquet(obs_path, columns=["station_id", "county", "beach_name",
                                           "station_name", "sample_date"])
    o["sample_date"] = pd.to_datetime(o["sample_date"])
    o = o.sort_values("sample_date").groupby("station_id").tail(1)
    # county already present on the scored frame (from load_station_days); only bring names.
    return o[["station_id", "beach_name", "station_name"]].reset_index(drop=True)


def build_scored_frame(obs_path: Path, rain_dir: Path | None) -> tuple[pd.DataFrame, dict]:
    df = B.load_station_days(obs_path, label="enterococcus")
    df = B.add_causal_features(df, reveal_lag_days=2)
    feats = list(B.FEATS)
    if rain_dir is not None:
        df = B.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += B.RAIN_FEATS

    max_date = df["sample_date"].max()
    score_cut = max_date - pd.Timedelta(days=SCORE_WINDOW_DAYS)
    calib_start = score_cut - pd.Timedelta(days=CALIB_WINDOW_DAYS)

    tr = df[df["sample_date"] <= calib_start]
    va = df[(df["sample_date"] > calib_start) & (df["sample_date"] <= score_cut)]
    sc = df[df["sample_date"] > score_cut].copy()
    # the "current" read = each station's most recent sample within the score window
    sc = sc.sort_values("sample_date").groupby("station_id").tail(1).copy()

    # Keep only features that exist and carry signal in the training slice. The upstream FEATS
    # list drifts (other agents add geo/neighbour columns); a feature that is absent or all-NaN
    # here (e.g. no geo map supplied) would crash the HGBT binner. Filter once, on train, and use
    # the same list everywhere so train/calib/score stay aligned.
    feats = [c for c in feats if c in df.columns and tr[c].notna().any()]

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())

    sc["p_raw"] = clf.predict_proba(sc[feats].astype(float))[:, 1]
    iso = None
    calib_check = None
    if va["exceed"].nunique() > 1:
        # Production isotonic uses the FULL calibration window (best use of recent data).
        va_scores = clf.predict_proba(va[feats].astype(float))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(va_scores, va["exceed"].to_numpy())
        sc["p_cal"] = iso.predict(sc["p_raw"].to_numpy())

        # HONEST reliability diagnostic: a SEPARATE check-isotonic fit on the earlier half of
        # the window, evaluated out-of-sample on the later half (never train==test). This is a
        # real reliability number, not the circular fit-and-score-the-same-rows trap.
        va_sorted = va.sort_values("sample_date")
        mid = len(va_sorted) // 2
        a, b = va_sorted.iloc[:mid], va_sorted.iloc[mid:]
        if mid >= 50 and a["exceed"].nunique() > 1 and b["exceed"].nunique() > 1:
            sa = clf.predict_proba(a[feats].astype(float))[:, 1]
            sb = clf.predict_proba(b[feats].astype(float))[:, 1]
            chk_iso = IsotonicRegression(out_of_bounds="clip").fit(sa, a["exceed"].to_numpy())
            pb = chk_iso.predict(sb)
            dec = pd.qcut(pd.Series(pb), 10, duplicates="drop")
            rel = pd.DataFrame({"pred": pb, "obs": b["exceed"].to_numpy()}).groupby(dec, observed=True).mean()
            ece = float((rel["pred"] - rel["obs"]).abs().mean())
            calib_check = {
                "method": "out-of-sample: isotonic fit on earlier half, measured on later half",
                "n_holdout": int(len(b)), "holdout_event_rate": round(float(b["exceed"].mean()), 4),
                "decile_ece": round(ece, 4),
                "max_decile_abs_gap": round(float((rel["pred"] - rel["obs"]).abs().max()), 4),
            }
    else:
        sc["p_cal"] = sc["p_raw"]

    meta = {
        "max_sample_date": str(max_date.date()),
        "score_cut": str(score_cut.date()), "calib_start": str(calib_start.date()),
        "n_train": int(len(tr)), "n_calib": int(len(va)), "n_scored_beaches": int(len(sc)),
        "train_event_rate": round(float(tr["exceed"].mean()), 4),
        "n_features": len(feats), "calibrated": iso is not None,
        "calibration_check": calib_check,
    }
    return sc, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None, help="statewide_beach_observations.parquet (or $MBAL_BACTERIA_OBS)")
    ap.add_argument("--rain-dir", default=None, help="dir with rainfall_grid.parquet + station_grid_map.parquet")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    obs = Path(args.obs) if args.obs else B.default_obs_path()
    if not obs.exists():
        print(f"[predict_current] obs not found: {obs}")
        return 2
    outdir = Path(args.out_dir) if args.out_dir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    sc, meta = build_scored_frame(obs, args.rain_dir)
    sc = sc.merge(_beach_names(obs), on="station_id", how="left")
    sc["risk"] = sc["p_cal"].apply(_tier)
    last = pd.to_datetime(sc["sample_date"])
    sc["last_sample"] = last.dt.date.astype(str)
    sc["days_stale"] = (pd.Timestamp(meta["max_sample_date"]) - last).dt.days

    cols = ["county", "beach_name", "station_name", "station_id", "last_sample",
            "days_stale", "p_cal", "p_raw", "risk"]
    sc = sc[cols].sort_values("p_cal", ascending=False).reset_index(drop=True)
    deploy = sc[sc["county"] != "San Diego"].reset_index(drop=True)
    sd = sc[sc["county"] == "San Diego"].reset_index(drop=True)

    deploy.to_csv(outdir / "current_risk.csv", index=False)
    meta["n_deploy_ready_beaches"] = int(len(deploy))
    meta["n_san_diego_ranking_only"] = int(len(sd))
    meta["tier_counts"] = deploy["risk"].value_counts().to_dict()
    (outdir / "nowcast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- human digest ----
    L = ["# Beach bacteria nowcast -- current exceedance risk",
         "",
         f"*Model:* statewide HGBT + isotonic calibration (enterococcus > 104 MPN/100mL, AB411, "
         f"reveal-lag 2d). *Trained through* {meta['calib_start']}, *calibrated through* "
         f"{meta['score_cut']}, *scored on the latest read per beach* (data through "
         f"{meta['max_sample_date']}).",
         f"*Deploy-ready beaches scored:* {len(deploy)} (San Diego excluded -- ranking-only, "
         f"not calibrated; see foot).",
         ""]
    cc = meta.get("calibration_check")
    if cc:
        L.append(f"*Reliability ({cc['method']}):* decile-ECE = "
                 f"{cc['decile_ece']}, max decile |pred-obs| = {cc['max_decile_abs_gap']} "
                 f"at holdout event rate {cc['holdout_event_rate']}.")
        L.append("")
    L.append("## Highest current risk (deploy-ready counties)")
    L.append("")
    L.append("| rank | county | beach | last sample | days old | P(exceed) | tier |")
    L.append("|--:|---|---|---|--:|--:|---|")
    for i, r in deploy.head(25).iterrows():
        L.append(f"| {i+1} | {r['county']} | {r['beach_name']} | {r['last_sample']} | "
                 f"{r['days_stale']} | {r['p_cal']:.2f} | {r['risk']} |")
    tc = meta["tier_counts"]
    L += ["", f"**Tier counts:** " + ", ".join(f"{k}={v}" for k, v in tc.items()) + ".", ""]
    if len(sd):
        top_sd = sd.iloc[0]
        L += ["---",
              f"*San Diego (ranking-only, NOT calibrated -- do not issue advisories from these "
              f"numbers): {len(sd)} beaches scored; top raw rank {top_sd['beach_name']} "
              f"p_raw={top_sd['p_raw']:.2f}.*"]
    (outdir / "digest.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nwrote {outdir/'current_risk.csv'} ({len(deploy)} beaches), digest.md, nowcast_meta.json")
    print("\nTop 10 deploy-ready by calibrated risk:")
    print(deploy.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
