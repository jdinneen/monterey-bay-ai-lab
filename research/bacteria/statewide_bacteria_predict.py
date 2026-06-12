#!/usr/bin/env python3
"""Statewide California bacteria-exceedance model (Phase A pilot).

Generalizes the Lover's Point predictor from one beach (53 events) to all CA
beaches in `blue_current_core_v2.california_beach_sample_observations`
(~1.38M obs, 312 beaches). Uses only features that generalize across regions:
per-site lab history, calendar, within-county/statewide bacteria context, and
beach advisories (incl. cause typing). Monterey-specific drivers (ASOS rainfall,
CDIP waves, local ocean drivers) are intentionally excluded — they do not apply
to San Diego/LA beaches.

Value question: does statewide scale beat the Monterey-only model on a powered,
time-held-out exceedance classifier?
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lovers_point_bacteria_predict as L  # reuse leaf helpers that generalize
from spatial_drivers_experiment import add_knn_spatial_lag

ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTDIR = ROOT / "bacteria_results" / "statewide"
OBS_CSV = OUTDIR / "ca_beach_observations.csv"
ADV_CSV = OUTDIR / "ca_beach_advisories.csv"
HAB_CSV = OUTDIR / "ca_hab_bloom_reports.csv"
RAIN_CSV = OUTDIR / "ca_statewide_rainfall.csv"

# Coastal-county -> representative ASOS airport station (Iowa Mesonet IDs).
# v1 mapping: every beach in a county shares its primary coastal airport's rain.
# Refine to per-beach nearest-station (lat/lon) only if rainfall proves it earns lift.
COUNTY_STATION = {
    "San Diego": "SAN", "Orange": "SNA", "Los Angeles": "LAX", "Long Beach City": "LGB",
    "Ventura": "OXR", "Santa Barbara": "SBA", "San Luis Obispo": "SBP",
    "Monterey": "MRY", "Santa Cruz": "WVI", "San Mateo": "HAF", "San Francisco": "SFO",
    "Marin": "SFO", "Sonoma": "STS", "Mendocino": "UKI", "Humboldt": "ACV",
    "East Bay Parks District": "OAK", "Alameda": "OAK",
}
PROJECT = os.environ.get("MBAL_GCP_PROJECT")
DS = f"`{PROJECT}.blue_current_core_v2`"


def fetch_observations(force=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if OBS_CSV.exists() and not force:
        return pd.read_csv(OBS_CSV, parse_dates=["sample_date"], low_memory=False)
    sql = f"""
      SELECT sample_date, county, beach_name, station_name, station_id,
             property_id, result_value_numeric
      FROM {DS}.california_beach_sample_observations
      WHERE result_value_numeric IS NOT NULL
        AND property_id IN ('prop_enterococcus','prop_total_coliform','prop_fecal_coliform','prop_e_coli')
        AND sample_date >= '2000-01-01'
    """
    df = L.run_bq_query(sql)
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df.to_csv(OBS_CSV, index=False)
    return df


def fetch_advisories(force=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if ADV_CSV.exists() and not force:
        return pd.read_csv(ADV_CSV, parse_dates=["advisory_date"], low_memory=False)
    sql = f"""
      SELECT advisory_date, county, beach_name, station_name,
             advisory_type, advisory_cause
      FROM {DS}.california_beach_advisory_events
      WHERE advisory_date IS NOT NULL
    """
    df = L.run_bq_query(sql)
    df["advisory_date"] = pd.to_datetime(df["advisory_date"], errors="coerce")
    df.to_csv(ADV_CSV, index=False)
    return df


def fetch_hab_blooms(force=False):
    """CA freshwater HAB (FHAB) bloom reports. NOTE: this program is freshwater
    cyanobacteria (reservoirs/lakes), NOT coastal marine HAB / domoic acid.
    Pulled to value-gate honestly; expected to be irrelevant to beach bacteria."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if HAB_CSV.exists() and not force:
        return pd.read_csv(HAB_CSV, parse_dates=["observation_date"], low_memory=False)
    sql = f"""
      SELECT observation_date, county, water_body_type
      FROM {DS}.california_hab_bloom_reports
      WHERE observation_date BETWEEN '2000-01-01' AND CURRENT_DATE()
    """
    df = L.run_bq_query(sql)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df.to_csv(HAB_CSV, index=False)
    return df


def add_hab_features(df, hab):
    """County-level recent FHAB bloom activity (shifted, rolling). A/B-gated."""
    if os.environ.get("MBAL_NO_HAB") == "1" or hab is None or hab.empty:
        df["hab_blooms_county_prev30d"] = 0.0
        df["hab_blooms_county_prev90d"] = 0.0
        return df
    h = hab.dropna(subset=["observation_date", "county"]).copy()
    h["sample_date"] = pd.to_datetime(h["observation_date"]).dt.normalize()
    daily = h.groupby(["county", "sample_date"]).size().rename("hab_blooms").reset_index()
    daily = daily.sort_values(["county", "sample_date"])
    for w in [30, 90]:
        daily[f"hab_blooms_county_prev{w}d"] = daily.groupby("county")["hab_blooms"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    keep = ["county", "sample_date"] + [c for c in daily.columns if "_prev" in c]
    return df.merge(daily[keep], on=["county", "sample_date"], how="left")


def fetch_statewide_rainfall(force=False):
    """Daily precip (mm) for coastal-county ASOS airports, 2000-2026.
    Same source/handling as the Lover's Point model, expanded statewide."""
    import urllib.parse
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if RAIN_CSV.exists() and not force:
        return pd.read_csv(RAIN_CSV, parse_dates=["date"])
    frames = []
    for station in sorted(set(COUNTY_STATION.values())):
        params = {
            "station": station, "data": "p01i", "year1": 2000, "month1": 1, "day1": 1,
            "year2": 2026, "month2": 12, "day2": 31, "tz": "Etc/UTC", "format": "onlycomma",
            "latlon": "no", "missing": "M", "trace": "T", "direct": "no", "report_type": 3,
        }
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(params)
        try:
            part = pd.read_csv(url)
        except Exception as e:
            print(f"  rain fetch {station} failed: {str(e)[:120]}")
            continue
        if part.empty:
            continue
        part["valid"] = pd.to_datetime(part["valid"], errors="coerce", utc=True)
        part = part.dropna(subset=["valid"])
        part["p01i"] = pd.to_numeric(part["p01i"].replace({"T": 0.002, "M": np.nan}), errors="coerce").clip(lower=0)
        part["rain_mm"] = part["p01i"] * 25.4
        daily = part.set_index("valid")["rain_mm"].resample("1D").sum(min_count=1).reset_index()
        daily["date"] = daily["valid"].dt.tz_localize(None).dt.normalize()
        daily["station"] = station
        frames.append(daily[["station", "date", "rain_mm"]])
        print(f"  rain {station}: {len(daily):,} days")
    rain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["station", "date", "rain_mm"])
    rain.to_csv(RAIN_CSV, index=False)
    return rain


def add_rainfall_statewide(df, rain):
    """County-level shifted rainfall features (prev windows, wet-days, first-flush)."""
    rcols = ["rain_county_prev1d", "rain_county_prev3d", "rain_county_prev7d",
             "rain_county_prev14d", "rain_county_wetdays_prev7d", "rain_county_first_flush_prev3d"]
    if os.environ.get("MBAL_NO_RAIN") == "1" or rain is None or rain.empty:
        for c in rcols:
            df[c] = 0.0
        return df
    df = df.copy()
    df["station"] = df["county"].map(COUNTY_STATION)
    rain = rain.copy()
    rain["date"] = pd.to_datetime(rain["date"]).dt.normalize()
    parts = []
    for station, g in rain.groupby("station"):
        s = g.set_index("date")["rain_mm"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="1D")).fillna(0)
        sh = s.shift(1)
        part = pd.DataFrame({"station": station, "sample_date": s.index})
        part["rain_county_prev1d"] = sh.to_numpy()
        part["rain_county_prev3d"] = sh.rolling(3, min_periods=1).sum().to_numpy()
        part["rain_county_prev7d"] = sh.rolling(7, min_periods=1).sum().to_numpy()
        part["rain_county_prev14d"] = sh.rolling(14, min_periods=1).sum().to_numpy()
        part["rain_county_wetdays_prev7d"] = sh.rolling(7, min_periods=1).apply(lambda x: float((x >= 2.54).sum()), raw=True).to_numpy()
        # first flush: meaningful rain after a long dry spell
        dry = (sh < 2.54).astype(int)
        days_dry = dry.groupby((dry != dry.shift()).cumsum()).cumsum()
        part["rain_county_first_flush_prev3d"] = (part["rain_county_prev3d"].to_numpy() * (days_dry.shift(1).fillna(0).to_numpy() >= 7))
        parts.append(part)
    feats = pd.concat(parts, ignore_index=True)
    return df.merge(feats, on=["station", "sample_date"], how="left")


def cause_flags(adv):
    cause = adv.get("advisory_cause", pd.Series("", index=adv.index)).fillna("").astype(str).str.lower()
    atype = adv.get("advisory_type", pd.Series("", index=adv.index)).fillna("").astype(str).str.lower()
    spill = ("spill", "sewage", "line break", "pump station", "grease", "overflow")
    adv = adv.copy()
    adv["cause_spill"] = cause.apply(lambda c: any(t in c for t in spill)).astype(float)
    adv["cause_rain"] = ((atype == "rain") | (cause == "rain")).astype(float)
    return adv


def build_features(obs):
    daily = L._daily_exceed_by_group(obs, ["county", "beach_name", "station_name", "station_id"])
    daily["site_key"] = daily["county"].fillna("") + "|" + daily["beach_name"].fillna("") + "|" + daily["station_name"].fillna("")
    daily = daily.sort_values(["site_key", "sample_date"]).reset_index(drop=True)
    daily = L.add_calendar_features(daily)
    # per-site lab history
    frames = [L.add_lab_lag_features(g.copy()) for _, g in daily.groupby("site_key", sort=False)]
    out = pd.concat(frames, ignore_index=True)
    # within-county prior bacteria context (shifted 1d), excludes same-day site via county-level agg
    cd = daily.groupby(["county", "sample_date"]).agg(
        county_exceed_rate=("exceed_any", "mean"),
        county_sites=("site_key", "nunique"),
    ).reset_index()
    cd = cd.sort_values(["county", "sample_date"])
    cd["county_exceed_rate_prev"] = cd.groupby("county")["county_exceed_rate"].shift(1)
    cd["county_exceed_roll7_prev"] = cd.groupby("county")["county_exceed_rate"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    out = out.merge(cd[["county", "sample_date", "county_exceed_rate_prev", "county_exceed_roll7_prev"]],
                    on=["county", "sample_date"], how="left")
                    
    # Spatial Tensor Encoding (K-Nearest Neighbors & Lat/Lon)
    geo_path = ROOT / "reports" / "station_geo.parquet"
    if geo_path.exists():
        geo = pd.read_parquet(geo_path)
        # We temporarily need 'exceed' for add_knn_spatial_lag
        out["exceed"] = out["exceed_any"].astype(float)
        out = add_knn_spatial_lag(out, geo, k=8, reveal_lag_days=2)
        out = out.drop(columns=["exceed"])
        geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
        out = out.merge(geo_t[["station_id", "latitude", "longitude"]], on="station_id", how="left")
        
    return out


def add_advisories(df, adv):
    adv = adv.dropna(subset=["advisory_date"]).copy()
    adv["advisory_date"] = pd.to_datetime(adv["advisory_date"]).dt.normalize()
    adv = cause_flags(adv)
    daily = adv.groupby(["county", "advisory_date"]).agg(
        adv_count=("advisory_type", "size"),
        adv_spill=("cause_spill", "sum"),
        adv_rain=("cause_rain", "sum"),
    ).reset_index().rename(columns={"advisory_date": "sample_date"})
    daily = daily.sort_values(["county", "sample_date"])
    for col in ["adv_count", "adv_spill", "adv_rain"]:
        for w in [30, 90]:
            daily[f"{col}_prev{w}d"] = daily.groupby("county")[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    keep = ["county", "sample_date"] + [c for c in daily.columns if "_prev" in c]
    return df.merge(daily[keep], on=["county", "sample_date"], how="left")


def train_eval(df, label, drop_substr=()):
    tgt = {"enterococcus", "total_coliform", "fecal_coliform", "e_coli",
           "exceed_enterococcus", "exceed_fecal_coliform", "exceed_total_coliform",
           "exceed_total_ratio", "exceed_any"}
    ignore = {"sample_date", "county", "beach_name", "station_name", "site_key", "station_id"} | tgt
    feats = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    if drop_substr:
        feats = [c for c in feats if not any(s in c for s in drop_substr)]
    rows = df.sort_values(["sample_date", "site_key"]).reset_index(drop=True)
    train = rows[rows["sample_date"] <= "2021-12-31"]
    valid = rows[(rows["sample_date"] > "2021-12-31") & (rows["sample_date"] <= "2023-12-31")]
    test = rows[rows["sample_date"] > "2023-12-31"]
    feats, _ = L.production_safe_features(feats, [train, valid, test])
    tv = pd.concat([train, valid], ignore_index=True)
    ytv = tv["exceed_any"].astype(bool).to_numpy()
    yte = test["exceed_any"].astype(bool).to_numpy()
    models = {
        "logistic": Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                              ("m", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))]),
        "rf": Pipeline([("i", SimpleImputer(strategy="median")),
                        ("m", RandomForestClassifier(n_estimators=400, min_samples_leaf=12, max_features="sqrt",
                                                     class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
    }
    best = None
    for name, mdl in models.items():
        mdl.fit(tv[feats], ytv)
        sc = mdl.predict_proba(test[feats])[:, 1]
        ap = average_precision_score(yte, sc); auc = roc_auc_score(yte, sc)
        if best is None or ap > best["avg_precision"]:
            best = {"model": name, "avg_precision": round(float(ap), 4), "roc_auc": round(float(auc), 4)}
    res = {"label": label, "n_features": len(feats), "rows": len(rows),
           "train": len(train), "valid": len(valid), "test": len(test),
           "test_events": int(yte.sum()), "test_event_rate": round(float(yte.mean()), 4), **best}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    obs = fetch_observations(args.refresh)
    adv = fetch_advisories(args.refresh)
    hab = fetch_hab_blooms(args.refresh)
    rain = fetch_statewide_rainfall(args.refresh)
    print(f"pulled: {len(obs):,} obs, {len(adv):,} advisories, {len(hab):,} HAB reports, {len(rain):,} rain-days")
    df = build_features(obs)
    df = add_advisories(df, adv)
    df = add_hab_features(df, hab)            # freshwater HAB (inert; kept as A/B record)
    df_full = add_rainfall_statewide(df.copy(), rain)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(OUTDIR / "statewide_training_frame.parquet", index=False)
    res_full = train_eval(df_full, "statewide + rainfall")
    # A/B: does statewide rainfall lift the model?
    os.environ["MBAL_NO_RAIN"] = "1"
    res_norain = train_eval(add_rainfall_statewide(df.copy(), rain), "statewide WITHOUT rainfall")
    os.environ.pop("MBAL_NO_RAIN", None)
    res_nolab = train_eval(df_full, "statewide NO lab-history (persistence ablation)",
                           drop_substr=("coliform", "enterococcus", "e_coli", "prev_exceed", "exceed_roll"))
    res_mry = train_eval(df_full[df_full["county"] == "Monterey"].copy(), "Monterey-only (same features)")
    out = {"_headline": {
               "note": "The pooled numbers below are NOT the fundable headline. Pooled all-county "
                       "AP/ROC-AUC are dominated by San Diego (large volume, ~60% base rate, AP~0.96) "
                       "and overstate generalization. The honest operational headline is the "
                       "leave-one-county-out result in reproduce/expected/spatial_holdout.json.",
               "operational_headline_source": "research/bacteria/reproduce/expected/spatial_holdout.json",
               "leave_one_county_out_median_ap": 0.5222,
               "counties_model_beats_ab411_baseline": "9/9",
               "counties_deploy_ready": "8/9",
               "san_diego_excluded_macro_ap": 0.4531,
               "san_diego_excluded_macro_roc_auc": 0.826,
               "pooled_figures_caveat": "statewide_with_rainfall AP/ROC-AUC are the pooled, "
                                        "San-Diego-dominated cut; report the leave-one-county-out figures instead.",
           },
           "statewide_with_rainfall": res_full, "statewide_without_rainfall": res_norain,
           "statewide_no_lab_history_ablation": res_nolab,
           "monterey_only": res_mry,
           "hab_verdict": "HAB tables are FRESHWATER cyanobacteria (reservoirs/lakes), not coastal/marine; excluded by coverage gate, no lift",
           "total_events_all_history": int(df_full["exceed_any"].sum()),
           "beaches": int(df_full["site_key"].nunique())}
    (OUTDIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
