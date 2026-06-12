#!/usr/bin/env python3
"""HAB / algae SOTA sweep over normalized local signals.

This is intentionally separate from the claimable bacteria registry. It asks:

1. Can the marine HAB / domoic-acid target use more of the normalized lakehouse signals?
2. Which model-family analogs help on the same time holdout?
3. Does an "all data" model actually beat the existing lean DA+Pseudo-nitzschia model?

Strict causality: every feature for sample i is known at the previous sample for that
station, and external daily signals are joined as-of that previous sample time with a
one-day availability lag. No same-day label or same-day driver enters its own row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - dependency varies by machine
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet"
OUTDIR = ROOT / "reports" / "hab"
OUT_JSON = OUTDIR / "hab_sota_sweep.json"
OUT_MD = OUTDIR / "hab_sota_sweep.md"
NORMALIZED_PANEL = ROOT / "lakehouse" / "silver" / "hab" / "hab_sota_panel.parquet"

THRESH_NG_ML = 0.5
TRAIN_END = 2018
VALID_END = 2020
AVAILABILITY_LAG_DAYS = 1

PN_COLS = ["Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group"]
LOCAL_DRIVER_COLS = ["Temp", "Avg_Chloro", "Nitrate", "Phosphate", "Silicate"]

FEAT_DA = [
    "month", "doy_sin", "doy_cos", "days_since_prev",
    "pda_prev", "pda_roll3_prev", "exc_prev", "exc_prev2", "exc_roll4_prev",
    "station_prior_rate", "station_prior_n",
]
FEAT_PN = ["pn_seriata_prev", "pn_delica_prev", "pn_total_prev"]
FEAT_LOCAL_DRIVER = [f"{c.lower()}_prev" for c in LOCAL_DRIVER_COLS] + ["si_n_ratio_prev"]
FEAT_SPATIAL = ["latitude", "longitude", "lat_sin", "lon_cos", "lon_sin"]


def _daily_mean(path: Path, time_col: str, mapping: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["signal_date", *mapping.values()])
    d = pd.read_parquet(path)
    if d.empty or time_col not in d.columns:
        return pd.DataFrame(columns=["signal_date", *mapping.values()])
    keep = [time_col] + [c for c in mapping if c in d.columns]
    d = d[keep].copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col])
    for c in mapping:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["signal_date"] = d[time_col].dt.tz_localize(None).dt.normalize()
    out = d.groupby("signal_date")[[c for c in mapping if c in d.columns]].mean().reset_index()
    return out.rename(columns=mapping)


def _hf_radar_daily() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "hf_radar" / "hf_radar.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["signal_date", "hf_current_speed_ms"])
    d = pd.read_parquet(path)
    if d.empty or not {"time", "u", "v"}.issubset(d.columns):
        return pd.DataFrame(columns=["signal_date", "hf_current_speed_ms"])
    d = d.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["hf_current_speed_ms"] = np.sqrt(pd.to_numeric(d["u"], errors="coerce") ** 2 + pd.to_numeric(d["v"], errors="coerce") ** 2)
    d["signal_date"] = d["time"].dt.tz_localize(None).dt.normalize()
    return d.dropna(subset=["signal_date"]).groupby("signal_date")["hf_current_speed_ms"].mean().reset_index()


def _cencoos_mbal_daily() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "cencoos_mbal_moorings" / "cencoos_mbal_moorings.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["signal_date"])
    d = pd.read_parquet(path)
    if d.empty or "time" not in d.columns:
        return pd.DataFrame(columns=["signal_date"])
    d = d.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["signal_date"] = d["time"].dt.tz_localize(None).dt.normalize()
    d["depth_m"] = pd.to_numeric(d.get("z"), errors="coerce").abs()
    d["temp"] = pd.to_numeric(d.get("sea_water_temperature"), errors="coerce").where(
        pd.to_numeric(d.get("sea_water_temperature"), errors="coerce").between(-2.5, 35.0)
    )
    d["sal"] = pd.to_numeric(d.get("sea_water_practical_salinity"), errors="coerce").where(
        pd.to_numeric(d.get("sea_water_practical_salinity"), errors="coerce").between(0, 42)
    )
    frames = []
    for label, lo, hi in [("surface", 0, 10), ("mid", 10, 75), ("deep", 75, 500)]:
        g = d[d["depth_m"].between(lo, hi, inclusive="left")]
        if g.empty:
            continue
        frames.append(
            g.groupby("signal_date").agg(
                **{
                    f"mbal_temp_{label}_c": ("temp", "mean"),
                    f"mbal_sal_{label}_psu": ("sal", "mean"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["signal_date"])
    return pd.concat(frames, axis=1).reset_index()


def _caloes_daily() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "caloes_spills" / "caloes_spills.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["signal_date"])
    d = pd.read_parquet(path)
    if d.empty or "event_time" not in d.columns:
        return pd.DataFrame(columns=["signal_date"])
    d = d.copy()
    d["signal_date"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    d = d.dropna(subset=["signal_date"])
    water = d.get("water", pd.Series("", index=d.index)).astype(str).str.lower().str.contains("yes")
    coast = d.get("county", pd.Series("", index=d.index)).astype(str).str.contains(
        "Monterey|Santa Cruz|San Mateo|San Luis Obispo|Santa Barbara|San Diego|Orange",
        case=False,
        regex=True,
        na=False,
    )
    idx = pd.date_range(d["signal_date"].min(), d["signal_date"].max(), freq="D")
    out = pd.DataFrame({"signal_date": idx})
    total = d.groupby("signal_date").size().reindex(idx, fill_value=0).astype(float)
    water_total = d[water].groupby("signal_date").size().reindex(idx, fill_value=0).astype(float)
    coast_water = d[water & coast].groupby("signal_date").size().reindex(idx, fill_value=0).astype(float)
    out["caloes_spills_30d"] = total.shift(1).rolling(30, min_periods=1).sum().fillna(0).to_numpy()
    out["caloes_water_spills_30d"] = water_total.shift(1).rolling(30, min_periods=1).sum().fillna(0).to_numpy()
    out["caloes_coast_water_spills_30d"] = coast_water.shift(1).rolling(30, min_periods=1).sum().fillna(0).to_numpy()
    return out


def external_daily_signals() -> tuple[pd.DataFrame, dict]:
    frames = [
        _daily_mean(ROOT / "data" / "external_curated" / "mur_sst" / "mur_sst.parquet", "time", {"sst_c": "mur_sst_c"}),
        _daily_mean(ROOT / "data" / "external_curated" / "viirs_chl" / "viirs_chl.parquet", "time", {"chlor_a": "viirs_chlor_a"}),
        _daily_mean(
            ROOT / "data" / "external_curated" / "cencoos_ocean_acidification" / "cencoos_ocean_acidification.parquet",
            "time",
            {
                "sea_water_temperature": "oa_temp_c",
                "sea_water_practical_salinity": "oa_salinity_psu",
                "mass_concentration_of_chlorophyll_in_sea_water": "oa_chlorophyll",
                "moles_of_oxygen_per_unit_mass_in_sea_water": "oa_oxygen_umol_kg",
                "sea_water_ph_reported_on_total_scale": "oa_ph_total",
                "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": "oa_pco2_uatm",
            },
        ),
        _cencoos_mbal_daily(),
        _hf_radar_daily(),
        _caloes_daily(),
        _daily_mean(ROOT / "mbal_history" / "noaa" / "noaa_upwelling.parquet", "time", {"cuti_37n": "cuti_37n", "beuti_37n": "beuti_37n"}),
    ]
    out = pd.DataFrame()
    coverage = {}
    for f in frames:
        if f.empty or "signal_date" not in f.columns:
            continue
        if out.empty:
            out = f
        else:
            out = out.merge(f, on="signal_date", how="outer")
    if out.empty:
        return pd.DataFrame(columns=["signal_date"]), {}
    out = out.sort_values("signal_date").reset_index(drop=True)
    for c in out.columns:
        if c != "signal_date":
            coverage[c] = int(out[c].notna().sum())
    return out, coverage


def load_panel() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
    d = d[d["pDA"].notna()].copy()
    d["exceed"] = (d["pDA"] >= THRESH_NG_ML).astype(int)
    d["pn_total"] = d[PN_COLS].sum(axis=1, min_count=1)
    d["lat_sin"] = np.sin(np.radians(d["latitude"]))
    d["lon_sin"] = np.sin(np.radians(d["longitude"]))
    d["lon_cos"] = np.cos(np.radians(d["longitude"]))
    return d.sort_values(["station", "time"]).reset_index(drop=True)


def add_causal_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    doy = d["time"].dt.dayofyear
    d["month"] = d["time"].dt.month
    d["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    parts = []
    for _, g in d.groupby("station", sort=False):
        g = g.sort_values("time").copy()
        prev = g.shift(1)
        g["feature_time"] = prev["time"]
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
        for c in LOCAL_DRIVER_COLS:
            g[f"{c.lower()}_prev"] = pd.to_numeric(prev[c], errors="coerce")
        g["si_n_ratio_prev"] = pd.to_numeric(prev["Silicate"], errors="coerce") / pd.to_numeric(prev["Nitrate"], errors="coerce").replace(0, np.nan)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["year"] = out["time"].dt.year
    return out[out["exc_prev"].notna()].reset_index(drop=True)


def join_external(d: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return d
    left = d.sort_values("feature_time").copy()
    left["available_signal_date"] = (left["feature_time"] - pd.Timedelta(days=AVAILABILITY_LAG_DAYS)).dt.normalize()
    right = daily.sort_values("signal_date").copy()
    out = pd.merge_asof(
        left,
        right,
        left_on="available_signal_date",
        right_on="signal_date",
        direction="backward",
        tolerance=pd.Timedelta(days=45),
    )
    return out.sort_index()


def _scores(y: Iterable[int], p: Iterable[float]) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "ap": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
    }


def _fit_predict(model_name: str, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray | None, str | None]:
    usable = []
    for f in feats:
        if f not in tr.columns:
            continue
        s = pd.to_numeric(tr[f], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if s.dropna().nunique() < 2:
            continue
        usable.append(f)
    if not usable:
        return None, "no usable training features after leakage/coverage gate"

    Xtr, ytr = tr[usable].astype(float), tr["exceed"].to_numpy()
    Xva, yva = va[usable].astype(float), va["exceed"].to_numpy()
    Xte = te[usable].astype(float)

    if model_name in {"bacteria_hgbt_isotonic", "bacteria_hgbt_spatial", "hab_sota_hgbt"}:
        model = HistGradientBoostingClassifier(
            max_iter=700,
            learning_rate=0.035,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
    elif model_name == "bacteria_xgboost":
        if XGBClassifier is None:
            return None, "xgboost dependency unavailable"
        pos = max(1, int(ytr.sum()))
        neg = max(1, int(len(ytr) - ytr.sum()))
        model = XGBClassifier(
            n_estimators=450,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=42,
        )
    elif model_name == "logistic_regularized":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        )
    elif model_name == "random_forest_context":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        )
    else:
        return None, "unknown model"

    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        p_te = model.predict_proba(Xte)[:, 1]
        p_va = model.predict_proba(Xva)[:, 1]
    else:
        p_te = model[-1].predict_proba(model[:-1].transform(Xte))[:, 1]
        p_va = model[-1].predict_proba(model[:-1].transform(Xva))[:, 1]
    if len(np.unique(yva)) > 1:
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
        p_te = iso.predict(p_te)
    return p_te, None


def run(output_dir: Path | None = None, panel_out: Path | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir else OUTDIR
    out_json = out_dir / "hab_sota_sweep.json"
    out_md = out_dir / "hab_sota_sweep.md"
    panel_path = Path(panel_out) if panel_out else NORMALIZED_PANEL

    daily, daily_coverage = external_daily_signals()
    panel = join_external(add_causal_features(load_panel()), daily)
    external_feats = [
        c for c in panel.columns
        if c
        not in {
            "time", "feature_time", "signal_date", "available_signal_date", "station",
            "Location_Code", "exceed", "year", "pDA", "tDA", "dDA", "pn_total",
        }
        and (
            c.startswith(("mur_", "viirs_", "oa_", "mbal_", "hf_", "caloes_", "cuti_", "beuti_"))
        )
    ]
    all_feats = FEAT_DA + FEAT_PN + FEAT_LOCAL_DRIVER + FEAT_SPATIAL + external_feats
    feature_groups = {
        "DA_history": FEAT_DA,
        "DA_plus_precursor": FEAT_DA + FEAT_PN,
        "local_HAB_all": FEAT_DA + FEAT_PN + FEAT_LOCAL_DRIVER,
        "spatial": FEAT_DA + FEAT_PN + FEAT_SPATIAL,
        "external_context": FEAT_DA + FEAT_PN + external_feats,
        "all_normalized_signals": all_feats,
    }
    tr = panel[panel["year"] <= TRAIN_END]
    va = panel[(panel["year"] > TRAIN_END) & (panel["year"] <= VALID_END)]
    te = panel[panel["year"] > VALID_END].copy()
    yte = te["exceed"].to_numpy()

    baselines = {
        "seasonal_naive": te["month"].map(tr.groupby("month")["exceed"].mean()).fillna(float(tr["exceed"].mean())),
        "persistence": te["exc_prev"].astype(float),
        "station_memory": te["station_prior_rate"].fillna(float(tr["exceed"].mean())),
    }
    results = {f"baseline::{k}": _scores(yte, v) for k, v in baselines.items()}

    sequence = []
    compatible_models = [
        ("bacteria_hgbt_isotonic", "DA_plus_precursor"),
        ("bacteria_hgbt_spatial", "spatial"),
        ("bacteria_xgboost", "all_normalized_signals"),
        ("logistic_regularized", "all_normalized_signals"),
        ("random_forest_context", "all_normalized_signals"),
        ("hab_sota_hgbt", "all_normalized_signals"),
    ]
    for model_id, group in compatible_models:
        feats = [f for f in feature_groups[group] if f in panel.columns]
        p, reason = _fit_predict(model_id, tr, va, te, feats)
        key = f"{model_id}::{group}"
        if p is None:
            results[key] = {"skipped": True, "reason": reason}
            sequence.append({"model_id": model_id, "feature_group": group, "status": "skipped", "reason": reason})
        else:
            results[key] = _scores(yte, p)
            results[key]["feature_count"] = len(feats)
            sequence.append({"model_id": model_id, "feature_group": group, "status": "done", **results[key]})

    # Feature-group ablation under the incumbent HGBT family.
    ablations = {}
    for group, feats0 in feature_groups.items():
        feats = [f for f in feats0 if f in panel.columns]
        p, reason = _fit_predict("bacteria_hgbt_isotonic", tr, va, te, feats)
        ablations[group] = {"reason": reason, "feature_count": len(feats), **(_scores(yte, p) if p is not None else {"skipped": True})}

    best_key = max(
        (k for k, v in results.items() if not v.get("skipped") and k.startswith(("bacteria_", "logistic", "random", "hab_"))),
        key=lambda k: results[k].get("ap") or -1,
    )
    baseline_ap = max(results["baseline::seasonal_naive"]["ap"], results["baseline::persistence"]["ap"], results["baseline::station_memory"]["ap"])
    incumbent_ap = ablations["DA_plus_precursor"]["ap"]
    out = {
        "target": f"next-sample pDA >= {THRESH_NG_ML} ng/mL, marine HAB / domoic-acid risk",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "normalization": {
            "panel_path": str(panel_path),
            "rows": int(len(panel)),
            "stations": int(panel["station"].nunique()),
            "date_min": str(panel["time"].min()),
            "date_max": str(panel["time"].max()),
            "external_daily_signal_coverage": daily_coverage,
            "availability_lag_days": AVAILABILITY_LAG_DAYS,
        },
        "splits": {
            "train_end": TRAIN_END,
            "valid_end": VALID_END,
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "n_test": int(len(te)),
            "test_events": int(te["exceed"].sum()),
        },
        "compatibility": {
            "xgboost_forecast_v2": "not run on HAB target: M1 regression forecaster, not DA classifier",
            "patchtst": "not run on HAB target: M1 neural forecasting benchmark; use only after building a weekly DA forecasting harness",
            "nhits": "not run on HAB target: M1 neural forecasting research candidate; use only after building a weekly DA forecasting harness",
        },
        "sequence": sequence,
        "ablation_hgbt": ablations,
        "results": results,
        "headline": {
            "best_model": best_key,
            "best_ap": results[best_key]["ap"],
            "best_roc_auc": results[best_key]["roc_auc"],
            "best_baseline_ap": baseline_ap,
            "incumbent_DA_plus_precursor_ap": incumbent_ap,
            "beats_baselines": (results[best_key]["ap"] or 0) > (baseline_ap or 0),
            "beats_incumbent": (results[best_key]["ap"] or 0) > (incumbent_ap or 0),
        },
        "critic": (
            "All normalized signals were tested, but promotion is based on held-out AP. "
            "If all-data context underperforms the lean DA+precursor model, the correct SOTA remains lean."
        ),
    }
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(out, out_md)
    return out


def write_md(out: dict, out_md: Path = OUT_MD) -> None:
    lines = [
        "# HAB SOTA Sweep",
        "",
        f"Target: {out['target']}",
        "",
        "## Headline",
        "",
        f"- Best model: `{out['headline']['best_model']}`",
        f"- AP: {out['headline']['best_ap']}",
        f"- ROC-AUC: {out['headline']['best_roc_auc']}",
        f"- Best baseline AP: {out['headline']['best_baseline_ap']}",
        f"- Incumbent DA+precursor AP: {out['headline']['incumbent_DA_plus_precursor_ap']}",
        f"- Beats incumbent: {out['headline']['beats_incumbent']}",
        "",
        "## Sequence",
        "",
        "| model | feature group | status | AP | ROC-AUC | note |",
        "|---|---|---|--:|--:|---|",
    ]
    for row in out["sequence"]:
        lines.append(
            f"| {row['model_id']} | {row['feature_group']} | {row['status']} | "
            f"{row.get('ap', '')} | {row.get('roc_auc', '')} | {row.get('reason', '')} |"
        )
    lines += [
        "",
        "## HGBT Feature Ablation",
        "",
        "| feature group | features | AP | ROC-AUC |",
        "|---|--:|--:|--:|",
    ]
    for group, row in out["ablation_hgbt"].items():
        lines.append(f"| {group} | {row.get('feature_count')} | {row.get('ap')} | {row.get('roc_auc')} |")
    lines += [
        "",
        "## Critic",
        "",
        out["critic"],
        "",
        "The M1 forecasting models (`xgboost_forecast_v2`, `patchtst`, `nhits`) were not run as DA classifiers.",
        "They need a separate weekly DA sequence harness before they are target-compatible.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTDIR),
                    help="where to write hab_sota_sweep.json/.md (default reports/hab/).")
    ap.add_argument("--panel-out", default=None,
                    help="where to write the normalized panel parquet "
                         "(default lakehouse/silver/hab/hab_sota_panel.parquet; "
                         "override for output-isolated confirmation runs).")
    args = ap.parse_args(argv)
    out_dir = Path(args.output_dir)
    panel_out = Path(args.panel_out) if args.panel_out else (
        out_dir / "hab_sota_panel.parquet" if args.output_dir != str(OUTDIR) else NORMALIZED_PANEL)

    out = run(output_dir=out_dir, panel_out=panel_out)
    print(json.dumps(out["headline"], indent=2))
    print(f"wrote {out_dir / 'hab_sota_sweep.json'}")
    print(f"wrote {out_dir / 'hab_sota_sweep.md'}")
    print(f"wrote {panel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
