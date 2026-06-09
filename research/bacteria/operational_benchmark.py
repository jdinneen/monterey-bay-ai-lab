#!/usr/bin/env python3
"""Operational beach-bacteria exceedance benchmark — honest, stratified, calibrated.

The pooled statewide headline (ROC-AUC ~0.89) does not survive domain scrutiny: the
project's own external review traced most of the train->test base-rate jump to an
enterococcus / San Diego (Tijuana River) regime break sitting on the split boundary.
This harness answers the question that actually matters operationally, WITHOUT being
flattered by that artifact:

    Within each region/era, does a model beat the baseline a public-health official
    ALREADY has -- the prior lab result (what they know 18-24h later) and station
    memory ("a beach that was recently dirty tends to be dirty again") -- and is it
    well-calibrated enough to support a beach-advisory decision?

Discipline (non-negotiable):
- Strictly causal features: every feature uses only information available BEFORE the
  predicted sample_date (no same-day lab feeds its own prediction). Ported verbatim
  from reports/statewide_bacteria_reframe.py.
- Region-stratified test eval (ALL / EXCLUDE_SAN_DIEGO / SAN_DIEGO_ONLY / MONTEREY)
  so the regime break is visible, not hidden in a pooled number.
- Operational baselines: prior-lab (last exceedance) and station memory, not just the
  base rate -- the model must beat what officials already use, per stratum.
- Calibration (Brier + expected calibration error) because an advisory decision needs
  trustworthy probabilities, not just ranking.
- AB411 rain rule is NOT evaluated statewide: the statewide rainfall driver is not yet
  ingested (only Lovers Point ASOS). That gap is reported explicitly, never faked.

Portable: data path via --obs / $MBARI_BACTERIA_OBS, no hardcoded project root.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# CA single-sample standards (same as the project's existing models).
THR = dict(enterococcus=104.0, fecal=400.0, total=10000.0, total_ratio_gate=1000.0,
           fecal_total_ratio=0.1)
PARAM_MAP = {
    "Enterococcus": "ent", "Fecal Coliforms": "fec",
    "Total Coliforms": "tot", "E. Coli": "ecoli",
}
FEATS = [
    "month", "doy_sin", "doy_cos",
    "ent_prev", "fec_prev", "tot_prev", "ecoli_prev",
    "ent_roll3_prev", "fec_roll3_prev", "tot_roll3_prev", "ecoli_roll3_prev",
    "exc_prev", "exc_prev2", "exc_roll5_prev", "exc_roll10_prev",
    "station_prior_rate", "station_prior_n", "days_since_prev",
    "cty_prev7", "sw_prev7",
]
# County -> evaluation region. Monterey Bay = Monterey + Santa Cruz counties.
MONTEREY_COUNTIES = {"Monterey", "Santa Cruz"}

# Rain features (added only when a rainfall dir is supplied) and the AB411 threshold.
RAIN_FEATS = ["rain_0d", "rain_3d", "rain_7d", "days_since_rain"]
AB411_RAIN_MM = 2.54  # 0.1 inch — the common CA wet-weather (AB411) advisory threshold
# First-flush river-discharge features (added only when a discharge dir is supplied).
DISCHARGE_FEATS = ["discharge_log", "discharge_3d", "discharge_anom", "days_since_highq"]


def _days_since_wet(precip: np.ndarray, thr: float) -> np.ndarray:
    """Days since the last wet day (>= thr) on a continuous daily series (NaN until
    the first wet day) — a first-flush proxy."""
    out = np.full(len(precip), np.nan)
    cnt = np.nan
    for i, v in enumerate(precip):
        if v >= thr:
            cnt = 0.0
        elif not np.isnan(cnt):
            cnt += 1.0
        out[i] = cnt
    return out


def load_rainfall(rain_dir: Path):
    rain_dir = Path(rain_dir)
    grid = pd.read_parquet(rain_dir / "rainfall_grid.parquet")
    smap = pd.read_parquet(rain_dir / "station_grid_map.parquet")
    return grid, smap


def add_rain_features(df: pd.DataFrame, rain_dir: Path) -> pd.DataFrame:
    """Join causal rainfall (known at sample time) to station-days via the 0.1deg grid.

    Windows END on sample_date: at the moment of sampling, today's and prior rain is
    known, while the lab result we predict is not available for another 18-24h, so
    these features do not leak the target.
    """
    grid, smap = load_rainfall(rain_dir)
    grid = grid.dropna(subset=["date"]).copy()
    grid["date"] = pd.to_datetime(grid["date"])
    grid["precip_mm"] = grid["precip_mm"].fillna(0.0)
    grid = grid.sort_values(["grid_lat", "grid_lon", "date"])
    g = grid.groupby(["grid_lat", "grid_lon"], group_keys=False)["precip_mm"]
    grid["rain_0d"] = grid["precip_mm"]
    grid["rain_3d"] = g.apply(lambda s: s.rolling(3, min_periods=1).sum())
    grid["rain_7d"] = g.apply(lambda s: s.rolling(7, min_periods=1).sum())
    grid["days_since_rain"] = g.transform(lambda s: _days_since_wet(s.to_numpy(), AB411_RAIN_MM))
    cols = ["grid_lat", "grid_lon", "date", *RAIN_FEATS]
    out = df.merge(smap, on="station_id", how="left").merge(
        grid[cols], left_on=["grid_lat", "grid_lon", "sample_date"],
        right_on=["grid_lat", "grid_lon", "date"], how="left")
    return out.drop(columns=["date"], errors="ignore")


def add_discharge_features(df: pd.DataFrame, discharge_dir: Path) -> pd.DataFrame:
    """Causal first-flush river-discharge features via each station's nearest
    data-bearing USGS gauge. merge_asof takes the last reading on/before sample_date
    (within 7 days), so a gauge's reporting gaps degrade gracefully to NaN.
    """
    discharge_dir = Path(discharge_dir)
    dis = pd.read_parquet(discharge_dir / "discharge_gauge.parquet")
    smap = pd.read_parquet(discharge_dir / "station_gauge_map.parquet")
    dis = dis.dropna(subset=["date"]).copy()
    dis["date"] = pd.to_datetime(dis["date"])
    parts = []
    for gid, d in dis.sort_values(["gauge_id", "date"]).groupby("gauge_id"):
        q = d.set_index("date")["discharge_cfs"].clip(lower=0)
        feat = pd.DataFrame(index=q.index)
        feat["gauge_id"] = gid
        feat["discharge_log"] = np.log1p(q)
        feat["discharge_3d"] = q.rolling("3D").mean()
        feat["discharge_anom"] = feat["discharge_3d"] / (q.rolling("30D").median() + 1.0)
        # days since a high-flow (>=p90) day — first-flush timing
        feat["days_since_highq"] = _days_since_wet(q.to_numpy(), float(q.quantile(0.9)))
        parts.append(feat.reset_index())
    f = pd.concat(parts, ignore_index=True).sort_values("date")
    left = (df.merge(smap[["station_id", "gauge_id"]], on="station_id", how="left")
              .sort_values("sample_date"))
    merged = pd.merge_asof(
        left, f[["date", "gauge_id", *DISCHARGE_FEATS]],
        left_on="sample_date", right_on="date", by="gauge_id",
        direction="backward", tolerance=pd.Timedelta("7D"))
    return merged.drop(columns=["date"], errors="ignore")


def default_obs_path() -> Path:
    env = os.environ.get("MBARI_BACTERIA_OBS")
    if env:
        return Path(env)
    root = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    return root / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet"


def load_station_days(obs_path: Path, label: str = "any") -> pd.DataFrame:
    """Worst-of-day per station/analyte -> exceedance label.

    label="enterococcus" (recommended canonical): the AB411 MARINE single-sample
    standard, enterococcus > 104 MPN/100mL, on station-days that actually have an
    enterococcus reading. This is a single, era-stable, standard-aligned target — unlike
    label="any" (legacy), the OR over a changing analyte panel (ent/fec/tot/ratio) that
    the 2022 enterococcus regime break exploits. Other analytes remain as FEATURES.
    """
    o = pd.read_parquet(obs_path)
    o = o[o["result_value_numeric"] >= 0].copy()  # drop -999.99 sentinels
    o["p"] = o["source_parameter"].map(PARAM_MAP)
    o = o.dropna(subset=["p"])
    o["sample_date"] = pd.to_datetime(o["sample_date"])
    agg = (o.groupby(["station_id", "county", "sample_date", "p"])
             ["result_value_numeric"].max().unstack("p").reset_index())
    for c in ["ent", "fec", "tot", "ecoli"]:
        if c not in agg:
            agg[c] = np.nan
    if label == "enterococcus":
        agg = agg[agg["ent"].notna()].copy()  # marine standard requires an ent reading
        agg["exceed"] = (agg["ent"] > THR["enterococcus"]).astype(int)
    elif label == "any":
        exc = (
            (agg["ent"] > THR["enterococcus"])
            | (agg["fec"] > THR["fecal"])
            | (agg["tot"] > THR["total"])
            | ((agg["tot"] > THR["total_ratio_gate"]) & (agg["fec"] / agg["tot"] > THR["fecal_total_ratio"]))
        )
        agg["exceed"] = exc.fillna(False).astype(int)
    else:
        raise ValueError(f"unknown label '{label}' (use 'enterococcus' or 'any')")
    return agg.sort_values(["station_id", "sample_date"]).reset_index(drop=True)


def _predecessor_idx(dates_ns: np.ndarray, lag_ns: int) -> np.ndarray:
    """Within a date-sorted station, the position of the latest prior sample whose lab
    result had RETURNED by each row's sample time (date <= sample_date - lag); -1 if none.
    lag_ns=0 reduces exactly to "the previous sample" (shift(1))."""
    pos = np.searchsorted(dates_ns, dates_ns - lag_ns, side="right") - 1
    i = np.arange(len(dates_ns))
    return np.where(pos >= i, i - 1, pos)  # lag=0 ties -> previous row


def add_causal_features(df: pd.DataFrame, reveal_lag_days: int = 0) -> pd.DataFrame:
    """Strictly-causal features. With ``reveal_lag_days`` > 0, every "prior lab / memory"
    feature uses the latest prior sample whose result had returned by sample time (date <=
    sample_date - lag), so a same-/next-day resample cannot feed a label that had not yet
    come back. ``reveal_lag_days=0`` reproduces the historical shift(1) behavior exactly."""
    df = df.sort_values(["station_id", "sample_date"]).copy().reset_index(drop=True)
    d = df["sample_date"]
    doy = d.dt.dayofyear
    df["month"] = d.dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    for c in ["ent", "fec", "tot", "ecoli"]:
        df[f"{c}_logp"] = np.log1p(df[c].clip(lower=0))

    lag_ns = int(reveal_lag_days) * 86_400_000_000_000

    def _gather(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        out = np.full(len(idx), np.nan)
        m = idx >= 0
        out[m] = np.asarray(arr, dtype=float)[idx[m]]
        return out

    parts = []
    for _, g in df.groupby("station_id", sort=False):
        dates = g["sample_date"].to_numpy(dtype="datetime64[ns]")
        pidx = _predecessor_idx(dates.view("int64"), lag_ns)
        pidx2 = np.full(len(pidx), -1, dtype=np.int64)
        ok = pidx >= 0
        pidx2[ok] = pidx[pidx[ok]]  # predecessor of the predecessor (reveal-safe "2 back")
        exc = g["exceed"].to_numpy(dtype=float)
        feat = {
            "exc_prev": _gather(exc, pidx),
            "exc_prev2": _gather(exc, pidx2),
            "exc_roll5_prev": _gather(pd.Series(exc).rolling(5, min_periods=1).mean().to_numpy(), pidx),
            "exc_roll10_prev": _gather(pd.Series(exc).rolling(10, min_periods=1).mean().to_numpy(), pidx),
            "station_prior_rate": _gather(pd.Series(exc).expanding().mean().to_numpy(), pidx),
            "station_prior_n": _gather(pd.Series(exc).expanding().count().to_numpy(), pidx),
        }
        for c in ["ent", "fec", "tot", "ecoli"]:
            lp = g[f"{c}_logp"].to_numpy(dtype=float)
            feat[f"{c}_prev"] = _gather(lp, pidx)
            feat[f"{c}_roll3_prev"] = _gather(pd.Series(lp).rolling(3, min_periods=1).mean().to_numpy(), pidx)
        dsp = np.full(len(pidx), np.nan)
        dsp[ok] = (dates[ok] - dates[pidx[ok]]).astype("timedelta64[D]").astype(float)
        feat["days_since_prev"] = dsp
        parts.append(pd.DataFrame(feat, index=g.index))
    df = pd.concat([df, pd.concat(parts).sort_index()], axis=1)

    cty = (df.groupby(["county", "sample_date"])["exceed"].mean().rename("cty_day_rate").reset_index()
             .sort_values(["county", "sample_date"]))
    cty["cty_prev7"] = (cty.groupby("county")["cty_day_rate"]
                          .apply(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
                          .reset_index(level=0, drop=True))
    df = df.merge(cty[["county", "sample_date", "cty_prev7"]], on=["county", "sample_date"], how="left")
    sw = (df.groupby("sample_date")["exceed"].mean().rename("sw_day_rate").reset_index().sort_values("sample_date"))
    sw["sw_prev7"] = sw["sw_day_rate"].shift(1).rolling(7, min_periods=1).mean()
    df = df.merge(sw[["sample_date", "sw_prev7"]], on="sample_date", how="left")
    return df


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Weighted |confidence - accuracy| over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr_budget: float = 0.20) -> float:
    """Recall (TPR) when the false-alarm rate is held at fpr_budget — the operational
    question: 'if officials tolerate flagging fpr_budget of clean days, what share of
    real exceedances do we catch?'"""
    neg = p[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr_budget)
    return float(((p >= thr) & (y == 1)).sum() / y.sum())


def score(y: np.ndarray, p: np.ndarray, base: float) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out = {"n": int(len(y)), "events": int(y.sum()), "base_rate": round(float(y.mean()), 4)}
    if y.sum() == 0 or y.sum() == len(y):
        out["note"] = "degenerate stratum (single class) — ranking metrics undefined"
        return out
    k = max(1, int(0.10 * len(p)))
    idx = np.argsort(-p)[:k]
    flagged = np.zeros(len(p), dtype=bool)
    flagged[idx] = True
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    ap = float(average_precision_score(y, p))
    stratum_base = float(y.mean())
    # recall@FPR is ill-defined for binary/degenerate scores (a 0/1 rule's negative
    # quantile collapses and flags everything -> a spurious 1.0); report NaN there.
    rec_fpr = _recall_at_fpr(y, p, 0.20) if len(np.unique(p)) > 2 else float("nan")
    out.update(
        ap=round(ap, 4),
        roc_auc=round(float(roc_auc_score(y, p)), 4),
        ap_lift_vs_stratum_base=round(ap / stratum_base, 2) if stratum_base > 0 else None,
        recall_at_10pct=round(tp / int(y.sum()), 4),
        precision_at_10pct=round(tp / (tp + fp), 4) if (tp + fp) else None,
        recall_at_20pct_fpr=(None if np.isnan(rec_fpr) else round(rec_fpr, 4)),
        brier=round(float(brier_score_loss(y, p)), 5),
        ece=round(_expected_calibration_error(y, p), 4),
    )
    return out


def _vb_class_mlr(tr: pd.DataFrame, te: pd.DataFrame, feats_vb: list[str],
                  min_train_events: int = 8) -> pd.Series:
    """Virtual-Beach-class baseline: a per-station logistic regression on causal hydromet
    predictors (VB's modeling approach), with a pooled fallback where a station has too
    few training events. A faithful reimplementation of the VB *method*, not the software.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _mk():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))

    pooled = _mk()
    pooled.fit(tr[feats_vb].fillna(0.0), tr["exceed"].to_numpy())
    p = pd.Series(pooled.predict_proba(te[feats_vb].fillna(0.0))[:, 1], index=te.index)
    tr_by = dict(tuple(tr.groupby("station_id")))
    for sid, g in te.groupby("station_id"):
        s = tr_by.get(sid)
        if s is not None and s["exceed"].nunique() > 1 and int(s["exceed"].sum()) >= min_train_events:
            try:
                m = _mk()
                m.fit(s[feats_vb].fillna(0.0), s["exceed"].to_numpy())
                p.loc[g.index] = m.predict_proba(g[feats_vb].fillna(0.0))[:, 1]
            except Exception:
                pass  # keep the pooled prediction for this station
    return p


def run(obs_path: Path, clf=None, rain_dir=None, discharge_dir=None, reveal_lag_days: int = 0,
        label: str = "any") -> dict:
    """Run the benchmark. ``clf`` is an injectable sklearn classifier (a test seam);
    when None, the production HistGradientBoosting config is used. ``rain_dir`` adds
    causal rainfall features + the AB411 wet-weather rule baseline + a Virtual-Beach-class
    per-site MLR baseline; ``discharge_dir`` adds first-flush features. ``reveal_lag_days``
    enforces the lab-reveal lag (0 = legacy). ``label`` selects the target: "enterococcus"
    (the AB411 marine standard, recommended) or "any" (legacy multi-analyte OR)."""
    df = load_station_days(obs_path, label=label)
    df = add_causal_features(df, reveal_lag_days=reveal_lag_days)
    df["region"] = np.where(
        df["county"] == "San Diego", "SAN_DIEGO",
        np.where(df["county"].isin(MONTEREY_COUNTIES), "MONTEREY", "OTHER"),
    )

    feats = list(FEATS)
    has_rain = False
    if rain_dir is not None:
        df = add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += RAIN_FEATS
            has_rain = True
    has_discharge = False
    if discharge_dir is not None:
        df = add_discharge_features(df, Path(discharge_dir))
        if "discharge_3d" in df.columns and df["discharge_3d"].notna().any():
            feats += DISCHARGE_FEATS
            has_discharge = True

    tr = df[df["sample_date"] <= "2019-12-31"]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    base = float(tr["exceed"].mean())

    if clf is None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    te["p_model"] = clf.predict_proba(te[feats].astype(float))[:, 1]

    # Isotonic calibration fit on the 2020-21 validation era -> the deploy-readiness
    # question: can the model's RANKING lift be turned into TRUSTWORTHY probabilities
    # for an advisory threshold? (The raw class_weight='balanced' scores are inflated.)
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te["p_model_cal"] = te["p_model"]
    if va["exceed"].nunique() > 1:
        # Manual isotonic map (score -> empirical exceedance) fit on the validation era;
        # version-robust vs CalibratedClassifierCV's changing prefit API.
        from sklearn.isotonic import IsotonicRegression

        va_scores = clf.predict_proba(va[feats].astype(float))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(va_scores, va["exceed"].to_numpy())
        te["p_model_cal"] = iso.predict(te["p_model"].to_numpy())

    # Operational baselines available WITHOUT external rain data.
    month_rate = tr.groupby("month")["exceed"].mean()
    te["p_base"] = base
    te["p_month"] = te["month"].map(month_rate).fillna(base)
    te["p_station_memory"] = te["station_prior_rate"].fillna(base)
    te["p_prior_lab"] = te["exc_prev"].fillna(base)  # the operational "was it dirty last time"
    if has_rain:
        # AB411 wet-weather rule: advise if >= 0.1in (2.54mm) fell in the prior ~3 days.
        te["p_ab411"] = (te["rain_3d"].fillna(0.0) >= AB411_RAIN_MM).astype(float)
        # Virtual-Beach-class per-site MLR on causal hydromet predictors (the deployed tool
        # we'd actually replace; AB411 alone is a weak strawman baseline).
        vb_feats = [c for c in ["rain_0d", "rain_3d", "rain_7d", "days_since_rain",
                                "doy_sin", "doy_cos", "exc_prev"] if c in df.columns]
        te["p_vb_mlr"] = _vb_class_mlr(tr, te, vb_feats)

    strata = {
        "ALL": te,
        "EXCLUDE_SAN_DIEGO": te[te["region"] != "SAN_DIEGO"],
        "SAN_DIEGO_ONLY": te[te["region"] == "SAN_DIEGO"],
        "MONTEREY": te[te["region"] == "MONTEREY"],
    }
    score_cols = {
        "baseline_global_rate": "p_base",
        "baseline_month_climatology": "p_month",
        "baseline_prior_lab": "p_prior_lab",
        "baseline_station_memory": "p_station_memory",
    }
    if has_rain:
        score_cols["baseline_ab411_rain"] = "p_ab411"
        score_cols["baseline_vb_mlr"] = "p_vb_mlr"
    score_cols["model_hgbt"] = "p_model"
    score_cols["model_hgbt_calibrated"] = "p_model_cal"
    out: dict = {
        "obs_path": str(obs_path),
        "train_events": int(tr["exceed"].sum()),
        "train_base_rate": round(base, 4),
        "ab411_rain_rule": (
            "EVALUATED - rain_3d >= 2.54mm (0.1in) wet-weather advisory rule; gridded "
            "Open-Meteo daily rainfall at 0.1deg, joined causally by station cell."
            if has_rain else
            "NOT EVALUATED - pass --rain-dir; baseline here is prior-lab + station memory."
        ),
        "discharge_first_flush": (
            "EVALUATED - causal USGS river-discharge first-flush features (nearest "
            "data-bearing gauge within 30km)." if has_discharge else "NOT EVALUATED - pass --discharge-dir."
        ),
        "strata": {},
    }
    for sname, part in strata.items():
        y = part["exceed"].to_numpy()
        rows = {m: score(y, part[col].to_numpy(), base) for m, col in score_cols.items()}
        # Honest headline per stratum: does the model beat the strongest operational
        # baseline (prior-lab, station memory, and the AB411 rain rule when available)?
        op_names = ["baseline_prior_lab", "baseline_station_memory"]
        for b in ("baseline_ab411_rain", "baseline_vb_mlr"):
            if b in rows:
                op_names.append(b)
        op_aps = [rows[b].get("ap") for b in op_names if rows[b].get("ap") is not None]
        op_eces = [rows[b].get("ece") for b in op_names if rows[b].get("ece") is not None]
        verdict = None
        if op_aps and rows["model_hgbt"].get("ap") is not None:
            best_op = max(op_aps)
            best_op_ece = min(op_eces) if op_eces else None
            cal_ece = rows["model_hgbt_calibrated"].get("ece")
            ab411_ap = rows.get("baseline_ab411_rain", {}).get("ap")
            verdict = {
                "best_operational_ap": round(best_op, 4),
                "model_ap": rows["model_hgbt"]["ap"],
                "model_ap_minus_operational": round(rows["model_hgbt"]["ap"] - best_op, 4),
                "model_beats_operational_ranking": bool(rows["model_hgbt"]["ap"] - best_op > 0),
                # Does the model beat the *regulatory* AB411 rain rule specifically?
                "ab411_ap": ab411_ap,
                "model_beats_ab411": (None if ab411_ap is None
                                      else bool(rows["model_hgbt"]["ap"] - ab411_ap > 0)),
                # Does the model beat the deployed-practice baseline (Virtual-Beach-class MLR)?
                "vb_mlr_ap": rows.get("baseline_vb_mlr", {}).get("ap"),
                "model_beats_vb_mlr": (None if rows.get("baseline_vb_mlr", {}).get("ap") is None
                                       else bool(rows["model_hgbt"]["ap"] - rows["baseline_vb_mlr"]["ap"] > 0)),
                # Calibration / deploy-readiness: ranking lift is worthless for an advisory
                # threshold if the probabilities are not trustworthy.
                "model_raw_ece": rows["model_hgbt"].get("ece"),
                "model_calibrated_ece": cal_ece,
                "best_operational_ece": best_op_ece,
                "calibrated_deploy_ready": bool(
                    cal_ece is not None and best_op_ece is not None and cal_ece <= best_op_ece + 0.05
                ),
            }
        out["strata"][sname] = {"models": rows, "operational_verdict": verdict}
    return out


def to_markdown(res: dict) -> str:
    lines = ["# Operational beach-bacteria benchmark (honest, stratified, calibrated)", ""]
    lines.append(f"- train events: {res['train_events']:,} | train base rate: {res['train_base_rate']}")
    lines.append(f"- AB411 rain rule: {res['ab411_rain_rule']}")
    lines.append("")
    for sname, s in res["strata"].items():
        lines.append(f"## {sname}")
        v = s["operational_verdict"]
        if v:
            beat = "BEATS" if v["model_beats_operational_ranking"] else "does NOT beat"
            deploy = "deploy-ready" if v["calibrated_deploy_ready"] else "NOT deploy-ready (recalibrate)"
            lines.append(f"- Ranking: model {beat} the operational baseline - AP {v['model_ap']} vs "
                         f"best-operational {v['best_operational_ap']} (delta {v['model_ap_minus_operational']:+}).")
            if v.get("ab411_ap") is not None:
                ab = "BEATS" if v["model_beats_ab411"] else "does NOT beat"
                lines.append(f"- AB411 regulatory rule: model {ab} it - model AP {v['model_ap']} vs "
                             f"AB411 rain-rule AP {v['ab411_ap']}.")
            lines.append(f"- Calibration: raw ECE {v['model_raw_ece']} -> calibrated ECE "
                         f"{v['model_calibrated_ece']} vs best-operational ECE {v['best_operational_ece']} "
                         f"=> {deploy}.")
        lines.append("")
        lines.append("| model | n | events | base | AP | AUC | rec@20%FPR | Brier | ECE |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for m, r in s["models"].items():
            if "ap" not in r:
                lines.append(f"| {m} | {r['n']} | {r['events']} | {r['base_rate']} | - | - | - | - | - |")
                continue
            lines.append(f"| {m} | {r['n']} | {r['events']} | {r['base_rate']} | {r['ap']} | "
                         f"{r['roc_auc']} | {r.get('recall_at_20pct_fpr')} | {r['brier']} | {r['ece']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None, help="statewide_beach_observations.parquet (or $MBARI_BACTERIA_OBS)")
    ap.add_argument("--out-dir", default=None, help="where to write JSON+MD (default: reports/operational_benchmark)")
    ap.add_argument("--rain-dir", default=None,
                    help="dir with rainfall_grid.parquet + station_grid_map.parquet (adds AB411 baseline + rain feats)")
    ap.add_argument("--discharge-dir", default=None,
                    help="dir with discharge_gauge.parquet + station_gauge_map.parquet (adds first-flush feats)")
    ap.add_argument("--reveal-lag-days", type=int, default=0,
                    help="enforce lab-result reveal lag on prior-lab/memory features (0 = legacy)")
    ap.add_argument("--label", default="any", choices=["any", "enterococcus"],
                    help="target: 'enterococcus' (AB411 marine standard, recommended) or 'any' (legacy)")
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else default_obs_path()
    if not obs.exists():
        print(f"[operational_benchmark] obs not found: {obs}")
        return 2
    res = run(obs, rain_dir=args.rain_dir, discharge_dir=args.discharge_dir,
              reveal_lag_days=args.reveal_lag_days, label=args.label)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[2] / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "operational_benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "operational_benchmark.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print(f"\nwrote {out_dir/'operational_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
