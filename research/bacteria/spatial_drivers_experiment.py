#!/usr/bin/env python3
"""Cracking the spatial gap: does a causal k-nearest-beach spatial driver beat site-memory
AND reduce the residual spatial autocorrelation found by spatial_autocorr.py?

Motivation: leave-one-beach-out (spatial_autocorr.py) showed the nowcast generalizes to
unseen beaches but its residuals stay spatially clustered (Moran's I ~0.25, p=0.002,
San-Diego-excluded) -- i.e. there is real sub-county spatial signal the current features
(prior-lab, station-memory, county/statewide rate, rainfall) do not capture. The current
"county prior-7d" feature is the only spatial term and counties are coarse.

This experiment adds a **k-nearest-beach spatial lag**: for each station-day, the mean
recent exceedance *state* of its k geographically nearest OTHER beaches, as known at sample
time under the lab-reveal lag. It is leakage-safe by construction -- a neighbour's result is
only visible `reveal_lag_days` after the neighbour was sampled, and only the target's own
future is ever the label. We then compare, on the honest San-Diego-excluded 2022+ holdout:

    site-memory  vs  driver model (no spatial)  vs  driver model + spatial lag

on AP / ROC-AUC / ECE, and we recompute residual Moran's I for the no-spatial and
+spatial models. The reproducible result is a wash: raw coordinates and nearest-neighbour
lag do not beat the no-spatial driver model or reduce residual Moran's I. That negative
result is reported straight and keeps the frontier focused on physical spatial covariates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob  # noqa: E402
from research.bacteria.spatial_autocorr import morans_i, _load_geo, leave_one_beach_out  # noqa: E402

SPATIAL_FEATS = ["nbr_prev1", "nbr_prev7"]


def _unique_features(feats: list[str]) -> list[str]:
    """Preserve feature order while avoiding duplicate DataFrame columns at fit time."""
    return list(dict.fromkeys(feats))


def add_knn_spatial_lag(df: pd.DataFrame, geo: pd.DataFrame, k: int = 8,
                        reveal_lag_days: int = 0) -> pd.DataFrame:
    """Add `nbr_prev1` (latest known exceedance state of the k nearest other beaches) and
    `nbr_prev7` (their 7-day mean state). Causal: a neighbour's value at target date t is its
    most recent sample dated <= t - reveal_lag_days, forward-filled on a daily calendar."""
    from sklearn.neighbors import BallTree

    df = df.copy()
    geo = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    stations = [s for s in df["station_id"].unique() if s in set(geo["station_id"])]
    gidx = geo.set_index("station_id").loc[stations]
    coords = np.radians(gidx[["latitude", "longitude"]].to_numpy())
    n = len(stations)
    if n <= k:
        df["nbr_prev1"] = np.nan
        df["nbr_prev7"] = np.nan
        return df

    tree = BallTree(coords, metric="haversine")
    _, nbr = tree.query(coords, k=k + 1)
    nbr = nbr[:, 1:]  # drop self -> (n, k) indices into `stations`

    # daily calendar + per-station forward-filled exceedance state matrix S (n_stations x n_days)
    cal = pd.date_range(df["sample_date"].min(), df["sample_date"].max(), freq="D")
    col_of = {d: i for i, d in enumerate(cal)}
    D = len(cal)
    S = np.full((n, D), np.nan)
    pos = {s: i for i, s in enumerate(stations)}
    for sid, g in df[df["station_id"].isin(set(stations))].groupby("station_id"):
        ser = g.set_index("sample_date")["exceed"].groupby(level=0).max().reindex(cal).ffill()
        S[pos[sid]] = ser.to_numpy()
    # shift right by the reveal lag: column t now holds the value known as-of (t - lag)
    lag = int(reveal_lag_days)
    if lag > 0:
        Sl = np.full_like(S, np.nan)
        Sl[:, lag:] = S[:, :-lag]
        S = Sl

    # neighbour-mean state per station (nanmean over its k neighbours), and a 7-day mean
    nbr_state = np.full((n, D), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices before first obs
        for i in range(n):
            nbr_state[i] = np.nanmean(S[nbr[i]], axis=0)
    nbr7 = pd.DataFrame(nbr_state.T).rolling(7, min_periods=1).mean().to_numpy().T

    p1 = np.full(len(df), np.nan)
    p7 = np.full(len(df), np.nan)
    sid_arr = df["station_id"].to_numpy()
    date_arr = df["sample_date"].to_numpy()
    for r in range(len(df)):
        i = pos.get(sid_arr[r])
        if i is None:
            continue
        c = col_of.get(pd.Timestamp(date_arr[r]))
        if c is None:
            continue
        p1[r] = nbr_state[i, c]
        p7[r] = nbr7[i, c]
    df["nbr_prev1"] = p1
    df["nbr_prev7"] = p7
    return df


def _fit_eval(tr, va, te, feats, base):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42)
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    raw = clf.predict_proba(te[feats].astype(float))[:, 1]
    p = raw
    if va["exceed"].nunique() > 1:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(va[feats].astype(float))[:, 1], va["exceed"].to_numpy())
        p = iso.predict(raw)
    return p


def run(obs_path: Path, rain_dir=None, reveal_lag_days: int = 2, label: str = "enterococcus",
        exclude_counties=("San Diego",), k: int = 8, n_perm: int = 499) -> dict:
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    # add_causal_features now injects spatial columns of its own (the geo block added when
    # station_geo started shipping). Drop them so THIS experiment owns the spatial arms: a
    # second merge of latitude/longitude below would otherwise collide into latitude_x/_y and
    # _ensure_optional_feature_columns would silently re-create latitude/longitude as all-NaN,
    # making every "+spatial" arm identical to no-spatial (an inert-feature false wash).
    df = df.drop(columns=[c for c in ("latitude", "longitude", "nbr_prev1", "nbr_prev7")
                          if c in df.columns])
    feats_base = list(ob.FEATS)
    if rain_dir is not None:
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats_base += ob.RAIN_FEATS

    geo = _load_geo()
    if geo is None:
        return {"error": "station_geo.parquet not found"}
    df = add_knn_spatial_lag(df, geo, k=k, reveal_lag_days=reveal_lag_days)
    # raw coordinates so the trees can learn arbitrary STATIC spatial structure
    # (per-location bias + regionally-varying driver response) the neighbour-lag missed.
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    df = df.merge(geo_t[["station_id", "latitude", "longitude"]], on="station_id", how="left")
    df = ob._ensure_optional_feature_columns(df)

    # candidate spatial encodings, each ADDED to the same driver baseline
    arms = {
        "driver_model_no_spatial": _unique_features(feats_base),
        "plus_nbr_lag": _unique_features(feats_base + SPATIAL_FEATS),            # neighbours' recent exceedance state
        "plus_latlon": _unique_features(feats_base + ["latitude", "longitude"]),  # static spatial structure
        "plus_nbr_and_latlon": _unique_features(feats_base + SPATIAL_FEATS + ["latitude", "longitude"]),
    }

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    base = float(tr["exceed"].mean())
    y = te["exceed"].to_numpy()

    scores = {"site_memory": ob.score(y, te["station_prior_rate"].fillna(base).to_numpy(), base)}
    moran = {}
    for name, feats in arms.items():
        te[f"p_{name}"] = _fit_eval(tr, va, te, feats, base)
        scores[name] = ob.score(y, te[f"p_{name}"].to_numpy(), base)
        resid = (te.assign(r=te["exceed"] - te[f"p_{name}"]).groupby("station_id")["r"].mean()
                   .rename("resid").reset_index())
        j = resid.merge(geo_t, on="station_id", how="inner")
        moran[name] = (morans_i(j["resid"].to_numpy(), j[["latitude", "longitude"]].to_numpy(),
                                k=k, n_perm=n_perm) if len(j) >= 3 else {"morans_i": None})

    ns_ap = scores["driver_model_no_spatial"]["ap"]
    ns_mi = moran["driver_model_no_spatial"].get("morans_i")
    mem_ap = scores["site_memory"]["ap"]
    best = max((a for a in arms if a != "driver_model_no_spatial"),
               key=lambda a: scores[a]["ap"])
    verdict = {
        "no_spatial_ap": round(ns_ap, 4), "site_memory_ap": round(mem_ap, 4),
        "no_spatial_morans_i": ns_mi,
        "best_spatial_arm": best,
        "best_spatial_ap": round(scores[best]["ap"], 4),
        "best_ap_minus_no_spatial": round(scores[best]["ap"] - ns_ap, 4),
        "best_arm_morans_i": moran[best].get("morans_i"),
        "any_arm_beats_no_spatial": bool(any(scores[a]["ap"] > ns_ap for a in arms if a != "driver_model_no_spatial")),
        "any_arm_reduces_clustering": bool(any(
            (moran[a].get("morans_i") or 1) < (ns_mi or 0) for a in arms if a != "driver_model_no_spatial")),
    }
    # CRITICAL verification: does the lat/lon lift survive on NEVER-SEEN beaches? If the gain
    # is only per-station memorization it vanishes under leave-one-beach-out; if it survives,
    # the model learned a genuine spatial RISK SURFACE that interpolates to new beaches.
    lobo_base = leave_one_beach_out(df, _unique_features(feats_base), n_splits=5)
    lobo_latlon = leave_one_beach_out(df, _unique_features(feats_base + ["latitude", "longitude"]), n_splits=5)
    lobo_ap_base = lobo_base.get("models", {}).get("model_calibrated", {}).get("ap")
    lobo_ap_latlon = lobo_latlon.get("models", {}).get("model_calibrated", {}).get("ap")
    verdict["lobo_ap_no_spatial"] = lobo_ap_base
    verdict["lobo_ap_plus_latlon"] = lobo_ap_latlon
    verdict["lobo_latlon_delta"] = (round(lobo_ap_latlon - lobo_ap_base, 4)
                                    if (lobo_ap_base and lobo_ap_latlon) else None)
    verdict["spatial_surface_generalizes"] = bool(
        lobo_ap_base and lobo_ap_latlon and lobo_ap_latlon > lobo_ap_base)

    return {"label": label, "excluded_counties": list(exclude_counties or []),
            "reveal_lag_days": reveal_lag_days, "k_neighbors": k,
            "n_test": int(len(te)), "events": int(y.sum()), "base_rate": round(float(y.mean()), 4),
            "scores": scores, "residual_morans_i": moran, "verdict": verdict}


def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"spatial_drivers_experiment error: {res['error']}"
    s, m, v = res["scores"], res["residual_morans_i"], res["verdict"]
    lines = [
        "# Spatial-driver experiment: do spatial encodings beat the no-spatial driver model?",
        "",
        f"- label={res['label']} | excluded={res['excluded_counties']} | lag={res['reveal_lag_days']}d "
        f"| k={res['k_neighbors']} | test rows={res['n_test']} | events={res['events']} | base={res['base_rate']}",
        "",
        "| model | AP | ROC-AUC | ECE | residual Moran's I |",
        "|---|--:|--:|--:|--:|",
        f"| site-memory | {s['site_memory']['ap']} | {s['site_memory']['roc_auc']} | {s['site_memory']['ece']} | - |",
    ]
    for name in ["driver_model_no_spatial", "plus_nbr_lag", "plus_latlon", "plus_nbr_and_latlon"]:
        sc = s[name]
        lines.append(f"| {name} | {sc['ap']} | {sc['roc_auc']} | {sc['ece']} | {m[name].get('morans_i')} |")
    lines += [
        "",
        f"- best spatial arm: **{v['best_spatial_arm']}** AP {v['best_spatial_ap']} "
        f"(vs no-spatial {v['no_spatial_ap']}, delta **{v['best_ap_minus_no_spatial']:+}**)",
        f"- residual Moran's I: no-spatial **{v['no_spatial_morans_i']}** -> best arm **{v['best_arm_morans_i']}**",
        f"- any spatial arm beats no-spatial AP: **{v['any_arm_beats_no_spatial']}** | "
        f"any arm reduces clustering: **{v['any_arm_reduces_clustering']}**",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default="bacteria_results/rainfall")
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--label", default="enterococcus", choices=["any", "enterococcus"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--include-san-diego", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[spatial_drivers_experiment] obs not found: {obs}")
        return 2
    excl = () if args.include_san_diego else ("San Diego",)
    res = run(obs, rain_dir=args.rain_dir, reveal_lag_days=args.reveal_lag_days,
              label=args.label, exclude_counties=excl, k=args.k, n_perm=args.n_perm)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spatial_drivers_experiment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "spatial_drivers_experiment.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
