#!/usr/bin/env python3
"""Physical spatial covariates vs the raw lat/lon surface — does mechanism beat black-box?

spatial_drivers_experiment.py shows raw lat/lon (a black-box spatial surface) is a wash:
it does not lift AP or reduce residual Moran's I once the station/county/statewide rate
features are present. The transferable positive result is that the base driver model
generalises to never-seen *beaches*. Physical covariates -- distance to the nearest
freshwater discharge point, local monitoring density as a proxy for urbanisation/embayment
-- test whether mechanism can explain the residual spatial structure that raw coordinates
do not capture.

This compares, on the honest San-Diego-excluded enterococcus holdout:
    no-spatial  vs  +lat/lon  vs  +physical  vs  +lat/lon+physical
on two evaluations:
  1. temporal hold-out (same beaches, 2022+ test) -- AP / ROC-AUC / ECE / residual Moran's I;
  2. **leave-one-COUNTY-out** -- the decisive test: a held-out region has NO training beaches
     nearby, so raw lat/lon must extrapolate (and should fail) while a physical covariate
     transfers. If +physical beats +lat/lon under LOCO, that is the fundable "predict an
     unmonitored REGION" capability.

Physical covariates are sourced from data already in the repo (no network):
  - `dist_to_gauge_km`  : distance to nearest USGS discharge gauge (river/creek influence),
                          from bacteria_results/discharge/station_gauge_map.parquet
  - `stn_density_10km`, `stn_density_25km` : count of other monitoring stations within R km
                          (urbanised / enclosed shorelines are densely monitored), from coords.
  - `dist_to_npdes_km`, `npdes_count_*`, `dist_to_sso_km`, `sso_count_*` : source-proximity
                          features from the staged station_static_features table when present.
  - `ccap_developed`     : NOAA C-CAP 2021 developed-land indicator sampled at the station.
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
from research.bacteria.spatial_autocorr import morans_i, _load_geo  # noqa: E402

PHYS_FEATS = [
    "dist_to_gauge_km",
    "stn_density_10km",
    "stn_density_25km",
    "dist_to_npdes_km",
    "npdes_count_5km",
    "npdes_count_25km",
    "dist_to_sso_km",
    "sso_count_5km",
    "sso_count_25km",
    "ccap_developed",
]
LATLON = ["latitude", "longitude"]
_EARTH_KM = 6371.0
STATION_STATIC_PATH = _REPO_ROOT / "data" / "external_curated" / "station_static_features" / "station_static_features.parquet"
_SPATIAL_EXCLUDE_FROM_BASE = set(LATLON + PHYS_FEATS + ["nbr_prev1", "nbr_prev7"])


def _unique(seq):
    return list(dict.fromkeys(seq))


def add_physical_covariates(df: pd.DataFrame, geo: pd.DataFrame,
                            gauge_map_path: Path | None = None,
                            station_static_path: Path | None = None) -> pd.DataFrame:
    """Attach static physical spatial covariates per station (no leakage: all are fixed
    geography, known a priori, identical in train and test)."""
    from sklearn.neighbors import BallTree

    static_path = Path(station_static_path) if station_static_path else STATION_STATIC_PATH
    if static_path.exists():
        static = pd.read_parquet(static_path).drop_duplicates("station_id")
        cols = [
            "station_id",
            "latitude",
            "longitude",
            "station_density_10km",
            "station_density_25km",
            "dist_to_usgs_gauge_km",
            "dist_to_npdes_km",
            "npdes_count_5km",
            "npdes_count_25km",
            "dist_to_sso_km",
            "sso_count_5km",
            "sso_count_25km",
            "ccap_developed",
        ]
        static = static[[c for c in cols if c in static.columns]].copy()
        static = static.rename(columns={
            "station_density_10km": "stn_density_10km",
            "station_density_25km": "stn_density_25km",
            "dist_to_usgs_gauge_km": "dist_to_gauge_km",
        })
        replace_cols = [c for c in static.columns if c != "station_id" and c in df.columns]
        base = df.drop(columns=replace_cols)
        out = base.merge(static, on="station_id", how="left")
        for col in PHYS_FEATS:
            if col not in out.columns:
                out[col] = np.nan
        if "ccap_developed" in out.columns:
            out["ccap_developed"] = out["ccap_developed"].astype("float")
        return out

    geo = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    df = df.merge(geo[["station_id", "latitude", "longitude"]], on="station_id", how="left")

    # local monitoring density (other stations within R km) via haversine radius query
    g = geo.set_index("station_id")
    coords = np.radians(g[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    dens = {}
    for r_km, col in [(10.0, "stn_density_10km"), (25.0, "stn_density_25km")]:
        counts = tree.query_radius(coords, r=r_km / _EARTH_KM, count_only=True) - 1  # drop self
        dens[col] = pd.Series(counts, index=g.index)
    dens_df = pd.DataFrame(dens)
    df = df.merge(dens_df, left_on="station_id", right_index=True, how="left")

    # distance to nearest USGS discharge gauge (freshwater influence)
    gm_path = Path(gauge_map_path) if gauge_map_path else (
        _REPO_ROOT / "bacteria_results" / "discharge" / "station_gauge_map.parquet")
    if gm_path.exists():
        gm = pd.read_parquet(gm_path)[["station_id", "distance_km"]].rename(
            columns={"distance_km": "dist_to_gauge_km"})
        df = df.merge(gm, on="station_id", how="left")
    else:
        df["dist_to_gauge_km"] = np.nan
    for col in PHYS_FEATS:
        if col not in df.columns:
            df[col] = np.nan
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


def _temporal_holdout(df, arms, geo_t, k, n_perm):
    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    base = float(tr["exceed"].mean())
    y = te["exceed"].to_numpy()
    out = {}
    for name, feats in arms.items():
        p = _fit_eval(tr, va, te, feats, base)
        sc = ob.score(y, p, base)
        resid = (te.assign(r=te["exceed"] - p).groupby("station_id")["r"].mean()
                   .rename("resid").reset_index().merge(geo_t, on="station_id", how="inner"))
        mi = (morans_i(resid["resid"].to_numpy(), resid[["latitude", "longitude"]].to_numpy(),
                       k=k, n_perm=n_perm).get("morans_i") if len(resid) >= 3 else None)
        out[name] = {"ap": sc["ap"], "roc_auc": sc["roc_auc"], "ece": sc["ece"], "morans_i": mi}
    return {"n_test": int(len(te)), "events": int(y.sum()),
            "base_rate": round(float(y.mean()), 4), "arms": out}


def _leave_one_county_out(df, arms, min_test_events=150, max_counties=9):
    """For each arm, train on every OTHER county (<=2021), calibrate on other-county 2020-21,
    test on the held-out county 2022+. Aggregates each arm's median AP across held-out counties.
    sw_prev7 is recomputed per fold excluding the held-out county (reuses the LOCO discipline)."""
    df = df.sort_values(["station_id", "sample_date"]).reset_index(drop=True)
    test_era = df[df["sample_date"] >= "2022-01-01"]
    ev = test_era.groupby("county")["exceed"].sum().sort_values(ascending=False)
    counties = [c for c, n in ev.items() if n >= min_test_events][:max_counties]

    def _sw_excl(d, county):
        day = d.groupby("sample_date")["exceed"].agg(s="sum", n="count")
        c = d[d["county"] == county].groupby("sample_date")["exceed"].agg(cs="sum", cn="count")
        j = day.join(c, how="left").fillna({"cs": 0, "cn": 0})
        denom = (j["n"] - j["cn"]).replace(0, np.nan)
        rate = ((j["s"] - j["cs"]) / denom).sort_index()
        return d["sample_date"].map(rate.shift(1).rolling(7, min_periods=1).mean())

    per_arm = {name: [] for name in arms}
    for county in counties:
        d = df.copy()
        d["sw_prev7"] = _sw_excl(d, county)
        tr = d[(d["county"] != county) & (d["sample_date"] <= "2019-12-31")]
        va = d[(d["county"] != county) & (d["sample_date"] >= "2020-01-01") & (d["sample_date"] <= "2021-12-31")]
        te = d[(d["county"] == county) & (d["sample_date"] >= "2022-01-01")].copy()
        if te.empty or te["exceed"].nunique() < 2 or tr["exceed"].nunique() < 2:
            continue
        base = float(tr["exceed"].mean())
        y = te["exceed"].to_numpy()
        for name, feats in arms.items():
            p = _fit_eval(tr, va, te, feats, base)
            per_arm[name].append(ob.score(y, p, base)["ap"])
    return {"counties_evaluated": counties,
            "median_ap": {name: (round(float(np.median(v)), 4) if v else None)
                          for name, v in per_arm.items()}}


def run(obs_path: Path, rain_dir="bacteria_results/rainfall", reveal_lag_days=2,
        label="enterococcus", exclude_counties=("San Diego",), k=8, n_perm=499,
        do_loco=True) -> dict:
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    feats_base = [f for f in list(ob.FEATS) if f not in _SPATIAL_EXCLUDE_FROM_BASE]
    if rain_dir is not None:
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats_base += ob.RAIN_FEATS

    geo = _load_geo()
    if geo is None:
        return {"error": "station_geo.parquet not found"}
    df = add_physical_covariates(df, geo)
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")

    arms = {
        "no_spatial": _unique(feats_base),
        "plus_latlon": _unique(feats_base + LATLON),
        "plus_physical": _unique(feats_base + PHYS_FEATS),
        "plus_latlon_physical": _unique(feats_base + LATLON + PHYS_FEATS),
    }
    res = {"label": label, "excluded_counties": list(exclude_counties or []),
           "reveal_lag_days": reveal_lag_days,
           "temporal_holdout": _temporal_holdout(df, arms, geo_t, k, n_perm)}
    if do_loco:
        res["leave_one_county_out"] = _leave_one_county_out(df, arms)

    # verdicts
    th = res["temporal_holdout"]["arms"]
    base_ap = th["no_spatial"]["ap"]
    res["verdict"] = {
        "physical_ap_minus_no_spatial": round(th["plus_physical"]["ap"] - base_ap, 4),
        "physical_ap_minus_latlon": round(th["plus_physical"]["ap"] - th["plus_latlon"]["ap"], 4),
        "both_ap_minus_latlon": round(th["plus_latlon_physical"]["ap"] - th["plus_latlon"]["ap"], 4),
        "moran_no_spatial": th["no_spatial"]["morans_i"],
        "moran_both": th["plus_latlon_physical"]["morans_i"],
    }
    if do_loco:
        lm = res["leave_one_county_out"]["median_ap"]
        res["verdict"]["loco_physical_minus_latlon"] = (
            round(lm["plus_physical"] - lm["plus_latlon"], 4)
            if lm.get("plus_physical") and lm.get("plus_latlon") else None)
    return res


def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"physical_spatial_covariates error: {res['error']}"
    th = res["temporal_holdout"]
    lines = [
        "# Physical spatial covariates vs the raw lat/lon surface",
        "",
        f"- label={res['label']} | excluded={res['excluded_counties']} | lag={res['reveal_lag_days']}d "
        f"| test rows={th['n_test']} | events={th['events']} | base={th['base_rate']}",
        "",
        "## Temporal hold-out (same beaches)",
        "| arm | AP | ROC-AUC | ECE | residual Moran's I |",
        "|---|--:|--:|--:|--:|",
    ]
    for name in ["no_spatial", "plus_latlon", "plus_physical", "plus_latlon_physical"]:
        a = th["arms"][name]
        lines.append(f"| {name} | {a['ap']} | {a['roc_auc']} | {a['ece']} | {a['morans_i']} |")
    if "leave_one_county_out" in res:
        lm = res["leave_one_county_out"]["median_ap"]
        lines += [
            "",
            "## Leave-one-COUNTY-out (held-out region — the extrapolation test)",
            f"- counties: {res['leave_one_county_out']['counties_evaluated']}",
            "| arm | median held-out-county AP |",
            "|---|--:|",
        ]
        for name in ["no_spatial", "plus_latlon", "plus_physical", "plus_latlon_physical"]:
            lines.append(f"| {name} | {lm.get(name)} |")
    v = res["verdict"]
    lines += [
        "",
        f"- physical vs no-spatial AP: **{v['physical_ap_minus_no_spatial']:+}** | "
        f"physical vs lat/lon: **{v['physical_ap_minus_latlon']:+}** | "
        f"physical on top of lat/lon: **{v['both_ap_minus_latlon']:+}**",
        f"- Moran's I no-spatial **{v['moran_no_spatial']}** -> +both **{v['moran_both']}**",
    ]
    if "loco_physical_minus_latlon" in v:
        lines.append(f"- **LOCO physical - lat/lon median AP: {v['loco_physical_minus_latlon']:+}** "
                     "(positive => physical extrapolates to new regions better than coordinates)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default="bacteria_results/rainfall")
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--label", default="enterococcus", choices=["any", "enterococcus"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--include-san-diego", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[physical_spatial_covariates] obs not found: {obs}")
        return 2
    excl = () if args.include_san_diego else ("San Diego",)
    res = run(obs, rain_dir=args.rain_dir, reveal_lag_days=args.reveal_lag_days, label=args.label,
              exclude_counties=excl, k=args.k, n_perm=args.n_perm, do_loco=not args.no_loco)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "physical_spatial_covariates.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "physical_spatial_covariates.md").write_text(to_markdown(res), encoding="utf-8")
    try:
        print(to_markdown(res))
    except UnicodeEncodeError:
        print("(wrote physical_spatial_covariates.{json,md}; console cp1252 cannot print glyphs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
