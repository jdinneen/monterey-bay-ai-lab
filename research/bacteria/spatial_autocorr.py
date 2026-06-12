#!/usr/bin/env python3
"""Leave-one-beach-out (LOBO) generalization + residual spatial-autocorrelation (Moran's I).

`spatial_holdout.py` answers the *county*-level reviewer objection ("predict a county you
never trained on"). A harder reviewer objection remains: counties are coarse, and
**co-located beaches** can still split across a county fold, so a county holdout can be
flattered by near-neighbor stations the model *did* train on. This module closes that gap
two ways:

1. **Leave-one-beach-out (grouped by station):** a GroupKFold over `station_id` puts every
   beach in exactly one held-out fold. For each fold we train on the *other* beaches
   (through 2021), calibrate on other-beach 2020-21, and test on the held-out beaches'
   2022+ samples. The county/statewide prior-rate aggregates (`cty_prev7`, `sw_prev7`) are
   **recomputed per fold with the held-out beaches removed**, so a held-out beach's own
   labels cannot leak into the aggregates other beaches train on (the LOCO discipline,
   applied at beach granularity). Out-of-fold predictions over all beaches are then scored
   once — this is skill on beaches the model has *never* seen.

2. **Moran's I on per-beach residuals:** even a model that scores well can be quietly
   leaning on spatial proximity. We compute Moran's I of each beach's mean calibrated
   residual (using real lat/lon, k-nearest-neighbour row-standardised weights, haversine).
   Residuals that cluster spatially (I >> expected, low permutation p) signal unmodelled
   local structure / proximity leakage; residuals that look spatially random (I near the
   expected -1/(n-1)) are the reassuring outcome.

Reuses the canonical leakage-safe pipeline from operational_benchmark.
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


def _default_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42,
    )


def _agg_prev7_excluding(df: pd.DataFrame, holdout_ids: set) -> tuple[np.ndarray, np.ndarray]:
    """Recompute (cty_prev7, sw_prev7) using ONLY non-held-out beaches, mapped back onto
    every row of ``df`` (order-preserving). Held-out beaches' labels therefore never enter
    the county/statewide prior-rate aggregates that training rows see."""
    keep = df[~df["station_id"].isin(holdout_ids)]
    cty = (keep.groupby(["county", "sample_date"])["exceed"].mean().rename("r")
               .reset_index().sort_values(["county", "sample_date"]))
    cty["cty_prev7"] = (cty.groupby("county")["r"]
                        .apply(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
                        .reset_index(level=0, drop=True))
    sw = (keep.groupby("sample_date")["exceed"].mean().rename("r")
              .reset_index().sort_values("sample_date"))
    sw["sw_prev7"] = sw["r"].shift(1).rolling(7, min_periods=1).mean()
    # drop any pre-existing aggregate cols so the re-merge doesn't collide (suffix _x/_y)
    base = df.drop(columns=["cty_prev7", "sw_prev7"], errors="ignore")
    cty_map = base.merge(cty[["county", "sample_date", "cty_prev7"]],
                         on=["county", "sample_date"], how="left")["cty_prev7"].to_numpy()
    sw_map = base.merge(sw[["sample_date", "sw_prev7"]],
                        on="sample_date", how="left")["sw_prev7"].to_numpy()
    return cty_map, sw_map


def leave_one_beach_out(df: pd.DataFrame, feats: list[str], clf_factory=None,
                        n_splits: int = 5, seed: int = 42) -> dict:
    """GroupKFold over station_id. Returns out-of-fold calibrated predictions on 2022+
    rows of never-trained-on beaches, plus the comparison baselines."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    clf_factory = clf_factory or _default_clf
    df = df.sort_values(["station_id", "sample_date"]).reset_index(drop=True)
    stations = df["station_id"].to_numpy()
    # deterministic group assignment independent of row order
    uniq = np.array(sorted(df["station_id"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_station = {sid: int(perm[i] % n_splits) for i, sid in enumerate(uniq)}
    df["_fold"] = df["station_id"].map(fold_of_station)

    oof = []  # held-out test rows with predictions
    for fold in range(n_splits):
        holdout_ids = set(uniq[[fold_of_station[s] == fold for s in uniq]])
        if not holdout_ids:
            continue
        d = df.copy()
        cty_map, sw_map = _agg_prev7_excluding(d, holdout_ids)
        d["cty_prev7"] = cty_map
        d["sw_prev7"] = sw_map

        is_hold = d["station_id"].isin(holdout_ids)
        train = d[~is_hold & (d["sample_date"] <= "2019-12-31")]
        cal = d[~is_hold & (d["sample_date"] >= "2020-01-01") & (d["sample_date"] <= "2021-12-31")]
        test = d[is_hold & (d["sample_date"] >= "2022-01-01")].copy()
        if test.empty or train["exceed"].nunique() < 2:
            continue

        clf = clf_factory()
        # drop features constant in THIS fold's training set (sklearn>=1.9 HGBT binning
        # errors on a single-distinct-value feature); mirrors operational_benchmark's guard.
        # Fold-local (do NOT mutate `feats` — it must stay full for the next fold).
        ff = [c for c in feats if train[c].notna().sum() >= 2 and train[c].nunique(dropna=True) > 1]
        clf.fit(train[ff].astype(float), train["exceed"].to_numpy())
        raw = clf.predict_proba(test[ff].astype(float))[:, 1]
        p_cal = raw
        if cal["exceed"].nunique() > 1:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(clf.predict_proba(cal[ff].astype(float))[:, 1], cal["exceed"].to_numpy())
            p_cal = iso.predict(raw)
        test["p_model"] = p_cal
        oof.append(test)

    if not oof:
        return {"error": "no out-of-fold test rows produced"}
    O = pd.concat(oof, ignore_index=True)
    base = float(df[df["sample_date"] <= "2019-12-31"]["exceed"].mean())
    y = O["exceed"].to_numpy()
    rows = {
        "model_calibrated": ob.score(y, O["p_model"].to_numpy(), base),
        "baseline_station_memory": ob.score(y, O["station_prior_rate"].fillna(base).to_numpy(), base),
        "baseline_prior_lab": ob.score(y, O["exc_prev"].fillna(base).to_numpy(), base),
    }
    if "rain_3d" in O.columns:
        ab = (O["rain_3d"].fillna(0.0) >= ob.AB411_RAIN_MM).astype(float).to_numpy()
        rows["baseline_ab411_rain"] = ob.score(y, ab, base)

    m_ap = rows["model_calibrated"].get("ap")
    mem_ap = rows["baseline_station_memory"].get("ap")
    ab_ap = rows.get("baseline_ab411_rain", {}).get("ap")
    return {
        "n_splits": n_splits,
        "n_test_beaches": int(O["station_id"].nunique()),
        "n_test_rows": int(len(O)),
        "events": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "models": rows,
        "model_beats_memory": (None if (m_ap is None or mem_ap is None) else bool(m_ap > mem_ap)),
        "model_beats_ab411": (None if (m_ap is None or ab_ap is None) else bool(m_ap > ab_ap)),
        "_oof": O[["station_id", "exceed", "p_model"]],  # for Moran's I; stripped before JSON
    }


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 8,
             n_perm: int = 999, seed: int = 42) -> dict:
    """Moran's I of ``values`` over points ``coords`` (lat, lon in degrees), using
    k-nearest-neighbour, row-standardised binary weights and haversine distance. A
    permutation test gives a one-sided p-value for *positive* spatial autocorrelation
    (clustered residuals). Expected I under spatial randomness is -1/(n-1)."""
    from sklearn.neighbors import BallTree

    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return {"n": int(n), "morans_i": None, "note": "too few points"}
    k = int(min(k, n - 1))
    rad = np.radians(coords)
    tree = BallTree(rad, metric="haversine")
    _, idx = tree.query(rad, k=k + 1)
    idx = idx[:, 1:]  # drop self

    def _I(v: np.ndarray) -> float:
        z = v - v.mean()
        denom = float((z ** 2).sum())
        if denom == 0:
            return 0.0
        # row-standardised weights => S0 = n => leading factor n/S0 = 1
        num = float(np.sum(z * (z[idx].sum(axis=1) / k)))
        return num / denom

    i_obs = _I(values)
    rng = np.random.default_rng(seed)
    perm = np.array([_I(rng.permutation(values)) for _ in range(n_perm)])
    p_one_sided = float((1 + np.sum(perm >= i_obs)) / (1 + n_perm))
    return {
        "n": int(n), "k_neighbors": k, "n_perm": n_perm,
        "morans_i": round(i_obs, 4),
        "expected_i": round(-1.0 / (n - 1), 4),
        "perm_mean_i": round(float(perm.mean()), 4),
        "perm_std_i": round(float(perm.std()), 4),
        "p_value_positive_autocorr": round(p_one_sided, 4),
        "interpretation": (
            "residuals spatially clustered (possible unmodelled local structure)"
            if p_one_sided < 0.05 else
            "residuals ~spatially random (no proximity-leakage signal)"),
    }


def _load_geo() -> pd.DataFrame | None:
    p = _REPO_ROOT / "reports" / "station_geo.parquet"
    if not p.exists():
        return None
    g = pd.read_parquet(p)
    return g[["station_id", "latitude", "longitude"]].dropna()


def run(obs_path: Path, rain_dir=None, clf_factory=None, n_splits: int = 5,
        reveal_lag_days: int = 0, label: str = "any", moran_k: int = 8,
        n_perm: int = 999, exclude_counties: list[str] | None = None) -> dict:
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        # honest-headline parity with operational_benchmark's EXCLUDE_SAN_DIEGO stratum:
        # the 2022 San Diego/Tijuana regime break otherwise dominates both AP and residuals.
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    feats = list(ob.FEATS)
    has_rain = False
    if rain_dir is not None:
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += ob.RAIN_FEATS
            has_rain = True

    lobo = leave_one_beach_out(df, feats, clf_factory=clf_factory, n_splits=n_splits)
    oof = lobo.pop("_oof", None)

    moran = {"note": "station_geo.parquet not found; skipped"}
    if oof is not None and isinstance(oof, pd.DataFrame):
        geo = _load_geo()
        if geo is not None:
            resid = (oof.assign(r=oof["exceed"] - oof["p_model"])
                        .groupby("station_id")["r"].mean().rename("resid").reset_index())
            j = resid.merge(geo, on="station_id", how="inner")
            if len(j) >= 3:
                moran = morans_i(j["resid"].to_numpy(),
                                 j[["latitude", "longitude"]].to_numpy(),
                                 k=moran_k, n_perm=n_perm)
                moran["beaches_geolocated"] = int(len(j))
    return {"has_rain": has_rain, "label": label,
            "excluded_counties": exclude_counties or [],
            "leave_one_beach_out": lobo, "residual_morans_i": moran}


def to_markdown(res: dict) -> str:
    l = res["leave_one_beach_out"]
    m = res["residual_morans_i"]
    lines = ["# Leave-one-beach-out generalization + residual Moran's I", ""]
    if "error" in l:
        return "\n".join(lines + [f"LOBO error: {l['error']}"])
    mc = l["models"]["model_calibrated"]
    mem = l["models"]["baseline_station_memory"]
    ab = l["models"].get("baseline_ab411_rain", {})
    lines += [
        f"- rainfall used: {res['has_rain']} | label: {res['label']} | folds: {l['n_splits']}",
        f"- held-out beaches scored: {l['n_test_beaches']} | test rows: {l['n_test_rows']} "
        f"| events: {l['events']} | base rate: {l['base_rate']}",
        "",
        "| metric | model (never-seen beaches) | station-memory | AB411 rain |",
        "|---|--:|--:|--:|",
        f"| AP | {mc.get('ap','-')} | {mem.get('ap','-')} | {ab.get('ap','-')} |",
        f"| ROC-AUC | {mc.get('roc_auc','-')} | {mem.get('roc_auc','-')} | {ab.get('roc_auc','-')} |",
        f"| ECE | {mc.get('ece','-')} | {mem.get('ece','-')} | {ab.get('ece','-')} |",
        "",
        f"- model beats station-memory: {l['model_beats_memory']} | beats AB411: {l['model_beats_ab411']}",
        "",
        "## Residual spatial autocorrelation (Moran's I)",
        f"- Moran's I = {m.get('morans_i')} (expected under randomness {m.get('expected_i')}), "
        f"k={m.get('k_neighbors')} neighbours, {m.get('beaches_geolocated','?')} beaches",
        f"- one-sided p (positive autocorr) = {m.get('p_value_positive_autocorr')} -> {m.get('interpretation')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--reveal-lag-days", type=int, default=0)
    ap.add_argument("--label", default="any", choices=["any", "enterococcus"])
    ap.add_argument("--moran-k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--exclude-counties", default=None,
                    help="comma-separated counties to drop (e.g. 'San Diego' for honest-headline parity)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[spatial_autocorr] obs not found: {obs}")
        return 2
    excl = [c.strip() for c in args.exclude_counties.split(",")] if args.exclude_counties else None
    res = run(obs, rain_dir=args.rain_dir, n_splits=args.n_splits,
              reveal_lag_days=args.reveal_lag_days, label=args.label,
              moran_k=args.moran_k, n_perm=args.n_perm, exclude_counties=excl)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spatial_autocorr.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "spatial_autocorr.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
