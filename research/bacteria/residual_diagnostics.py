#!/usr/bin/env python3
"""Residual diagnostics — the DISCOVERY front-end for the signal lab.

`signal_lab.py` can GATE a derived signal once you've thought of it, but it can't tell you
WHICH signal to try. This module does: it fits the current best model, takes its calibrated
holdout residuals, and asks "where is the model systematically wrong?" along several axes.
Each axis with real structure is a place a new level-2 signal can live — so the output is a
*ranked candidate backlog*, not a guess. This is exactly how lat/lon was found (residual
Moran's I flagged unmodelled spatial structure); here we generalise that to every axis.

Axes screened (each → an effect size + the signal_lab candidate it suggests):
  - spatial      : Moran's I of per-station mean residual (reuses spatial_autocorr.morans_i).
  - temporal     : how much residual variance month / day-of-year bins explain (seasonal
                   miscalibration the doy_sin/doy_cos pair didn't capture).
  - station      : between-station vs within-station residual variance (ICC) — systematically
                   mis-predicted beaches the features don't separate.
  - county       : same at county granularity.
  - driver-decile: residual mean across deciles of each PRESENT driver (rain_3d, discharge_log,
                   Hs, water_level_m). A monotone/curved residual-vs-decile profile = unmodelled
                   nonlinearity or interaction in that driver.

Headroom floor. Before chasing any of this we estimate an irreducible-error band: among
near-duplicate covariate neighbourhoods (k-NN in standardised feature space) the label still
disagrees at some rate — that disagreement is a lower bound on achievable error (a Bayes-error
proxy). If the model already sits near the floor, deeper signal-stacking won't pay; the report
says so. This is the honest "is there headroom?" check a reviewer demands.

Nothing is materialised. This is a screen; it writes a report, not a feature.

CLI:
    python -m research.bacteria.residual_diagnostics \
        --obs bacteria_results/statewide/statewide_beach_observations.parquet
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
from research.bacteria import signal_lab as sl  # noqa: E402
from research.bacteria.spatial_autocorr import _load_geo, morans_i  # noqa: E402

# Effect-size thresholds above which an axis is worth a candidate. Deliberately conservative
# so we don't manufacture work from noise (the value gate's anti-bloat bias).
SPATIAL_MORANS_MIN = 0.05      # residual Moran's I (vs ~0 expected if spatially random)
TEMPORAL_ETA2_MIN = 0.01       # fraction of residual variance explained by the calendar bin
GROUP_ICC_MIN = 0.05           # between-group residual ICC
DRIVER_RANGE_MIN = 0.05        # spread of mean residual across a driver's deciles


def _present_drivers(df: pd.DataFrame) -> dict[str, str]:
    """Map a usable driver column -> the signal_lab candidate that would add it, but only
    for drivers whose data actually landed (non-NaN, non-sentinel)."""
    out = {}
    if "rain_3d" in df.columns and df["rain_3d"].notna().any():
        out["rain_3d"] = "rain"
    if "discharge_log" in df.columns and df["discharge_log"].notna().any():
        out["discharge_log"] = "discharge"
    if "Hs" in df.columns and df["Hs"].notna().any():
        out["Hs"] = "waves"
    if "water_level_m" in df.columns and (df["water_level_m"] != -999.0).any():
        out["water_level_m"] = "tide"
    return out


def _eta_squared(resid: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of residual variance explained by a categorical grouping (one-way ANOVA eta^2,
    = ICC numerator/total). 0 → groups carry no residual structure; →1 → they carry all of it."""
    s = pd.DataFrame({"r": resid, "g": groups}).dropna()
    if s["g"].nunique() < 2 or len(s) < 3:
        return 0.0
    grand = s["r"].mean()
    ss_total = float(((s["r"] - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    gm = s.groupby("g")["r"].agg(["mean", "count"])
    ss_between = float((gm["count"] * (gm["mean"] - grand) ** 2).sum())
    return max(0.0, min(1.0, ss_between / ss_total))


def _driver_decile_profile(resid: np.ndarray, x: np.ndarray, q: int = 10):
    """Mean residual within each decile of driver x. Returns (range, table). A large range
    means the model is biased high in some driver regimes and low in others → unmodelled shape."""
    s = pd.DataFrame({"r": resid, "x": x}).replace({-999.0: np.nan}).dropna()
    if len(s) < q * 5 or s["x"].nunique() < q:
        return 0.0, []
    s["bin"] = pd.qcut(s["x"].rank(method="first"), q, labels=False)
    prof = s.groupby("bin")["r"].mean()
    return float(prof.max() - prof.min()), [round(v, 4) for v in prof.tolist()]


def _irreducible_floor(df: pd.DataFrame, feats: list[str], k: int = 15, sample: int = 6000,
                       seed: int = 42) -> dict:
    """Bayes-error proxy for an IMBALANCED label. In standardised feature space, estimate the
    local exceedance probability p as the positive rate among a point's k nearest neighbours.
    The irreducible Brier under that local oracle is mean p(1-p) — no model with these features
    can score below it, because covariate-near days still disagree on the outcome. Reported as
    a Brier floor directly comparable to the model's Brier (the headroom = model - floor). A
    threshold-disagreement rate is kept as a coarse secondary read.

    Caveat (stated, not hidden): with a ~9% base rate this is a *lower bound* and a coarse one —
    kNN smooths, so the true floor is at least this high. Use it to sanity-check headroom, not
    as a precise ceiling."""
    from sklearn.neighbors import NearestNeighbors

    d = df.dropna(subset=["exceed"]).copy()
    X = d[feats].astype(float).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    y = d["exceed"].to_numpy().astype(float)
    if len(d) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(d), sample, replace=False)
        X, y = X[idx], y[idx]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xs))).fit(Xs)
    _, nbr = nn.kneighbors(Xs)
    nbr = nbr[:, 1:]  # drop self
    p_local = y[nbr].mean(axis=1)               # local exceedance probability (Bayes proxy)
    bayes_brier = float(np.mean(p_local * (1.0 - p_local)))
    disagreement = float(np.mean(np.abs(y - (p_local >= 0.5))))
    base = float(y.mean())
    return {
        "k": k, "n_used": int(len(y)), "base_rate": round(base, 4),
        "bayes_brier_floor": round(bayes_brier, 5),
        "label_disagreement_rate": round(disagreement, 4),
        "note": ("Brier floor = mean p(1-p) over local neighbourhoods (lower bound on any "
                 "model's Brier with these features). Compare to the model's Brier above: a "
                 "small gap ⇒ little headroom for new signals."),
    }


def run(obs_path: Path, *, rain_dir=None, discharge_dir=None, cdip_dir=None, tide_dir=None,
        reveal_lag_days: int = 2, label: str = "enterococcus",
        exclude_counties=("San Diego",), k: int = 8, n_perm: int = 199) -> dict:
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    feats = list(sl.BASE_FEATS)  # pinned baseline (excludes spatial cols promoted into ob.FEATS)
    if rain_dir and Path(rain_dir).exists():
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += ob.RAIN_FEATS
    if discharge_dir and Path(discharge_dir).exists():
        df = ob.add_discharge_features(df, Path(discharge_dir))
    if cdip_dir and Path(cdip_dir).exists():
        df = ob.add_wave_features(df, Path(cdip_dir))
    if tide_dir and Path(tide_dir).exists():
        df = ob.add_tide_stage_features(df, Path(tide_dir))

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    base_rate = float(tr["exceed"].mean())
    te["p"] = sl._fit_eval(tr, va, te, feats)
    te["resid"] = te["exceed"] - te["p"]
    y = te["exceed"].to_numpy()
    model_score = ob.score(y, te["p"].to_numpy(), base_rate)

    axes = []  # each: {axis, effect, metric, suggested_candidate, flagged, detail}

    # --- spatial ---
    geo = _load_geo()
    if geo is not None:
        geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
        per_st = te.groupby("station_id")["resid"].mean().rename("resid").reset_index()
        j = per_st.merge(geo_t, on="station_id", how="inner")
        if len(j) >= 3:
            mi = morans_i(j["resid"].to_numpy(), j[["latitude", "longitude"]].to_numpy(),
                          k=k, n_perm=n_perm)
            axes.append({
                "axis": "spatial", "metric": "residual_morans_i",
                "effect": mi.get("morans_i"), "p_value": mi.get("p_value_positive_autocorr"),
                "suggested_candidate": "latlon",
                "flagged": bool((mi.get("morans_i") or 0) >= SPATIAL_MORANS_MIN),
                "detail": mi.get("interpretation"),
            })

    # --- temporal (calendar bins) ---
    for col, name in [("month", "month"), (te["sample_date"].dt.isocalendar().week.values, "isoweek")]:
        g = te[col].to_numpy() if isinstance(col, str) else col
        eta = _eta_squared(te["resid"].to_numpy(), g)
        axes.append({
            "axis": f"temporal:{name}", "metric": "resid_var_explained_eta2",
            "effect": round(eta, 4), "p_value": None,
            "suggested_candidate": "(new seasonal candidate — none built yet)",
            "flagged": bool(eta >= TEMPORAL_ETA2_MIN), "detail": None,
        })

    # --- station / county grouping ---
    # cty_prev7/sw_prev7 (in the baseline) already encode county/statewide structure; a static
    # region prior (region_embed) was tried and washed, so a NEW signal is only worth it if this
    # ICC clears the bar despite those features already being present.
    for col in ["station_id", "county"]:
        icc = _eta_squared(te["resid"].to_numpy(), te[col].to_numpy())
        axes.append({
            "axis": f"group:{col}", "metric": "between_group_resid_icc",
            "effect": round(icc, 4), "p_value": None,
            "suggested_candidate": "(structure beyond cty_prev7 — no candidate beats it yet)",
            "flagged": bool(icc >= GROUP_ICC_MIN), "detail": None,
        })

    # --- driver deciles ---
    for col, cand in _present_drivers(te).items():
        rng_, prof = _driver_decile_profile(te["resid"].to_numpy(), te[col].to_numpy())
        axes.append({
            "axis": f"driver:{col}", "metric": "resid_range_across_deciles",
            "effect": round(rng_, 4), "p_value": None, "suggested_candidate": cand,
            "flagged": bool(rng_ >= DRIVER_RANGE_MIN), "detail": {"decile_resid": prof},
        })

    axes.sort(key=lambda a: (not a["flagged"], -abs(a["effect"] or 0)))
    floor = _irreducible_floor(df, feats)

    return {
        "label": label, "excluded_counties": list(exclude_counties or []),
        "reveal_lag_days": reveal_lag_days, "n_test": int(len(te)), "events": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "model": {"feats": feats, "ap": model_score["ap"], "roc_auc": model_score["roc_auc"],
                  "ece": model_score["ece"], "brier": model_score["brier"]},
        "axes": axes,
        "irreducible_floor": floor,
        "backlog": [a["suggested_candidate"] for a in axes if a["flagged"]],
    }


def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"residual_diagnostics error: {res['error']}"
    m = res["model"]
    lines = [
        "# Residual diagnostics — where do new level-2 signals live?",
        "",
        f"- label={res['label']} | excluded={res['excluded_counties']} | test rows={res['n_test']} "
        f"| events={res['events']} | base={res['base_rate']}",
        f"- current model: AP **{m['ap']}** | ROC-AUC {m['roc_auc']} | ECE {m['ece']} | Brier {m['brier']}",
        "",
        "## Ranked residual axes (flagged = worth a candidate)",
        "",
        "| axis | metric | effect | flagged | suggested candidate |",
        "|---|---|--:|:-:|---|",
    ]
    for a in res["axes"]:
        flag = "✅" if a["flagged"] else "·"
        lines.append(f"| {a['axis']} | {a['metric']} | {a['effect']} | {flag} | {a['suggested_candidate']} |")
    fl = res["irreducible_floor"]
    gap = round(m["brier"] - fl["bayes_brier_floor"], 5)
    lines += [
        "",
        "## Headroom (irreducible-error floor)",
        f"- model Brier **{m['brier']}** vs Bayes-Brier floor **{fl['bayes_brier_floor']}** "
        f"→ headroom **{gap:+}** (base rate {fl['base_rate']}, k={fl['k']}, n={fl['n_used']})",
        f"- {fl['note']}",
        "",
        "## Backlog (run these through signal_lab next)",
        ("- " + "\n- ".join(dict.fromkeys(res["backlog"]))) if res["backlog"]
        else "- (no axis cleared its threshold — no new candidate suggested)",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default="bacteria_results/rainfall")
    ap.add_argument("--discharge-dir", default="bacteria_results/discharge")
    ap.add_argument("--cdip-dir", default="bacteria_results/cdip_waves")
    ap.add_argument("--tide-dir", default="bacteria_results/tide_stages")
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--label", default="enterococcus", choices=["any", "enterococcus"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--include-san-diego", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[residual_diagnostics] obs not found: {obs}")
        return 2

    res = run(obs, rain_dir=sl.existing_dir(args.rain_dir), discharge_dir=sl.existing_dir(args.discharge_dir),
              cdip_dir=sl.existing_dir(args.cdip_dir), tide_dir=sl.existing_dir(args.tide_dir),
              reveal_lag_days=args.reveal_lag_days, label=args.label,
              exclude_counties=() if args.include_san_diego else ("San Diego",),
              k=args.k, n_perm=args.n_perm)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "residual_diagnostics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "residual_diagnostics.md").write_text(md, encoding="utf-8")
    sl.echo_markdown(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
