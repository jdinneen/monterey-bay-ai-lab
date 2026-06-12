#!/usr/bin/env python3
"""Signal Lab — a reusable harness for discovering, building, and HONESTLY gating
derived ("level-2") signals for the bacteria nowcast, so we can keep doing it as more
data arrives instead of writing a new bespoke 250-line experiment per idea.

Background. Two one-off experiments already proved the loop works:
`spatial_drivers_experiment.py` and `physical_spatial_covariates.py` each (a) found a
residual the model wasn't using, (b) built a leakage-safe candidate signal, and (c)
measured its lift on the honest San-Diego-excluded 2022+ holdout. The lat/lon spatial
surface earned its place (AP 0.4614 -> 0.4969, +0.0355); the neighbour-state lag alone
was a wash (+0.0). This module turns that *recipe* into a registry so the next signal is
~15 lines, not a new script, and every candidate is forced through the same gate.

What a "signal" is here. A `Candidate` contributes one or more leakage-safe feature
columns and is scored as: base driver model  vs  base + candidate, on TWO tests that must
agree before we keep it —

  1. Temporal holdout (train<=2019, calibrate 2020-21, test 2022+): the operational A/B.
  2. Leave-one-beach-out (GroupKFold over station_id): does the lift SURVIVE on beaches
     the model never trained on, or was it just per-station memorisation? A signal that
     wins (1) but loses (2) is rejected — that distinction is the whole point.

Levels. A Candidate may `requires=[...]` other candidates, whose columns are built first.
A *level-2* signal builds from raw/causal features; a *level-3* signal `requires` one or
more level-2 signals and builds on their columns. The dependency plumbing is the same, so
L3 is just "a candidate that depends on L2 candidates" — no new machinery.

Honesty. The verdict is reported straight: a wash is a wash, a temporal win that fails
LOBO is REJECT (memorisation), a regression is a regression. Nothing is materialised into
the gold layer here — this is the screen that decides whether a signal deserves to exist.

CLI:
    python -m research.bacteria.signal_lab \
        --candidates latlon,nbr_lag,nbr_and_latlon \
        --obs bacteria_results/statewide/statewide_beach_observations.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob  # noqa: E402
from research.bacteria import spatial_drivers_experiment as sde  # noqa: E402
from research.bacteria.spatial_autocorr import _load_geo, leave_one_beach_out, morans_i  # noqa: E402

# A meaningful AP lift. Below this, in either direction, we call it a wash — small AP
# wobbles are within run-to-run noise and must not be sold as a signal.
MIN_LIFT_AP = 0.005

# The discovery engine PINS its own baseline rather than inheriting ob.FEATS, which other
# workstreams mutate (concurrent oceanography work folded latitude/longitude/nbr_prev1/nbr_prev7
# straight into ob.FEATS). Inheriting that would silently zero out the very candidates we gate —
# a `latlon` signal can't show lift if the baseline already contains lat/lon. We therefore strip
# the gateable/promoted columns back out so the A/B stays meaningful and reproducible regardless
# of what gets promoted into ob.FEATS upstream.
_GATEABLE_NOT_BASELINE = {"latitude", "longitude", "nbr_prev1", "nbr_prev7"}
BASE_FEATS = [f for f in ob.FEATS if f not in _GATEABLE_NOT_BASELINE]


# ----------------------------------------------------------------- shared CLI helpers
def existing_dir(p):
    """Return ``p`` if it exists on disk, else None — used to skip absent driver dirs."""
    return p if (p and Path(p).exists()) else None


def echo_markdown(md: str) -> None:
    """Echo a report to stdout encoding-safely. Windows consoles default to cp1252 and choke
    on the reports' Δ/≥/× glyphs; we encode through stdout's own codec (the .md keeps them)."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.buffer.write(md.encode(enc, errors="replace") + b"\n")


# --------------------------------------------------------------------------- registry
@dataclass
class Candidate:
    """One derived signal. ``build`` mutates ``df`` to add ``feats`` (leakage-safe).
    ``requires`` names other candidates whose columns must exist first (this is what makes
    a level-3 signal possible: it requires level-2 candidates and reads their columns)."""
    name: str
    level: int
    feats: list[str]
    build: Callable[[pd.DataFrame, dict], pd.DataFrame]
    requires: list[str] = field(default_factory=list)
    note: str = ""


REGISTRY: dict[str, Candidate] = {}


def register(cand: Candidate) -> Candidate:
    if cand.name in REGISTRY:
        raise ValueError(f"duplicate candidate name: {cand.name}")
    REGISTRY[cand.name] = cand
    return cand


def resolved_feats(name: str) -> list[str]:
    """All feature columns a candidate brings, including those from its requirements."""
    cand = REGISTRY[name]
    out: list[str] = []
    for req in cand.requires:
        for f in resolved_feats(req):
            if f not in out:
                out.append(f)
    for f in cand.feats:
        if f not in out:
            out.append(f)
    return out


def _build_order(names: list[str]) -> list[str]:
    """Topological order so every requirement is built before the candidate needing it."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str, stack: tuple[str, ...]) -> None:
        if n in seen:
            return
        if n in stack:
            raise ValueError(f"cyclic requires: {' -> '.join(stack + (n,))}")
        for req in REGISTRY[n].requires:
            visit(req, stack + (n,))
        seen.add(n)
        order.append(n)

    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"unknown candidate '{n}' (known: {sorted(REGISTRY)})")
        visit(n, ())
    return order


# --------------------------------------------------------------- built-in candidates
def _b_latlon(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    geo = ctx["geo"].dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    if "latitude" in df.columns:
        return df
    return df.merge(geo[["station_id", "latitude", "longitude"]], on="station_id", how="left")


def _b_nbr_lag(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    if "nbr_prev1" in df.columns:
        return df
    return sde.add_knn_spatial_lag(df, ctx["geo"], k=ctx["k"], reveal_lag_days=ctx["reveal_lag_days"])


def _b_noop(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    return df


def _driver_builder(dir_key: str, feats: list[str], adder):
    """Make a builder that adds a driver's feature columns from its data dir (via the existing
    leakage-safe ob.add_*_features), or fills NaN if the dir is absent so the candidate still
    resolves (and honestly scores as a wash)."""
    def _b(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        if all(f in df.columns for f in feats):
            return df
        d = ctx.get(dir_key)
        if not d or not Path(d).exists():
            df = df.copy()
            for f in feats:
                df[f] = np.nan
            return df
        return adder(df, Path(d))
    return _b


def _b_nbr7_x_rain3(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """LEVEL-3 example: interact the level-2 neighbour-state signal (nbr_prev7) with recent
    rain (rain_3d). Hypothesis: a dirty neighbourhood matters more during a wet first flush.
    Reads columns produced by the `nbr_lag` L2 candidate and by add_rain_features."""
    if "rain_3d" not in df.columns or df["rain_3d"].isna().all():
        df = df.copy()
        df["nbr7_x_rain3"] = np.nan
        return df
    df = df.copy()
    df["nbr7_x_rain3"] = df["nbr_prev7"].astype(float) * df["rain_3d"].astype(float)
    return df


# A candidate's single representative numeric column, used to auto-build an L3 product
# interaction in propose_l3. None ⇒ no well-defined scalar (skip auto-build, still recommend).
PRIMARY_COL = {
    "rain": "rain_3d", "discharge": "discharge_log", "waves": "Hs", "tide": "water_level_m",
    "nbr_lag": "nbr_prev7", "latlon": None,
}

# An L3 is only worth composing when the two L2 residuals are genuinely independent. Below this,
# the interaction just re-expresses signal both parents already carry — propose_l3 declines to
# build it (this is what stops the auto-register branch from manufacturing dead L3 candidates).
MIN_INDEPENDENCE = 0.15

register(Candidate("latlon", 2, ["latitude", "longitude"], _b_latlon,
                   note="static spatial risk surface (proven: +0.0355 AP over a rain baseline)"))
register(Candidate("nbr_lag", 2, list(sde.SPATIAL_FEATS), _b_nbr_lag,
                   note="k-nearest-beach recent exceedance state (proven wash alone)"))
register(Candidate("nbr_and_latlon", 2, [], _b_noop, requires=["nbr_lag", "latlon"],
                   note="composite: neighbour lag + static surface"))
register(Candidate("rain", 2, list(ob.RAIN_FEATS), _driver_builder("rain_dir", list(ob.RAIN_FEATS), ob.add_rain_features),
                   note="rainfall driver (known strong driver — expect KEEP)"))
register(Candidate("discharge", 2, list(ob.DISCHARGE_FEATS),
                   _driver_builder("discharge_dir", list(ob.DISCHARGE_FEATS), ob.add_discharge_features),
                   note="river first-flush discharge (driver-null regression test — expect WASH)"))
register(Candidate("waves", 2, list(ob.WAVE_FEATS),
                   _driver_builder("cdip_dir", list(ob.WAVE_FEATS), ob.add_wave_features),
                   note="CDIP wave height/period/dir (driver-null regression test — expect WASH)"))
register(Candidate("tide", 2, list(ob.TIDE_STAGES_FEATS),
                   _driver_builder("tide_dir", list(ob.TIDE_STAGES_FEATS), ob.add_tide_stage_features),
                   note="CO-OPS tide water level (driver-null regression test — expect WASH)"))
register(Candidate("nbr7_x_rain3", 3, ["nbr7_x_rain3"], _b_nbr7_x_rain3, requires=["nbr_lag", "rain"],
                   note="L3 hypothesis: neighbourhood x first-flush rain interaction"))


# ------------------------------------------------------------------------ evaluation
def _fit_eval(tr, va, te, feats):
    """Calibrated probabilities on ``te`` from a HGB trained on ``tr``, isotonic-fit on ``va``."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    # Drop features constant in THIS training set (sklearn>=1.9 HGBT binning errors on a
    # single-distinct-value feature). A candidate whose columns are all-NaN/constant in train
    # (e.g. a source with no pre-2019 coverage) thus collapses to the base -> reported as a
    # WASH, not a crash. Mirrors operational_benchmark's global guard.
    feats = [c for c in feats if tr[c].notna().sum() >= 2 and tr[c].nunique(dropna=True) > 1]
    clf = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1, random_state=42)
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    raw = clf.predict_proba(te[feats].astype(float))[:, 1]
    if va["exceed"].nunique() > 1:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(va[feats].astype(float))[:, 1], va["exceed"].to_numpy())
        return iso.predict(raw)
    return raw


def _residual_morans(te: pd.DataFrame, pcol: str, geo: pd.DataFrame, k: int, n_perm: int):
    resid = (te.assign(r=te["exceed"] - te[pcol]).groupby("station_id")["r"].mean()
               .rename("resid").reset_index())
    j = resid.merge(geo, on="station_id", how="inner")
    if len(j) < 3:
        return None
    return morans_i(j["resid"].to_numpy(), j[["latitude", "longitude"]].to_numpy(),
                    k=k, n_perm=n_perm).get("morans_i")


def _verdict(delta_ap: float, lobo_delta, did_lobo: bool) -> tuple[str, str]:
    if delta_ap < -MIN_LIFT_AP:
        return "REJECT", f"regresses temporal AP ({delta_ap:+.4f})"
    if delta_ap < MIN_LIFT_AP:
        return "WASH", f"temporal AP move {delta_ap:+.4f} < {MIN_LIFT_AP} (noise)"
    if did_lobo and lobo_delta is not None and lobo_delta < 0:
        return "REJECT", (f"temporal AP +{delta_ap:.4f} but LOBO {lobo_delta:+.4f} — "
                          "lift is per-station memorisation, fails on unseen beaches")
    return "KEEP", (f"temporal AP +{delta_ap:.4f}"
                    + (f", survives LOBO ({lobo_delta:+.4f})" if did_lobo and lobo_delta is not None else ""))


def _prepare_base(obs_path: Path, *, base_drivers=("rain",), rain_dir=None, reveal_lag_days=2,
                  label="enterococcus", exclude_counties=("San Diego",)):
    """Load obs and build the HONEST baseline: pure causal features PLUS the already-established
    drivers (rainfall by default). Testing a new signal over a rain-less baseline is misleading —
    rain-correlated drivers (discharge/waves/tide) would proxy for rainfall and falsely 'win';
    latlon's proven +0.0355 was measured over FEATS+RAIN, so that is the baseline we reproduce."""
    df = ob.load_station_days(obs_path, label=label)
    if exclude_counties:
        df = df[~df["county"].isin(set(exclude_counties))].copy()
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    base_feats = list(BASE_FEATS)  # pinned baseline, NOT ob.FEATS (see BASE_FEATS note)
    if "rain" in base_drivers and rain_dir and Path(rain_dir).exists():
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            base_feats += list(ob.RAIN_FEATS)
    return df, base_feats


def evaluate(obs_path: Path, candidates: list[str], *, base_drivers=("rain",), rain_dir=None,
             discharge_dir=None, cdip_dir=None, tide_dir=None, reveal_lag_days: int = 2,
             label: str = "enterococcus", exclude_counties=("San Diego",), k: int = 8,
             n_perm: int = 199, do_lobo: bool = True) -> dict:
    # Baseline = pure causal features + established drivers (rain). New drivers and spatial
    # signals are gated as candidates over THAT, so a rain-correlated driver can't false-win.
    df, base_feats = _prepare_base(obs_path, base_drivers=base_drivers, rain_dir=rain_dir,
                                   reveal_lag_days=reveal_lag_days, label=label,
                                   exclude_counties=exclude_counties)

    geo = _load_geo()
    if geo is None:
        return {"error": "station_geo.parquet not found"}
    geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    ctx = {"geo": geo, "k": k, "reveal_lag_days": reveal_lag_days, "rain_dir": rain_dir,
           "discharge_dir": discharge_dir, "cdip_dir": cdip_dir, "tide_dir": tide_dir}

    # build every needed column once, in dependency order
    for name in _build_order(candidates):
        df = REGISTRY[name].build(df, ctx)

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    y = te["exceed"].to_numpy()
    base_rate = float(tr["exceed"].mean())

    # baseline once
    te["p_base"] = _fit_eval(tr, va, te, base_feats)
    base_score = ob.score(y, te["p_base"].to_numpy(), base_rate)
    base_ap = base_score["ap"]
    base_mi = _residual_morans(te, "p_base", geo_t, k, n_perm)
    lobo_base_ap = None
    if do_lobo:
        lb = leave_one_beach_out(df, base_feats)
        lobo_base_ap = lb.get("models", {}).get("model_calibrated", {}).get("ap")

    results = []
    for name in candidates:
        feats = base_feats + [f for f in resolved_feats(name) if f not in base_feats]
        pcol = f"p_{name}"
        te[pcol] = _fit_eval(tr, va, te, feats)
        sc = ob.score(y, te[pcol].to_numpy(), base_rate)
        delta_ap = round(sc["ap"] - base_ap, 4)
        mi = _residual_morans(te, pcol, geo_t, k, n_perm)
        lobo_delta = None
        if do_lobo and lobo_base_ap is not None:
            lc = leave_one_beach_out(df, feats)
            lobo_cand_ap = lc.get("models", {}).get("model_calibrated", {}).get("ap")
            if lobo_cand_ap is not None:
                lobo_delta = round(lobo_cand_ap - lobo_base_ap, 4)
        verdict, reason = _verdict(delta_ap, lobo_delta, do_lobo)
        results.append({
            "name": name, "level": REGISTRY[name].level, "note": REGISTRY[name].note,
            "added_feats": [f for f in resolved_feats(name) if f not in base_feats],
            "ap": sc["ap"], "delta_ap": delta_ap, "ece": sc["ece"],
            "residual_morans_i": mi, "morans_delta": (round(mi - base_mi, 4)
                                                      if (mi is not None and base_mi is not None) else None),
            "lobo_delta_ap": lobo_delta, "verdict": verdict, "reason": reason,
        })

    results.sort(key=lambda r: (r["verdict"] != "KEEP", -(r["delta_ap"] or 0)))
    return {
        "label": label, "excluded_counties": list(exclude_counties or []),
        "reveal_lag_days": reveal_lag_days, "k_neighbors": k, "min_lift_ap": MIN_LIFT_AP,
        "n_test": int(len(te)), "events": int(y.sum()), "base_rate": round(float(y.mean()), 4),
        "baseline": {"feats": "ob.FEATS + rainfall (established operational model)"
                     if any(f.startswith("rain") for f in base_feats) else "ob.FEATS (pure causal)",
                     "ap": base_ap, "ece": base_score["ece"], "residual_morans_i": base_mi,
                     "lobo_ap": lobo_base_ap},
        "candidates": results,
    }


def _rank_pairs(resid: dict) -> list[dict]:
    """Rank candidate pairs by residual INDEPENDENCE (1 - |corr|). Two signals whose holdout
    residuals are uncorrelated capture different structure → their interaction can carry signal;
    a highly-correlated pair already overlaps and an L3 between them is dead on arrival."""
    names = list(resid)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            r = float(np.corrcoef(resid[a], resid[b])[0, 1])
            if not np.isfinite(r):
                r = 0.0
            pairs.append({"pair": [a, b], "residual_corr": round(r, 4),
                          "independence": round(1 - abs(r), 4)})
    pairs.sort(key=lambda p: -p["independence"])
    return pairs


def propose_l3(obs_path: Path, kept_l2: list[str], *, base_drivers=("rain",), rain_dir=None,
               discharge_dir=None, cdip_dir=None, tide_dir=None, reveal_lag_days: int = 2,
               label: str = "enterococcus", exclude_counties=("San Diego",), k: int = 8) -> dict:
    """Answer 'how much L2 do we need for L3?' concretely: you need >=2 KEPT L2 signals whose
    holdout RESIDUALS are independent — interacting two signals that already capture the same
    structure (high residual correlation) can't add anything. We fit each kept L2 signal's
    model, correlate their per-row residuals, rank pairs by independence (low |corr|), and for
    the most independent pair that has a well-defined scalar product, register an L3 interaction
    Candidate so the orchestrator can gate it like any other signal.

    Returns {pairs: [...ranked...], recommended_pair, registered: <l3 name or None>}.
    Lightweight (temporal residuals only; no LOBO) — the gating of the L3 itself uses evaluate().
    """
    if len(kept_l2) < 2:
        return {"pairs": [], "recommended_pair": None, "registered": None,
                "note": "need >=2 KEPT L2 signals to compose an L3"}

    df, base_feats = _prepare_base(obs_path, base_drivers=base_drivers, rain_dir=rain_dir,
                                   reveal_lag_days=reveal_lag_days, label=label,
                                   exclude_counties=exclude_counties)
    geo = _load_geo()
    if geo is None:
        return {"error": "station_geo.parquet not found"}
    ctx = {"geo": geo, "k": k, "reveal_lag_days": reveal_lag_days, "rain_dir": rain_dir,
           "discharge_dir": discharge_dir, "cdip_dir": cdip_dir, "tide_dir": tide_dir}
    for name in _build_order(kept_l2):
        df = REGISTRY[name].build(df, ctx)

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()
    resid = {}
    for name in kept_l2:
        feats = base_feats + [f for f in resolved_feats(name) if f not in base_feats]
        p = _fit_eval(tr, va, te, feats)
        resid[name] = te["exceed"].to_numpy() - p

    pairs = _rank_pairs(resid)

    registered = None
    for p in pairs:  # register the most-independent BUILDABLE pair — but only if it is actually
        a, b = p["pair"]               # independent; a redundant pair makes a dead L3 (Vapor).
        if p["independence"] < MIN_INDEPENDENCE:
            break  # pairs are independence-sorted, so nothing below here qualifies either
        ca, cb = PRIMARY_COL.get(a), PRIMARY_COL.get(b)
        if ca and cb:
            name = f"l3_{a}_x_{b}"
            if name not in REGISTRY:
                def _mk(ca=ca, cb=cb):
                    def _b(d: pd.DataFrame, ctx: dict, _ca=ca, _cb=cb) -> pd.DataFrame:
                        d = d.copy()
                        d[f"{_ca}_x_{_cb}"] = (d[_ca].astype(float).replace(-999.0, np.nan)
                                               * d[_cb].astype(float).replace(-999.0, np.nan))
                        return d
                    return _b
                register(Candidate(name, 3, [f"{ca}_x_{cb}"], _mk(), requires=[a, b],
                                   note=f"L3 auto-proposed: most-independent KEPT pair ({a} x {b})"))
            registered = name
            p["registered_as"] = name
            break

    top_indep = pairs[0]["independence"] if pairs else 0.0
    reason = ("registered " + registered if registered
              else f"no L3 worth composing: best pair independence {top_indep} < {MIN_INDEPENDENCE} "
                   "(KEPT L2 signals are redundant)" if top_indep < MIN_INDEPENDENCE
              else "most-independent pair has no scalar product to auto-build")
    return {"pairs": pairs, "recommended_pair": (pairs[0]["pair"] if pairs else None),
            "registered": registered, "min_independence": MIN_INDEPENDENCE, "reason": reason}


def to_markdown(res: dict) -> str:
    if "error" in res:
        return f"signal_lab error: {res['error']}"
    b = res["baseline"]
    lines = [
        "# Signal Lab — derived-signal discovery gate",
        "",
        f"- label={res['label']} | excluded={res['excluded_counties']} | lag={res['reveal_lag_days']}d "
        f"| k={res['k_neighbors']} | test rows={res['n_test']} | events={res['events']} | base={res['base_rate']}",
        f"- baseline ({b['feats']}): AP **{b['ap']}** | ECE {b['ece']} | resid Moran's I {b['residual_morans_i']} "
        f"| LOBO AP {b['lobo_ap']}",
        f"- keep threshold: temporal ΔAP ≥ {res['min_lift_ap']} AND survives leave-one-beach-out",
        "",
        "| candidate | lvl | ΔAP (temporal) | ΔAP (LOBO) | ECE | resid Moran's I (Δ) | verdict |",
        "|---|:-:|--:|--:|--:|--:|:--|",
    ]
    for r in res["candidates"]:
        mi = r["residual_morans_i"]
        mid = r["morans_delta"]
        mi_s = "-" if mi is None else f"{mi} ({mid:+})" if mid is not None else f"{mi}"
        lobo = "-" if r["lobo_delta_ap"] is None else f"{r['lobo_delta_ap']:+}"
        lines.append(f"| {r['name']} | {r['level']} | {r['delta_ap']:+} | {lobo} | {r['ece']} | {mi_s} | "
                     f"**{r['verdict']}** |")
    lines.append("")
    for r in res["candidates"]:
        lines.append(f"- **{r['name']}** — {r['verdict']}: {r['reason']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--candidates", default=",".join(REGISTRY),
                    help=f"comma-separated; known: {','.join(REGISTRY)}")
    ap.add_argument("--rain-dir", default="bacteria_results/rainfall")
    ap.add_argument("--discharge-dir", default="bacteria_results/discharge")
    ap.add_argument("--cdip-dir", default="bacteria_results/cdip_waves")
    ap.add_argument("--tide-dir", default="bacteria_results/tide_stages")
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--label", default="enterococcus", choices=["any", "enterococcus"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--include-san-diego", action="store_true")
    ap.add_argument("--no-lobo", action="store_true", help="skip leave-one-beach-out (faster, less honest)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[signal_lab] obs not found: {obs}")
        return 2
    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    res = evaluate(obs, cands, rain_dir=existing_dir(args.rain_dir),
                   discharge_dir=existing_dir(args.discharge_dir), cdip_dir=existing_dir(args.cdip_dir),
                   tide_dir=existing_dir(args.tide_dir), reveal_lag_days=args.reveal_lag_days,
                   label=args.label, exclude_counties=() if args.include_san_diego else ("San Diego",),
                   k=args.k, n_perm=args.n_perm, do_lobo=not args.no_lobo)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signal_lab.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "signal_lab.md").write_text(md, encoding="utf-8")
    echo_markdown(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
