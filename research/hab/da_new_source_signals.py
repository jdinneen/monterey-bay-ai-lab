#!/usr/bin/env python3
"""Do the NEW buildout sources improve the domoic-acid (DA) forecast? — gated secondary-signal test.

The data buildout added ~20 new curated sources. This asks the one honest question the lab cares
about: does injecting any of them as a feature beat the lean DA+precursor incumbent
(`research/hab/da_forecast.py`) on the SAME leakage-safe split — gated, washes reported as washes.

It does NOT reimplement anything. It reuses:
  * `hab_sota_sweep.load_panel` + `add_causal_features`  — the CalHABMAP pier panel with a
    `feature_time` = the PRIOR visit (so every feature is known one visit ahead; strict causality).
  * `da_forecast.fit_predict` / `_scores` / `HEADLINE_FEATS` — the exact incumbent model + metric.
  * `signal_lab._verdict` / `MIN_LIFT_AP` — the canonical KEEP/WASH/REJECT gate (temporal
    AP lift >= 0.005 AND survives leave-one-station-out; else WASH/REJECT).

Honest scoping (verified in planning, see reports/hab/da_new_source_signals.md):
  * Only sources with BOTH training (<=2018) and test (>2020) coverage can be evaluated. CDIP
    SST/wave (2021+), WCOFS (2022-12+) and ROMS (test ends 2022) are EXCLUDED — including them
    would fake a "wash" that is really "never trained". They are listed, not tested.
  * Temperature/SST is already covered by mur_sst/oa_temp in the existing sweep, so per-source we
    keep only the mechanistically-NEW levers (upwelling wind, river discharge/turbidity, solar,
    precip, pressure) and drop redundant temp columns.

Join: for each pier, the NEAREST source cell/station (haversine, within a max radius; NaN — never
imputed — if none), daily-aggregated, merged as-of `feature_time - 1 day` (backward, tight
tolerance). A column with <30% training coverage after the join is dropped (a wash must not be a
no-data artifact).

Usage:  python research/hab/da_new_source_signals.py [--output-dir reports/hab]
Outputs (reports/hab/): da_new_source_signals.json, da_new_source_signals.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # research/hab on path
import da_forecast as F          # noqa: E402  (reuse fit_predict/_scores/HEADLINE_FEATS/splits)
import hab_sota_sweep as S       # noqa: E402  (reuse load_panel/add_causal_features panel)
sys.path.insert(0, str(ROOT))
from research.bacteria.signal_lab import _verdict, MIN_LIFT_AP  # noqa: E402  (the canonical gate)

OUTDIR = ROOT / "reports" / "hab"
CURATED = ROOT / "data" / "external_curated"
CA_LAT, CA_LON = (32.0, 42.0), (-125.0, -117.0)   # California coast bbox, prefilter every source
AVAIL_LAG_DAYS = 1                                 # external signal must predate the prior visit
TOL_DAYS = 10                                      # as-of fill horizon (45d was too loose / stale)
MIN_TRAIN_COVERAGE = 0.30                          # drop a feature column below this in train
LOSO_MIN_EVENTS = 10

# Each NEW source: where, how to key it spatially, and the mechanistically-distinct variables to
# keep (temp/SST dropped as redundant with mur_sst/oa_temp). long=(param_col,value_col)+keep, or
# wide=[cols]. All have verified train(<=2018) AND test(>2020) coverage.
SOURCES = [
    {"name": "usgs_runoff", "dir": "usgs_dv_statewide", "lat": "latitude", "lon": "longitude",
     "time": "date", "param": "parameter", "value": "result_value",
     "keep": ["discharge_cfs", "turbidity_fnu"], "max_km": 75,
     "why": "river first-flush discharge + turbidity (runoff/nutrient pulse)"},
    {"name": "gridmet", "dir": "gridmet_daily", "lat": "grid_lat", "lon": "grid_lon",
     "time": "date", "param": "parameter", "value": "value",
     "keep": ["wind_ms", "vpd_kpa", "precip_mm", "srad_wm2"], "max_km": 60,
     "why": "gridMET 4km met: wind (upwelling), vapor-pressure deficit, precip, solar"},
    {"name": "nasa_power", "dir": "nasa_power_daily", "lat": "grid_lat", "lon": "grid_lon",
     "time": "date", "param": "parameter", "value": "value",
     "keep": ["WS10M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"], "max_km": 80,
     "why": "NASA POWER satellite met: wind, precip, surface shortwave (light)"},
    {"name": "era5_openmeteo", "dir": "open_meteo_archive", "lat": "grid_lat", "lon": "grid_lon",
     "time": "time", "param": "parameter", "value": "value",
     "keep": ["wind_speed_10m", "precipitation", "shortwave_radiation", "surface_pressure"],
     "max_km": 60, "why": "ERA5 reanalysis (hourly->daily): wind, precip, shortwave, pressure"},
    {"name": "ncei_isd", "dir": "ncei_isd_hourly", "lat": "latitude", "lon": "longitude",
     "time": "time", "wide": ["wind_speed_ms", "slp_hpa"], "max_km": 75,
     "why": "in-situ coastal station met (hourly->daily): wind + sea-level pressure"},
]

# Excluded up front (verified spans) — reported, NOT tested, so a real null is never confused with
# an un-trainable source.
EXCLUDED = [
    {"name": "cdip_sst_network", "span": "2021-2026", "reason": "no training coverage (<=2018)"},
    {"name": "cdip_wave_network", "span": "2021-2026", "reason": "no training coverage (<=2018)"},
    {"name": "wcofs_circulation", "span": "2022-12 to 2026", "reason": "no training coverage (<=2018)"},
    {"name": "roms_circulation", "span": "2015 to 2022-12", "reason": "test coverage ends 2022 (non-representative test AP)"},
]


def _haversine_km(lat1, lon1, lat2, lon2):
    """Scalar pier (lat1,lon1) to array of cells (lat2,lon2). Correct at CA's 33–41N lat span."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _load_daily(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (per-cell daily-mean wide frame with clat/clon/day + prefixed value cols, feat_cols)."""
    pq = CURATED / cfg["dir"] / f"{cfg['dir']}.parquet"
    df = pd.read_parquet(pq)
    lat, lon, t = cfg["lat"], cfg["lon"], cfg["time"]
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df = df[df[lat].between(*CA_LAT) & df[lon].between(*CA_LON)].copy()
    df["day"] = pd.to_datetime(df[t], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=[lat, lon, "day"])

    if "param" in cfg:
        df = df[df[cfg["param"]].isin(cfg["keep"])].copy()
        df[cfg["value"]] = pd.to_numeric(df[cfg["value"]], errors="coerce")
        wide = df.pivot_table(index=[lat, lon, "day"], columns=cfg["param"],
                              values=cfg["value"], aggfunc="mean").reset_index()
        valcols = [c for c in cfg["keep"] if c in wide.columns]
    else:
        for c in cfg["wide"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        wide = df.groupby([lat, lon, "day"])[cfg["wide"]].mean().reset_index()
        valcols = [c for c in cfg["wide"] if c in wide.columns]

    wide = wide.rename(columns={lat: "clat", lon: "clon"})
    pref = cfg["name"]
    wide = wide.rename(columns={c: f"{pref}__{c}" for c in valcols})
    return wide, [f"{pref}__{c}" for c in valcols]


def attach_source(cfg: dict, panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join cfg's nearest-cell daily series onto the panel, leakage-safe (as-of feature_time-1d)."""
    daily, feat_cols = _load_daily(cfg)
    panel = panel.copy()
    for c in feat_cols:
        panel[c] = np.nan
    if daily.empty:
        return panel, feat_cols
    cells = daily[["clat", "clon"]].drop_duplicates().reset_index(drop=True)
    for st, sub in panel.groupby("station", sort=False):
        plat, plon = float(sub["latitude"].iloc[0]), float(sub["longitude"].iloc[0])
        dist = _haversine_km(plat, plon, cells["clat"].to_numpy(), cells["clon"].to_numpy())
        j = int(np.argmin(dist))
        if dist[j] > cfg["max_km"]:
            continue   # no source cell within radius -> leave NaN (never impute across stations)
        nlat, nlon = cells.loc[j, "clat"], cells.loc[j, "clon"]
        series = (daily[(daily["clat"] == nlat) & (daily["clon"] == nlon)]
                  [["day", *feat_cols]].dropna(subset=["day"]).sort_values("day"))
        left = sub.sort_values("available_signal_date")
        merged = pd.merge_asof(left[["available_signal_date"]], series,
                               left_on="available_signal_date", right_on="day",
                               direction="backward", tolerance=pd.Timedelta(days=TOL_DAYS))
        for c in feat_cols:
            panel.loc[left.index, c] = merged[c].to_numpy()
    return panel, feat_cols


def _pooled_loso_delta(panel: pd.DataFrame, lean: list[str], feats: list[str]) -> float | None:
    """Leave-one-station-out: pooled AP(lean+source) - pooled AP(lean) over held-out stations."""
    ev = panel.groupby("station")["exceed"].sum()
    ys, p_lean, p_src = [], [], []
    for st in ev[ev >= LOSO_MIN_EVENTS].index:
        tr_all = panel[panel["station"] != st]
        tr = tr_all[tr_all["year"] <= F.TRAIN_END]
        va = tr_all[(tr_all["year"] > F.TRAIN_END) & (tr_all["year"] <= F.VALID_END)]
        te = panel[panel["station"] == st]
        if te["exceed"].sum() == 0 or tr["exceed"].nunique() < 2:
            continue
        ys.append(te["exceed"].to_numpy())
        p_lean.append(F.fit_predict(tr, va, te, lean))
        p_src.append(F.fit_predict(tr, va, te, feats))
    if not ys:
        return None
    y = np.concatenate(ys)
    return round(float(average_precision_score(y, np.concatenate(p_src))
                       - average_precision_score(y, np.concatenate(p_lean))), 4)


def run(output_dir: Path | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir else OUTDIR
    panel = S.add_causal_features(S.load_panel())
    panel["available_signal_date"] = (
        panel["feature_time"] - pd.Timedelta(days=AVAIL_LAG_DAYS)).dt.normalize()

    src_feats: dict[str, list[str]] = {}
    for cfg in SOURCES:
        panel, fcols = attach_source(cfg, panel)
        src_feats[cfg["name"]] = fcols

    tr = panel[panel["year"] <= F.TRAIN_END]
    va = panel[(panel["year"] > F.TRAIN_END) & (panel["year"] <= F.VALID_END)]
    te = panel[panel["year"] > F.VALID_END].copy()
    base = float(tr["exceed"].mean())
    yte = te["exceed"].to_numpy()
    lean = list(F.HEADLINE_FEATS)
    lean_ap = F._scores(yte, F.fit_predict(tr, va, te, lean), base)["ap"]

    sources = {}
    for cfg in SOURCES:
        name = cfg["name"]
        fcols = src_feats[name]
        coverage = {c: round(float(tr[c].notna().mean()), 3) for c in fcols}
        kept = [c for c in fcols if tr[c].notna().mean() >= MIN_TRAIN_COVERAGE]
        if not kept:
            sources[name] = {"why": cfg["why"], "verdict": "EXCLUDED_COVERAGE",
                             "reason": f"all columns <{MIN_TRAIN_COVERAGE:.0%} train coverage",
                             "train_coverage": coverage, "kept_features": []}
            continue
        feats = lean + kept
        ap = F._scores(yte, F.fit_predict(tr, va, te, feats), base)["ap"]
        d_ap = round((ap or 0) - (lean_ap or 0), 4)
        loso_delta = _pooled_loso_delta(panel, lean, feats)
        verdict, reason = _verdict(d_ap, loso_delta, loso_delta is not None)
        sources[name] = {
            "why": cfg["why"], "kept_features": kept, "train_coverage": coverage,
            "ap_lean_plus_source": ap, "delta_ap_vs_lean": d_ap, "loso_delta_ap": loso_delta,
            "verdict": verdict, "reason": reason,
        }

    out = {
        "target": "next-visit pDA >= 0.5 ng/mL (=500 ng/L) -- does any NEW buildout source beat lean?",
        "incumbent": "da_forecast DA+precursor (lean)",
        "gate": f"signal_lab: temporal dAP >= {MIN_LIFT_AP} AND LOSO dAP >= 0 (else WASH/REJECT)",
        "splits": {"train_end": F.TRAIN_END, "valid_end": F.VALID_END,
                   "n_train": int(len(tr)), "n_test": int(len(te)),
                   "test_events": int(yte.sum()), "test_base_rate": round(float(yte.mean()), 4)},
        "join": {"availability_lag_days": AVAIL_LAG_DAYS, "as_of_tolerance_days": TOL_DAYS,
                 "min_train_coverage": MIN_TRAIN_COVERAGE, "nearest_cell": "haversine within max_km"},
        "lean_test_ap": lean_ap,
        "sources_tested": sources,
        "sources_excluded": EXCLUDED,
        "headline": {
            "n_tested": len(sources),
            "kept": [n for n, r in sources.items() if r["verdict"] == "KEEP"],
            "washed": [n for n, r in sources.items() if r["verdict"] == "WASH"],
            "rejected": [n for n, r in sources.items() if r["verdict"] == "REJECT"],
            "excluded_no_coverage": [e["name"] for e in EXCLUDED]
                                    + [n for n, r in sources.items() if r["verdict"] == "EXCLUDED_COVERAGE"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "da_new_source_signals.json").write_text(json.dumps(out, indent=2, default=str),
                                                        encoding="utf-8")
    (out_dir / "da_new_source_signals.md").write_text(_render_md(out), encoding="utf-8")
    return out


def _render_md(out: dict) -> str:
    h = out["headline"]
    L = [f"# New-source signal test on the DA forecast", "",
         f"Target: {out['target']}", "",
         f"Incumbent (baseline to beat): **{out['incumbent']}**, lean test AP = {out['lean_test_ap']}.",
         f"Gate: {out['gate']}.",
         f"Test split: train<={out['splits']['train_end']} / calib-{out['splits']['valid_end']} / "
         f"test>{out['splits']['valid_end']} (n_test={out['splits']['n_test']}, "
         f"{out['splits']['test_events']} events, base {out['splits']['test_base_rate']}).",
         f"Join: nearest-cell haversine, as-of (prior visit - {out['join']['availability_lag_days']}d), "
         f"tolerance {out['join']['as_of_tolerance_days']}d, drop <"
         f"{int(out['join']['min_train_coverage']*100)}% train-coverage columns.", "",
         "## Verdicts (per new source)", "",
         "| source | new signals | dAP vs lean | LOSO dAP | verdict | why it was tested |",
         "|---|---|--:|--:|:-:|---|"]
    for name, r in out["sources_tested"].items():
        feats = ", ".join(c.split("__", 1)[-1] for c in r.get("kept_features") or []) or "(none)"
        L.append(f"| `{name}` | {feats} | {r.get('delta_ap_vs_lean','-')} | "
                 f"{r.get('loso_delta_ap','-')} | **{r['verdict']}** | {r['why']} |")
    L += ["", "## Excluded up front (insufficient train/test coverage -- NOT tested)", "",
          "| source | span | reason |", "|---|---|---|"]
    for e in out["sources_excluded"]:
        L.append(f"| `{e['name']}` | {e['span']} | {e['reason']} |")
    L += ["",
          f"**Result:** KEEP={h['kept'] or 'none'} | WASH={h['washed'] or 'none'} | "
          f"REJECT={h['rejected'] or 'none'}.",
          "",
          "A WASH/REJECT is the honest deliverable: it confirms the documented driver-null pattern "
          "(physical/met/runoff drivers do not beat the lean DA-history + Pseudo-nitzschia model) "
          "now extends to the new buildout sources. A KEEP would be flagged for cross-review and is "
          "NOT auto-wired into da_forecast (promotion stays gated)."]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTDIR),
                    help="where to write da_new_source_signals.json/.md (default reports/hab/).")
    args = ap.parse_args(argv)
    out = run(Path(args.output_dir))
    print(json.dumps(out["headline"], indent=2))
    print(f"\nwrote {Path(args.output_dir) / 'da_new_source_signals.json'}, .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
