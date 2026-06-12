#!/usr/bin/env python3
"""Gate the NEW ingested data sources as BACTERIA signals — the symmetric pass to the
HAB-target injection (claude-fetch-A's lane).

Each of the 20+ new curated sources is turned into a leakage-safe `Candidate` and put
through the *existing* Signal-Lab gate (`research.bacteria.signal_lab.evaluate`): scored
base (FEATS + rain, San-Diego-excluded) vs base + source, and KEPT only if it lifts AP by
>= 0.005 on the 2022+ temporal holdout AND survives leave-one-beach-out (so the lift is
real generalisation, not per-station memorisation). A wash is reported as a wash.

This module does NOT edit signal_lab.py — it registers candidates into its REGISTRY at
import time and calls its `evaluate()`. The only new code is the leakage-safe causal join.

Causal join (strictly past-only): for each beach, the NEAREST source location (or a
statewide daily mean when the source has no per-row coordinates); the source's daily
series is shifted (prev = shift(1), roll7 = shift(1).rolling(7)) and merged as-of the
beach sample date minus the lab reveal lag — so only data available before the sample is
ever used.

    python -m research.bacteria.new_source_signal_gate            # all configured sources
    python -m research.bacteria.new_source_signal_gate --sources gridmet_precip,sst_buoy
    python -m research.bacteria.new_source_signal_gate --no-lobo  # fast screen (no KEEP without LOBO)
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.bacteria import signal_lab as sl  # noqa: E402

CURATED = _REPO / "data" / "external_curated"
OUT = _REPO / "reports" / "bacteria" / "new_source_signal_gate"

# Each config maps a new curated source -> one representative, bacteria-relevant signal.
# loc=("lat","lon") => nearest-location join; loc=None => statewide daily-mean fallback.
# param=(col,val) selects one row-type from a long (parameter/value) table.
SOURCE_CONFIGS = {
    "gridmet_precip":  dict(src="gridmet_daily",     time="date", value="value", loc=("grid_lat", "grid_lon"), param=("parameter", "precip_mm")),
    "nasa_t2m":        dict(src="nasa_power_daily",  time="date", value="value", loc=("grid_lat", "grid_lon"), param=("parameter", "T2M")),
    "openmeteo_temp":  dict(src="open_meteo_archive",time="time", value="value", loc=("grid_lat", "grid_lon"), param=("parameter", "temperature_2m")),
    "usgs_turbidity":  dict(src="usgs_dv_statewide", time="date", value="result_value", loc=("latitude", "longitude"), param=("parameter", "turbidity_fnu")),
    "usgs_watertemp":  dict(src="usgs_dv_statewide", time="date", value="result_value", loc=("latitude", "longitude"), param=("parameter", "water_temp_c")),
    "sst_buoy":        dict(src="cdip_sst_network",  time="time", value="sst_c", loc=("latitude", "longitude")),
    "wave_network":    dict(src="cdip_wave_network", time="time", value="waveHs", loc=("latitude", "longitude")),
    "buoy_wind":       dict(src="ndbc_cwind",        time="time", value="WSPD", loc=None),
    "isd_airtemp":     dict(src="ncei_isd_hourly",   time="time", value="air_temp_c", loc=("latitude", "longitude")),
    "ghcnd_precip":    dict(src="ghcnd",             time="date", value="value", loc=None, param=("element", "PRCP")),
}


def _read_src(src: str) -> pd.DataFrame | None:
    p = CURATED / src / f"{src}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _nan_feats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{prefix}_prev"] = np.nan
    df[f"{prefix}_roll7"] = np.nan
    return df


def _causal_join(df: pd.DataFrame, ctx: dict, cfg: dict, prefix: str) -> pd.DataFrame:
    """Add `<prefix>_prev` + `<prefix>_roll7` — nearest-source-location daily series, shifted
    and merged as-of (sample_date - reveal_lag). Strictly past-only. Falls back to a statewide
    daily mean when the source has no per-row coordinates."""
    if f"{prefix}_prev" in df.columns:
        return df
    src = _read_src(cfg["src"])
    if src is None or src.empty:
        return _nan_feats(df, prefix)
    if cfg.get("param"):
        pc, pv = cfg["param"]
        if pc in src.columns:
            src = src[src[pc].astype(str) == pv]
    src = src.copy()
    src["__v"] = pd.to_numeric(src[cfg["value"]], errors="coerce")
    src["__d"] = pd.to_datetime(src[cfg["time"]], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    src = src.dropna(subset=["__v", "__d"])
    if src.empty:
        return _nan_feats(df, prefix)

    geo = ctx["geo"].dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    loc = cfg.get("loc")
    has_loc = bool(loc) and loc[0] in src.columns and loc[1] in src.columns
    reveal = int(ctx.get("reveal_lag_days", 2))

    if has_loc:
        from scipy.spatial import cKDTree
        src["__la"] = pd.to_numeric(src[loc[0]], errors="coerce")
        src["__lo"] = pd.to_numeric(src[loc[1]], errors="coerce")
        src = src.dropna(subset=["__la", "__lo"])
        locs = src[["__la", "__lo"]].drop_duplicates().reset_index(drop=True)
        locs["loc"] = locs.index
        tree = cKDTree(locs[["__la", "__lo"]].to_numpy())
        _, bidx = tree.query(geo[["latitude", "longitude"]].to_numpy())
        beach_loc = geo[["station_id"]].assign(loc=bidx)
        src = src.merge(locs, on=["__la", "__lo"], how="left")
        daily = src.groupby(["loc", "__d"])["__v"].mean().reset_index().sort_values(["loc", "__d"])
        # value AT the matched (most-recent) day + trailing 7-obs mean PER location. Causality is
        # enforced by the as-of merge below (only days <= sample_date - reveal_lag are matched),
        # so no extra shift is needed (an extra shift just throws away the freshest legal value).
        daily["v_prev"] = daily["__v"]
        daily["v_roll7"] = daily.groupby("loc")["__v"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    else:
        daily = src.groupby("__d")["__v"].mean().reset_index().sort_values("__d")
        daily["loc"] = 0
        daily["v_prev"] = daily["__v"]
        daily["v_roll7"] = daily["__v"].rolling(7, min_periods=1).mean()
        beach_loc = geo[["station_id"]].assign(loc=0)

    d = df.merge(beach_loc, on="station_id", how="left")
    d["__asof"] = (pd.to_datetime(d["sample_date"]) - pd.Timedelta(days=reveal)).dt.normalize()
    d["__rid"] = np.arange(len(d))
    d = d.dropna(subset=["loc"]).copy()
    d["loc"] = d["loc"].astype(int)
    merged = pd.merge_asof(
        d.sort_values("__asof"), daily[["loc", "__d", "v_prev", "v_roll7"]].sort_values("__d"),
        left_on="__asof", right_on="__d", by="loc", direction="backward",
        tolerance=pd.Timedelta(days=21))
    out = df.copy()
    pv = pd.Series(merged["v_prev"].to_numpy(), index=merged["__rid"].to_numpy())
    rv = pd.Series(merged["v_roll7"].to_numpy(), index=merged["__rid"].to_numpy())
    out[f"{prefix}_prev"] = pv.reindex(np.arange(len(df))).to_numpy()
    out[f"{prefix}_roll7"] = rv.reindex(np.arange(len(df))).to_numpy()
    return out


def register_new_source_candidates() -> list[str]:
    """Register one Candidate per configured source into the signal_lab REGISTRY. Idempotent."""
    names = []
    for name, cfg in SOURCE_CONFIGS.items():
        names.append(name)
        if name in sl.REGISTRY:
            continue
        feats = [f"{name}_prev", f"{name}_roll7"]
        sl.register(sl.Candidate(
            name=name, level=2, feats=feats,
            build=partial(_causal_join, cfg=cfg, prefix=name),
            note=f"NEW source '{cfg['src']}' as a bacteria signal (driver-null prior — expect WASH)"))
    return names


def run(sources: list[str] | None = None, *, do_lobo: bool = True) -> dict:
    names = register_new_source_candidates()
    if sources:
        names = [n for n in names if n in set(sources)]
    obs = _REPO / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet"
    rain = _REPO / "bacteria_results" / "rainfall"
    res = sl.evaluate(obs, names, rain_dir=str(rain), do_lobo=do_lobo)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdicts.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def render_md(res: dict) -> str:
    L = ["# New-source BACTERIA signal gate", "",
         f"Baseline (FEATS+rain, San-Diego-excluded): AP {res.get('baseline',{}).get('ap')} "
         f"· LOBO AP {res.get('baseline',{}).get('lobo_ap')} · n_test {res.get('n_test')} "
         f"· events {res.get('events')}", "",
         "KEEP only if temporal ΔAP ≥ 0.005 AND survives leave-one-beach-out.", "",
         "| source signal | ΔAP (temporal) | LOBO ΔAP | verdict | reason |",
         "|---|--:|--:|:-:|---|"]
    for c in res.get("candidates", []):
        L.append(f"| `{c['name']}` | {c.get('delta_ap'):+.4f} | "
                 f"{('%+.4f' % c['lobo_delta_ap']) if c.get('lobo_delta_ap') is not None else '—'} | "
                 f"**{c['verdict']}** | {c.get('reason','')} |")
    keeps = [c['name'] for c in res.get('candidates', []) if c['verdict'] == 'KEEP']
    L += ["", f"**KEEP: {keeps or 'none — all new sources are washes over FEATS+rain'}**",
          "", "_Honest: a wash is a wash. The bench already showed the full driver set lifts AP only "
          "~+0.01 for boosted trees, so most new sources are expected to wash over the rain baseline._"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=None, help="comma subset (default: all configured)")
    ap.add_argument("--no-lobo", action="store_true", help="skip leave-one-beach-out (fast screen)")
    args = ap.parse_args(argv)
    srcs = [s.strip() for s in args.sources.split(",")] if args.sources else None
    res = run(srcs, do_lobo=not args.no_lobo)
    if "error" in res:
        print("ERROR:", res["error"]); return 1
    md = render_md(res)
    (OUT / "verdicts.md").write_text(md + "\n", encoding="utf-8")
    sl.echo_markdown(md)
    print(f"\nwrote {OUT / 'verdicts.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
