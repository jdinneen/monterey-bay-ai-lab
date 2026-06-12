#!/usr/bin/env python3
"""Cross-source correlation discovery over the normalized lakehouse — no target, no gate.

The question this answers is deliberately open-ended: *what moves with what?* It reduces each
numeric measurement variable to one statewide DAILY series, computes every pairwise correlation
across sources, and ranks the strongest positives — even when we have no hypothesis for why.

This is NOT a predictive model and has NO target/baseline gate (that work lives in
research/bacteria/ and research/hab/). It is also the shared "load each source as a clean daily
series" library that `source_consistency_check.py` (the data-QA gate) builds on, so the catalog
and loaders live here once.

Honesty layer (shown, never used to hide anything):
  * Pearson (linear) AND Spearman (rank/monotonic) for every pair.
  * A DESEASONALIZED Pearson (each series minus its monthly climatology) so you can see which
    correlations are just shared seasonal swing vs co-movement that survives removing the season.
  * Overlap n per pair + a min-overlap floor; daily series are autocorrelated so the effective
    sample size (and thus significance) is smaller than n — treat r, not p, as the signal.

Usage:  python research/data_lab/correlation_discovery.py [--output-dir reports/data_lab]
                 [--min-overlap 180] [--top 50] [--min-abs-r 0.3] [--cluster-abs-r 0.6]
Outputs (reports/data_lab/): correlation_discovery.json, .md, correlation_matrix.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "data" / "external_curated"
OUTDIR = ROOT / "reports" / "data_lab"
SEP = "::"   # column = "<source>::<var>"; source = col.split(SEP)[0]

# Map per-source variable names to a canonical physical QUANTITY, so "same thing, different product"
# (airtemp~tavg, srad~swrad, pres~slp) is a data-consistency check, and only DIFFERENT quantities
# (discharge~precip, wind~waveheight, water~air temp) count as a cross-quantity discovery. Reused
# by source_consistency_check.py to define which variables SHOULD agree.
CANON = {
    "airtemp": "air_temp", "tavg": "air_temp", "tmax": "air_temp", "tmin": "air_temp",
    "temp": "air_temp", "dewpt": "air_temp",
    "sst": "water_temp", "watertemp": "water_temp",
    "pres": "pressure", "slp": "pressure",
    "srad": "solar", "swrad": "solar", "solar": "solar",
    "wind": "wind", "wspd": "wind",
    "precip": "precip", "prcp": "precip",
    "rh": "humidity", "rhmax": "humidity", "vpd": "humidity",
    "sal": "salinity", "chl": "chl",
    "hs": "wave", "tp": "wave", "dp": "wave", "wvht": "wave", "dpd": "wave", "apd": "wave",
}

# One statewide daily series per numeric measurement worth correlating.
# kind="wide": value columns already exist.  kind="long": one (param,value) pair, pivoted.
# kind="speed": vector magnitude sqrt(u^2+v^2) (+ optional extra wide cols).
CATALOG = [
    {"name": "mursst", "dir": "mur_sst", "time": "time", "kind": "wide", "vars": {"sst_c": "sst"}},
    {"name": "viirs", "dir": "viirs_chl", "time": "time", "kind": "wide", "vars": {"chlor_a": "chl"}},
    {"name": "cencoos_oa", "dir": "cencoos_ocean_acidification", "time": "time", "kind": "wide",
     "vars": {"sea_water_temperature": "temp", "sea_water_practical_salinity": "sal",
              "mass_concentration_of_chlorophyll_in_sea_water": "chl",
              "moles_of_oxygen_per_unit_mass_in_sea_water": "o2",
              "sea_water_ph_reported_on_total_scale": "ph",
              "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": "pco2"}},
    {"name": "mbal_moor", "dir": "cencoos_mbal_moorings", "time": "time", "kind": "wide",
     "vars": {"sea_water_temperature": "temp", "sea_water_practical_salinity": "sal",
              "air_temperature": "airtemp"}},
    {"name": "hfradar", "dir": "hf_radar", "time": "time", "kind": "speed", "u": "u", "v": "v",
     "speed_name": "current_speed"},
    {"name": "ndbc", "dir": "ndbc_stdmet", "time": "time", "kind": "wide",
     "vars": {"WSPD": "wspd", "WVHT": "wvht", "DPD": "dpd", "APD": "apd", "PRES": "pres",
              "ATMP": "airtemp", "WTMP": "watertemp"}},
    {"name": "ndbc_cwind", "dir": "ndbc_cwind", "time": "time", "kind": "wide", "vars": {"WSPD": "wspd"}},
    {"name": "cdip_wave", "dir": "cdip_wave_network", "time": "time", "kind": "wide",
     "vars": {"waveHs": "hs", "waveTp": "tp", "waveDp": "dp"}},
    {"name": "cdip_sst", "dir": "cdip_sst_network", "time": "time", "kind": "wide", "vars": {"sst_c": "sst"}},
    {"name": "isd", "dir": "ncei_isd_hourly", "time": "time", "kind": "wide",
     "vars": {"air_temp_c": "airtemp", "dewpoint_c": "dewpt", "slp_hpa": "slp", "wind_speed_ms": "wspd"}},
    {"name": "tide", "dir": "coops_tide_staged", "time": "sample_date", "kind": "wide",
     "vars": {"water_level_m": "water_level"}},
    {"name": "gridmet", "dir": "gridmet_daily", "time": "date", "kind": "long",
     "param": "parameter", "value": "value",
     "vars": {"wind_ms": "wind", "vpd_kpa": "vpd", "precip_mm": "precip", "srad_wm2": "srad",
              "tmax_k": "tmax", "rh_max_pct": "rhmax"}},
    {"name": "nasapower", "dir": "nasa_power_daily", "time": "date", "kind": "long",
     "param": "parameter", "value": "value",
     "vars": {"WS10M": "wind", "PRECTOTCORR": "precip", "ALLSKY_SFC_SW_DWN": "solar",
              "T2M": "temp", "RH2M": "rh"}},
    {"name": "era5", "dir": "open_meteo_archive", "time": "time", "kind": "long",
     "param": "parameter", "value": "value",
     "vars": {"wind_speed_10m": "wind", "precipitation": "precip", "shortwave_radiation": "swrad",
              "surface_pressure": "pres", "temperature_2m": "temp", "relative_humidity_2m": "rh",
              "cloud_cover": "cloud"}},
    {"name": "usgs", "dir": "usgs_dv_statewide", "time": "date", "kind": "long",
     "param": "parameter", "value": "result_value",
     "vars": {"discharge_cfs": "discharge", "turbidity_fnu": "turbidity", "water_temp_c": "watertemp",
              "dissolved_oxygen_mgl": "do", "ph": "ph", "sp_conductance_uscm": "spc"}},
    {"name": "ghcnd", "dir": "ghcnd", "time": "date", "kind": "long", "param": "element", "value": "value",
     "vars": {"PRCP": "precip", "TMAX": "tmax", "TMIN": "tmin", "TAVG": "tavg"}},
    {"name": "charm", "dir": "c_harm", "time": "time", "kind": "wide",
     "vars": {"pseudo_nitzschia": "pn", "particulate_domoic": "pda", "cellular_domoic": "cda"}},
    {"name": "habmap", "dir": "habmap_cdph", "time": "time", "kind": "wide",
     "vars": {"pDA": "pda", "Pseudo_nitzschia_seriata_group": "pn_seriata",
              "Pseudo_nitzschia_delicatissima_group": "pn_delica", "Temp": "temp",
              "Nitrate": "nitrate", "Avg_Chloro": "chl"}},
    {"name": "surfrider", "dir": "surfrider_bwtf", "time": "sample_date", "kind": "wide",
     "vars": {"bacteria_result": "bacteria"}},
    {"name": "roms", "dir": "roms_circulation", "time": "time", "kind": "speed", "u": "u", "v": "v",
     "speed_name": "current_speed", "extra": {"temp": "temp", "salt": "salt"}},
]


def _daily(spec: dict) -> pd.DataFrame | None:
    """Reduce one source to a statewide daily-mean wide frame indexed by 'day', cols '<name>::<var>'."""
    pq = CURATED / spec["dir"] / f"{spec['dir']}.parquet"
    if not pq.exists():
        return None
    name, tcol = spec["name"], spec["time"]

    if spec["kind"] == "long":
        df = pd.read_parquet(pq, columns=[tcol, spec["param"], spec["value"]])
        df = df[df[spec["param"]].isin(spec["vars"])].copy()
        df[spec["value"]] = pd.to_numeric(df[spec["value"]], errors="coerce")
        df["day"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["day"])
        g = df.groupby(["day", spec["param"]])[spec["value"]].mean().unstack(spec["param"])
        g = g.rename(columns=spec["vars"])
        out = g[[v for v in spec["vars"].values() if v in g.columns]]
    elif spec["kind"] == "speed":
        cols = [tcol, spec["u"], spec["v"], *spec.get("extra", {}).keys()]
        df = pd.read_parquet(pq, columns=cols)
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[spec["speed_name"]] = np.sqrt(df[spec["u"]] ** 2 + df[spec["v"]] ** 2)
        df["day"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["day"])
        agg = {spec["speed_name"]: "mean", **{k: "mean" for k in spec.get("extra", {})}}
        out = df.groupby("day").agg(agg).rename(columns=spec.get("extra", {}))
    else:  # wide
        keys = list(spec["vars"])
        df = pd.read_parquet(pq, columns=[tcol, *keys])
        for c in keys:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["day"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["day"])
        out = df.groupby("day")[keys].mean().rename(columns=spec["vars"])

    return out.rename(columns={c: f"{name}{SEP}{c}" for c in out.columns})


def build_matrix() -> tuple[pd.DataFrame, dict]:
    frames, coverage = [], {}
    for spec in CATALOG:
        try:
            d = _daily(spec)
        except Exception as e:  # one bad source never sinks the run
            coverage[spec["name"]] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if d is None or d.empty:
            coverage[spec["name"]] = {"status": "missing/empty"}
            continue
        coverage[spec["name"]] = {"days": int(d.notna().any(axis=1).sum()), "vars": list(d.columns)}
        frames.append(d)
        print(f"  loaded {spec['name']:12} {len(d.columns)} vars, {len(d)} days", flush=True)
    wide = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return wide, coverage


def _deseasonalize(wide: pd.DataFrame) -> pd.DataFrame:
    """Subtract each variable's monthly climatology so shared seasonality is removed."""
    m = wide.index.month
    return wide - wide.groupby(m).transform("mean")


def _rank_pairs(pear, spear, pear_anom, overlap, min_overlap, min_abs_r) -> list[dict]:
    pairs, cols = [], list(pear.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            n = int(overlap.loc[a, b])
            r = pear.loc[a, b]
            if n < min_overlap or pd.isna(r) or abs(r) < min_abs_r:
                continue
            qa = CANON.get(a.split(SEP)[1], a.split(SEP)[1])
            qb = CANON.get(b.split(SEP)[1], b.split(SEP)[1])
            pairs.append({
                "a": a, "b": b, "cross_source": a.split(SEP)[0] != b.split(SEP)[0],
                "same_quantity": qa == qb, "n": n,
                "pearson": round(float(r), 3),
                "spearman": round(float(spear.loc[a, b]), 3) if not pd.isna(spear.loc[a, b]) else None,
                "pearson_deseasonalized": round(float(pear_anom.loc[a, b]), 3)
                if not pd.isna(pear_anom.loc[a, b]) else None,
            })
    return pairs


def cluster_variables(pear: pd.DataFrame, abs_r_threshold: float = 0.6) -> list[list[str]]:
    """Group variables into correlated families: average-linkage on 1-|pearson|, cut so members
    are linked at |pearson| >= abs_r_threshold. Singletons are dropped from the reported groups."""
    cols = list(pear.columns)
    if len(cols) < 3:
        return []
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        dist = np.nan_to_num((1 - pear.abs()).to_numpy(), nan=1.0)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2
        Z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(Z, t=1 - abs_r_threshold, criterion="distance")
    except Exception:
        return []
    groups: dict[int, list[str]] = {}
    for col, lab in zip(cols, labels):
        groups.setdefault(int(lab), []).append(col)
    return sorted([g for g in groups.values() if len(g) > 1], key=len, reverse=True)


def run(output_dir: Path | None = None, min_overlap: int = 180, top: int = 50,
        min_abs_r: float = 0.3, cluster_abs_r: float = 0.6) -> dict:
    out_dir = Path(output_dir) if output_dir else OUTDIR
    print("[corr] building statewide-daily matrix ...", flush=True)
    wide, coverage = build_matrix()
    if wide.empty:
        raise SystemExit("no sources loaded")
    wide = wide[wide.index.notna()]
    print(f"[corr] matrix: {wide.shape[1]} variables x {wide.shape[0]} days; correlating ...", flush=True)

    notna = wide.notna().astype(int)
    overlap = notna.T.dot(notna)
    pear = wide.corr(method="pearson", min_periods=min_overlap)
    spear = wide.corr(method="spearman", min_periods=min_overlap)
    pear_anom = _deseasonalize(wide).corr(method="pearson", min_periods=min_overlap)

    pairs = _rank_pairs(pear, spear, pear_anom, overlap, min_overlap, min_abs_r)
    positive = sorted([p for p in pairs if p["pearson"] > 0], key=lambda p: -p["pearson"])
    cross_pos = [p for p in positive if p["cross_source"]]
    cross_quantity = [p for p in cross_pos if not p["same_quantity"]]
    consistency = [p for p in cross_pos if p["same_quantity"]]   # same quantity, diff product
    survives = [p for p in cross_quantity if (p["pearson_deseasonalized"] or 0) >= max(0.2, min_abs_r * 0.66)]
    clusters = cluster_variables(pear, cluster_abs_r)

    out = {
        "what": "cross-source positive correlation discovery (statewide daily, no target, no gate)",
        "params": {"min_overlap_days": min_overlap, "min_abs_r": min_abs_r, "top": top,
                   "cluster_abs_r": cluster_abs_r},
        "matrix": {"n_variables": int(wide.shape[1]), "n_days": int(wide.shape[0]),
                   "date_min": str(wide.index.min().date()), "date_max": str(wide.index.max().date())},
        "coverage": coverage,
        "n_pairs_considered": len(pairs),
        "top_cross_quantity_positive": cross_quantity[:top],
        "top_survives_deseasonalized": survives[:top],
        "data_consistency_same_quantity": consistency[:top],
        "variable_clusters": clusters,
        "caveats": [
            "Correlation is not causation; these are co-movements, surfaced without a hypothesis.",
            "Daily series are autocorrelated -> effective sample size << n, so significance is "
            "overstated; read the correlation magnitude, not a p-value.",
            "Many pairs are tested; expect some strong correlations by chance.",
            "raw pearson can be inflated by shared seasonality -> compare pearson_deseasonalized.",
            "correlation is scale/offset invariant: high r confirms co-movement, NOT that two "
            "products agree in absolute units -- that check is source_consistency_check.py.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "correlation_discovery.json").write_text(json.dumps(out, indent=2, default=str),
                                                        encoding="utf-8")
    pear.to_csv(out_dir / "correlation_matrix.csv")
    (out_dir / "correlation_discovery.md").write_text(_render_md(out), encoding="utf-8")
    return out


def _render_md(out: dict) -> str:
    m = out["matrix"]
    L = ["# Cross-source correlation discovery (no target, no gate)", "",
         f"What: {out['what']}.",
         f"Matrix: **{m['n_variables']} variables x {m['n_days']} days** "
         f"({m['date_min']} to {m['date_max']}), statewide daily means. "
         f"min overlap {out['params']['min_overlap_days']}d, |r| >= {out['params']['min_abs_r']}, "
         f"{out['n_pairs_considered']} qualifying pairs.", "",
         "## Discoveries: strongest POSITIVE cross-source, cross-QUANTITY correlations", "",
         "_(different measured things, in different sources, that move together. A `deseason. "
         "pearson` near the raw value = real co-movement; much lower = mostly shared seasonality.)_", "",
         "| a | b | pearson | spearman | deseason. pearson | n days |",
         "|---|---|--:|--:|--:|--:|"]
    for p in out["top_cross_quantity_positive"]:
        L.append(f"| `{p['a']}` | `{p['b']}` | {p['pearson']} | {p['spearman']} | "
                 f"{p['pearson_deseasonalized']} | {p['n']} |")
    L += ["", "## Variable clusters (families that move together, |pearson| >= "
          f"{out['params']['cluster_abs_r']})", ""]
    if out["variable_clusters"]:
        for i, g in enumerate(out["variable_clusters"], 1):
            L.append(f"- **cluster {i}** ({len(g)}): " + ", ".join(f"`{c}`" for c in g))
    else:
        L.append("- (no multi-member clusters at this threshold)")
    L += ["", "## Data-consistency check: same quantity across independent sources", "",
          "_(these we DO understand. High r = the products co-vary. For whether they AGREE in "
          "absolute units, see source_consistency_check.py.)_", "",
          "| a | b | pearson | deseason. pearson | n days |", "|---|---|--:|--:|--:|"]
    for p in out["data_consistency_same_quantity"][:20]:
        L.append(f"| `{p['a']}` | `{p['b']}` | {p['pearson']} | {p['pearson_deseasonalized']} | {p['n']} |")
    L += ["", "## How to read this", ""] + [f"- {c}" for c in out["caveats"]]
    L += ["", "_Columns are `source::variable`. The full pairwise matrix is in "
          "`correlation_matrix.csv`._"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTDIR))
    ap.add_argument("--min-overlap", type=int, default=180, help="min overlapping days per pair")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-abs-r", type=float, default=0.3, help="ignore |pearson| below this")
    ap.add_argument("--cluster-abs-r", type=float, default=0.6, help="|pearson| link threshold for clusters")
    args = ap.parse_args(argv)
    out = run(Path(args.output_dir), args.min_overlap, args.top, args.min_abs_r, args.cluster_abs_r)
    print(f"\n[corr] {out['matrix']['n_variables']} vars, "
          f"{len(out['top_cross_quantity_positive'])} cross-quantity positives, "
          f"{len(out['variable_clusters'])} clusters")
    print(f"[corr] wrote {Path(args.output_dir) / 'correlation_discovery.md'} (+ .json, matrix.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
