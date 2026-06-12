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

Portable: data path via --obs / $MBAL_BACTERIA_OBS, no hardcoded project root.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# CA single-sample standards (same as the project's existing models).
THR = dict(enterococcus=104.0, fecal=400.0, total=10000.0, total_ratio_gate=1000.0,
           fecal_total_ratio=0.1)
PARAM_MAP = {
    "Enterococcus": "ent", "Fecal Coliforms": "fec",
    "Total Coliforms": "tot", "E. Coli": "ecoli",
}
TIMED_ANALYTE_MAP = {
    "prop_enterococcus": "ent",
    "prop_fecal_coliform": "fec",
    "prop_total_coliform": "tot",
    "prop_e_coli": "ecoli",
}
FEATS = [
    "month", "doy_sin", "doy_cos",
    "ent_prev", "fec_prev", "tot_prev", "ecoli_prev",
    "ent_roll3_prev", "fec_roll3_prev", "tot_roll3_prev", "ecoli_roll3_prev",
    "exc_prev", "exc_prev2", "exc_roll5_prev", "exc_roll10_prev",
    "station_prior_rate", "station_prior_n", "days_since_prev",
    "cty_prev7", "sw_prev7", "latitude", "longitude", "nbr_prev1", "nbr_prev7"
]
# County -> evaluation region. Monterey Bay = Monterey + Santa Cruz counties.
MONTEREY_COUNTIES = {"Monterey", "Santa Cruz"}

# Rain features (added only when a rainfall dir is supplied) and the AB411 threshold.
RAIN_FEATS = ["rain_0d", "rain_3d", "rain_7d", "days_since_rain"]
AB411_RAIN_MM = 2.54  # 0.1 inch — the common CA wet-weather (AB411) advisory threshold

# First-flush river-discharge features (added only when a discharge dir is supplied).
DISCHARGE_FEATS = ["discharge_log", "discharge_3d", "discharge_anom", "days_since_highq"]

# Tide stage (CO-OPS water level) features (added only when tide_stages dir is supplied).
TIDE_STAGES_FEATS = ["water_level_m"]

# CDIP Wave features (added only when a CDIP dir is supplied).
WAVE_FEATS = ["Hs", "Tp", "Dm"]

# Oceanography features (MUR SST only, as HF Radar lacks 20-year historical coverage for training).
OCEANOGRAPHY_FEATS = ["sst_c_prev1d", "sst_c_roll7_prev"]
_AIR_JOIN_FEATS = ["air_temp_prev1d", "air_temp_roll7_prev"]
# warmspell = prev1d - 7-day mean (warmer-than-usual). Gated KEEP, +0.0158 AP, survives LOBO.
AIR_TEMP_FEATS = _AIR_JOIN_FEATS + ["air_temp_warmspell"]

EVENTS_JSONL: Path | None = None


def set_events_jsonl(path: str | Path | None) -> None:
    global EVENTS_JSONL
    EVENTS_JSONL = Path(path) if path else None
    if EVENTS_JSONL:
        EVENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        EVENTS_JSONL.write_text("", encoding="utf-8")


def emit_stage(stage: str, status: str = "start", **extra) -> None:
    if not EVENTS_JSONL:
        return
    payload = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **extra,
    }
    with EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")



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


def add_wave_features(df: pd.DataFrame, cdip_dir: Path) -> pd.DataFrame:
    """Causal wave features via geographic proximity to CDIP/NDBC buoys.
    Joins aggregated daily wave metrics (height, period, direction) to station-days.
    """
    cdip_dir = Path(cdip_dir)
    waves_path = cdip_dir / "cdip_waves.parquet"
    map_path = cdip_dir / "station_map.parquet"

    if not waves_path.exists() or not map_path.exists():
        print(f"[!] CDIP waves or map not found in {cdip_dir}")
        return df

    waves = pd.read_parquet(waves_path)
    smap = pd.read_parquet(map_path)

    # Aggregate hourly -> daily (mean)
    waves["sample_date"] = pd.to_datetime(waves["sample_date"]).dt.tz_localize(None)
    daily = waves.groupby(["station_id", waves["sample_date"].dt.normalize()]).agg({
        "Hs": "mean", "Tp": "mean", "Dm": "mean"
    }).reset_index()
    daily.rename(columns={"sample_date": "date"}, inplace=True)

    # Join beach -> buoy -> daily wave data
    df["sample_date"] = pd.to_datetime(df["sample_date"]).dt.tz_localize(None)
    left = df.merge(smap, on="station_id", how="left")
    out = left.merge(
        daily, left_on=["ndbc_id", "sample_date"],
        right_on=["station_id", "date"], how="left", suffixes=("", "_buoy")
    )
    return out.drop(columns=["date", "station_id_buoy", "ndbc_id", "distance_km"], errors="ignore")


def add_tide_stage_features(df: pd.DataFrame, tide_stages_dir: Path) -> pd.DataFrame:
    """Causal first-flush tide-stage (water level) features via geographic proximity
    to NOAA CO-OPS tide gauges. Joins daily-averaged water_level_m from nearest station.
    
    Uses tide_station_mapping.parquet to map beach UUIDs → nearest CO-OPS tide station ID,
    then merge_asof to get the last recorded water level on/before sample_date.
    """
    tide_stages_dir = Path(tide_stages_dir)

    # Find directories to search
    possible_dirs = []
    
    if "tide_stages" in str(tide_stages_dir):
        # Look for mapping in sibling directory
        possible_dirs.append(tide_stages_dir)  # tide_stages (data)
        possible_dirs.append((tide_stages_dir / "..").resolve() / "tide_stations")  # tide_stations (map)
    elif "tide_stations" in str(tide_stages_dir):
        possible_dirs.append(tide_stages_dir)  # tide_stations (map)
        possible_dirs.append((tide_stages_dir / "..").resolve() / "tide_stages")  # tide_stages (data)
    else:
        possible_dirs.append(tide_stages_dir)
    
    tide_path = None
    map_path = None
    
    for base_dir in possible_dirs:
        if not base_dir.exists():
            continue
        
        ts_path = base_dir / "tide_stages.parquet"
        map_file = base_dir / "tide_station_mapping.parquet"
        
        if ts_path.exists() and map_path is None:
            tide_path = ts_path
        if map_file.exists() and map_path is None:
            map_path = map_file
    
    if not tide_path or not map_path:
        print(f"[!] Tide stages or mapping not found in: {[str(d) for d in possible_dirs]}")
        return df

    # Load data
    tide = pd.read_parquet(tide_path)
    smap = pd.read_parquet(map_path)

    # Ensure datetime types match (both should be timezone-naive for merge_asof)
    tide["sample_date"] = pd.to_datetime(tide["sample_date"]).dt.tz_localize(None).dt.normalize()
    
    # Map beach UUIDs to CO-OPS station IDs
    left = df.merge(smap[["beach_station_uuid", "tide_station_id"]],
                    left_on="station_id", right_on="beach_station_uuid",
                    how="left")
    
    # Ensure sample_date is timezone-naive too
    left = left.copy()
    left["sample_date"] = pd.to_datetime(left["sample_date"]).dt.tz_localize(None)
    
    # Use merge_asof for temporal join: get the latest tide measurement on/before sample_date.
    # The tide frame's gauge id is also named "station_id"; rename it so it does not collide
    # with the beach station_id (a collision makes merge_asof emit station_id_x/_y and drops the
    # beach key the rest of the benchmark groups on).
    left = left.sort_values("sample_date")
    tide_sorted = tide.rename(columns={"station_id": "tide_gauge_id"}).sort_values("sample_date")

    merged = pd.merge_asof(
        left,
        tide_sorted[["tide_gauge_id", "sample_date", "water_level_m"]],
        on="sample_date",
        left_by="tide_station_id",
        right_by="tide_gauge_id",
        direction="backward"
    )

    # Clean up (keep the beach station_id; drop only the join helpers)
    result = merged.drop(columns=["date", "beach_station_uuid", "tide_station_id",
                                  "tide_gauge_id", "distance_km"], errors="ignore")
    
    # Fill NaNs in water_level_m with a sentinel value that HGBT can handle
    if "water_level_m" in result.columns:
        result["water_level_m"] = result["water_level_m"].fillna(-999.0)
    
    return result


def add_oceanography_features(df: pd.DataFrame, ocean_dir: Path) -> pd.DataFrame:
    """Joins strictly causal MUR SST (timeseries) and HF Radar (nearest lat/lon)."""
    out = df.copy()
    ocean_dir = Path(ocean_dir)
    out["sample_date_tznaive"] = pd.to_datetime(out["sample_date"]).dt.tz_localize(None).dt.normalize()

    # 1. SST (1D Timeseries)
    sst_path = ocean_dir / "mur_sst" / "mur_sst.parquet"
    if sst_path.exists():
        sst = pd.read_parquet(sst_path)
        sst["date"] = pd.to_datetime(sst["time"]).dt.tz_localize(None).dt.normalize()
        daily_sst = sst.groupby("date")["sst_c"].mean().reset_index()
        daily_sst["sst_c_prev1d"] = daily_sst["sst_c"].shift(1)
        daily_sst["sst_c_roll7_prev"] = daily_sst["sst_c"].shift(1).rolling(7, min_periods=1).mean()
        
        out = out.merge(daily_sst[["date", "sst_c_prev1d", "sst_c_roll7_prev"]], 
                        left_on="sample_date_tznaive", right_on="date", how="left")
        out = out.drop(columns=["date"])

    # 2. HF Radar
    hf_path = ocean_dir / "hf_radar" / "hf_radar.parquet"
    geo_path = Path(__file__).resolve().parents[2] / "reports" / "station_geo.parquet"
    if hf_path.exists() and geo_path.exists():
        hf = pd.read_parquet(hf_path)
        geo = pd.read_parquet(geo_path).dropna(subset=["latitude", "longitude"])
        
        hf["date"] = pd.to_datetime(hf["time"]).dt.tz_localize(None).dt.normalize()
        daily_hf = hf.groupby(["lat", "lon", "date"])[["u", "v"]].mean().reset_index()
        daily_hf = daily_hf.sort_values(["lat", "lon", "date"])
        
        cells = daily_hf[["lat", "lon"]].drop_duplicates()
        if not cells.empty:
            from sklearn.neighbors import BallTree
            tree = BallTree(np.radians(cells[["lat", "lon"]].to_numpy()), metric="haversine")
            
            geo_stations = geo[geo["station_id"].isin(out["station_id"])].copy()
            if not geo_stations.empty:
                _, idx = tree.query(np.radians(geo_stations[["latitude", "longitude"]].to_numpy()), k=1)
                geo_stations["nearest_hf_lat"] = cells.iloc[idx.flatten()]["lat"].to_numpy()
                geo_stations["nearest_hf_lon"] = cells.iloc[idx.flatten()]["lon"].to_numpy()
                
                out = out.merge(geo_stations[["station_id", "nearest_hf_lat", "nearest_hf_lon"]], on="station_id", how="left")
                
                daily_hf["u_prev1d"] = daily_hf.groupby(["lat", "lon"])["u"].shift(1)
                daily_hf["v_prev1d"] = daily_hf.groupby(["lat", "lon"])["v"].shift(1)
                daily_hf["hf_speed_prev1d"] = np.sqrt(daily_hf["u_prev1d"]**2 + daily_hf["v_prev1d"]**2)
                
                out = out.merge(
                    daily_hf[["lat", "lon", "date", "u_prev1d", "v_prev1d", "hf_speed_prev1d"]],
                    left_on=["nearest_hf_lat", "nearest_hf_lon", "sample_date_tznaive"],
                    right_on=["lat", "lon", "date"],
                    how="left"
                )
                out = out.drop(columns=["nearest_hf_lat", "nearest_hf_lon", "lat", "lon", "date"])

    return out.drop(columns=["sample_date_tznaive"])


def add_air_temp_features(df: pd.DataFrame, air_dir: Path) -> pd.DataFrame:
    """Nearest-grid-cell ERA5 (Open-Meteo archive) 2-m air temperature, strictly causal
    (previous-day value + 7-day trailing mean). This is the ONE new ingested source that
    cleared the bacteria signal gate (temporal ΔAP +0.0131, survives leave-one-beach-out);
    `air_dir` is the curated root holding ``open_meteo_archive/open_meteo_archive.parquet``."""
    out = df.copy()
    path = Path(air_dir) / "open_meteo_archive" / "open_meteo_archive.parquet"
    geo_path = Path(__file__).resolve().parents[2] / "reports" / "station_geo.parquet"
    if not (path.exists() and geo_path.exists()):
        return out
    src = pd.read_parquet(path, columns=["grid_lat", "grid_lon", "time", "parameter", "value"])
    src = src[src["parameter"] == "temperature_2m"]
    if src.empty:
        return out
    # distinct cell-column names so we don't collide with the rainfall grid (grid_lat/grid_lon)
    src = src.rename(columns={"grid_lat": "air_glat", "grid_lon": "air_glon"})
    src["date"] = pd.to_datetime(src["time"], utc=True).dt.tz_localize(None).dt.normalize()
    src["value"] = pd.to_numeric(src["value"], errors="coerce")
    daily = (src.groupby(["air_glat", "air_glon", "date"])["value"].mean().reset_index()
                .sort_values(["air_glat", "air_glon", "date"]))
    g = daily.groupby(["air_glat", "air_glon"])["value"]
    daily["air_temp_prev1d"] = g.transform(lambda s: s.shift(1))
    daily["air_temp_roll7_prev"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    cells = daily[["air_glat", "air_glon"]].drop_duplicates()
    geo = pd.read_parquet(geo_path).dropna(subset=["latitude", "longitude"])
    geo = geo[geo["station_id"].isin(out["station_id"])].copy()
    if cells.empty or geo.empty:
        return out
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(cells[["air_glat", "air_glon"]].to_numpy()), metric="haversine")
    _, idx = tree.query(np.radians(geo[["latitude", "longitude"]].to_numpy()), k=1)
    geo["air_glat"] = cells.iloc[idx.flatten()]["air_glat"].to_numpy()
    geo["air_glon"] = cells.iloc[idx.flatten()]["air_glon"].to_numpy()
    out["sample_date_tznaive"] = pd.to_datetime(out["sample_date"]).dt.tz_localize(None).dt.normalize()
    out = out.merge(geo[["station_id", "air_glat", "air_glon"]], on="station_id", how="left")
    out = out.merge(daily[["air_glat", "air_glon", "date", *_AIR_JOIN_FEATS]],
                    left_on=["air_glat", "air_glon", "sample_date_tznaive"],
                    right_on=["air_glat", "air_glon", "date"], how="left")
    # derived warm-spell: how much warmer than the trailing week (a gated KEEP, +0.0158 AP)
    out["air_temp_warmspell"] = out["air_temp_prev1d"] - out["air_temp_roll7_prev"]
    return out.drop(columns=[c for c in ["date", "air_glat", "air_glon", "sample_date_tznaive"]
                             if c in out.columns])


def default_obs_path() -> Path:
    env = os.environ.get("MBAL_BACTERIA_OBS")
    if env:
        return Path(env)
    root = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    return root / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet"


def portable_obs_path(obs_path: Path) -> str:
    """Project-root-relative POSIX path for the emitted payload.

    An absolute path both leaks the local filesystem and makes the frozen
    expected/ outputs impossible to reproduce byte-identically on another
    machine. Record the path relative to the project root instead.
    """
    root = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    try:
        return Path(obs_path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(obs_path).name


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


def _ensure_optional_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["latitude", "longitude"]:
        if c not in out:
            for cand in (f"{c}_y", f"{c}_x"):
                if cand in out:
                    out[c] = out[cand]
                    break
        if c not in out:
            out[c] = np.nan
        out = out.drop(columns=[f"{c}_x", f"{c}_y"], errors="ignore")
    for c in ["nbr_prev1", "nbr_prev7"]:
        if c not in out:
            out[c] = np.nan
    return out


def load_timed_station_days(timed_path: Path, label: str = "enterococcus") -> pd.DataFrame:
    """Worst-of-day station panel from timed labels with record-level `available_at`.

    Expected input is a timed-labels parquet (one row per analyte result) with
    `record_status`, `analyte`, `value`, `sampledate`, `station_id`, `county`,
    and `available_at`.
    """
    o = pd.read_parquet(timed_path)
    o = o[o["record_status"] == "accepted"].copy()
    o["p"] = o["analyte"].map(TIMED_ANALYTE_MAP)
    o = o.dropna(subset=["p"])
    o["value"] = pd.to_numeric(o["value"], errors="coerce")
    o = o[o["value"] >= 0]
    o["sample_date"] = pd.to_datetime(o["sampledate"])
    o["available_at"] = pd.to_datetime(o["available_at"])
    o = o.dropna(subset=["sample_date", "available_at"])
    o = _apply_timed_station_bridge(o)

    agg = (o.groupby(["station_id", "county", "sample_date", "p"])["value"]
             .max().unstack("p").reset_index())
    for c in ["ent", "fec", "tot", "ecoli"]:
        if c not in agg:
            agg[c] = np.nan
    avail = (o.groupby(["station_id", "county", "sample_date"])["available_at"]
               .max().reset_index())
    agg = agg.merge(avail, on=["station_id", "county", "sample_date"], how="left")

    if label == "enterococcus":
        agg = agg[agg["ent"].notna()].copy()
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
    agg = agg.dropna(subset=["available_at"])
    return _ensure_optional_feature_columns(agg.sort_values(["station_id", "sample_date"]).reset_index(drop=True))


def _station_name_key(df: pd.DataFrame) -> pd.Series:
    def norm(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("").str.strip().str.lower()
    return norm(df["county"]) + "|" + norm(df["beach_name"]) + "|" + norm(df["station_name"])


def _apply_timed_station_bridge(o: pd.DataFrame) -> pd.DataFrame:
    """Map raw BeachWatch numeric station ids to the hashed research station ids.

    Rainfall/discharge/spatial driver maps were built against the public research
    observation parquet's hashed `station_id`. The timed foundation preserves the
    raw BeachWatch station id. The stable bridge is the provider station identity
    triple: county + beach_name + station_name. If the research obs file is not
    present (e.g. a tiny unit test), the timed ids are left unchanged.
    """
    needed = {"county", "beach_name", "station_name"}
    if not needed <= set(o.columns):
        return o
    obs_path = default_obs_path()
    if not obs_path.exists():
        return o
    obs = pd.read_parquet(obs_path, columns=["station_id", "county", "beach_name", "station_name"]).drop_duplicates()
    obs["_station_bridge_key"] = _station_name_key(obs)
    if obs["_station_bridge_key"].duplicated().any():
        return o
    bridge = obs[["_station_bridge_key", "station_id"]].rename(columns={"station_id": "research_station_id"})
    out = o.copy()
    out["source_station_id"] = out["station_id"].astype("string")
    out["_station_bridge_key"] = _station_name_key(out)
    out = out.merge(bridge, on="_station_bridge_key", how="left")
    mapped = out["research_station_id"].notna()
    out.loc[mapped, "station_id"] = out.loc[mapped, "research_station_id"].astype("string")
    return out.drop(columns=["_station_bridge_key", "research_station_id"], errors="ignore")


def _predecessor_idx(dates_ns: np.ndarray, lag_ns: int) -> np.ndarray:
    """Within a date-sorted station, the position of the latest prior sample whose lab
    result had RETURNED by each row's sample time (date <= sample_date - lag); -1 if none.
    lag_ns=0 reduces exactly to "the previous sample" (shift(1))."""
    pos = np.searchsorted(dates_ns, dates_ns - lag_ns, side="right") - 1
    i = np.arange(len(dates_ns))
    return np.where(pos >= i, i - 1, pos)  # lag=0 ties -> previous row


def _avail_predecessor_idx(sample_ns: np.ndarray, avail_ns: np.ndarray) -> np.ndarray:
    out = np.full(len(sample_ns), -1, dtype=np.int64)
    for i, ti in enumerate(sample_ns):
        j = i - 1
        while j >= 0:
            if avail_ns[j] <= ti:
                out[i] = j
                break
            j -= 1
    return out


def _gather_float(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    m = idx >= 0
    out[m] = np.asarray(arr, dtype=float)[idx[m]]
    return out


def _add_calendar_lab_features(df: pd.DataFrame, predecessor_fn) -> pd.DataFrame:
    df = df.sort_values(["station_id", "sample_date"]).copy().reset_index(drop=True)
    d = df["sample_date"]
    doy = d.dt.dayofyear
    df["month"] = d.dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    for c in ["ent", "fec", "tot", "ecoli"]:
        df[f"{c}_logp"] = np.log1p(df[c].clip(lower=0))

    parts = []
    for _, g in df.groupby("station_id", sort=False):
        pidx = predecessor_fn(g)
        pidx2 = np.full(len(pidx), -1, dtype=np.int64)
        ok = pidx >= 0
        pidx2[ok] = pidx[pidx[ok]]
        dates = g["sample_date"].to_numpy(dtype="datetime64[ns]")
        exc = g["exceed"].to_numpy(dtype=float)
        feat = {
            "exc_prev": _gather_float(exc, pidx),
            "exc_prev2": _gather_float(exc, pidx2),
            "exc_roll5_prev": _gather_float(pd.Series(exc).rolling(5, min_periods=1).mean().to_numpy(), pidx),
            "exc_roll10_prev": _gather_float(pd.Series(exc).rolling(10, min_periods=1).mean().to_numpy(), pidx),
            "station_prior_rate": _gather_float(pd.Series(exc).expanding().mean().to_numpy(), pidx),
            "station_prior_n": _gather_float(pd.Series(exc).expanding().count().to_numpy(), pidx),
        }
        for c in ["ent", "fec", "tot", "ecoli"]:
            lp = g[f"{c}_logp"].to_numpy(dtype=float)
            feat[f"{c}_prev"] = _gather_float(lp, pidx)
            feat[f"{c}_roll3_prev"] = _gather_float(pd.Series(lp).rolling(3, min_periods=1).mean().to_numpy(), pidx)
        dsp = np.full(len(pidx), np.nan)
        dsp[ok] = (dates[ok] - dates[pidx[ok]]).astype("timedelta64[D]").astype(float)
        feat["days_since_prev"] = dsp
        parts.append(pd.DataFrame(feat, index=g.index))
    return pd.concat([df, pd.concat(parts).sort_index()], axis=1)


def add_causal_features(df: pd.DataFrame, reveal_lag_days: int = 0) -> pd.DataFrame:
    """Strictly-causal features. With ``reveal_lag_days`` > 0, every "prior lab / memory"
    feature uses the latest prior sample whose result had returned by sample time (date <=
    sample_date - lag), so a same-/next-day resample cannot feed a label that had not yet
    come back. ``reveal_lag_days=0`` reproduces the historical shift(1) behavior exactly."""
    lag_ns = int(reveal_lag_days) * 86_400_000_000_000
    def predecessor(g: pd.DataFrame) -> np.ndarray:
        dates = g["sample_date"].to_numpy(dtype="datetime64[ns]")
        return _predecessor_idx(dates.view("int64"), lag_ns)
    df = _add_calendar_lab_features(df, predecessor)

    cty = (df.groupby(["county", "sample_date"])["exceed"].mean().rename("cty_day_rate").reset_index()
             .sort_values(["county", "sample_date"]))
    cty["cty_prev7"] = (cty.groupby("county")["cty_day_rate"]
                          .apply(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
                          .reset_index(level=0, drop=True))
    df = df.merge(cty[["county", "sample_date", "cty_prev7"]], on=["county", "sample_date"], how="left")
    sw = (df.groupby("sample_date")["exceed"].mean().rename("sw_day_rate").reset_index().sort_values("sample_date"))
    sw["sw_prev7"] = sw["sw_day_rate"].shift(1).rolling(7, min_periods=1).mean()
    df = df.merge(sw[["sample_date", "sw_prev7"]], on="sample_date", how="left")
    
    # Add Spatial Tensor Features (K-Nearest Neighbors and static Lat/Lon)
    geo_path = Path(__file__).resolve().parents[2] / "reports" / "station_geo.parquet"
    if geo_path.exists():
        from research.bacteria.spatial_drivers_experiment import add_knn_spatial_lag
        geo = pd.read_parquet(geo_path)
        # add_knn_spatial_lag needs 'exceed' column to compute neighbors' prior states
        df = add_knn_spatial_lag(df, geo, k=8, reveal_lag_days=reveal_lag_days)
        geo_t = geo.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
        df = df.merge(geo_t[["station_id", "latitude", "longitude"]], on="station_id", how="left")

    return _ensure_optional_feature_columns(df)


def add_causal_features_available(df: pd.DataFrame, reveal_lag_days: int = 0) -> pd.DataFrame:
    """Strictly causal features using real record-level `available_at`.

    Prior lab, memory, county, and statewide features only use labels whose
    provider record-entry time is known by the target sample time. Spatial
    neighbor features are intentionally left null until an available-at-aware
    neighbor implementation is added.
    """
    if "available_at" not in df.columns:
        raise ValueError("availability_mode='available_at' requires an available_at column")

    def predecessor(g: pd.DataFrame) -> np.ndarray:
        dates = g["sample_date"].to_numpy(dtype="datetime64[ns]")
        avail = g["available_at"].to_numpy(dtype="datetime64[ns]")
        return _avail_predecessor_idx(dates.view("int64"), avail.view("int64"))

    df = _add_calendar_lab_features(df, predecessor)

    cty_rows = []
    for county, g in df.groupby("county", sort=False):
        events = g[["sample_date", "available_at", "exceed"]].dropna(subset=["sample_date", "available_at"])
        dates = pd.Series(pd.to_datetime(g["sample_date"].drop_duplicates()).sort_values().to_numpy())
        vals = []
        for d in dates:
            revealed = events[events["available_at"] <= d]
            recent = revealed[revealed["sample_date"] >= (d - pd.Timedelta(days=7))]
            vals.append(float(recent["exceed"].mean()) if not recent.empty else np.nan)
        cty_rows.append(pd.DataFrame({"county": county, "sample_date": dates, "cty_prev7": vals}))
    if cty_rows:
        cty = pd.concat(cty_rows, ignore_index=True)
        df = df.merge(cty, on=["county", "sample_date"], how="left")
    else:
        df["cty_prev7"] = np.nan

    events = df[["sample_date", "available_at", "exceed"]].dropna(subset=["sample_date", "available_at"])
    dates = pd.Series(pd.to_datetime(df["sample_date"].drop_duplicates()).sort_values().to_numpy())
    vals = []
    for d in dates:
        revealed = events[events["available_at"] <= d]
        recent = revealed[revealed["sample_date"] >= (d - pd.Timedelta(days=7))]
        vals.append(float(recent["exceed"].mean()) if not recent.empty else np.nan)
    sw = pd.DataFrame({"sample_date": dates, "sw_prev7": vals})
    df = df.merge(sw, on="sample_date", how="left")
    return _ensure_optional_feature_columns(df)


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


def run(obs_path: Path, clf=None, rain_dir=None, discharge_dir=None, cdip_dir=None,
        tide_stages_dir=None, ocean_dir=None, air_dir=None, reveal_lag_days: int = 0,
        label: str = "any", availability_mode: str = "fixed_lag") -> dict:
    """Run the benchmark. ``clf`` is an injectable sklearn classifier (a test seam);
    when None, the production HistGradientBoosting config is used. ``rain_dir`` adds
    causal rainfall features + the AB411 wet-weather rule baseline + a Virtual-Beach-class
    per-site MLR baseline; ``discharge_dir`` adds first-flush features; ``cdip_dir``
    adds SOTA wave features (height, period, direction); ``tide_stages_dir`` adds
    NOAA CO-OPS water level features. ``reveal_lag_days`` enforces the lab-reveal lag
    (0 = legacy). ``label`` selects the target: "enterococcus" (the AB411 marine standard,
    recommended) or "any" (legacy multi-analyte OR)."""
    if availability_mode not in {"fixed_lag", "available_at"}:
        raise ValueError("availability_mode must be 'fixed_lag' or 'available_at'")
    emit_stage("loading labels", obs_path=str(obs_path), label=label, availability_mode=availability_mode)
    df = load_timed_station_days(obs_path, label=label) if availability_mode == "available_at" else load_station_days(obs_path, label=label)
    emit_stage("loading labels", "done", rows=len(df), events=int(df["exceed"].sum()))
    emit_stage("building causal features", reveal_lag_days=reveal_lag_days, availability_mode=availability_mode)
    if availability_mode == "available_at":
        df = add_causal_features_available(df, reveal_lag_days=reveal_lag_days)
    else:
        df = add_causal_features(df, reveal_lag_days=reveal_lag_days)
    emit_stage("building causal features", "done", rows=len(df))
    df["region"] = np.where(
        df["county"] == "San Diego", "SAN_DIEGO",
        np.where(df["county"].isin(MONTEREY_COUNTIES), "MONTEREY", "OTHER"),
    )

    feats = list(FEATS)
    has_rain = False
    if rain_dir is not None:
        emit_stage("joining rainfall", rain_dir=str(rain_dir))
        df = add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += RAIN_FEATS
            has_rain = True
        emit_stage("joining rainfall", "done", enabled=has_rain, rows_with_rain=int(df.get("rain_3d", pd.Series(dtype=float)).notna().sum()))
    has_discharge = False
    if discharge_dir is not None:
        emit_stage("joining discharge", discharge_dir=str(discharge_dir))
        df = add_discharge_features(df, Path(discharge_dir))
        if "discharge_3d" in df.columns and df["discharge_3d"].notna().any():
            feats += DISCHARGE_FEATS
            has_discharge = True
        emit_stage("joining discharge", "done", enabled=has_discharge)
    has_cdip = False
    if cdip_dir is not None:
        emit_stage("joining waves", cdip_dir=str(cdip_dir))
        df = add_wave_features(df, Path(cdip_dir))
        if "Hs" in df.columns and df["Hs"].notna().any():
            feats += WAVE_FEATS
            has_cdip = True
        emit_stage("joining waves", "done", enabled=has_cdip)
    has_tide = False
    if tide_stages_dir is not None:
        emit_stage("joining tide", tide_stages_dir=str(tide_stages_dir))
        df = add_tide_stage_features(df, Path(tide_stages_dir))
        if "water_level_m" in df.columns and df["water_level_m"].notna().any():
            feats += TIDE_STAGES_FEATS
            has_tide = True
        emit_stage("joining tide", "done", enabled=has_tide)
    has_ocean = False
    if ocean_dir is not None:
        emit_stage("joining oceanography", ocean_dir=str(ocean_dir))
        df = add_oceanography_features(df, Path(ocean_dir))
        if "sst_c_prev1d" in df.columns and df["sst_c_prev1d"].notna().any():
            feats += ["sst_c_prev1d", "sst_c_roll7_prev"]
            has_ocean = True
        emit_stage("joining oceanography", "done", enabled=has_ocean)
    has_air = False
    if air_dir is not None:
        emit_stage("joining air temperature", air_dir=str(air_dir))
        df = add_air_temp_features(df, Path(air_dir))
        if "air_temp_prev1d" in df.columns and df["air_temp_prev1d"].notna().any():
            feats += AIR_TEMP_FEATS
            has_air = True
        emit_stage("joining air temperature", "done", enabled=has_air)
            
    tr = df[df["sample_date"] <= "2019-12-31"]
    
    # HGBT binning crashes if a feature has fewer than 2 distinct values in the training set.
    # Filter features based on training-set coverage (protects against missing historical data).
    valid_feats = []
    for c in feats:
        if tr[c].notna().sum() >= 2 and tr[c].nunique(dropna=True) > 1:
            valid_feats.append(c)
    feats = valid_feats
    emit_stage("feature gate", "done", feature_count=len(feats), train_rows=len(tr))

    te = df[df["sample_date"] >= "2022-01-01"].copy()
    base = float(tr["exceed"].mean())

    if clf is None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )
    emit_stage("training HGBT", train_rows=len(tr), test_rows=len(te), features=len(feats))
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    te["p_model"] = clf.predict_proba(te[feats].astype(float))[:, 1]
    emit_stage("training HGBT", "done")

    # Isotonic calibration fit on the 2020-21 validation era -> the deploy-readiness
    # question: can the model's RANKING lift be turned into TRUSTWORTHY probabilities
    # for an advisory threshold? (The raw class_weight='balanced' scores are inflated.)
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te["p_model_cal"] = te["p_model"]
    if va["exceed"].nunique() > 1:
        emit_stage("calibrating", validation_rows=len(va))
        # Manual isotonic map (score -> empirical exceedance) fit on the validation era;
        # version-robust vs CalibratedClassifierCV's changing prefit API.
        from sklearn.isotonic import IsotonicRegression

        va_scores = clf.predict_proba(va[feats].astype(float))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(va_scores, va["exceed"].to_numpy())
        te["p_model_cal"] = iso.predict(te["p_model"].to_numpy())
        emit_stage("calibrating", "done")
    else:
        emit_stage("calibrating", "skipped", reason="validation era has one class")

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
        "obs_path": portable_obs_path(obs_path),
        "availability_mode": availability_mode,
        "reveal_lag_days": int(reveal_lag_days),
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
        "cdip_waves": (
            "EVALUATED - SOTA causal CDIP wave features (Hs, Tp, Dm) joined via "
            "geographic proximity to nearest buoy." if has_cdip else "NOT EVALUATED - pass --cdip-dir."
        ),
        "tide_stages": (
            "EVALUATED - NOAA CO-OPS water level features (daily average mLLW) joined via "
            "haversine distance to nearest tide gauge." if has_tide else "NOT EVALUATED - pass --tide-stages-dir."
        ),
        "oceanography": (
            "EVALUATED - HF Radar currents (nearest lat/lon) and MUR SST (regional) included."
            if has_ocean else "NOT EVALUATED - pass --ocean-dir."
        ),
        "strata": {},
    }
    for sname, part in strata.items():
        emit_stage("evaluating stratum", stratum=sname, rows=len(part), events=int(part["exceed"].sum()))
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
        emit_stage("evaluating stratum", "done", stratum=sname, verdict=verdict)
    return out


def to_markdown(res: dict) -> str:
    lines = ["# Operational beach-bacteria benchmark (honest, stratified, calibrated)", ""]
    lines.append(f"- availability mode: {res.get('availability_mode', 'fixed_lag')} | reveal lag days: {res.get('reveal_lag_days', 0)}")
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
    ap.add_argument("--obs", default=None, help="statewide_beach_observations.parquet (or $MBAL_BACTERIA_OBS)")
    ap.add_argument("--out-dir", default=None, help="where to write JSON+MD (default: reports/operational_benchmark)")
    ap.add_argument("--rain-dir", default=None,
                    help="dir with rainfall_grid.parquet + station_grid_map.parquet (adds AB411 baseline + rain feats)")
    ap.add_argument("--discharge-dir", default=None,
                    help="dir with discharge_gauge.parquet + station_gauge_map.parquet (adds first-flush feats)")
    ap.add_argument("--cdip-dir", default=None,
                    help="dir with cdip_waves.parquet + station_map.parquet (adds wave feats)")
    ap.add_argument("--tide-stages-dir", default=None,
                    help="dir with tide_stages.parquet + station_mapping.parquet (adds water level feat)")
    ap.add_argument("--ocean-dir", default=None,
                    help="dir with hf_radar/ and mur_sst/ (adds current/sst feats)")
    ap.add_argument("--air-dir", default=None,
                    help="curated root with open_meteo_archive/ (adds gated ERA5 air-temp feats)")
    ap.add_argument("--reveal-lag-days", type=int, default=0,
                    help="enforce lab-result reveal lag on prior-lab/memory features (0 = legacy)")
    ap.add_argument("--availability-mode", choices=["fixed_lag", "available_at"], default="fixed_lag",
                    help="fixed_lag uses sample_date - reveal_lag_days; available_at uses timed label record availability")
    ap.add_argument("--timed", default=None,
                    help="timed accepted_training_labels parquet; shortcut for --availability-mode available_at")
    ap.add_argument("--label", default="any", choices=["any", "enterococcus"],
                    help="target: 'enterococcus' (AB411 marine standard, recommended) or 'any' (legacy)")
    ap.add_argument("--events-jsonl", default=None,
                    help="write structured stage events to this JSONL file")
    args = ap.parse_args(argv)
    set_events_jsonl(args.events_jsonl)
    if args.timed:
        args.availability_mode = "available_at"
        obs = Path(args.timed)
    else:
        obs = Path(args.obs) if args.obs else default_obs_path()
    if not obs.exists():
        emit_stage("benchmark", "failed", reason=f"obs not found: {obs}")
        print(f"[operational_benchmark] obs not found: {obs}")
        return 2
    emit_stage("benchmark", label=args.label, reveal_lag_days=args.reveal_lag_days,
               availability_mode=args.availability_mode)
    res = run(obs, rain_dir=args.rain_dir, discharge_dir=args.discharge_dir,
              cdip_dir=args.cdip_dir, tide_stages_dir=args.tide_stages_dir,
              ocean_dir=args.ocean_dir, air_dir=args.air_dir, reveal_lag_days=args.reveal_lag_days,
              label=args.label, availability_mode=args.availability_mode)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[2] / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_stage("writing evidence", out_dir=str(out_dir))
    (out_dir / "operational_benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out_dir / "operational_benchmark.md").write_text(to_markdown(res), encoding="utf-8")
    emit_stage("writing evidence", "done", json=str(out_dir / "operational_benchmark.json"), md=str(out_dir / "operational_benchmark.md"))
    emit_stage("benchmark", "done")
    print(to_markdown(res))
    print(f"\nwrote {out_dir/'operational_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
