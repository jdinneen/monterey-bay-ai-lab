#!/usr/bin/env python3
"""
Lovers Point bacteria exceedance predictor.

Goal: predict whether the next Lovers Point beach sample will exceed California
single-sample bacteria standards, using only prior lab history plus already-known
or prior environmental drivers. This is a decision-support model, not a public
health advisory engine.

This is the archived single-site precursor to the statewide bacteria headline
(see ``operational_benchmark.py`` and ``reproduce/PAPER_DRAFT.md``); the 15-site
Monterey-only panel was too small to validate, which motivated the statewide pivot.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTDIR = ROOT / "bacteria_results" / "lovers_point"
CACHE_CSV = OUTDIR / "lovers_point_beachwatch.csv"
MONTEREY_CACHE_CSV = OUTDIR / "monterey_county_beachwatch.csv"
ADVISORY_CACHE_CSV = OUTDIR / "monterey_advisories.csv"
RAIN_CACHE_CSV = OUTDIR / "asos_rainfall.csv"
CDIP_WAVE_CACHE_CSV = OUTDIR / "cdip_158_wave.csv"
STORM_DRAIN_CACHE_CSV = OUTDIR / "lovers_storm_drain_wqp.csv"
DRIVERS = ROOT / "nn_cache" / "drivers_hourly.parquet"

# Set MBAL_GCP_PROJECT; no specific cloud project is baked into the published source.
PROJECT = os.environ.get("MBAL_GCP_PROJECT", "your-gcp-project")
TABLE = f"`{PROJECT}.blue_current_core_v2.california_beach_sample_observations`"
RAIN_STATIONS = {
    "MRY": "monterey_airport",
    "SNS": "salinas_airport",
    "WVI": "watsonville_airport",
}
CDIP_STATION = "158"

# California ocean-water single sample limits per 100 mL:
# enterococcus 104, fecal coliform 400, total coliform 10,000, or total coliform
# >1,000 when fecal/total ratio exceeds 0.1.
THRESHOLDS = {
    "enterococcus": 104.0,
    "fecal_coliform": 400.0,
    "total_coliform": 10000.0,
    "total_coliform_ratio_gate": 1000.0,
    "fecal_total_ratio": 0.1,
}


def _run_bq_query_client(sql: str) -> pd.DataFrame:
    """Primary path: google-cloud-bigquery python client. No 100k row cap, native
    types preserved (no CSV round-trip). Raises ImportError when the client library
    is absent so the caller can fall back to the bq CLI."""
    from google.cloud import bigquery  # local import so CLI-only envs still work
    client = bigquery.Client(project=os.environ.get("MBAL_GCP_PROJECT"))
    rows = client.query(" ".join(sql.split())).result()
    try:
        # bqstorage is optional; degrades to a warning + tabledata.list when
        # google-cloud-bigquery-storage is not installed. Some versions raise
        # instead of warning, so retry the dataframe conversion without it.
        return rows.to_dataframe(create_bqstorage_client=True)
    except Exception:
        return rows.to_dataframe(create_bqstorage_client=False)


def _run_bq_query_cli(sql: str) -> pd.DataFrame:
    """Fallback path: shell out to the `bq` CLI (capped at 100k rows, CSV typed)."""
    bq = shutil.which("bq.cmd") if os.name == "nt" else shutil.which("bq")
    bq = bq or shutil.which("bq")
    if not bq:
        raise RuntimeError("bq CLI not found. Install Google Cloud SDK or provide --cache-only data.")
    sql = " ".join(sql.split())
    cmd = [bq, "query", "--use_legacy_sql=false", "--format=csv", "--max_rows=100000", sql]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return pd.read_csv(io.StringIO(result.stdout))


def run_bq_query(sql: str) -> pd.DataFrame:
    """Run a read-only query. Prefer the python client (no row cap, native types);
    fall back to the bq CLI when the client library is missing or its query fails."""
    try:
        return _run_bq_query_client(sql)
    except ImportError:
        return _run_bq_query_cli(sql)
    except Exception:
        # Auth/runtime failure in the client path: retry via the CLI so CLI-only
        # environments (and gcloud-authenticated shells) still work.
        return _run_bq_query_cli(sql)


def fetch_lovers_point(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if CACHE_CSV.exists() and not force:
        return pd.read_csv(CACHE_CSV, parse_dates=["sample_date"])

    sql = f"""
    SELECT
      sample_observation_id,
      sample_date,
      sample_datetime,
      beach_name,
      station_name,
      station_id,
      source_parameter,
      property_id,
      result_comparator,
      result_value_numeric,
      weather,
      tidalheight,
      surfheight,
      turbidity,
      stormdrainflow,
      watercolor,
      odor
    FROM {TABLE}
    WHERE beach_name = "Lover's Point"
      AND station_name = "LOP"
      AND result_value_numeric IS NOT NULL
      AND property_id IN (
        'prop_enterococcus',
        'prop_total_coliform',
        'prop_fecal_coliform',
        'prop_e_coli'
      )
    ORDER BY sample_date, property_id
    """
    df = run_bq_query(sql)
    df.to_csv(CACHE_CSV, index=False)
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    return df


def fetch_monterey_county(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if MONTEREY_CACHE_CSV.exists() and not force:
        return pd.read_csv(MONTEREY_CACHE_CSV, parse_dates=["sample_date"])

    sql = f"""
    SELECT
      sample_date,
      beach_name,
      station_name,
      station_id,
      source_parameter,
      property_id,
      result_comparator,
      result_value_numeric
    FROM {TABLE}
    WHERE county = 'Monterey'
      AND result_value_numeric IS NOT NULL
      AND property_id IN (
        'prop_enterococcus',
        'prop_total_coliform',
        'prop_fecal_coliform',
        'prop_e_coli'
      )
    ORDER BY sample_date, beach_name, station_name, property_id
    """
    df = run_bq_query(sql)
    df.to_csv(MONTEREY_CACHE_CSV, index=False)
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    return df


def fetch_advisories(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if ADVISORY_CACHE_CSV.exists() and not force:
        return pd.read_csv(ADVISORY_CACHE_CSV, parse_dates=["advisory_date", "opened_date"])

    sql = f"""
    SELECT
      advisory_date,
      opened_date,
      beach_name,
      station_name,
      advisory_type,
      advisory_cause,
      enterococcus_trigger_raw,
      fecal_coliforms_trigger_raw,
      total_coliforms_trigger_raw,
      e_coli_trigger_raw,
      ratio_trigger_raw
    FROM `{PROJECT}.blue_current_core_v2.california_beach_advisory_events`
    WHERE county = 'Monterey'
    ORDER BY advisory_date, beach_name, station_name
    """
    df = run_bq_query(sql)
    df.to_csv(ADVISORY_CACHE_CSV, index=False)
    for col in ["advisory_date", "opened_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_asos_rainfall(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if RAIN_CACHE_CSV.exists() and not force:
        return pd.read_csv(RAIN_CACHE_CSV, parse_dates=["valid"])

    frames = []
    for station in RAIN_STATIONS:
        params = {
            "station": station,
            "data": "p01i",
            "year1": 2010,
            "month1": 1,
            "day1": 1,
            "year2": 2026,
            "month2": 12,
            "day2": 31,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "elev": "no",
            "missing": "M",
            "trace": "T",
            "direct": "no",
            # Routine hourly observations avoid double-counting 5-minute special reports.
            "report_type": 3,
        }
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(params)
        part = pd.read_csv(url)
        if not part.empty:
            frames.append(part)
    if not frames:
        raise RuntimeError("No ASOS rainfall rows were returned.")

    rain = pd.concat(frames, ignore_index=True)
    rain["valid"] = pd.to_datetime(rain["valid"], errors="coerce", utc=True)
    rain = rain.dropna(subset=["valid"])
    rain["p01i"] = rain["p01i"].replace({"T": 0.002, "M": np.nan})
    rain["p01i"] = pd.to_numeric(rain["p01i"], errors="coerce").clip(lower=0)
    rain.to_csv(RAIN_CACHE_CSV, index=False)
    return rain


def fetch_cdip_wave(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if CDIP_WAVE_CACHE_CSV.exists() and not force:
        return pd.read_csv(CDIP_WAVE_CACHE_CSV, parse_dates=["waveTime"])

    url = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{CDIP_STATION}p1/{CDIP_STATION}p1_historic.nc"
    ds = xr.open_dataset(url)
    cols = ["waveHs", "waveTp", "waveTa", "waveDp", "waveTz"]
    wave = ds[cols].to_dataframe().reset_index()
    ds.close()
    wave = wave.dropna(subset=["waveTime"])
    wave = wave.rename(columns={
        "waveHs": "cdip158_wave_hs_m",
        "waveTp": "cdip158_wave_tp_s",
        "waveTa": "cdip158_wave_ta_s",
        "waveDp": "cdip158_wave_dp_deg",
        "waveTz": "cdip158_wave_tz_s",
    })
    for col in [c for c in wave.columns if c.startswith("cdip158_")]:
        wave[col] = pd.to_numeric(wave[col], errors="coerce")
    wave.to_csv(CDIP_WAVE_CACHE_CSV, index=False)
    return wave


def fetch_storm_drain_wqp(force: bool = False) -> pd.DataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if STORM_DRAIN_CACHE_CSV.exists() and not force:
        return pd.read_csv(STORM_DRAIN_CACHE_CSV, parse_dates=["ActivityStartDate"])
    url = "https://www.waterqualitydata.us/data/Result/search?siteid=CEDEN-309-PGSD-03&mimeType=csv"
    df = pd.read_csv(url, low_memory=False)
    df.to_csv(STORM_DRAIN_CACHE_CSV, index=False)
    df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
    return df


def daily_lab_frame(raw: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "prop_enterococcus": "enterococcus",
        "prop_total_coliform": "total_coliform",
        "prop_fecal_coliform": "fecal_coliform",
        "prop_e_coli": "e_coli",
    }
    df = raw.copy()
    df["analyte"] = df["property_id"].map(mapping)
    df = df[df["analyte"].notna()].copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"]).dt.date
    daily = (
        df.pivot_table(
            index="sample_date",
            columns="analyte",
            values="result_value_numeric",
            aggfunc="max",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    daily["sample_date"] = pd.to_datetime(daily["sample_date"])
    daily = daily.sort_values("sample_date").reset_index(drop=True)
    for col in ["enterococcus", "total_coliform", "fecal_coliform", "e_coli"]:
        if col not in daily.columns:
            daily[col] = np.nan

    ratio = daily["fecal_coliform"] / daily["total_coliform"]
    daily["exceed_enterococcus"] = daily["enterococcus"] > THRESHOLDS["enterococcus"]
    daily["exceed_fecal_coliform"] = daily["fecal_coliform"] > THRESHOLDS["fecal_coliform"]
    daily["exceed_total_coliform"] = daily["total_coliform"] > THRESHOLDS["total_coliform"]
    daily["exceed_total_ratio"] = (
        (daily["total_coliform"] > THRESHOLDS["total_coliform_ratio_gate"])
        & (ratio > THRESHOLDS["fecal_total_ratio"])
    )
    daily["exceed_any"] = daily[
        [
            "exceed_enterococcus",
            "exceed_fecal_coliform",
            "exceed_total_coliform",
            "exceed_total_ratio",
        ]
    ].any(axis=1)
    return daily


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["sample_date"].dt.dayofyear.astype(float)
    month = out["sample_date"].dt.month.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["year"] = out["sample_date"].dt.year
    return out


def add_lab_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["days_since_prev_sample"] = out["sample_date"].diff().dt.days
    analytes = ["enterococcus", "total_coliform", "fecal_coliform", "e_coli"]
    for col in analytes:
        x = np.log1p(out[col].clip(lower=0))
        out[f"{col}_prev"] = x.shift(1)
        out[f"{col}_prev2"] = x.shift(2)
        out[f"{col}_roll3_mean"] = x.shift(1).rolling(3, min_periods=1).mean()
        out[f"{col}_roll3_max"] = x.shift(1).rolling(3, min_periods=1).max()
        out[f"{col}_roll8_mean"] = x.shift(1).rolling(8, min_periods=2).mean()
        out[f"{col}_roll8_max"] = x.shift(1).rolling(8, min_periods=2).max()
    y_prev = out["exceed_any"].astype(float).shift(1)
    out["prev_exceed"] = y_prev
    out["exceed_roll3_rate"] = y_prev.rolling(3, min_periods=1).mean()
    out["exceed_roll8_rate"] = y_prev.rolling(8, min_periods=2).mean()
    return out


def add_driver_features(df: pd.DataFrame, drivers_path: Path = DRIVERS) -> pd.DataFrame:
    if not drivers_path.exists():
        return df
    drv = pd.read_parquet(drivers_path)
    if "ds" not in drv.columns:
        drv = drv.reset_index().rename(columns={drv.index.name or "index": "ds"})
    drv["ds"] = pd.to_datetime(drv["ds"], utc=True, errors="coerce")
    drv = drv.dropna(subset=["ds"]).set_index("ds").sort_index()

    wanted = [
        "tide_M2_cos",
        "tide_M2_sin",
        "tide_S2_cos",
        "tide_S2_sin",
        "tide_K1_cos",
        "tide_K1_sin",
        "tide_O1_cos",
        "tide_O1_sin",
        "solar_elev_deg",
        "is_daylight",
        "ndbc46042_wind_speed_ms",
        "ndbc46042_wind_u_ms",
        "ndbc46042_wind_v_ms",
        "ndbc46042_wind_alongshore_ms",
        "ndbc46042_wind_stress_pa",
        "ndbc46042_wind_stress_alongshore_pa",
        "ndbc46042_pressure_mb",
        "ndbc46042_air_temp_c",
        "ndbc46042_water_temp_c",
        "coops_water_temp_c",
        "coops_air_pressure_mb",
        "coops_wind_speed_ms",
        "coops_wind_alongshore_ms",
        "cuti_37n",
        "beuti_37n",
    ]
    wanted = [c for c in wanted if c in drv.columns]
    if not wanted:
        return df

    numeric = drv[wanted].apply(pd.to_numeric, errors="coerce")
    observed_cols = [
        c for c in numeric.columns
        if c.startswith(("ndbc", "coops", "cuti", "beuti"))
    ]
    years = numeric.index.year
    for col in observed_cols:
        for year in pd.Index(years).unique():
            mask = years == year
            if mask.sum() >= 24 * 30 and numeric.loc[mask, col].nunique(dropna=True) <= 2:
                numeric.loc[mask, col] = np.nan
    daily_mean = numeric.resample("1D").mean().shift(1)
    daily_max = numeric.resample("1D").max().shift(1)
    daily_min = numeric.resample("1D").min().shift(1)
    feats = daily_mean.add_suffix("_prev1d_mean").join(
        [daily_max.add_suffix("_prev1d_max"), daily_min.add_suffix("_prev1d_min")]
    )
    for window in [3, 7]:
        roll = numeric.resample("1D").mean().shift(1).rolling(window, min_periods=1).mean()
        feats = feats.join(roll.add_suffix(f"_prev{window}d_mean"))
    feats = feats.reset_index().rename(columns={"ds": "sample_date"})
    feats["sample_date"] = feats["sample_date"].dt.tz_localize(None).dt.normalize()
    return df.merge(feats, on="sample_date", how="left")


def add_rainfall_features(df: pd.DataFrame, rainfall: pd.DataFrame | None) -> pd.DataFrame:
    if rainfall is None or rainfall.empty:
        return df
    rain = rainfall.copy()
    if "valid" not in rain.columns or "p01i" not in rain.columns or "station" not in rain.columns:
        return df
    rain["valid"] = pd.to_datetime(rain["valid"], errors="coerce", utc=True)
    rain = rain.dropna(subset=["valid"])
    rain["p01i"] = rain["p01i"].replace({"T": 0.002, "M": np.nan})
    rain["p01i"] = pd.to_numeric(rain["p01i"], errors="coerce").clip(lower=0)
    rain["rain_mm"] = rain["p01i"] * 25.4
    if rain.empty:
        return df

    min_d = pd.to_datetime(df["sample_date"]).min().normalize()
    max_d = pd.to_datetime(df["sample_date"]).max().normalize()
    full_idx = pd.date_range(min_d, max_d, freq="1D")
    feature_frames = []
    for station, label in RAIN_STATIONS.items():
        s = (
            rain[rain["station"] == station]
            .set_index("valid")["rain_mm"]
            .sort_index()
            .resample("1D")
            .sum(min_count=1)
        )
        s.index = s.index.tz_localize(None).normalize()
        s = s.reindex(full_idx).fillna(0)
        shifted = s.shift(1)
        part = pd.DataFrame({"sample_date": full_idx})
        part[f"rain_{label}_prev1d_mm"] = shifted.to_numpy()
        for window in [2, 3, 7, 14]:
            roll = shifted.rolling(window, min_periods=1)
            part[f"rain_{label}_prev{window}d_mm"] = roll.sum().to_numpy()
            part[f"rain_{label}_prev{window}d_max_daily_mm"] = roll.max().to_numpy()
            part[f"rain_{label}_wet_days_prev{window}d"] = roll.apply(lambda x: float((x >= 2.54).sum()), raw=False).to_numpy()
        days_since = []
        last_wet = None
        for d, amount in s.items():
            days_since.append(np.nan if last_wet is None else (d - last_wet).days)
            if amount >= 2.54:
                last_wet = d
        part[f"rain_{label}_days_since_wet_day"] = days_since
        part[f"rain_{label}_first_flush_prev3d"] = (
            part[f"rain_{label}_prev3d_mm"] * (part[f"rain_{label}_days_since_wet_day"].fillna(99) >= 7).astype(float)
        )
        feature_frames.append(part)

    feats = feature_frames[0]
    for part in feature_frames[1:]:
        feats = feats.merge(part, on="sample_date", how="outer")
    rain_cols = [c for c in feats.columns if c != "sample_date"]
    feats["rain_regional_prev3d_mm_max"] = feats[
        [c for c in rain_cols if c.endswith("_prev3d_mm")]
    ].max(axis=1)
    feats["rain_regional_prev7d_mm_max"] = feats[
        [c for c in rain_cols if c.endswith("_prev7d_mm")]
    ].max(axis=1)
    feats["rain_regional_wet_station_count_prev3d"] = (
        feats[[c for c in rain_cols if c.endswith("_prev3d_mm")]] >= 2.54
    ).sum(axis=1)
    return df.merge(feats, on="sample_date", how="left")


def add_wave_features(df: pd.DataFrame, wave: pd.DataFrame | None) -> pd.DataFrame:
    if wave is None or wave.empty or "waveTime" not in wave.columns:
        return df
    w = wave.copy()
    w["waveTime"] = pd.to_datetime(w["waveTime"], errors="coerce", utc=True)
    w = w.dropna(subset=["waveTime"])
    if w.empty:
        return df
    for col in [c for c in w.columns if c.startswith("cdip158_")]:
        w[col] = pd.to_numeric(w[col], errors="coerce")
    w["cdip158_wave_energy_proxy"] = (w["cdip158_wave_hs_m"] ** 2) * w["cdip158_wave_tp_s"]
    direction_rad = np.deg2rad(w["cdip158_wave_dp_deg"])
    w["cdip158_wave_dir_sin"] = np.sin(direction_rad)
    w["cdip158_wave_dir_cos"] = np.cos(direction_rad)

    wanted = [
        "cdip158_wave_hs_m",
        "cdip158_wave_tp_s",
        "cdip158_wave_ta_s",
        "cdip158_wave_dp_deg",
        "cdip158_wave_tz_s",
        "cdip158_wave_energy_proxy",
        "cdip158_wave_dir_sin",
        "cdip158_wave_dir_cos",
    ]
    numeric = w.set_index("waveTime")[wanted].sort_index()
    min_d = pd.to_datetime(df["sample_date"]).min().normalize()
    max_d = pd.to_datetime(df["sample_date"]).max().normalize()
    full_idx = pd.date_range(min_d, max_d, freq="1D")

    daily_mean = numeric.resample("1D").mean()
    daily_max = numeric.resample("1D").max()
    daily_min = numeric.resample("1D").min()
    daily_mean.index = daily_mean.index.tz_localize(None).normalize()
    daily_max.index = daily_max.index.tz_localize(None).normalize()
    daily_min.index = daily_min.index.tz_localize(None).normalize()
    shifted_mean = daily_mean.reindex(full_idx).shift(1)
    shifted_max = daily_max.reindex(full_idx).shift(1)
    shifted_min = daily_min.reindex(full_idx).shift(1)
    feats = shifted_mean.add_suffix("_prev1d_mean").join([
        shifted_max.add_suffix("_prev1d_max"),
        shifted_min.add_suffix("_prev1d_min"),
    ])
    for window in [3, 7, 14]:
        roll = shifted_mean.rolling(window, min_periods=1)
        feats = feats.join(roll.mean().add_suffix(f"_prev{window}d_mean"))
        feats = feats.join(shifted_max.rolling(window, min_periods=1).max().add_suffix(f"_prev{window}d_max"))
    feats = feats.reset_index().rename(columns={"index": "sample_date"})
    return df.merge(feats, on="sample_date", how="left")


def fusion_quality_report(df: pd.DataFrame, mode: str, outdir: Path) -> dict:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    groups = {
        "lab_lag": [c for c in numeric_cols if any(s in c for s in ["_prev", "_roll", "days_since_prev_sample", "exceed_roll"])],
        "drivers": [c for c in numeric_cols if any(s in c for s in ["ndbc", "coops", "tide", "cuti", "beuti", "solar"])],
        "rain": [c for c in numeric_cols if c.startswith("rain_")],
        "waves": [c for c in numeric_cols if c.startswith("cdip158_")],
        "regional_or_nearby": [c for c in numeric_cols if c.startswith("regional_") or c.startswith("nearby_")],
        "advisories": [c for c in numeric_cols if "advis" in c],
    }
    coverage = {}
    for name, cols in groups.items():
        if not cols:
            coverage[name] = {"feature_count": 0, "mean_non_null_rate": None, "min_non_null_rate": None}
            continue
        rates = df[cols].notna().mean()
        coverage[name] = {
            "feature_count": int(len(cols)),
            "mean_non_null_rate": round(float(rates.mean()), 4),
            "min_non_null_rate": round(float(rates.min()), 4),
            "lowest_coverage_features": {
                k: round(float(v), 4) for k, v in rates.sort_values().head(8).items()
            },
        }
    duplicate_keys = ["sample_date"]
    if "site_key" in df.columns:
        duplicate_keys.append("site_key")
    report = {
        "mode": mode,
        "rows": int(len(df)),
        "date_range": [str(df["sample_date"].min().date()), str(df["sample_date"].max().date())],
        "duplicate_sample_keys": int(df.duplicated(duplicate_keys).sum()),
        "events": int(df["exceed_any"].sum()) if "exceed_any" in df.columns else None,
        "event_rate": round(float(df["exceed_any"].mean()), 4) if "exceed_any" in df.columns else None,
        "feature_coverage": coverage,
        "leakage_rules": {
            "lab_history": "uses shift(1)/prior rolling windows only",
            "environmental_drivers": "daily aggregates are shifted one day before merge",
            "stale_observed_drivers": "observed driver columns flat for a full year are nulled for that year before aggregation",
            "rainfall": "rain windows are shifted one day before merge",
            "waves": "CDIP wave windows are shifted one day before merge",
            "nearby_or_regional_bacteria": "nearby/regional aggregates are shifted one day before merge",
            "advisories": "advisory counts are shifted one day before merge",
            "feature_availability": "model fitting keeps only features with at least 50% non-null coverage in train, validation, and holdout partitions",
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fusion_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _daily_exceed_by_group(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    mapping = {
        "prop_enterococcus": "enterococcus",
        "prop_total_coliform": "total_coliform",
        "prop_fecal_coliform": "fecal_coliform",
        "prop_e_coli": "e_coli",
    }
    df = raw.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"]).dt.normalize()
    df["analyte"] = df["property_id"].map(mapping)
    df = df[df["analyte"].notna()].copy()
    piv = (
        df.pivot_table(
            index=["sample_date", *group_cols],
            columns="analyte",
            values="result_value_numeric",
            aggfunc="max",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["enterococcus", "total_coliform", "fecal_coliform", "e_coli"]:
        if col not in piv.columns:
            piv[col] = np.nan
    ratio = piv["fecal_coliform"] / piv["total_coliform"]
    piv["exceed_any"] = (
        (piv["enterococcus"] > THRESHOLDS["enterococcus"])
        | (piv["fecal_coliform"] > THRESHOLDS["fecal_coliform"])
        | (piv["total_coliform"] > THRESHOLDS["total_coliform"])
        | (
            (piv["total_coliform"] > THRESHOLDS["total_coliform_ratio_gate"])
            & (ratio > THRESHOLDS["fecal_total_ratio"])
        )
    )
    return piv


def add_nearby_bacteria_features(df: pd.DataFrame, raw_county: pd.DataFrame) -> pd.DataFrame:
    county = _daily_exceed_by_group(raw_county, ["beach_name", "station_name"])
    is_lovers = (county["beach_name"] == "Lover's Point") & (county["station_name"] == "LOP")
    nearby = county[~is_lovers].copy()
    if nearby.empty:
        return df
    daily = nearby.groupby("sample_date").agg(
        nearby_sample_sites=("station_name", "nunique"),
        nearby_exceed_sites=("exceed_any", "sum"),
        nearby_enterococcus_max=("enterococcus", "max"),
        nearby_total_coliform_max=("total_coliform", "max"),
        nearby_fecal_coliform_max=("fecal_coliform", "max"),
    ).sort_index()
    full_idx = pd.date_range(df["sample_date"].min(), df["sample_date"].max(), freq="1D")
    daily = daily.reindex(full_idx)
    daily.index.name = "sample_date"
    base_cols = list(daily.columns)
    feature_parts = []
    for window in [1, 3, 7, 14, 30]:
        shifted = daily.shift(1).rolling(window, min_periods=1)
        part = {}
        for col in base_cols:
            part[f"{col}_prev{window}d_sum"] = shifted[col].sum()
            part[f"{col}_prev{window}d_max"] = shifted[col].max()
        feature_parts.append(pd.DataFrame(part, index=daily.index))
    feats = pd.concat(feature_parts, axis=1).reset_index()
    return df.merge(feats, on="sample_date", how="left")


def add_advisory_features(df: pd.DataFrame, advisories: pd.DataFrame) -> pd.DataFrame:
    if advisories.empty or "advisory_date" not in advisories.columns:
        return df
    adv = advisories.copy()
    adv["advisory_date"] = pd.to_datetime(adv["advisory_date"], errors="coerce").dt.normalize()
    adv = adv.dropna(subset=["advisory_date"])
    if adv.empty:
        return df
    adv["is_lovers"] = (adv["beach_name"] == "Lover's Point") | (adv["station_name"] == "LOP")
    daily = adv.groupby("advisory_date").agg(
        monterey_advisory_count=("advisory_type", "size"),
        lovers_advisory_count=("is_lovers", "sum"),
    ).sort_index()
    full_idx = pd.date_range(df["sample_date"].min(), df["sample_date"].max(), freq="1D")
    daily = daily.reindex(full_idx).fillna(0)
    daily.index.name = "sample_date"
    for window in [7, 30, 90]:
        shifted = daily.shift(1).rolling(window, min_periods=1)
        daily[f"monterey_advisories_prev{window}d"] = shifted["monterey_advisory_count"].sum()
        daily[f"lovers_advisories_prev{window}d"] = shifted["lovers_advisory_count"].sum()
    last_lovers = None
    days_since = []
    lovers_daily = daily["lovers_advisory_count"]
    for d, count in lovers_daily.items():
        days_since.append(np.nan if last_lovers is None else (d - last_lovers).days)
        if count > 0:
            last_lovers = d
    daily["days_since_lovers_advisory"] = days_since
    keep = [c for c in daily.columns if "_prev" in c or c == "days_since_lovers_advisory"]
    return df.merge(daily[keep].reset_index(), on="sample_date", how="left")


def add_advisory_features_regional(df: pd.DataFrame, advisories: pd.DataFrame) -> pd.DataFrame:
    if advisories.empty or "advisory_date" not in advisories.columns:
        return df
    adv = advisories.copy()
    adv["advisory_date"] = pd.to_datetime(adv["advisory_date"], errors="coerce").dt.normalize()
    adv = adv.dropna(subset=["advisory_date"])
    if adv.empty:
        return df
    adv["site_key"] = adv["beach_name"].fillna("") + "|" + adv["station_name"].fillna("")
    adv_daily = adv.groupby(["advisory_date", "site_key"]).size().rename("site_advisory_count").reset_index()
    frames = []
    for site, g in df[["site_key", "sample_date"]].drop_duplicates().groupby("site_key"):
        min_d, max_d = g["sample_date"].min(), g["sample_date"].max()
        idx = pd.date_range(min_d, max_d, freq="1D")
        s = (
            adv_daily[adv_daily["site_key"] == site]
            .set_index("advisory_date")["site_advisory_count"]
            .reindex(idx)
            .fillna(0)
        )
        part = pd.DataFrame({"sample_date": idx, "site_key": site})
        for window in [30, 90]:
            part[f"site_advisories_prev{window}d"] = s.shift(1).rolling(window, min_periods=1).sum().to_numpy()
        frames.append(part)
    if not frames:
        return df
    site_feats = pd.concat(frames, ignore_index=True)

    regional = adv.groupby("advisory_date").size().rename("regional_advisory_count").sort_index()
    full_idx = pd.date_range(df["sample_date"].min(), df["sample_date"].max(), freq="1D")
    regional = regional.reindex(full_idx).fillna(0)
    reg_feats = pd.DataFrame({"sample_date": full_idx})
    for window in [7, 30, 90]:
        reg_feats[f"regional_advisories_prev{window}d"] = regional.shift(1).rolling(window, min_periods=1).sum().to_numpy()
    return df.merge(site_feats, on=["sample_date", "site_key"], how="left").merge(reg_feats, on="sample_date", how="left")


def build_features(raw: pd.DataFrame, raw_county: pd.DataFrame | None = None,
                   advisories: pd.DataFrame | None = None,
                   rainfall: pd.DataFrame | None = None,
                   wave: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = daily_lab_frame(raw)
    daily = add_calendar_features(daily)
    daily = add_lab_lag_features(daily)
    daily = add_driver_features(daily)
    daily = add_rainfall_features(daily, rainfall)
    daily = add_wave_features(daily, wave)
    if raw_county is not None:
        daily = add_nearby_bacteria_features(daily, raw_county)
    if advisories is not None:
        daily = add_advisory_features(daily, advisories)
    return daily


def build_regional_features(raw_county: pd.DataFrame, advisories: pd.DataFrame | None = None,
                            rainfall: pd.DataFrame | None = None,
                            wave: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = _daily_exceed_by_group(raw_county, ["beach_name", "station_name", "station_id"])
    daily["site_key"] = daily["beach_name"].fillna("") + "|" + daily["station_name"].fillna("")
    daily = daily.sort_values(["site_key", "sample_date"]).reset_index(drop=True)
    daily = add_calendar_features(daily)

    frames = []
    for _, g in daily.groupby("site_key", sort=False):
        frames.append(add_lab_lag_features(g.copy()))
    out = pd.concat(frames, ignore_index=True).sort_values(["sample_date", "site_key"]).reset_index(drop=True)

    # Prior regional bacteria context. The current site's same-day value is excluded,
    # then all aggregates are shifted by one calendar day to avoid leakage.
    regional_daily = daily.groupby("sample_date").agg(
        regional_sample_sites=("site_key", "nunique"),
        regional_exceed_sites=("exceed_any", "sum"),
        regional_enterococcus_max=("enterococcus", "max"),
        regional_total_coliform_max=("total_coliform", "max"),
        regional_fecal_coliform_max=("fecal_coliform", "max"),
    ).sort_index()
    idx = pd.date_range(daily["sample_date"].min(), daily["sample_date"].max(), freq="1D")
    regional_daily = regional_daily.reindex(idx)
    regional_daily.index.name = "sample_date"
    parts = []
    base_cols = list(regional_daily.columns)
    for window in [1, 3, 7, 14, 30]:
        roll = regional_daily.shift(1).rolling(window, min_periods=1)
        part = {}
        for col in base_cols:
            part[f"{col}_prev{window}d_sum"] = roll[col].sum()
            part[f"{col}_prev{window}d_max"] = roll[col].max()
        parts.append(pd.DataFrame(part, index=regional_daily.index))
    reg_feats = pd.concat(parts, axis=1).reset_index()
    out = out.merge(reg_feats, on="sample_date", how="left")
    out = add_driver_features(out)
    out = add_rainfall_features(out, rainfall)
    out = add_wave_features(out, wave)
    if advisories is not None:
        out = add_advisory_features_regional(out, advisories)
    out["is_lovers_point"] = ((out["beach_name"] == "Lover's Point") & (out["station_name"] == "LOP")).astype(int)
    return out


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = df.sort_values("sample_date").reset_index(drop=True)
    train_end = pd.Timestamp("2021-12-31")
    valid_end = pd.Timestamp("2023-12-31")
    train = rows[rows["sample_date"] <= train_end]
    valid = rows[(rows["sample_date"] > train_end) & (rows["sample_date"] <= valid_end)]
    test = rows[rows["sample_date"] > valid_end]
    if len(test) < 40 or test["exceed_any"].sum() < 3:
        n = len(rows)
        train = rows.iloc[: int(n * 0.65)]
        valid = rows.iloc[int(n * 0.65): int(n * 0.82)]
        test = rows.iloc[int(n * 0.82):]
    return train, valid, test


def choose_threshold(y_true: np.ndarray, score: np.ndarray, min_recall: float = 0.70) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall:
            candidates.append((p, r, t))
    if not candidates:
        return float(np.quantile(score, 0.85))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return float(candidates[0][2])


def oversample_minority(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(bool)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    if len(pos) >= len(neg):
        return X, y
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def production_safe_features(feature_cols: list[str], partitions: list[pd.DataFrame],
                             min_coverage: float = 0.50) -> tuple[list[str], list[str]]:
    keep = []
    dropped = []
    for col in feature_cols:
        rates = [part[col].notna().mean() for part in partitions if col in part.columns]
        if rates and min(rates) >= min_coverage:
            keep.append(col)
        else:
            dropped.append(col)
    return keep, dropped


def metric_block(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    cm = confusion_matrix(y_true, pred, labels=[False, True])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out = {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": round(float(np.mean(y_true)), 4),
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else None,
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, score)), 4)
        out["avg_precision"] = round(float(average_precision_score(y_true, score)), 4)
    return out


def train_and_eval(df: pd.DataFrame) -> dict:
    target_cols = {
        "enterococcus",
        "total_coliform",
        "fecal_coliform",
        "e_coli",
        "exceed_enterococcus",
        "exceed_fecal_coliform",
        "exceed_total_coliform",
        "exceed_total_ratio",
        "exceed_any",
    }
    ignore = {"sample_date"} | target_cols
    feature_cols = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]

    train, valid, test = split_by_time(df)
    feature_cols, dropped_features = production_safe_features(feature_cols, [train, valid, test])
    train_valid = pd.concat([train, valid], ignore_index=True)
    y_train = train["exceed_any"].astype(bool).to_numpy()
    y_valid = valid["exceed_any"].astype(bool).to_numpy()
    y_train_valid = train_valid["exceed_any"].astype(bool).to_numpy()
    y_test = test["exceed_any"].astype(bool).to_numpy()

    logistic = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    neural_net = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 24),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=40,
            random_state=42,
        )),
    ])

    candidates = {"logistic": logistic, "random_forest": forest, "neural_net_mlp": neural_net}
    validation = {}
    for name, model in candidates.items():
        fit_X, fit_y = train[feature_cols], y_train
        if name == "neural_net_mlp":
            fit_X, fit_y = oversample_minority(fit_X, fit_y)
        model.fit(fit_X, fit_y)
        score = model.predict_proba(valid[feature_cols])[:, 1]
        threshold = choose_threshold(y_valid, score)
        validation[name] = metric_block(y_valid, score, threshold)
        validation[name]["threshold"] = threshold

    chosen_name = max(
        validation,
        key=lambda k: (
            validation[k].get("avg_precision", -1),
            validation[k].get("recall") or 0,
        ),
    )
    threshold = float(validation[chosen_name]["threshold"])
    test_candidates = {}
    for name, candidate in candidates.items():
        fit_X_c, fit_y_c = train_valid[feature_cols], y_train_valid
        if name == "neural_net_mlp":
            fit_X_c, fit_y_c = oversample_minority(fit_X_c, fit_y_c)
        candidate.fit(fit_X_c, fit_y_c)
        score_c = candidate.predict_proba(test[feature_cols])[:, 1]
        test_candidates[name] = metric_block(y_test, score_c, float(validation[name]["threshold"]))

    model = candidates[chosen_name]
    fit_X, fit_y = train_valid[feature_cols], y_train_valid
    if chosen_name == "neural_net_mlp":
        fit_X, fit_y = oversample_minority(fit_X, fit_y)
    model.fit(fit_X, fit_y)
    test_score = model.predict_proba(test[feature_cols])[:, 1]

    pred = test[["sample_date", "enterococcus", "total_coliform", "fecal_coliform", "e_coli", "exceed_any"]].copy()
    pred["risk_score"] = test_score
    pred["predicted_exceed"] = pred["risk_score"] >= threshold
    pred["risk_band"] = pd.cut(
        pred["risk_score"],
        bins=[-math.inf, threshold * 0.5, threshold, 1.0],
        labels=["low", "watch", "high"],
    )

    metrics = {
        "target": "any California single-sample bacteria exceedance",
        "thresholds": THRESHOLDS,
        "rows": {
            "all_sample_days": int(len(df)),
            "all_events": int(df["exceed_any"].sum()),
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
        },
        "date_ranges": {
            "all": [str(df["sample_date"].min().date()), str(df["sample_date"].max().date())],
            "train": [str(train["sample_date"].min().date()), str(train["sample_date"].max().date())],
            "valid": [str(valid["sample_date"].min().date()), str(valid["sample_date"].max().date())],
            "test": [str(test["sample_date"].min().date()), str(test["sample_date"].max().date())],
        },
        "validation": validation,
        "chosen_model": chosen_name,
        "chosen_threshold": round(threshold, 4),
        "test": metric_block(y_test, test_score, threshold),
        "test_candidates": test_candidates,
        "feature_count": len(feature_cols),
        "dropped_unavailable_feature_count": len(dropped_features),
        "dropped_unavailable_features_sample": dropped_features[:25],
        "signal_groups": {
            "prior_labs": len([c for c in feature_cols if "coliform" in c or "enterococcus" in c or "e_coli" in c]),
            "ocean_weather_drivers": len([c for c in feature_cols if "ndbc" in c or "coops" in c or "tide" in c or "cuti" in c or "beuti" in c or "solar" in c]),
            "rainfall_runoff_proxy": len([c for c in feature_cols if c.startswith("rain_")]),
            "wave_transport": len([c for c in feature_cols if c.startswith("cdip158_")]),
            "nearby_beaches": len([c for c in feature_cols if c.startswith("nearby_")]),
            "advisories": len([c for c in feature_cols if "advis" in c]),
        },
    }

    if chosen_name == "logistic":
        coefs = model.named_steps["model"].coef_[0]
        importance = pd.DataFrame({"feature": feature_cols, "importance": np.abs(coefs), "coef": coefs})
    elif chosen_name == "random_forest":
        imps = model.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"feature": feature_cols, "importance": imps})
    else:
        importance = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    importance = importance.sort_values("importance", ascending=False)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / "training_frame.csv", index=False)
    pred.to_csv(OUTDIR / "test_predictions.csv", index=False)
    importance.to_csv(OUTDIR / "feature_importance.csv", index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fusion_quality_report(df, "lovers_point", OUTDIR)
    return metrics


def train_and_eval_regional(df: pd.DataFrame) -> dict:
    target_cols = {
        "enterococcus",
        "total_coliform",
        "fecal_coliform",
        "e_coli",
        "exceed_enterococcus",
        "exceed_fecal_coliform",
        "exceed_total_coliform",
        "exceed_total_ratio",
        "exceed_any",
    }
    ignore = {"sample_date", "beach_name", "station_name", "station_id", "site_key"} | target_cols
    feature_cols = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    rows = df.sort_values(["sample_date", "site_key"]).reset_index(drop=True)
    lovers = (rows["beach_name"] == "Lover's Point") & (rows["station_name"] == "LOP")
    train = rows[rows["sample_date"] <= pd.Timestamp("2021-12-31")]
    valid = rows[(rows["sample_date"] > pd.Timestamp("2021-12-31")) & (rows["sample_date"] <= pd.Timestamp("2023-12-31"))]
    test = rows[(rows["sample_date"] > pd.Timestamp("2023-12-31")) & lovers]
    feature_cols, dropped_features = production_safe_features(feature_cols, [train, valid, test])

    y_train = train["exceed_any"].astype(bool).to_numpy()
    y_valid = valid["exceed_any"].astype(bool).to_numpy()
    train_valid = pd.concat([train, valid], ignore_index=True)
    y_train_valid = train_valid["exceed_any"].astype(bool).to_numpy()
    y_test = test["exceed_any"].astype(bool).to_numpy()

    logistic = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    neural_net = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(96, 32),
            activation="relu",
            alpha=0.02,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=40,
            random_state=42,
        )),
    ])
    candidates = {"logistic": logistic, "random_forest": forest, "neural_net_mlp": neural_net}
    validation = {}
    for name, model in candidates.items():
        fit_X, fit_y = train[feature_cols], y_train
        if name == "neural_net_mlp":
            fit_X, fit_y = oversample_minority(fit_X, fit_y)
        model.fit(fit_X, fit_y)
        score = model.predict_proba(valid[feature_cols])[:, 1]
        threshold = choose_threshold(y_valid, score)
        validation[name] = metric_block(y_valid, score, threshold)
        validation[name]["threshold"] = threshold

    chosen_name = max(
        validation,
        key=lambda k: (
            validation[k].get("avg_precision", -1),
            validation[k].get("recall") or 0,
        ),
    )
    test_candidates = {}
    test_scores = {}
    for name, model in candidates.items():
        fit_X, fit_y = train_valid[feature_cols], y_train_valid
        if name == "neural_net_mlp":
            fit_X, fit_y = oversample_minority(fit_X, fit_y)
        model.fit(fit_X, fit_y)
        score = model.predict_proba(test[feature_cols])[:, 1]
        test_scores[name] = score
        test_candidates[name] = metric_block(y_test, score, float(validation[name]["threshold"]))

    threshold = float(validation[chosen_name]["threshold"])
    test_score = test_scores[chosen_name]
    pred = test[["sample_date", "beach_name", "station_name", "enterococcus", "total_coliform", "fecal_coliform", "e_coli", "exceed_any"]].copy()
    pred["risk_score"] = test_score
    pred["predicted_exceed"] = pred["risk_score"] >= threshold
    pred["risk_band"] = pd.cut(
        pred["risk_score"],
        bins=[-math.inf, threshold * 0.5, threshold, 1.0],
        labels=["low", "watch", "high"],
    )

    chosen_model = candidates[chosen_name]
    if chosen_name == "logistic":
        coefs = chosen_model.named_steps["model"].coef_[0]
        importance = pd.DataFrame({"feature": feature_cols, "importance": np.abs(coefs), "coef": coefs})
    elif chosen_name == "random_forest":
        imps = chosen_model.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"feature": feature_cols, "importance": imps})
    else:
        importance = pd.DataFrame({"feature": feature_cols, "importance": np.nan})
    importance = importance.sort_values("importance", ascending=False)

    metrics = {
        "mode": "regional_train_lovers_point_holdout",
        "target": "any California single-sample bacteria exceedance",
        "thresholds": THRESHOLDS,
        "rows": {
            "all_sample_site_days": int(len(rows)),
            "all_events": int(rows["exceed_any"].sum()),
            "sites": int(rows["site_key"].nunique()),
            "lovers_sample_days": int(lovers.sum()),
            "lovers_events": int(rows.loc[lovers, "exceed_any"].sum()),
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test_lovers": int(len(test)),
        },
        "date_ranges": {
            "all": [str(rows["sample_date"].min().date()), str(rows["sample_date"].max().date())],
            "train": [str(train["sample_date"].min().date()), str(train["sample_date"].max().date())],
            "valid": [str(valid["sample_date"].min().date()), str(valid["sample_date"].max().date())],
            "test_lovers": [str(test["sample_date"].min().date()), str(test["sample_date"].max().date())],
        },
        "validation_regional": validation,
        "chosen_model": chosen_name,
        "chosen_threshold": round(threshold, 4),
        "test_lovers": metric_block(y_test, test_score, threshold),
        "test_lovers_candidates": test_candidates,
        "feature_count": len(feature_cols),
        "dropped_unavailable_feature_count": len(dropped_features),
        "dropped_unavailable_features_sample": dropped_features[:25],
        "signal_groups": {
            "site_prior_labs": len([c for c in feature_cols if "coliform" in c or "enterococcus" in c or "e_coli" in c]),
            "ocean_weather_drivers": len([c for c in feature_cols if "ndbc" in c or "coops" in c or "tide" in c or "cuti" in c or "beuti" in c or "solar" in c]),
            "rainfall_runoff_proxy": len([c for c in feature_cols if c.startswith("rain_")]),
            "wave_transport": len([c for c in feature_cols if c.startswith("cdip158_")]),
            "regional_beaches": len([c for c in feature_cols if c.startswith("regional_")]),
            "advisories": len([c for c in feature_cols if "advis" in c]),
        },
    }

    regional_dir = OUTDIR / "regional_train"
    regional_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(regional_dir / "training_frame.csv", index=False)
    pred.to_csv(regional_dir / "lovers_point_test_predictions.csv", index=False)
    importance.to_csv(regional_dir / "feature_importance.csv", index=False)
    (regional_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fusion_quality_report(df, "regional_train_lovers_point_holdout", regional_dir)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-query BigQuery instead of using cached CSV")
    ap.add_argument("--cache-only", action="store_true", help="fail if cached Lovers Point CSV is absent")
    ap.add_argument("--no-extra-signals", action="store_true",
                    help="only use Lovers Point lab history and local driver cache")
    ap.add_argument("--no-wave", action="store_true",
                    help="disable CDIP wave features while keeping other extra signals")
    ap.add_argument("--regional-train", action="store_true",
                    help="train on all Monterey beach sites, evaluate on Lovers Point holdout")
    args = ap.parse_args()

    if args.cache_only and not CACHE_CSV.exists():
        raise SystemExit(f"cache not found: {CACHE_CSV}")
    raw = pd.read_csv(CACHE_CSV, parse_dates=["sample_date"]) if args.cache_only else fetch_lovers_point(args.refresh)
    raw_county = None
    advisories = None
    rainfall = None
    wave = None
    if not args.no_extra_signals:
        if args.cache_only:
            raw_county = pd.read_csv(MONTEREY_CACHE_CSV, parse_dates=["sample_date"]) if MONTEREY_CACHE_CSV.exists() else None
            advisories = pd.read_csv(ADVISORY_CACHE_CSV, parse_dates=["advisory_date", "opened_date"]) if ADVISORY_CACHE_CSV.exists() else None
            rainfall = pd.read_csv(RAIN_CACHE_CSV, parse_dates=["valid"]) if RAIN_CACHE_CSV.exists() else None
            wave = pd.read_csv(CDIP_WAVE_CACHE_CSV, parse_dates=["waveTime"]) if CDIP_WAVE_CACHE_CSV.exists() and not args.no_wave else None
        else:
            raw_county = fetch_monterey_county(args.refresh)
            advisories = fetch_advisories(args.refresh)
            rainfall = fetch_asos_rainfall(args.refresh)
            wave = None if args.no_wave else fetch_cdip_wave(args.refresh)
            fetch_storm_drain_wqp(args.refresh)
    if args.regional_train:
        if raw_county is None:
            raise SystemExit("regional training requires Monterey County cache; rerun without --cache-only or omit --no-extra-signals")
        frame = build_regional_features(raw_county, advisories=advisories, rainfall=rainfall, wave=wave)
        metrics = train_and_eval_regional(frame)
    else:
        frame = build_features(raw, raw_county=raw_county, advisories=advisories, rainfall=rainfall, wave=wave)
        metrics = train_and_eval(frame)
    print(json.dumps(metrics, indent=2))
    print(f"wrote: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
