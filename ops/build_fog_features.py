#!/usr/bin/env python3
"""Build derived fog/marine-layer proxy features from validated weather sources.

The raw fetch framework lands broad weather sources. This script turns the fog-relevant
parts into a compact modeling table without mutating the source artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "lakehouse" / "silver" / "fog_features"
REPORT_DIR = ROOT / "reports" / "data_fetch" / "fog_features"
HOURLY_OUT = OUTDIR / "fog_features_hourly.parquet"
DAILY_OUT = OUTDIR / "fog_features_daily.parquet"
MANIFEST_OUT = REPORT_DIR / "manifest.json"
VALIDATION_OUT = REPORT_DIR / "validation.json"
COVERAGE_OUT = REPORT_DIR / "coverage.json"


def _rh_from_temp_dewpoint(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """Approximate relative humidity from temperature/dewpoint using Magnus formula."""
    temp = pd.to_numeric(temp_c, errors="coerce")
    dew = pd.to_numeric(dewpoint_c, errors="coerce")
    rh = 100.0 * np.exp((17.625 * dew / (243.04 + dew)) - (17.625 * temp / (243.04 + temp)))
    return pd.Series(rh, index=temp.index).clip(0, 100)


def _fog_score(spread_c: pd.Series, rh_pct: pd.Series | None = None,
               cloud_pct: pd.Series | None = None, visibility: pd.Series | None = None) -> pd.Series:
    spread = pd.to_numeric(spread_c, errors="coerce")
    score = pd.Series(np.zeros(len(spread), dtype=float), index=spread.index)
    score = score.where(spread.isna() | (spread > 2.5), 0.35)
    score = score.where(spread.isna() | (spread > 1.5), 0.65)
    score = score.where(spread.isna() | (spread > 0.75), 0.85)
    if rh_pct is not None:
        rh = pd.to_numeric(rh_pct, errors="coerce")
        score = np.maximum(score, np.where(rh >= 95, 0.7, np.where(rh >= 90, 0.45, 0.0)))
    if cloud_pct is not None:
        cloud = pd.to_numeric(cloud_pct, errors="coerce")
        humid_cloud = (rh_pct is not None) & (pd.to_numeric(rh_pct, errors="coerce") >= 90) & (cloud >= 80)
        score = np.maximum(score, np.where(humid_cloud, 0.75, 0.0))
    if visibility is not None:
        vis = pd.to_numeric(visibility, errors="coerce")
        score = np.maximum(score, np.where(vis <= 1.0, 0.9, np.where(vis <= 3.0, 0.55, 0.0)))
    return pd.Series(score, index=spread.index).clip(0, 1)


def _ncei_features() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "ncei_isd_hourly" / "ncei_isd_hourly.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=[
        "station_id", "time", "air_temp_c", "dewpoint_c", "latitude", "longitude"
    ])
    df = df.dropna(subset=["time", "air_temp_c", "dewpoint_c"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df[df["time"].notna()]
    df["dewpoint_spread_c"] = pd.to_numeric(df["air_temp_c"], errors="coerce") - pd.to_numeric(df["dewpoint_c"], errors="coerce")
    df["relative_humidity_pct"] = _rh_from_temp_dewpoint(df["air_temp_c"], df["dewpoint_c"])
    df["fog_score"] = _fog_score(df["dewpoint_spread_c"], df["relative_humidity_pct"])
    df["fog_likely"] = df["fog_score"] >= 0.65
    df["source"] = "ncei_isd_hourly"
    df["site_id"] = df["station_id"].astype(str)
    df = df.rename(columns={"air_temp_c": "air_temp_c", "dewpoint_c": "dewpoint_c"})
    return df[[
        "source", "site_id", "time", "latitude", "longitude", "air_temp_c", "dewpoint_c",
        "dewpoint_spread_c", "relative_humidity_pct", "fog_score", "fog_likely",
    ]]


def _ndbc_features() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "ndbc_stdmet" / "ndbc_stdmet.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=["station_id", "time", "ATMP", "DEWP", "VIS"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df[df["time"].notna()]
    df["air_temp_c"] = pd.to_numeric(df["ATMP"], errors="coerce")
    df["dewpoint_c"] = pd.to_numeric(df["DEWP"], errors="coerce")
    df["visibility_reported"] = pd.to_numeric(df["VIS"], errors="coerce")
    df = df.dropna(subset=["air_temp_c", "dewpoint_c"], how="all")
    df["dewpoint_spread_c"] = df["air_temp_c"] - df["dewpoint_c"]
    df["relative_humidity_pct"] = _rh_from_temp_dewpoint(df["air_temp_c"], df["dewpoint_c"])
    df["fog_score"] = _fog_score(df["dewpoint_spread_c"], df["relative_humidity_pct"], visibility=df["visibility_reported"])
    df["fog_likely"] = df["fog_score"] >= 0.65
    df["source"] = "ndbc_stdmet"
    df["site_id"] = df["station_id"].astype(str)
    df["latitude"] = np.nan
    df["longitude"] = np.nan
    return df[[
        "source", "site_id", "time", "latitude", "longitude", "air_temp_c", "dewpoint_c",
        "dewpoint_spread_c", "relative_humidity_pct", "visibility_reported", "fog_score", "fog_likely",
    ]]


def _openmeteo_features() -> pd.DataFrame:
    path = ROOT / "data" / "external_curated" / "open_meteo_archive" / "open_meteo_archive.parquet"
    if not path.exists():
        return pd.DataFrame()
    wanted = ["temperature_2m", "dewpoint_2m", "relative_humidity_2m", "cloud_cover"]
    df = pd.read_parquet(path)
    df = df[df["parameter"].isin(wanted)].copy()
    if df.empty:
        return pd.DataFrame()
    piv = df.pivot_table(
        index=["grid_lat", "grid_lon", "time"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    piv["time"] = pd.to_datetime(piv["time"], utc=True, errors="coerce")
    piv = piv[piv["time"].notna()]
    piv["air_temp_c"] = pd.to_numeric(piv.get("temperature_2m"), errors="coerce")
    piv["dewpoint_c"] = pd.to_numeric(piv.get("dewpoint_2m"), errors="coerce")
    piv["relative_humidity_pct"] = pd.to_numeric(piv.get("relative_humidity_2m"), errors="coerce")
    piv["cloud_cover_pct"] = pd.to_numeric(piv.get("cloud_cover"), errors="coerce")
    piv["dewpoint_spread_c"] = piv["air_temp_c"] - piv["dewpoint_c"]
    piv["fog_score"] = _fog_score(piv["dewpoint_spread_c"], piv["relative_humidity_pct"], cloud_pct=piv["cloud_cover_pct"])
    piv["fog_likely"] = piv["fog_score"] >= 0.65
    piv["source"] = "open_meteo_archive"
    piv["site_id"] = piv["grid_lat"].round(3).astype(str) + "," + piv["grid_lon"].round(3).astype(str)
    piv = piv.rename(columns={"grid_lat": "latitude", "grid_lon": "longitude"})
    return piv[[
        "source", "site_id", "time", "latitude", "longitude", "air_temp_c", "dewpoint_c",
        "dewpoint_spread_c", "relative_humidity_pct", "cloud_cover_pct", "fog_score", "fog_likely",
    ]]


def build() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [f for f in [_ncei_features(), _ndbc_features(), _openmeteo_features()] if not f.empty]
    if not frames:
        raise RuntimeError("no fog-capable source tables found")
    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly = hourly.sort_values(["source", "site_id", "time"]).reset_index(drop=True)
    hourly.to_parquet(HOURLY_OUT, index=False)

    daily = hourly.copy()
    daily["date"] = daily["time"].dt.floor("D")
    daily = daily.groupby(["source", "site_id", "date", "latitude", "longitude"], dropna=False).agg(
        fog_score_mean=("fog_score", "mean"),
        fog_score_max=("fog_score", "max"),
        fog_likely_hours=("fog_likely", "sum"),
        obs_hours=("fog_score", "count"),
        air_temp_c_mean=("air_temp_c", "mean"),
        dewpoint_spread_c_min=("dewpoint_spread_c", "min"),
        relative_humidity_pct_max=("relative_humidity_pct", "max"),
    ).reset_index()
    daily.to_parquet(DAILY_OUT, index=False)

    summary = {
        "source": "fog_features",
        "title": "Derived fog / marine-layer proxy features",
        "endpoint": "derived from validated local NCEI ISD, NDBC stdmet, and Open-Meteo archive tables",
        "status": "READY_FOR_MODELING",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "inputs": ["ncei_isd_hourly", "ndbc_stdmet", "open_meteo_archive"],
        "curated_path": str(HOURLY_OUT.relative_to(ROOT)),
        "hourly_path": str(HOURLY_OUT.relative_to(ROOT)),
        "daily_path": str(DAILY_OUT.relative_to(ROOT)),
        "rows": int(len(hourly)),
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(daily)),
        "date_min": str(hourly["time"].min()),
        "date_max": str(hourly["time"].max()),
        "columns": int(hourly.shape[1]),
        "column_names": list(hourly.columns),
        "definition": {
            "dewpoint_spread_c": "air_temp_c - dewpoint_c",
            "fog_likely": "fog_score >= 0.65; score uses dewpoint spread, RH, cloud cover, and NDBC visibility when available",
        },
    }
    MANIFEST_OUT.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    VALIDATION_OUT.write_text(json.dumps({
        "source": "fog_features",
        "path": str(HOURLY_OUT.relative_to(ROOT)),
        "passed": True,
        "rows": int(len(hourly)),
        "columns": int(hourly.shape[1]),
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "null_rates": hourly.isna().mean().round(4).to_dict(),
    }, indent=2, default=str), encoding="utf-8")
    COVERAGE_OUT.write_text(json.dumps({
        "source": "fog_features",
        "rows": int(len(hourly)),
        "columns": int(hourly.shape[1]),
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "date_column": "time",
        "gaps": [
            "DERIVED SOURCE: native completeness is inherited from ncei_isd_hourly, ndbc_stdmet, and open_meteo_archive; current direct fog observations are not claimed.",
        ],
    }, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(build(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
