#!/usr/bin/env python3
"""Broad leakage-aware analysis across all local MBARI-related data assets.

This is intentionally an exploratory analysis, not a promotion gate. It inventories
available data, aligns sources where the time contracts make sense, and screens
driver signal against M1 forecast targets without using future observations as
features.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mbari_forecast_v2 import build_hourly_matrix, load_source
from mbari_core import apply_physical_quality_filters

PROJECT_ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = PROJECT_ROOT / "mbari_big_analysis_results"

M1 = PROJECT_ROOT / "mbari_history" / "opendap" / "m1_history.parquet"
M2 = PROJECT_ROOT / "mbari_history" / "opendap" / "m2_history.parquet"
DRIVERS = PROJECT_ROOT / "nn_cache" / "drivers_hourly.parquet"
DRIVERS_MANIFEST = PROJECT_ROOT / "nn_cache" / "drivers_manifest.json"
NOAA_DAILY = PROJECT_ROOT / "mbari_history" / "noaa" / "noaa_drivers_daily.parquet"
NDBC = PROJECT_ROOT / "mbari_history" / "noaa" / "noaa_ndbc46042.parquet"
COOPS = PROJECT_ROOT / "mbari_history" / "noaa" / "noaa_coops.parquet"
UPWELLING = PROJECT_ROOT / "mbari_history" / "noaa" / "noaa_upwelling.parquet"

BACTERIA_DIR = PROJECT_ROOT / "bacteria_results" / "lovers_point"

FOCUS_TARGETS = [
    "temp_d1p0",
    "temp_d10p0",
    "temp_d20p0",
    "temp_d100p0",
    "sal_d1p0",
    "sal_d10p0",
    "sal_d20p0",
    "sal_d100p0",
    "air_temperature",
    "air_pressure",
]
HORIZONS = [6, 24, 72, 168]


def _time_summary(df: pd.DataFrame, time_col: str | None = None) -> dict[str, Any]:
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    else:
        t = pd.to_datetime(df.index, errors="coerce", utc=True)
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "time_min": str(t.min()) if t.notna().any() else None,
        "time_max": str(t.max()) if t.notna().any() else None,
    }


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
    return df


def _asset_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    assets = [
        ("m1_history", M1, "parquet", "M1 mooring observations; primary forecast source"),
        ("m2_history", M2, "parquet", "M2 mooring observations; not yet in main forecast-v2"),
        ("drivers_hourly", DRIVERS, "parquet", "canonical leakage-classed hourly driver table"),
        ("noaa_daily", NOAA_DAILY, "parquet", "daily NOAA/NDBC/CO-OPS/upwelling aggregate"),
        ("noaa_ndbc46042", NDBC, "parquet", "offshore buoy observations including wave fields"),
        ("noaa_coops", COOPS, "parquet", "Monterey harbor met/ocean observations"),
        ("noaa_upwelling", UPWELLING, "parquet", "CUTI/BEUTI daily upwelling indices"),
        ("cdip_158_wave", BACTERIA_DIR / "cdip_158_wave.csv", "csv", "nearshore wave station used by bacteria experiment"),
        ("asos_rainfall", BACTERIA_DIR / "asos_rainfall.csv", "csv", "airport rainfall/runoff proxies"),
        ("storm_drain_wqp", BACTERIA_DIR / "lovers_storm_drain_wqp.csv", "csv", "Lovers Point storm-drain public records"),
        ("monterey_beachwatch", BACTERIA_DIR / "monterey_county_beachwatch.csv", "csv", "regional beach bacteria history"),
        ("monterey_advisories", BACTERIA_DIR / "monterey_advisories.csv", "csv", "beach advisory history"),
    ]
    for name, path, kind, note in assets:
        row = {"asset": name, "path": str(path.relative_to(PROJECT_ROOT)), "exists": path.exists(), "kind": kind, "note": note}
        if path.exists():
            row["size_mb"] = round(path.stat().st_size / 1024 / 1024, 3)
            try:
                if kind == "parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                time_col = next(
                    (
                        c
                        for c in [
                            "time",
                            "ds",
                            "date",
                            "sample_date",
                            "valid",
                            "waveTime",
                            "ActivityStartDate",
                            "advisory_date",
                            "opened_date",
                        ]
                        if c in df.columns
                    ),
                    None,
                )
                row.update(_time_summary(df, time_col))
                row["numeric_columns"] = int(len(df.select_dtypes(include="number").columns))
            except Exception as exc:
                row["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_for(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_source("parquet", str(path), None)
    clean, qc = apply_physical_quality_filters(raw)
    matrix, coverage = build_hourly_matrix(clean)
    return raw, matrix, coverage


def _coverage_compare(mats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station, matrix in mats.items():
        for col in matrix.columns:
            if not col.startswith(("temp_d", "sal_d", "air_", "wind_", "current", "relative", "eastward", "northward")):
                continue
            s = matrix[col]
            rows.append(
                {
                    "station": station,
                    "series": col,
                    "non_null_hours": int(s.notna().sum()),
                    "coverage_pct": round(float(s.notna().mean() * 100), 3),
                    "first_time": str(s.first_valid_index()),
                    "last_time": str(s.last_valid_index()),
                }
            )
    return pd.DataFrame(rows).sort_values(["station", "non_null_hours"], ascending=[True, False])


def _m1_m2_overlap(m1: pd.DataFrame, m2: pd.DataFrame) -> pd.DataFrame:
    common_cols = sorted(set(m1.columns) & set(m2.columns))
    rows: list[dict[str, Any]] = []
    common_idx = m1.index.intersection(m2.index)
    for col in common_cols:
        if not col.startswith(("temp_d", "sal_d", "air_temperature", "air_pressure")):
            continue
        a = m1.loc[common_idx, col]
        b = m2.loc[common_idx, col]
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 500:
            continue
        rows.append(
            {
                "series": col,
                "overlap_hours": int(mask.sum()),
                "m1_mean": float(a[mask].mean()),
                "m2_mean": float(b[mask].mean()),
                "m2_minus_m1_mean": float((b[mask] - a[mask]).mean()),
                "corr": float(a[mask].corr(b[mask])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["series", "overlap_hours", "m1_mean", "m2_mean", "m2_minus_m1_mean", "corr"]
        )
    return out.sort_values("overlap_hours", ascending=False)


def _driver_screen(m1: pd.DataFrame, drivers: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    drivers = drivers.copy()
    drivers.index = pd.to_datetime(drivers.index, utc=True).floor("h")
    drivers = drivers[~drivers.index.duplicated(keep="last")].sort_index()
    aligned = m1.join(drivers, how="inner", rsuffix="_drv")

    driver_cols = [c for c in drivers.columns if c in aligned.columns]
    hist_cols = set(manifest.get("hist", []))
    futr_cols = set(manifest.get("futr", []))
    rows: list[dict[str, Any]] = []
    for target in [c for c in FOCUS_TARGETS if c in aligned.columns]:
        y0 = aligned[target]
        for h in HORIZONS:
            future = y0.shift(-h)
            delta = future - y0
            for driver in driver_cols:
                x = aligned[driver]
                mask = x.notna() & future.notna() & y0.notna()
                n = int(mask.sum())
                if n < 5000:
                    continue
                xv = x[mask]
                fv = future[mask]
                dv = delta[mask]
                corr_future = xv.corr(fv)
                corr_delta = xv.corr(dv)
                if pd.isna(corr_future) and pd.isna(corr_delta):
                    continue
                rows.append(
                    {
                        "target": target,
                        "horizon_h": h,
                        "driver": driver,
                        "driver_class": "hist" if driver in hist_cols else "futr" if driver in futr_cols else "unknown",
                        "n": n,
                        "corr_driver_to_future_y": float(corr_future) if not pd.isna(corr_future) else np.nan,
                        "corr_driver_to_future_delta": float(corr_delta) if not pd.isna(corr_delta) else np.nan,
                        "abs_corr_delta": float(abs(corr_delta)) if not pd.isna(corr_delta) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_corr_delta", ascending=False)


def _wave_driver_summary(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_hist = set(manifest.get("hist", []))
    manifest_futr = set(manifest.get("futr", []))
    for path, prefix in [(NDBC, "ndbc46042"), (NOAA_DAILY, "daily")]:
        df = _read_parquet(path)
        if df is None:
            continue
        wave_cols = [c for c in df.columns if "wave" in c.lower()]
        for col in wave_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            rows.append(
                {
                    "asset": path.name,
                    "column": col,
                    "non_null": int(s.notna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "std": float(s.std()) if s.notna().any() else np.nan,
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                    "status": "hist_driver_manifested"
                    if col in manifest_hist
                    else "futr_driver_manifested"
                    if col in manifest_futr
                    else "available_not_in_canonical_driver_manifest",
                }
            )
    return pd.DataFrame(rows)


def _bacteria_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_files = [
        BACTERIA_DIR / "metrics.json",
        BACTERIA_DIR / "wave_fusion" / "metrics.json",
        BACTERIA_DIR / "regional_train" / "metrics.json",
    ]
    for path in metric_files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        test = data.get("test") or data.get("test_lovers") or {}
        rows.append(
            {
                "run": str(path.relative_to(PROJECT_ROOT)),
                "chosen_model": data.get("chosen_model"),
                "test_n": test.get("n"),
                "test_events": test.get("events"),
                "tp": test.get("tp"),
                "fp": test.get("fp"),
                "tn": test.get("tn"),
                "fn": test.get("fn"),
                "recall": test.get("recall"),
                "precision": test.get("precision"),
                "roc_auc": test.get("roc_auc"),
                "avg_precision": test.get("avg_precision"),
                "feature_count": data.get("feature_count"),
            }
        )
    return pd.DataFrame(rows)


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# MBARI Big All-Data Analysis",
        "",
        "This report uses all local MBARI-related assets that can be read safely. It is exploratory: it screens signal and coverage, but it does not promote a model.",
        "",
        "## Inputs",
        "",
    ]
    inv = pd.read_csv(OUT / "asset_inventory.csv")
    for _, r in inv.iterrows():
        state = "present" if bool(r.get("exists")) else "missing"
        rows = "" if pd.isna(r.get("rows")) else f", rows={int(r['rows']):,}"
        tmin = "" if pd.isna(r.get("time_min")) else f", {r['time_min']} -> {r['time_max']}"
        lines.append(f"- `{r['asset']}`: {state}{rows}{tmin}")

    lines += [
        "",
        "## What Is Newly Usable",
        "",
        "- M2 history is present and profiled, but should be treated as a separate station/domain rather than pooled blindly with M1.",
        "- Canonical hourly drivers are present with leakage classes: deterministic future regressors plus observed-only historical regressors.",
        "- NDBC wave fields are checked against the canonical driver manifest and can be isolated with the `ndbc_wave` ablation manifest; the 2026-06-08 bounded rerun did not support a general wave-driver claim.",
        "- Bacteria-side wave/rain/storm-drain/regional beach data is useful for the Lovers Point risk-screen workflow, not directly for M1 water-column forecasts without a new target contract.",
        "",
        "## Driver Signal Screen",
        "",
    ]
    ds = pd.read_csv(OUT / "driver_signal_screen.csv") if (OUT / "driver_signal_screen.csv").exists() else pd.DataFrame()
    if ds.empty:
        lines.append("No driver rows met the minimum overlap threshold.")
    else:
        top = ds.head(25).copy()
        lines += [
            "| target | h | driver | class | n | corr driver->future delta |",
            "|---|---:|---|---|---:|---:|",
        ]
        for _, r in top.iterrows():
            lines.append(
                f"| `{r['target']}` | {int(r['horizon_h'])} | `{r['driver']}` | {r['driver_class']} | "
                f"{int(r['n']):,} | {r['corr_driver_to_future_delta']:.3f} |"
            )

    lines += [
        "",
        "## M1/M2 Overlap",
        "",
    ]
    ov = pd.read_csv(OUT / "m1_m2_overlap.csv") if (OUT / "m1_m2_overlap.csv").exists() else pd.DataFrame()
    if ov.empty:
        lines.append("No M1/M2 common series had enough overlapping hourly observations.")
    else:
        lines += [
            "| series | overlap hours | corr | M2-M1 mean |",
            "|---|---:|---:|---:|",
        ]
        for _, r in ov.head(20).iterrows():
            lines.append(f"| `{r['series']}` | {int(r['overlap_hours']):,} | {r['corr']:.3f} | {r['m2_minus_m1_mean']:.3f} |")

    lines += [
        "",
        "## Bacteria Workflow",
        "",
    ]
    bact = pd.read_csv(OUT / "bacteria_model_summary.csv") if (OUT / "bacteria_model_summary.csv").exists() else pd.DataFrame()
    if bact.empty:
        lines.append("No bacteria metrics were found.")
    else:
        lines += [
            "| run | model | test events | TP | FP | FN | recall | precision | AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in bact.iterrows():
            lines.append(
                f"| `{r['run']}` | `{r['chosen_model']}` | {int(r['test_events'])} | {int(r['tp'])} | "
                f"{int(r['fp'])} | {int(r['fn'])} | {r['recall']:.3f} | {r['precision']:.3f} | {r['roc_auc']:.3f} |"
            )

    lines += [
        "",
        "## Caveats",
        "",
        "- Driver correlations are not model skill; they identify candidates for shared-split reruns.",
        "- Observed-only drivers must remain historical exogenous inputs, not known-future inputs.",
        "- M2 is spatially distinct and older; pooling it with M1 needs a station feature or transfer-learning design.",
        "- NDBC hourly waves are manifest-backed historical drivers, but the current bounded ablation verdict is research-only / not supported as a general driver claim.",
        "- CDIP and daily aggregated wave assets remain outside the canonical forecast driver contract.",
        "",
        "## Artifacts",
        "",
        "- `asset_inventory.csv`",
        "- `station_series_coverage.csv`",
        "- `m1_m2_overlap.csv`",
        "- `driver_signal_screen.csv`",
        "- `wave_fields_available.csv`",
        "- `bacteria_model_summary.csv`",
        "- `summary.json`",
    ]
    (OUT / "BIG_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    inventory = _asset_inventory()
    inventory.to_csv(OUT / "asset_inventory.csv", index=False)

    raw_m1, m1_matrix, m1_cov = _matrix_for(M1)
    raw_m2, m2_matrix, m2_cov = _matrix_for(M2)
    m1_matrix.to_parquet(OUT / "m1_hourly_matrix.parquet")
    m2_matrix.to_parquet(OUT / "m2_hourly_matrix.parquet")
    station_cov = _coverage_compare({"M1": m1_matrix, "M2": m2_matrix})
    station_cov.to_csv(OUT / "station_series_coverage.csv", index=False)

    overlap = _m1_m2_overlap(m1_matrix, m2_matrix)
    overlap.to_csv(OUT / "m1_m2_overlap.csv", index=False)

    manifest = json.loads(DRIVERS_MANIFEST.read_text(encoding="utf-8")) if DRIVERS_MANIFEST.exists() else {}
    drivers = pd.read_parquet(DRIVERS) if DRIVERS.exists() else pd.DataFrame()
    driver_screen = _driver_screen(m1_matrix, drivers, manifest) if not drivers.empty else pd.DataFrame()
    driver_screen.to_csv(OUT / "driver_signal_screen.csv", index=False)

    wave = _wave_driver_summary(manifest)
    wave.to_csv(OUT / "wave_fields_available.csv", index=False)

    bacteria = _bacteria_summary()
    bacteria.to_csv(OUT / "bacteria_model_summary.csv", index=False)

    summary = {
        "m1_raw_rows": int(len(raw_m1)),
        "m2_raw_rows": int(len(raw_m2)),
        "m1_matrix_shape": list(m1_matrix.shape),
        "m2_matrix_shape": list(m2_matrix.shape),
        "asset_count": int(len(inventory)),
        "present_asset_count": int(inventory["exists"].sum()),
        "driver_signal_rows": int(len(driver_screen)),
        "top_driver_signals": driver_screen.head(20).to_dict(orient="records") if not driver_screen.empty else [],
        "m1_m2_overlap_rows": int(len(overlap)),
        "wave_field_count": int(len(wave)),
        "bacteria_runs": int(len(bacteria)),
    }
    _write_report(summary)
    print(f"Wrote big analysis to {OUT}")
    print(f"Report: {OUT / 'BIG_ANALYSIS_REPORT.md'}")


if __name__ == "__main__":
    main()
