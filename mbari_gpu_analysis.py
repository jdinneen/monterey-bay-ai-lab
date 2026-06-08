#!/usr/bin/env python3
"""
Robust statistical analysis for MBARI M1 BigQuery data.

This version avoids the main failure mode in the earlier script: sparse sensor
families were combined into one dense design matrix, then every row was dropped.
It profiles the real table, deduplicates repeated ingests, models each target
with the features that exist for that target, and only claims GPU acceleration
after a real CUDA kernel executes successfully.
"""

from __future__ import annotations

import json
import math
import os
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None

warnings.filterwarnings("ignore", category=UserWarning, module="google.cloud.bigquery")

# GCP project is environment-configured (set MBARI_GCP_PROJECT) so no specific
# cloud project is baked into the published source.
PROJECT_ID = os.environ.get("MBARI_GCP_PROJECT", "your-gcp-project")
DATASET_ID = "blue_current_raw"
TABLE_ID = "mbari_m1_realtime"
OUT_DIR = Path("mbari_analysis_results")

RAW_COLUMNS = [
    "source",
    "dataset_id",
    "station",
    "time",
    "latitude",
    "longitude",
    "z",
    "air_pressure",
    "relative_humidity",
    "air_temperature",
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "wind_speed_sonic",
    "wind_from_direction_sonic",
    "wind_speed_windbird",
    "wind_from_direction_windbird",
    "air_pressure_qc_agg",
    "relative_humidity_qc_agg",
    "air_temperature_qc_agg",
    "sea_water_temperature_qc_agg",
    "sea_water_practical_salinity_qc_agg",
    "eastward_sea_water_velocity_qc_agg",
    "northward_sea_water_velocity_qc_agg",
    "wind_speed_sonic_qc_agg",
    "wind_from_direction_sonic_qc_agg",
    "ingested_at",
]

MEASURE_COLUMNS = [
    "latitude",
    "longitude",
    "z",
    "air_pressure",
    "relative_humidity",
    "air_temperature",
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
    "wind_speed_sonic",
    "wind_from_direction_sonic",
    "wind_speed_windbird",
    "wind_from_direction_windbird",
]

TARGETS = [
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "air_temperature",
    "relative_humidity",
    "air_pressure",
]

PHYSICAL_RANGES = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "z": (-11000.0, 100.0),
    "air_pressure": (870.0, 1085.0),
    "relative_humidity": (0.0, 100.0),
    "air_temperature": (-10.0, 45.0),
    "sea_water_temperature": (-3.0, 35.0),
    "sea_water_practical_salinity": (20.0, 40.0),
    "eastward_sea_water_velocity": (-5.0, 5.0),
    "northward_sea_water_velocity": (-5.0, 5.0),
    "wind_speed_sonic": (0.0, 60.0),
    "wind_from_direction_sonic": (0.0, 360.0),
    "wind_speed_windbird": (0.0, 60.0),
    "wind_from_direction_windbird": (0.0, 360.0),
}


@dataclass
class ComputeStatus:
    backend: str
    gpu_name: str | None
    cuda_available: bool
    cuda_kernel_ok: bool
    note: str


def detect_compute() -> ComputeStatus:
    if torch is None:
        return ComputeStatus("numpy/sklearn-cpu", None, False, False, "PyTorch is not importable.")

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return ComputeStatus("numpy/sklearn-cpu", None, False, False, "CUDA is not available to PyTorch.")

    gpu_name = torch.cuda.get_device_name(0)
    try:
        x = torch.randn(512, 512, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        _ = float(y[0, 0].detach().cpu())
        return ComputeStatus("torch-cuda+sklearn-cpu", gpu_name, True, True, "CUDA tensor kernels are usable.")
    except Exception as exc:
        return ComputeStatus(
            "numpy/sklearn-cpu",
            gpu_name,
            True,
            False,
            f"CUDA is visible but unusable for PyTorch kernels: {type(exc).__name__}: {exc}",
        )


def read_gbq(query: str) -> pd.DataFrame:
    try:
        import pandas_gbq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "BigQuery reads require the optional analysis/geo dependencies. "
            "Install with `python -m pip install -e .[analysis,geo]`."
        ) from exc
    return pandas_gbq.read_gbq(query, project_id=PROJECT_ID, progress_bar_type=None)


def load_profile() -> pd.DataFrame:
    select_parts = [
        "COUNT(*) AS row_count",
        "COUNT(DISTINCT CONCAT(CAST(time AS STRING), '|', CAST(z AS STRING), '|', CAST(ingested_at AS STRING))) AS distinct_time_depth_ingest",
        "COUNT(DISTINCT CONCAT(CAST(time AS STRING), '|', CAST(z AS STRING))) AS distinct_time_depth",
        "MIN(time) AS min_time",
        "MAX(time) AS max_time",
    ]
    for col in MEASURE_COLUMNS:
        select_parts.append(f"COUNTIF({col} IS NOT NULL) AS {col}_n")
    query = f"""
        SELECT {", ".join(select_parts)}
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    return read_gbq(query)


def load_mbari_data(limit: int | None = None) -> pd.DataFrame:
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        WITH latest AS (
          SELECT
            {", ".join(RAW_COLUMNS)},
            ROW_NUMBER() OVER (
              PARTITION BY time, FORMAT("%0.6f", z)
              ORDER BY ingested_at DESC
            ) AS rn
          FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
          WHERE time IS NOT NULL
        )
        SELECT {", ".join(RAW_COLUMNS)}
        FROM latest
        WHERE rn = 1
        ORDER BY time, z
        {limit_clause}
    """
    df = read_gbq(query)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    if "z" in out:
        out["depth_m"] = out["z"].abs()
    out["hour"] = out["time"].dt.hour.astype(float)
    out["day_of_year"] = out["time"].dt.dayofyear.astype(float)
    out["days_since_start"] = (out["time"] - out["time"].min()).dt.total_seconds() / 86400.0
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)
    if {"eastward_sea_water_velocity", "northward_sea_water_velocity"}.issubset(out.columns):
        out["current_speed"] = np.sqrt(
            out["eastward_sea_water_velocity"].pow(2) + out["northward_sea_water_velocity"].pow(2)
        )
    if {"wind_speed_sonic", "wind_from_direction_sonic"}.issubset(out.columns):
        out["wind_u_sonic"] = out["wind_speed_sonic"] * np.sin(np.deg2rad(out["wind_from_direction_sonic"]))
        out["wind_v_sonic"] = out["wind_speed_sonic"] * np.cos(np.deg2rad(out["wind_from_direction_sonic"]))
    return out


def apply_physical_quality_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        present = values.notna()
        invalid = present & ((values < lo) | (values > hi) | ~np.isfinite(values))
        rows.append(
            {
                "variable": col,
                "non_null_before": int(present.sum()),
                "invalidated": int(invalid.sum()),
                "valid_after": int((present & ~invalid).sum()),
                "min_allowed": lo,
                "max_allowed": hi,
            }
        )
        out.loc[invalid, col] = np.nan
    return out, pd.DataFrame(rows)


def coverage_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for depth, group in df.groupby("z", dropna=False):
        row: dict[str, Any] = {
            "z": depth,
            "row_count": len(group),
            "min_time": str(group["time"].min()),
            "max_time": str(group["time"].max()),
        }
        for col in MEASURE_COLUMNS:
            row[f"{col}_n"] = int(group[col].notna().sum()) if col in group else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["row_count", "z"], ascending=[False, True])


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MEASURE_COLUMNS + ["depth_m", "current_speed", "wind_u_sonic", "wind_v_sonic"]:
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "variable": col,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def robust_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "depth_m",
        "days_since_start",
        "hour_sin",
        "hour_cos",
        "sea_water_temperature",
        "sea_water_practical_salinity",
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "current_speed",
        "wind_speed_sonic",
    ]
    available = [c for c in cols if c in df and df[c].notna().sum() >= 100]
    corr = df[available].corr(method="spearman", min_periods=100)
    corr.to_csv(OUT_DIR / "spearman_correlation_matrix.csv")

    pairs = []
    for i, left in enumerate(available):
        for right in available[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": left, "right": right, "spearman_r": float(value), "abs_r": abs(float(value))})
    return pd.DataFrame(pairs).sort_values("abs_r", ascending=False)


# (fit_target_model removed; see mbari_forecast_v2.py)


def anomaly_detection(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    features = [
        "depth_m",
        "sea_water_temperature",
        "sea_water_practical_salinity",
        "current_speed",
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "wind_speed_sonic",
        "hour_sin",
        "hour_cos",
        "days_since_start",
    ]
    features = [c for c in features if c in df and df[c].notna().sum() >= 500]
    work = df[["time", "z"] + features].copy()
    work = work.loc[work[features].notna().sum(axis=1) >= 3].copy()
    if len(work) < 1000:
        return pd.DataFrame(), {"rows": int(len(work)), "status": "not enough rows"}

    x = SimpleImputer(strategy="median").fit_transform(work[features])
    x = StandardScaler().fit_transform(x)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(x)
    scores = -iso.decision_function(x)
    work["anomaly_score"] = scores
    work["is_anomaly"] = labels == -1
    top = work.loc[work["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(100)
    summary = {
        "rows": int(len(work)),
        "features": features,
        "anomaly_count": int(work["is_anomaly"].sum()),
        "anomaly_rate": float(work["is_anomaly"].mean()),
    }
    return top, summary


def temporal_signals(df: pd.DataFrame) -> dict[str, Any]:
    try:
        from scipy import stats
        from scipy.signal import periodogram
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Temporal signal diagnostics require scipy. "
            "Install with `python -m pip install -e .[analysis]`."
        ) from exc

    results: dict[str, Any] = {}
    for target in ["sea_water_temperature", "sea_water_practical_salinity", "air_temperature"]:
        work = df.dropna(subset=[target]).sort_values("time")
        if len(work) < 500:
            continue
        hourly = work.groupby(work["time"].dt.hour)[target].mean()
        depth_corr = np.nan
        if work["depth_m"].notna().sum() >= 100:
            depth_corr = float(stats.spearmanr(work["depth_m"], work[target], nan_policy="omit").statistic)

        hourly_series = (
            work.set_index("time")[target]
            .resample("1h")
            .mean()
            .interpolate(limit=12)
            .dropna()
            .astype(float)
        )
        dominant_period_hours = None
        if len(hourly_series) >= 72:
            freq, power = periodogram(hourly_series.values, fs=1.0)
            mask = freq > 0
            if mask.any():
                peak_freq = freq[mask][np.argmax(power[mask])]
                dominant_period_hours = float(1.0 / peak_freq) if peak_freq > 0 else None

        results[target] = {
            "rows": int(len(work)),
            "hourly_peak_hour": int(hourly.idxmax()),
            "hourly_trough_hour": int(hourly.idxmin()),
            "hourly_amplitude": float(hourly.max() - hourly.min()),
            "spearman_depth_r": depth_corr,
            "dominant_period_hours": dominant_period_hours,
        }
    return results


def write_markdown_report(
    compute: ComputeStatus,
    profile: pd.DataFrame,
    quality: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    anomalies: dict[str, Any],
    temporal: dict[str, Any],
) -> None:
    p = profile.iloc[0].to_dict()
    lines = [
        "# MBARI M1 Statistical Analysis",
        "",
        f"Generated on: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"Project/table: `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`",
        "",
        "## Compute",
        "",
        f"- Backend: `{compute.backend}`",
        f"- GPU: `{compute.gpu_name or 'none'}`",
        f"- CUDA visible: `{compute.cuda_available}`",
        f"- CUDA kernel usable: `{compute.cuda_kernel_ok}`",
        f"- Note: {compute.note}",
        "",
        "## Data Profile",
        "",
        f"- Raw rows: {int(p['row_count']):,}",
        f"- Distinct time-depth rows before latest-ingest dedupe: {int(p['distinct_time_depth']):,}",
        f"- Time range: {p['min_time']} to {p['max_time']}",
        f"- Rows after latest-ingest dedupe: {int(coverage['row_count'].sum()):,}",
        "",
        "## Quality Filters",
        "",
    ]
    invalid = quality[quality["invalidated"] > 0].sort_values("invalidated", ascending=False)
    if invalid.empty:
        lines.append("No physical-range violations were found in the configured measured fields.")
    for row in invalid.itertuples(index=False):
        lines.append(
            f"- `{row.variable}`: invalidated {row.invalidated:,} of {row.non_null_before:,} non-null values "
            f"(allowed {row.min_allowed:g} to {row.max_allowed:g})"
        )
    lines += [
        "",
        "## Strongest Spearman Relationships",
        "",
    ]
    for row in corr_pairs.head(12).itertuples(index=False):
        lines.append(f"- `{row.left}` vs `{row.right}`: r={row.spearman_r:.3f}")

    lines += ["", "## Temporal Signals", ""]
    for target, info in temporal.items():
        period = info["dominant_period_hours"]
        period_text = "n/a" if period is None or not math.isfinite(period) else f"{period:.1f} h"
        lines.append(
            f"- `{target}`: hourly amplitude={info['hourly_amplitude']:.4g}, "
            f"peak hour={info['hourly_peak_hour']}, trough hour={info['hourly_trough_hour']}, "
            f"depth Spearman r={info['spearman_depth_r']:.3f}, dominant period={period_text}"
        )

    lines += [
        "",
        "## Anomalies",
        "",
        f"- Rows scored: {anomalies.get('rows', 0):,}",
        f"- Anomalies: {anomalies.get('anomaly_count', 0):,} ({anomalies.get('anomaly_rate', 0.0) * 100:.2f}%)",
        "",
        "## Output Files",
        "",
        "- `profile.json`",
        "- `quality_filter_counts.csv`",
        "- `coverage_by_depth.csv`",
        "- `numeric_summary.csv`",
        "- `spearman_correlation_matrix.csv`",
        "- `strong_correlation_pairs.csv`",
        "- `top_anomalies.csv`",
    ]
    (OUT_DIR / "MBARI_STATISTICAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    compute = detect_compute()
    print(f"Compute backend: {compute.backend}")
    print(f"GPU: {compute.gpu_name or 'none'}")
    print(f"CUDA kernel usable: {compute.cuda_kernel_ok}")
    print(f"Note: {compute.note}")

    profile = load_profile()
    profile.to_json(OUT_DIR / "profile.json", orient="records", indent=2, date_format="iso")
    print("Loaded BigQuery profile.")

    raw_df = load_mbari_data()
    clean_df, quality = apply_physical_quality_filters(raw_df)
    quality.to_csv(OUT_DIR / "quality_filter_counts.csv", index=False)

    df = add_derived_features(clean_df)
    print(f"Loaded {len(df):,} latest-ingest time-depth rows.")

    coverage = coverage_by_depth(df)
    coverage.to_csv(OUT_DIR / "coverage_by_depth.csv", index=False)

    summary = numeric_summary(df)
    summary.to_csv(OUT_DIR / "numeric_summary.csv", index=False)

    corr_pairs = robust_correlations(df)
    corr_pairs.to_csv(OUT_DIR / "strong_correlation_pairs.csv", index=False)

    top_anomalies, anomaly_summary = anomaly_detection(df)
    top_anomalies.to_csv(OUT_DIR / "top_anomalies.csv", index=False)
    (OUT_DIR / "anomaly_summary.json").write_text(json.dumps(anomaly_summary, indent=2), encoding="utf-8")

    temporal = temporal_signals(df)
    (OUT_DIR / "temporal_signals.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")

    run_meta = {
        "compute": asdict(compute),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "row_count_after_dedupe": int(len(df)),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    write_markdown_report(compute, profile, quality, coverage, summary, corr_pairs, anomaly_summary, temporal)
    print(f"Analysis complete. Report: {OUT_DIR / 'MBARI_STATISTICAL_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
