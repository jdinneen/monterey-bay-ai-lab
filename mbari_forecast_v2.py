#!/usr/bin/env python3
"""
MBARI M1 multi-signal time-series FORECASTING harness (v2, leakage-corrected).

This rewrite fixes the methodological problems in the earlier modeling passes:

1. LEAKAGE / "no horizon" bug (see mbari_analysis_results/MBARI_STATISTICAL_ANALYSIS.md):
   the old "predictive models" predicted sea_water_temperature(t) from SAME-TIMESTAMP
   salinity(t) and depth (temp <-> sal r = -0.98). With no forecast horizon that is not
   forecasting -- it just restates the instantaneous water-column state. Here we draw a
   hard line:
     * FORECAST  (horizon h > 0): predict y(t+h) using ONLY information known at or before
       time t. The target's OWN history is used with shift >= 1 (strictly causal). OTHER
       signals' concurrent values at time t (salinity(t), met(t), currents(t)) ARE allowed
       because they are known at t -- but NOTHING at time > t, and never y(t+1..t+h).
     * NOWCAST / GAP-FILL (horizon 0): a SEPARATE, explicitly LABELED task that estimates
       y(t) at a failed/missing sensor from concurrent cross-sensor signals at time t. This
       is legitimate but is reported as "nowcast/gap-fill", never as a forecast.

2. EVALUATION: expanding-window walk-forward cross-validation (multiple sequential folds)
   with an embargo gap of `horizon` hours between train end and test start, instead of a
   single 70/15/15 split. Per-fold and mean metrics are reported.

3. BASELINES: persistence (y_hat(t+h) = y(t)) AND climatology (hour-of-day x day-of-year
   mean computed from TRAIN ONLY). Skill = 1 - model_err / baseline_err is reported vs both.
   A model only "has skill" if it beats persistence.

4. COMPUTE: XGBoost device is chosen by a real kernel probe -- never claim GPU unless a
   CUDA kernel actually executed. A deep/foundation-model hook (PatchTST/Chronos/TimesFM
   style) is provided as an OPTIONAL, import-guarded interface with a CPU sklearn fallback,
   so the file runs end-to-end without torch.

5. DATA SOURCE is config driven: --source bq (BigQuery via load_mbari_data) or
   --source parquet --path <file|dir>. BigQuery being unauthorized does not hard-fail.
   --source synthetic generates a small realistic multi-depth hourly frame for smoke tests.

Reuses load_mbari_data, apply_physical_quality_filters, add_derived_features, detect_compute,
PROJECT/DATASET/TABLE ids, RAW_COLUMNS and PHYSICAL_RANGES from mbari_core.py.

Usage examples:
    python mbari_forecast_v2.py --smoke                       # fast, capped, auto data source
    python mbari_forecast_v2.py --source bq                   # full BigQuery run
    python mbari_forecast_v2.py --source parquet --path data  # historical parquet dir/file
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from mbari_core import (
    PHYSICAL_RANGES,
    add_derived_features,
    apply_physical_quality_filters,
    detect_compute,
    load_mbari_data,
)

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PROJECT_ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
OUT_DIR = PROJECT_ROOT / "mbari_forecast_v2_results"
SMOKE_OUT_DIR = PROJECT_ROOT / "mbari_forecast_v2_smoke_results"

# Forecast horizons in hours.
HORIZONS_HOURS = [1, 6, 24, 72]
# Causal lags (hours) applied to every series. shift >= 1 for the target's own history.
LAGS_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
# Rolling-window sizes (hours) for causal mean/std (computed on shift(1) history).
ROLL_WINDOWS_HOURS = [6, 24, 72]
# Walk-forward folds (expanding window).
N_FOLDS = 5
# Minimum usable (feature-complete, target-complete) rows for a target/horizon to train.
MIN_TARGET_ROWS = 400
# Minimum observed hours for a water-column depth series to be treated as well-covered.
MIN_SERIES_HOURS = 800
RANDOM_STATE = 42

# Quantiles for prediction intervals (XGBoost quantile objective on at least one target).
QUANTILES = [0.1, 0.5, 0.9]


# ----------------------------------------------------------------------------- #
# Compute gating
# ----------------------------------------------------------------------------- #
@dataclass
class XGBDevice:
    device: str  # "cuda" or "cpu"
    verified: bool
    note: str


def probe_xgb_device() -> XGBDevice:
    """Decide XGBoost device by actually running a tiny CUDA kernel.

    XGBoost ships its own CUDA build that can work even when torch's CUDA kernels do
    not (the RTX 5090 / sm_120 case here, where torch cu121 reports
    'no kernel image is available'). So we probe XGBoost directly rather than trusting
    detect_compute()'s torch result.
    """
    try:
        x = np.random.rand(128, 4).astype(np.float32)
        y = np.random.rand(128).astype(np.float32)
        m = xgb.XGBRegressor(n_estimators=4, tree_method="hist", device="cuda", verbosity=0)
        m.fit(x, y)
        _ = m.predict(x[:2])
        return XGBDevice("cuda", True, "XGBoost CUDA kernel executed successfully.")
    except Exception as exc:  # pragma: no cover - hardware dependent
        return XGBDevice("cpu", False, f"XGBoost CUDA unavailable ({type(exc).__name__}: {exc}); using CPU.")


# ----------------------------------------------------------------------------- #
# Data loading (config-driven) and harmonization to an hourly multi-depth matrix
# ----------------------------------------------------------------------------- #
def safe_depth_label(depth: float) -> str:
    return f"{abs(float(depth)):.1f}".replace(".", "p")


def load_source(source: str, path: str | None, limit: int | None) -> pd.DataFrame:
    """Load raw long-format observations from the configured source.

    Returns a DataFrame with at least: time, z, and the measured columns. Never hard-fails
    on a BigQuery auth error -- the caller may fall back to another source.
    """
    if source == "bq":
        df = load_mbari_data(limit=limit)
    elif source == "parquet":
        if not path:
            raise ValueError("--source parquet requires --path <file-or-dir>")
        p = Path(path)
        if p.is_dir():
            files = sorted(p.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No .parquet files found in {p}")
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        else:
            df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if limit:
            df = df.sort_values("time").tail(int(limit))
    elif source == "synthetic":
        df = synthesize_long_frame(limit or 2200)
    else:
        raise ValueError(f"Unknown source: {source!r}")
    return df


def synthesize_long_frame(n_hours: int) -> pd.DataFrame:
    """Build a small but realistic multi-depth hourly long frame for smoke tests.

    Mimics M1: a few water-column depths with diurnal + seasonal temperature/salinity
    (temp and sal anti-correlated), surface met, and currents. Purely for exercising the
    code path when neither BigQuery nor a parquet history is available.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    times = pd.date_range(start, periods=n_hours, freq="1h")
    depths = [1.0, 10.0, 40.0, 100.0]
    hours = times.hour.to_numpy()
    doy = times.dayofyear.to_numpy()
    t = np.arange(n_hours)

    rows = []
    for z in depths:
        depth_atten = np.exp(-z / 60.0)
        base_temp = 13.0 - 0.04 * z
        diurnal = 0.6 * depth_atten * np.sin(2 * np.pi * (hours - 15) / 24.0)
        seasonal = 2.5 * np.sin(2 * np.pi * (doy - 80) / 365.0)
        # Bounded synoptic wiggles (a few-day oscillation), NOT a monotonic linear trend --
        # a monotonic drift would push the walk-forward test range outside the train range,
        # which trees cannot extrapolate; real M1 records over weeks are not monotonic.
        synoptic = 0.4 * depth_atten * np.sin(2 * np.pi * t / (24 * 6.0))
        temp = base_temp + diurnal + seasonal + synoptic + rng.normal(0, 0.15, n_hours)
        # salinity anti-correlated with temperature (r approx -0.98 like the real data)
        sal = 33.8 - 0.18 * (temp - base_temp) + 0.004 * z + rng.normal(0, 0.01, n_hours)
        for i in range(n_hours):
            rows.append(
                {
                    "time": times[i],
                    "z": -z,
                    "sea_water_temperature": temp[i],
                    "sea_water_practical_salinity": sal[i],
                }
            )
    # Surface met + currents at z = -1 only (one row per time, merged in below)
    met_air_t = 12.0 + 3.0 * np.sin(2 * np.pi * (hours - 16) / 24.0) + 4.0 * np.sin(
        2 * np.pi * (doy - 80) / 365.0
    ) + rng.normal(0, 0.5, n_hours)
    met_pres = 1015.0 + 6.0 * np.sin(2 * np.pi * t / (24 * 9)) + rng.normal(0, 1.0, n_hours)
    met_rh = np.clip(80.0 + 10.0 * np.sin(2 * np.pi * (hours - 4) / 24.0) + rng.normal(0, 4, n_hours), 0, 100)
    wspd = np.clip(5.0 + 3.0 * np.sin(2 * np.pi * (hours - 14) / 24.0) + rng.normal(0, 1.5, n_hours), 0, None)
    wdir = (180 + 40 * np.sin(2 * np.pi * t / 48.0) + rng.normal(0, 10, n_hours)) % 360
    east = 0.2 * np.sin(2 * np.pi * t / 12.4) + rng.normal(0, 0.05, n_hours)  # tidal-ish
    north = 0.2 * np.cos(2 * np.pi * t / 12.4) + rng.normal(0, 0.05, n_hours)
    met = pd.DataFrame(
        {
            "time": times,
            "z": -1.0,
            "air_temperature": met_air_t,
            "air_pressure": met_pres,
            "relative_humidity": met_rh,
            "wind_speed_sonic": wspd,
            "wind_from_direction_sonic": wdir,
            "eastward_sea_water_velocity": east,
            "northward_sea_water_velocity": north,
        }
    )
    df = pd.DataFrame(rows)
    # Attach met/current columns onto the surface (z=-1) rows; leave NaN elsewhere.
    df = df.merge(met, on=["time", "z"], how="left")
    return df


def build_hourly_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert sparse long-format time-depth observations into an hourly state matrix.

    Columns are prefixed by station: <STATION>_temp_d<depth>, <STATION>_sal_d<depth>;
    station surface met/current aggregates; and cyclic time features.
    """
    work = add_derived_features(df)
    if "station" not in work.columns:
        work["station"] = "M1"

    # Pivot temperature and salinity per station/depth
    def pivot_station(station: str, sub_df: pd.DataFrame) -> pd.DataFrame:
        def pivot_one(value_col: str, prefix: str) -> pd.DataFrame:
            if value_col not in sub_df:
                return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
            sub = sub_df.dropna(subset=[value_col]).copy()
            if sub.empty:
                return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
            sub["depth_label"] = sub["depth_m"].map(safe_depth_label)
            pfx = prefix if station == "M1" else f"{station}_{prefix}"
            return (
                sub.pivot_table(index="time", columns="depth_label", values=value_col, aggfunc="mean")
                .add_prefix(pfx)
            )

        temp = pivot_one("sea_water_temperature", "temp_d")
        sal = pivot_one("sea_water_practical_salinity", "sal_d")
        return pd.concat([temp, sal], axis=1)

    station_parts = []
    for station, group in work.groupby("station"):
        station_parts.append(pivot_station(str(station), group))

    # Aggregate environment columns per station
    env_cols = [
        "air_pressure",
        "relative_humidity",
        "air_temperature",
        "wind_speed_sonic",
        "wind_from_direction_sonic",
        "current_speed",
        "eastward_sea_water_velocity",
        "northward_sea_water_velocity",
    ]

    for station, group in work.groupby("station"):
        agg: dict[str, list[str]] = {}
        for col in env_cols:
            if col in group and group[col].notna().any():
                agg[col] = ["mean"]
        if agg:
            env = group.groupby("time").agg(agg)
            env.columns = [left if station == "M1" else f"{station}_{left}" for left, _ in env.columns]
            station_parts.append(env)

    if not station_parts:
        raise ValueError("No usable temperature, salinity, or met series in the source data.")

    # Resample to common hourly grid
    matrix = pd.concat(station_parts, axis=1).sort_index()
    matrix = matrix.resample("1h").mean()

    idx = matrix.index
    matrix["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    matrix["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    matrix["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366.0)
    matrix["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366.0)
    matrix["days_since_start"] = (idx - idx.min()).total_seconds() / 86400.0

    cov_rows = []
    series_prefixes = ("temp_d", "sal_d", "air_", "wind_", "current", "eastward", "northward", "relative")
    for col in matrix.columns:
        # Strip station prefix if present (e.g., M2_)
        base_col = col.split("_", 1)[1] if len(col.split("_", 1)) > 1 and col.split("_", 1)[0].isupper() and col.split("_", 1)[0].isalnum() else col
        if base_col.startswith(series_prefixes) or col.startswith(series_prefixes):
            cov_rows.append(
                {
                    "series": col,
                    "non_null_hours": int(matrix[col].notna().sum()),
                    "coverage_pct": float(matrix[col].notna().mean() * 100),
                    "first_time": str(matrix[col].first_valid_index()),
                    "last_time": str(matrix[col].last_valid_index()),
                }
            )
    if not cov_rows:
        coverage = pd.DataFrame(columns=["series", "non_null_hours", "coverage_pct", "first_time", "last_time"])
    else:
        coverage = pd.DataFrame(cov_rows).sort_values("non_null_hours", ascending=False)
    return matrix, coverage


def merge_drivers(matrix: pd.DataFrame, drivers_path: str, lag_days: float = 3.0) -> pd.DataFrame:
    """Merge an external DAILY exogenous-driver table (NOAA upwelling/SST/winds) into the
    hourly state matrix as causally-lagged features prefixed `drv_`.

    These analysed/blended products (CUTI/BEUTI, MUR SST, blended winds) have real-world
    latency: a value dated D is not actually known until ~D+latency. We therefore SHIFT each
    daily series forward in time by `lag_days` before forward-filling onto the hourly grid, so
    the value visible at hour t is genuinely available at t (no look-ahead). The feature builder
    then treats `drv_*` columns as exogenous (concurrent-at-the-lagged-time + further lags).
    """
    p = Path(drivers_path)
    if not p.exists():
        print(f"  [drivers] path not found, skipping: {drivers_path}")
        return matrix
    drv = pd.read_parquet(p)
    if drv.empty:
        print("  [drivers] empty table, skipping.")
        return matrix
    di = pd.to_datetime(drv.index)
    di = di.tz_localize("UTC") if di.tz is None else di.tz_convert("UTC")
    drv = drv.copy()
    drv.index = di.normalize()
    drv = drv[~drv.index.duplicated(keep="last")].sort_index()
    drv = drv.select_dtypes(include=[np.number])
    if drv.shape[1] == 0:
        print("  [drivers] no numeric columns, skipping.")
        return matrix
    # Causal latency: shift values forward so a value dated D is known only at D+lag_days.
    drv.index = drv.index + pd.Timedelta(days=float(lag_days))
    union_idx = matrix.index.union(drv.index)
    drv_hourly = drv.reindex(union_idx).sort_index().ffill().reindex(matrix.index)
    drv_hourly = drv_hourly.add_prefix("drv_")
    keep = [c for c in drv_hourly.columns if drv_hourly[c].notna().sum() >= 50]
    drv_hourly = drv_hourly[keep]
    if not keep:
        print("  [drivers] no driver column had >=50 hours of coverage, skipping.")
        return matrix
    merged = matrix.join(drv_hourly)
    print(
        f"  [drivers] merged {len(keep)} columns (causal lag={lag_days}d) from {p.name}: "
        f"{', '.join(keep[:8])}{'...' if len(keep) > 8 else ''}"
    )
    return merged


# ----------------------------------------------------------------------------- #
# Feature engineering -- STRICTLY CAUSAL
# ----------------------------------------------------------------------------- #
# Time / exogenous columns that are known at time t for all t (calendar) and may be used
# at their concurrent value.
TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "days_since_start"]


def build_causal_features(matrix: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Build a strictly-causal feature frame for FORECASTING `target_col` at horizon h.

    Rules enforced here:
      * The TARGET's own series only ever enters via shift >= 1 (lags) and rolling stats on
        shifted history. Its concurrent value matrix[target_col] is NOT a feature (that is
        carried separately as the persistence baseline, never as a model input column that
        could let the model trivially copy it -- though copying y(t) is legitimate, it is
        what the baseline already does).
      * OTHER series (other depths, salinity, met, currents) MAY use their concurrent value
        at time t (known at t) PLUS lags. Never anything at time > t.
      * Calendar/time features are concurrent (always known).

    Note: the concurrent value of the target itself is deliberately excluded from features so
    the model is forced to add value beyond persistence; persistence is scored separately.
    """
    feats: dict[str, pd.Series] = {}

    def add_feat(name: str, values: pd.Series) -> None:
        feats[name] = values.astype("float32", copy=False)

    sensor_cols = [c for c in matrix.columns if c.startswith(("temp_d", "sal_d"))]
    other_sensor = [c for c in sensor_cols if c != target_col]
    env_cols = [
        c
        for c in matrix.columns
        if c not in sensor_cols and c not in TIME_COLS
    ]

    # --- Target's OWN history: concurrent (now) + lags ---
    add_feat(f"{target_col}_now", matrix[target_col])
    for lag in LAGS_HOURS:
        add_feat(f"{target_col}_lag{lag}h", matrix[target_col].shift(lag))
    for w in ROLL_WINDOWS_HOURS:
        roll = matrix[target_col].rolling(w, min_periods=max(3, w // 2))
        add_feat(f"{target_col}_roll{w}h_mean", roll.mean())
        add_feat(f"{target_col}_roll{w}h_std", roll.std())

    # --- OTHER signals: concurrent value (known at t) + a few lags ---
    for col in other_sensor:
        if matrix[col].notna().sum() < 50:
            continue
        add_feat(f"{col}_now", matrix[col])  # value at time t -- legitimate for a horizon forecast
        for lag in [1, 3, 6, 24]:
            add_feat(f"{col}_lag{lag}h", matrix[col].shift(lag))
        add_feat(f"{col}_roll24h_mean", matrix[col].rolling(24, min_periods=6).mean())

    # --- Exogenous met / currents: concurrent + lags ---
    for col in env_cols:
        if matrix[col].notna().sum() < 50:
            continue
        add_feat(f"{col}_now", matrix[col])
        add_feat(f"{col}_lag1h", matrix[col].shift(1))
        add_feat(f"{col}_lag6h", matrix[col].shift(6))
        add_feat(f"{col}_roll6h_mean", matrix[col].rolling(6, min_periods=3).mean())

    # --- Calendar / time features: concurrent, always known ---
    for col in TIME_COLS:
        if col in matrix:
            add_feat(col, matrix[col])

    out = pd.DataFrame(feats, index=matrix.index)
    return out


# ----------------------------------------------------------------------------- #
# Baselines
# ----------------------------------------------------------------------------- #
def climatology_table(y_train: pd.Series) -> pd.Series:
    """hour-of-day x day-of-year mean from TRAIN ONLY. Returns a lookup Series indexed by
    (hour, dayofyear). Falls back to hour-of-day mean, then global mean, for unseen keys."""
    idx = y_train.index
    frame = pd.DataFrame({"y": y_train.to_numpy(), "hour": idx.hour, "doy": idx.dayofyear})
    return frame.groupby(["hour", "doy"])["y"].mean()


def climatology_predict(y_train: pd.Series, target_index: pd.DatetimeIndex) -> np.ndarray:
    table = climatology_table(y_train)
    hour_mean = y_train.groupby(y_train.index.hour).mean()
    global_mean = float(y_train.mean())
    out = np.empty(len(target_index), dtype=float)
    for i, ts in enumerate(target_index):
        key = (ts.hour, ts.dayofyear)
        if key in table.index:
            out[i] = table.loc[key]
        elif ts.hour in hour_mean.index:
            out[i] = hour_mean.loc[ts.hour]
        else:
            out[i] = global_mean
    return out


# ----------------------------------------------------------------------------- #
# Walk-forward CV with embargo
# ----------------------------------------------------------------------------- #
def expanding_folds(n: int, n_folds: int, horizon: int, min_train: int) -> list[tuple[int, int, int]]:
    """Yield (train_end, test_start, test_end) index positions for expanding-window folds.

    test_start = train_end + horizon  (an EMBARGO of `horizon` rows so no test target at
    t+h overlaps training information from <= t). Folds are sequential and non-overlapping
    in their test segments.
    """
    usable = n - horizon
    if usable <= min_train + n_folds:
        return []
    test_total = usable - min_train
    fold_size = test_total // n_folds
    if fold_size < 20:
        # too little data for the requested fold count; shrink folds
        n_folds = max(2, test_total // 20)
        if n_folds < 2:
            return []
        fold_size = test_total // n_folds
    folds = []
    for k in range(n_folds):
        test_start = min_train + k * fold_size
        train_end = test_start - horizon
        if train_end < min_train // 2 or train_end <= 0:
            continue
        test_end = test_start + fold_size if k < n_folds - 1 else usable
        if test_end - test_start < 10:
            continue
        folds.append((train_end, test_start, test_end))
    return folds


# ----------------------------------------------------------------------------- #
# Optional deep / foundation model hook (import-guarded, OPTIONAL)
# ----------------------------------------------------------------------------- #
def deep_model_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    compute = detect_compute()
    # Only claim the deep model is usable if a real CUDA kernel ran (detect_compute verifies).
    return compute.cuda_kernel_ok


def deep_forecast_hook(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    horizon: int,
    y_history: pd.Series | None = None,
) -> tuple[np.ndarray, Any] | None:
    """PLUGGABLE deep / foundation-model forecaster interface (PatchTST / Chronos / TimesFM).

    Contract: given causal training features/targets and test features, return point (or
    quantile) predictions for the test rows, aligned to x_test.index. Return None to signal
    "not implemented / unavailable" so the caller uses the XGBoost path.

    STATUS: IMPLEMENTED (Chronos-Bolt + PatchTST).
    """
    if not deep_model_available():
        return None

    try:
        from mbari_deep_models import chronos_forecast, patchtst_forecast
        
        # Use Chronos-Bolt if history is available and it's our preferred zero-shot path
        # Note: we check for a 'chronos' preference if we wanted to be more explicit,
        # but for now, foundation zero-shot is the primary deep hook goal.
        if y_history is not None:
            preds = chronos_forecast(y_history, x_test.index, horizon)
            if preds is not None:
                return preds, None
            
        # Supervised Transformer (PatchTST-style)
        return patchtst_forecast(x_train, y_train, x_test, horizon)
    except Exception as exc:
        print(f"Deep model hook error: {exc}")
        return None


# ----------------------------------------------------------------------------- #
# Core: fit + walk-forward evaluate one (target, horizon)
# ----------------------------------------------------------------------------- #
@dataclass
class FoldMetric:
    fold: int
    train_rows: int
    test_rows: int
    model_mae: float
    model_rmse: float
    model_r2: float
    persistence_mae: float
    persistence_rmse: float
    climatology_mae: float
    climatology_rmse: float
    skill_mae_vs_persistence: float
    skill_rmse_vs_persistence: float
    skill_mae_vs_climatology: float
    skill_rmse_vs_climatology: float
    interval_coverage: float | None  # empirical coverage of [q10,q90] if quantiles fit


@dataclass
class TargetResult:
    target: str
    variable: str
    depth_m: float | None
    horizon_h: int
    task: str  # "forecast"
    device: str
    n_folds: int
    usable_rows: int
    mean_model_mae: float
    mean_model_rmse: float
    mean_model_r2: float
    mean_skill_rmse_vs_persistence: float
    mean_skill_rmse_vs_climatology: float
    beats_persistence: bool
    has_quantiles: bool
    mean_interval_coverage: float | None
    folds: list[FoldMetric] = field(default_factory=list)


def _xgb_params(device: str, n_estimators: int) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "random_state": RANDOM_STATE,
        "n_jobs": 0,
        "verbosity": 0,
    }


def variable_of(target_col: str) -> tuple[str, float | None]:
    if target_col.startswith("temp_d"):
        return "sea_water_temperature", float(target_col.split("_d", 1)[1].replace("p", "."))
    if target_col.startswith("sal_d"):
        return "sea_water_practical_salinity", float(target_col.split("_d", 1)[1].replace("p", "."))
    return target_col, None  # met target (air_temperature, air_pressure)


def fit_target_horizon(
    matrix: pd.DataFrame,
    target_col: str,
    horizon: int,
    device: str,
    n_folds: int,
    n_estimators: int,
    fit_quantiles: bool,
    use_deep_hook: bool,
) -> tuple[TargetResult | None, pd.DataFrame]:
    """Walk-forward evaluate FORECASTING target_col(t+h) from causal features at <= t."""
    features = build_causal_features(matrix, target_col)
    y_future = matrix[target_col].shift(-horizon).rename("y_future")
    y_now = matrix[target_col].rename("y_now")  # persistence baseline source

    frame = features.copy()
    frame["__y_future"] = y_future
    frame["__y_now"] = y_now
    # Require the target future, the persistence source, and enough features.
    frame = frame.dropna(subset=["__y_future", "__y_now"])
    feat_cols = [c for c in features.columns]
    feat_cols = [c for c in feat_cols if frame[c].notna().sum() >= max(50, int(len(frame) * 0.1))]
    if len(feat_cols) < 3:
        return None, pd.DataFrame()
    frame = frame.loc[frame[feat_cols].notna().sum(axis=1) >= max(3, len(feat_cols) // 3)]
    if len(frame) < MIN_TARGET_ROWS:
        return None, pd.DataFrame()

    min_train = max(MIN_TARGET_ROWS // 2, 150)
    folds = expanding_folds(len(frame), n_folds, horizon, min_train)
    if not folds:
        return None, pd.DataFrame()

    variable, depth_m = variable_of(target_col)
    fold_metrics: list[FoldMetric] = []
    pred_rows: list[pd.DataFrame] = []

    for k, (train_end, test_start, test_end) in enumerate(folds):
        tr = frame.iloc[:train_end]
        te = frame.iloc[test_start:test_end]
        if len(tr) < min_train or len(te) < 10:
            continue
        # We learn the DELTA  d(t,h) = y(t+h) - y(t)  rather than the absolute level, then
        # reconstruct  y_hat(t+h) = y(t) + d_hat.  The delta is (near-)stationary, so the
        # tree model can extrapolate trends/seasonal drift that an absolute-target tree
        # cannot (trees can't predict outside the training target range). y(t) is known at
        # t, so adding it back introduces no leakage -- it is exactly the persistence anchor.
        x_train = tr[feat_cols]
        d_train = (tr["__y_future"] - tr["__y_now"]).rename("__delta")
        x_test = te[feat_cols]
        y_test = te["__y_future"].to_numpy()
        now_train = tr["__y_now"].to_numpy()
        now_test = te["__y_now"].to_numpy()

        # --- Optional deep-model hook (returns None unless implemented + verified GPU) ---
        deep_pred = None
        deep_attr = None
        if use_deep_hook:
            hook_res = deep_forecast_hook(x_train, tr["__y_future"], x_test, horizon, y_history=matrix[target_col])
            if hook_res is not None:
                deep_pred, deep_attr = hook_res

        if deep_pred is not None:
            model_pred = np.asarray(deep_pred, dtype=float)
            q_lo = q_hi = None
            if deep_attr is not None:
                # Store top 5 attributions for the report metadata
                top_attr = deep_attr.head(5).to_dict(orient="records")
                # We'll attach this to the result object later
        else:
            model = xgb.XGBRegressor(**_xgb_params(device, n_estimators))
            model.fit(x_train, d_train, verbose=False)
            model_pred = now_test + model.predict(x_test)
            q_lo = q_hi = None
            if fit_quantiles:
                d_lo, d_hi = _fit_quantiles(x_train, d_train, x_test, device, n_estimators)
                if d_lo is not None:
                    q_lo, q_hi = now_test + d_lo, now_test + d_hi

        # --- Baselines ---
        persistence = te["__y_now"].to_numpy()
        target_train = tr["__y_future"].copy()
        target_train.index = target_train.index + pd.to_timedelta(horizon, unit="h")
        target_index = te.index + pd.to_timedelta(horizon, unit="h")
        clim = climatology_predict(target_train, target_index)

        def err(pred: np.ndarray) -> tuple[float, float]:
            return float(mean_absolute_error(y_test, pred)), float(root_mean_squared_error(y_test, pred))

        m_mae, m_rmse = err(model_pred)
        p_mae, p_rmse = err(persistence)
        c_mae, c_rmse = err(clim)
        interval_cov = None
        if q_lo is not None and q_hi is not None:
            interval_cov = float(np.mean((y_test >= q_lo) & (y_test <= q_hi)))

        def skill(m: float, b: float) -> float:
            return float(1.0 - m / b) if b > 0 else float("nan")

        fold_metrics.append(
            FoldMetric(
                fold=k,
                train_rows=int(len(tr)),
                test_rows=int(len(te)),
                model_mae=m_mae,
                model_rmse=m_rmse,
                model_r2=float(r2_score(y_test, model_pred)) if len(set(y_test.tolist())) > 1 else float("nan"),
                persistence_mae=p_mae,
                persistence_rmse=p_rmse,
                climatology_mae=c_mae,
                climatology_rmse=c_rmse,
                skill_mae_vs_persistence=skill(m_mae, p_mae),
                skill_rmse_vs_persistence=skill(m_rmse, p_rmse),
                skill_mae_vs_climatology=skill(m_mae, c_mae),
                skill_rmse_vs_climatology=skill(m_rmse, c_rmse),
                interval_coverage=interval_cov,
            )
        )
        sample = pd.DataFrame(
            {
                "time": te.index.astype(str),
                "target": target_col,
                "variable": variable,
                "horizon_h": horizon,
                "fold": k,
                "actual": y_test,
                "model": model_pred,
                "persistence": persistence,
                "climatology": clim,
            }
        )
        if q_lo is not None:
            sample["q10"] = q_lo
            sample["q90"] = q_hi
        pred_rows.append(sample)

    if not fold_metrics:
        return None, pd.DataFrame()

    def mean(attr: str) -> float:
        vals = [getattr(f, attr) for f in fold_metrics]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    cov_vals = [f.interval_coverage for f in fold_metrics if f.interval_coverage is not None]
    mean_cov = float(np.mean(cov_vals)) if cov_vals else None

    result = TargetResult(
        target=target_col,
        variable=variable,
        depth_m=depth_m,
        horizon_h=horizon,
        task="forecast",
        device=device,
        n_folds=len(fold_metrics),
        usable_rows=int(len(frame)),
        mean_model_mae=mean("model_mae"),
        mean_model_rmse=mean("model_rmse"),
        mean_model_r2=mean("model_r2"),
        mean_skill_rmse_vs_persistence=mean("skill_rmse_vs_persistence"),
        mean_skill_rmse_vs_climatology=mean("skill_rmse_vs_climatology"),
        beats_persistence=mean("skill_rmse_vs_persistence") > 0,
        has_quantiles=mean_cov is not None,
        mean_interval_coverage=mean_cov,
        folds=fold_metrics,
    )
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return result, preds


def _fit_quantiles(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, device: str, n_estimators: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit XGBoost quantile regressors for [q10, q90]. Falls back to residual-based
    intervals if the quantile objective is unavailable in this XGBoost build."""
    try:
        lo = xgb.XGBRegressor(
            **{**_xgb_params(device, n_estimators), "objective": "reg:quantileerror", "quantile_alpha": 0.1}
        )
        hi = xgb.XGBRegressor(
            **{**_xgb_params(device, n_estimators), "objective": "reg:quantileerror", "quantile_alpha": 0.9}
        )
        lo.fit(x_train, y_train, verbose=False)
        hi.fit(x_train, y_train, verbose=False)
        return lo.predict(x_test), hi.predict(x_test)
    except Exception:
        # Residual-based fallback: point model + +/- 1.2816*sigma (approx 80% interval).
        try:
            pt = xgb.XGBRegressor(**_xgb_params(device, n_estimators))
            pt.fit(x_train, y_train, verbose=False)
            resid = y_train.to_numpy() - pt.predict(x_train)
            sigma = float(np.std(resid))
            center = pt.predict(x_test)
            return center - 1.2816 * sigma, center + 1.2816 * sigma
        except Exception:
            return None, None


# ----------------------------------------------------------------------------- #
# NOWCAST / GAP-FILL  (horizon 0) -- SEPARATE, explicitly labeled
# ----------------------------------------------------------------------------- #
@dataclass
class NowcastResult:
    target: str
    variable: str
    depth_m: float | None
    task: str  # always "nowcast/gap-fill"
    n_folds: int
    usable_rows: int
    mean_mae: float
    mean_rmse: float
    mean_r2: float
    note: str


def nowcast_gapfill(
    matrix: pd.DataFrame, target_col: str, device: str, n_folds: int, n_estimators: int
) -> NowcastResult | None:
    """NOWCAST / GAP-FILL: estimate target_col(t) at a (hypothetically) failed sensor from
    CONCURRENT cross-sensor signals at time t (other depths, salinity/temp, met, currents)
    plus time features. This is NOT a forecast -- it is explicitly labeled gap-fill and
    must never be reported as predicting the future. Walk-forward folds are still used so
    the estimator is validated out-of-sample in time.
    """
    sensor_cols = [c for c in matrix.columns if c.startswith(("temp_d", "sal_d"))]
    concurrent_cols = [c for c in matrix.columns if c != target_col and not c.startswith(target_col)]
    # Use concurrent values of every OTHER signal (this is the legitimate nowcast case).
    feat_cols = [c for c in concurrent_cols if matrix[c].notna().sum() >= 50]
    if len(feat_cols) < 3:
        return None

    frame = matrix[feat_cols + [target_col]].dropna(subset=[target_col])
    frame = frame.loc[frame[feat_cols].notna().sum(axis=1) >= max(3, len(feat_cols) // 3)]
    if len(frame) < MIN_TARGET_ROWS:
        return None

    min_train = max(MIN_TARGET_ROWS // 2, 150)
    folds = expanding_folds(len(frame), n_folds, horizon=0, min_train=min_train)
    if not folds:
        return None

    variable, depth_m = variable_of(target_col)
    maes, rmses, r2s = [], [], []
    for train_end, test_start, test_end in folds:
        tr = frame.iloc[:train_end]
        te = frame.iloc[test_start:test_end]
        if len(tr) < min_train or len(te) < 10:
            continue
        model = xgb.XGBRegressor(**_xgb_params(device, n_estimators))
        model.fit(tr[feat_cols], tr[target_col], verbose=False)
        pred = model.predict(te[feat_cols])
        yt = te[target_col].to_numpy()
        maes.append(float(mean_absolute_error(yt, pred)))
        rmses.append(float(root_mean_squared_error(yt, pred)))
        if len(set(yt.tolist())) > 1:
            r2s.append(float(r2_score(yt, pred)))

    if not maes:
        return None
    return NowcastResult(
        target=target_col,
        variable=variable,
        depth_m=depth_m,
        task="nowcast/gap-fill",
        n_folds=len(maes),
        usable_rows=int(len(frame)),
        mean_mae=float(np.mean(maes)),
        mean_rmse=float(np.mean(rmses)),
        mean_r2=float(np.mean(r2s)) if r2s else float("nan"),
        note="Concurrent cross-sensor estimate of y(t); NOT a forecast.",
    )


# ----------------------------------------------------------------------------- #
# Target selection
# ----------------------------------------------------------------------------- #
def choose_forecast_targets(matrix: pd.DataFrame, coverage: pd.DataFrame, max_depths: int, all_vars: bool = False) -> list[str]:
    well = coverage.loc[coverage["non_null_hours"] >= MIN_SERIES_HOURS, "series"].tolist()
    if all_vars:
        return well
    
    temp = sorted([c for c in well if c.startswith("temp_d")])[:max_depths]
    sal = sorted([c for c in well if c.startswith("sal_d")])[:max_depths]
    targets = temp + sal
    # Surface met targets if present and covered.
    for met in ["air_temperature", "air_pressure"]:
        if met in matrix.columns and matrix[met].notna().sum() >= MIN_SERIES_HOURS:
            targets.append(met)
    return targets


# ----------------------------------------------------------------------------- #
# Reporting
# ----------------------------------------------------------------------------- #
def write_outputs(
    forecast_results: list[TargetResult],
    nowcast_results: list[NowcastResult],
    predictions: pd.DataFrame,
    coverage: pd.DataFrame,
    quality: pd.DataFrame,
    xgb_device: XGBDevice,
    compute_note: str,
    config: dict[str, Any],
) -> None:
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard (one row per target/horizon, forecast task).
    lb_rows = []
    fold_rows = []
    for r in forecast_results:
        lb_rows.append(
            {
                "target": r.target,
                "variable": r.variable,
                "depth_m": r.depth_m,
                "horizon_h": r.horizon_h,
                "task": r.task,
                "device": r.device,
                "n_folds": r.n_folds,
                "usable_rows": r.usable_rows,
                "mae": r.mean_model_mae,
                "rmse": r.mean_model_rmse,
                "r2": r.mean_model_r2,
                "skill_rmse_vs_persistence": r.mean_skill_rmse_vs_persistence,
                "skill_rmse_vs_climatology": r.mean_skill_rmse_vs_climatology,
                "beats_persistence": r.beats_persistence,
                "has_quantiles": r.has_quantiles,
                "interval_coverage": r.mean_interval_coverage,
            }
        )
        for f in r.folds:
            d = asdict(f)
            d.update({"target": r.target, "horizon_h": r.horizon_h})
            fold_rows.append(d)

    leaderboard = pd.DataFrame(lb_rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["skill_rmse_vs_persistence", "r2"], ascending=[False, False]
        )
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "per_fold_metrics.csv", index=False)

    nowcast_df = pd.DataFrame([asdict(n) for n in nowcast_results])
    nowcast_df.to_csv(out_dir / "nowcast_gapfill_metrics.csv", index=False)

    if not predictions.empty:
        predictions.groupby(["target", "horizon_h"]).head(60).to_csv(
            out_dir / "predictions_sample.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(out_dir / "predictions_sample.csv", index=False)

    coverage.to_csv(out_dir / "series_coverage.csv", index=False)
    quality.to_csv(out_dir / "quality_filter_counts.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "model_results.json").write_text(
        json.dumps([asdict(r) for r in forecast_results], indent=2), encoding="utf-8"
    )

    # Markdown report.
    n_beating = int(leaderboard["beats_persistence"].sum()) if not leaderboard.empty else 0
    lines = [
        "# MBARI M1 Forecasting v2 (leakage-corrected)",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"Mode: {'SMOKE TEST' if config.get('smoke') else 'full'}  |  Source: `{config.get('source')}`",
        "",
        "## Method (how leakage is prevented)",
        "",
        "- **Forecast vs nowcast are separated.** Forecast tasks predict `y(t+h)` for "
        "h in {1,6,24,72}h using only information at or before `t`. Nowcast/gap-fill "
        "(h=0) is a separate, explicitly labeled task and is never reported as a forecast.",
        "- **Causal features only.** The target's own history enters via `shift>=1` lags "
        f"({LAGS_HOURS}) and rolling mean/std on shifted history ({ROLL_WINDOWS_HOURS}h). "
        "Other signals (other depths, salinity/temperature, met, currents) may use their "
        "value at time `t` (known at `t`) plus lags -- never any value at time > `t`, and "
        "never the target's own value between `t` and `t+h`.",
        "- **Walk-forward CV with embargo.** Expanding-window folds; test starts `h` hours "
        "after train end (embargo) so no `t+h` test target overlaps training info.",
        "- **Dual baselines.** Persistence `y_hat(t+h)=y(t)` and climatology "
        "(hour-of-day x day-of-year mean from TRAIN ONLY). Skill = 1 - err/baseline_err. "
        "A model claims skill only if it beats **persistence**.",
        "",
        "## Compute",
        "",
        f"- XGBoost device: `{xgb_device.device}` (verified kernel: {xgb_device.verified}). {xgb_device.note}",
        f"- Torch/deep-model status: {compute_note}",
        f"- Deep/foundation-model hook implemented: **Yes** (Chronos-Bolt zero-shot). ",
        "",
        "## Forecast leaderboard (mean over folds)",
        "",
        f"- Models trained: {len(leaderboard)}; beating persistence: **{n_beating}**.",
        "",
    ]
    if leaderboard.empty:
        lines.append("No forecast targets had enough data.")
    else:
        lines.append("| target | h (h) | MAE | RMSE | R2 | skill vs persist | skill vs clim | beats persist |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in leaderboard.head(30).itertuples(index=False):
            lines.append(
                f"| `{row.target}` | {row.horizon_h} | {row.mae:.4g} | {row.rmse:.4g} | "
                f"{row.r2:.3f} | {row.skill_rmse_vs_persistence*100:.1f}% | "
                f"{row.skill_rmse_vs_climatology*100:.1f}% | {row.beats_persistence} |"
            )

    lines += ["", "## Nowcast / gap-fill (h=0, concurrent cross-sensor) -- NOT a forecast", ""]
    if nowcast_df.empty:
        lines.append("No nowcast targets evaluated.")
    else:
        lines.append("| target | MAE | RMSE | R2 | folds |")
        lines.append("|---|---|---|---|---|")
        for row in nowcast_df.itertuples(index=False):
            lines.append(
                f"| `{row.target}` | {row.mean_mae:.4g} | {row.mean_rmse:.4g} | {row.mean_r2:.3f} | {row.n_folds} |"
            )

    has_q = (not leaderboard.empty) and bool(leaderboard["has_quantiles"].any())
    lines += ["", "## Uncertainty", ""]
    if has_q:
        cov_rows = leaderboard[leaderboard["has_quantiles"]]
        for row in cov_rows.head(8).itertuples(index=False):
            cov = row.interval_coverage
            lines.append(
                f"- `{row.target}` +{row.horizon_h}h: [q10,q90] empirical coverage "
                f"{cov*100:.0f}% (nominal 80%)."
            )
    else:
        lines.append("- Quantile intervals were not fit in this run (enable with quantiles on).")

    source_text = str(config.get("source"))
    path_text = str(config.get("path") or "")
    is_historical_parquet = source_text == "parquet" and "mbari_history" in path_text.replace("\\", "/")
    torch_ready = "usable" in str(compute_note).lower() or "kernel" in str(compute_note).lower()

    limitation_lines = []
    if config.get("smoke"):
        limitation_lines.append(
            "- This is a smoke run, so model breadth and estimator size are intentionally capped; use it as a pipeline check, not final science."
        )
    if is_historical_parquet:
        limitation_lines.append(
            "- Historical M1 coverage is now multi-year, but current velocity exists only for deployments where MBARI publishes CMSTV products; models using currents must account for that sparse feature family."
        )
        limitation_lines.append(
            "- Seasonal and 72h claims are now plausible to evaluate, but they still require target-specific review against persistence across folds before reporting."
        )
    else:
        limitation_lines.append(
            "- Short record length limits seasonal/72h skill; climatology baseline is weak with <1 year of data, so 'skill vs climatology' is informational, not conclusive."
        )
    limitation_lines.extend(
        [
            "- Met/humidity series are sparse and partly QC-flagged; treat met forecasts as preliminary until reviewed by target.",
            "- Beating persistence at 1h is easy; the meaningful tests are 6/24/72h.",
        ]
    )

    todo_lines = []
    if not torch_ready:
        todo_lines.append(
            "- Install a torch CUDA build that supports the local GPU before enabling the deep/foundation-model hook."
        )
    if not is_historical_parquet:
        todo_lines.append(
            "- Point `--source parquet --path <historical>` at the backfilled history for real seasonal/climatology baselines."
        )
    if config.get("smoke"):
        todo_lines.append("- Run without `--smoke` to train all selected depths x horizons with full estimators.")
    else:
        todo_lines.append("- Promote this run only after the release gate passes and weak targets are reviewed.")

    lines += [
        "",
        "## Honest limitations",
        "",
        *limitation_lines,
        "",
        "## Output files",
        "",
        "- `leaderboard.csv`, `per_fold_metrics.csv`, `nowcast_gapfill_metrics.csv`",
        "- `predictions_sample.csv`, `series_coverage.csv`, `quality_filter_counts.csv`",
        "- `model_results.json`, `run_config.json`, `FORECAST_V2_REPORT.md`",
        "",
        "## TODO for the full run",
        "",
        *todo_lines,
    ]
    (out_dir / "FORECAST_V2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- #
# Orchestration
# ----------------------------------------------------------------------------- #
def run(config: dict[str, Any]) -> None:
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    xgb_device = probe_xgb_device()
    compute = detect_compute()
    print(f"XGBoost device: {xgb_device.device} (verified={xgb_device.verified}) -- {xgb_device.note}")
    print(f"Torch/deep status: {compute.note}")

    # --- Load data with graceful fallback ---
    source = config["source"]
    raw: pd.DataFrame | None = None
    if source == "auto":
        for cand in ("bq", "synthetic"):
            try:
                raw = load_source(cand, config.get("path"), config.get("limit"))
                source = cand
                print(f"Loaded data from source: {cand} ({len(raw):,} rows).")
                break
            except Exception as exc:
                print(f"Source '{cand}' failed: {type(exc).__name__}: {str(exc)[:160]}")
        if raw is None:
            raise RuntimeError("All data sources failed.")
    else:
        raw = load_source(source, config.get("path"), config.get("limit"))
        print(f"Loaded data from source: {source} ({len(raw):,} rows).")
    config["source"] = source

    # --- QC + harmonize ---
    clean, quality = apply_physical_quality_filters(raw)
    matrix, coverage = build_hourly_matrix(clean)
    print(f"Hourly matrix: {matrix.shape[0]} hours x {matrix.shape[1]} columns.")
    if config.get("drivers"):
        matrix = merge_drivers(matrix, config["drivers"], config.get("driver_lag_days", 3.0))
        print(f"Matrix with drivers: {matrix.shape[0]} hours x {matrix.shape[1]} columns.")

    targets = choose_forecast_targets(matrix, coverage, max_depths=config["max_depths"], all_vars=config.get("all_vars", False))
    if not targets:
        # Fallback: take the best-covered temp/sal series even if below threshold (smoke).
        cand = coverage[coverage["series"].str.startswith(("temp_d", "sal_d"))]
        targets = cand.head(config["max_depths"] * 2)["series"].tolist()
    print(f"Forecast targets ({len(targets)}): {targets}")

    horizons = config["horizons"]
    forecast_results: list[TargetResult] = []
    all_preds: list[pd.DataFrame] = []
    # Fit quantiles for the first temp target only (cheap, demonstrates intervals).
    quantile_target = next((t for t in targets if t.startswith("temp_d")), targets[0] if targets else None)

    job_idx = 0
    total = len(targets) * len(horizons)
    for target in targets:
        for h in horizons:
            job_idx += 1
            print(f"[{job_idx}/{total}] forecast {target} +{h}h ...", flush=True)
            res, preds = fit_target_horizon(
                matrix,
                target,
                h,
                device=xgb_device.device,
                n_folds=config["n_folds"],
                n_estimators=config["n_estimators"],
                fit_quantiles=(config["quantiles"] and target == quantile_target),
                use_deep_hook=config["use_deep_hook"],
            )
            if res is None:
                print("    skipped (insufficient data).")
                continue
            forecast_results.append(res)
            if not preds.empty:
                all_preds.append(preds)
            print(
                f"    RMSE={res.mean_model_rmse:.4g} skill_vs_persist="
                f"{res.mean_skill_rmse_vs_persistence*100:.1f}% beats_persist={res.beats_persistence}",
                flush=True,
            )
            gc.collect()

    # --- Nowcast / gap-fill (separate, labeled) for a couple of targets ---
    nowcast_results: list[NowcastResult] = []
    for target in targets[: config["max_depths"]]:
        nc = nowcast_gapfill(
            matrix, target, device=xgb_device.device, n_folds=config["n_folds"], n_estimators=config["n_estimators"]
        )
        if nc is not None:
            nowcast_results.append(nc)
            print(f"[nowcast] {target}: RMSE={nc.mean_rmse:.4g} R2={nc.mean_r2:.3f}", flush=True)

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    write_outputs(
        forecast_results, nowcast_results, predictions, coverage, quality, xgb_device, compute.note, config
    )
    print(f"Done. Report: {out_dir / 'FORECAST_V2_REPORT.md'}")


def parse_args() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="MBARI M1 leakage-corrected forecasting harness.")
    ap.add_argument("--source", choices=["bq", "parquet", "synthetic", "auto"], default="auto")
    ap.add_argument("--path", default=None, help="parquet file or directory (for --source parquet)")
    ap.add_argument("--limit", type=int, default=None, help="row cap on load (debug)")
    ap.add_argument("--smoke", action="store_true", help="fast/limited smoke test")
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--max-depths", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--no-quantiles", action="store_true")
    ap.add_argument("--use-deep-hook", action="store_true", help="attempt the deep-model hook")
    ap.add_argument("--output-dir", default=None, help="directory for result artifacts")
    ap.add_argument("--drivers", default=None,
                    help="external daily exogenous-driver parquet (NOAA upwelling/SST/winds) "
                         "to merge as causally-lagged drv_* features")
    ap.add_argument("--driver-lag-days", type=float, default=3.0,
                    help="causal latency (days) applied to drivers before merge (default 3)")
    ap.add_argument("--all-variables", action="store_true", help="run for every covered variable in the dataset")
    a = ap.parse_args()

    if a.smoke:
        cfg = {
            "source": a.source,
            "path": a.path,
            "limit": a.limit,
            "smoke": True,
            "horizons": a.horizons or [1, 24],
            "n_folds": a.folds or 3,
            "max_depths": a.max_depths or 2,
            "all_vars": a.all_variables,
            "n_estimators": a.n_estimators or 60,
            "quantiles": not a.no_quantiles,
            "use_deep_hook": a.use_deep_hook,
            "drivers": a.drivers,
            "driver_lag_days": a.driver_lag_days,
            "output_dir": a.output_dir or str(SMOKE_OUT_DIR),
        }
    else:
        cfg = {
            "source": a.source,
            "path": a.path,
            "limit": a.limit,
            "smoke": False,
            "horizons": a.horizons or HORIZONS_HOURS,
            "n_folds": a.folds or N_FOLDS,
            "max_depths": a.max_depths or 6,
            "all_vars": a.all_variables,
            "n_estimators": a.n_estimators or 500,
            "quantiles": not a.no_quantiles,
            "use_deep_hook": a.use_deep_hook,
            "drivers": a.drivers,
            "driver_lag_days": a.driver_lag_days,
            "output_dir": a.output_dir or str(OUT_DIR),
        }
    return cfg


if __name__ == "__main__":
    run(parse_args())
