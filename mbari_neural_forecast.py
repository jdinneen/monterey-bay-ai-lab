#!/usr/bin/env python3
"""
MBARI M1 neural forecasting harness (concurrency-safe).

Trains ONE neural model (chosen by --model) on the 21-year M1 hourly record and evaluates
it with expanding walk-forward cross-validation against persistence + climatology, at
horizons {1,6,24,72,168}h, for temp/salinity across depth + surface met.

CONCURRENCY: every run writes ONLY into its own --outdir. No shared files are touched, so
many models can train at once (different agents / processes) without clobbering each other.
A separate research/model-lab merge step can read all outdirs into one leaderboard.

Honest evaluation: gaps are filled with past-only rules ONLY so the model has a continuous
series to train on; metrics are computed ONLY where the forecast target and forecast-origin
value were ORIGINALLY OBSERVED (a mask), never on filled test targets.

Models (Nixtla neuralforecast): tft, nhits, nbeatsx, itransformer, patchtst, tsmixerx,
dlinear, nlinear.  Interpretable: tft (variable selection + attention), nhits/nbeatsx (basis).
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from mbari_forecast_v2 import load_source, build_hourly_matrix
from mbari_core import apply_physical_quality_filters
from mbari_split_contracts import (
    SplitContract,
    build_split_contract as build_shared_split_contract,
    write_split_contract,
)

PROJECT_ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
PARQUET = os.environ.get(
    "MBARI_SOURCE_PARQUET",
    str(PROJECT_ROOT / "mbari_history" / "opendap" / "m1_history.parquet"),
)
HORIZONS = [1, 6, 24, 72, 168]
FUTR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]  # calendar, known into the future
TARGET_HIST_EXOG = ["target_was_observed", "target_was_filled", "target_hours_since_observed"]
QUANTILES = [0.1, 0.5, 0.9]  # for --loss quantile (MQLoss); 0.8 central interval = q0.9 - q0.1
STD_DEPTHS = ["1p0", "10p0", "20p0", "40p0", "60p0", "80p0", "100p0", "150p0", "200p0", "250p0", "300p0"]


def target_series() -> list[str]:
    s = [f"temp_d{d}" for d in STD_DEPTHS] + [f"sal_d{d}" for d in STD_DEPTHS]
    return s + ["air_temperature", "air_pressure"]


CACHE = Path(os.environ.get("MBARI_CACHE_DIR", PROJECT_ROOT / "nn_cache"))
CACHE_VERSION = "v3_past_only_fill_origin_observed_missingness"
LAKEHOUSE = Path(os.environ.get("MBARI_LAKEHOUSE_DIR", PROJECT_ROOT / "lakehouse"))


def build_long_frame(smoke: bool, series_filter: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df for neuralforecast, obs_mask df). Cached to parquet so concurrent model
    processes don't each rebuild the 4.55M-row frame from the 3M-row source (RAM/page-file fix)."""
    CACHE.mkdir(exist_ok=True)
    tag = f"{CACHE_VERSION}_{'smoke' if smoke else 'full'}"
    lf_p, mk_p, meta_p = CACHE / f"long_{tag}.parquet", CACHE / f"mask_{tag}.parquet", CACHE / f"meta_{tag}.json"
    if lf_p.exists() and mk_p.exists():
        long_df, mask = pd.read_parquet(lf_p), pd.read_parquet(mk_p)
    else:
        long_df, mask = _build_long_frame(smoke)
        long_df.to_parquet(lf_p); mask.to_parquet(mk_p)
        meta = {
            "cache_version": CACHE_VERSION,
            "source": PARQUET,
            "smoke": smoke,
            "rows": int(len(long_df)),
            "series": sorted(long_df["unique_id"].unique()),
            "fill": "observed values plus forward-fill <=48h, then same-day-of-year prior-observation mean, then prior global mean",
            "target_hist_exog": TARGET_HIST_EXOG,
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if series_filter:
        keep = set(series_filter)
        long_df = long_df[long_df["unique_id"].isin(keep)].copy()
        mask = mask[mask["unique_id"].isin(keep)].copy()
        missing = sorted(keep - set(long_df["unique_id"].unique()))
        if missing:
            raise ValueError(f"requested series not available: {missing}")
    return long_df, mask


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def append_parquet_table(path: Path, rows: pd.DataFrame, key_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = rows
    combined.to_parquet(path, index=False)


def run_manifest_row(manifest: dict) -> dict:
    return {
        "run_id": manifest["run_id"],
        "split_id": manifest["split_id"],
        "created_at_utc": manifest["created_at_utc"],
        "outdir": manifest["outdir"],
        "model": manifest["model"],
        "loss": manifest["loss"],
        "cache_version": manifest["cache_version"],
        "status": manifest["status"],
        "drivers_enabled": manifest["drivers"] is not None,
        "params_json": json.dumps(manifest["params"], sort_keys=True, default=str),
        "drivers_json": json.dumps(manifest["drivers"], sort_keys=True, default=str),
        "env_json": json.dumps(manifest["env"], sort_keys=True, default=str),
        "source_json": json.dumps(manifest["source"], sort_keys=True, default=str),
        "artifacts_json": json.dumps(manifest["artifacts"], sort_keys=True, default=str),
    }


def build_split_contract(cv: pd.DataFrame, long_df: pd.DataFrame, config: dict) -> tuple[str, pd.DataFrame, dict]:
    cutoffs = sorted(pd.to_datetime(cv["cutoff"].drop_duplicates()).tolist())
    train_start = pd.to_datetime(long_df["ds"].min())
    contract = build_shared_split_contract(
        source_path=PARQUET,
        cutoffs=cutoffs,
        train_start=train_start,
        horizons=HORIZONS,
        cache_version=CACHE_VERSION,
        config=config,
        family="neural",
    )
    return contract.split_id, contract.rows, contract.manifest


def emit_lakehouse_contracts(
    out: Path,
    cv: pd.DataFrame,
    res: pd.DataFrame,
    long_df: pd.DataFrame,
    summary: dict,
    model_col: str,
    config: dict,
    pred_cols: list[str],
) -> dict:
    """Emit local Delta-shaped tables: silver split contracts and gold run/metric/prediction outputs."""
    run_id = stable_hash({
        "created_at_utc": utc_now(),
        "outdir": str(out),
        "model": summary["model"],
        "loss": summary["loss"],
        "drivers": summary.get("drivers"),
        "config": config,
    }) + "_" + uuid.uuid4().hex[:8]
    split_id, split_rows, split_manifest = build_split_contract(cv, long_df, config)

    write_split_contract(SplitContract(split_id, split_rows, split_manifest), LAKEHOUSE)

    run_manifest = {
        "run_id": run_id,
        "split_id": split_id,
        "created_at_utc": utc_now(),
        "outdir": str(out),
        "model": summary["model"],
        "loss": summary["loss"],
        "cache_version": CACHE_VERSION,
        "source": file_fingerprint(PARQUET),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": {
            "MBARI_ACCEL": os.environ.get("MBARI_ACCEL"),
            "MBARI_BATCH_SIZE": os.environ.get("MBARI_BATCH_SIZE"),
            "MBARI_WINDOWS_BATCH": os.environ.get("MBARI_WINDOWS_BATCH"),
            "MBARI_INFER_WINDOWS_BATCH": os.environ.get("MBARI_INFER_WINDOWS_BATCH"),
            "MBARI_VALID_BATCH": os.environ.get("MBARI_VALID_BATCH"),
        },
        "params": config,
        "drivers": summary.get("drivers"),
        "artifacts": {
            "summary_json": str(out / "summary.json"),
            "leaderboard_csv": str(out / "leaderboard.csv"),
            "cv_predictions_parquet": str(out / "cv_predictions.parquet"),
        },
        "status": "completed",
    }
    run_dir = LAKEHOUSE / "gold" / "forecast_runs" / f"run_id={run_id}"
    write_json(run_dir / "run_manifest.json", run_manifest)
    append_parquet_table(
        LAKEHOUSE / "gold" / "forecast_runs" / "run_manifest.parquet",
        pd.DataFrame([run_manifest_row(run_manifest)]),
        ["run_id"],
    )

    metrics = res.copy()
    if not metrics.empty:
        metrics.insert(0, "run_id", run_id)
        metrics.insert(1, "split_id", split_id)
        metrics["model"] = summary["model"]
        metrics["loss"] = summary["loss"]
        metrics["drivers_enabled"] = summary.get("drivers") is not None
        metrics["cache_version"] = CACHE_VERSION
        metrics["created_at_utc"] = utc_now()
    write_parquet(LAKEHOUSE / "gold" / "forecast_metrics" / f"run_id={run_id}" / "metrics.parquet", metrics)
    if not metrics.empty:
        append_parquet_table(
            LAKEHOUSE / "gold" / "forecast_metrics" / "metrics.parquet",
            metrics,
            # 'model' is REQUIRED in the key: foundation runs pack multiple models
            # (chronos, timesfm) under one run_id, so a (run_id,unique_id,horizon_h)
            # key would let a global dedup silently drop all but the last model.
            ["run_id", "split_id", "unique_id", "horizon_h", "model"],
        )

    pred_cols_to_keep = ["unique_id", "ds", "cutoff", "y"] + pred_cols
    predictions = cv[pred_cols_to_keep].copy()
    predictions.insert(0, "run_id", run_id)
    predictions.insert(1, "split_id", split_id)
    predictions["model_prediction_col"] = model_col
    predictions["cache_version"] = CACHE_VERSION
    write_parquet(LAKEHOUSE / "gold" / "forecast_predictions" / f"run_id={run_id}" / "predictions.parquet", predictions)

    return {"run_id": run_id, "split_id": split_id}


def past_only_fill(y_obs: pd.Series) -> pd.Series:
    """Fill gaps without using values later than the timestamp being filled.

    NeuralForecast needs a dense target history for training windows. This keeps observed
    values unchanged, fills short outages from the latest past observation, then uses a
    same-day-of-year climatology computed only from previous observed years. Remaining
    gaps use the previous global observed mean. Leading gaps before the first observation
    are dropped by the caller.
    """
    y = y_obs.copy()
    observed = y.notna()
    y = y.ffill(limit=48)
    missing = y.isna()
    if not missing.any():
        return y

    doy = pd.Series(y.index.dayofyear, index=y.index)
    prior_doy_mean = pd.Series(index=y.index, dtype="float64")
    for _, idx_for_doy in doy.groupby(doy).groups.items():
        obs_at_doy = y_obs.loc[idx_for_doy]
        prior_doy_mean.loc[idx_for_doy] = obs_at_doy.expanding().mean().shift(1).to_numpy()
    y = y.fillna(prior_doy_mean)

    prior_global_mean = y_obs.where(observed).expanding().mean().shift(1)
    y = y.fillna(prior_global_mean)
    return y


def target_missingness_features(observed: pd.Series) -> pd.DataFrame:
    """Build causal target-availability features for historical exogenous use."""
    obs = observed.astype(bool).copy()
    positions = pd.Series(np.arange(len(obs), dtype="float64"), index=obs.index)
    last_obs_position = positions.where(obs).ffill()
    age = (positions - last_obs_position).fillna(0.0)
    return pd.DataFrame(
        {
            "target_was_observed": obs.astype("float32"),
            "target_was_filled": (~obs).astype("float32"),
            "target_hours_since_observed": age.astype("float32"),
        },
        index=obs.index,
    )


def _build_long_frame(smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_source("parquet", PARQUET, None)
    clean, _ = apply_physical_quality_filters(raw)
    matrix, _ = build_hourly_matrix(clean)
    matrix = matrix.sort_index()
    full = pd.date_range(matrix.index.min(), matrix.index.max(), freq="1h")
    matrix = matrix.reindex(full)

    series = [c for c in target_series() if c in matrix.columns]
    if smoke:
        series = [c for c in ["temp_d10p0", "temp_d100p0", "sal_d10p0", "air_pressure"] if c in series]
        matrix = matrix.tail(24 * 365 * 3)  # last ~3 years for a fast smoke
        full = matrix.index

    # calendar futures
    idx = matrix.index
    cal = pd.DataFrame(index=idx)
    cal["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    cal["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    cal["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 366.0)
    cal["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 366.0)

    long_parts, mask_parts = [], []
    for col in series:
        y_obs = matrix[col]
        observed = y_obs.notna()
        if observed.sum() < 5000:
            continue
        first_obs = y_obs.first_valid_index()
        if first_obs is None:
            continue
        # Fill with past-only information. Drop leading rows where no past value exists.
        y = past_only_fill(y_obs.loc[first_obs:]).dropna()
        series_idx = y.index
        df = pd.DataFrame({"unique_id": col, "ds": series_idx, "y": y.values})
        for f in FUTR:
            df[f] = cal.loc[series_idx, f].values
        miss = target_missingness_features(observed.loc[series_idx])
        for f in TARGET_HIST_EXOG:
            df[f] = miss[f].values
        long_parts.append(df)
        mask_parts.append(pd.DataFrame({
            "unique_id": col,
            "ds": series_idx,
            "observed": observed.loc[series_idx].values,
        }))

    long_df = pd.concat(long_parts, ignore_index=True)
    mask = pd.concat(mask_parts, ignore_index=True)
    return long_df, mask


# Per-model exogenous capability map. Verified against the installed neuralforecast (3.1.9)
# class attributes EXOGENOUS_FUTR / EXOGENOUS_HIST / EXOGENOUS_STAT (the authoritative flags
# the library itself uses to validate exog inputs). multivariate iTransformer takes NO exog.
EXOG_CAPS = {
    "tft":          {"futr": True,  "hist": True,  "stat": True},
    "nhits":        {"futr": True,  "hist": True,  "stat": True},
    "nbeatsx":      {"futr": True,  "hist": True,  "stat": True},
    "itransformer": {"futr": False, "hist": False, "stat": False},
    "patchtst":     {"futr": False, "hist": False, "stat": False},
    "tsmixerx":     {"futr": True,  "hist": True,  "stat": True},
    "dlinear":      {"futr": False, "hist": False, "stat": False},
    "nlinear":      {"futr": False, "hist": False, "stat": False},
}


def make_model(name: str, h: int, input_size: int, max_steps: int,
               loss_kind: str = "mae", futr_extra: list[str] | None = None,
               hist_cols: list[str] | None = None, logfn=print):
    import os
    from neuralforecast.models import TFT, NHITS, NBEATSx, iTransformer, PatchTST, TSMixerx, DLinear, NLinear
    from neuralforecast.losses.pytorch import MAE, MQLoss
    # Bounded batches keep memory sane: windows_batch_size caps train-step windows;
    # inference_windows_batch_size caps the predict allocation (the source of the 81 GiB OOM).
    # All four are env-overridable so a diagnostic run can shrink batches (sm_120 cudaErrorUnknown) without code edits.
    def _bi(name, default):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
    bounded = dict(batch_size=_bi("MBARI_BATCH_SIZE", 24),
                   windows_batch_size=_bi("MBARI_WINDOWS_BATCH", 256),
                   inference_windows_batch_size=_bi("MBARI_INFER_WINDOWS_BATCH", 512),
                   valid_batch_size=_bi("MBARI_VALID_BATCH", 256),
                   # num_workers=0: avoid Windows spawn-worker DLL crashes.
                   # pin_memory=False: pinned host buffers are non-pageable commit that adds to
                   # the system-commit pressure behind the terminal OOMs, for no H2D win at these batch sizes.
                   dataloader_kwargs=dict(num_workers=0, pin_memory=False))
    accel = os.environ.get("MBARI_ACCEL", "auto")  # cpu|gpu|auto -- gpu hard-crashes (sm_120/torch2.11) at full scale
    if accel != "auto":
        bounded["accelerator"] = accel
        bounded["devices"] = 1

    n = name.lower()
    caps = EXOG_CAPS.get(n, {"futr": False, "hist": False, "stat": False})
    loss = MQLoss(quantiles=QUANTILES) if loss_kind == "quantile" else MAE()

    # Build the futr/hist exog lists this model is allowed to receive.
    # Calendar FUTR is the existing baseline; futr_extra are driver columns known into the future.
    futr_extra = futr_extra or []
    hist_cols = hist_cols or []
    full_futr = FUTR + [c for c in futr_extra if c not in FUTR]
    # Gate by capability: warn-and-drop exog kinds a model cannot accept (never crash).
    if futr_extra and not caps["futr"]:
        logfn(f"[exog] WARNING: model {n} has no futr-exog support; dropping {len(futr_extra)} driver futr col(s): {futr_extra}")
    if hist_cols and not caps["hist"]:
        logfn(f"[exog] WARNING: model {n} has no hist-exog support; dropping {len(hist_cols)} driver hist col(s): {hist_cols}")
    use_futr = full_futr if caps["futr"] else None
    hist_with_target_state = list(dict.fromkeys(TARGET_HIST_EXOG + (hist_cols or [])))
    use_hist = (hist_with_target_state if hist_with_target_state else None) if caps["hist"] else None
    if caps["futr"] and futr_extra:
        logfn(f"[exog] model {n} futr_exog_list (calendar+drivers) = {use_futr}")
    if caps["hist"] and use_hist:
        logfn(f"[exog] model {n} hist_exog_list (target-state+observed-only drivers) = {use_hist}")

    base = dict(h=h, input_size=input_size, max_steps=max_steps, scaler_type="standard",
                loss=loss, random_seed=42, enable_progress_bar=False, **bounded)
    exog = {}
    if caps["futr"] and use_futr:
        exog["futr_exog_list"] = use_futr
    if caps["hist"] and use_hist:
        exog["hist_exog_list"] = use_hist

    if n == "tft":
        return TFT(**base, **exog, hidden_size=64, n_head=4)
    if n == "nhits":
        return NHITS(**base, **exog)
    if n == "nbeatsx":
        return NBEATSx(**base, **exog)
    if n == "itransformer":
        return iTransformer(**base, n_series=common_n_series)
    if n == "patchtst":
        return PatchTST(**base)
    if n == "tsmixerx":
        return TSMixerx(**base, **exog, n_series=common_n_series)
    if n == "dlinear":
        return DLinear(**base)
    if n == "nlinear":
        return NLinear(**base)
    raise ValueError(f"unknown model {name}")


common_n_series = 24  # set at runtime


def evaluate(cv: pd.DataFrame, mask: pd.DataFrame, long_df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Score ONLY originally-observed test points. horizon=(ds-cutoff) in hours.
    Persistence prediction = the series value at the forecast origin (ds==cutoff), compared on
    the exact same masked test points as the model -> rigorous skill = 1 - rmse_model/rmse_persist."""
    cv = cv.merge(mask, on=["unique_id", "ds"], how="left")
    cv = cv[cv["observed"].fillna(False)].copy()
    cv["horizon_h"] = ((cv["ds"] - cv["cutoff"]) / pd.Timedelta(hours=1)).round().astype(int)
    origin = long_df[["unique_id", "ds", "y"]].rename(columns={"ds": "cutoff", "y": "y_origin"})
    cv = cv.merge(origin, on=["unique_id", "cutoff"], how="left")
    origin_mask = mask.rename(columns={"ds": "cutoff", "observed": "origin_observed"})
    cv = cv.merge(origin_mask, on=["unique_id", "cutoff"], how="left")
    cv = cv[cv["origin_observed"].fillna(False)].copy()
    rows = []
    for (uid, hh), g in cv.groupby(["unique_id", "horizon_h"]):
        if hh not in HORIZONS or len(g) < 3:
            continue
        m_err = (g[model_col] - g["y"]).to_numpy()
        p_err = (g["y_origin"] - g["y"]).to_numpy()
        m_rmse = float(np.sqrt(np.mean(m_err ** 2)))
        p_rmse = float(np.sqrt(np.mean(p_err ** 2))) if np.isfinite(p_err).all() else float("nan")
        rows.append({
            "unique_id": uid, "horizon_h": hh,
            "model_rmse": round(m_rmse, 5), "model_mae": round(float(np.mean(np.abs(m_err))), 5),
            "persistence_rmse": round(p_rmse, 5),
            "skill_vs_persistence_pct": round((1 - m_rmse / p_rmse) * 100, 2) if p_rmse and np.isfinite(p_rmse) else None,
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


def load_drivers(drivers_parquet: str, drivers_manifest: str, logfn=print) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a WIDE drivers parquet (indexed by ds, hourly UTC) + JSON manifest.
    Returns (wide_df_with_ds_column, futr_cols, hist_cols).
    futr = deterministic/known-future drivers; hist = observed-only drivers (no leakage)."""
    manifest = json.loads(Path(drivers_manifest).read_text(encoding="utf-8"))
    futr_cols = list(manifest.get("futr", []))
    hist_cols = list(manifest.get("hist", []))
    wide = pd.read_parquet(drivers_parquet)
    # Normalise the time index into a 'ds' column.
    if "ds" not in wide.columns:
        wide = wide.reset_index()
        if "ds" not in wide.columns:
            wide = wide.rename(columns={wide.columns[0]: "ds"})
    wide["ds"] = pd.to_datetime(wide["ds"])
    requested = futr_cols + hist_cols
    missing = [c for c in requested if c not in wide.columns]
    if missing:
        raise ValueError(f"drivers parquet missing manifest columns: {missing}")
    wide = wide[["ds"] + [c for c in requested if c in wide.columns]].drop_duplicates("ds")
    logfn(f"[drivers] loaded {len(wide):,} rows, futr={futr_cols} hist={hist_cols} "
          f"span={wide.ds.min()}->{wide.ds.max()}")
    return wide, futr_cols, hist_cols


def merge_drivers(long_df: pd.DataFrame, drivers_wide: pd.DataFrame, logfn=print) -> pd.DataFrame:
    """Broadcast the same driver values to every unique_id by merging on ds.
    Driver gaps are filled with 0 so NeuralForecast gets a dense exog frame without
    extending stale observations or using full-period summary statistics."""
    driver_cols = [c for c in drivers_wide.columns if c != "ds"]
    merged = long_df.merge(drivers_wide, on="ds", how="left")
    n_missing = int(merged[driver_cols].isna().any(axis=1).sum()) if driver_cols else 0
    if n_missing:
        logfn(f"[drivers] {n_missing:,} long rows had missing driver values for ds; zero-filled gaps")
    merged = merged.sort_values(["unique_id", "ds"])
    for c in driver_cols:
        merged[c] = merged[c].fillna(0.0)
    return merged.reset_index(drop=True)


def apply_tail_weeks(long_df: pd.DataFrame, mask: pd.DataFrame, tail_weeks: float | None,
                     logfn=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the most recent N weeks of hourly data.

    This is used for exogenous models whose full 20+ year history causes NeuralForecast to
    materialize impossible host-memory window tensors. The data stays hourly and keeps all
    selected series/drivers, but bounds the trainable history length.
    """
    if tail_weeks is None or tail_weeks <= 0:
        return long_df, mask
    max_ds = pd.to_datetime(long_df["ds"].max())
    cutoff = max_ds - pd.Timedelta(weeks=float(tail_weeks))
    before_rows = len(long_df)
    before_mask = len(mask)
    long_df = long_df[long_df["ds"] >= cutoff].reset_index(drop=True)
    mask = mask[mask["ds"] >= cutoff].reset_index(drop=True)
    logfn(
        f"[tail] kept last {tail_weeks:g} weeks from {cutoff} to {max_ds}: "
        f"rows {before_rows:,}->{len(long_df):,}, mask {before_mask:,}->{len(mask):,}"
    )
    return long_df, mask


def synthesize_dummy_drivers(long_df: pd.DataFrame, out_parquet: Path, out_manifest: Path) -> None:
    """Write a tiny dummy drivers parquet + manifest spanning the long_df ds range, to exercise
    the futr/hist plumbing before a real driver file exists. 2 futr cols + 1 hist col."""
    ds = pd.date_range(long_df.ds.min(), long_df.ds.max(), freq="1h")
    h = ds.hour.to_numpy()
    doy = ds.dayofyear.to_numpy()
    wide = pd.DataFrame({
        "ds": ds,
        "tide_sin": np.sin(2 * np.pi * h / 12.42),          # futr: deterministic tidal proxy
        "season_cos": np.cos(2 * np.pi * doy / 365.25),     # futr: deterministic seasonal proxy
        "upwelling_index": np.cos(2 * np.pi * h / 24.0) * 0.5,  # hist: pretend observed-only driver
    })
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_parquet, index=False)
    out_manifest.write_text(json.dumps({
        "futr": ["tide_sin", "season_cos"],
        "hist": ["upwelling_index"],
        "note": "synthetic dummy drivers for contract test",
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--n-windows", type=int, default=None)
    ap.add_argument("--step-size", type=int, default=168, help="hours between rolling forecast origins")
    ap.add_argument("--series", nargs="+", default=None, help="optional unique_id list, e.g. temp_d10p0 air_pressure")
    ap.add_argument("--tail-weeks", type=float, default=None,
                    help="keep only the most recent N weeks before training/CV; bounds host-memory windows")
    ap.add_argument("--drivers-parquet", default=None, help="WIDE drivers parquet indexed by ds (hourly UTC)")
    ap.add_argument("--drivers-manifest", default=None, help="JSON manifest: {'futr':[known-future cols], 'hist':[observed-only cols]}")
    ap.add_argument("--loss", choices=["mae", "quantile"], default="mae",
                    help="mae (default, point) or quantile (MQLoss q=[0.1,0.5,0.9], adds interval coverage)")
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("w", encoding="utf-8")
    def P(*x):
        print(*x, flush=True); print(*x, file=log, flush=True)

    t0 = time.time()
    import torch
    torch.set_float32_matmul_precision("high")
    P(f"model={a.model} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    h = 168
    input_size = a.input_size or (96 if a.smoke else 168)  # 1-week lookback; annual cycle comes from doy futr exog. >168 hard-crashes GPU (sm_120)
    max_steps = a.max_steps or (200 if a.smoke else 1500)
    n_windows = a.n_windows or (8 if a.smoke else 52)  # weekly origins; 1yr seasonal test in full
    run_config = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "n_windows": n_windows,
        "step_size": a.step_size,
        "smoke": a.smoke,
        "series_filter": a.series,
        "tail_weeks": a.tail_weeks,
        "loss": a.loss,
    }

    long_df, mask = build_long_frame(a.smoke, a.series)
    long_df, mask = apply_tail_weeks(long_df, mask, a.tail_weeks, P)
    global common_n_series
    common_n_series = long_df["unique_id"].nunique()
    if common_n_series <= 0:
        raise ValueError("no series selected for neural run")
    P(f"series={common_n_series} rows={len(long_df):,} span={long_df.ds.min()}->{long_df.ds.max()}")
    P(f"unique_ids={sorted(long_df['unique_id'].unique())}")
    P(f"h={h} input_size={input_size} max_steps={max_steps} n_windows={n_windows}")

    # Optional exogenous drivers. --smoke with no real driver file synthesizes a dummy one to
    # exercise the futr/hist plumbing (contract test).
    drivers_parquet, drivers_manifest = a.drivers_parquet, a.drivers_manifest
    futr_extra, hist_cols = [], []
    if drivers_parquet and drivers_manifest:
        drivers_wide, futr_extra, hist_cols = load_drivers(drivers_parquet, drivers_manifest, P)
        long_df = merge_drivers(long_df, drivers_wide, P)
        P(f"[drivers] merged; long_df now has cols {[c for c in long_df.columns if c not in ('unique_id','ds','y')]}")
    run_config["drivers_parquet"] = drivers_parquet
    run_config["drivers_manifest"] = drivers_manifest
    run_config["driver_futr_cols"] = futr_extra
    run_config["driver_hist_cols"] = hist_cols

    P(f"loss={a.loss}")
    from neuralforecast import NeuralForecast
    model = make_model(a.model, h, input_size, max_steps,
                       loss_kind=a.loss, futr_extra=futr_extra, hist_cols=hist_cols, logfn=P)
    nf = NeuralForecast(models=[model], freq="h")

    P("cross_validation starting ...")
    cv = nf.cross_validation(df=long_df, n_windows=n_windows, step_size=a.step_size)
    reserved = ("unique_id", "ds", "cutoff", "y")
    pred_cols = [c for c in cv.columns if c not in reserved]
    cv.to_parquet(out / "cv_predictions.parquet")

    # Column selection: MAE -> one point column; MQLoss -> *-lo-80.0 / *-median / *-hi-80.0.
    median_col = lo_col = hi_col = None
    if a.loss == "quantile":
        median_col = next((c for c in pred_cols if c.endswith("-median")), None)
        lo_col = next((c for c in pred_cols if "-lo-" in c), None)
        hi_col = next((c for c in pred_cols if "-hi-" in c), None)
        if median_col is None or lo_col is None or hi_col is None:
            raise ValueError(f"quantile loss: could not locate median/lo/hi among {pred_cols}")
        model_col = median_col  # score point skill on the median
        P(f"quantile columns: median={median_col} lo={lo_col} hi={hi_col}")
    else:
        if len(pred_cols) != 1:
            raise ValueError(f"expected exactly one model prediction column, found {pred_cols}")
        model_col = pred_cols[0]

    res = evaluate(cv, mask, long_df, model_col)
    # attach persistence/climatology reference from the data profile if present, else compute
    res["model"] = a.model
    res.to_csv(out / "leaderboard.csv", index=False)

    mean_rmse_by_h = {}
    if not res.empty:
        mean_rmse_by_h = res.groupby("horizon_h")["model_rmse"].mean().round(4).to_dict()

    # Interval coverage for quantile loss: fraction of observed test points inside [q0.1, q0.9].
    interval_metrics = {}
    if a.loss == "quantile":
        cov = cv.merge(mask, on=["unique_id", "ds"], how="left")
        cov = cov[cov["observed"].fillna(False)].copy()
        # restrict to scored horizons for consistency with skill scoring
        cov["horizon_h"] = ((cov["ds"] - cov["cutoff"]) / pd.Timedelta(hours=1)).round().astype(int)
        cov = cov[cov["horizon_h"].isin(HORIZONS)].copy()
        if len(cov):
            inside = (cov["y"] >= cov[lo_col]) & (cov["y"] <= cov[hi_col])
            interval_metrics = {
                "interval_coverage_0.8": round(float(inside.mean()), 4),
                "mean_interval_width_0.8": round(float((cov[hi_col] - cov[lo_col]).mean()), 5),
                "interval_n": int(len(cov)),
                "quantiles": QUANTILES,
            }
        else:
            interval_metrics = {"interval_coverage_0.8": None, "mean_interval_width_0.8": None,
                                "interval_n": 0, "quantiles": QUANTILES}
        P(f"interval_coverage_0.8={interval_metrics['interval_coverage_0.8']} "
          f"mean_width={interval_metrics['mean_interval_width_0.8']} n={interval_metrics['interval_n']}")

    summary = {
        "model": a.model, "n_series": int(common_n_series), "h": h, "input_size": input_size,
        "max_steps": max_steps, "n_windows": n_windows, "smoke": a.smoke,
        "loss": a.loss, "tail_weeks": a.tail_weeks,
        "drivers": {"parquet": drivers_parquet, "manifest": drivers_manifest,
                    "futr_cols": futr_extra, "hist_cols": hist_cols} if (drivers_parquet and drivers_manifest) else None,
        "cache_version": CACHE_VERSION,
        "preprocessing": {
            "target_fill": "past-only: observed values, ffill<=48h, prior same-doy mean, prior global mean",
            "scoring_mask": "target ds observed and cutoff/origin observed",
            "target_hist_exog": TARGET_HIST_EXOG,
        },
        "series": sorted(long_df["unique_id"].unique()),
        "minutes": round((time.time() - t0) / 60, 2),
        "scored_rows": int(len(res)),
        "mean_rmse_by_h": mean_rmse_by_h,
    }
    summary.update(interval_metrics)  # adds interval_coverage_0.8 / mean_interval_width_0.8 for --loss quantile
    contract_ids = emit_lakehouse_contracts(
        out=out,
        cv=cv,
        res=res,
        long_df=long_df,
        summary=summary,
        model_col=model_col,
        config=run_config,
        pred_cols=pred_cols,
    )
    summary.update(contract_ids)
    P(f"lakehouse run_id={contract_ids['run_id']} split_id={contract_ids['split_id']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    P(f"DONE {a.model} in {summary['minutes']} min")
    P("mean RMSE by horizon: " + json.dumps(summary["mean_rmse_by_h"]))
    log.close()


if __name__ == "__main__":
    main()
