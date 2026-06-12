#!/usr/bin/env python
"""Release gate for the local Monterey Bay AI Lab analysis setup.

The gate is intentionally read-only outside this directory. It inspects the
GPU/ML stack, historical Parquet artifacts, and model result JSON files, then
writes machine-readable and human-readable reports.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbal_lakehouse import read_forecast_metrics


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
REQUIRED_PARQUET_COLUMNS = {
    "source",
    "dataset_id",
    "station",
    "time",
    "depth_m",
    "latitude",
    "longitude",
}
CORE_SIGNAL_COLUMNS = (
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "air_pressure",
    "air_temperature",
    "relative_humidity",
    "wind_speed_sonic",
    "current_speed",
)
RANGE_LIMITS = {
    "sea_water_temperature": (-5.0, 40.0),
    "sea_water_practical_salinity": (0.0, 45.0),
    "air_pressure": (800.0, 1100.0),
    "air_temperature": (-20.0, 50.0),
    "relative_humidity": (0.0, 100.0),
    "wind_speed_sonic": (0.0, 75.0),
    "wind_speed_windbird": (0.0, 75.0),
    "eastward_sea_water_velocity": (-5.0, 5.0),
    "northward_sea_water_velocity": (-5.0, 5.0),
    "current_speed": (0.0, 7.0),
    "wind_from_direction_sonic": (0.0, 360.0),
    "wind_from_direction_windbird": (0.0, 360.0),
}
QC_COLUMNS = (
    "sea_water_temperature_qc",
    "sea_water_practical_salinity_qc",
    "air_pressure_qc",
    "air_temperature_qc",
    "relative_humidity_qc",
    "wind_speed_sonic_qc",
    "wind_from_direction_sonic_qc",
)
QC_VALUE_COLUMNS = {
    "sea_water_temperature_qc": "sea_water_temperature",
    "sea_water_practical_salinity_qc": "sea_water_practical_salinity",
    "air_pressure_qc": "air_pressure",
    "air_temperature_qc": "air_temperature",
    "relative_humidity_qc": "relative_humidity",
    "wind_speed_sonic_qc": "wind_speed_sonic",
    "wind_from_direction_sonic_qc": "wind_from_direction_sonic",
}


@dataclass
class Check:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def combine_status(*statuses: str) -> str:
    worst = "PASS"
    for status in statuses:
        if STATUS_ORDER[status] > STATUS_ORDER[worst]:
            worst = status
    return worst


def module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def run_command(args: list[str], timeout_s: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "error": "not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout_s}s"}


def check_gpu_stack() -> Check:
    details: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout_s=15,
        ),
    }
    statuses: list[str] = []
    messages: list[str] = []

    try:
        import torch

        details["torch_version"] = torch.__version__
        details["torch_cuda_version"] = torch.version.cuda
        details["torch_cuda_available"] = bool(torch.cuda.is_available())
        details["torch_arch_list"] = list(torch.cuda.get_arch_list())
        if not torch.cuda.is_available():
            statuses.append("FAIL")
            messages.append("PyTorch CUDA is unavailable")
        else:
            device_name = torch.cuda.get_device_name(0)
            details["torch_device_name"] = device_name
            details["torch_device_capability"] = torch.cuda.get_device_capability(0)
            x = torch.randn((1024, 1024), device="cuda")
            y = torch.randn((1024, 1024), device="cuda")
            z = x @ y
            torch.cuda.synchronize()
            details["torch_kernel_mean"] = float(z.mean().detach().cpu())
            if "RTX 5090" in device_name and "sm_120" not in details["torch_arch_list"]:
                statuses.append("FAIL")
                messages.append("RTX 5090 detected but PyTorch build lacks sm_120")
            else:
                statuses.append("PASS")
                messages.append(f"PyTorch CUDA kernel passed on {device_name}")
    except Exception as exc:
        statuses.append("FAIL")
        details["torch_error"] = f"{type(exc).__name__}: {exc}"
        details["torch_traceback"] = traceback.format_exc(limit=4)
        messages.append("PyTorch CUDA check failed")

    try:
        import numpy as np
        import xgboost as xgb

        details["xgboost_version"] = xgb.__version__
        rng = np.random.default_rng(17)
        x_train = rng.normal(size=(2048, 8))
        y_train = x_train[:, 0] * 0.7 - x_train[:, 3] * 0.2 + rng.normal(0, 0.01, 2048)
        model = xgb.XGBRegressor(
            n_estimators=16,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            random_state=17,
        )
        model.fit(x_train, y_train)
        booster_config = json.loads(model.get_booster().save_config())
        details["xgboost_booster_config"] = booster_config.get("learner", {})
        statuses.append("PASS")
        messages.append("XGBoost CUDA smoke training passed")
    except Exception as exc:
        statuses.append("FAIL")
        details["xgboost_error"] = f"{type(exc).__name__}: {exc}"
        details["xgboost_traceback"] = traceback.format_exc(limit=4)
        messages.append("XGBoost CUDA check failed")

    status = combine_status(*statuses) if statuses else "FAIL"
    return Check("gpu_stack", status, "; ".join(messages), details)


def check_package_versions() -> Check:
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "xgboost",
        "numpy",
        "pandas",
        "pyarrow",
        "sklearn",
        "polars",
        "duckdb",
        "xarray",
        "netCDF4",
    ]
    versions = {pkg: module_version(pkg) for pkg in packages}
    details: dict[str, Any] = {"versions": versions}
    statuses: list[str] = []
    messages: list[str] = []

    missing_required = [pkg for pkg in ("torch", "xgboost", "pandas", "pyarrow") if not versions[pkg]]
    if missing_required:
        statuses.append("FAIL")
        messages.append(f"missing required packages: {', '.join(missing_required)}")

    torch_family = {pkg: versions[pkg] for pkg in ("torch", "torchvision", "torchaudio") if versions[pkg]}
    cuda_suffixes = {
        pkg: version.split("+", 1)[1] if "+" in version else None
        for pkg, version in torch_family.items()
    }
    details["torch_cuda_suffixes"] = cuda_suffixes
    suffix_values = {suffix for suffix in cuda_suffixes.values() if suffix}
    if torch_family and len(torch_family) < 3:
        statuses.append("WARN")
        messages.append("torch/torchvision/torchaudio family is incomplete")
    if suffix_values and len(suffix_values) > 1:
        statuses.append("FAIL")
        messages.append("torch package CUDA suffixes are inconsistent")
    elif torch_family and not suffix_values:
        statuses.append("WARN")
        messages.append("torch packages do not expose CUDA build suffixes")

    missing_optional = [pkg for pkg in ("polars", "duckdb", "xarray", "netCDF4") if not versions[pkg]]
    if missing_optional:
        statuses.append("WARN")
        messages.append(f"optional data packages missing: {', '.join(missing_optional)}")

    if not statuses:
        statuses.append("PASS")
        messages.append("required package versions are present and consistent")

    return Check(
        "package_versions",
        combine_status(*statuses),
        "; ".join(messages),
        details,
    )


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def check_single_parquet(path: Path) -> Check:
    details: dict[str, Any] = {"path": str(path)}
    statuses: list[str] = []
    messages: list[str] = []
    try:
        import pandas as pd
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema.names)
        details["num_rows_metadata"] = parquet_file.metadata.num_rows
        details["num_row_groups"] = parquet_file.metadata.num_row_groups
        details["columns"] = list(parquet_file.schema.names)
        if parquet_file.metadata.num_rows <= 0:
            statuses.append("FAIL")
            messages.append("Parquet file has zero rows")
        missing_columns = sorted(REQUIRED_PARQUET_COLUMNS - schema_names)
        if missing_columns:
            statuses.append("FAIL")
            messages.append(f"missing required columns: {', '.join(missing_columns)}")

        needed_columns = sorted(
            (REQUIRED_PARQUET_COLUMNS | set(CORE_SIGNAL_COLUMNS) | set(RANGE_LIMITS) | set(QC_COLUMNS))
            & schema_names
        )
        df = pd.read_parquet(path, columns=needed_columns)
        details["rows_read"] = int(len(df))
        details["stations"] = sorted(str(x) for x in df["station"].dropna().unique()) if "station" in df else []
        if {"station", "time", "depth_m"}.issubset(df.columns):
            dup_count = int(df.duplicated(["station", "time", "depth_m"]).sum())
            details["duplicate_station_time_depth_rows"] = dup_count
            if dup_count:
                statuses.append("FAIL")
                messages.append(f"{dup_count} duplicate station/time/depth rows")
            time_values = pd.to_datetime(df["time"], errors="coerce", utc=True)
            null_times = int(time_values.isna().sum())
            details["null_or_invalid_times"] = null_times
            if null_times:
                statuses.append("FAIL")
                messages.append(f"{null_times} null/invalid timestamps")
            else:
                details["time_min"] = str(time_values.min())
                details["time_max"] = str(time_values.max())

        non_null_counts: dict[str, int] = {}
        range_violations: dict[str, int] = {}
        for column in CORE_SIGNAL_COLUMNS:
            if column in df.columns:
                non_null_counts[column] = int(df[column].notna().sum())
        details["non_null_counts"] = non_null_counts
        if non_null_counts:
            useful_columns = [name for name, count in non_null_counts.items() if count > 0]
            if not useful_columns:
                statuses.append("FAIL")
                messages.append("no non-null core sensor signals")
            elif len(useful_columns) < 3:
                statuses.append("WARN")
                messages.append("fewer than three core sensor signals have data")

        for column, (low, high) in RANGE_LIMITS.items():
            if column not in df.columns:
                continue
            series = df[column].dropna()
            if series.empty:
                continue
            bad = int(((series < low) | (series > high)).sum())
            if bad:
                range_violations[column] = bad
        details["range_violations"] = range_violations
        if range_violations:
            statuses.append("FAIL")
            messages.append("physical range violations found")

        qc_retained_bad_flags: dict[str, list[float]] = {}
        qc_bad_value_leaks: dict[str, int] = {}
        for column in QC_COLUMNS:
            if column not in df.columns:
                continue
            values = sorted(float(x) for x in df[column].dropna().unique())
            bad_values = [x for x in values if x not in (1.0, 2.0)]
            if bad_values:
                qc_retained_bad_flags[column] = bad_values[:20]
                value_column = QC_VALUE_COLUMNS.get(column)
                if value_column and value_column in df.columns:
                    leaked = int((df[column].round().isin(bad_values) & df[value_column].notna()).sum())
                    if leaked:
                        qc_bad_value_leaks[value_column] = leaked
        details["qc_retained_bad_flags"] = qc_retained_bad_flags
        details["qc_bad_value_leaks"] = qc_bad_value_leaks
        if qc_bad_value_leaks:
            statuses.append("FAIL")
            messages.append("bad-QC measurements remain non-null")

        if not statuses:
            statuses.append("PASS")
            messages.append("Parquet integrity checks passed")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate Parquet file")

    return Check(f"parquet::{path.name}", combine_status(*statuses), "; ".join(messages), details)


def check_auxiliary_parquet(path: Path) -> Check:
    details: dict[str, Any] = {"path": str(path)}
    statuses: list[str] = []
    messages: list[str] = []
    try:
        import pandas as pd
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema_arrow.names)
        details["columns"] = list(parquet_file.schema_arrow.names)
        details["num_rows_metadata"] = int(parquet_file.metadata.num_rows)
        details["num_row_groups"] = int(parquet_file.metadata.num_row_groups)
        if parquet_file.metadata.num_rows <= 0:
            statuses.append("FAIL")
            messages.append("auxiliary parquet has no rows")
        if "time" not in schema_names:
            statuses.append("FAIL")
            messages.append("auxiliary parquet is missing time column")
            read_columns = list(schema_names)[:10]
        else:
            read_columns = ["time"] + sorted(c for c in schema_names if c != "time")[:12]
        df = pd.read_parquet(path, columns=read_columns)
        details["rows_read"] = int(len(df))
        if "time" in df.columns:
            time_values = pd.to_datetime(df["time"], errors="coerce", utc=True)
            null_times = int(time_values.isna().sum())
            details["null_or_invalid_times"] = null_times
            if null_times:
                statuses.append("FAIL")
                messages.append(f"{null_times} null/invalid timestamps")
            else:
                details["time_min"] = str(time_values.min())
                details["time_max"] = str(time_values.max())
        numeric_cols = [c for c in df.columns if c != "time"]
        non_null_counts = {c: int(df[c].notna().sum()) for c in numeric_cols}
        details["non_null_counts"] = non_null_counts
        if not any(count > 0 for count in non_null_counts.values()):
            statuses.append("FAIL")
            messages.append("auxiliary parquet has no non-null driver values")
        if not statuses:
            statuses.append("PASS")
            messages.append("auxiliary driver parquet sanity checks passed")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate auxiliary parquet")

    return Check(f"aux_parquet::{path.name}", combine_status(*statuses), "; ".join(messages), details)


def check_historical_parquet(project_root: Path) -> Check:
    parquet_paths = sorted(project_root.glob("mbal_history/**/*.parquet"))
    details: dict[str, Any] = {"files": [str(path) for path in parquet_paths]}
    if not parquet_paths:
        return Check(
            "historical_parquet",
            "WARN",
            "no historical Parquet files found under mbal_history",
            details,
        )

    production_paths = [path for path in parquet_paths if path.parent.name == "opendap" and path.name.endswith("_history.parquet")]
    auxiliary_paths = [path for path in parquet_paths if path not in production_paths]
    production_checks = [check_single_parquet(path) for path in production_paths]
    auxiliary_checks = [check_auxiliary_parquet(path) for path in auxiliary_paths]
    details["production_checks"] = [check.__dict__ for check in production_checks]
    details["auxiliary_checks"] = [check.__dict__ for check in auxiliary_checks]
    details["checks"] = details["production_checks"] + details["auxiliary_checks"]
    if not production_checks:
        return Check(
            "historical_parquet",
            "FAIL",
            "trusted OPeNDAP M1/M2 history parquet files are missing",
            details,
        )
    production_status = combine_status(*(check.status for check in production_checks))
    auxiliary_status = combine_status(*(check.status for check in auxiliary_checks)) if auxiliary_checks else "PASS"
    if production_status == "FAIL":
        status = "FAIL"
    elif auxiliary_status in {"FAIL", "WARN"}:
        status = "WARN"
    else:
        status = production_status
    pass_count = sum(1 for check in production_checks if check.status == "PASS")
    warn_count = sum(1 for check in production_checks if check.status == "WARN")
    fail_count = sum(1 for check in production_checks if check.status == "FAIL")
    aux_pass_count = sum(1 for check in auxiliary_checks if check.status == "PASS")
    aux_warn_count = sum(1 for check in auxiliary_checks if check.status == "WARN")
    aux_fail_count = sum(1 for check in auxiliary_checks if check.status == "FAIL")
    return Check(
        "historical_parquet",
        status,
        (
            f"{pass_count} PASS, {warn_count} WARN, {fail_count} FAIL across "
            f"{len(production_checks)} trusted OPeNDAP Parquet files; "
            f"{aux_pass_count} PASS, {aux_warn_count} WARN, {aux_fail_count} FAIL across "
            f"{len(auxiliary_checks)} auxiliary Parquet files inspected"
        ),
        details,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_curated_dataset(project_root: Path) -> Check:
    dataset_dir = project_root / "mbal_pipeline" / "curated_history"
    report_path = project_root / "mbal_pipeline" / "reports" / "history_validation.json"
    details: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "validation_report": str(report_path),
    }
    statuses: list[str] = []
    messages: list[str] = []

    if not dataset_dir.exists():
        return Check("curated_dataset", "WARN", "curated partitioned dataset is missing", details)
    if not report_path.exists():
        return Check("curated_dataset", "WARN", "curated dataset validation report is missing", details)

    try:
        report = load_json(report_path)
        expected_rows = int(report.get("total_rows", -1))
        total_errors = int(report.get("total_errors", -1))
        total_warnings = int(report.get("total_warnings", -1))
        details["expected_rows"] = expected_rows
        details["reported_errors"] = total_errors
        details["reported_warnings"] = total_warnings
        if total_errors:
            statuses.append("FAIL")
            messages.append(f"validation report has {total_errors} errors")
        if total_warnings:
            statuses.append("WARN")
            messages.append(f"validation report has {total_warnings} warnings")

        parquet_files = sorted(dataset_dir.rglob("*.parquet"))
        details["parquet_file_count"] = len(parquet_files)
        if not parquet_files:
            statuses.append("FAIL")
            messages.append("curated dataset has no parquet files")
        else:
            import pyarrow.dataset as ds

            dataset = ds.dataset(dataset_dir, format="parquet", partitioning="hive")
            actual_rows = int(dataset.count_rows())
            details["actual_rows"] = actual_rows
            details["schema_columns"] = dataset.schema.names
            for column in ("station", "year", "month"):
                if column not in dataset.schema.names:
                    statuses.append("FAIL")
                    messages.append(f"missing partition column {column}")
            if expected_rows >= 0 and actual_rows != expected_rows:
                statuses.append("FAIL")
                messages.append(f"curated row count {actual_rows} != validation report {expected_rows}")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate curated dataset")

    if not statuses:
        statuses.append("PASS")
        messages.append("curated partitioned dataset matches validation report")
    return Check("curated_dataset", combine_status(*statuses), "; ".join(messages), details)


def find_metric_values(record: dict[str, Any]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for key, value in record.items():
        if key in {"mae", "rmse", "r2"} or key.endswith(("_mae", "_rmse", "_r2")) or "skill" in key:
            if isinstance(value, (int, float)):
                values.append((key, float(value)))
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                for metric_name, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        values.append((f"metrics.{model_name}.{metric_name}", float(value)))
    return values


def check_model_result_file(path: Path) -> Check:
    details: dict[str, Any] = {"path": str(path)}
    statuses: list[str] = []
    messages: list[str] = []
    try:
        payload = load_json(path)
        run_config_path = path.parent / "run_config.json"
        run_config = load_json(run_config_path) if run_config_path.exists() else {}
        details["run_config"] = run_config
        is_production_result = path.parent.name == "mbal_forecast_v2_results"
        if is_production_result:
            if not run_config_path.exists():
                statuses.append("FAIL")
                messages.append("production run_config.json is missing")
            if run_config.get("smoke") is not False:
                statuses.append("FAIL")
                messages.append("production result is not a full non-smoke run")
            if run_config.get("source") != "parquet":
                statuses.append("FAIL")
                messages.append("production result is not sourced from historical parquet")
            if not str(run_config.get("path", "")).replace("\\", "/").endswith("mbal_history/opendap/m1_history.parquet"):
                statuses.append("FAIL")
                messages.append("production result is not pointed at trusted M1 OPeNDAP history")
        records = payload if isinstance(payload, list) else payload.get("results", [])
        if not isinstance(records, list) or not records:
            return Check(f"models::{path.parent.name}", "FAIL", "model_results.json has no result records", details)
        details["record_count"] = len(records)

        non_finite_metrics: list[dict[str, Any]] = []
        missing_split: list[str] = []
        gpu_records = 0
        benchmarked_records = 0
        beats_persistence = 0
        fails_persistence: list[str] = []
        negative_fold_skill: list[str] = []
        weak_records: list[str] = []
        tiny_records: list[str] = []

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                non_finite_metrics.append({"index": index, "reason": "record is not an object"})
                continue
            target = str(record.get("target", f"record_{index}"))
            rows = record.get("rows", record.get("usable_rows"))
            train_rows = record.get("train_rows")
            test_rows = record.get("test_rows")
            folds = record.get("folds")
            if finite_number(rows) and int(rows) < 500:
                tiny_records.append(target)
            has_split = (
                finite_number(train_rows)
                and int(train_rows) > 0
                and finite_number(test_rows)
                and int(test_rows) > 0
            ) or (isinstance(folds, list) and len(folds) > 0)
            if not has_split:
                missing_split.append(target)
            if str(record.get("device", "")).lower() == "cuda":
                gpu_records += 1
            best_params = record.get("best_params")
            if isinstance(best_params, dict) and str(best_params.get("device", "")).lower() == "cuda":
                gpu_records += 1
            if any(key in record for key in ("baseline_rmse", "mean_skill_rmse_vs_persistence", "beats_persistence")):
                benchmarked_records += 1
            if record.get("beats_persistence") is True:
                beats_persistence += 1
            elif finite_number(record.get("skill_rmse_pct")) and float(record["skill_rmse_pct"]) > 0:
                beats_persistence += 1
            elif finite_number(record.get("mean_skill_rmse_vs_persistence")) and float(
                record["mean_skill_rmse_vs_persistence"]
            ) > 0:
                beats_persistence += 1
            if finite_number(record.get("mean_skill_rmse_vs_persistence")) and float(
                record["mean_skill_rmse_vs_persistence"]
            ) <= 0:
                horizon = record.get("horizon_h", "?")
                fails_persistence.append(f"{target}+{horizon}h")
            if isinstance(folds, list):
                for fold in folds:
                    if not isinstance(fold, dict):
                        continue
                    fold_skill = fold.get("skill_rmse_vs_persistence")
                    if finite_number(fold_skill) and float(fold_skill) < 0:
                        horizon = record.get("horizon_h", "?")
                        negative_fold_skill.append(f"{target}+{horizon}h")
                        break
            for metric_name, value in find_metric_values(record):
                if not math.isfinite(value):
                    non_finite_metrics.append({"target": target, "metric": metric_name, "value": value})
                if metric_name.endswith("r2") and value < -5.0:
                    weak_records.append(f"{target}:{metric_name}={value:.3f}")

        details.update(
            {
                "gpu_records": gpu_records,
                "benchmarked_records": benchmarked_records,
                "beats_persistence_records": beats_persistence,
                "fails_persistence_records": len(fails_persistence),
                "fails_persistence_targets": fails_persistence[:30],
                "negative_fold_skill_records": len(negative_fold_skill),
                "negative_fold_skill_targets": negative_fold_skill[:30],
                "missing_split_targets": missing_split[:30],
                "tiny_record_targets": tiny_records[:30],
                "weak_records": weak_records[:30],
                "non_finite_metrics": non_finite_metrics[:30],
            }
        )

        if non_finite_metrics:
            statuses.append("FAIL")
            messages.append("non-finite metric values found")
        if missing_split:
            statuses.append("FAIL")
            messages.append("one or more records lack a train/test split or folds")
        if benchmarked_records == 0:
            statuses.append("WARN")
            messages.append("no persistence/baseline benchmark fields found")
        if gpu_records == 0:
            statuses.append("WARN")
            messages.append("no CUDA device evidence in model results")
        if beats_persistence == 0 and benchmarked_records:
            statuses.append("WARN")
            messages.append("no records beat persistence")
        if fails_persistence:
            statuses.append("WARN")
            messages.append(f"{len(fails_persistence)} records do not beat persistence")
        if negative_fold_skill:
            statuses.append("WARN")
            messages.append(f"{len(negative_fold_skill)} records have at least one negative-skill fold")
        if weak_records:
            statuses.append("WARN")
            messages.append("very weak model records found")
        if tiny_records:
            statuses.append("WARN")
            messages.append("some model records have fewer than 500 rows")
        if not statuses:
            statuses.append("PASS")
            messages.append("model result sanity checks passed")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate model results")

    return Check(f"models::{path.parent.name}", combine_status(*statuses), "; ".join(messages), details)


def check_model_outputs(project_root: Path) -> Check:
    result_paths = sorted(project_root.glob("mbal*_results/model_results.json"))
    details: dict[str, Any] = {"files": [str(path) for path in result_paths]}
    if not result_paths:
        return Check(
            "model_outputs",
            "WARN",
            "no Monterey Bay AI Lab model_results.json files found",
            details,
        )
    production_paths = [path for path in result_paths if path.parent.name == "mbal_forecast_v2_results"]
    legacy_paths = [path for path in result_paths if path not in production_paths]
    production_checks = [check_model_result_file(path) for path in production_paths]
    legacy_checks = [check_model_result_file(path) for path in legacy_paths]
    details["production_checks"] = [check.__dict__ for check in production_checks]
    details["legacy_checks"] = [check.__dict__ for check in legacy_checks]
    details["checks"] = details["production_checks"] + details["legacy_checks"]
    if not production_checks:
        return Check(
            "model_outputs",
            "FAIL",
            "production forecast v2 model_results.json is missing",
            details,
        )
    status = combine_status(*(check.status for check in production_checks))
    pass_count = sum(1 for check in production_checks if check.status == "PASS")
    warn_count = sum(1 for check in production_checks if check.status == "WARN")
    fail_count = sum(1 for check in production_checks if check.status == "FAIL")
    return Check(
        "model_outputs",
        status,
        (
            f"{pass_count} PASS, {warn_count} WARN, {fail_count} FAIL across "
            f"{len(production_checks)} production result files; {len(legacy_checks)} legacy result files inspected"
        ),
        details,
    )


def check_neural_lakehouse_outputs(project_root: Path) -> Check:
    """Validate the completed neural sweep and local lakehouse contracts.

    This is deliberately advisory for now: XGBoost remains the production model
    surface, but neural evidence should be visible in the release gate so model
    promotion decisions are not based on scattered CSV/Markdown artifacts.
    """
    phase2_path = project_root / "nn_results" / "phase2_run_summary.csv"
    metrics_path = project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet"
    run_manifest_path = project_root / "lakehouse" / "gold" / "forecast_runs" / "run_manifest.parquet"
    details: dict[str, Any] = {
        "phase2_run_summary": str(phase2_path),
        "forecast_metrics": str(metrics_path),
        "forecast_run_manifest": str(run_manifest_path),
    }
    statuses: list[str] = []
    messages: list[str] = []

    try:
        import pandas as pd

        if not phase2_path.exists():
            statuses.append("WARN")
            messages.append("Phase-2 neural comparison summary is missing")
        else:
            phase2 = pd.read_csv(phase2_path)
            details["phase2_rows"] = int(len(phase2))
            required = {"outdir", "model_name", "loss", "drivers_enabled", "mean_skill", "median_skill", "mean_rmse"}
            missing = sorted(required - set(phase2.columns))
            if missing:
                statuses.append("FAIL")
                messages.append(f"Phase-2 summary missing columns: {', '.join(missing)}")
            elif phase2.empty:
                statuses.append("FAIL")
                messages.append("Phase-2 summary has no rows")
            else:
                phase2 = phase2.copy()
                phase2["mean_skill"] = pd.to_numeric(phase2["mean_skill"], errors="coerce")
                phase2["median_skill"] = pd.to_numeric(phase2["median_skill"], errors="coerce")
                phase2["mean_rmse"] = pd.to_numeric(phase2["mean_rmse"], errors="coerce")
                if phase2[["mean_skill", "median_skill", "mean_rmse"]].isna().any().any():
                    statuses.append("FAIL")
                    messages.append("Phase-2 summary contains non-numeric key metrics")
                else:
                    best = phase2.sort_values(["mean_skill", "median_skill"], ascending=False).iloc[0].to_dict()
                    details["phase2_best_run"] = {
                        "outdir": str(best["outdir"]),
                        "model_name": str(best["model_name"]),
                        "loss": str(best["loss"]),
                        "drivers_enabled": bool(best["drivers_enabled"]),
                        "mean_skill": float(best["mean_skill"]),
                        "median_skill": float(best["median_skill"]),
                        "mean_rmse": float(best["mean_rmse"]),
                    }
                    positive_runs = phase2[phase2["mean_skill"] > 0]
                    details["phase2_positive_mean_skill_runs"] = int(len(positive_runs))
                    driver_rows = phase2[phase2["drivers_enabled"].astype(str).str.lower().isin(("true", "1"))]
                    details["phase2_driver_runs"] = int(len(driver_rows))
                    if float(best["mean_skill"]) <= 0:
                        statuses.append("WARN")
                        messages.append("no completed neural run beats persistence on mean skill")
                    if not driver_rows.empty and float(driver_rows["mean_skill"].median()) < 0:
                        statuses.append("WARN")
                        messages.append("driver-enabled neural runs are negative on median mean-skill")

        if not metrics_path.exists():
            statuses.append("WARN")
            messages.append("lakehouse forecast_metrics table is missing")
        else:
            metrics = read_forecast_metrics(project_root, include_partitions=True)
            details["lakehouse_metric_rows"] = int(len(metrics))
            required_metrics = {"run_id", "split_id", "unique_id", "horizon_h", "model_rmse", "persistence_rmse", "skill_vs_persistence_pct"}
            missing_metrics = sorted(required_metrics - set(metrics.columns))
            if missing_metrics:
                statuses.append("FAIL")
                messages.append(f"lakehouse metrics missing columns: {', '.join(missing_metrics)}")
            elif metrics.empty:
                statuses.append("FAIL")
                messages.append("lakehouse metrics table has no rows")
            else:
                details["lakehouse_run_count"] = int(metrics["run_id"].nunique())
                details["lakehouse_split_count"] = int(metrics["split_id"].nunique())
                skill = pd.to_numeric(metrics["skill_vs_persistence_pct"], errors="coerce")
                if skill.isna().any():
                    statuses.append("FAIL")
                    messages.append("lakehouse metrics contain non-numeric skill values")
                else:
                    details["lakehouse_positive_skill_cells"] = int((skill > 0).sum())
                    details["lakehouse_total_skill_cells"] = int(len(skill))
                    details["lakehouse_mean_skill"] = float(round(skill.mean(), 4))
                    details["lakehouse_median_skill"] = float(round(skill.median(), 4))
                    if not (skill > 0).any():
                        statuses.append("WARN")
                        messages.append("lakehouse neural metrics have no positive-skill cells")

        if not run_manifest_path.exists():
            statuses.append("WARN")
            messages.append("lakehouse run manifest table is missing")

        if not statuses:
            statuses.append("PASS")
            best = details.get("phase2_best_run", {})
            if best:
                messages.append(
                    "neural/lakehouse contracts present; "
                    f"best completed run is {best.get('outdir')} mean_skill={best.get('mean_skill')}"
                )
            else:
                messages.append("neural/lakehouse contracts present")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate neural/lakehouse outputs")

    return Check("neural_lakehouse_outputs", combine_status(*statuses), "; ".join(messages), details)


def check_promotion_matrix(project_root: Path) -> Check:
    matrix_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    summary_path = project_root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json"
    details: dict[str, Any] = {
        "promotion_matrix": str(matrix_path),
        "promotion_summary": str(summary_path),
    }
    statuses: list[str] = []
    messages: list[str] = []
    if not matrix_path.exists():
        return Check(
            "promotion_matrix",
            "WARN",
            "promotion matrix is missing; run release_gate/mbal_promotion_matrix.py",
            details,
        )
    try:
        import pandas as pd

        matrix = pd.read_parquet(matrix_path)
        required = {
            "schema_version",
            "target",
            "horizon_h",
            "candidate_model",
            "candidate_loss",
            "candidate_drivers_enabled",
            "candidate_run_id",
            "candidate_split_id",
            "xgb_run_id",
            "xgb_split_id",
            "candidate_skill_vs_persistence_pct",
            "xgb_skill_vs_persistence_pct",
            "xgb_delta_skill_pct",
            "status",
            "reason",
        }
        missing = sorted(required - set(matrix.columns))
        if missing:
            statuses.append("FAIL")
            messages.append(f"promotion matrix missing columns: {', '.join(missing)}")
        details["rows"] = int(len(matrix))
        if matrix.empty:
            statuses.append("WARN")
            messages.append("promotion matrix has no candidate rows")
        else:
            status_counts = matrix["status"].value_counts().sort_index().to_dict()
            details["status_counts"] = {str(k): int(v) for k, v in status_counts.items()}
            promoted = matrix[matrix["status"] == "promote"].copy()
            promotable = int(len(promoted))
            candidates = int(matrix["status"].isin(["candidate", "candidate_split_mismatch"]).sum())
            details["promote_count"] = promotable
            details["promoted_row_count"] = promotable
            details["unique_promoted_target_horizon_count"] = int(
                promoted.drop_duplicates(["target", "horizon_h"]).shape[0]
            )
            details["unique_promoted_model_cell_count"] = int(
                promoted.drop_duplicates(
                    [
                        "target",
                        "horizon_h",
                        "candidate_model",
                        "candidate_loss",
                        "candidate_drivers_enabled",
                    ]
                ).shape[0]
            )
            details["candidate_count"] = candidates
            if promotable == 0 and candidates == 0:
                statuses.append("WARN")
                messages.append("promotion matrix has no promote/candidate rows")
            elif promotable == 0:
                statuses.append("WARN")
                messages.append("promotion matrix has candidates but no enforceable promotions")
        if summary_path.exists():
            details["summary"] = load_json(summary_path)
        else:
            statuses.append("WARN")
            messages.append("promotion summary JSON is missing")
        if not statuses:
            statuses.append("PASS")
            messages.append("promotion matrix is present and has enforceable promotions")
    except Exception as exc:
        statuses.append("FAIL")
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc(limit=4)
        messages.append("could not validate promotion matrix")
    return Check("promotion_matrix", combine_status(*statuses), "; ".join(messages), details)


STRUCTURAL_CHECKS = {
    "gpu_stack",
    "package_versions",
    "historical_parquet",
    "curated_dataset",
    "promotion_matrix",
}
ADVISORY_CHECKS = {
    "model_outputs",
    "neural_lakehouse_outputs",
}


def compute_overall_status(checks: list[Check]) -> str:
    """Aggregate release status while keeping advisory caveats visible.

    Model quality warnings, such as weak XGBoost target/horizon records or
    broadly weak driver-enabled neural runs, should stay in the report but not
    block release once structural checks pass and promotions are enforceable.
    Invalid data/artifacts still surface as FAIL from their source checks and
    always fail the gate.
    """
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"

    by_name = {check.name: check for check in checks}
    promotion_check = by_name.get("promotion_matrix")
    structural_ready = all(by_name.get(name) and by_name[name].status == "PASS" for name in STRUCTURAL_CHECKS)
    advisory_only_warnings = all(
        check.status == "PASS" or check.name in ADVISORY_CHECKS
        for check in checks
    )
    if (
        promotion_check
        and promotion_check.status == "PASS"
        and structural_ready
        and advisory_only_warnings
    ):
        return "PASS"

    return combine_status(*(check.status for check in checks))


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Monterey Bay AI Lab Release Gate Report",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Project root: `{report['project_root']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.extend(
            [
                f"### {check['status']} - {check['name']}",
                "",
                check["summary"],
                "",
            ]
        )
        key_details = check.get("details", {})
        if check["name"] == "gpu_stack":
            lines.append("- GPU stack details:")
            for key in ("torch_version", "torch_cuda_version", "torch_device_name", "xgboost_version"):
                if key in key_details:
                    lines.append(f"  - `{key}`: `{key_details[key]}`")
            if key_details.get("nvidia_smi", {}).get("stdout"):
                lines.append(f"  - `nvidia-smi`: `{key_details['nvidia_smi']['stdout']}`")
            lines.append("")
        elif check["name"] == "package_versions":
            lines.append("- Package versions:")
            for package, version in key_details.get("versions", {}).items():
                lines.append(f"  - `{package}`: `{version or 'missing'}`")
            lines.append("")
        elif check["name"] == "historical_parquet":
            lines.append("- Trusted OPeNDAP history:")
            for child in key_details.get("production_checks", key_details.get("checks", [])):
                lines.append(f"- **{child['status']}** `{child['name']}`: {child['summary']}")
            auxiliary = key_details.get("auxiliary_checks", [])
            if auxiliary:
                lines.append("- Auxiliary Parquet inspected but not release-blocking:")
                for child in auxiliary:
                    lines.append(f"  - **{child['status']}** `{child['name']}`: {child['summary']}")
            lines.append("")
        elif check["name"] == "curated_dataset":
            for key in ("expected_rows", "actual_rows", "parquet_file_count", "reported_errors", "reported_warnings"):
                if key in key_details:
                    lines.append(f"- `{key}`: `{key_details[key]}`")
            lines.append("")
        elif check["name"] == "model_outputs":
            lines.append("- Production model outputs:")
            for child in key_details.get("production_checks", []):
                lines.append(f"  - **{child['status']}** `{child['name']}`: {child['summary']}")
                child_details = child.get("details", {})
                for key in ("fails_persistence_records", "negative_fold_skill_records"):
                    if key in child_details:
                        lines.append(f"    - `{key}`: `{child_details[key]}`")
            legacy = key_details.get("legacy_checks", [])
            if legacy:
                lines.append("- Legacy/exploratory outputs inspected but not release-blocking:")
                for child in legacy:
                    lines.append(f"  - **{child['status']}** `{child['name']}`: {child['summary']}")
            lines.append("")
        elif check["name"] == "neural_lakehouse_outputs":
            best = key_details.get("phase2_best_run")
            if best:
                lines.append("- Best completed neural run:")
                for key in ("outdir", "model_name", "loss", "drivers_enabled", "mean_skill", "median_skill", "mean_rmse"):
                    if key in best:
                        lines.append(f"  - `{key}`: `{best[key]}`")
            for key in (
                "phase2_rows",
                "phase2_positive_mean_skill_runs",
                "phase2_driver_runs",
                "lakehouse_run_count",
                "lakehouse_split_count",
                "lakehouse_positive_skill_cells",
                "lakehouse_total_skill_cells",
                "lakehouse_mean_skill",
                "lakehouse_median_skill",
            ):
                if key in key_details:
                    lines.append(f"- `{key}`: `{key_details[key]}`")
            lines.append("")
        elif check["name"] == "promotion_matrix":
            if "status_counts" in key_details:
                lines.append("- Promotion status counts:")
                for status, count in key_details["status_counts"].items():
                    lines.append(f"  - `{status}`: `{count}`")
            for key in (
                "rows",
                "promoted_row_count",
                "unique_promoted_target_horizon_count",
                "unique_promoted_model_cell_count",
                "candidate_count",
            ):
                if key in key_details:
                    lines.append(f"- `{key}`: `{key_details[key]}`")
            lines.append("")
    lines.extend(
        [
            "## Policy",
            "",
            "- `FAIL` means the setup should not be treated as production-grade until fixed.",
            "- `WARN` means the setup is usable with caveats that should be tracked.",
            "- `PASS` means the checked surface met the current gate criteria.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(project_root: Path, output_dir: Path) -> dict[str, Any]:
    checks = [
        check_gpu_stack(),
        check_package_versions(),
        check_historical_parquet(project_root),
        check_curated_dataset(project_root),
        check_model_outputs(project_root),
        check_neural_lakehouse_outputs(project_root),
        check_promotion_matrix(project_root),
    ]
    overall_status = compute_overall_status(checks)
    report = {
        "schema_version": 1,
        "overall_status": overall_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "checks": [check.__dict__ for check in checks],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "release_gate_report.json"
    md_path = output_dir / "release_gate_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(make_markdown(report), encoding="utf-8")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Monterey Bay AI Lab release gate.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root to inspect. Defaults to this repository's parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for JSON and markdown reports.",
    )
    parser.add_argument(
        "--no-exit-code",
        action="store_true",
        help="Always exit 0 after writing reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    report = run_gate(project_root, output_dir)
    print(json.dumps({k: report[k] for k in ("overall_status", "report_paths")}, indent=2))
    if args.no_exit_code:
        return 0
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

