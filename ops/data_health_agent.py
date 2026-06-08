#!/usr/bin/env python3
"""Data health agent for MBARI local artifacts.

The agent monitors coverage, gaps, driver freshness contracts, and duplicate metric
identities. It writes JSON + Markdown and returns nonzero only on hard structural
failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mbari_lakehouse import read_forecast_metrics


STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
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


@dataclass
class Check:
    name: str
    status: str
    summary: str
    details: dict[str, Any]


def combine_status(checks: list[Check]) -> str:
    if not checks:
        return "PASS"
    return max((check.status for check in checks), key=lambda s: STATUS_ORDER[s])


def max_missing_run_hours(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    best = cur = 0
    for value in missing:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def current_missing_run_hours(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    cur = 0
    for value in missing[::-1]:
        if not value:
            break
        cur += 1
    return int(cur)


def check_driver_manifest(project_root: Path) -> Check:
    manifest_path = project_root / "nn_cache" / "drivers_manifest.json"
    parquet_path = project_root / "nn_cache" / "drivers_hourly.parquet"
    details: dict[str, Any] = {"manifest": str(manifest_path), "parquet": str(parquet_path)}
    if not manifest_path.exists() or not parquet_path.exists():
        missing = [str(path) for path in [manifest_path, parquet_path] if not path.exists()]
        details["missing"] = missing
        return Check("driver_manifest", "FAIL", "driver artifacts are missing", details)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    drivers = pd.read_parquet(parquet_path)
    hist_cols = list(manifest.get("hist", []))
    futr_cols = list(manifest.get("futr", []))
    requested_cols = futr_cols + hist_cols
    raw_coverage = manifest.get("hist_raw_coverage", {})
    max_staleness = manifest.get("hist_max_staleness_hours", {})
    cap = manifest.get("max_hist_ffill_hours")
    details.update(
        {
            "hist_driver_count": len(hist_cols),
            "max_hist_ffill_hours": cap,
            "observed_driver_availability_lag_hours": manifest.get("observed_driver_availability_lag_hours"),
            "daily_observed_driver_availability_lag_days": manifest.get("daily_observed_driver_availability_lag_days"),
        }
    )
    failures: list[str] = []
    warnings: list[str] = []
    missing_driver_cols = sorted(c for c in requested_cols if c not in drivers.columns)
    details["missing_driver_columns"] = missing_driver_cols
    if missing_driver_cols:
        failures.append(f"{len(missing_driver_cols)} manifest driver columns are missing from parquet")
    if isinstance(drivers.index, pd.DatetimeIndex):
        index = drivers.index
    else:
        index = pd.to_datetime(drivers.index, errors="coerce", utc=True)
    details["driver_rows"] = int(len(drivers))
    details["driver_duplicate_timestamps"] = int(len(index) - pd.Index(index).nunique())
    if pd.isna(index).any():
        failures.append("driver parquet index contains non-datetime values")
    elif details["driver_duplicate_timestamps"]:
        failures.append("driver parquet contains duplicate timestamps")
    elif len(index) > 1:
        diffs = pd.Series(index.sort_values()).diff().dropna()
        details["driver_median_step"] = str(diffs.median())
        details["driver_max_step"] = str(diffs.max())
        if diffs.median() != pd.Timedelta(hours=1):
            warnings.append("driver parquet median timestamp step is not 1h")
    missing_raw = sorted(c for c in hist_cols if c not in raw_coverage)
    missing_stale = sorted(c for c in hist_cols if c not in max_staleness)
    if missing_raw:
        failures.append(f"{len(missing_raw)} hist drivers missing raw coverage")
    if missing_stale:
        failures.append(f"{len(missing_stale)} hist drivers missing staleness metadata")
    if cap is not None:
        over_cap = {
            col: value
            for col, value in max_staleness.items()
            if value is not None and float(value) > float(cap) + 1e-9
        }
        details["staleness_over_cap"] = over_cap
        if over_cap:
            failures.append(f"{len(over_cap)} hist drivers exceed max staleness cap")
    filled_coverage = manifest.get("coverage", {})
    coverage_mismatches = {}
    for col in requested_cols:
        if col in drivers.columns and col in filled_coverage:
            actual = float(drivers[col].notna().mean())
            expected = float(filled_coverage[col])
            if abs(actual - expected) > 0.01:
                coverage_mismatches[col] = {"actual": round(actual, 4), "manifest": round(expected, 4)}
    details["coverage_mismatches"] = coverage_mismatches
    if coverage_mismatches:
        warnings.append(f"{len(coverage_mismatches)} driver coverage values differ from manifest by >1 pp")
    inflated = {
        col: round(float(filled_coverage.get(col, 0.0)) - float(raw_coverage.get(col, 0.0)), 4)
        for col in hist_cols
        if col in raw_coverage and col in filled_coverage and float(filled_coverage[col]) - float(raw_coverage[col]) > 0.25
    }
    details["large_fill_coverage_lifts"] = inflated
    if inflated:
        warnings.append(f"{len(inflated)} hist drivers rely heavily on forward-filled coverage")
    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    return Check("driver_manifest", status, "; ".join(failures or warnings or ["driver freshness contract passes"]), details)


def _load_target_matrix(project_root: Path) -> pd.DataFrame | None:
    candidates = [
        project_root / "mbari_big_analysis_results" / "m1_hourly_matrix.parquet",
        project_root / "nn_cache" / "long_v2_past_only_fill_origin_observed_full.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "unique_id" in df.columns and "ds" in df.columns and "y" in df.columns:
                wide = df.pivot(index="ds", columns="unique_id", values="y")
                wide.index = pd.to_datetime(wide.index, utc=True)
                return wide
            if not isinstance(df.index, pd.DatetimeIndex):
                first = df.columns[0]
                try:
                    df[first] = pd.to_datetime(df[first], utc=True)
                    df = df.set_index(first)
                except Exception:
                    pass
            return df
    return None


def check_target_coverage(
    project_root: Path,
    warn_coverage: float,
    warn_gap_hours: int,
    warn_current_gap_hours: int = 24 * 14,
) -> Check:
    matrix = _load_target_matrix(project_root)
    details: dict[str, Any] = {
        "warn_coverage": warn_coverage,
        "warn_gap_hours": warn_gap_hours,
        "warn_current_gap_hours": warn_current_gap_hours,
    }
    if matrix is None:
        return Check("target_coverage", "WARN", "no target matrix artifact found for coverage audit", details)
    if isinstance(matrix.index, pd.DatetimeIndex) and len(matrix.index):
        full = pd.date_range(matrix.index.min(), matrix.index.max(), freq="1h")
        matrix = matrix.reindex(full)
    rows: list[dict[str, Any]] = []
    for target in FOCUS_TARGETS:
        if target not in matrix.columns:
            rows.append({"target": target, "present": False})
            continue
        series = matrix[target]
        rows.append(
            {
                "target": target,
                "present": True,
                "coverage": round(float(series.notna().mean()), 4),
                "non_null_hours": int(series.notna().sum()),
                "max_missing_run_h": max_missing_run_hours(series),
                "current_missing_run_h": current_missing_run_hours(series),
                "first_valid": str(series.first_valid_index()),
                "last_valid": str(series.last_valid_index()),
            }
        )
    details["targets"] = rows
    missing = [row["target"] for row in rows if not row.get("present")]
    low = [row for row in rows if row.get("present") and row["coverage"] < warn_coverage]
    long_gaps = [row for row in rows if row.get("present") and row["max_missing_run_h"] > warn_gap_hours]
    current_gaps = [row for row in rows if row.get("present") and row["current_missing_run_h"] > warn_current_gap_hours]
    if missing:
        return Check("target_coverage", "FAIL", f"{len(missing)} focus targets are missing from the target matrix", details)
    warnings = []
    if low:
        warnings.append(f"{len(low)} focus targets have coverage below {warn_coverage:.0%}")
    if current_gaps:
        warnings.append(f"{len(current_gaps)} focus targets are currently stale longer than {warn_current_gap_hours}h")
    details["historical_long_gap_targets"] = [row["target"] for row in long_gaps]
    return Check("target_coverage", "WARN" if warnings else "PASS", "; ".join(warnings or ["focus target coverage is acceptable"]), details)


def check_metric_duplicates(project_root: Path) -> Check:
    path = project_root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet"
    details: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        return Check("metric_duplicates", "WARN", "aggregate metrics parquet is missing", details)
    metrics = pd.read_parquet(path)
    deduped_recursive = read_forecast_metrics(project_root, include_partitions=True)
    keys = [c for c in ["run_id", "split_id", "unique_id", "horizon_h", "model", "loss"] if c in metrics.columns]
    details["rows"] = int(len(metrics))
    details["recursive_deduped_rows"] = int(len(deduped_recursive))
    details["identity_keys"] = keys
    if len(keys) < 4:
        return Check("metric_duplicates", "FAIL", "metrics table lacks identity columns", details)
    dup_count = int(metrics.duplicated(keys).sum())
    details["duplicate_metric_identities"] = dup_count
    if dup_count:
        details["examples"] = metrics[metrics.duplicated(keys, keep=False)][keys].head(20).to_dict(orient="records")
        return Check("metric_duplicates", "FAIL", f"{dup_count} duplicate metric identities found", details)
    return Check("metric_duplicates", "PASS", "no duplicate metric identities found", details)


def check_data_inventory(project_root: Path) -> Check:
    expected = [
        "mbari_history/opendap/m1_history.parquet",
        "mbari_history/opendap/m2_history.parquet",
        "mbari_history/noaa/noaa_ndbc46042.parquet",
        "mbari_history/noaa/noaa_coops.parquet",
        "mbari_history/noaa/noaa_upwelling.parquet",
        "mbari_history/noaa/noaa_drivers_daily.parquet",
        "nn_cache/drivers_hourly.parquet",
        "nn_cache/long_v3_past_only_fill_origin_observed_missingness_full.parquet",
    ]
    rows = []
    for rel in expected:
        path = project_root / rel
        rows.append({"path": rel, "exists": path.exists(), "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None})
    mur_cache = project_root / "mbari_history" / "noaa" / "mur_sst_cache"
    if mur_cache.exists():
        mur_files = sorted({*mur_cache.glob("mur_sst_*.parquet"), *mur_cache.glob("mur_*.parquet")})
    else:
        mur_files = []
    details = {"assets": rows, "mur_sst_cache_files": len(mur_files)}
    missing = [row["path"] for row in rows if not row["exists"] and "long_v3" not in row["path"]]
    warnings = []
    if not mur_files:
        warnings.append("MUR SST cache has no yearly parquet files")
    if missing:
        return Check("data_inventory", "FAIL", f"{len(missing)} required source/cache assets are missing", details)
    return Check("data_inventory", "WARN" if warnings else "PASS", "; ".join(warnings or ["expected data assets are present"]), details)


def build_report(project_root: Path, warn_coverage: float, warn_gap_hours: int) -> dict[str, Any]:
    checks = [
        check_data_inventory(project_root),
        check_driver_manifest(project_root),
        check_target_coverage(project_root, warn_coverage=warn_coverage, warn_gap_hours=warn_gap_hours),
        check_metric_duplicates(project_root),
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "overall_status": combine_status(checks),
        "checks": [asdict(check) for check in checks],
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_health_summary.json"
    md_path = output_dir / "DATA_HEALTH_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        "# MBARI Data Health Agent",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Generated: `{report['generated_at_utc']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.extend([f"### {check['status']} - {check['name']}", "", check["summary"], ""])
        if check["name"] == "target_coverage":
            rows = check["details"].get("targets", [])
            lines.extend(["| target | coverage | max missing h |", "|---|---:|---:|"])
            for row in rows:
                if not row.get("present"):
                    lines.append(f"| `{row['target']}` | missing | missing |")
                else:
                    lines.append(f"| `{row['target']}` | {row['coverage']:.4f} | {row['max_missing_run_h']} |")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--warn-coverage", type=float, default=0.50)
    parser.add_argument("--warn-gap-hours", type=int, default=24 * 30)
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = build_report(root, warn_coverage=args.warn_coverage, warn_gap_hours=args.warn_gap_hours)
    paths = write_outputs(report, args.output_dir or root / "reports" / "data_health")
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, indent=2, sort_keys=True))
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
