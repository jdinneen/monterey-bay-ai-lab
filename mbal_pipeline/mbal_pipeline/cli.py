#!/usr/bin/env python
"""Validate and partition long-format MBAL historical parquet files."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: dict[str, str] = {
    "source": "string",
    "dataset_id": "string",
    "station": "string",
    "time": "datetime64[ns, UTC]",
    "latitude": "float64",
    "longitude": "float64",
    "z": "float64",
    "depth_m": "float64",
    "sea_water_temperature": "float64",
    "sea_water_temperature_qc": "float64",
    "sea_water_practical_salinity": "float64",
    "sea_water_practical_salinity_qc": "float64",
    "air_pressure": "float64",
    "air_pressure_qc": "float64",
    "air_temperature": "float64",
    "air_temperature_qc": "float64",
    "relative_humidity": "float64",
    "relative_humidity_qc": "float64",
    "eastward_sea_water_velocity": "float64",
    "northward_sea_water_velocity": "float64",
    "current_speed": "float64",
    "wind_speed_sonic": "float64",
    "wind_speed_sonic_qc": "float64",
    "wind_from_direction_sonic": "float64",
    "wind_from_direction_sonic_qc": "float64",
    "wind_speed_windbird": "float64",
    "wind_from_direction_windbird": "float64",
    "downwelling_shortwave_flux": "float64",
}

VALUE_RANGES: dict[str, tuple[float, float]] = {
    "sea_water_temperature": (-3.0, 35.0),
    "sea_water_practical_salinity": (20.0, 40.0),
    "air_pressure": (870.0, 1085.0),
    "air_temperature": (-40.0, 50.0),
    "relative_humidity": (0.0, 100.0),
    "wind_speed_sonic": (0.0, 60.0),
    "wind_from_direction_sonic": (0.0, 360.0),
    "wind_speed_windbird": (0.0, 60.0),
    "wind_from_direction_windbird": (0.0, 360.0),
    "downwelling_shortwave_flux": (-20.0, 1500.0),
    "eastward_sea_water_velocity": (-5.0, 5.0),
    "northward_sea_water_velocity": (-5.0, 5.0),
    "current_speed": (0.0, 5.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "depth_m": (0.0, 6000.0),
}

QC_PAIRS: dict[str, str] = {
    "sea_water_temperature": "sea_water_temperature_qc",
    "sea_water_practical_salinity": "sea_water_practical_salinity_qc",
    "air_pressure": "air_pressure_qc",
    "air_temperature": "air_temperature_qc",
    "relative_humidity": "relative_humidity_qc",
    "wind_speed_sonic": "wind_speed_sonic_qc",
    "wind_from_direction_sonic": "wind_from_direction_sonic_qc",
}

GOOD_QC = {1, 2}
KNOWN_QC = {0, 1, 2, 3, 4, 5, 8, 9}
DEFAULT_COVERAGE_COLUMNS = (
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
)


@dataclass
class Issue:
    severity: str
    check: str
    message: str
    count: int | None = None
    sample: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
        }
        if self.count is not None:
            out["count"] = int(self.count)
        if self.sample:
            out["sample"] = self.sample
        return out


@dataclass
class ValidationReport:
    input_path: str
    rows: int
    columns: list[str]
    station_summary: list[dict[str, Any]]
    variable_summary: list[dict[str, Any]]
    coverage_summary: list[dict[str, Any]]
    issues: list[Issue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "rows": self.rows,
            "columns": self.columns,
            "station_summary": self.station_summary,
            "variable_summary": self.variable_summary,
            "coverage_summary": self.coverage_summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


def _json_default(value: Any) -> str | float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def read_mbal_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if "station" in df.columns:
        df["station"] = df["station"].astype("string").str.upper().str.strip()
    return df


def validate_dataframe(df: pd.DataFrame, input_path: Path) -> ValidationReport:
    issues: list[Issue] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if missing:
        issues.append(Issue("error", "schema", f"Missing required columns: {', '.join(missing)}", len(missing)))
    if extra:
        issues.append(Issue("warning", "schema", f"Unexpected columns present: {', '.join(extra)}", len(extra)))

    for col, expected in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        dtype = str(df[col].dtype)
        if expected.startswith("datetime"):
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(Issue("error", "schema", f"{col} must be datetime-like, found {dtype}"))
            elif getattr(df[col].dt, "tz", None) is None:
                issues.append(Issue("error", "schema", f"{col} must be timezone-aware UTC, found naive datetime"))
        elif expected == "string":
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                issues.append(Issue("warning", "schema", f"{col} should be string-like, found {dtype}"))
        elif expected == "float64" and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(Issue("error", "schema", f"{col} must be numeric, found {dtype}"))

    if {"z", "depth_m"}.issubset(df.columns):
        bad_z = df["z"].notna() & df["depth_m"].notna() & ~np.isclose(df["z"], -df["depth_m"], atol=0.02)
        if bad_z.any():
            issues.append(Issue("error", "depth", "z must equal -depth_m within 0.02 m", int(bad_z.sum()), _sample_rows(df, bad_z)))

    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        bad = df[col].notna() & ~df[col].between(lo, hi, inclusive="both")
        if bad.any():
            issues.append(Issue("error", "range", f"{col} outside expected range [{lo}, {hi}]", int(bad.sum()), _sample_rows(df, bad)))

    for value_col, qc_col in QC_PAIRS.items():
        if value_col not in df.columns or qc_col not in df.columns:
            continue
        qc = pd.to_numeric(df[qc_col], errors="coerce")
        qc_non_null = qc.dropna()
        unknown_qc = qc_non_null[~qc_non_null.astype(int).isin(KNOWN_QC)]
        if not unknown_qc.empty:
            issues.append(Issue("warning", "qc", f"{qc_col} contains non-standard OceanSITES QC flags", int(unknown_qc.shape[0])))

        bad_qc_has_value = df[value_col].notna() & qc.notna() & ~qc.astype("Int64").isin(GOOD_QC)
        if bad_qc_has_value.any():
            issues.append(
                Issue(
                    "error",
                    "qc",
                    f"{value_col} has non-null values where {qc_col} is not good/probably-good",
                    int(bad_qc_has_value.sum()),
                    _sample_rows(df, bad_qc_has_value, [value_col, qc_col]),
                )
            )

    duplicate_keys = [col for col in ("station", "time", "depth_m") if col in df.columns]
    if len(duplicate_keys) == 3:
        dup = df.duplicated(duplicate_keys, keep=False)
        if dup.any():
            issues.append(Issue("error", "duplicates", "Duplicate station/time/depth_m rows found", int(dup.sum()), _sample_rows(df, dup)))

    if "time" in df.columns:
        null_time = df["time"].isna()
        if null_time.any():
            issues.append(Issue("error", "time", "Rows with null or unparsable time values", int(null_time.sum()), _sample_rows(df, null_time)))

    _validate_current_units(df, issues)

    return ValidationReport(
        input_path=str(input_path),
        rows=int(len(df)),
        columns=list(df.columns),
        station_summary=_station_summary(df),
        variable_summary=_variable_summary(df),
        coverage_summary=_coverage_summary(df),
        issues=issues,
    )


def _sample_rows(df: pd.DataFrame, mask: pd.Series, cols: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    base_cols = [col for col in ("station", "dataset_id", "time", "depth_m") if col in df.columns]
    if cols:
        base_cols.extend(col for col in cols if col in df.columns and col not in base_cols)
    if not base_cols:
        base_cols = list(df.columns[: min(6, len(df.columns))])
    rows = df.loc[mask, base_cols].head(limit).to_dict(orient="records")
    return [{key: _json_default(value) for key, value in row.items()} for row in rows]


def _validate_current_units(df: pd.DataFrame, issues: list[Issue]) -> None:
    required = {"eastward_sea_water_velocity", "northward_sea_water_velocity", "current_speed"}
    if not required.issubset(df.columns):
        return

    u = pd.to_numeric(df["eastward_sea_water_velocity"], errors="coerce")
    v = pd.to_numeric(df["northward_sea_water_velocity"], errors="coerce")
    speed = pd.to_numeric(df["current_speed"], errors="coerce")
    has_uv = u.notna() & v.notna()
    if not has_uv.any():
        return

    suspected_cms = has_uv & ((u.abs() > 5) | (v.abs() > 5)) & ((u.abs() <= 500) & (v.abs() <= 500))
    if suspected_cms.any():
        issues.append(
            Issue(
                "error",
                "current_units",
                "Current velocities look like cm/s; expected m/s after multiplying UCUR/VCUR by 0.01",
                int(suspected_cms.sum()),
                _sample_rows(df, suspected_cms, ["eastward_sea_water_velocity", "northward_sea_water_velocity"]),
            )
        )

    derived = np.sqrt(np.square(u) + np.square(v))
    comparable = has_uv & speed.notna()
    mismatch = comparable & ~np.isclose(speed, derived, atol=0.02, rtol=0.02)
    if mismatch.any():
        issues.append(
            Issue(
                "error",
                "current_speed",
                "current_speed must match sqrt(eastward^2 + northward^2) within tolerance",
                int(mismatch.sum()),
                _sample_rows(df.assign(_derived_current_speed=derived), mismatch, ["current_speed", "_derived_current_speed"]),
            )
        )

    missing_speed = has_uv & speed.isna()
    if missing_speed.any():
        issues.append(Issue("warning", "current_speed", "Rows with U/V currents but null current_speed", int(missing_speed.sum()), _sample_rows(df, missing_speed)))


def _station_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    if not {"station", "time", "depth_m"}.issubset(df.columns):
        return []
    summary = []
    for station, group in df.groupby("station", dropna=False, observed=True):
        time = group["time"].dropna()
        summary.append(
            {
                "station": str(station),
                "rows": int(len(group)),
                "first_time": _json_default(time.min()) if not time.empty else None,
                "last_time": _json_default(time.max()) if not time.empty else None,
                "depth_count": int(group["depth_m"].nunique(dropna=True)),
                "min_depth_m": _finite_float(group["depth_m"].min()),
                "max_depth_m": _finite_float(group["depth_m"].max()),
            }
        )
    return summary


def _variable_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for col in VALUE_RANGES:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out.append(
            {
                "column": col,
                "non_null": int(values.notna().sum()),
                "null": int(values.isna().sum()),
                "coverage_pct": round(float(values.notna().mean() * 100), 3) if len(values) else 0.0,
                "min": _finite_float(values.min()),
                "max": _finite_float(values.max()),
                "mean": _finite_float(values.mean()),
            }
        )
    return out


def _coverage_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"station", "time", "depth_m"}
    if not required.issubset(df.columns):
        return []
    out = []
    coverage_cols = [col for col in DEFAULT_COVERAGE_COLUMNS if col in df.columns]
    keyed = df.dropna(subset=["time", "depth_m"]).sort_values(["station", "depth_m", "time"])
    for station, station_group in keyed.groupby("station", observed=True):
        for depth, group in station_group.groupby("depth_m", observed=True):
            times = group["time"].drop_duplicates().sort_values()
            cadence_minutes = _infer_cadence_minutes(times)
            record = {
                "station": str(station),
                "depth_m": _finite_float(depth),
                "rows": int(len(group)),
                "unique_times": int(times.shape[0]),
                "first_time": _json_default(times.min()) if not times.empty else None,
                "last_time": _json_default(times.max()) if not times.empty else None,
                "median_cadence_minutes": cadence_minutes,
                "large_gap_count": _large_gap_count(times, cadence_minutes),
            }
            for col in coverage_cols:
                record[f"{col}_non_null"] = int(group[col].notna().sum())
            out.append(record)
    return out


def _infer_cadence_minutes(times: pd.Series) -> float | None:
    if times.shape[0] < 3:
        return None
    diffs = times.diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    return round(float(diffs.median()), 3)


def _large_gap_count(times: pd.Series, cadence_minutes: float | None) -> int | None:
    if cadence_minutes is None or times.shape[0] < 3:
        return None
    threshold = cadence_minutes * 1.5
    diffs = times.diff().dropna().dt.total_seconds() / 60.0
    return int((diffs > threshold).sum())


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def write_partitioned(df: pd.DataFrame, output_dir: Path, overwrite: bool = False) -> None:
    if "time" not in df.columns or "station" not in df.columns:
        raise ValueError("Cannot partition without station and time columns")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        if output_dir.anchor == str(output_dir):
            raise ValueError(f"Refusing to overwrite filesystem root: {output_dir}")
        shutil.rmtree(output_dir)

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["station"] = out["station"].astype("string").str.upper().str.strip()
    out["year"] = out["time"].dt.year.astype("int16")
    out["month"] = out["time"].dt.month.astype("int8")
    out = out.sort_values(["station", "time", "depth_m"], kind="mergesort")
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(
        output_dir,
        engine="pyarrow",
        partition_cols=["station", "year", "month"],
        index=False,
        compression="zstd",
    )


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MBAL long-format historical parquet files and optionally write station/year/month partitioned parquet."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input long-format parquet file(s).")
    parser.add_argument("--output-dir", type=Path, help="Write validated rows as partitioned parquet under this directory.")
    parser.add_argument("--report", type=Path, help="Write a JSON validation report.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --output-dir or --report if present.")
    parser.add_argument("--allow-errors", action="store_true", help="Write partitioned output even when validation errors are found.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Validate a deterministic row sample; useful for quick smoke checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frames: list[pd.DataFrame] = []
    reports: list[ValidationReport] = []

    for path in args.inputs:
        if not path.exists():
            print(f"ERROR: input does not exist: {path}", file=sys.stderr)
            return 2
        df = read_mbal_parquet(path)
        if args.sample_frac is not None:
            if not (0 < args.sample_frac <= 1):
                print("ERROR: --sample-frac must be in (0, 1]", file=sys.stderr)
                return 2
            df = df.sample(frac=args.sample_frac, random_state=17).sort_index()
        reports.append(validate_dataframe(df, path))
        frames.append(df)

    report_dict = {
        "generated_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "inputs": [report.to_dict() for report in reports],
        "total_rows": int(sum(len(frame) for frame in frames)),
        "total_errors": int(sum(report.error_count for report in reports)),
        "total_warnings": int(sum(report.warning_count for report in reports)),
    }

    if args.report:
        if args.report.exists() and not args.overwrite:
            print(f"ERROR: report exists, use --overwrite: {args.report}", file=sys.stderr)
            return 2
        write_report(report_dict, args.report)

    print(json.dumps({
        "total_rows": report_dict["total_rows"],
        "total_errors": report_dict["total_errors"],
        "total_warnings": report_dict["total_warnings"],
        "report": str(args.report) if args.report else None,
    }, indent=2))

    if args.output_dir:
        if report_dict["total_errors"] and not args.allow_errors:
            print("ERROR: validation errors found; refusing to write partitioned output without --allow-errors", file=sys.stderr)
            return 1
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        write_partitioned(combined, args.output_dir, overwrite=args.overwrite)
        print(f"Wrote partitioned parquet: {args.output_dir}")

    return 1 if report_dict["total_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
