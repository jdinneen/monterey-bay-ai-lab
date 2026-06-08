#!/usr/bin/env python3
"""Merge driver-compatible hourly tables and manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_driver_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ds" not in df.columns:
        df = df.reset_index()
        if "ds" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ds"})
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    return df.drop_duplicates("ds").sort_values("ds")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def merge_drivers(inputs: list[tuple[Path, Path]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not inputs:
        raise ValueError("at least one --input parquet manifest pair is required")
    merged: pd.DataFrame | None = None
    futr: list[str] = []
    hist: list[str] = []
    sources: list[dict[str, str]] = []
    for parquet_path, manifest_path in inputs:
        df = read_driver_table(parquet_path)
        manifest = read_manifest(manifest_path)
        requested = list(manifest.get("futr", [])) + list(manifest.get("hist", []))
        missing = [col for col in requested if col not in df.columns]
        if missing:
            raise ValueError(f"{parquet_path} missing manifest columns: {missing}")
        keep = ["ds"] + requested
        df = df[keep]
        if merged is None:
            merged = df
        else:
            overlap = sorted((set(merged.columns) & set(df.columns)) - {"ds"})
            if overlap:
                raise ValueError(f"driver column overlap would be ambiguous: {overlap}")
            merged = merged.merge(df, on="ds", how="outer")
        futr.extend(str(col) for col in manifest.get("futr", []))
        hist.extend(str(col) for col in manifest.get("hist", []))
        sources.append({"parquet": str(parquet_path), "manifest": str(manifest_path)})
    assert merged is not None
    merged = merged.sort_values("ds").reset_index(drop=True)
    for col in [c for c in merged.columns if c != "ds"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    out_manifest = {
        "schema_version": 1,
        "driver_family": "merged",
        "sources": sources,
        "futr": ordered_unique(futr),
        "hist": ordered_unique(hist),
        "coverage": {
            col: round(float((merged[col] != 0).mean()), 4)
            for col in merged.columns
            if col != "ds"
        },
        "ds_min": str(merged["ds"].min()),
        "ds_max": str(merged["ds"].max()),
    }
    return merged, out_manifest


def parse_input(values: list[str]) -> list[tuple[Path, Path]]:
    if len(values) % 2:
        raise ValueError("--input expects pairs: parquet manifest")
    return [(Path(values[i]), Path(values[i + 1])) for i in range(0, len(values), 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="pairs of driver parquet and manifest paths",
    )
    parser.add_argument("--out-parquet", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged, manifest = merge_drivers(parse_input(args.input))
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out_parquet, index=False)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": int(len(merged)),
                "columns": int(len(merged.columns) - 1),
                "futr": len(manifest["futr"]),
                "hist": len(manifest["hist"]),
                "parquet": str(args.out_parquet),
                "manifest": str(args.out_manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
