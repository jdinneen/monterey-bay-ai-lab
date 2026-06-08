#!/usr/bin/env python3
"""Build hourly observed-only news/event driver features.

Reads curated silver news events and writes a driver-compatible hourly table:

  nn_cache/news_drivers_hourly.parquet
  nn_cache/news_drivers_manifest.json

All emitted columns are historical exogenous drivers. Feature visibility is
controlled by ``available_at_utc``; event time is descriptive only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = DEFAULT_PROJECT_ROOT / "research" / "news_events_taxonomy.json"
DEFAULT_EVENTS = DEFAULT_PROJECT_ROOT / "lakehouse" / "silver" / "news_events" / "events.parquet"
DEFAULT_OUT_PARQUET = DEFAULT_PROJECT_ROOT / "nn_cache" / "news_drivers_hourly.parquet"
DEFAULT_OUT_MANIFEST = DEFAULT_PROJECT_ROOT / "nn_cache" / "news_drivers_manifest.json"
DEFAULT_GRID = DEFAULT_PROJECT_ROOT / "nn_cache" / "drivers_hourly.parquet"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_taxonomy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_types_from_taxonomy(taxonomy: dict[str, Any]) -> list[str]:
    return [str(item["event_type"]) for item in taxonomy.get("event_classes", [])]


def feature_windows_from_taxonomy(taxonomy: dict[str, Any]) -> list[int]:
    windows = [int(w) for w in taxonomy.get("feature_windows_hours", [24, 72, 168])]
    return sorted(set(windows))


def load_grid(grid_parquet: Path | None, events: pd.DataFrame, start_utc: str | None, end_utc: str | None) -> pd.DatetimeIndex:
    if grid_parquet and grid_parquet.exists():
        grid_df = pd.read_parquet(grid_parquet)
        if "ds" not in grid_df.columns:
            grid_df = grid_df.reset_index()
            if "ds" not in grid_df.columns:
                grid_df = grid_df.rename(columns={grid_df.columns[0]: "ds"})
        idx = pd.to_datetime(grid_df["ds"], utc=True).drop_duplicates().sort_values()
        return pd.DatetimeIndex(idx, name="ds")

    if start_utc and end_utc:
        return pd.date_range(pd.Timestamp(start_utc), pd.Timestamp(end_utc), freq="1h", inclusive="left", tz="UTC", name="ds")

    if not events.empty and "available_at_utc" in events.columns:
        available = pd.to_datetime(events["available_at_utc"], utc=True, errors="coerce").dropna()
        if not available.empty:
            start = available.min().floor("h") - pd.Timedelta(days=1)
            end = available.max().ceil("h") + pd.Timedelta(days=1)
            return pd.date_range(start, end, freq="1h", tz="UTC", name="ds")

    raise ValueError("no grid available; pass --grid-parquet, --start-utc/--end-utc, or non-empty events")


def load_events(path: Path, event_types: list[str], allow_empty: bool) -> pd.DataFrame:
    if not path.exists():
        if not allow_empty:
            raise FileNotFoundError(f"news events parquet not found: {path}")
        return pd.DataFrame(columns=["event_type", "available_at_utc", "severity_score", "confidence_score"])
    events = pd.read_parquet(path)
    required = {"event_type", "available_at_utc"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"news events missing columns: {missing}")
    events = events.copy()
    events["event_type"] = events["event_type"].astype(str)
    events = events[events["event_type"].isin(event_types)].copy()
    events["available_at_utc"] = pd.to_datetime(events["available_at_utc"], utc=True, errors="coerce")
    events = events[events["available_at_utc"].notna()].copy()
    for col, default in [("severity_score", 1.0), ("confidence_score", 1.0), ("relevance_score", 1.0)]:
        if col not in events.columns:
            events[col] = default
        events[col] = pd.to_numeric(events[col], errors="coerce").fillna(default).clip(lower=0.0)
    return events


def build_features(grid: pd.DatetimeIndex, events: pd.DataFrame, event_types: list[str], windows: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    out.index.name = "ds"
    if events.empty:
        for event_type in event_types:
            for window in windows:
                out[f"news_{event_type}_count_{window}h"] = 0.0
                out[f"news_{event_type}_severity_sum_{window}h"] = 0.0
        out["news_any_event_count_168h"] = 0.0
        out["news_any_event_weighted_score_168h"] = 0.0
        return out.reset_index()

    visible = events.copy()
    visible["available_hour"] = visible["available_at_utc"].dt.floor("h")
    visible = visible[visible["available_hour"].between(grid.min(), grid.max())].copy()
    weight = (
        visible["severity_score"].astype(float)
        * visible["confidence_score"].astype(float)
        * visible["relevance_score"].astype(float)
    )
    visible["weighted_score"] = weight

    any_count = pd.Series(0.0, index=grid)
    any_weight = pd.Series(0.0, index=grid)
    for event_type in event_types:
        typed = visible[visible["event_type"].eq(event_type)]
        count = pd.Series(0.0, index=grid)
        sev = pd.Series(0.0, index=grid)
        if not typed.empty:
            count_updates = typed.groupby("available_hour").size().astype(float)
            score_updates = typed.groupby("available_hour")["weighted_score"].sum().astype(float)
            count.loc[count_updates.index] = count.loc[count_updates.index].add(count_updates, fill_value=0.0)
            sev.loc[score_updates.index] = sev.loc[score_updates.index].add(score_updates, fill_value=0.0)
            any_count = any_count.add(count, fill_value=0.0)
            any_weight = any_weight.add(sev, fill_value=0.0)
        for window in windows:
            out[f"news_{event_type}_count_{window}h"] = count.rolling(window=window, min_periods=1).sum().to_numpy()
            out[f"news_{event_type}_severity_sum_{window}h"] = sev.rolling(window=window, min_periods=1).sum().to_numpy()

    out["news_any_event_count_168h"] = any_count.rolling(window=168, min_periods=1).sum().to_numpy()
    out["news_any_event_weighted_score_168h"] = any_weight.rolling(window=168, min_periods=1).sum().to_numpy()
    return out.reset_index()


def write_outputs(
    features: pd.DataFrame,
    events: pd.DataFrame,
    taxonomy: dict[str, Any],
    out_parquet: Path,
    out_manifest: Path,
    source_events_path: Path,
) -> dict[str, Any]:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_parquet, index=False)
    hist_cols = [c for c in features.columns if c != "ds"]
    coverage = {col: round(float((features[col] != 0).mean()), 4) for col in hist_cols}
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "driver_family": "news_events",
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "source_events_path": str(source_events_path),
        "futr": [],
        "hist": hist_cols,
        "coverage": coverage,
        "event_rows": int(len(events)),
        "feature_rows": int(len(features)),
        "ds_min": str(pd.to_datetime(features["ds"]).min()),
        "ds_max": str(pd.to_datetime(features["ds"]).max()),
        "leakage_rule": "news features are computed from available_at_utc <= ds and are hist-only",
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--grid-parquet", type=Path, default=None)
    parser.add_argument("--start-utc", default=None)
    parser.add_argument("--end-utc", default=None)
    parser.add_argument("--out-parquet", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true", help="write zero-valued news features if silver events are missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    taxonomy_path = args.taxonomy or root / "research" / "news_events_taxonomy.json"
    events_path = args.events or root / "lakehouse" / "silver" / "news_events" / "events.parquet"
    grid_path = args.grid_parquet if args.grid_parquet is not None else root / "nn_cache" / "drivers_hourly.parquet"
    out_parquet = args.out_parquet or root / "nn_cache" / "news_drivers_hourly.parquet"
    out_manifest = args.out_manifest or root / "nn_cache" / "news_drivers_manifest.json"

    taxonomy = load_taxonomy(taxonomy_path)
    event_types = event_types_from_taxonomy(taxonomy)
    windows = feature_windows_from_taxonomy(taxonomy)
    events = load_events(events_path, event_types, allow_empty=args.allow_empty)
    grid = load_grid(grid_path, events, args.start_utc, args.end_utc)
    features = build_features(grid, events, event_types, windows)
    manifest = write_outputs(features, events, taxonomy, out_parquet, out_manifest, events_path)
    print(
        json.dumps(
            {
                "event_rows": manifest["event_rows"],
                "feature_rows": manifest["feature_rows"],
                "hist_feature_count": len(manifest["hist"]),
                "parquet": str(out_parquet),
                "manifest": str(out_manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
