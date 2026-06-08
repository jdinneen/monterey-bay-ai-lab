#!/usr/bin/env python
"""Shared forecast split contracts for MBARI model comparison.

This module is intentionally small and dependency-light so XGBoost, neural, and
foundation runners can all use the same split identifiers and row schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_SCHEMA_VERSION = 1
DEFAULT_HORIZONS = [1, 6, 24, 72, 168]


@dataclass(frozen=True)
class SplitContract:
    split_id: str
    rows: pd.DataFrame
    manifest: dict
    alias: str | None = None


def _alias_filename(alias: str) -> str:
    normalized = alias.strip()
    if not normalized:
        raise ValueError("split alias must be non-empty")
    if any(ch in normalized for ch in ("/", "\\", ":", "\0")):
        raise ValueError(f"invalid split alias: {alias!r}")
    return f"alias={normalized}.json"


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


def normalize_timestamps(values: Iterable) -> list[pd.Timestamp]:
    timestamps = pd.to_datetime(list(values), utc=True)
    return sorted(pd.Timestamp(ts).tz_convert("UTC") for ts in timestamps.dropna().unique())


def build_split_contract(
    *,
    source_path: str | Path,
    cutoffs: Iterable,
    train_start,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    cache_version: str,
    config: dict | None = None,
    family: str = "shared",
    alias: str | None = None,
) -> SplitContract:
    cutoff_values = normalize_timestamps(cutoffs)
    horizon_values = [int(h) for h in horizons]
    train_start_ts = pd.Timestamp(train_start)
    if train_start_ts.tzinfo is None:
        train_start_ts = train_start_ts.tz_localize("UTC")
    else:
        train_start_ts = train_start_ts.tz_convert("UTC")
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "family": family,
        "cache_version": cache_version,
        "source": file_fingerprint(source_path),
        "config": config or {},
        "cutoffs": [c.isoformat() for c in cutoff_values],
        "train_start": train_start_ts.isoformat(),
        "horizons": horizon_values,
    }
    split_id = stable_hash(payload)
    created = utc_now()
    rows = []
    for split_index, cutoff in enumerate(cutoff_values):
        for horizon_h in horizon_values:
            target_ds = cutoff + pd.Timedelta(hours=int(horizon_h))
            rows.append(
                {
                    "split_id": split_id,
                    "split_index": int(split_index),
                    "cutoff": cutoff,
                    "horizon_h": int(horizon_h),
                    "train_start": train_start_ts,
                    "train_end": cutoff,
                    "target_start": target_ds,
                    "target_end": target_ds,
                    "cache_version": cache_version,
                    "schema_version": SPLIT_SCHEMA_VERSION,
                    "created_at_utc": created,
                }
            )
    manifest = {
        "split_id": split_id,
        "alias": alias,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "family": family,
        "cache_version": cache_version,
        "source": payload["source"],
        "config": payload["config"],
        "n_cutoffs": len(cutoff_values),
        "horizons": horizon_values,
        "train_start": train_start_ts.isoformat(),
        "cutoff_min": cutoff_values[0].isoformat() if cutoff_values else None,
        "cutoff_max": cutoff_values[-1].isoformat() if cutoff_values else None,
        "created_at_utc": created,
    }
    return SplitContract(split_id=split_id, rows=pd.DataFrame(rows), manifest=manifest, alias=alias)


def write_split_contract(contract: SplitContract, lakehouse_dir: str | Path) -> Path:
    split_dir = Path(lakehouse_dir) / "silver" / "forecast_splits" / f"split_id={contract.split_id}"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_manifest.json").write_text(
        json.dumps(contract.manifest, indent=2, default=str), encoding="utf-8"
    )
    if contract.alias:
        alias_path = split_dir.parent / _alias_filename(contract.alias)
        alias_payload = {"split_id": contract.split_id, "alias": contract.alias}
        alias_path.write_text(json.dumps(alias_payload, indent=2), encoding="utf-8")
    contract.rows.to_parquet(split_dir / "split_rows.parquet", index=False)
    return split_dir


def load_split_contract(lakehouse_dir: str | Path, split_id: str) -> SplitContract:
    split_dir = Path(lakehouse_dir) / "silver" / "forecast_splits" / f"split_id={split_id}"
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(split_dir / "split_rows.parquet")
    return SplitContract(split_id=split_id, rows=rows, manifest=manifest, alias=manifest.get("alias"))
