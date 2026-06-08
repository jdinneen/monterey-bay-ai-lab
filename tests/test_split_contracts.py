#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbari_split_contracts import build_split_contract, load_split_contract, write_split_contract  # noqa: E402


def test_split_contract_is_deterministic_and_writable(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fixture")
    cutoffs = pd.to_datetime(["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"])
    a = build_split_contract(
        source_path=source,
        cutoffs=cutoffs,
        train_start="2025-01-01T00:00:00Z",
        horizons=[1, 6],
        cache_version="v1",
        config={"n_windows": 2},
        family="unit",
    )
    b = build_split_contract(
        source_path=source,
        cutoffs=reversed(cutoffs),
        train_start="2025-01-01T00:00:00Z",
        horizons=[1, 6],
        cache_version="v1",
        config={"n_windows": 2},
        family="unit",
    )
    assert a.split_id == b.split_id
    assert len(a.rows) == 4
    assert a.manifest["n_cutoffs"] == 2

    split_dir = write_split_contract(a, tmp_path / "lakehouse")
    assert (split_dir / "split_manifest.json").exists()
    loaded = load_split_contract(tmp_path / "lakehouse", a.split_id)
    assert loaded.split_id == a.split_id
    assert len(loaded.rows) == 4


def test_split_alias_does_not_change_split_id_and_writes_lookup(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"fixture")
    common = {
        "source_path": source,
        "cutoffs": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "train_start": "2025-01-01T00:00:00Z",
        "horizons": [6],
        "cache_version": "v1",
        "config": {"n_windows": 1},
        "family": "unit",
    }
    canonical = build_split_contract(**common)
    aliased = build_split_contract(**common, alias="latest_shared")

    assert aliased.split_id == canonical.split_id
    assert aliased.manifest["alias"] == "latest_shared"

    write_split_contract(aliased, tmp_path / "lakehouse")
    alias_path = (
        tmp_path
        / "lakehouse"
        / "silver"
        / "forecast_splits"
        / "alias=latest_shared.json"
    )
    assert alias_path.exists()

    loaded = load_split_contract(tmp_path / "lakehouse", aliased.split_id)
    assert loaded.alias == "latest_shared"
