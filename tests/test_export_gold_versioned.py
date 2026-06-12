#!/usr/bin/env python3
"""Tests for the content-addressed versioned-gold snapshot layer.

All fixtures use ``tmp_path`` - the real ``lakehouse/`` is never touched. We set
``MBAL_LAKEHOUSE_DIR`` per-test so the exporter resolves the gold root inside
the tmp tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import export_gold_versioned as egv  # noqa: E402


@pytest.fixture
def gold(tmp_path, monkeypatch):
    """A tmp lakehouse with a tiny forecast_metrics source parquet."""
    lake = tmp_path / "lakehouse"
    metrics_dir = lake / "gold" / "forecast_metrics"
    metrics_dir.mkdir(parents=True)
    monkeypatch.setenv("MBAL_LAKEHOUSE_DIR", str(lake))
    df = pd.DataFrame(
        [
            {"run_id": "r1", "horizon_h": 6, "model_rmse": 1.0},
            {"run_id": "r2", "horizon_h": 6, "model_rmse": 2.0},
        ]
    )
    src = metrics_dir / "metrics.parquet"
    df.to_parquet(src, index=False)
    return tmp_path, src, df


def _log(tmp_path, table="forecast_metrics"):
    return egv._read_log(egv.versioned_dir(tmp_path, table))


def test_snapshot_written_and_logged(gold):
    tmp_path, _src, df = gold
    entry = egv.export_snapshot(tmp_path, "forecast_metrics", as_of="2026-06-08T00:00:00+00:00")

    assert entry is not None
    snap_dir = egv.versioned_dir(tmp_path, "forecast_metrics") / "snapshots" / entry["snapshot_id"]
    assert (snap_dir / "data.parquet").exists()

    log = _log(tmp_path)
    assert len(log) == 1
    assert log[0]["snapshot_id"] == entry["snapshot_id"]
    assert log[0]["parent_snapshot_id"] is None
    assert log[0]["row_count"] == len(df)
    assert log[0]["n_columns"] == df.shape[1]
    assert log[0]["created_at_utc"] == "2026-06-08T00:00:00+00:00"
    assert log[0]["schema_fingerprint"]


def test_rerun_identical_is_idempotent(gold):
    tmp_path, _src, _df = gold
    first = egv.export_snapshot(tmp_path, "forecast_metrics")
    second = egv.export_snapshot(tmp_path, "forecast_metrics")

    assert first["snapshot_id"] == second["snapshot_id"]
    log = _log(tmp_path)
    assert len(log) == 1  # no duplicate entry


def test_idempotent_under_row_reordering(gold):
    """Physical row order is incidental; content hash must be stable."""
    tmp_path, src, df = gold
    egv.export_snapshot(tmp_path, "forecast_metrics")
    # Rewrite source with rows reversed - same content, different byte layout.
    df.iloc[::-1].reset_index(drop=True).to_parquet(src, index=False)
    egv.export_snapshot(tmp_path, "forecast_metrics")
    assert len(_log(tmp_path)) == 1


def test_read_latest_returns_equal_data(gold):
    tmp_path, _src, df = gold
    egv.export_snapshot(tmp_path, "forecast_metrics")
    got = egv.read_snapshot(tmp_path, "forecast_metrics")  # None -> latest

    assert got is not None
    # Compare content order-independently.
    a = df.sort_values(list(df.columns)).reset_index(drop=True)
    b = got.sort_values(list(got.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_like=True)


def test_mutation_creates_child_snapshot(gold):
    tmp_path, src, df = gold
    first = egv.export_snapshot(tmp_path, "forecast_metrics")

    mutated = df.copy()
    mutated.loc[0, "model_rmse"] = 99.0
    mutated.to_parquet(src, index=False)
    second = egv.export_snapshot(tmp_path, "forecast_metrics")

    assert second["snapshot_id"] != first["snapshot_id"]
    assert second["parent_snapshot_id"] == first["snapshot_id"]
    log = _log(tmp_path)
    assert len(log) == 2

    # Time-travel: old id still returns the pre-mutation content.
    old = egv.read_snapshot(tmp_path, "forecast_metrics", first["snapshot_id"])
    assert float(old.sort_values("run_id").iloc[0]["model_rmse"]) == 1.0
    new = egv.read_snapshot(tmp_path, "forecast_metrics", second["snapshot_id"])
    assert 99.0 in set(new["model_rmse"].tolist())


def test_unknown_table_returns_none(gold):
    tmp_path, _src, _df = gold
    assert egv.export_snapshot(tmp_path, "no_such_table") is None
    assert egv.read_snapshot(tmp_path, "no_such_table") is None


def test_missing_source_skipped_exit_zero(tmp_path, monkeypatch, capsys):
    lake = tmp_path / "lakehouse"
    (lake / "gold").mkdir(parents=True)
    monkeypatch.setenv("MBAL_LAKEHOUSE_DIR", str(lake))
    results = egv.export_tables(tmp_path, ["forecast_metrics", "promotion_matrix"])
    assert results == {"forecast_metrics": None, "promotion_matrix": None}
    assert "skipping" in capsys.readouterr().err


def test_read_snapshot_unknown_id_returns_none(gold):
    tmp_path, _src, _df = gold
    egv.export_snapshot(tmp_path, "forecast_metrics")
    assert egv.read_snapshot(tmp_path, "forecast_metrics", "deadbeef") is None
