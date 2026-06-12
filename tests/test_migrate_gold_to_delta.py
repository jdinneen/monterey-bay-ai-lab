"""Offline tests for the Delta cutover tool (ops/migrate_gold_to_delta.py).

These run without the optional ``deltalake`` dependency: they assert the module is
import-safe, that the migrate/validate paths fail with a clear install hint rather
than a traceback, and that path/derivation helpers are correct. The real
migrate/parity behavior is covered once ``deltalake`` is installed (Phase 0/1).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ops import migrate_gold_to_delta as m

_HAS_DELTALAKE = importlib.util.find_spec("deltalake") is not None


def test_module_imports_and_reuses_snapshot_table_map():
    # Same logical tables as the snapshot layer — single source of truth.
    assert set(m._TABLE_SOURCES) >= {"forecast_metrics", "promotion_matrix"}


def test_delta_warehouse_is_sibling_of_gold(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    root = tmp_path
    # gold_delta sits next to gold, never inside it (non-destructive, parallel).
    assert m.delta_root(root) == root / "lakehouse" / "gold_delta"
    assert m.delta_table_path(root, "forecast_metrics").name == "forecast_metrics"
    assert "gold_delta" in str(m.delta_table_path(root, "forecast_metrics"))


@pytest.mark.skipif(_HAS_DELTALAKE, reason="deltalake installed; absent-path test N/A")
def test_require_deltalake_raises_clear_hint_when_absent():
    with pytest.raises(SystemExit) as exc:
        m._require_deltalake()
    msg = str(exc.value)
    assert "deltalake is not installed" in msg
    assert "pip install" in msg and "deltalake" in msg


@pytest.mark.skipif(_HAS_DELTALAKE, reason="deltalake installed; absent-path test N/A")
def test_validate_only_cli_exits_with_hint_without_dep():
    # SystemExit IS the exit-1 mechanism (verified: `rc=1` at the CLI). The dep
    # hint surfaces rather than a raw ImportError traceback.
    with pytest.raises(SystemExit) as exc:
        m.main(["--validate-only", "--tables", "forecast_metrics", "--project-root", "."])
    assert "deltalake is not installed" in str(exc.value)


def test_unknown_table_is_skipped_not_fatal(tmp_path, monkeypatch):
    # An unknown table returns None (skip) before any deltalake import is attempted.
    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    (tmp_path / "lakehouse" / "gold").mkdir(parents=True)
    assert m.migrate_table(tmp_path, "does_not_exist") is None


# --- present-path tests: exercise the real delta-rs engine (Phase 0/1) ----------

import pandas as pd  # noqa: E402


def _seed_metrics(tmp_path: Path, df: pd.DataFrame) -> Path:
    (tmp_path / "lakehouse" / "gold" / "forecast_metrics").mkdir(parents=True)
    (df).to_parquet(
        tmp_path / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet", index=False
    )
    return tmp_path


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_migrate_single_commit_and_parity(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    df = pd.DataFrame({"unique_id": ["a", "b"], "horizon_h": [1, 6], "n": pd.array([5, 7], dtype="Int64")})
    root = _seed_metrics(tmp_path, df)
    summary = m.migrate_table(root, "forecast_metrics", replay_history=False)
    assert summary["parity_ok"] is True
    assert summary["delta_rows"] == 2
    # Delta table written to the sibling gold_delta/, gold/ untouched.
    assert (root / "lakehouse" / "gold_delta" / "forecast_metrics" / "_delta_log").exists()
    assert (root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet").exists()
    got = m.read_delta_table(root, "forecast_metrics")
    assert sorted(got["unique_id"]) == ["a", "b"]


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_replay_history_yields_versions_and_time_travel(tmp_path, monkeypatch):
    from ops import export_gold_versioned as ev

    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    root = _seed_metrics(tmp_path, pd.DataFrame({"unique_id": ["a"], "v": [1]}))
    # Two snapshots in the versioned log: v=1 then v=2.
    ev.export_snapshot(root, "forecast_metrics", df=pd.DataFrame({"unique_id": ["a"], "v": [1]}))
    ev.export_snapshot(root, "forecast_metrics", df=pd.DataFrame({"unique_id": ["a"], "v": [2]}))
    # Source-of-truth parquet reflects the latest content for the parity check.
    pd.DataFrame({"unique_id": ["a"], "v": [2]}).to_parquet(
        root / "lakehouse" / "gold" / "forecast_metrics" / "metrics.parquet", index=False
    )
    summary = m.migrate_table(root, "forecast_metrics", replay_history=True)
    assert summary["parity_ok"] is True
    assert summary["commits"] == 2
    # Time-travel: version 0 is the first snapshot (v=1), latest is v=2.
    assert m.read_delta_table(root, "forecast_metrics", version=0)["v"].tolist() == [1]
    assert m.read_delta_table(root, "forecast_metrics")["v"].tolist() == [2]


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_validate_only_missing_delta_is_friendly_not_traceback(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    root = _seed_metrics(tmp_path, pd.DataFrame({"unique_id": ["a"], "v": [1]}))
    # Never migrated -> validate_parity raises a friendly AssertionError (caught by main).
    with pytest.raises(AssertionError) as exc:
        m.validate_parity(root, "forecast_metrics")
    assert "run the migration first" in str(exc.value)
    rc = m.main(["--validate-only", "--tables", "forecast_metrics", "--project-root", str(root)])
    assert rc == 1


def test_shadow_dual_write_is_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_GOLD_SHADOW_DELTA", raising=False)
    gold = tmp_path / "lakehouse" / "gold"
    gold.mkdir(parents=True)
    out = m.shadow_dual_write("forecast_metrics", pd.DataFrame({"a": [1]}), gold_dir=gold)
    assert out is None
    assert not (tmp_path / "lakehouse" / "gold_delta").exists()


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_shadow_dual_write_mirrors_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("MBAL_GOLD_SHADOW_DELTA", "1")
    gold = tmp_path / "lakehouse" / "gold"
    gold.mkdir(parents=True)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = m.shadow_dual_write("forecast_metrics", df, gold_dir=gold)
    assert out is not None
    assert (tmp_path / "lakehouse" / "gold_delta" / "forecast_metrics" / "_delta_log").exists()
    # gold/ itself is never written by the shadow path.
    assert not (gold / "forecast_metrics").exists()


# --- reader-parity contract: pins the Delta read to the live aggregate reader, and
# --- characterizes the Phase-3 reader-switch BLOCKER (include_partitions union) -----

def _seed_gold_metrics(root, agg_df, partition=None):
    """Write an aggregate metrics.parquet (+ optional un-folded run_id=* partition)."""
    base = root / "lakehouse" / "gold" / "forecast_metrics"
    base.mkdir(parents=True, exist_ok=True)
    agg_df.to_parquet(base / "metrics.parquet", index=False)
    if partition is not None:
        run_id, pdf = partition
        pdir = base / f"run_id={run_id}"
        pdir.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(pdir / "metrics.parquet", index=False)
    return root


def _metrics_row(run_id, horizon, rmse):
    return {
        "run_id": run_id, "split_id": "s1", "unique_id": "air_pressure",
        "horizon_h": horizon, "model": "xgboost", "loss": "mae",
        "drivers_enabled": True, "n": pd.array([16], dtype="Int64")[0],
        "model_rmse": rmse,
    }


def _metrics_df(rows):
    df = pd.DataFrame(rows)
    df["n"] = df["n"].astype("Int64")  # preserve the correctness-critical nullable dtype
    df["drivers_enabled"] = df["drivers_enabled"].astype(bool)
    return df


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_delta_read_equals_aggregate_only_reader(tmp_path, monkeypatch):
    """read_delta_table ≡ read_forecast_metrics(include_partitions=False).

    Pins the Delta read to the aggregate-only reader semantics and proves the
    Parquet->Delta dtype round-trip (Int64->int64, bool->bool) does not drift.
    """
    from mbal_lakehouse import read_forecast_metrics

    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    agg = _metrics_df([_metrics_row("r1", 1, 1.8), _metrics_row("r1", 6, 2.7)])
    root = _seed_gold_metrics(tmp_path, agg)
    summary = m.migrate_table(root, "forecast_metrics", replay_history=False)
    assert summary["parity_ok"] is True
    delta_sig = m._parity_signature(m.read_delta_table(root, "forecast_metrics"))
    agg_sig = m._parity_signature(read_forecast_metrics(root, include_partitions=False))
    assert delta_sig == agg_sig


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_delta_read_undercounts_unfolded_partitions_phase3_blocker(tmp_path, monkeypatch):
    """CHARACTERIZATION of the Phase-3 reader-switch blocker.

    Consumers call read_forecast_metrics(include_partitions=True), which unions the
    aggregate with un-folded run_id=* partitions. The Delta table is built from the
    aggregate ONLY, so it legitimately undercounts until partitions are folded. A
    reader switch MUST fold partitions into the aggregate first; this test fails
    loudly the day that assumption changes.
    """
    from mbal_lakehouse import read_forecast_metrics

    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    agg = _metrics_df([_metrics_row("r1", 1, 1.8), _metrics_row("r1", 6, 2.7)])
    unfolded = _metrics_df([_metrics_row("r2", 1, 0.9)])  # a run NOT yet in the aggregate
    root = _seed_gold_metrics(tmp_path, agg, partition=("r2", unfolded))
    m.migrate_table(root, "forecast_metrics", replay_history=False)

    delta_rows = len(m.read_delta_table(root, "forecast_metrics"))            # aggregate only
    union_rows = len(read_forecast_metrics(root, include_partitions=True))    # aggregate + partition
    assert delta_rows == 2
    assert union_rows == 3
    assert union_rows - delta_rows == 1, (
        "Phase-3 reader switch is unsafe while the aggregate lags un-folded "
        "run_id=* partitions; fold partitions before serving reads from Delta."
    )


@pytest.mark.skipif(not _HAS_DELTALAKE, reason="requires deltalake (Phase-0 dep)")
def test_delta_backfill_time_travel_across_reruns(tmp_path, monkeypatch):
    monkeypatch.delenv("MBAL_LAKEHOUSE_DIR", raising=False)
    base = tmp_path / "lakehouse" / "gold" / "forecast_metrics"
    agg = _metrics_df([_metrics_row("r1", 1, 1.8)])
    root = _seed_gold_metrics(tmp_path, agg)
    m.migrate_table(root, "forecast_metrics", replay_history=False)
    assert m._latest_version(root, "forecast_metrics") == 0
    # Mutate the source aggregate and re-migrate -> a new Delta version.
    (base / "metrics.parquet").unlink()
    _metrics_df([_metrics_row("r1", 1, 9.99)]).to_parquet(base / "metrics.parquet", index=False)
    m.migrate_table(root, "forecast_metrics", replay_history=False)
    assert m._latest_version(root, "forecast_metrics") == 1
    # Time-travel: version 0 still holds the original value.
    assert m.read_delta_table(root, "forecast_metrics", version=0)["model_rmse"].tolist() == [1.8]
    assert m.read_delta_table(root, "forecast_metrics")["model_rmse"].tolist() == [9.99]
