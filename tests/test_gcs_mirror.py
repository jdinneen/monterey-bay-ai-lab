"""Offline safety tests for ops/gcs_mirror.py.

These never touch the network: they assert the dry-run-by-default contract and the
two-key (--execute + --confirm) gate, so ambient credentials alone can never turn a
stray --execute into a real upload. The actual upload helpers are never invoked.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ops import gcs_mirror


def _make_lakehouse(tmp_path: Path) -> Path:
    root = tmp_path / "lakehouse"
    (root / "gold" / "forecast_metrics").mkdir(parents=True)
    (root / "gold" / "forecast_metrics" / "metrics.parquet").write_bytes(b"PAR1data")
    (root / "silver").mkdir(parents=True)
    (root / "silver" / "splits.parquet").write_bytes(b"xy")
    return root


def test_plan_mirror_builds_forward_slash_keys(tmp_path):
    root = _make_lakehouse(tmp_path)
    planned = gcs_mirror.plan_mirror(root, prefix="mbal")
    keys = sorted(p.object_name for p in planned)
    assert keys == [
        "mbal/lakehouse/gold/forecast_metrics/metrics.parquet",
        "mbal/lakehouse/silver/splits.parquet",
    ]
    # forward slashes only, regardless of local OS separator
    assert all("\\" not in k for k in keys)
    assert sum(p.size_bytes for p in planned) == len(b"PAR1data") + len(b"xy")


def test_dry_run_is_default_and_uploads_nothing(tmp_path, monkeypatch, capsys):
    root = _make_lakehouse(tmp_path)

    def _boom(*a, **k):  # pragma: no cover - must never be called in dry-run
        raise AssertionError("execute_mirror must not run in dry-run mode")

    monkeypatch.setattr(gcs_mirror, "execute_mirror", _boom)
    rc = gcs_mirror.main(["--lakehouse-dir", str(root), "--bucket", "any-bucket"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "no uploads performed" in out


def test_execute_without_confirm_is_blocked(tmp_path, monkeypatch):
    root = _make_lakehouse(tmp_path)

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("execute_mirror must not run without --confirm")

    monkeypatch.setattr(gcs_mirror, "execute_mirror", _boom)
    # credentials present should NOT matter: the --confirm gate comes first.
    monkeypatch.setattr(gcs_mirror, "_credentials_present", lambda: True)
    rc = gcs_mirror.main(
        ["--lakehouse-dir", str(root), "--bucket", "any-bucket", "--execute"]
    )
    assert rc == 2


def test_execute_confirm_without_bucket_is_blocked(tmp_path, monkeypatch):
    root = _make_lakehouse(tmp_path)
    monkeypatch.delenv("MBAL_GCS_BUCKET", raising=False)
    monkeypatch.setattr(gcs_mirror, "_credentials_present", lambda: True)
    rc = gcs_mirror.main(
        ["--lakehouse-dir", str(root), "--execute", "--confirm"]
    )
    assert rc == 2


def test_execute_confirm_without_credentials_is_blocked(tmp_path, monkeypatch):
    root = _make_lakehouse(tmp_path)
    monkeypatch.setattr(gcs_mirror, "_credentials_present", lambda: False)

    def _boom(*a, **k):  # pragma: no cover
        raise AssertionError("must not reach upload without credentials")

    monkeypatch.setattr(gcs_mirror, "execute_mirror", _boom)
    rc = gcs_mirror.main(
        ["--lakehouse-dir", str(root), "--bucket", "any-bucket", "--execute", "--confirm"]
    )
    assert rc == 2
