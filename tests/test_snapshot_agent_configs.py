#!/usr/bin/env python3
"""Tests for the agent-config snapshot backup.

Covers the two safety-critical behaviours — secrets are never copied, and an
unchanged source set is a no-op — plus rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ops.snapshot_agent_configs as snap  # noqa: E402


def test_is_secret_flags_credentials():
    for name in (
        "oauth_creds.json",
        ".credentials.json",
        "google_accounts.json",
        "api-keys.csv",
        "apikey.txt",
        "id_rsa",
        "server.pem",
        "auth.json",  # .codex/auth.json exists on the host; belt-and-suspenders
        ".netrc",
    ):
        assert snap.is_secret(name), name


def test_is_secret_allows_config_and_memory():
    for name in ("GEMINI.md", "CLAUDE.md", "MEMORY.md", "settings.json", "config.toml", "default.rules"):
        assert not snap.is_secret(name), name


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point the module at an isolated HOME + backup root with one fake agent."""
    home = tmp_path / "home"
    (home / ".gemini").mkdir(parents=True)
    (home / ".gemini" / "GEMINI.md").write_text("rules v1", encoding="utf-8")
    # A secret sitting right next to a curated file must NOT be copied.
    (home / ".gemini" / "oauth_creds.json").write_text("SECRET-TOKEN", encoding="utf-8")
    monkeypatch.setattr(snap, "HOME", home)
    monkeypatch.setattr(snap, "BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(snap, "SOURCES", {"gemini": [".gemini/GEMINI.md", ".gemini/oauth_creds.json"]})
    return home


def test_secret_is_never_copied(fake_home):
    snap.run(stamp="2026-01-01_000000")
    copied = [p.name for p in (snap.BACKUP_ROOT / "2026-01-01_000000").rglob("*") if p.is_file()]
    assert "GEMINI.md" in copied
    assert "oauth_creds.json" not in copied  # the secret was excluded


def test_unchanged_set_is_a_noop(fake_home):
    snap.run(stamp="2026-01-01_000000")
    snap.run(stamp="2026-01-02_000000")  # nothing changed
    dirs = sorted(d.name for d in snap.BACKUP_ROOT.iterdir() if d.is_dir())
    assert dirs == ["2026-01-01_000000"]  # no second snapshot written


def test_changed_content_makes_new_snapshot(fake_home):
    snap.run(stamp="2026-01-01_000000")
    (fake_home / ".gemini" / "GEMINI.md").write_text("rules v2", encoding="utf-8")
    snap.run(stamp="2026-01-02_000000")
    dirs = sorted(d.name for d in snap.BACKUP_ROOT.iterdir() if d.is_dir())
    assert dirs == ["2026-01-01_000000", "2026-01-02_000000"]


def test_partial_copy_does_not_poison_marker(fake_home, monkeypatch):
    # If a copy fails, the dedup marker must NOT advance — otherwise the next
    # identical run no-ops and the snapshot stays permanently incomplete.
    real_copy = snap.shutil.copy2

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(snap.shutil, "copy2", boom)
    snap.run(stamp="2026-01-01_000000")
    assert not (snap.BACKUP_ROOT / "LATEST.sha").exists()  # marker not written

    # Restore ONLY copy2 (keep the fake_home patches); the next run must capture.
    monkeypatch.setattr(snap.shutil, "copy2", real_copy)
    snap.run(stamp="2026-01-02_000000")
    assert (snap.BACKUP_ROOT / "LATEST.sha").exists()
    assert (snap.BACKUP_ROOT / "2026-01-02_000000").is_dir()


def test_rotation_keeps_last_n(fake_home, monkeypatch):
    monkeypatch.setattr(snap, "KEEP", 2)
    for i, content in enumerate(["a", "b", "c", "d"]):
        (fake_home / ".gemini" / "GEMINI.md").write_text(content, encoding="utf-8")
        snap.run(stamp=f"2026-01-0{i + 1}_000000")
    dirs = sorted(d.name for d in snap.BACKUP_ROOT.iterdir() if d.is_dir())
    assert dirs == ["2026-01-03_000000", "2026-01-04_000000"]  # oldest two rotated out
