#!/usr/bin/env python3
"""Tests for the precious-file overwrite guard (Claude PreToolUse hook).

Mirrors tests/test_agent_lock.py: the sibling guard ships with a test, so this
one must too (VALUE_GATE.md criterion 3). Exercises the case-fold match, the
exists/new-file gate, the override+snapshot path, and fail-open behaviour.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ops.precious_guard as pg  # noqa: E402


def _run_guard(monkeypatch, payload: dict) -> int:
    """Drive guard() with a synthetic Claude hook payload on stdin."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    return pg.guard()


# --- _is_precious: the membership logic, including the Windows case trap ------

def test_is_precious_exact_names():
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "MEMORY.md", ".qwenrules", "VALUE_GATE.md"):
        assert pg._is_precious(Path(f"/some/dir/{name}")), name


def test_is_precious_is_case_insensitive():
    # The original silent hole: a lowercased path hits the same file on Windows.
    assert pg._is_precious(Path("/repo/agents.md"))
    assert pg._is_precious(Path("/repo/Gemini.Md"))
    assert pg._is_precious(Path("/repo/memory.MD"))


def test_is_precious_rejects_ordinary_files():
    assert not pg._is_precious(Path("/repo/ops/precious_guard.py"))
    assert not pg._is_precious(Path("/repo/README.rst"))
    assert not pg._is_precious(Path("/repo/notes.md"))  # not a charter name


# --- guard(): the block/allow contract (exit 2 = block, 0 = allow) -----------

def test_guard_blocks_overwrite_of_existing_precious(tmp_path, monkeypatch):
    f = tmp_path / "AGENTS.md"
    f.write_text("real charter", encoding="utf-8")
    rc = _run_guard(monkeypatch, {"tool_input": {"file_path": str(f)}})
    assert rc == 2


def test_guard_blocks_lowercase_variant(tmp_path, monkeypatch):
    # Create the real file; address it via a lowercase name (same file on a
    # case-insensitive FS). Use the real on-disk path so .exists() is true.
    f = tmp_path / "AGENTS.md"
    f.write_text("real charter", encoding="utf-8")
    lower = tmp_path / "agents.md"  # resolves to f on Win/mac; the name is what matters
    # Force the name to be lowercase while pointing at an existing file:
    monkeypatch.setattr(pg.Path, "exists", lambda self: True)
    rc = _run_guard(monkeypatch, {"tool_input": {"file_path": str(lower)}})
    assert rc == 2


def test_guard_allows_new_precious_file(tmp_path, monkeypatch):
    # Creating a brand-new charter/memory file is fine — only clobbering an
    # existing one is blocked.
    missing = tmp_path / "nope" / "MEMORY.md"
    rc = _run_guard(monkeypatch, {"tool_input": {"file_path": str(missing)}})
    assert rc == 0


def test_guard_allows_ordinary_file(tmp_path, monkeypatch):
    f = tmp_path / "script.py"
    f.write_text("print(1)", encoding="utf-8")
    rc = _run_guard(monkeypatch, {"tool_input": {"file_path": str(f)}})
    assert rc == 0


def test_guard_override_allows_and_snapshots(tmp_path, monkeypatch):
    f = tmp_path / "CLAUDE.md"
    f.write_text("original content", encoding="utf-8")
    snaps = tmp_path / "snaps"
    monkeypatch.setattr(pg, "SNAPSHOT_DIR", snaps)
    monkeypatch.setenv("ALLOW_PRECIOUS_OVERWRITE", "1")
    rc = _run_guard(monkeypatch, {"tool_input": {"file_path": str(f)}})
    assert rc == 0
    # The prior content must have been snapshotted before the overwrite.
    backups = list(snaps.glob("*.bak"))
    assert backups, "override path must snapshot the prior content"
    assert backups[0].read_text(encoding="utf-8") == "original content"


def test_guard_fails_open_on_garbage_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json{{"))
    assert pg.guard() == 0


def test_guard_allows_when_no_file_path(monkeypatch):
    assert _run_guard(monkeypatch, {"tool_input": {}}) == 0
    assert _run_guard(monkeypatch, {}) == 0
