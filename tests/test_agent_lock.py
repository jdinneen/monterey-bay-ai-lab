#!/usr/bin/env python3
"""Tests for the cooperative multi-agent file-claim lock."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ops.agent_lock as agent_lock  # noqa: E402


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    """Point the lock module at an isolated repo root + lock dir."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("y", encoding="utf-8")
    monkeypatch.setattr(agent_lock, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(agent_lock, "LOCK_DIR", tmp_path / ".agent_locks")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_claim_blocks_second_agent(lock_env):
    assert agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=1800) == 0
    # Different agent claiming the same path conflicts.
    assert agent_lock.claim(["a.py"], agent="claude-9", task="t", ttl=1800) == 2
    # check() agrees.
    assert agent_lock.check(["a.py"], agent="claude-9") == 2


def test_disjoint_files_run_in_parallel(lock_env):
    assert agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=1800) == 0
    assert agent_lock.claim(["b.py"], agent="claude-9", task="t", ttl=1800) == 0
    assert agent_lock.check(["b.py"], agent="claude-9") == 0


def test_same_agent_reclaims_its_own(lock_env):
    assert agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=1800) == 0
    assert agent_lock.claim(["a.py"], agent="codex-1", task="t2", ttl=1800) == 0
    assert agent_lock.check(["a.py"], agent="codex-1") == 0


def test_expired_lock_is_reclaimable(lock_env):
    assert agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=0) == 0
    # TTL 0 -> immediately stale -> a different agent can take it.
    assert agent_lock.check(["a.py"], agent="claude-9") == 0
    assert agent_lock.claim(["a.py"], agent="claude-9", task="t", ttl=1800) == 0


def test_release_mine_only_releases_owner(lock_env):
    agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=1800)
    agent_lock.claim(["b.py"], agent="claude-9", task="t", ttl=1800)
    agent_lock.release_mine("codex-1")
    assert agent_lock.check(["a.py"], agent="claude-9") == 0  # freed
    assert agent_lock.check(["b.py"], agent="codex-1") == 2  # still held


def test_release_force_overrides_owner(lock_env):
    agent_lock.claim(["a.py"], agent="codex-1", task="t", ttl=1800)
    agent_lock.release(["a.py"], agent="someone-else", force=True)
    assert agent_lock.check(["a.py"], agent="codex-1") == 0


def test_guard_auto_claims_and_blocks(lock_env, monkeypatch, capsys):
    target = str(lock_env / "a.py")

    def feed(payload):
        monkeypatch.setattr(agent_lock.sys, "stdin", _Stdin(json.dumps(payload)))

    # First agent edits a.py -> allowed, auto-claimed.
    feed({"session_id": "claude-1", "tool_input": {"file_path": target}})
    assert agent_lock.guard() == 0
    # Second agent editing the same file is blocked.
    feed({"session_id": "claude-2", "tool_input": {"file_path": target}})
    assert agent_lock.guard() == 2


def test_guard_fails_open_on_garbage(lock_env, monkeypatch):
    monkeypatch.setattr(agent_lock.sys, "stdin", _Stdin("not json at all"))
    assert agent_lock.guard() == 0  # never block on a lock-layer error


def test_guard_ignores_paths_outside_repo(lock_env, monkeypatch):
    monkeypatch.setattr(
        agent_lock.sys,
        "stdin",
        _Stdin(json.dumps({"session_id": "x", "tool_input": {"file_path": "/etc/passwd"}})),
    )
    assert agent_lock.guard() == 0


class _Stdin:
    def __init__(self, data: str):
        self._data = data

    def read(self) -> str:
        return self._data
