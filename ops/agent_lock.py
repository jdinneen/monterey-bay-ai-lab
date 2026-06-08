#!/usr/bin/env python3
"""Cooperative file-claim locks for the MBARI multi-agent repo.

Multiple autonomous agents (Codex / Claude / Gemini / Qwen) share one working
tree and step on each other: two agents edit the same file, one clobbers the
other, or both redo the same task. This is a *lightweight, advisory* mutual
exclusion layer on file paths so that does not happen — while leaving disjoint
work fully parallel.

Design rules (so it is neither bloat nor a foot-gun):
- A claim is one small JSON file under ``.agent_locks/``. No daemon, no server.
- Every claim has a TTL and records the owner PID. A claim is *stale* (freely
  reclaimable) once its TTL passes or its owning process is gone, so a crashed
  agent never deadlocks the tree.
- The ``guard`` subcommand is wired to the Claude ``PreToolUse`` hook and
  **fails open**: any internal error lets the edit proceed rather than blocking
  work. It only blocks on a cleanly detected live conflict.

CLI:
  claim PATHS...   --agent ID [--task STR] [--ttl SEC]   # returns 2 on conflict
  release PATHS... --agent ID [--force]
  release-mine     --agent ID            (or reads session_id from stdin JSON)
  check PATHS...   --agent ID                            # returns 2 on conflict
  status [--all]
  guard                                  # stdin = Claude hook JSON; auto-claim
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = REPO_ROOT / ".agent_locks"
DEFAULT_TTL = 1800  # 30 minutes; refreshed on every edit by the hook
HOST = socket.gethostname()


def _now() -> float:
    return time.time()


def _normalize(path: str) -> str:
    # Accept MSYS/Git-Bash drive paths like /c/Users/... on Windows.
    if len(path) >= 3 and path[0] == "/" and path[2] == "/" and path[1].isalpha():
        path = f"{path[1].upper()}:\\{path[3:].replace('/', os.sep)}"
    return path


def _rel(path: str) -> str:
    p = Path(_normalize(path))
    if not p.is_absolute():
        p = (Path.cwd() / p)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Outside the repo: not our business to lock.
        return ""


def _lock_file(rel: str) -> Path:
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return LOCK_DIR / f"{digest}.json"


def _is_stale(rec: dict) -> bool:
    """A claim is stale once its TTL passes.

    Liveness is intentionally TTL-based, not PID-based: claims are written by
    short-lived CLI / hook subprocesses, so the recorded PID is dead almost
    immediately and would make every lock look stale. An active agent keeps its
    claims fresh by editing (the hook refreshes the TTL); a crashed or idle
    agent's claims simply expire after the TTL. ``pid``/``host`` are recorded
    for visibility only.
    """
    return float(rec.get("expires_at", 0)) < _now()


def _read(lock_file: Path) -> dict | None:
    try:
        return json.loads(lock_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write(lock_file: Path, rec: dict) -> None:
    LOCK_DIR.mkdir(exist_ok=True)
    tmp = lock_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, lock_file)


def _refresh_agent(agent: str) -> None:
    """Extend the TTL on every live claim owned by ``agent`` (proof of life)."""
    if not LOCK_DIR.exists():
        return
    for lock_file in LOCK_DIR.glob("*.json"):
        rec = _read(lock_file)
        if rec and rec.get("agent") == agent and not _is_stale(rec):
            rec["expires_at"] = _now() + DEFAULT_TTL
            _write(lock_file, rec)


def _holder(rel: str, agent: str) -> dict | None:
    """Return the live conflicting record for ``rel``, or None if free/mine/stale."""
    lock_file = _lock_file(rel)
    if not lock_file.exists():
        return None
    rec = _read(lock_file)
    if rec is None or _is_stale(rec) or rec.get("agent") == agent:
        return None
    return rec


def claim(paths: list[str], agent: str, task: str, ttl: int) -> int:
    conflicts: list[dict] = []
    to_write: list[tuple[Path, dict]] = []
    for raw in paths:
        rel = _rel(raw)
        if not rel:
            continue
        holder = _holder(rel, agent)
        if holder is not None:
            conflicts.append({"path": rel, **holder})
            continue
        rec = {
            "path": rel,
            "agent": agent,
            "task": task,
            "pid": os.getpid(),
            "host": HOST,
            "created_at": _now(),
            "expires_at": _now() + ttl,
        }
        to_write.append((_lock_file(rel), rec))
    if conflicts:
        for c in conflicts:
            mins = max(0, int((float(c["expires_at"]) - _now()) / 60))
            print(
                f"CONFLICT: {c['path']} is held by '{c['agent']}' (task: {c.get('task') or 'n/a'}; ~{mins}m left)",
                file=sys.stderr,
            )
        return 2
    for lock_file, rec in to_write:
        _write(lock_file, rec)
    print(json.dumps({"claimed": [r["path"] for _, r in to_write], "agent": agent}, sort_keys=True))
    return 0


def check(paths: list[str], agent: str) -> int:
    conflicts = []
    for raw in paths:
        rel = _rel(raw)
        if not rel:
            continue
        holder = _holder(rel, agent)
        if holder is not None:
            conflicts.append({"path": rel, "agent": holder.get("agent"), "task": holder.get("task")})
    if conflicts:
        print(json.dumps({"conflicts": conflicts}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


def release(paths: list[str], agent: str, force: bool) -> int:
    released = []
    for raw in paths:
        rel = _rel(raw)
        if not rel:
            continue
        lock_file = _lock_file(rel)
        rec = _read(lock_file) if lock_file.exists() else None
        if rec is None:
            continue
        if force or rec.get("agent") == agent:
            lock_file.unlink(missing_ok=True)
            released.append(rel)
    print(json.dumps({"released": released, "agent": agent}, sort_keys=True))
    return 0


def release_mine(agent: str) -> int:
    released = []
    if LOCK_DIR.exists():
        for lock_file in LOCK_DIR.glob("*.json"):
            rec = _read(lock_file)
            if rec and rec.get("agent") == agent:
                lock_file.unlink(missing_ok=True)
                released.append(rec.get("path"))
    print(json.dumps({"released": released, "agent": agent}, sort_keys=True))
    return 0


def status(show_all: bool) -> int:
    rows = []
    if LOCK_DIR.exists():
        for lock_file in sorted(LOCK_DIR.glob("*.json")):
            rec = _read(lock_file)
            if rec is None:
                continue
            stale = _is_stale(rec)
            if stale and not show_all:
                continue
            mins = int((float(rec.get("expires_at", 0)) - _now()) / 60)
            rows.append((rec.get("path"), rec.get("agent"), rec.get("task") or "", mins, "STALE" if stale else "live"))
    if not rows:
        print("no active claims")
        return 0
    width = max(len(str(r[0])) for r in rows)
    for path, agent, task, mins, state in rows:
        print(f"{state:5} {str(path):<{width}}  {agent}  ({mins:+d}m)  {task}")
    return 0


def _stdin_payload() -> dict:
    try:
        data = sys.stdin.read()
        return json.loads(data) if data.strip() else {}
    except Exception:
        return {}


def guard() -> int:
    """Claude PreToolUse hook: auto-claim the edited file, block on live conflict.

    Fails open on any error — the coordination layer must never brick edits.
    """
    try:
        payload = _stdin_payload()
        agent = str(payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "claude")
        tool_input = payload.get("tool_input") or {}
        file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not file_path:
            return 0
        rel = _rel(str(file_path))
        if not rel:
            return 0
        holder = _holder(rel, agent)
        if holder is not None:
            mins = max(0, int((float(holder["expires_at"]) - _now()) / 60))
            print(
                f"[agent-lock] BLOCKED: {rel} is being edited by another agent "
                f"('{holder.get('agent')}', task: {holder.get('task') or 'n/a'}, ~{mins}m left). "
                f"Work on a different file, or run `python ops/agent_lock.py release {rel} --force` "
                f"if that agent is dead.",
                file=sys.stderr,
            )
            return 2
        # Free / mine / stale -> (re)claim and refresh the TTL.
        _write(
            _lock_file(rel),
            {
                "path": rel,
                "agent": agent,
                "task": str(payload.get("task") or "claude edit session"),
                "pid": os.getpid(),
                "host": HOST,
                "created_at": _now(),
                "expires_at": _now() + DEFAULT_TTL,
            },
        )
        # The agent is clearly alive: extend all of its other claims too, so a
        # long edit on one file does not let another agent's claim expire.
        _refresh_agent(agent)
        return 0
    except Exception as exc:  # fail open
        print(f"[agent-lock] guard error (failing open): {exc}", file=sys.stderr)
        return 0


def release_mine_from_stdin() -> int:
    payload = _stdin_payload()
    agent = str(payload.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "claude")
    return release_mine(agent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("paths", nargs="+")
    p_claim.add_argument("--agent", required=True)
    p_claim.add_argument("--task", default="")
    p_claim.add_argument("--ttl", type=int, default=DEFAULT_TTL)

    p_check = sub.add_parser("check")
    p_check.add_argument("paths", nargs="+")
    p_check.add_argument("--agent", required=True)

    p_rel = sub.add_parser("release")
    p_rel.add_argument("paths", nargs="+")
    p_rel.add_argument("--agent", required=True)
    p_rel.add_argument("--force", action="store_true")

    p_relmine = sub.add_parser("release-mine")
    p_relmine.add_argument("--agent", default=None)

    p_status = sub.add_parser("status")
    p_status.add_argument("--all", action="store_true")

    sub.add_parser("guard")

    args = parser.parse_args()
    if args.cmd == "claim":
        return claim(args.paths, args.agent, args.task, args.ttl)
    if args.cmd == "check":
        return check(args.paths, args.agent)
    if args.cmd == "release":
        return release(args.paths, args.agent, args.force)
    if args.cmd == "release-mine":
        return release_mine(args.agent) if args.agent else release_mine_from_stdin()
    if args.cmd == "status":
        return status(args.all)
    if args.cmd == "guard":
        return guard()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
