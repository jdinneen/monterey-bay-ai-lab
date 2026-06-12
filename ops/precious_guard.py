#!/usr/bin/env python3
"""Block blind full-file overwrites of "constitution" files.

Motive (a real incident, 2026-06-10): an autonomous Gemini session, while
"distilling" value-gate rules, used a full-file *Write* to overwrite the global
``~/.gemini/GEMINI.md`` and a Qwen ``MEMORY.md`` — replacing whatever was there
with new content. No diff, no backup, no way to know what was lost. These files
are the agents' standing instructions and memory; a blind overwrite silently
rewrites the rules every agent boots with.

This guard encodes one rule: **you may not clobber an existing precious file
with a whole-file Write. Use a targeted Edit instead** (which preserves the rest
of the file and is reviewable as a diff). Creating a *new* precious file is
fine; editing one is fine; only blind overwrite of an existing one is blocked.

Scope & honesty about limits (do not oversell this):
- Wired as a Claude ``PreToolUse`` hook on the ``Write`` tool only, *by design*.
  Targeted ``Edit``/``MultiEdit`` are intentionally trusted — they cannot clobber
  blindly (they must match existing text) and a targeted edit is exactly the
  behaviour we want on these files. Only a whole-file ``Write`` is blocked.
- Binds *Claude only*. Gemini/Qwen/Codex don't run Claude hooks, and a blind write
  from them — or any ``Bash``/PowerShell redirect (``> AGENTS.md``) from inside
  Claude — never reaches this hook, so it is NOT blocked and leaves NO snapshot.
  For those paths the real floor is git history + periodic manual snapshots + the
  prose rule in each agent's config. Treat this as a Claude-side seatbelt, NOT a
  cross-agent safety net.
- Fails **open**: any internal error lets the write proceed. A guard must never
  brick legitimate work.
- Escape hatch: set ``ALLOW_PRECIOUS_OVERWRITE=1`` to intentionally overwrite via
  the Write tool; the guard snapshots the prior content into the snapshot dir
  first (this snapshot only happens on this Claude Write path, per the limit above).

CLI:
  guard           # stdin = Claude hook JSON; exit 2 blocks the Write
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# Files that are an agent's standing rules or memory. Matched by basename so the
# guard covers the same-named file wherever it lives (global config dirs, repo
# root, memory stores). Keep this list TIGHT — only true "constitution" files,
# or it becomes friction.
PRECIOUS_NAMES = {
    "GEMINI.md",
    "CLAUDE.md",
    "AGENTS.md",
    "QWEN.md",
    "MEMORY.md",
    ".qwenrules",
    "VALUE_GATE.md",
}
# Windows/macOS filesystems are case-insensitive: 'agents.md' is the SAME file as
# 'AGENTS.md', so an exact-case match has a silent hole (a lowercased path sails
# through and clobbers the real file). Match case-folded. Precomputed at import.
_PRECIOUS_FOLDED = {n.casefold() for n in PRECIOUS_NAMES}

SNAPSHOT_DIR = Path(__file__).resolve().parent / ".precious_snapshots"


def _stdin_payload() -> dict:
    try:
        data = sys.stdin.read()
        return json.loads(data) if data.strip() else {}
    except Exception:
        return {}


def _is_precious(path: Path) -> bool:
    return path.name.casefold() in _PRECIOUS_FOLDED


def _snapshot(path: Path) -> Path | None:
    """Copy the current file into the snapshot dir before an allowed overwrite."""
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        safe = str(path.resolve()).replace(":", "_").replace(os.sep, "_").replace("/", "_")
        dest = SNAPSHOT_DIR / f"{safe}.bak"
        shutil.copy2(path, dest)
        return dest
    except Exception:
        return None


def guard() -> int:
    try:
        payload = _stdin_payload()
        tool_input = payload.get("tool_input") or {}
        file_path = tool_input.get("file_path")
        if not file_path:
            return 0
        path = Path(str(file_path))

        # Only blind overwrites of an *existing* precious file are blocked.
        # New-file creation and targeted Edits are unaffected.
        if not _is_precious(path) or not path.exists():
            return 0

        if os.environ.get("ALLOW_PRECIOUS_OVERWRITE") == "1":
            snap = _snapshot(path)
            note = f" (prior content saved to {snap})" if snap else ""
            print(f"[precious-guard] override: overwriting {path.name}{note}", file=sys.stderr)
            return 0

        print(
            f"[precious-guard] BLOCKED: '{path.name}' is a standing rules/memory file; "
            f"a full-file Write would clobber it with no diff. Use a targeted Edit to "
            f"change specific lines, or -- if you truly mean to replace the whole file -- "
            f"re-run with ALLOW_PRECIOUS_OVERWRITE=1 (the old content is snapshotted first).",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # fail open
        print(f"[precious-guard] error (failing open): {exc}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "guard":
        raise SystemExit(guard())
    # Default to guard so a bare invocation from the hook still works.
    raise SystemExit(guard())
