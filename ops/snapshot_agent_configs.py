#!/usr/bin/env python3
"""Periodic snapshot of every terminal agent's UNTRACKED rules/memory/config.

Why this exists (incident 2026-06-10): an autonomous Gemini YOLO session
blind-overwrote the global ``~/.gemini/GEMINI.md`` and a Qwen ``MEMORY.md``.
Those files live OUTSIDE any git repo, so git history — the recovery floor for
the repo's tracked charters — does not cover them. The Claude PreToolUse
``precious_guard`` only binds Claude. So for the cross-agent case (Codex /
Gemini / Qwen / Claude all run in YOLO on one box), the honest floor is a
versioned backup of these files taken on a timer.

This is that backup. It is deliberately NARROW and SAFE:
- **Curated allow-list only** — the durable rules/memory/config each terminal
  agent boots from. NOT caches, sessions, task-state, or transcripts (those are
  already mirrored by ai-transcript-mirror.ps1).
- **Secrets are excluded, hard** — any file whose name looks like a credential
  (oauth/token/cred/api-key/account/.pem/password) is skipped even if a glob
  catches it. Backups must never become a secret-sprawl vector.
- **Content-hash dedup** — if nothing changed since the last run, no new
  snapshot is written (anti-bloat; this runs on the mirror timer).
- **Rotation** — keep the most recent ``KEEP`` snapshots.
- Backups live under ``~/.agent-config-backups/`` — OUTSIDE the repo, so this
  content can never be committed.

Run: ``python ops/snapshot_agent_configs.py`` (also callable from the mirror PS1).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
BACKUP_ROOT = HOME / ".agent-config-backups"
KEEP = 30

# Defense-in-depth: never copy anything that looks like a secret, even if a glob
# below were to catch it. Matched against the file name, case-insensitively.
SECRET_RE = re.compile(
    r"(cred|oauth|token|secret|api[-_]?key|apikey|account|password|auth|\.netrc|\.pem$|\.key$|id_rsa)",
    re.IGNORECASE,
)

# Curated, per-agent: the UNTRACKED durable rules/memory/config. Globs are
# resolved under HOME unless absolute. Missing paths are simply skipped.
SOURCES: dict[str, list[str]] = {
    "codex": [".codex/config.toml", ".codex/AGENTS.md", ".codex/rules/*.rules"],
    "gemini": [".gemini/GEMINI.md", ".gemini/settings.json"],
    "qwen": [
        ".qwen/settings.json",
        ".qwen/output-language.md",
        "Desktop/.qwen-memory/memory/*.md",
    ],
    "claude": [
        ".claude/CLAUDE.md",
        ".claude/settings.json",
        ".claude/projects/*/memory/*.md",
    ],
}


def is_secret(name: str) -> bool:
    return bool(SECRET_RE.search(name))


def iter_source_files() -> list[tuple[str, Path]]:
    """Yield (agent, file) for every existing, non-secret curated source."""
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for agent, patterns in SOURCES.items():
        for pat in patterns:
            base = Path(pat)
            matches = [base] if base.is_absolute() else list(HOME.glob(pat))
            for f in matches:
                if not f.is_file():
                    continue
                if is_secret(f.name):
                    continue
                rf = f.resolve()
                if rf in seen:
                    continue
                seen.add(rf)
                out.append((agent, f))
    return out


def manifest_hash(files: list[tuple[str, Path]]) -> str:
    """Stable hash over (agent, name, bytes) so an unchanged set is a no-op."""
    h = hashlib.sha256()
    for agent, f in sorted(files, key=lambda t: (t[0], t[1].name)):
        h.update(agent.encode("utf-8"))
        h.update(b"\0")
        h.update(f.name.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def _latest_marker() -> Path:
    return BACKUP_ROOT / "LATEST.sha"


def _rotate(keep: int) -> None:
    if not BACKUP_ROOT.exists():
        return
    snaps = sorted(d for d in BACKUP_ROOT.iterdir() if d.is_dir())
    for old in snaps[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def run(stamp: str | None = None) -> int:
    files = iter_source_files()
    if not files:
        print("[config-snapshot] no curated source files found; nothing to do")
        return 0

    digest = manifest_hash(files)
    marker = _latest_marker()
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == digest:
        print(f"[config-snapshot] no change since last snapshot ({len(files)} files)")
        return 0

    stamp = stamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest_root = BACKUP_ROOT / stamp
    n = 0
    for agent, f in files:
        # Key the destination on the source's path under HOME (not the bare name)
        # so two same-named files (e.g. MEMORY.md from two project dirs) can't
        # collide and silently drop one.
        try:
            rel = f.resolve().relative_to(HOME)
        except ValueError:
            rel = Path(f.name)
        dest = dest_root / agent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(f, dest)
            n += 1
        except OSError as exc:
            print(f"[config-snapshot] skip {f}: {exc}", file=sys.stderr)

    _rotate(KEEP)
    # Only advance the dedup marker if EVERY source copied. A partial snapshot
    # must not poison the marker — otherwise the next identical run no-ops and
    # the gap is permanent. Leaving the marker stale makes the next run retry.
    if n == len(files):
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        marker.write_text(digest, encoding="utf-8")
        (BACKUP_ROOT / "_heartbeat.txt").write_text(
            f"last snapshot : {stamp}\nfiles         : {n}\ndigest        : {digest[:16]}\n",
            encoding="utf-8",
        )
        print(f"[config-snapshot] wrote {n} file(s) -> {dest_root}")
    else:
        print(
            f"[config-snapshot] PARTIAL {n}/{len(files)} copied -> {dest_root}; "
            f"marker NOT advanced (next run retries)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
