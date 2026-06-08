#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class QuarantineCandidate:
    path: Path
    reasons: tuple[str, ...]
    destination: Path | None = None
    moved: bool = False


def _leaderboard_is_empty_or_header_only(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return True
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = [row for row in csv.reader(fh) if any(cell.strip() for cell in row)]
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = [row for row in csv.reader(fh) if any(cell.strip() for cell in row)]
    return len(rows) <= 1


def _summary_cache_version(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("cache_version")
    return None if value is None else str(value)


def incomplete_reasons(run_dir: Path, cache_version: str | None = None) -> tuple[str, ...]:
    reasons: list[str] = []

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        reasons.append("missing summary.json")
    elif cache_version is not None:
        found = _summary_cache_version(summary_path)
        if found != str(cache_version):
            reasons.append(f"stale cache_version: {found!r} != {str(cache_version)!r}")

    run_log = run_dir / "run.log"
    try:
        log_text = run_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""
    if "DONE" not in log_text:
        reasons.append("run.log missing DONE")

    leaderboard = run_dir / "leaderboard.csv"
    if _leaderboard_is_empty_or_header_only(leaderboard):
        reasons.append("leaderboard.csv missing, empty, or header-only")

    return tuple(reasons)


def iter_run_dirs(nn_results: Path) -> Iterable[Path]:
    if not nn_results.exists():
        return ()
    return (
        child
        for child in sorted(nn_results.iterdir(), key=lambda p: p.name.lower())
        if child.is_dir() and child.name != "_quarantine"
    )


def find_incomplete_runs(nn_results: Path, cache_version: str | None = None) -> list[QuarantineCandidate]:
    candidates: list[QuarantineCandidate] = []
    for run_dir in iter_run_dirs(nn_results):
        reasons = incomplete_reasons(run_dir, cache_version=cache_version)
        if reasons:
            candidates.append(QuarantineCandidate(path=run_dir, reasons=reasons))
    return candidates


def collision_safe_destination(quarantine_root: Path, name: str) -> Path:
    destination = quarantine_root / name
    if not destination.exists():
        return destination
    index = 1
    while True:
        candidate = quarantine_root / f"{name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def quarantine_incomplete_runs(
    nn_results: Path,
    *,
    apply: bool = False,
    cache_version: str | None = None,
    date: str | None = None,
) -> list[QuarantineCandidate]:
    nn_results = nn_results.resolve()
    candidates = find_incomplete_runs(nn_results, cache_version=cache_version)
    stamp = date or datetime.now().strftime("%Y%m%d")
    quarantine_root = nn_results / "_quarantine" / stamp

    results: list[QuarantineCandidate] = []
    for candidate in candidates:
        destination = collision_safe_destination(quarantine_root, candidate.path.name)
        moved = False
        if apply:
            quarantine_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate.path), str(destination))
            moved = True
        results.append(
            QuarantineCandidate(
                path=candidate.path,
                reasons=candidate.reasons,
                destination=destination,
                moved=moved,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first quarantine for incomplete nn_results run directories."
    )
    parser.add_argument(
        "--nn-results",
        type=Path,
        default=Path("nn_results"),
        help="Path to nn_results. Defaults to ./nn_results.",
    )
    parser.add_argument(
        "--cache-version",
        default=None,
        help="Expected summary.json cache_version. Runs with a different value are incomplete.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move incomplete runs into nn_results/_quarantine/YYYYMMDD. Default is dry-run.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override quarantine date stamp as YYYYMMDD. Primarily useful for repeatable automation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = quarantine_incomplete_runs(
        args.nn_results,
        apply=args.apply,
        cache_version=args.cache_version,
        date=args.date,
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: {len(results)} incomplete run(s) found")
    for result in results:
        action = "moved" if result.moved else "would move"
        reasons = "; ".join(result.reasons)
        print(f"- {result.path} -> {result.destination} [{action}; {reasons}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
