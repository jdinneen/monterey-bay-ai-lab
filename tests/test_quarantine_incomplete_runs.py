#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import quarantine_incomplete_runs as qir  # noqa: E402


def write_complete_run(run_dir: Path, cache_version: str = "v1") -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"cache_version": cache_version}), encoding="utf-8"
    )
    (run_dir / "run.log").write_text("training\nDONE\n", encoding="utf-8")
    (run_dir / "leaderboard.csv").write_text("model,rmse\npatchtst,0.1\n", encoding="utf-8")


def test_find_incomplete_runs_reports_each_required_reason(tmp_path):
    nn_results = tmp_path / "nn_results"
    nn_results.mkdir()

    complete = nn_results / "complete"
    write_complete_run(complete)

    missing_summary = nn_results / "missing_summary"
    missing_summary.mkdir()
    (missing_summary / "run.log").write_text("DONE\n", encoding="utf-8")
    (missing_summary / "leaderboard.csv").write_text("model,rmse\nx,1.0\n", encoding="utf-8")

    no_done = nn_results / "no_done"
    write_complete_run(no_done)
    (no_done / "run.log").write_text("still running\n", encoding="utf-8")

    header_only = nn_results / "header_only"
    write_complete_run(header_only)
    (header_only / "leaderboard.csv").write_text("model,rmse\n", encoding="utf-8")

    found = {candidate.path.name: candidate.reasons for candidate in qir.find_incomplete_runs(nn_results)}

    assert "complete" not in found
    assert found["missing_summary"] == ("missing summary.json",)
    assert found["no_done"] == ("run.log missing DONE",)
    assert found["header_only"] == ("leaderboard.csv missing, empty, or header-only",)


def test_cache_version_marks_stale_summary_incomplete(tmp_path):
    nn_results = tmp_path / "nn_results"
    nn_results.mkdir()
    write_complete_run(nn_results / "stale", cache_version="old")
    write_complete_run(nn_results / "fresh", cache_version="expected")

    found = {
        candidate.path.name: candidate.reasons
        for candidate in qir.find_incomplete_runs(nn_results, cache_version="expected")
    }

    assert "fresh" not in found
    assert found["stale"] == ("stale cache_version: 'old' != 'expected'",)


def test_default_dry_run_does_not_move_candidates(tmp_path):
    nn_results = tmp_path / "nn_results"
    nn_results.mkdir()
    run_dir = nn_results / "incomplete"
    run_dir.mkdir()

    results = qir.quarantine_incomplete_runs(nn_results, date="20260607")

    assert len(results) == 1
    assert results[0].moved is False
    assert results[0].destination == nn_results.resolve() / "_quarantine" / "20260607" / "incomplete"
    assert run_dir.exists()
    assert not (nn_results / "_quarantine").exists()


def test_apply_moves_candidates_and_handles_name_collisions(tmp_path):
    nn_results = tmp_path / "nn_results"
    nn_results.mkdir()
    run_dir = nn_results / "incomplete"
    run_dir.mkdir()
    existing = nn_results / "_quarantine" / "20260607" / "incomplete"
    existing.mkdir(parents=True)

    results = qir.quarantine_incomplete_runs(nn_results, apply=True, date="20260607")

    assert len(results) == 1
    assert results[0].moved is True
    assert results[0].destination == nn_results.resolve() / "_quarantine" / "20260607" / "incomplete_1"
    assert not run_dir.exists()
    assert existing.exists()
    assert (nn_results / "_quarantine" / "20260607" / "incomplete_1").exists()


def test_quarantine_directory_is_not_scanned(tmp_path):
    nn_results = tmp_path / "nn_results"
    quarantined = nn_results / "_quarantine" / "20260607" / "old_incomplete"
    quarantined.mkdir(parents=True)
    write_complete_run(nn_results / "complete")

    assert qir.find_incomplete_runs(nn_results) == []


def _run_with_tmp_path(test_func):
    with tempfile.TemporaryDirectory() as td:
        test_func(Path(td))


if __name__ == "__main__":
    tests = [
        test_find_incomplete_runs_reports_each_required_reason,
        test_cache_version_marks_stale_summary_incomplete,
        test_default_dry_run_does_not_move_candidates,
        test_apply_moves_candidates_and_handles_name_collisions,
        test_quarantine_directory_is_not_scanned,
    ]
    for test in tests:
        _run_with_tmp_path(test)
    print(f"{len(tests)} passed")
