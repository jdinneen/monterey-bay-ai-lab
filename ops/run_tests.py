#!/usr/bin/env python
"""Run the local MBARI pipeline tests.

This is the stable test entrypoint for release/promotion work. It uses pytest
when available so new tests can stay idiomatic, and returns a clear install hint
instead of silently falling back to partial coverage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TESTS = [
    "tests/test_mbari_train.py",
    "tests/test_mbari_neural_forecast.py",
    "tests/test_release_gate.py",
    "tests/test_promotion_matrix.py",
    "tests/test_split_contracts.py",
    "tests/test_normalize_xgb_to_lakehouse.py",
    "tests/test_normalize_foundation_to_lakehouse.py",
    "tests/test_run_xgb_on_candidate_splits.py",
    "tests/test_sr_manager_gate.py",
    "tests/test_quarantine_incomplete_runs.py",
    "tests/test_generate_daily_learnings.py",
    "tests/test_autonomous_rate_limit_fetcher.py",
    "tests/test_repair_driver_manifest.py",
    "tests/test_fetch_news_events.py",
    "tests/test_curate_news_event_evidence.py",
    "tests/test_build_news_event_features.py",
    "tests/test_lakehouse_metrics.py",
    "tests/test_agent_lock.py",
    "tests/test_seasonal_naive.py",
    "tests/test_champion_selector.py",
    "tests/test_evidence_gate_agent.py",
    "tests/test_data_health_agent.py",
    "tests/test_split_closure_agent.py",
    "tests/test_foundation_rerun_agent.py",
    "tests/test_promotion_critic_agent.py",
    "tests/test_report_consistency_agent.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MBARI local pipeline tests with pytest.")
    parser.add_argument("tests", nargs="*", default=DEFAULT_TESTS, help="optional test paths")
    parser.add_argument("--quiet", action="store_true", help="pass -q to pytest")
    parser.add_argument(
        "--install-hint-only",
        action="store_true",
        help="only check pytest availability and print the install hint if missing",
    )
    return parser.parse_args()


def pytest_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def main() -> int:
    args = parse_args()
    if not pytest_available():
        print(
            "pytest is not installed. Install the MBARI test dependency with:\n"
            f"  {sys.executable} -m pip install -r ops/requirements-mbari-gpu.txt\n"
            "or just:\n"
            f"  {sys.executable} -m pip install pytest",
            file=sys.stderr,
        )
        return 2
    if args.install_hint_only:
        print("pytest is available")
        return 0

    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "pytest"]
    if args.quiet:
        cmd.append("-q")
    cmd.extend(args.tests)
    return subprocess.run(cmd, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
