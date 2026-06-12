#!/usr/bin/env python3
"""Entrypoint for the unified environmental data-fetch framework.

The "ffmpeg of environmental data fetching" for Monterey Bay AI Lab: one command
surface over a registry of source adapters (discover / dry-run / fetch / resume /
validate) that wrap or call the repo's existing fetchers and emit a common
manifest + coverage + validation artifact triad under data/external_* and
reports/data_fetch/.

    python ops/data_fetch.py list
    python ops/data_fetch.py inventory
    python ops/data_fetch.py discover --source ciwqs_sso
    python ops/data_fetch.py dry-run  --source wqp --start 2020-01-01 --end 2026-06-10
    python ops/data_fetch.py fetch     --source ciwqs_sso
    python ops/data_fetch.py validate  --source ciwqs_sso
    python ops/data_fetch.py report
    python ops/data_fetch.py fetch-all --priority high --max-workers 2

See reports/data_fetch/EXISTING_FETCHER_INVENTORY.md and CRITIC_PROOF_REPORT.md.
"""
from __future__ import annotations

import os
import sys

# Allow `python ops/data_fetch.py …` to import the ops.data_fetch package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.data_fetch.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
