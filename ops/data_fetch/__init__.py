"""Unified environmental data-fetch framework for Monterey Bay AI Lab.

The "ffmpeg of environmental data fetching": one command surface
(`python ops/data_fetch.py …`) over a registry of source adapters that each
support discover / dry-run / fetch / resume / validate, and emit a common
manifest + coverage + validation artifact triad.

This is NOT a rewrite of the existing fetchers; adapters wrap or call them.
See reports/data_fetch/EXISTING_FETCHER_INVENTORY.md.
"""
from __future__ import annotations

from .core import (
    Adapter,
    AdapterResult,
    Status,
    PROJECT_ROOT,
    external_raw_dir,
    external_curated_dir,
    report_dir,
    is_trusted_path,
)
from .registry import REGISTRY, SourceSpec, get_spec, validate_registry

__all__ = [
    "Adapter",
    "AdapterResult",
    "Status",
    "PROJECT_ROOT",
    "external_raw_dir",
    "external_curated_dir",
    "report_dir",
    "is_trusted_path",
    "REGISTRY",
    "SourceSpec",
    "get_spec",
    "validate_registry",
]
