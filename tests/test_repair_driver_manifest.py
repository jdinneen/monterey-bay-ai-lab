#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.repair_driver_manifest import repair_manifest  # noqa: E402


def test_repair_manifest_drops_high_fill_lift_hist_driver():
    manifest = {
        "hist": ["good", "filled"],
        "coverage": {"good": 0.8, "filled": 0.99},
        "hist_raw_coverage": {"good": 0.75, "filled": 0.04},
    }

    repaired, dropped = repair_manifest(manifest, max_fill_lift=0.25, min_raw_coverage=0.25)

    assert dropped == ["filled"]
    assert repaired["hist"] == ["good"]
    assert repaired["production_policy"]["excluded_hist_drivers"] == ["filled"]
