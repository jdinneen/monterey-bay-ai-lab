#!/usr/bin/env python3
"""Repair the production driver manifest by excluding weak raw-coverage hist drivers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repair_manifest(manifest: dict[str, Any], *, max_fill_lift: float, min_raw_coverage: float) -> tuple[dict[str, Any], list[str]]:
    out = json.loads(json.dumps(manifest))
    hist = list(out.get("hist", []))
    coverage = out.get("coverage", {})
    raw = out.get("hist_raw_coverage", {})
    dropped: list[str] = []
    for col in hist:
        raw_cov = float(raw.get(col, 0.0))
        filled_cov = float(coverage.get(col, raw_cov))
        if raw_cov < min_raw_coverage or filled_cov - raw_cov > max_fill_lift:
            dropped.append(col)
    out["hist"] = [col for col in hist if col not in set(dropped)]
    policy = out.setdefault("production_policy", {})
    policy["excluded_hist_drivers"] = dropped
    policy["exclusion_reason"] = (
        f"Excluded hist drivers with raw coverage < {min_raw_coverage:.2f} "
        f"or fill coverage lift > {max_fill_lift:.2f}."
    )
    return out, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("nn_cache/drivers_manifest.json"))
    parser.add_argument("--max-fill-lift", type=float, default=0.25)
    parser.add_argument("--min-raw-coverage", type=float, default=0.25)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repaired, dropped = repair_manifest(
        manifest,
        max_fill_lift=args.max_fill_lift,
        min_raw_coverage=args.min_raw_coverage,
    )
    if args.apply:
        args.manifest.write_text(json.dumps(repaired, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"applied": args.apply, "dropped": dropped, "hist_count": len(repaired.get("hist", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
