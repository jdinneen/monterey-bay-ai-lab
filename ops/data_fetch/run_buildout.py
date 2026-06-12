"""Repeatable end-to-end driver for the data-source buildout.

ONE command reproduces the whole "add N new sources" pipeline for a chosen source set:
fetch (resumable/checkpointed) -> validate (>=50k labeled + columns + bounds + dup keys)
-> coverage audit (explain every gap) -> a consolidated machine- and human-readable
report. Idempotent: a re-run only does outstanding chunk work, so it is safe to run on a
schedule or in CI.

    # the new buildout sources, end-to-end:
    python -m ops.data_fetch.run_buildout

    # a specific set / band / single source:
    python -m ops.data_fetch.run_buildout --sources ndbc_stdmet,nasa_power_daily
    python -m ops.data_fetch.run_buildout --priority medium
    python -m ops.data_fetch.run_buildout --no-fetch        # just validate+audit what's landed
    python -m ops.data_fetch.run_buildout --min-rows 50000  # gate threshold (default 50k)

Exit code is non-zero if any TARGETED source is not READY_FOR_MODELING at >= min-rows,
so this doubles as a pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import PROJECT_ROOT, Status
from .registry import REGISTRY, get_spec
from . import coverage_audit

# The sources this buildout set out to ADD (beyond the original registry). Override
# with --sources / --priority. Kept here (not hardcoded in many places) so "what is the
# buildout" has one definition.
BUILDOUT_SOURCES = [
    # Lane B (claude-fetch-2) — 7
    "ndbc_stdmet", "nasa_power_daily", "gridmet_daily",
    "ceden_water_chem", "ceden_toxicity", "usgs_iv_turbidity", "open_meteo_archive",
    # Lane A (claude-fetch-A) — 8 (corrected to the real landed keys by claude-fetch-A;
    # the earlier 'modis_chl'/'obis' were never built — modis_chl was dropped as a slow
    # ERDDAP griddap aggregate, 'obis' shipped as 'obis_hab').
    "ghcnd", "cdec", "ca_snow_swc", "coops_met", "dwr_gw_continuous",
    "inaturalist_ca", "obis_hab", "dwr_groundwater",
    # Round 2 (claude-fetch-2) — 5 more large driver sources
    "usgs_dv_statewide", "cdip_wave_network", "cdip_sst_network",
    "ndbc_cwind", "ncei_isd_hourly",
]

_REPORTS = PROJECT_ROOT / "reports" / "data_fetch"


def _select(args) -> list[str]:
    if args.sources:
        return [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.priority:
        bands = {"high": lambda p: p <= 4, "medium": lambda p: 5 <= p <= 9,
                 "low": lambda p: p >= 10, "all": lambda p: True}
        test = bands.get(args.priority, bands["all"])
        return sorted([k for k, s in REGISTRY.items() if test(s.priority)],
                      key=lambda k: REGISTRY[k].priority)
    return [s for s in BUILDOUT_SOURCES if s in REGISTRY]


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", help="comma-separated source keys (default: buildout set)")
    ap.add_argument("--priority", choices=["high", "medium", "low", "all"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-rows", type=int, default=50_000)
    ap.add_argument("--no-fetch", action="store_true", help="skip fetch; validate+audit only")
    ap.add_argument("--limit-chunks", type=int, default=None)
    args = ap.parse_args(argv)

    keys = _select(args)
    print(f"[run_buildout] {len(keys)} sources: {keys}")
    rows = []
    failures = []
    for k in keys:
        if k not in REGISTRY:
            print(f"  [SKIP] unknown source {k}")
            continue
        spec = get_spec(k)
        adapter = spec.build_adapter()
        try:
            if not args.no_fetch:
                adapter.fetch(args.start, args.end, resume=True, limit_chunks=args.limit_chunks)
            v = adapter.validate(write=True)
        except Exception as exc:  # noqa: BLE001 — one source must not kill the run
            print(f"  [ERROR] {k}: {type(exc).__name__}: {exc}")
            rows.append({"source": k, "status": Status.FAILED, "rows": 0, "error": str(exc)})
            failures.append(k)
            continue
        n = int(v.get("rows", 0))
        passed = bool(v.get("passed")) and n >= args.min_rows
        status = Status.READY_FOR_MODELING if passed else (
            Status.FETCHED_NEEDS_REVIEW if n else Status.IMPLEMENTED_NOT_FETCHED)
        try:
            audit = coverage_audit.audit_source(k, probe_native=False)
            gaps = audit.get("gaps", [])
        except Exception as exc:  # noqa: BLE001
            gaps = [f"audit error: {exc}"]
        rows.append({
            "source": k, "status": status, "rows": n,
            "date_min": v.get("date_min"), "date_max": v.get("date_max"),
            "dup_keys": v.get("duplicate_key_count"),
            "bound_violations": v.get("bound_violations"),
            "coverage_gaps": gaps,
        })
        if not passed:
            failures.append(k)
        print(f"  [{status:22}] {k}: {n:,} rows"
              + ("" if passed else "  <-- NOT READY (>= %d gate)" % args.min_rows))

    summary = {
        "min_rows": args.min_rows,
        "targeted": keys,
        "ready": [r["source"] for r in rows if r["status"] == Status.READY_FOR_MODELING],
        "not_ready": failures,
        "sources": rows,
    }
    _REPORTS.mkdir(parents=True, exist_ok=True)
    (_REPORTS / "buildout_run.json").write_text(json.dumps(summary, indent=2, default=str),
                                                encoding="utf-8")
    ready = len(summary["ready"])
    print(f"\n[run_buildout] READY {ready}/{len(keys)}  |  not-ready: {failures or 'none'}")
    print(f"wrote {_REPORTS / 'buildout_run.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
