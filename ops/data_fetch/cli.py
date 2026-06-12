"""Command surface for the data-fetch framework.

    python ops/data_fetch.py list
    python ops/data_fetch.py inventory
    python ops/data_fetch.py discover --source ciwqs_sso
    python ops/data_fetch.py dry-run  --source wqp --start 2020-01-01 --end 2026-06-10
    python ops/data_fetch.py fetch     --source ciwqs_sso
    python ops/data_fetch.py validate  --source ciwqs_sso
    python ops/data_fetch.py report
    python ops/data_fetch.py fetch-all --priority high --max-workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import PROJECT_ROOT, Status, report_dir
from .registry import REGISTRY, get_spec, validate_registry

_REPORTS_ROOT = PROJECT_ROOT / "reports" / "data_fetch"

PRIORITY_BANDS = {
    "high": lambda p: p <= 4,
    "medium": lambda p: 5 <= p <= 9,
    "low": lambda p: p >= 10,
    "all": lambda p: True,
}


def _sources_by_priority(band: str) -> list[str]:
    test = PRIORITY_BANDS.get(band, PRIORITY_BANDS["all"])
    keys = [k for k, s in REGISTRY.items() if test(s.priority)]
    return sorted(keys, key=lambda k: REGISTRY[k].priority)


def cmd_list(args) -> int:
    rows = sorted(REGISTRY.values(), key=lambda s: s.priority)
    print(f"{'prio':>4}  {'source':24} {'type':14} {'mode':12} title")
    print("-" * 100)
    for s in rows:
        mode = "wrap_trusted" if s.wraps_trusted else ("creds" if s.needs_credentials else "staged")
        print(f"{s.priority:>4}  {s.key:24} {s.type:14} {mode:12} {s.title}")
    problems = validate_registry()
    if problems:
        print("\n[registry problems]")
        for p in problems:
            print("  -", p)
        return 1
    print(f"\n{len(rows)} sources · registry valid")
    return 0


def cmd_inventory(args) -> int:
    inv = _REPORTS_ROOT / "EXISTING_FETCHER_INVENTORY.md"
    print(f"Existing-fetcher inventory: {inv}")
    print("Sources wrapping trusted outputs (read-only):")
    for s in sorted(REGISTRY.values(), key=lambda x: x.priority):
        if s.wraps_trusted:
            p = PROJECT_ROOT / s.wraps_trusted
            print(f"  - {s.key:22} -> {s.wraps_trusted}  [{'present' if p.exists() else 'MISSING'}]")
    return 0


def cmd_discover(args) -> int:
    spec = get_spec(args.source)
    info = spec.build_adapter().discover(args.start, args.end)
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_dry_run(args) -> int:
    spec = get_spec(args.source)
    plan = spec.build_adapter().dry_run(args.start, args.end)
    print(json.dumps(plan, indent=2, default=str))
    return 0


def cmd_fetch(args) -> int:
    spec = get_spec(args.source)
    res = spec.build_adapter().fetch(
        args.start, args.end, resume=not args.no_resume, limit_chunks=args.limit_chunks
    )
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0 if res.status in (Status.READY_FOR_MODELING, Status.FETCHED_NEEDS_REVIEW) else 0


def cmd_validate(args) -> int:
    spec = get_spec(args.source)
    v = spec.build_adapter().validate(write=True)
    print(json.dumps(v, indent=2, default=str))
    return 0 if v.get("passed") else 1


def _status_for_source(key: str) -> dict:
    spec = REGISTRY[key]
    rd = report_dir(key)
    entry = {
        "source": key,
        "title": spec.title,
        "priority": spec.priority,
        "type": spec.type,
        "mode": "wrap_trusted" if spec.wraps_trusted else ("creds" if spec.needs_credentials else "staged"),
        "status": Status.NOT_STARTED,
    }
    vpath = rd / "validation.json"
    mpath = rd / "manifest.json"
    if vpath.exists():
        try:
            v = json.loads(vpath.read_text())
        except Exception:
            v = {}
        rows = v.get("rows", 0)
        entry["rows"] = rows
        entry["columns"] = v.get("columns", 0)
        entry["date_min"] = v.get("date_min")
        entry["date_max"] = v.get("date_max")
        entry["duplicate_key_count"] = v.get("duplicate_key_count")
        bounded_sample = bool(v.get("bounded_sample") or spec.bounded_sample)
        if v.get("status") == Status.IMPLEMENTED_NOT_FETCHED:
            entry["status"] = Status.IMPLEMENTED_NOT_FETCHED
        elif not rows:
            entry["status"] = Status.IMPLEMENTED_NOT_FETCHED if not v.get("error") else Status.FAILED
        elif v.get("passed") and not bounded_sample:
            entry["status"] = Status.READY_FOR_MODELING
        else:
            entry["status"] = Status.FETCHED_NEEDS_REVIEW
    if mpath.exists():
        try:
            entry["curated_path"] = json.loads(mpath.read_text()).get("curated_path")
        except Exception:
            pass
    return entry


def cmd_report(args) -> int:
    entries = [_status_for_source(k) for k in sorted(REGISTRY, key=lambda k: REGISTRY[k].priority)]
    counts: dict[str, int] = {s: 0 for s in Status.ALL}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    matrix = {"generated_by": "ops/data_fetch.py report", "counts": counts, "sources": entries}
    _REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = _REPORTS_ROOT / "fetch_status_matrix.json"
    out.write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
    print(f"{'prio':>4}  {'source':24} {'status':24} rows")
    print("-" * 70)
    for e in entries:
        print(f"{e['priority']:>4}  {e['source']:24} {e['status']:24} {e.get('rows', 0)}")
    print("\ncounts:", json.dumps(counts))
    print("wrote", out)
    return 0


def cmd_fetch_all(args) -> int:
    keys = _sources_by_priority(args.priority)
    print(f"[fetch-all] priority={args.priority} -> {keys}")
    results = []
    for k in keys:
        spec = REGISTRY[k]
        try:
            res = spec.build_adapter().fetch(args.start, args.end, resume=True,
                                             limit_chunks=args.limit_chunks)
            results.append((k, res.status, res.rows))
            print(f"  [{res.status}] {k}: {res.rows} rows — {res.message}")
        except Exception as exc:  # noqa: BLE001 — one source failing must not crash fetch-all
            results.append((k, Status.FAILED, 0))
            print(f"  [FAILED] {k}: {type(exc).__name__}: {exc}")
    cmd_report(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="data_fetch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("inventory").set_defaults(func=cmd_inventory)

    def _src(sp, with_dates=True):
        sp.add_argument("--source", required=True)
        if with_dates:
            sp.add_argument("--start", default=None)
            sp.add_argument("--end", default=None)

    _src(d := sub.add_parser("discover")); d.set_defaults(func=cmd_discover)
    _src(dr := sub.add_parser("dry-run")); dr.set_defaults(func=cmd_dry_run)
    f = sub.add_parser("fetch"); _src(f)
    f.add_argument("--no-resume", action="store_true")
    f.add_argument("--limit-chunks", type=int, default=None)
    f.set_defaults(func=cmd_fetch)
    v = sub.add_parser("validate"); v.add_argument("--source", required=True); v.set_defaults(func=cmd_validate)

    r = sub.add_parser("report")
    r.add_argument("--start", default=None); r.add_argument("--end", default=None)
    r.add_argument("--limit-chunks", type=int, default=None)
    r.set_defaults(func=cmd_report)

    fa = sub.add_parser("fetch-all")
    fa.add_argument("--priority", default="high", choices=list(PRIORITY_BANDS))
    fa.add_argument("--max-workers", type=int, default=1,
                    help="reserved; fetch-all runs sequentially for rate-limit safety")
    fa.add_argument("--start", default=None); fa.add_argument("--end", default=None)
    fa.add_argument("--limit-chunks", type=int, default=None)
    fa.set_defaults(func=cmd_fetch_all)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
