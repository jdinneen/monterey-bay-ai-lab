"""Coverage validator for the data-fetch framework.

Answers the question "WHY do we have gaps?" for each source by comparing what we
actually fetched (the curated parquet) against the dataset's NATIVE availability,
and classifying every gap as one of:

  * UNFETCHED HISTORY / RECENT  -> data exists upstream but we didn't request it
                                   (RECOVERABLE: re-fetch the missing range)
  * ENDPOINT-LIMITED            -> the endpoint itself only serves a short/rolling
                                   window; the archive lives elsewhere (NOT here)
  * INTRINSIC SPARSITY          -> within the fetched span, many days simply have no
                                   data (e.g. cloud cover for ocean color) -> not
                                   fully recoverable from any source

This exists so a bounded `--start 2020` fetch can never again silently masquerade as
"complete" when the dataset goes back to 2002. Run it after any fetch.

    python -m ops.data_fetch.coverage_audit            # all sources
    python -m ops.data_fetch.coverage_audit --source mur_sst
    python -m ops.data_fetch.coverage_audit --json
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import urllib.request
from datetime import date
from typing import Optional

import pandas as pd

from .core import external_curated_dir
from .registry import REGISTRY, get_spec

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap"
DENSITY_FLOOR = 80.0       # below this %, flag intrinsic sparsity
ROLLING_WINDOW_DAYS = 200  # native span <= this => endpoint-limited (rolling)


def _erddap_dataset_id(spec) -> Optional[str]:
    """Best-effort discovery of an ERDDAP griddap dataset id from the adapter."""
    try:
        adapter = spec.build_adapter()
    except Exception:
        return None
    for attr in ("DATASET_ID", "DATASET"):
        val = getattr(adapter, attr, None) or getattr(type(adapter), attr, None)
        if val:
            return val
    try:  # module-level DATASET (e.g. hf_radar)
        mod = importlib.import_module(type(adapter).__module__)
        return getattr(mod, "DATASET", None)
    except Exception:
        return None


def _native_window(dataset_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Native time_coverage_start/end from an ERDDAP griddap .das (None if unknown)."""
    if not dataset_id:
        return None, None
    try:
        das = urllib.request.urlopen(f"{ERDDAP_BASE}/griddap/{dataset_id}.das", timeout=40)
        text = das.read().decode("utf-8", "replace")
    except Exception:
        return None, None
    s = re.search(r'time_coverage_start\s+"(.*?)"', text)
    e = re.search(r'time_coverage_end\s+"(.*?)"', text)
    return (s.group(1)[:10] if s else None, e.group(1)[:10] if e else None)


def audit_source(key: str, probe_native: bool = True) -> dict:
    spec = get_spec(key)
    curated = external_curated_dir(key) / f"{key}.parquet"
    rep: dict = {"source": key, "curated_exists": curated.exists()}
    if not curated.exists():
        rep["gaps"] = ["NO CURATED FILE: source has never been fetched/consolidated"]
        return rep

    df = pd.read_parquet(curated)
    rep["rows"] = int(len(df))
    dc = spec.date_column if (spec.date_column and spec.date_column in df.columns) else (
        "time" if "time" in df.columns else None)
    if not dc or df.empty:
        rep["gaps"] = ["EMPTY or NO DATE COLUMN: cannot assess temporal coverage"]
        return rep

    t = pd.to_datetime(df[dc], utc=True, errors="coerce").dropna()
    if t.empty:
        rep["gaps"] = ["NO PARSEABLE DATES"]
        return rep
    rep["fetched_start"], rep["fetched_end"] = str(t.min().date()), str(t.max().date())
    distinct_days = int(t.dt.normalize().nunique())
    span_days = int((t.max().normalize() - t.min().normalize()).days) + 1
    rep["distinct_days"] = distinct_days
    rep["span_days"] = span_days
    rep["density_pct"] = round(100 * distinct_days / span_days, 1) if span_days else None

    gaps: list[str] = []
    if probe_native:
        did = _erddap_dataset_id(spec)
        nat_s, nat_e = _native_window(did)
        rep["dataset_id"] = did
        rep["native_start"], rep["native_end"] = nat_s, nat_e
        if nat_s and nat_e:
            nat_span = (date.fromisoformat(nat_e) - date.fromisoformat(nat_s)).days
            if nat_span <= ROLLING_WINDOW_DAYS:
                gaps.append(
                    f"ENDPOINT-LIMITED: this endpoint only serves ~{nat_span} days "
                    f"({nat_s}..{nat_e}); the historical archive is NOT here -> needs a different source")
            else:
                if rep["fetched_start"] > nat_s:
                    gaps.append(
                        f"UNFETCHED HISTORY: data available back to {nat_s} but earliest fetched "
                        f"is {rep['fetched_start']} -> RECOVERABLE (re-fetch --start {nat_s[:4]})")
                if rep["fetched_end"] < nat_e:
                    gaps.append(
                        f"UNFETCHED RECENT: data available through {nat_e} but latest fetched "
                        f"is {rep['fetched_end']} -> RECOVERABLE")
        elif did:
            gaps.append(f"NATIVE WINDOW UNKNOWN: could not probe {did} (.das unreachable)")

    if rep.get("density_pct") is not None and rep["density_pct"] < DENSITY_FLOOR:
        gaps.append(
            f"INTRINSIC SPARSITY: only {rep['density_pct']}% of days in the fetched span have data "
            f"({distinct_days}/{span_days}) -> source-limited (e.g. cloud cover), NOT fully recoverable")

    rep["gaps"] = gaps or ["OK: fetched coverage matches native availability and is dense"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Explain coverage gaps per data-fetch source.")
    ap.add_argument("--source", help="audit one source key (default: all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-native", action="store_true", help="skip the network native-window probe")
    args = ap.parse_args(argv)

    keys = [args.source] if args.source else list(REGISTRY)
    reports = []
    for k in keys:
        try:
            reports.append(audit_source(k, probe_native=not args.no_native))
        except Exception as exc:  # noqa: BLE001
            reports.append({"source": k, "error": str(exc)})

    if args.json:
        print(json.dumps(reports, indent=2))
        return 0

    for r in reports:
        print(f"\n=== {r['source']} ===")
        if r.get("error"):
            print(f"  error: {r['error']}")
            continue
        if not r.get("curated_exists"):
            print(f"  - {r['gaps'][0]}")
            continue
        print(f"  fetched {r.get('fetched_start')} -> {r.get('fetched_end')} | "
              f"{r.get('rows'):,} rows | {r.get('distinct_days')} days | density {r.get('density_pct')}%")
        if r.get("native_start"):
            print(f"  native  {r['native_start']} -> {r['native_end']}  ({r.get('dataset_id')})")
        for g in r.get("gaps", []):
            print(f"  - {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
