#!/usr/bin/env python3
"""Cache guard + auditor for the NOAA / physical-driver fetchers.

Motivation: a fetcher rewrite that changes a cache FILENAME silently orphans the prior
cache, so the next run re-downloads hours of data it already has (this happened: a v2
rewrite changed mur_<year>.parquet -> mur_sst_<year>.parquet). Two small helpers make a
fetcher robust to that and FLAG the problem instead of wasting bandwidth:

  resolve_cache(cache_dir, key, patterns) -> Path | None
      Return an existing cache file for `key` (e.g. a year) under ANY known name pattern
      (canonical first). A fetcher should call this BEFORE fetching: if it returns a path,
      load + skip. This is the "smart" primitive — adopt it in the per-year cache check so
      a future filename change can never orphan good data again.

  audit_dir(cache_dir, patterns) -> list[str]
      Flag problems for a human/agent: ORPHANED caches (data under a non-canonical name,
      canonical missing -> a rerun would re-fetch), DUPLICATE caches (both names present),
      and EMPTY/corrupt parquet. Run standalone: `python ops/fetch_cache_guard.py
      <cache_dir> --patterns "mur_sst_{key}.parquet,mur_{key}.parquet"`.

Read-only over the cache (audit). resolve_cache never writes. No new dependencies.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Default to the case that motivated this (NOAA MUR SST yearly cache).
DEFAULT_PATTERNS = ["mur_sst_{key}.parquet", "mur_{key}.parquet"]


def resolve_cache(cache_dir, key, patterns=DEFAULT_PATTERNS) -> Path | None:
    """First existing cache file for `key` across `patterns` (canonical = patterns[0])."""
    cache_dir = Path(cache_dir)
    for pat in patterns:
        p = cache_dir / pat.format(key=key)
        if p.exists():
            return p
    return None


def _assign_files(cache_dir: Path, patterns: list[str]) -> dict[str, dict[int, Path]]:
    """Assign each cache file to its FIRST-matching pattern (canonical=index 0), so a more
    general pattern (e.g. mur_{key}) cannot greedily steal a canonical file (mur_sst_{key}).
    Returns {key: {pattern_index: path}}."""
    compiled = [re.compile("^" + re.escape(p).replace(r"\{key\}", r"(?P<key>.+)") + "$")
                for p in patterns]
    files: set[Path] = set()
    for pat in patterns:
        files |= set(cache_dir.glob(pat.replace("{key}", "*")))
    out: dict[str, dict[int, Path]] = {}
    for f in sorted(files):
        for i, rx in enumerate(compiled):
            m = rx.match(f.name)
            if m:
                out.setdefault(m.group("key"), {})[i] = f
                break  # first (most specific) match wins
    return out


def _same_content(a: Path, b: Path) -> bool | None:
    """Cheap identity check: equal size, else compare numeric values via pandas."""
    try:
        if a.stat().st_size == b.stat().st_size:
            return True
        import pandas as pd

        da, db = pd.read_parquet(a), pd.read_parquet(b)
        na = da.select_dtypes("number").reset_index(drop=True)
        nb = db.select_dtypes("number").reset_index(drop=True)
        return na.shape == nb.shape and bool((na.fillna(0).values == nb.fillna(0).values).all())
    except Exception:
        return None


def audit_dir(cache_dir, patterns=DEFAULT_PATTERNS) -> list[str]:
    """Return human-readable flags for orphaned / duplicate / empty caches."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return [f"cache dir not found: {cache_dir}"]
    canonical = patterns[0]
    assigned = _assign_files(cache_dir, patterns)
    flags: list[str] = []
    for k, by_idx in sorted(assigned.items()):
        canon_name = canonical.format(key=k)
        if 0 not in by_idx:  # only under a non-canonical name -> a rerun re-fetches
            p = by_idx[min(by_idx)]
            flags.append(
                f"ORPHANED: key={k} has data '{p.name}' under a non-canonical name but the "
                f"canonical '{canon_name}' is MISSING -> a rerun WILL re-fetch it. "
                f"Fix: use resolve_cache() or migrate the file.")
        elif len(by_idx) > 1:  # canonical + a non-canonical copy
            alt = by_idx[max(by_idx)]
            same = _same_content(alt, by_idx[0])
            tag = "identical" if same else ("DIFFERENT -- investigate" if same is False else "unknown")
            flags.append(f"DUPLICATE: key={k} present as both '{alt.name}' and '{by_idx[0].name}' ({tag}).")
        try:
            if 0 in by_idx and by_idx[0].stat().st_size == 0:
                flags.append(f"EMPTY: canonical cache '{canon_name}' is zero bytes.")
        except OSError:
            pass
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_dir")
    ap.add_argument("--patterns", default=",".join(DEFAULT_PATTERNS),
                    help="comma-separated name templates with {key}; first = canonical")
    args = ap.parse_args(argv)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    flags = audit_dir(args.cache_dir, patterns)
    if not flags:
        print(f"[fetch_cache_guard] OK — no cache problems in {args.cache_dir}")
        return 0
    print(f"[fetch_cache_guard] {len(flags)} flag(s) in {args.cache_dir}:")
    for f in flags:
        print("  -", f)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
