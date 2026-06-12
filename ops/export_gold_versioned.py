#!/usr/bin/env python3
"""Additive, content-addressed snapshot layer over the gold Parquet tables.

This is an *interim* step toward a real ACID table format (Delta-rs / Apache
Iceberg) which we cannot adopt yet because the dependency is not approved and we
may not hit the network. Until then the gold tables are plain partitioned
Parquet: no time-travel, no snapshot isolation, just a hand-rolled aggregate file
plus a cooperative ``ops/agent_lock.py`` papering over concurrency.

What this module adds, in pure Python + pyarrow/pandas and *without touching the
existing gold files*, is Iceberg-style table semantics:

* Immutable snapshots written to
  ``lakehouse/gold/_versioned/<table>/snapshots/<snapshot_id>/data.parquet``.
* ``snapshot_id`` is content-addressed (sha256 over the canonicalized table
  content), so re-exporting identical data is **idempotent** - same id, no new
  snapshot, no new log entry.
* An append-only commit log
  ``lakehouse/gold/_versioned/<table>/_log.json`` recording the lineage of every
  snapshot ``{snapshot_id, parent_snapshot_id, row_count, n_columns,
  created_at_utc, schema_fingerprint}``.
* ``read_snapshot`` for time-travel (latest, or any historical id).

Read-only over ``lakehouse/gold/`` proper. The only thing it writes lives under
the new ``_versioned/`` sibling directory. See
``docs/versioned_gold_migration_plan.md`` for the upgrade path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Map a logical table name to its source-of-truth Parquet file (relative to the
# gold root). These are the hand-rolled aggregate files the rest of the pipeline
# already treats as the canonical table; the per-run partitions are derived.
_TABLE_SOURCES: dict[str, str] = {
    "forecast_metrics": "forecast_metrics/metrics.parquet",
    "promotion_matrix": "promotion_matrix/promotion_matrix.parquet",
    "forecast_runs": "forecast_runs/run_manifest.parquet",
}

_VERSIONED_DIRNAME = "_versioned"
_LOG_FILENAME = "_log.json"
_DATA_FILENAME = "data.parquet"


def gold_root(project_root: Path) -> Path:
    """Gold root, honoring ``MBAL_LAKEHOUSE_DIR`` like the rest of the repo."""
    lake = Path(os.environ.get("MBAL_LAKEHOUSE_DIR", Path(project_root) / "lakehouse"))
    return lake / "gold"


def source_path(project_root: Path, table: str) -> Path | None:
    """Absolute path to a table's source Parquet, or ``None`` if unknown table."""
    rel = _TABLE_SOURCES.get(table)
    if rel is None:
        return None
    return gold_root(project_root) / rel


def versioned_dir(project_root: Path, table: str) -> Path:
    return gold_root(project_root) / _VERSIONED_DIRNAME / table


def _canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Order-independent canonical form so content addressing is stable.

    Row order and column order in the source Parquet are incidental (a re-run of
    the aggregate may reorder rows), but the *table content* is what we version.
    We sort columns, then sort rows by their stringified values, so two frames
    with identical content hash identically regardless of physical ordering.
    """
    out = df.reindex(sorted(df.columns), axis=1)
    if len(out) > 1:
        order = out.astype(str).agg("".join, axis=1).argsort(kind="stable")
        out = out.iloc[order].reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _content_hash(df: pd.DataFrame) -> str:
    """sha256 over the canonicalized table content (stdlib hashlib only)."""
    canon = _canonical_frame(df)
    h = hashlib.sha256()
    # Hash the schema first so a pure schema change yields a new id even when the
    # (empty) row content is identical.
    for col in canon.columns:
        h.update(str(col).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(canon[col].dtype).encode("utf-8"))
        h.update(b"\x01")
    h.update(b"\x02")
    # Then the row content, via a deterministic CSV-ish serialization. We avoid
    # raw Parquet bytes because those embed non-content metadata (timestamps,
    # writer version, compression framing) that would break idempotency.
    h.update(canon.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def _schema_fingerprint(df: pd.DataFrame) -> str:
    """Short, stable fingerprint of (column, dtype) pairs in canonical order."""
    parts = [f"{c}:{df[c].dtype}" for c in sorted(df.columns)]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _read_log(table_dir: Path) -> list[dict]:
    log_path = table_dir / _LOG_FILENAME
    if not log_path.exists():
        return []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _write_log(table_dir: Path, log: list[dict]) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / _LOG_FILENAME).write_text(
        json.dumps(log, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _resolve_created_at(as_of: str | None, src: Path | None) -> str:
    """Snapshot timestamp.

    Deliberately *not* part of the content hash, so it never breaks idempotency.
    Preference order: explicit ``--as-of`` -> source-file mtime -> now(). The
    first two keep re-runs fully reproducible.
    """
    if as_of:
        return as_of
    if src is not None and src.exists():
        return datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat()
    return datetime.now(tz=timezone.utc).isoformat()


def export_snapshot(
    project_root: Path,
    table: str,
    *,
    as_of: str | None = None,
    df: pd.DataFrame | None = None,
) -> dict | None:
    """Write an immutable, content-addressed snapshot of ``table``.

    Idempotent: if the latest snapshot already has the same content hash, no new
    snapshot directory and no new log entry are created; the existing log entry
    is returned. Returns the (new or pre-existing) log entry, or ``None`` if the
    table is unknown / its source file is missing.
    """
    project_root = Path(project_root)
    if df is None:
        src = source_path(project_root, table)
        if src is None or not src.exists():
            return None
        df = pd.read_parquet(src)
    else:
        src = source_path(project_root, table)

    snapshot_id = _content_hash(df)
    table_dir = versioned_dir(project_root, table)
    log = _read_log(table_dir)

    # Idempotency: identical content as the current head -> no-op.
    if log and log[-1].get("snapshot_id") == snapshot_id:
        return log[-1]
    # Also short-circuit if this exact snapshot already exists anywhere in the
    # log (e.g. a revert to a prior content state re-points head but we still do
    # not duplicate the immutable data file).
    existing = next((e for e in log if e.get("snapshot_id") == snapshot_id), None)

    parent = log[-1]["snapshot_id"] if log else None
    snap_dir = table_dir / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    data_path = snap_dir / _DATA_FILENAME
    if not data_path.exists():
        df.to_parquet(data_path, index=False)

    entry = {
        "snapshot_id": snapshot_id,
        "parent_snapshot_id": parent,
        "row_count": int(len(df)),
        "n_columns": int(df.shape[1]),
        "created_at_utc": _resolve_created_at(as_of, src),
        "schema_fingerprint": _schema_fingerprint(df),
    }
    if existing is not None:
        # Re-pointing head to a previously-seen content state: append a fresh log
        # entry capturing the new lineage edge but reuse the immutable data file.
        entry["created_at_utc"] = _resolve_created_at(as_of, src)
    log.append(entry)
    _write_log(table_dir, log)
    return entry


def read_snapshot(
    project_root: Path, table: str, snapshot_id: str | None = None
) -> pd.DataFrame | None:
    """Time-travel read. ``snapshot_id=None`` returns the latest snapshot.

    Returns ``None`` when there is no log / no matching snapshot.
    """
    project_root = Path(project_root)
    table_dir = versioned_dir(project_root, table)
    log = _read_log(table_dir)
    if not log:
        return None
    if snapshot_id is None:
        snapshot_id = log[-1]["snapshot_id"]
    elif not any(e.get("snapshot_id") == snapshot_id for e in log):
        return None
    data_path = table_dir / "snapshots" / snapshot_id / _DATA_FILENAME
    if not data_path.exists():
        return None
    return pd.read_parquet(data_path)


def export_tables(
    project_root: Path, tables: list[str], *, as_of: str | None = None
) -> dict[str, dict | None]:
    """Export each requested table; missing tables are skipped with a warning."""
    results: dict[str, dict | None] = {}
    for table in tables:
        src = source_path(project_root, table)
        if src is None:
            print(f"[warn] unknown table '{table}', skipping", file=sys.stderr)
            results[table] = None
            continue
        if not src.exists():
            print(f"[warn] source missing for '{table}' ({src}), skipping", file=sys.stderr)
            results[table] = None
            continue
        entry = export_snapshot(project_root, table, as_of=as_of)
        results[table] = entry
        if entry is not None:
            print(
                f"[ok] {table}: snapshot {entry['snapshot_id'][:12]} "
                f"({entry['row_count']} rows, {entry['n_columns']} cols)"
            )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".", help="Repo root (default: cwd).")
    p.add_argument(
        "--tables",
        default="forecast_metrics,promotion_matrix",
        help="Comma-separated logical table names.",
    )
    p.add_argument(
        "--as-of",
        default=None,
        help="ISO-8601 created_at_utc to stamp on new snapshots (idempotency-safe).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    export_tables(Path(args.project_root), tables, as_of=args.as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
