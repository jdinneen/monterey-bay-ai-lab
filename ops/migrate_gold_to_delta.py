#!/usr/bin/env python3
"""Reference cutover tool: gold Parquet -> Delta Lake (delta-rs), non-destructively.

STATUS: DRAFT / reference implementation. It requires the ``deltalake`` package,
which is intentionally NOT yet a project dependency (see
``docs/delta_cutover_plan.md`` Phase 0). Until that dependency is approved this
module is import-safe and ``--help``-able but its migrate/validate paths will
exit with a clear install instruction.

Why Delta Lake here, and why it is NOT the forbidden product
------------------------------------------------------------
"Delta Lake" is an **open table format / open protocol** (Linux Foundation,
Apache-2.0). ``delta-rs`` (PyPI: ``deltalake``) is a pure-Rust/Python writer with
**no JVM, no Spark, and no Databricks runtime or account**. Adopting it is fully
consistent with the binding directive "build LIKE a lakehouse, do NOT use the
Databricks product": we gain ACID commits, snapshot isolation, time-travel, and
schema evolution on local/portable compute, and the tables stay readable later by
any Spark/Trino/managed runtime without lock-in. Do not confuse this open format
with the managed product; this tool adds no ``DATABRICKS_*`` env, no Unity
Catalog, no bundle.

What it does
------------
* Writes each gold table to ``lakehouse/gold_delta/<table>`` as a Delta table,
  leaving the existing ``lakehouse/gold/<table>`` Parquet untouched (parallel,
  reversible).
* With ``--replay-history`` (default), replays the content-addressed snapshot log
  produced by ``ops/export_gold_versioned.py`` so each historical snapshot becomes
  a Delta **version** (commit) in order -> the existing time-travel lineage carries
  over 1:1 into ``_delta_log``. Without it, only the current source state is
  written as a single commit.
* Validates parity (row count + canonical content hash) between the latest Delta
  version and the source Parquet, reusing the exact hash from the snapshot layer.

This is the Phase 2 engine of the cutover plan; readers are switched in Phase 3.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

# Allow both `python -m ops.migrate_gold_to_delta` and direct
# `python ops/migrate_gold_to_delta.py` by ensuring the repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the snapshot-layer helpers so hashing / table map / roots stay identical.
from ops.export_gold_versioned import (  # noqa: E402
    _TABLE_SOURCES,
    _content_hash,
    _read_log,
    gold_root,
    read_snapshot,
    source_path,
    versioned_dir,
)

_INSTALL_HINT = (
    "deltalake is not installed. This is a Phase-0 gated dependency. To run the "
    "cutover install it first (pure Rust/Python wheel, no JVM):\n"
    "    python -m pip install 'deltalake>=1.0'   # or: pip install -e .[delta]\n"
    "See docs/delta_cutover_plan.md."
)


def _require_deltalake():
    """Lazy import so the module stays import-safe before the dep is approved."""
    try:
        import deltalake  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without the dep
        raise SystemExit(f"[migrate_gold_to_delta] {_INSTALL_HINT}\n({exc})")
    return deltalake


def delta_root(project_root: Path) -> Path:
    """Sibling Delta warehouse next to the existing gold Parquet (never mixed)."""
    return gold_root(project_root).parent / "gold_delta"


def delta_table_path(project_root: Path, table: str) -> Path:
    return delta_root(project_root) / table


def _coerce_for_arrow(df: pd.DataFrame) -> "pd.DataFrame":
    """Make pandas nullable extension dtypes safe for the Arrow/Delta writer.

    The gold frames use pandas nullable Int64/boolean and ``string[python]``
    (e.g. forecast_metrics.n is Int64, drivers_enabled is bool, analyte is
    string). delta-rs accepts an Arrow table; pandas nullable dtypes convert
    cleanly via ``pyarrow`` with nulls preserved, but we normalize object columns
    that hold mixed None/str to a stable string dtype to avoid Arrow inferring a
    null-typed column on all-None edge cases.
    """
    import pyarrow as pa

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            # All-null object columns confuse Arrow type inference; pin to string.
            if out[col].notna().sum() == 0:
                out[col] = out[col].astype("string")
    # Round-trip through Arrow to surface any unsupported dtype early, loudly.
    pa.Table.from_pandas(out, preserve_index=False)
    return out


def _write_commit(deltalake, path: Path, df: pd.DataFrame, *, metadata: dict) -> None:
    """One Delta commit (overwrite), carrying snapshot provenance as commit meta."""
    # deltalake>=1.0 dropped the `engine` kwarg (rust is the only engine). All
    # custom_metadata values must be strings.
    deltalake.write_deltalake(
        str(path),
        _coerce_for_arrow(df),
        mode="overwrite",
        schema_mode="overwrite",  # allow schema evolution across historical snaps
        commit_properties=deltalake.CommitProperties(
            custom_metadata={k: str(v) for k, v in metadata.items()}
        ),
    )


def migrate_table(
    project_root: Path,
    table: str,
    *,
    replay_history: bool = True,
) -> dict | None:
    """Materialize ``table`` as a Delta table. Returns a summary dict or None."""
    project_root = Path(project_root)
    # Cheap skips (unknown / missing table) happen before we require the optional
    # dependency, so a bad table name never turns into an install error.
    src = source_path(project_root, table)
    if src is None:
        print(f"[warn] unknown table '{table}', skipping", file=sys.stderr)
        return None
    if not src.exists():
        print(f"[warn] source missing for '{table}' ({src}), skipping", file=sys.stderr)
        return None

    deltalake = _require_deltalake()
    path = delta_table_path(project_root, table)
    path.mkdir(parents=True, exist_ok=True)

    commits = 0
    last_sid = None
    if replay_history:
        log = _read_log(versioned_dir(project_root, table))
        for entry in log:
            sid = entry["snapshot_id"]
            df = read_snapshot(project_root, table, sid)
            if df is None:
                continue
            _write_commit(
                deltalake, path, df,
                metadata={
                    "source_snapshot_id": sid,
                    "snapshot_created_at_utc": str(entry.get("created_at_utc")),
                    "origin": "export_gold_versioned",
                },
            )
            commits += 1
            last_sid = sid
        # Reconcile the head with the CURRENT source Parquet so the latest Delta
        # version always matches the live gold table, even when the snapshot log
        # lagged behind a concurrent rebuild (this is a multi-agent shared tree).
        if commits:
            src_df = pd.read_parquet(src)
            if _content_hash(src_df) != last_sid:
                _write_commit(
                    deltalake, path, src_df,
                    metadata={"origin": "source_reconcile", "source_snapshot_id": _content_hash(src_df)},
                )
                commits += 1

    if commits == 0:
        # No snapshot history (or replay disabled): single commit of current state.
        df = pd.read_parquet(src)
        _write_commit(
            deltalake, path, df,
            metadata={"source_snapshot_id": _content_hash(df), "origin": "source_parquet"},
        )
        commits = 1

    summary = validate_parity(project_root, table)
    summary["commits"] = commits
    summary["delta_versions"] = _latest_version(project_root, table) + 1
    return summary


_SHADOW_FLAG = "MBAL_GOLD_SHADOW_DELTA"


def shadow_dual_write(table: str, df: pd.DataFrame, gold_dir: str | Path) -> str | None:
    """Phase-1 shadow dual-write: best-effort Delta mirror of a gold aggregate.

    Call this right after a gold aggregate Parquet write. It is:
    * **Opt-in** - a no-op unless ``MBAL_GOLD_SHADOW_DELTA=1`` (default off, so
      existing pipelines are unchanged).
    * **Best-effort and NON-FATAL** - if ``deltalake`` is absent or any write
      error occurs it logs to stderr and returns ``None``; it must never break the
      primary gold write.

    The Delta table lands in the ``gold_delta/`` sibling of ``gold_dir`` (the gold
    root, e.g. ``lakehouse/gold``), table ``<table>``. Returns the Delta path on
    success, else ``None``.
    """
    if os.environ.get(_SHADOW_FLAG) != "1":
        return None
    try:
        import deltalake  # type: ignore
    except Exception:
        print(
            f"[shadow_delta] {_SHADOW_FLAG}=1 but deltalake not installed; "
            f"skipping shadow write for {table}.",
            file=sys.stderr,
        )
        return None
    try:
        path = Path(gold_dir).parent / "gold_delta" / table
        path.mkdir(parents=True, exist_ok=True)
        _write_commit(deltalake, path, df, metadata={"origin": "shadow_dual_write", "table": table})
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive; shadow must not break runs
        print(f"[shadow_delta] non-fatal shadow write failed for {table}: {exc}", file=sys.stderr)
        return None


def read_delta_table(
    project_root: Path, table: str, version: int | None = None
) -> pd.DataFrame:
    """Read shim mirroring read_forecast_metrics semantics, with time-travel.

    ``version=None`` -> latest. This is the function Phase 3 wires behind the
    ``MBAL_GOLD_FORMAT=delta`` feature flag.
    """
    deltalake = _require_deltalake()
    dt = deltalake.DeltaTable(str(delta_table_path(project_root, table)))
    if version is not None:
        dt.load_as_version(version)
    return dt.to_pandas()


def _latest_version(project_root: Path, table: str) -> int:
    deltalake = _require_deltalake()
    dt = deltalake.DeltaTable(str(delta_table_path(project_root, table)))
    return dt.version()


def _parity_signature(df: pd.DataFrame) -> str:
    """Value-level, dtype-tolerant signature for cross-format parity.

    Unlike the snapshot layer's ``_content_hash`` (which folds the pandas dtype
    string into the hash), this normalizes before hashing so a benign dtype change
    across the Parquet->Delta round-trip (e.g. nullable ``Int64`` -> ``int64``,
    ``boolean`` -> ``bool``, ``string[python]`` -> object) is NOT flagged as drift.
    Numeric/bool columns are coerced to float64; everything else to nullable
    string; rows are then canonically ordered so physical row order is irrelevant.
    Real value differences still change the signature.
    """
    norm = pd.DataFrame(index=range(len(df)))
    for col in sorted(df.columns):
        s = df[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            norm[col] = s.astype("float64")
        else:
            norm[col] = s.astype("string")
    if len(norm) > 1:
        order = norm.astype("string").fillna("").agg("".join, axis=1).argsort(kind="stable")
        norm = norm.iloc[order].reset_index(drop=True)
    return hashlib.sha256(norm.to_csv(index=False).encode("utf-8")).hexdigest()


def validate_parity(project_root: Path, table: str) -> dict:
    """Compare the latest Delta version against the source Parquet.

    Parity is a value-level, dtype-tolerant signature (see ``_parity_signature``),
    so a benign row reorder or nullable-dtype normalization during the Delta write
    does not register as drift. Raises AssertionError on mismatch so the cutover
    fails loud rather than silently diverging the gold table.
    """
    project_root = Path(project_root)
    dpath = delta_table_path(project_root, table)
    if not (dpath / "_delta_log").exists():
        # Friendly, caught-by-main failure instead of a raw delta-rs traceback when
        # validating a table that was never migrated.
        raise AssertionError(
            f"[migrate_gold_to_delta] no Delta table for '{table}' at {dpath}; "
            f"run the migration first (omit --validate-only)."
        )
    src = pd.read_parquet(source_path(project_root, table))
    got = read_delta_table(project_root, table)
    src_hash, got_hash = _parity_signature(src), _parity_signature(got)
    ok = src_hash == got_hash and len(src) == len(got)
    result = {
        "table": table,
        "parity_ok": ok,
        "source_rows": int(len(src)),
        "delta_rows": int(len(got)),
        "source_hash": src_hash[:12],
        "delta_hash": got_hash[:12],
    }
    assert ok, f"[migrate_gold_to_delta] PARITY FAILURE for {table}: {result}"
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".", help="Repo root (default: cwd).")
    p.add_argument(
        "--tables",
        default=",".join(_TABLE_SOURCES),
        help="Comma-separated logical table names (default: all known gold tables).",
    )
    p.add_argument(
        "--no-replay-history",
        dest="replay_history",
        action="store_false",
        help="Write only the current source state as a single Delta commit.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip writing; only check Delta-vs-Parquet parity for existing tables.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    rc = 0
    for table in tables:
        try:
            if args.validate_only:
                summary = validate_parity(Path(args.project_root), table)
            else:
                summary = migrate_table(
                    Path(args.project_root), table, replay_history=args.replay_history
                )
            if summary is None:
                continue
            print(
                f"[ok] {table}: parity={summary['parity_ok']} "
                f"rows={summary.get('delta_rows')} "
                f"versions={summary.get('delta_versions', 'n/a')}"
            )
        except AssertionError as exc:
            print(str(exc), file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
