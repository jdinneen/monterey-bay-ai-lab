"""Tests for the fetcher cache guard (mbal_history/fetch_cache_guard)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ops import fetch_cache_guard as g  # noqa: E402

PATTERNS = ["mur_sst_{key}.parquet", "mur_{key}.parquet"]


def test_resolve_prefers_canonical_then_falls_back(tmp_path):
    (tmp_path / "mur_2010.parquet").write_text("old")          # only the non-canonical name
    assert g.resolve_cache(tmp_path, 2010, PATTERNS).name == "mur_2010.parquet"
    (tmp_path / "mur_sst_2010.parquet").write_text("new")      # canonical now present
    assert g.resolve_cache(tmp_path, 2010, PATTERNS).name == "mur_sst_2010.parquet"
    assert g.resolve_cache(tmp_path, 1999, PATTERNS) is None   # nothing for this key


def test_audit_flags_orphan_and_duplicate(tmp_path):
    (tmp_path / "mur_2008.parquet").write_text("data")         # orphan: canonical missing
    (tmp_path / "mur_sst_2009.parquet").write_text("x")        # canonical-only: fine
    (tmp_path / "mur_2009.parquet").write_text("x")            # -> 2009 becomes a duplicate
    flags = g.audit_dir(tmp_path, PATTERNS)
    joined = "\n".join(flags)
    assert any("ORPHANED" in f and "2008" in f for f in flags)
    assert any("DUPLICATE" in f and "2009" in f for f in flags)
    assert "2008" in joined and "2009" in joined


def test_audit_clean_dir_returns_no_flags(tmp_path):
    (tmp_path / "mur_sst_2002.parquet").write_text("x")
    (tmp_path / "mur_sst_2003.parquet").write_text("x")
    assert g.audit_dir(tmp_path, PATTERNS) == []
