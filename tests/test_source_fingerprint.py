#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbal_neural_forecast import file_fingerprint as nf_fingerprint  # noqa: E402
from mbal_split_contracts import file_fingerprint as sc_fingerprint  # noqa: E402


FINGERPRINTS = [nf_fingerprint, sc_fingerprint]


@pytest.mark.parametrize("fingerprint", FINGERPRINTS)
def test_content_sha256_present_and_64_hex(tmp_path, fingerprint):
    f = tmp_path / "data.parquet"
    f.write_bytes(b"hello mbal")
    fp = fingerprint(f)

    assert "content_sha256" in fp
    digest = fp["content_sha256"]
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)
    # additive: legacy keys still present
    assert "size_bytes" in fp
    assert "mtime_utc" in fp


@pytest.mark.parametrize("fingerprint", FINGERPRINTS)
def test_content_sha256_stable_across_mtime_change(tmp_path, fingerprint):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    payload = b"byte-identical content"
    a.write_bytes(payload)
    b.write_bytes(payload)

    # Force a different mtime on b so only the timestamp differs.
    os.utime(b, (1_000_000, 1_000_000))

    fp_a = fingerprint(a)
    fp_b = fingerprint(b)

    assert fp_a["mtime_utc"] != fp_b["mtime_utc"]
    assert fp_a["content_sha256"] == fp_b["content_sha256"]


@pytest.mark.parametrize("fingerprint", FINGERPRINTS)
def test_content_sha256_changes_with_content(tmp_path, fingerprint):
    f = tmp_path / "data.parquet"
    f.write_bytes(b"original content")
    first = fingerprint(f)["content_sha256"]

    f.write_bytes(b"different content")
    second = fingerprint(f)["content_sha256"]

    assert first != second
