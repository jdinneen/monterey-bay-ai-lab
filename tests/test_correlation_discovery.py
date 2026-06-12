"""Tests for the cross-source correlation discovery engine (research/data_lab/correlation_discovery.py).

Hermetic: synthesizes two tiny curated sources with a KNOWN correlation, so we assert the engine
actually recovers it, classifies cross-source/cross-quantity correctly, and writes ASCII-safe output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "data_lab"))
import correlation_discovery as C  # noqa: E402


def _make_sources(curated: Path):
    """src_a::wind and src_b::discharge are correlated by construction; src_c::noise is independent."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", "2020-12-31", freq="D", tz="UTC")
    base = rng.normal(size=len(days)) + np.sin(np.arange(len(days)) / 30.0)
    for d, frame in {
        "src_a": pd.DataFrame({"time": days, "WSPD": base + rng.normal(0, 0.1, len(days))}),
        "src_b": pd.DataFrame({"time": days, "result": base * 2 + rng.normal(0, 0.2, len(days))}),
        "src_c": pd.DataFrame({"time": days, "noise": rng.normal(size=len(days))}),
    }.items():
        (curated / d).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(curated / d / f"{d}.parquet", index=False)
    return [
        {"name": "a", "dir": "src_a", "time": "time", "kind": "wide", "vars": {"WSPD": "wind"}},
        {"name": "b", "dir": "src_b", "time": "time", "kind": "wide", "vars": {"result": "discharge"}},
        {"name": "c", "dir": "src_c", "time": "time", "kind": "wide", "vars": {"noise": "noise"}},
    ]


def test_recovers_known_correlation_and_classifies(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CURATED", tmp_path / "curated")
    monkeypatch.setattr(C, "CATALOG", _make_sources(tmp_path / "curated"))
    out = C.run(output_dir=tmp_path / "out", min_overlap=180, top=20, min_abs_r=0.3)

    assert out["matrix"]["n_variables"] == 3
    disc = out["top_cross_quantity_positive"]
    # the constructed wind~discharge correlation must be found, strong, cross-source, cross-quantity
    hit = [p for p in disc if {p["a"], p["b"]} == {"a::wind", "b::discharge"}]
    assert hit and hit[0]["pearson"] > 0.9 and hit[0]["cross_source"]
    # the independent noise series must NOT show up as a strong positive with wind/discharge
    assert not any("noise" in p["a"] or "noise" in p["b"] for p in disc)


def test_same_quantity_goes_to_consistency_not_discovery(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    rng = np.random.default_rng(1)
    days = pd.date_range("2015-01-01", "2020-12-31", freq="D", tz="UTC")
    base = rng.normal(size=len(days))
    for d, col in [("src_a", "WSPD"), ("src_b", "wind_speed_10m")]:
        (curated / d).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time": days, col: base + rng.normal(0, 0.1, len(days))}).to_parquet(
            curated / d / f"{d}.parquet", index=False)
    catalog = [
        {"name": "a", "dir": "src_a", "time": "time", "kind": "wide", "vars": {"WSPD": "wspd"}},
        {"name": "b", "dir": "src_b", "time": "time", "kind": "wide", "vars": {"wind_speed_10m": "wind"}},
    ]
    monkeypatch.setattr(C, "CURATED", curated)
    monkeypatch.setattr(C, "CATALOG", catalog)
    out = C.run(output_dir=tmp_path / "out", min_overlap=180, top=20, min_abs_r=0.3)
    # wspd and wind both canonicalize to "wind" -> data-consistency, NOT a cross-quantity discovery
    assert any({p["a"], p["b"]} == {"a::wspd", "b::wind"} for p in out["data_consistency_same_quantity"])
    assert not any({p["a"], p["b"]} == {"a::wspd", "b::wind"} for p in out["top_cross_quantity_positive"])


def test_cluster_groups_correlated_variables(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    days = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    base = rng.normal(size=len(days))
    other = rng.normal(size=len(days))
    wide = pd.DataFrame({
        "a::t": base, "b::t": base + rng.normal(0, 0.1, len(days)),          # correlated family
        "c::w": other, "d::w": other + rng.normal(0, 0.1, len(days)),        # a second family
    }, index=days)
    pear = wide.corr()
    clusters = C.cluster_variables(pear, abs_r_threshold=0.6)
    flat = {frozenset(g) for g in clusters}
    assert {"a::t", "b::t"} in flat and {"c::w", "d::w"} in flat


def test_artifacts_are_ascii_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CURATED", tmp_path / "curated")
    monkeypatch.setattr(C, "CATALOG", _make_sources(tmp_path / "curated"))
    C.run(output_dir=tmp_path / "out", min_overlap=180, top=20, min_abs_r=0.3)
    (tmp_path / "out" / "correlation_discovery.md").read_text(encoding="utf-8").encode("ascii")
    json.loads((tmp_path / "out" / "correlation_discovery.json").read_text(encoding="utf-8"))
