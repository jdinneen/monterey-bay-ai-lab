"""Tests for the cross-source consistency gate (research/data_lab/source_consistency_check.py).

The point of this check is to CATCH regressions, so the tests plant the two bug classes it exists
to find and assert it flags them:
  * a Kelvin-for-Celsius offset -> correlation stays ~1 but ABSOLUTE agreement must FAIL;
  * a time-scrambled series -> CO-MOVEMENT (deseasonalized correlation) must FAIL.
A clean, agreeing pair must PASS. (A check that only ever returns PASS hasn't earned its keep.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DL = Path(__file__).resolve().parents[1] / "research" / "data_lab"
sys.path.insert(0, str(_DL))
import correlation_discovery as CD  # noqa: E402
import source_consistency_check as SC  # noqa: E402


def _make_temp_sources(curated: Path):
    """Four 'temperature' sources: two agree, one is +273 (Kelvin bug), one is time-scrambled."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2014-01-01", "2020-12-31", freq="D", tz="UTC")
    doy = days.dayofyear.to_numpy()
    seasonal = 15 + 8 * np.sin(2 * np.pi * doy / 365.25)        # a real seasonal degC signal
    weather = rng.normal(0, 2, len(days))                       # shared day-to-day wiggle
    base = seasonal + weather
    series = {
        "good_a": base + rng.normal(0, 0.3, len(days)),
        "good_b": base + rng.normal(0, 0.3, len(days)),
        "kelvin_c": base + 273.15 + rng.normal(0, 0.3, len(days)),     # unit bug (corr-invisible)
        "scram_d": rng.permutation(base),                              # co-move bug
    }
    for name, vals in series.items():
        (curated / name).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time": days, "tC": vals}).to_parquet(curated / name / f"{name}.parquet", index=False)
    catalog = [{"name": n, "dir": n, "time": "time", "kind": "wide", "vars": {"tC": "t"}}
               for n in series]
    group = [{"q": "temp", "unit": "degC", "abs_bias": 2.5, "abs_mad": 3.0, "absolute": True,
              "members": [f"{n}::t" for n in series]}]
    return catalog, group


def _setup(tmp_path, monkeypatch):
    catalog, group = _make_temp_sources(tmp_path / "curated")
    monkeypatch.setattr(CD, "CURATED", tmp_path / "curated")
    monkeypatch.setattr(CD, "CATALOG", catalog)
    monkeypatch.setattr(SC, "GROUPS", group)


def _pair(out, x, y):
    pairs = out["groups"][0]["pairs"]
    return next(p for p in pairs if {p["a"], p["b"]} == {x, y})


def test_clean_pair_passes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = SC.run(output_dir=tmp_path / "out")
    p = _pair(out, "good_a::t", "good_b::t")
    assert p["co_move"] == "PASS" and p["absolute"] == "PASS" and p["verdict"] == "PASS"


def test_kelvin_offset_caught_by_absolute_not_correlation(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = SC.run(output_dir=tmp_path / "out")
    p = _pair(out, "good_a::t", "kelvin_c::t")
    # correlation is invariant to the +273 offset -> co-move still PASS ...
    assert p["co_move"] == "PASS"
    # ... but absolute agreement must catch it
    assert p["absolute"] == "FAIL" and abs(p["bias"]) > 200 and p["verdict"] == "FAIL"


def test_scrambled_series_caught_by_co_movement(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = SC.run(output_dir=tmp_path / "out")
    p = _pair(out, "good_a::t", "scram_d::t")
    assert p["co_move"] == "FAIL" and p["verdict"] == "FAIL"


def test_overall_fails_and_exit_nonzero_and_ascii(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = SC.run(output_dir=tmp_path / "out")
    assert out["overall"] == "FAIL" and out["n_failing_pairs"] >= 2
    assert SC.main(["--output-dir", str(tmp_path / "out2")]) == 1   # gate exits non-zero on FAIL
    (tmp_path / "out" / "source_consistency.md").read_text(encoding="utf-8").encode("ascii")
