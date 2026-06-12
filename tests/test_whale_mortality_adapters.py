"""Offline tests for the whale-mortality thread adapters.

Network-free: exercises registry wiring + the pure annotation-parsing / chunk-planning
logic. These two sources are deliberately NOT >=50k corpus sources:
  * marine_mammal_mortality is a bounded mortality-LABEL slice (bounded_sample=True),
  * cciea_forage is the COMPLETE published forage index (~500 annual rows).
So unlike the lane-B corpus sources they are exempt from the 50k gate; what matters is
correct labeling (dead_flag) and effort capture. Real fetched counts are checked by the
framework's validation.json.
"""
from __future__ import annotations

import pandas as pd

from ops.data_fetch.registry import get_spec, validate_registry
from ops.data_fetch.adapters.marine_mammal_mortality import (
    MarineMammalMortalityAdapter,
    TERM_ALIVE_OR_DEAD,
    VALUE_DEAD,
    VALUE_ALIVE,
)
from ops.data_fetch.adapters.cciea_forage import CcieaForageAdapter

WHALE_KEYS = ["marine_mammal_mortality", "cciea_forage"]


def test_registry_valid_and_sources_wired():
    assert not validate_registry(), validate_registry()
    for key in WHALE_KEYS:
        spec = get_spec(key)
        assert spec.required_columns and spec.date_column == "time"
        assert ":" in spec.adapter
        # both resolve to a real adapter class
        assert spec.build_adapter() is not None


def test_mortality_dead_flag_parsing():
    f = MarineMammalMortalityAdapter._dead_flag
    dead = {"annotations": [{"controlled_attribute_id": TERM_ALIVE_OR_DEAD,
                             "controlled_value_id": VALUE_DEAD}]}
    alive = {"annotations": [{"controlled_attribute_id": TERM_ALIVE_OR_DEAD,
                              "controlled_value_id": VALUE_ALIVE}]}
    other = {"annotations": [{"controlled_attribute_id": 1, "controlled_value_id": 2}]}
    none = {"annotations": []}
    assert f(dead) == 1                       # Dead -> 1
    assert f(alive) == 0                      # Alive -> 0
    assert pd.isna(f(other))                  # unrelated annotation -> <NA> (counts as effort)
    assert pd.isna(f(none))                   # no annotation -> <NA>
    assert pd.isna(f({}))                     # missing key -> <NA>
    # Dead wins if both present (a carcass annotated by two users)
    both = {"annotations": [
        {"controlled_attribute_id": TERM_ALIVE_OR_DEAD, "controlled_value_id": VALUE_ALIVE},
        {"controlled_attribute_id": TERM_ALIVE_OR_DEAD, "controlled_value_id": VALUE_DEAD}]}
    assert f(both) == 1


def test_mortality_chunks_are_years():
    adapter = get_spec("marine_mammal_mortality").build_adapter()
    chunks = list(adapter.iter_chunks("2015", "2018"))
    assert [c["key"] for c in chunks] == ["2015", "2016", "2017", "2018"]
    assert all("year" in c for c in chunks)


def test_mortality_bounded_sample_and_bounds():
    spec = get_spec("marine_mammal_mortality")
    assert spec.bounded_sample is True          # never auto-promotes to READY
    assert spec.dedup_keys == ["observation_id"]
    assert "dead_flag" in spec.required_columns
    assert spec.value_bounds["dead_flag"] == (0.0, 1.0)


def test_cciea_forage_single_chunk_and_query():
    adapter = get_spec("cciea_forage").build_adapter()
    chunks = list(adapter.iter_chunks(None, None))
    assert len(chunks) == 1 and chunks[0]["key"] == "all"
    assert isinstance(adapter, CcieaForageAdapter)
    spec = get_spec("cciea_forage")
    assert spec.dedup_keys == ["time", "species_group"]
    assert "mean_cpue" in spec.required_columns
