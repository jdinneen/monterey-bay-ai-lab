#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import build_news_event_features as news_features  # noqa: E402
from ops import merge_driver_tables  # noqa: E402


def taxonomy() -> dict:
    return {
        "taxonomy_version": "test",
        "event_classes": [
            {"event_type": "spill_release"},
            {"event_type": "hab_biotoxin"},
        ],
        "feature_windows_hours": [2],
    }


def test_news_features_use_available_at_not_event_time():
    grid = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h", name="ds")
    events = pd.DataFrame(
        [
            {
                "event_type": "spill_release",
                "event_time_start": "2026-01-01T00:00:00Z",
                "available_at_utc": "2026-01-01T03:00:00Z",
                "severity_score": 2.0,
                "confidence_score": 0.5,
                "relevance_score": 1.0,
            }
        ]
    )
    events["available_at_utc"] = pd.to_datetime(events["available_at_utc"], utc=True)

    features = news_features.build_features(
        grid,
        events,
        ["spill_release", "hab_biotoxin"],
        [2],
    )

    by_ds = features.set_index("ds")
    assert by_ds.loc[pd.Timestamp("2026-01-01T02:00:00Z"), "news_spill_release_count_2h"] == 0.0
    assert by_ds.loc[pd.Timestamp("2026-01-01T03:00:00Z"), "news_spill_release_count_2h"] == 1.0
    assert by_ds.loc[pd.Timestamp("2026-01-01T04:00:00Z"), "news_spill_release_severity_sum_2h"] == 1.0


def test_write_outputs_marks_news_features_hist_only(tmp_path):
    grid = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1h", name="ds")
    events = pd.DataFrame(columns=["event_type", "available_at_utc"])
    features = news_features.build_features(grid, events, ["spill_release"], [2])

    manifest = news_features.write_outputs(
        features,
        events,
        taxonomy(),
        tmp_path / "news.parquet",
        tmp_path / "news_manifest.json",
        tmp_path / "events.parquet",
    )

    assert manifest["futr"] == []
    assert manifest["hist"]
    assert all(col.startswith("news_") for col in manifest["hist"])
    saved = json.loads((tmp_path / "news_manifest.json").read_text(encoding="utf-8"))
    assert saved["leakage_rule"].endswith("hist-only")


def test_merge_driver_tables_combines_futr_and_news_hist(tmp_path):
    ds = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1h")
    physical = pd.DataFrame({"ds": ds, "tide_cos": [1.0, 0.5], "wind": [0.0, 3.0]})
    news = pd.DataFrame({"ds": ds, "news_spill_release_count_2h": [0.0, 1.0]})
    physical_path = tmp_path / "physical.parquet"
    news_path = tmp_path / "news.parquet"
    physical_manifest = tmp_path / "physical.json"
    news_manifest = tmp_path / "news.json"
    physical.to_parquet(physical_path, index=False)
    news.to_parquet(news_path, index=False)
    physical_manifest.write_text(json.dumps({"futr": ["tide_cos"], "hist": ["wind"]}), encoding="utf-8")
    news_manifest.write_text(json.dumps({"futr": [], "hist": ["news_spill_release_count_2h"]}), encoding="utf-8")

    merged, manifest = merge_driver_tables.merge_drivers(
        [(physical_path, physical_manifest), (news_path, news_manifest)]
    )

    assert list(merged.columns) == ["ds", "tide_cos", "wind", "news_spill_release_count_2h"]
    assert manifest["futr"] == ["tide_cos"]
    assert manifest["hist"] == ["wind", "news_spill_release_count_2h"]

