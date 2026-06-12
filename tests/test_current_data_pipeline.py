from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ops import current_data_pipeline as cdp


def test_materialize_source_inventory_and_driver_manifest(tmp_path, monkeypatch):
    root = tmp_path
    curated = root / "data" / "external_curated" / "ready" / "ready.parquet"
    curated.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(curated, index=False)

    monkeypatch.setattr(cdp, "ROOT", root)
    monkeypatch.setattr(cdp, "PIPELINE_DIR", root / "reports" / "current_data_pipeline")
    monkeypatch.setattr(cdp, "PIPELINE_JSON", root / "reports" / "current_data_pipeline" / "current_data_pipeline.json")
    monkeypatch.setattr(cdp, "LAKEHOUSE_SOURCE_DIR", root / "lakehouse" / "silver" / "source_inventory")
    monkeypatch.setattr(cdp, "LAKEHOUSE_SOURCE_PARQUET", root / "lakehouse" / "silver" / "source_inventory" / "source_inventory.parquet")
    monkeypatch.setattr(cdp, "LAKEHOUSE_SOURCE_JSON", root / "lakehouse" / "silver" / "source_inventory" / "source_inventory.json")
    monkeypatch.setattr(cdp, "DRIVER_MANIFEST", root / "lakehouse" / "silver" / "external_drivers" / "drivers_manifest.json")

    status = {
        "sources": [
            {
                "source": "ready",
                "title": "Ready Source",
                "priority": 1,
                "status": "READY_FOR_MODELING",
                "rows": 3,
                "columns": 1,
                "date_min": "2026-01-01",
                "date_max": "2026-01-03",
                "curated_path": "data/external_curated/ready/ready.parquet",
                "duplicate_key_count": 0,
            },
            {
                "source": "blocked",
                "title": "Blocked Source",
                "priority": 2,
                "status": "NOT_STARTED",
                "rows": 0,
            },
        ]
    }

    inv = cdp.materialize_source_inventory(status)
    assert inv["sources"] == 2
    assert inv["ready_sources"] == 1
    assert inv["not_ready"][0]["source"] == "blocked"
    assert cdp.LAKEHOUSE_SOURCE_PARQUET.exists()
    assert pd.read_parquet(cdp.LAKEHOUSE_SOURCE_PARQUET)["sha256"].notna().any()

    manifest = cdp.write_driver_manifest(status, inv)
    assert manifest["ready_sources"] == 1
    data = json.loads(cdp.DRIVER_MANIFEST.read_text(encoding="utf-8"))
    assert data["ready_sources"][0]["source"] == "ready"
    assert data["policy"]["data_tables_are_referenced_not_duplicated"] is True
