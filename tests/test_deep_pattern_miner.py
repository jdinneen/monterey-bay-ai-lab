from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.model_lab import deep_pattern_miner as dpm


def test_build_plan_is_traffic_gated_and_additive(monkeypatch, tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps({"source_inventory": {"ready_sources": 2, "sources": 2, "ready_rows": 123}}),
        encoding="utf-8",
    )
    root = tmp_path / "repo"
    (root / "data" / "external_curated").mkdir(parents=True)
    (root / "lakehouse" / "silver").mkdir(parents=True)

    monkeypatch.setattr(dpm, "PIPELINE_JSON", pipeline)
    monkeypatch.setattr(dpm, "REPO_ROOT", root)

    plan = dpm.build_plan("unit", hours=1.0)

    assert plan.gate["critic_gate"] == "PASS"
    assert plan.gate["traffic_gated"] is True
    assert plan.ready_sources == 2
    assert "ops/run_safe.py" in plan.commands["unsupervised_safe"]
    assert "--exclude-path-fragment" in plan.commands["unsupervised_safe"]
    assert "lakehouse/silver/forecast_splits" in plan.commands["unsupervised_safe"]
    assert "--hash-scope" in plan.commands["semi_supervised_safe"]
    assert "global" in plan.commands["semi_supervised_safe"]
    assert plan.run_root.startswith("reports/model_lab/deep_pattern_miner/")


def test_write_plan_outputs_json_and_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dpm, "PLAN_JSON", tmp_path / "plan.json")
    monkeypatch.setattr(dpm, "PLAN_MD", tmp_path / "plan.md")
    plan = dpm.MinerPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        ready_sources=1,
        total_sources=1,
        ready_rows=10,
        inputs=["data/external_curated"],
        run_root="reports/model_lab/deep_pattern_miner/run_id=x",
        commands={"unsupervised_safe": ["python", "x"], "semi_supervised_safe": ["python", "y"]},
        gate={"critic_gate": "PASS"},
        research_sources=[{"name": "SCARF", "url": "https://example.test", "use": "basis"}],
    )

    dpm.write_plan(plan)

    payload = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert payload["project"] == "Monterey Bay AI Lab"
    assert payload["unsupervised_artifact_excludes"] == dpm.UNSUPERVISED_ARTIFACT_EXCLUDES
    assert payload["gate"]["critic_gate"] == "PASS"
    assert "Deep Pattern Miner Plan" in (tmp_path / "plan.md").read_text(encoding="utf-8")


def test_smoke_runs_both_miners(tmp_path: Path) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()
    pd.DataFrame(
        {
            "station": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "exceed_any": [0, 1, 0, 0, 1, 0, 1, 0],
            "rain": [0.0, 1.2, 0.2, 0.0, 2.0, 0.1, 1.8, 0.0],
        }
    ).to_parquet(data / "toy.parquet", index=False)

    result = dpm.run_smoke(out, [str(data)])

    assert result["status"] == "PASS"
    assert (out / "unsupervised" / "summary.json").exists()
    assert (out / "semi_supervised" / "summary.json").exists()
    semi = json.loads((out / "semi_supervised" / "summary.json").read_text(encoding="utf-8"))
    assert semi["hash_scope"] == "global"
