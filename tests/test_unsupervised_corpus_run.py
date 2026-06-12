from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_lab.unsupervised_corpus_run import discover_files, encode_frame, read_batches, train


class Args:
    def __init__(self, root: Path, output: Path) -> None:
        self.inputs = [str(root)]
        self.output = str(output)
        self.run_id = "test_unsupervised"
        self.hours = 0.0
        self.batch_rows = 4
        self.input_dim = 64
        self.hidden_dim = 16
        self.latent_dim = 4
        self.dropout = 0.0
        self.noise = 0.0
        self.mask_prob = 0.0
        self.learning_rate = 1e-3
        self.device = "cpu"
        self.metrics_every = 1
        self.anomaly_cap = 10
        self.embedding_sample = 10
        self.evaluation_sample = 10
        self.hash_scope = "source"
        self.clusters = 4
        self.neighbors = 4
        self.checkpoint_every = 1
        self.watchdog_loss_floor = 1e-3
        self.exclude_path_fragment = []
        self.max_steps = 2
        self.append_metrics = False


def test_discover_files_keeps_supported_structured_inputs(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x,y\n1,a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ignored", encoding="utf-8")

    files = discover_files([tmp_path])

    assert [p.name for p in files] == ["a.csv"]


def test_discover_files_honors_exclude_path_fragments(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "observations.csv").write_text("x,y\n1,a\n", encoding="utf-8")
    split_dir = tmp_path / "lakehouse" / "silver" / "forecast_splits" / "split_id=abc"
    split_dir.mkdir(parents=True)
    (split_dir / "split_manifest.json").write_text('{"split_id":"abc"}', encoding="utf-8")

    files = discover_files([tmp_path], exclude_fragments=["lakehouse/silver/forecast_splits"])

    assert [p.name for p in files] == ["observations.csv"]


def test_read_batches_normalizes_mixed_json_payloads(tmp_path: Path) -> None:
    mixed = tmp_path / "metrics.json"
    mixed.write_text(
        json.dumps(
            {
                "summary": {"loss": 0.25, "rows": 3},
                "rows_seen": 9,
                "tags": ["alpha", "beta"],
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    scalar = tmp_path / "scalar.json"
    scalar.write_text("17", encoding="utf-8")
    mixed_list = tmp_path / "mixed_list.json"
    mixed_list.write_text(json.dumps([{"a": 1}, 2, {"b": {"c": 3}}]), encoding="utf-8")

    mixed_frame = list(read_batches(mixed, batch_rows=10))[0].frame
    assert mixed_frame.loc[0, "summary.loss"] == 0.25
    assert mixed_frame.loc[0, "rows_seen"] == 9
    assert json.loads(mixed_frame.loc[0, "tags"]) == ["alpha", "beta"]

    scalar_frame = list(read_batches(scalar, batch_rows=10))[0].frame
    assert scalar_frame.loc[0, "value"] == 17

    list_batches = list(read_batches(mixed_list, batch_rows=2))
    assert sum(len(batch.frame) for batch in list_batches) == 3
    assert list_batches[0].frame.loc[0, "a"] == 1
    assert list_batches[0].frame.loc[1, "value"] == 2
    assert list_batches[1].frame.iloc[0]["b.c"] == 3


def test_global_hash_scope_aligns_same_features_across_sources(tmp_path: Path) -> None:
    frame = pd.DataFrame({"temperature": [12.5], "site_type": ["surf"]})
    input_dim = 65536

    global_a, _ = encode_frame(frame, tmp_path / "a.csv", 0, input_dim, hash_scope="global")
    global_b, _ = encode_frame(frame, tmp_path / "b.csv", 0, input_dim, hash_scope="global")
    source_a, _ = encode_frame(frame, tmp_path / "a.csv", 0, input_dim, hash_scope="source")
    source_b, _ = encode_frame(frame, tmp_path / "b.csv", 0, input_dim, hash_scope="source")

    assert set(np.flatnonzero(global_a[0])) == set(np.flatnonzero(global_b[0]))
    assert set(np.flatnonzero(source_a[0])) != set(np.flatnonzero(source_b[0]))


def test_unsupervised_runner_writes_additive_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "run"
    data_root.mkdir()
    pd.DataFrame(
        {
            "site": ["a", "b", "c", "d", "e"],
            "label": [0, 1, 0, 1, 1],
            "value": [1.0, 2.5, 3.0, 4.0, 5.0],
        }
    ).to_csv(data_root / "labels.csv", index=False)
    pd.DataFrame({"driver": [0.1, 0.2, 0.3], "name": ["x", "y", "z"]}).to_parquet(
        data_root / "drivers.parquet", index=False
    )

    code = train(Args(data_root, output))

    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["destructive"] is False
    assert summary["exclude_path_fragments"] == []
    assert summary["mask_prob"] == 0.0
    assert summary["rows_seen"] > 0
    assert (output / "autoencoder.pt").exists()
    assert (output / "artifacts" / "anomaly_candidates.parquet").exists()
    assert (output / "artifacts" / "embeddings.parquet").exists()
    assert (output / "artifacts" / "clusters.parquet").exists()
    assert (output / "artifacts" / "nearest_neighbors.parquet").exists()
    assert (output / "artifacts" / "label_coverage.json").exists()
    assert (output / "artifacts" / "label_evaluation.json").exists()
    assert (output / "checkpoint.partial.json").exists()
    assert (output / "autoencoder.partial.pt").exists()
    assert (output / "artifacts" / "embeddings.partial.parquet").exists()
    assert (output / "metrics.jsonl").exists()


def test_label_coverage_and_evaluation_artifact_for_label_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "run"
    data_root.mkdir()
    rows = 32
    pd.DataFrame(
        {
            "site": [f"s{i % 4}" for i in range(rows)],
            "target_label": [i % 2 for i in range(rows)],
            "value": [float(i % 2) * 10.0 + i / 100.0 for i in range(rows)],
        }
    ).to_csv(data_root / "training_labels.csv", index=False)
    args = Args(data_root, output)
    args.batch_rows = 8
    args.embedding_sample = rows
    args.evaluation_sample = rows
    args.input_dim = 128
    args.hidden_dim = 24
    args.latent_dim = 6
    args.mask_prob = 0.25
    args.hours = 1.0
    args.max_steps = 4

    code = train(args)

    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["mask_prob"] == 0.25
    coverage = json.loads((output / "artifacts" / "label_coverage.json").read_text(encoding="utf-8"))
    assert coverage["labelish_column_count"] == 1
    assert coverage["labelish_columns"][0]["column"] == "target_label"
    assert coverage["labelish_columns"][0]["non_null_rows"] == rows
    assert coverage["evaluation_sample_rows"] >= 12

    evaluation = json.loads((output / "artifacts" / "label_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["status"] == "evaluated"
    assert "not causal evidence" in " ".join(evaluation["limitations"])
    assert evaluation["probes"][0]["status"] == "evaluated_holdout_linear_probe"
    assert evaluation["probes"][0]["label_column"] == "target_label"
