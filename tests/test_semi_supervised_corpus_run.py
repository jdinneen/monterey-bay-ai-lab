from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_lab.semi_supervised_corpus_run import coerce_binary_labels, train


class Args:
    def __init__(self, root: Path, output: Path) -> None:
        self.inputs = [str(root)]
        self.output = str(output)
        self.run_id = "test_semisup"
        self.hours = 0.0
        self.batch_rows = 4
        self.input_dim = 64
        self.hidden_dim = 16
        self.latent_dim = 4
        self.hash_scope = "global"
        self.dropout = 0.0
        self.noise = 0.0
        self.mask_prob = 0.0
        self.learning_rate = 1e-3
        self.supervised_weight = 0.05
        self.max_labels_per_batch = 4
        self.device = "cpu"
        self.metrics_every = 1
        self.anomaly_cap = 10
        self.embedding_sample = 10
        self.supervised_sample = 10
        self.checkpoint_every = 1
        self.watchdog_loss_floor = 1e-3
        self.max_steps = 2
        self.append_metrics = False


def test_coerce_binary_labels_prefers_exceed_any() -> None:
    labels, source = coerce_binary_labels(pd.DataFrame({"exceed_any": [True, False, None]}))

    assert source == "exceed_any"
    assert labels is not None
    assert labels[:2].tolist() == [1.0, 0.0]
    assert np.isnan(labels[2])


def test_semi_supervised_runner_writes_label_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output = tmp_path / "run"
    data_root.mkdir()
    pd.DataFrame(
        {
            "site": ["a", "b", "c", "d", "e", "f"],
            "exceed_any": [False, True, False, False, True, False],
            "value": [1.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        }
    ).to_parquet(data_root / "labels.parquet", index=False)

    code = train(Args(data_root, output))

    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["destructive"] is False
    assert summary["hash_scope"] == "global"
    assert summary["mask_prob"] == 0.0
    assert summary["labeled_seen"] > 0
    assert (output / "semi_supervised_autoencoder.pt").exists()
    assert (output / "artifacts" / "label_summary.json").exists()
    assert (output / "artifacts" / "supervised_examples.csv").exists()
    assert (output / "checkpoint.partial.json").exists()
    assert (output / "semi_supervised_autoencoder.partial.pt").exists()
    assert (output / "artifacts" / "label_summary.partial.json").exists()
