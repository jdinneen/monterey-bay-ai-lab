#!/usr/bin/env python3
"""Additive unsupervised corpus-discovery run for Monterey Bay AI Lab.

This script streams heterogeneous tabular files into a shared hashed feature
space and trains a denoising autoencoder. It never modifies source data.
Artifacts are written under a new run directory for later inspection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import heapq
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - exercised when pyarrow is unavailable
    pq = None


DEFAULT_INPUTS = [
    "data",
    "lakehouse",
    "bacteria_results",
    "nn_cache",
    "mbal_history",
    "research/bacteria/reproduce/expected",
    "signals",
]
DEFAULT_EXCLUDES = {
    ".git",
    ".agent_locks",
    "__pycache__",
    ".pytest_cache",
    "lightning_logs",
    "checkpoints",
    "outputs",
    "release",
    "runs",
    "tmp",
}
DEFAULT_EXCLUDED_MARKERS = {
    "forecast_predictions",
    "mlflow.db",
}
SUPPORTED_SUFFIXES = {".parquet", ".csv", ".json", ".jsonl"}
HASH_SCOPES = {"source", "global"}
LABELISH_FILENAME_TOKENS = ("label", "bacteria", "observation", "training")
LABELISH_COLUMN_EXACT = {
    "class",
    "ground_truth",
    "label",
    "outcome",
    "target",
    "truth",
    "y",
}
LABELISH_COLUMN_TOKENS = (
    "classification",
    "detected",
    "exceed",
    "flag",
    "label",
    "positive",
    "presence",
    "target",
)
EVALUATION_LIMITATIONS = [
    "Lightweight supervised probe over sampled autoencoder embeddings; it is not causal evidence.",
    "Label-like columns are detected heuristically and may mix semantics across sources.",
    "The sample is bounded for cheap corpus runs and may not represent the full data distribution.",
]


@dataclass
class Batch:
    frame: pd.DataFrame
    source: Path
    row_offset: int


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def corrupt_inputs(x: torch.Tensor, noise: float, mask_prob: float) -> torch.Tensor:
    """Apply masked-feature corruption used for self-supervised reconstruction."""
    train_x = x
    if noise > 0:
        train_x = train_x + torch.randn_like(train_x) * noise
    if mask_prob > 0:
        keep = torch.rand_like(train_x).ge(mask_prob).to(train_x.dtype)
        train_x = train_x * keep
    return train_x


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def numeric_transform(value: float) -> float:
    signed = math.copysign(math.log1p(abs(value)), value)
    return float(max(-12.0, min(12.0, signed)) / 12.0)


def json_safe_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def flatten_json_record(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        row: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(raw_value, dict):
                row.update(flatten_json_record(raw_value, name))
            else:
                row[name] = json_safe_value(raw_value)
        return row or {prefix or "value": "{}"}
    return {prefix or "value": json_safe_value(value)}


def json_payload_to_frame(payload: object) -> pd.DataFrame:
    if isinstance(payload, list):
        rows = [
            flatten_json_record(item) if isinstance(item, dict) else {"value": json_safe_value(item)}
            for item in payload
        ]
        return pd.DataFrame(rows)
    if isinstance(payload, dict):
        return pd.DataFrame([flatten_json_record(payload)])
    return pd.DataFrame([{"value": json_safe_value(payload)}])


def iter_jsonl_batches(path: Path, batch_rows: int) -> Iterable[Batch]:
    rows: list[dict[str, object]] = []
    offset = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            frame = json_payload_to_frame(json.loads(text))
            rows.extend(frame.to_dict(orient="records"))
            while len(rows) >= batch_rows:
                yield Batch(frame=pd.DataFrame(rows[:batch_rows]), source=path, row_offset=offset)
                rows = rows[batch_rows:]
                offset += batch_rows
    if rows:
        yield Batch(frame=pd.DataFrame(rows), source=path, row_offset=offset)


def feature_key(kind: str, source_key: str, column: object, hash_scope: str, value: str | None = None) -> str:
    if hash_scope not in HASH_SCOPES:
        raise ValueError(f"unsupported hash_scope={hash_scope!r}")
    column_key = str(column)
    if value is None:
        body = column_key
    else:
        body = f"{column_key}={value}"
    if hash_scope == "global":
        return f"{kind}:{body}"
    return f"{kind}:{source_key}:{body}"


def is_labelish_column(column: object) -> bool:
    name = str(column).strip().lower()
    if name in LABELISH_COLUMN_EXACT:
        return True
    if name.startswith(("has_", "is_")):
        return True
    return any(token in name for token in LABELISH_COLUMN_TOKENS)


def labelish_columns(frame: pd.DataFrame) -> list[str]:
    return [str(col) for col in frame.columns if is_labelish_column(col)]


def update_label_coverage(
    stats: dict[str, dict[str, object]],
    frame: pd.DataFrame,
    source: Path,
    columns: list[str],
) -> None:
    source_key = str(source).replace("\\", "/")
    for col in columns:
        series = frame[col]
        key = f"{source_key}\0{col}"
        stat = stats.setdefault(
            key,
            {
                "source": source_key,
                "column": col,
                "dtype": str(series.dtype),
                "rows_seen": 0,
                "non_null_rows": 0,
                "sample_values": [],
            },
        )
        stat["rows_seen"] = int(stat["rows_seen"]) + len(series)
        non_null = series.dropna()
        stat["non_null_rows"] = int(stat["non_null_rows"]) + int(len(non_null))
        samples = stat["sample_values"]
        assert isinstance(samples, list)
        seen = {json.dumps(value, sort_keys=True, default=str) for value in samples}
        for raw in non_null.head(50).tolist():
            value = json_safe_value(raw)
            marker = json.dumps(value, sort_keys=True, default=str)
            if marker not in seen:
                samples.append(value)
                seen.add(marker)
            if len(samples) >= 20:
                break


def label_coverage_payload(
    files: list[Path],
    label_stats: dict[str, dict[str, object]],
    evaluation_rows: list[dict[str, object]],
) -> dict[str, object]:
    labelish_files = [
        str(p)
        for p in files
        if any(token in str(p).lower() for token in LABELISH_FILENAME_TOKENS)
    ]
    columns: list[dict[str, object]] = []
    for stat in sorted(label_stats.values(), key=lambda item: (str(item["source"]), str(item["column"]))):
        rows_seen = int(stat["rows_seen"])
        non_null_rows = int(stat["non_null_rows"])
        item = dict(stat)
        item["coverage"] = float(non_null_rows / rows_seen) if rows_seen else 0.0
        columns.append(item)
    return {
        "labelish_file_count": len(labelish_files),
        "labelish_files": labelish_files[:500],
        "labelish_column_count": len(columns),
        "labelish_columns": columns[:500],
        "evaluation_sample_rows": len(evaluation_rows),
        "note": "Heuristic filename and column coverage for an unsupervised corpus pass.",
    }


def normalized_label_value(value: object) -> object:
    value = json_safe_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        boolish = {
            "1": 1,
            "true": 1,
            "yes": 1,
            "y": 1,
            "positive": 1,
            "present": 1,
            "detected": 1,
            "exceed": 1,
            "exceeded": 1,
            "0": 0,
            "false": 0,
            "no": 0,
            "n": 0,
            "negative": 0,
            "absent": 0,
            "not_detected": 0,
            "non_exceed": 0,
        }
        return boolish.get(lowered, text)
    if isinstance(value, bool):
        return int(value)
    return value


def coerce_binary_labels(values: list[object]) -> tuple[np.ndarray | None, dict[str, object]]:
    normalized = [normalized_label_value(value) for value in values]
    normalized = [value for value in normalized if value is not None]
    unique = list(dict.fromkeys(normalized))
    if len(unique) != 2:
        return None, {
            "unique_count": len(unique),
            "sample_values": [json_safe_value(value) for value in unique[:20]],
        }
    if all(isinstance(value, int | float | bool) for value in unique):
        unique = sorted(unique, key=float)
    else:
        unique = sorted(unique, key=lambda value: str(value))
    mapping = {unique[0]: 0, unique[1]: 1}
    labels = np.asarray([mapping[value] for value in normalized], dtype=np.int64)
    return labels, {
        "negative_label": json_safe_value(unique[0]),
        "positive_label": json_safe_value(unique[1]),
    }


def build_evaluation_payload(
    evaluation_rows: list[dict[str, object]],
    label_stats: dict[str, dict[str, object]],
    evaluation_sample_limit: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "skipped_no_labelish_columns",
        "evaluation_sample_limit": evaluation_sample_limit,
        "sampled_rows": len(evaluation_rows),
        "labelish_column_count": len(label_stats),
        "limitations": EVALUATION_LIMITATIONS,
        "probes": [],
    }
    if evaluation_sample_limit <= 0:
        payload["status"] = "skipped_disabled"
        return payload
    if not label_stats:
        return payload
    if not evaluation_rows:
        payload["status"] = "skipped_no_sampled_label_rows"
        return payload

    frame = pd.DataFrame(evaluation_rows)
    feature_cols = sorted(c for c in frame.columns if c.startswith("z"))
    if "reconstruction_loss" in frame.columns:
        feature_cols.append("reconstruction_loss")
    label_cols = sorted(c for c in frame.columns if c.startswith("label::"))
    if not feature_cols or not label_cols:
        payload["status"] = "skipped_no_probe_features"
        return payload

    probes: list[dict[str, object]] = []
    for label_col in label_cols:
        subset = frame[[label_col, *feature_cols]].dropna(subset=[label_col])
        normalized = [normalized_label_value(value) for value in subset[label_col].tolist()]
        valid_positions = [i for i, value in enumerate(normalized) if value is not None]
        subset = subset.iloc[valid_positions]
        labels, label_info = coerce_binary_labels([normalized[i] for i in valid_positions])
        probe: dict[str, object] = {
            "label_column": label_col.removeprefix("label::"),
            "rows": int(len(subset)),
            "feature_columns": feature_cols,
            "label_info": label_info,
        }
        if labels is None:
            probe["status"] = "skipped_not_binary"
            probes.append(probe)
            continue
        counts = Counter(labels.tolist())
        probe["class_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
        if len(labels) < 12 or min(counts.values()) < 2:
            probe["status"] = "skipped_insufficient_binary_samples"
            probes.append(probe)
            continue
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
            from sklearn.model_selection import train_test_split

            x_all = subset[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            x_train, x_test, y_train, y_test = train_test_split(
                x_all,
                labels,
                test_size=0.33,
                random_state=42,
                stratify=labels,
            )
            model = LogisticRegression(class_weight="balanced", max_iter=200, solver="liblinear")
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = model.predict_proba(x_test)[:, 1]
            baseline_accuracy = float(max(np.mean(y_test == 0), np.mean(y_test == 1)))
            probe.update(
                {
                    "status": "evaluated_holdout_linear_probe",
                    "train_rows": int(len(y_train)),
                    "test_rows": int(len(y_test)),
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "accuracy_delta_vs_baseline": float(accuracy_score(y_test, pred) - baseline_accuracy),
                    "roc_auc": float(roc_auc_score(y_test, score)),
                    "average_precision": float(average_precision_score(y_test, score)),
                }
            )
        except Exception as exc:  # non-blocking diagnostic artifact only
            probe["status"] = "failed_nonblocking"
            probe["error"] = repr(exc)
        probes.append(probe)

    payload["probes"] = probes
    if any(str(probe.get("status", "")).startswith("evaluated") for probe in probes):
        payload["status"] = "evaluated"
    elif probes:
        payload["status"] = "skipped_no_feasible_probe"
    return payload


def encode_frame(
    frame: pd.DataFrame,
    source: Path,
    row_offset: int,
    input_dim: int,
    hash_scope: str = "source",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if frame.empty:
        return np.zeros((0, input_dim), dtype=np.float32), []

    frame = frame.reset_index(drop=True)
    matrix = np.zeros((len(frame), input_dim), dtype=np.float32)
    refs: list[dict[str, object]] = []
    source_key = str(source).replace("\\", "/")

    numeric_cols = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    object_cols = [c for c in frame.columns if c not in numeric_cols][:32]

    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        slot = stable_hash(feature_key("num", source_key, col, hash_scope), input_dim)
        arr = values.to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(arr)
        if valid.any():
            transformed = np.zeros(len(arr), dtype=np.float32)
            transformed[valid] = np.vectorize(numeric_transform)(arr[valid]).astype(np.float32)
            matrix[:, slot] += transformed

    for col in object_cols:
        values = frame[col].astype("string").fillna("")
        for idx, raw in enumerate(values):
            text = str(raw)
            if not text or text == "<NA>":
                continue
            if len(text) > 96:
                text = text[:96]
            slot = stable_hash(feature_key("cat", source_key, col, hash_scope, text), input_dim)
            matrix[idx, slot] += 1.0

    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] = matrix[nonzero] / np.maximum(norms[nonzero, None], 1.0)

    for i in range(len(frame)):
        refs.append({"source": source_key, "row": int(row_offset + i)})
    return matrix, refs


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(DEFAULT_EXCLUDES):
        return True
    normalized = str(path).replace("\\", "/").lower()
    return any(marker in normalized for marker in DEFAULT_EXCLUDED_MARKERS)


def normalize_fragment(fragment: str) -> str:
    return fragment.strip().replace("\\", "/").lower()


def matches_excluded_fragment(path: Path, exclude_fragments: Iterable[str] | None) -> bool:
    if not exclude_fragments:
        return False
    normalized = str(path).replace("\\", "/").lower()
    return any(fragment and fragment in normalized for fragment in map(normalize_fragment, exclude_fragments))


def discover_files(inputs: Iterable[Path], exclude_fragments: Iterable[str] | None = None) -> list[Path]:
    files: list[Path] = []
    for root in inputs:
        if root.is_file() and root.suffix.lower() in SUPPORTED_SUFFIXES:
            if not should_skip(root) and not matches_excluded_fragment(root, exclude_fragments):
                files.append(root)
        elif root.is_dir():
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in SUPPORTED_SUFFIXES
                    and not should_skip(path)
                    and not matches_excluded_fragment(path, exclude_fragments)
                ):
                    files.append(path)
    return sorted(set(files), key=lambda p: str(p).lower())


def read_batches(path: Path, batch_rows: int) -> Iterable[Batch]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if pq is not None:
            parquet = pq.ParquetFile(path)
            offset = 0
            for record_batch in parquet.iter_batches(batch_size=batch_rows):
                frame = record_batch.to_pandas()
                yield Batch(frame=frame, source=path, row_offset=offset)
                offset += len(frame)
        else:
            frame = pd.read_parquet(path)
            for offset in range(0, len(frame), batch_rows):
                yield Batch(frame=frame.iloc[offset : offset + batch_rows], source=path, row_offset=offset)
    elif suffix == ".csv":
        offset = 0
        for frame in pd.read_csv(path, chunksize=batch_rows, low_memory=False):
            yield Batch(frame=frame, source=path, row_offset=offset)
            offset += len(frame)
    elif suffix == ".jsonl":
        yield from iter_jsonl_batches(path, batch_rows)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = json_payload_to_frame(payload)
        for offset in range(0, len(frame), batch_rows):
            yield Batch(frame=frame.iloc[offset : offset + batch_rows], source=path, row_offset=offset)


def write_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_anomalies(heap: list[tuple[float, int, dict[str, object]]], losses: np.ndarray, refs: list[dict[str, object]], cap: int) -> None:
    for loss, ref in zip(losses, refs, strict=False):
        item = (float(loss), random.randrange(2**31), ref)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def maybe_append_embedding(
    rows: list[dict[str, object]],
    z: torch.Tensor,
    losses: np.ndarray,
    refs: list[dict[str, object]],
    cap: int,
    frame: pd.DataFrame | None = None,
    label_cols: list[str] | None = None,
    evaluation_rows: list[dict[str, object]] | None = None,
    evaluation_cap: int = 0,
) -> None:
    latent = z.detach().float().cpu().numpy()
    embedding_take = min(len(latent), max(cap - len(rows), 0), 16)
    evaluation_take = 0
    eval_frame = None
    if (
        evaluation_rows is not None
        and evaluation_cap > 0
        and label_cols
        and frame is not None
        and len(evaluation_rows) < evaluation_cap
    ):
        evaluation_take = min(len(latent), evaluation_cap - len(evaluation_rows), 16)
        eval_frame = frame.reset_index(drop=True)
    take = max(embedding_take, evaluation_take)
    if take <= 0:
        return
    for i in range(take):
        item = dict(refs[i])
        item["reconstruction_loss"] = float(losses[i])
        for j, value in enumerate(latent[i, :16]):
            item[f"z{j:02d}"] = float(value)
        if i < embedding_take:
            rows.append(item)
        if eval_frame is not None and i < evaluation_take:
            eval_item = dict(item)
            for col in label_cols or []:
                value = json_safe_value(eval_frame.iloc[i][col])
                if value is not None:
                    eval_item[f"label::{col}"] = value
            if any(key.startswith("label::") for key in eval_item):
                evaluation_rows.append(eval_item)


def watchdog_loss_value(raw_loss: float, floor: float) -> float:
    if not math.isfinite(raw_loss):
        return raw_loss
    return max(float(raw_loss), float(floor))


def write_partial_checkpoint(
    output: Path,
    artifacts: Path,
    model: DenoisingAutoencoder,
    args: argparse.Namespace,
    step: int,
    rows_seen: int,
    files_seen: set[str],
    anomalies: list[tuple[float, int, dict[str, object]]],
    embeddings: list[dict[str, object]],
    errors: list[dict[str, str]],
    label_stats: dict[str, dict[str, object]],
    evaluation_rows: list[dict[str, object]],
    files: list[Path],
) -> None:
    anomaly_rows = [
        {"reconstruction_loss": loss, **ref}
        for loss, _, ref in sorted(anomalies, key=lambda x: x[0], reverse=True)
    ]
    pd.DataFrame(anomaly_rows).to_parquet(artifacts / "anomaly_candidates.partial.parquet", index=False)
    pd.DataFrame(embeddings).to_parquet(artifacts / "embeddings.partial.parquet", index=False)
    atomic_json(artifacts / "label_coverage.partial.json", label_coverage_payload(files, label_stats, evaluation_rows))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "run_id": args.run_id,
            "hash_scope": getattr(args, "hash_scope", "source"),
            "step": step,
        },
        output / "autoencoder.partial.pt",
    )
    atomic_json(
        output / "checkpoint.partial.json",
        {
            "event": "partial_checkpoint",
            "run_id": args.run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "rows_seen": rows_seen,
            "files_seen": len(files_seen),
            "error_count": len(errors),
            "embedding_rows": len(embeddings),
            "anomaly_rows": len(anomaly_rows),
            "evaluation_sample_rows": len(evaluation_rows),
            "destructive": False,
        },
    )


def train(args: argparse.Namespace) -> int:
    output = Path(args.output)
    artifacts = output / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    if metrics_path.exists() and not args.append_metrics:
        try:
            metrics_path.unlink()
        except PermissionError:
            # A previous run_safe watchdog can still hold the file briefly on Windows.
            pass

    hash_scope = getattr(args, "hash_scope", "source")
    if hash_scope not in HASH_SCOPES:
        raise ValueError(f"unsupported hash_scope={hash_scope!r}")
    evaluation_sample = max(0, int(getattr(args, "evaluation_sample", 1000)))
    cluster_limit = int(getattr(args, "clusters", 12))
    neighbor_limit = int(getattr(args, "neighbors", 6))
    checkpoint_every = int(getattr(args, "checkpoint_every", 500))
    watchdog_floor = float(getattr(args, "watchdog_loss_floor", 1e-3))

    inputs = [Path(p) for p in args.inputs]
    exclude_fragments = [
        normalize_fragment(fragment)
        for fragment in (getattr(args, "exclude_path_fragment", None) or [])
        if normalize_fragment(fragment)
    ]
    files = discover_files(inputs, exclude_fragments=exclude_fragments)
    manifest = {
        "run_id": args.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in inputs],
        "exclude_path_fragments": exclude_fragments,
        "file_count": len(files),
        "files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "hash_scope": hash_scope,
        "mask_prob": args.mask_prob,
        "evaluation_sample": evaluation_sample,
        "checkpoint_every": checkpoint_every,
        "watchdog_loss_floor": watchdog_floor,
        "hours": args.hours,
        "mode": "unsupervised_corpus_discovery",
        "destructive": False,
    }
    atomic_json(output / "manifest.json", manifest)
    write_jsonl(
        metrics_path,
        {
            "event": "start",
            "step": 0,
            "run_id": args.run_id,
            "file_count": len(files),
            "exclude_path_fragments": exclude_fragments,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "hash_scope": hash_scope,
            "mask_prob": args.mask_prob,
            "evaluation_sample": evaluation_sample,
            "checkpoint_every": checkpoint_every,
            "watchdog_loss_floor": watchdog_floor,
            "hours": args.hours,
            "destructive": False,
        },
    )

    if not files:
        write_jsonl(metrics_path, {"event": "safety_shutdown", "reasons": ["no_input_files"], "step": 0})
        return 2

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model = DenoisingAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    deadline = time.monotonic() + max(args.hours, 0.0) * 3600.0
    step = 0
    rows_seen = 0
    files_seen: set[str] = set()
    errors: list[dict[str, str]] = []
    anomalies: list[tuple[float, int, dict[str, object]]] = []
    embeddings: list[dict[str, object]] = []
    label_stats: dict[str, dict[str, object]] = {}
    evaluation_rows: list[dict[str, object]] = []
    started = time.monotonic()

    while time.monotonic() < deadline or step == 0:
        for file_path in files:
            if time.monotonic() >= deadline and step > 0:
                break
            try:
                for batch in read_batches(file_path, args.batch_rows):
                    if time.monotonic() >= deadline and step > 0:
                        break
                    labels = labelish_columns(batch.frame)
                    if labels:
                        update_label_coverage(label_stats, batch.frame, batch.source, labels)
                    x_np, refs = encode_frame(
                        batch.frame,
                        batch.source,
                        batch.row_offset,
                        args.input_dim,
                        hash_scope=hash_scope,
                    )
                    if len(x_np) == 0:
                        continue
                    x = torch.from_numpy(x_np).to(device)
                    train_x = corrupt_inputs(x, args.noise, args.mask_prob)
                    optimizer.zero_grad(set_to_none=True)
                    recon, z = model(train_x)
                    per_row = torch.mean((recon - x) ** 2, dim=1)
                    loss = per_row.mean()
                    if not torch.isfinite(loss):
                        write_jsonl(metrics_path, {"event": "safety_shutdown", "reasons": ["non_finite_loss"], "step": step})
                        return 5
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    step += 1
                    rows_seen += len(x_np)
                    files_seen.add(str(file_path))
                    losses = per_row.detach().float().cpu().numpy()
                    append_anomalies(anomalies, losses, refs, args.anomaly_cap)
                    maybe_append_embedding(
                        embeddings,
                        z,
                        losses,
                        refs,
                        args.embedding_sample,
                        frame=batch.frame,
                        label_cols=labels,
                        evaluation_rows=evaluation_rows,
                        evaluation_cap=evaluation_sample,
                    )

                    if step % args.metrics_every == 0 or step == 1:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        raw_loss = float(loss.detach().cpu())
                        write_jsonl(
                            metrics_path,
                            {
                                "event": "train",
                                "step": step,
                                "loss": watchdog_loss_value(raw_loss, watchdog_floor),
                                "raw_reconstruction_loss": raw_loss,
                                "rows_seen": rows_seen,
                                "files_seen": len(files_seen),
                                "speed_bps": float(rows_seen / elapsed),
                                "device": str(device),
                                "hash_scope": hash_scope,
                                "mask_prob": args.mask_prob,
                                "seconds_remaining": max(0.0, deadline - time.monotonic()),
                            },
                        )
                    if checkpoint_every > 0 and step % checkpoint_every == 0:
                        write_partial_checkpoint(
                            output,
                            artifacts,
                            model,
                            args,
                            step,
                            rows_seen,
                            files_seen,
                            anomalies,
                            embeddings,
                            errors,
                            label_stats,
                            evaluation_rows,
                            files,
                        )
                    if args.max_steps and step >= args.max_steps:
                        break
            except Exception as exc:  # keep corpus pass additive; record bad readers
                errors.append({"source": str(file_path), "error": repr(exc)})
                write_jsonl(metrics_path, {"event": "read_error", "step": step, "source": str(file_path), "error": repr(exc)})
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "run_id": args.run_id,
            "hash_scope": hash_scope,
        },
        output / "autoencoder.pt",
    )

    anomaly_rows = [
        {"reconstruction_loss": loss, **ref}
        for loss, _, ref in sorted(anomalies, key=lambda x: x[0], reverse=True)
    ]
    anomaly_path = artifacts / "anomaly_candidates.parquet"
    embedding_path = artifacts / "embeddings.parquet"
    cluster_path = artifacts / "clusters.parquet"
    neighbor_path = artifacts / "nearest_neighbors.parquet"
    label_coverage_path = artifacts / "label_coverage.json"
    evaluation_path = artifacts / "label_evaluation.json"

    pd.DataFrame(anomaly_rows).to_parquet(anomaly_path, index=False)
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_parquet(embedding_path, index=False)

    z_cols = [c for c in embedding_df.columns if c.startswith("z")]
    if len(embedding_df) >= 8 and z_cols:
        cluster_count = min(cluster_limit, max(2, len(embedding_df) // 25))
        z_matrix = embedding_df[z_cols].to_numpy(dtype=np.float32)
        kmeans = MiniBatchKMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(z_matrix)
        clusters = embedding_df[["source", "row", "reconstruction_loss"]].copy()
        clusters["cluster"] = cluster_labels
        clusters.to_parquet(cluster_path, index=False)

        nn_count = min(neighbor_limit, len(embedding_df))
        nbrs = NearestNeighbors(n_neighbors=nn_count).fit(z_matrix)
        distances, indices = nbrs.kneighbors(z_matrix[: min(200, len(embedding_df))])
        neighbor_rows = []
        for anchor_idx, (dist_row, idx_row) in enumerate(zip(distances, indices, strict=False)):
            anchor = embedding_df.iloc[anchor_idx]
            for rank, (distance, neighbor_idx) in enumerate(zip(dist_row[1:], idx_row[1:], strict=False), start=1):
                neighbor = embedding_df.iloc[int(neighbor_idx)]
                neighbor_rows.append(
                    {
                        "anchor_source": anchor["source"],
                        "anchor_row": int(anchor["row"]),
                        "neighbor_source": neighbor["source"],
                        "neighbor_row": int(neighbor["row"]),
                        "rank": rank,
                        "distance": float(distance),
                    }
                )
        pd.DataFrame(neighbor_rows).to_parquet(neighbor_path, index=False)
    else:
        pd.DataFrame().to_parquet(cluster_path, index=False)
        pd.DataFrame().to_parquet(neighbor_path, index=False)

    atomic_json(label_coverage_path, label_coverage_payload(files, label_stats, evaluation_rows))
    atomic_json(evaluation_path, build_evaluation_payload(evaluation_rows, label_stats, evaluation_sample))

    finished = {
        **manifest,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "steps": step,
        "rows_seen": rows_seen,
        "files_seen": len(files_seen),
        "errors": errors[:200],
        "error_count": len(errors),
        "artifacts": {
            "model": str(output / "autoencoder.pt"),
            "metrics": str(metrics_path),
            "anomalies": str(anomaly_path),
            "embeddings": str(embedding_path),
            "clusters": str(cluster_path),
            "nearest_neighbors": str(neighbor_path),
            "label_coverage": str(label_coverage_path),
            "label_evaluation": str(evaluation_path),
        },
    }
    atomic_json(output / "summary.json", finished)
    write_jsonl(
        metrics_path,
        {
            "event": "finish",
            "step": step,
            "rows_seen": rows_seen,
            "files_seen": len(files_seen),
            "error_count": len(errors),
            "hash_scope": hash_scope,
            "evaluation_sample_rows": len(evaluation_rows),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    run_id = datetime.now(timezone.utc).strftime("unsup_corpus_%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Run additive unsupervised corpus discovery.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=f"runs/unsupervised_corpus/run_id={run_id}")
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--batch-rows", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hash-scope", choices=sorted(HASH_SCOPES), default="source")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--anomaly-cap", type=int, default=500)
    parser.add_argument("--embedding-sample", type=int, default=5000)
    parser.add_argument("--evaluation-sample", type=int, default=1000)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--watchdog-loss-floor", type=float, default=1e-3)
    parser.add_argument(
        "--exclude-path-fragment",
        action="append",
        default=[],
        help="Skip discovered files whose normalized path contains this fragment. May be repeated.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--append-metrics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(train(parse_args()))
