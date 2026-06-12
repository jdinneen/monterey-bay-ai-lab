#!/usr/bin/env python3
"""Small semi-supervised corpus-discovery run for Monterey Bay AI Lab.

The main task is still unsupervised reconstruction. A weak auxiliary label
head is trained only on rows that already carry a binary event label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.model_lab.unsupervised_corpus_run import (
    DEFAULT_INPUTS,
    HASH_SCOPES,
    append_anomalies,
    atomic_json,
    corrupt_inputs,
    discover_files,
    encode_frame,
    maybe_append_embedding,
    read_batches,
    watchdog_loss_value,
    write_jsonl,
)


LABEL_COLUMNS = [
    "exceed_any",
    "caloes_spill_observed",
    "in_monterey_region",
]


class SemiSupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.label_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z, self.label_head(z).squeeze(-1)


def coerce_binary_labels(frame: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    for col in LABEL_COLUMNS:
        if col not in frame.columns:
            continue
        values = frame[col]
        if values.dtype == bool:
            labels = values.astype("float32").to_numpy()
        else:
            lowered = values.astype("string").str.lower()
            mapped = lowered.map(
                {
                    "true": 1.0,
                    "false": 0.0,
                    "1": 1.0,
                    "0": 0.0,
                    "yes": 1.0,
                    "no": 0.0,
                }
            )
            numeric = pd.to_numeric(values, errors="coerce")
            labels = mapped.fillna(numeric).to_numpy(dtype=np.float32, na_value=np.nan)
        mask = np.isfinite(labels)
        if mask.any():
            return labels, col

    if "bacteria_result" in frame.columns:
        numeric = pd.to_numeric(frame["bacteria_result"], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
        labels = np.where(np.isfinite(numeric), (numeric >= 104.0).astype(np.float32), np.nan)
        if np.isfinite(labels).any():
            return labels, "bacteria_result_ge_104"
    return None, None


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_partial_checkpoint(
    output: Path,
    artifacts: Path,
    model: SemiSupervisedAutoencoder,
    args: argparse.Namespace,
    step: int,
    rows_seen: int,
    files_seen: set[str],
    labeled_seen: int,
    positive_seen: int,
    anomalies: list[tuple[float, int, dict[str, object]]],
    embeddings: list[dict[str, object]],
    supervised_examples: list[dict[str, object]],
    label_sources: dict[str, int],
    errors: list[dict[str, str]],
) -> None:
    anomaly_rows = [
        {"reconstruction_loss": loss, **ref}
        for loss, _, ref in sorted(anomalies, key=lambda x: x[0], reverse=True)
    ]
    pd.DataFrame(anomaly_rows).to_parquet(artifacts / "anomaly_candidates.partial.parquet", index=False)
    pd.DataFrame(embeddings).to_parquet(artifacts / "embeddings.partial.parquet", index=False)
    write_csv_rows(artifacts / "supervised_examples.partial.csv", supervised_examples)
    atomic_json(
        artifacts / "label_summary.partial.json",
        {
            "labeled_seen": labeled_seen,
            "positive_seen": positive_seen,
            "positive_rate": (positive_seen / labeled_seen) if labeled_seen else None,
            "label_sources": label_sources,
        },
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "run_id": args.run_id,
            "step": step,
        },
        output / "semi_supervised_autoencoder.partial.pt",
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
            "labeled_seen": labeled_seen,
            "positive_seen": positive_seen,
            "error_count": len(errors),
            "embedding_rows": len(embeddings),
            "anomaly_rows": len(anomaly_rows),
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
    checkpoint_every = int(getattr(args, "checkpoint_every", 500))
    watchdog_floor = float(getattr(args, "watchdog_loss_floor", 1e-3))
    hash_scope = getattr(args, "hash_scope", "global")
    if hash_scope not in HASH_SCOPES:
        raise ValueError(f"unsupported hash_scope={hash_scope!r}")
    if args.hash_scope not in HASH_SCOPES:
        raise ValueError(f"unsupported hash_scope={args.hash_scope!r}")

    files = discover_files([Path(p) for p in args.inputs])
    manifest = {
        "run_id": args.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "semi_supervised_corpus_discovery",
        "destructive": False,
        "file_count": len(files),
        "files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
        "label_columns": LABEL_COLUMNS,
        "supervised_weight": args.supervised_weight,
        "checkpoint_every": checkpoint_every,
        "watchdog_loss_floor": watchdog_floor,
        "hash_scope": hash_scope,
        "mask_prob": args.mask_prob,
    }
    atomic_json(output / "manifest.json", manifest)
    write_jsonl(
        metrics_path,
        {
            "event": "start",
            "step": 0,
            "run_id": args.run_id,
            "file_count": len(files),
            "hours": args.hours,
            "supervised_weight": args.supervised_weight,
            "checkpoint_every": checkpoint_every,
            "watchdog_loss_floor": watchdog_floor,
            "hash_scope": hash_scope,
            "mask_prob": args.mask_prob,
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

    model = SemiSupervisedAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    deadline = time.monotonic() + args.hours * 3600.0
    started = time.monotonic()
    step = 0
    rows_seen = 0
    labeled_seen = 0
    positive_seen = 0
    files_seen: set[str] = set()
    errors: list[dict[str, str]] = []
    label_sources: dict[str, int] = {}
    anomalies: list[tuple[float, int, dict[str, object]]] = []
    embeddings: list[dict[str, object]] = []
    supervised_examples: list[dict[str, object]] = []

    while time.monotonic() < deadline or step == 0:
        for file_path in files:
            if time.monotonic() >= deadline and step > 0:
                break
            try:
                for batch in read_batches(file_path, args.batch_rows):
                    if time.monotonic() >= deadline and step > 0:
                        break
                    x_np, refs = encode_frame(
                        batch.frame,
                        batch.source,
                        batch.row_offset,
                        args.input_dim,
                        hash_scope=hash_scope,
                    )
                    if len(x_np) == 0:
                        continue
                    labels_np, label_source = coerce_binary_labels(batch.frame.reset_index(drop=True))
                    x = torch.from_numpy(x_np).to(device)
                    train_x = corrupt_inputs(x, args.noise, args.mask_prob)
                    optimizer.zero_grad(set_to_none=True)
                    recon, z, logits = model(train_x)
                    per_row = torch.mean((recon - x) ** 2, dim=1)
                    recon_loss = per_row.mean()
                    total_loss = recon_loss
                    sup_loss_value: float | None = None
                    label_count = 0
                    batch_positive = 0
                    if labels_np is not None and label_source is not None:
                        finite = np.isfinite(labels_np)
                        if finite.any():
                            label_indices = np.flatnonzero(finite)
                            if len(label_indices) > args.max_labels_per_batch:
                                label_indices = np.random.choice(label_indices, args.max_labels_per_batch, replace=False)
                            label_tensor = torch.from_numpy(labels_np[label_indices].astype(np.float32)).to(device)
                            sup_loss = bce(logits[label_indices], label_tensor)
                            total_loss = total_loss + args.supervised_weight * sup_loss
                            sup_loss_value = float(sup_loss.detach().cpu())
                            label_count = int(len(label_indices))
                            batch_positive = int(label_tensor.detach().sum().cpu())
                            labeled_seen += label_count
                            positive_seen += batch_positive
                            label_sources[label_source] = label_sources.get(label_source, 0) + label_count

                    if not torch.isfinite(total_loss):
                        write_jsonl(metrics_path, {"event": "safety_shutdown", "reasons": ["non_finite_loss"], "step": step})
                        return 5
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    step += 1
                    rows_seen += len(x_np)
                    files_seen.add(str(file_path))
                    losses = per_row.detach().float().cpu().numpy()
                    append_anomalies(anomalies, losses, refs, args.anomaly_cap)
                    maybe_append_embedding(embeddings, z, losses, refs, args.embedding_sample)

                    if label_count and len(supervised_examples) < args.supervised_sample:
                        probabilities = torch.sigmoid(logits.detach()).float().cpu().numpy()
                        for idx in np.flatnonzero(np.isfinite(labels_np))[: min(label_count, 16)]:
                            item = dict(refs[int(idx)])
                            item["label_source"] = label_source
                            item["label"] = float(labels_np[int(idx)])
                            item["predicted_probability"] = float(probabilities[int(idx)])
                            supervised_examples.append(item)
                            if len(supervised_examples) >= args.supervised_sample:
                                break

                    if step % args.metrics_every == 0 or step == 1:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        raw_recon_loss = float(recon_loss.detach().cpu())
                        raw_total_loss = float(total_loss.detach().cpu())
                        write_jsonl(
                            metrics_path,
                            {
                                "event": "train",
                                "step": step,
                                "loss": watchdog_loss_value(raw_recon_loss, watchdog_floor),
                                "raw_total_loss": raw_total_loss,
                                "reconstruction_loss": raw_recon_loss,
                                "supervised_loss": sup_loss_value,
                                "rows_seen": rows_seen,
                                "labeled_seen": labeled_seen,
                                "positive_seen": positive_seen,
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
                            labeled_seen,
                            positive_seen,
                            anomalies,
                            embeddings,
                            supervised_examples,
                            label_sources,
                            errors,
                        )
                    if args.max_steps and step >= args.max_steps:
                        break
            except Exception as exc:
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
        output / "semi_supervised_autoencoder.pt",
    )

    anomaly_rows = [
        {"reconstruction_loss": loss, **ref}
        for loss, _, ref in sorted(anomalies, key=lambda x: x[0], reverse=True)
    ]
    pd.DataFrame(anomaly_rows).to_parquet(artifacts / "anomaly_candidates.parquet", index=False)
    pd.DataFrame(embeddings).to_parquet(artifacts / "embeddings.parquet", index=False)
    write_csv_rows(artifacts / "supervised_examples.csv", supervised_examples)
    atomic_json(
        artifacts / "label_summary.json",
        {
            "labeled_seen": labeled_seen,
            "positive_seen": positive_seen,
            "positive_rate": (positive_seen / labeled_seen) if labeled_seen else None,
            "label_sources": label_sources,
        },
    )

    summary = {
        **manifest,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "steps": step,
        "rows_seen": rows_seen,
        "files_seen": len(files_seen),
        "labeled_seen": labeled_seen,
        "positive_seen": positive_seen,
        "label_sources": label_sources,
        "error_count": len(errors),
        "errors": errors[:200],
        "artifacts": {
            "model": str(output / "semi_supervised_autoencoder.pt"),
            "metrics": str(metrics_path),
            "embeddings": str(artifacts / "embeddings.parquet"),
            "anomalies": str(artifacts / "anomaly_candidates.parquet"),
            "label_summary": str(artifacts / "label_summary.json"),
            "supervised_examples": str(artifacts / "supervised_examples.csv"),
        },
    }
    atomic_json(output / "summary.json", summary)
    write_jsonl(
        metrics_path,
        {
            "event": "finish",
            "step": step,
            "rows_seen": rows_seen,
            "files_seen": len(files_seen),
            "labeled_seen": labeled_seen,
            "positive_seen": positive_seen,
            "error_count": len(errors),
            "hash_scope": hash_scope,
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    run_id = datetime.now(timezone.utc).strftime("semisup_corpus_%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description="Run weak semi-supervised corpus discovery.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=f"runs/semi_supervised_corpus/run_id={run_id}")
    parser.add_argument("--run-id", default=run_id)
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--batch-rows", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hash-scope", choices=sorted(HASH_SCOPES), default="global")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--supervised-weight", type=float, default=0.05)
    parser.add_argument("--max-labels-per-batch", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metrics-every", type=int, default=25)
    parser.add_argument("--anomaly-cap", type=int, default=500)
    parser.add_argument("--embedding-sample", type=int, default=5000)
    parser.add_argument("--supervised-sample", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--watchdog-loss-floor", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--append-metrics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(train(parse_args()))
