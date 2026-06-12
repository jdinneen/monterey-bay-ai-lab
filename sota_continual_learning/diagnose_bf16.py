#!/usr/bin/env python3
"""
bf16 NaN-divergence diagnostic for the SOTA continual-learning MoE.

A prior fp16 production run (run_production.py, hidden_dim=1024, experts=32,
batch=2, context=168) trained cleanly for a few hundred steps then went NaN
around step ~1000 and never recovered. The data was exonerated (worst-case
z-scored |x| across all 115k real windows = 6.2, far below fp16's 65504), so
the overflow was happening *inside* the model under fp16 autocast.

trainer.Trainer now uses bf16 autocast (same exponent range as fp32, so
activations can't overflow to inf) plus a non-finite-loss skip guard. This
script reproduces the exact failing configuration and runs PAST the death point
to confirm bf16 survives. It saves no checkpoints and writes only a small log.

Run: python sota_continual_learning/diagnose_bf16.py --steps 1500
"""

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import ContinualLearner            # noqa: E402
from trainer import Trainer                   # noqa: E402
from data import LakehouseDataLoader          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="bf16 NaN diagnostic")
    ap.add_argument("--steps", type=int, default=1500,
                    help="Steps to run (default 1500 — past the ~1000 death point)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--context-window", type=int, default=168)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--num-experts", type=int, default=32)
    ap.add_argument("--parquet-path", type=str,
                    default="lakehouse/gold/forecast_predictions")
    ap.add_argument("--log", type=str,
                    default="sota_continual_learning/bf16_diag.log")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(args.log, "w", encoding="utf-8")

    def emit(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    emit(f"device={device} | torch={torch.__version__}")
    if device == "cuda":
        emit(f"GPU={torch.cuda.get_device_name(0)}")

    model = ContinualLearner(input_dim=1, hidden_dim=args.hidden_dim,
                             num_experts=args.num_experts).to(device)
    trainer = Trainer(model, device=device, mixed_precision=True, grad_accum_steps=2)
    emit(f"amp_dtype={getattr(trainer, 'amp_dtype', 'n/a')} | "
         f"scaler={'set' if getattr(trainer, 'scaler', None) is not None else 'None'} | "
         f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    loader = LakehouseDataLoader(
        parquet_path=args.parquet_path,
        context_window=args.context_window,
        forecast_horizon=24,
        min_rows_per_series=800,
    )
    emit(f"loader ready: {len(loader.valid_series)} valid series")

    step = 0
    nan_steps = 0
    first_nan_step = None
    finite_losses = 0
    t0 = time.time()
    emit(f"=== running {args.steps} steps (bs={args.batch_size}, ctx={args.context_window}, "
         f"hidden={args.hidden_dim}, experts={args.num_experts}) ===")

    while step < args.steps:
        progressed = False
        for bx, bt in loader.stream_batched(batch_size=args.batch_size):
            if step >= args.steps:
                break
            progressed = True
            m = trainer.train_step_with_target(bx.to(device), bt.to(device))
            step += 1
            loss = m.get("loss", float("nan"))
            is_nan = (loss is None) or (isinstance(loss, float) and not math.isfinite(loss)) \
                or m.get("nan_skipped", False)
            if is_nan:
                nan_steps += 1
                if first_nan_step is None:
                    first_nan_step = step
                    emit(f"!! FIRST non-finite at step {step} "
                         f"(nan_skips_total={m.get('nan_skips_total')})")
            else:
                finite_losses += 1
            if step % 100 == 0:
                spd = step / (time.time() - t0)
                vram = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
                emit(f"step {step}/{args.steps} | loss={loss if isinstance(loss,(int,float)) else 'NaN'} "
                     f"| mape={m.get('mape')} | nan_steps={nan_steps} "
                     f"| {spd:.1f} st/s | VRAM {vram:.1f}GB")
        if not progressed:
            emit("stream exhausted with no batches — check data path")
            break

    dt = time.time() - t0
    verdict = "PASS — bf16 survived past the death point with no NaN" if nan_steps == 0 else (
        f"PARTIAL — {nan_steps} non-finite steps (guard skipped them; "
        f"first at {first_nan_step})" if finite_losses > nan_steps else
        f"FAIL — {nan_steps} non-finite steps dominate (first at {first_nan_step})")
    summary = {
        "steps_run": step,
        "finite_steps": finite_losses,
        "nan_steps": nan_steps,
        "first_nan_step": first_nan_step,
        "elapsed_s": round(dt, 1),
        "verdict": verdict,
    }
    emit("=== SUMMARY ===")
    emit(json.dumps(summary, indent=2))
    logf.close()


if __name__ == "__main__":
    main()
