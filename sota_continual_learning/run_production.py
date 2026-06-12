#!/usr/bin/env python3
"""
Production-grade training runner for SOTA Continual Learning System.

Uses actual MBAL forecast_predictions data from Lakehouse (gold layer).
Integrates with SafetyMonitor for RTX 5090 VRAM protection.

Run: python sota_continual_learning/run_production.py --total-steps 10000
"""

import argparse
import json
import os
import sys
import time
import torch
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.gpu_admission import enforce_gpu_admission, estimate_run_production_mib

# Import high-signal components
from core import ContinualLearner, SafetyMonitor
from trainer import Trainer
from data import create_high_performance_loader, CudaPrefetcher


logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging for production run."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # SafetyMonitor (core) logs VRAM/Runtime/Idle every step at INFO. Over a 50k-step
    # run that floods the log; keep the checks active but silence the per-step spam —
    # VRAM is captured in metrics.jsonl instead.
    logging.getLogger('core').setLevel(logging.WARNING)


def load_config(output_dir: str) -> dict:
    """
    Load or create configuration for production run.
    
    Configuration follows the Production Lakehouse Contract:
    - hidden_dim=1024 (keep this, NOT 2048 which causes overfitting on ~3.8M rows)
    - num_experts=32 (increase from default 16)
    - context_window=720 (30 days hourly)
    """
    config = {
        'model': {
            'input_dim': 1,  # Univariate time series
            'hidden_dim': 1024,  # MATCH AGENTS.md directive: keep hidden_dim=1024
            'num_experts': 32,   # INCREASE to 32 for capacity (was 16)
        },
        'training': {
            'batch_size': 8,      # Smaller batch for RTX 5090 with large context window
            'context_window': 720,  # 30 days hourly
            'forecast_horizon': 24,  # Predict next 24 hours
        },
        'trainer': {
            'mixed_precision': True,  # bf16 on RTX 5090 (see trainer.py: amp_dtype)
            # Dense-MoE (all 32 experts active) caps the micro-batch at ~2 before OOM,
            # so use gradient accumulation for a larger EFFECTIVE batch (2 x 8 = 16)
            # at micro-batch-2 memory. Smoother gradients -> real convergence vs the
            # very noisy batch-2 signal, with zero extra VRAM.
            'grad_accum_steps': 8,
            'learning_rate': 1e-4,
            'weight_decay': 0.1,
        },
        'safety': {
            'vram_threshold': 0.90,  # card is dedicated to this run; 0.80 was for shared use
            'max_runtime_hours': 72.0,
            'idle_shutdown_minutes': 30.0,
        },
        'output': {
            'checkpoint_dir': str(Path(output_dir) / 'checkpoints'),
            'metrics_dir': str(Path(output_dir) / 'metrics'),
        }
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Production SOTA Continual Learning Training Runner'
    )
    
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10000,
        help='Total training steps (default: 10000)'
    )
    
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1000,
        help='Evaluation interval in steps (default: 1000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='sota_continual_learning/output_production',
        help='Output directory for checkpoints and metrics'
    )
    
    parser.add_argument(
        '--parquet-path',
        type=str,
        default='lakehouse/gold/forecast_predictions',
        help='Path to Lakehouse Parquet data (default: gold layer)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size. The dense-MoE model is ~1.3B params; keep small to fit '
             'RTX 5090 VRAM and stay under the SafetyMonitor threshold (default: 2)'
    )

    parser.add_argument(
        '--context-window',
        type=int,
        default=168,
        help='Context window in hours of history per sample (default: 168 = 7 days). '
             'Larger windows (e.g. 720) raise activation memory ~linearly and may OOM.'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--use-latent',
        action='store_true',
        help='Enable LatentTSF: Compress raw sensors into abstract latent signals before MoE.'
    )

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    enforce_gpu_admission(
        label=f"run_production.py batch={args.batch_size} context={args.context_window}",
        request_mib=estimate_run_production_mib(args.batch_size, args.context_window),
        enabled=os.environ.get("MBAL_ACCEL", "gpu").lower() != "cpu",
    )
    
    # Create output directories
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    metrics_dir = Path(args.output_dir) / 'metrics'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.output_dir)
    config['training']['parquet_path'] = args.parquet_path
    config['training']['batch_size'] = args.batch_size
    config['training']['context_window'] = args.context_window
    
    logger.info(f"Starting production training run")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total steps: {args.total_steps}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({total_mem:.1f} GB VRAM)")
    
    # Initialize model with Production Lakehouse Contract settings
    input_dim = config['model']['input_dim']
    
    model = ContinualLearner(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_experts=config['model']['num_experts']
    ).to(device)
    
    logger.info(f"Model initialized: hidden_dim={config['model']['hidden_dim']}")
    
    # Initialize trainer with RTX 5090 optimizations
    trainer = Trainer(
        model,
        device=device,
        mixed_precision=config['trainer']['mixed_precision'],
        grad_accum_steps=config['trainer']['grad_accum_steps']
    )
    
    # Initialize SafetyMonitor for RTX 5090 protection
    safety_monitor = SafetyMonitor(
        vram_threshold=config['safety']['vram_threshold'],
        max_runtime_hours=config['safety']['max_runtime_hours'],
        idle_shutdown_minutes=config['safety']['idle_shutdown_minutes']
    )
    
    # Initialize high-performance Lakehouse data loader (streams gold/forecast_predictions ~5.8M rows)
    logger.info(f"Initializing high-performance loader from {config['training']['parquet_path']}")
    
    loader = create_high_performance_loader(
        parquet_path=args.parquet_path,
        batch_size=config['training']['batch_size'],
        num_workers=8,  # Scale workers for RTX 5090
        context_window=config['training']['context_window'],
        forecast_horizon=config['training']['forecast_horizon']
    )
    
    # Initialize CUDA Prefetcher (the Firehose)
    prefetcher = CudaPrefetcher(loader, device=torch.device(device))
    
    # Removed fake continual learning ExperienceReplayBuffer
    
    # ---- Structured metrics logger ("watch everything") ----
    metrics_path = metrics_dir / 'metrics.jsonl'
    metrics_f = open(metrics_path, 'a', encoding='utf-8')

    def log_metrics(record: dict):
        record.setdefault('wall', round(time.time(), 1))
        metrics_f.write(json.dumps(record) + '\n')
        metrics_f.flush()

    def rotate_checkpoints(keep: int = 3):
        """Keep only the most recent `keep` step_*.pt (each is ~15 GB)."""
        ckpts = sorted(checkpoint_dir.glob('step_*.pt'), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def vram_gb() -> float:
        return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    # Training loop — CONTINUOUS. The lakehouse stream is finite (one pass over the
    # series), so we re-open it each epoch and keep going until total_steps is reached.
    start_time = time.time()
    logger.info(f"Starting CONTINUOUS training for {args.total_steps} steps (stream cycles per epoch)")
    log_metrics({'event': 'run_start', 'total_steps': args.total_steps,
                 'batch_size': config['training']['batch_size'],
                 'context_window': config['training']['context_window'],
                 'num_experts': config['model']['num_experts'],
                 'hidden_dim': config['model']['hidden_dim']})

    step = 0
    batch_count = 0
    epoch = 0
    best_loss = float('inf')
    stop = False

    while step < args.total_steps and not stop:
        epoch += 1
        logger.info(f"--- Starting Epoch {epoch} ---")
        
        # CRITIC FIX: Streamlined prefetcher iteration and reset logic
        # Initialize or restart prefetcher at the start of each epoch
        prefetcher = CudaPrefetcher(loader, device=torch.device(device))
        
        while step < args.total_steps:
            batch = prefetcher.next()
            if batch is None:
                logger.info("End of data stream reached. Moving to next epoch.")
                break

            batch_x, batch_target = batch

            safety_monitor.heartbeat()

            loss_metrics = trainer.train_step_with_target(batch_x, batch_target)

            shutdown_reasons = safety_monitor.check_all()
            if shutdown_reasons:
                logger.warning(f"[SAFETY] Shutdown triggered: {shutdown_reasons}")
                log_metrics({'event': 'safety_shutdown', 'step': step,
                             'reasons': list(shutdown_reasons)})
                stop = True
                break

            step += 1
            batch_count += 1

            loss_val = loss_metrics['loss']
            if loss_val < best_loss:
                best_loss = loss_val

            # Record metrics every 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                speed = prefetcher.get_throughput()
                lr = trainer.optimizer.param_groups[0]['lr']
                v = vram_gb()
                logger.info(
                    f"Step {step}/{args.total_steps} | epoch {epoch} | "
                    f"Loss: {loss_val:.4f} | MAPE: {loss_metrics.get('mape', 0):.4f} | "
                    f"Throughput: {speed:.2f} b/s | VRAM: {v:.1f} GB"
                )
                log_metrics({'step': step, 'epoch': epoch, 'loss': loss_val,
                             'mape': loss_metrics.get('mape'), 'best_loss': best_loss,
                             'lr': lr, 'speed_bps': round(speed, 3),
                             'vram_gb': round(v, 2), 'elapsed_s': round(elapsed, 1)})

            # Periodic checkpoint save (rotated so disk use stays bounded)
            if step % args.eval_interval == 0:
                checkpoint_path = checkpoint_dir / f'step_{step}.pt'
                trainer.save_checkpoint(checkpoint_path)
                rotate_checkpoints(keep=3)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                log_metrics({'event': 'checkpoint', 'step': step,
                             'path': str(checkpoint_path)})
            
            # Next batch
            batch = prefetcher.next()

    # Final save
    final_path = checkpoint_dir / 'final.pt'
    trainer.save_checkpoint(final_path)

    elapsed_time = time.time() - start_time

    logger.info(f"Training complete!")
    logger.info(f"Steps completed: {step}/{args.total_steps} over {epoch} epoch(s)")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final checkpoint: {final_path}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    log_metrics({'event': 'run_complete', 'steps': step, 'epochs': epoch,
                 'best_loss': best_loss, 'total_time_s': round(elapsed_time, 1)})
    metrics_f.close()


if __name__ == '__main__':
    main()
