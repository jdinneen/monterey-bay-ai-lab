#!/usr/bin/env python3
"""
Smoke test for production training pipeline.
Runs a quick on actual MBARI Lakehouse data to verify integration.

Usage: python sota_continual_learning/smoke_test_production.py --steps 50
"""

import argparse
import os
import sys
import torch
import time
import logging
from pathlib import Path

# Disable TCMalloc warning on Windows (optional)
os.environ.pop('LD_PRELOAD', '')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.gpu_admission import enforce_gpu_admission, estimate_smoke_test_mib

from core import ContinualLearner, SafetyMonitor
from trainer import Trainer
from data import LakehouseDataLoader


logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='SOTA Continual Learning Smoke Test')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps to run')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (reduce for larger context windows)')
    parser.add_argument('--context-window', type=int, default=720, help='Context window (hours)')
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("Starting SOTA Continual Learning smoke test")

    enforce_gpu_admission(
        label=f"smoke_test_production.py batch={args.batch_size} context={args.context_window}",
        request_mib=estimate_smoke_test_mib(args.batch_size, args.context_window),
        enabled=os.environ.get("MBARI_ACCEL", "gpu").lower() != "cpu",
    )
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram:.1f} GB")
    
    # Initialize SafetyMonitor
    safety_monitor = SafetyMonitor(vram_threshold=0.85)
    
    # Model config (Production Lakehouse Contract compliant):
    # - hidden_dim=1024 (NOT 2048 which overfits on ~3.8M rows)
    # - num_experts=32 (increased capacity)
    model = ContinualLearner(
        input_dim=1,
        hidden_dim=1024,
        num_experts=32
    ).to(device)
    
    logger.info(f"Model: hidden_dim=1024, experts=32")

    # Trainer with FP16 for RTX 5090 (mixed precision)
    trainer = Trainer(
        model,
        device=device,
        mixed_precision=True,  # Enable mixed precision training
        grad_accum_steps=2  # Reduce accumulation since larger context uses more memory
    )

    #Lakehouse data loader (small smoke test - limit valid series)
    logger.info("Loading Lakehouse data...")
    
    lake_loader = LakehouseDataLoader(
        parquet_path='lakehouse/gold/forecast_predictions',
        context_window=args.context_window,
        forecast_horizon=24,
        min_rows_per_series=800
    )
    
    # Limit to fewer series for smoke test (use first 3 for quick validation)
    lake_loader.valid_series = lake_loader.valid_series[:3]
    logger.info(f"Loaded {len(lake_loader.valid_series)} series for smoke test")
    
    # Training loop
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    start_time = time.time()
    loss_sum = 0.0
    mape_sum = 0.0
    
    step = 0
    batch_count = 0
    
    for batch_x, batch_target in lake_loader.stream_batched(batch_size=args.batch_size):
        if step >= args.steps:
            break
        
        safety_monitor.heartbeat()
        
        # Move to device
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)
        
        # Train step
        metrics = trainer.train_step_with_target(batch_x, batch_target)
        
        # Debug: print input stats before loss computation
        if step == 0:
            logger.info(f"Batch X stats - min: {batch_x.min():.4f}, max: {batch_x.max():.4f}, mean: {batch_x.mean():.4f}")
            logger.info(f"Batch Target stats - min: {batch_target.min():.4f}, max: {batch_target.max():.4f}, mean: {batch_target.mean():.4f}")
            # Check for NaN in inputs
            if torch.isnan(batch_x).any():
                logger.error("NaN values found in batch_x!")
            if torch.isnan(batch_target).any():
                logger.error("NaN values found in batch_target!")
        
        loss_sum += metrics['loss']
        mape_sum += metrics.get('mape', 0)
        
        # Safety check
        shutdown_reasons = safety_monitor.check_all()
        if shutdown_reasons:
            logger.warning(f"Shutdown triggered: {shutdown_reasons}")
            break
        
        if step % 10 == 0:
            logger.info(
                f"Step {step+1}/{args.steps} | "
                f"Loss: {metrics['loss']:.4f}, MAPE: {metrics.get('mape', 'N/A')}"
            )
        
        step += 1
        batch_count += 1
    
    elapsed = time.time() - start_time
    logger.info(f"\nSmoke test complete!")
    logger.info(f"Steps: {step}")
    logger.info(f"Batches/sec: {batch_count/elapsed:.2f}")
    logger.info(f"Avg loss: {loss_sum/step:.4f}")
    logger.info(f"Avg MAPE: {mape_sum/step:.4f}")
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_reserved() / 1024**3
        vram_pct = vram_used / (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 100
        logger.info(f"Peak VRAM: {vram_used:.2f} GB ({vram_pct:.1f}%)")


if __name__ == '__main__':
    main()
