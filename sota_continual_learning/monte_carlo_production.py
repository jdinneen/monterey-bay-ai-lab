#!/usr/bin/env python3
"""
Monte Carlo Simulation for SOTA Continual Learning System.

Runs 1,000 randomized trials of the 10,000-step training process.
Aggregates statistical results (Mean, Std, Min, Max) for Loss and MAPE.
Optimized for RTX 5090 throughput.
"""

import argparse
import os
import sys
import time
import torch
import logging
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import SOTA components
from sota_continual_learning.core import ContinualLearner, SafetyMonitor
from sota_continual_learning.trainer import Trainer
from sota_continual_learning.data import LakehouseDataLoader

logger = logging.getLogger(__name__)

@dataclass
class TrialResult:
    trial_id: int
    final_loss: float
    final_mape: float
    duration_s: float
    status: str  # 'completed' or 'shutdown'

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

import gc

def run_trial(trial_id, args, device, config):
    """Run a single training trial."""
    logger.info(f"--- Starting Trial {trial_id}/{args.num_trials} ---")
    
    # Clear memory from previous trials
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Set seed for this trial
    torch.manual_seed(trial_id * 42)
    np.random.seed(trial_id * 42)
    
    # Initialize model
    model = ContinualLearner(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_experts=config['model']['num_experts']
    ).to(device)
    
    # Initialize trainer with more stability
    trainer = Trainer(
        model,
        device=device,
        mixed_precision=True,
        grad_accum_steps=8
    )
    # Lower learning rate for Monte Carlo stability
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 5e-5
    
    # Ensure metrics don't leak memory over hundreds of trials
    trainer.reset_metrics()
    
    # Initialize SafetyMonitor
    safety_monitor = SafetyMonitor(
        vram_threshold=0.85, # Be slightly more conservative
        max_runtime_hours=args.max_trial_hours
    )
    
    # Initialize DataLoader
    lake_loader = LakehouseDataLoader(
        parquet_path=args.parquet_path,
        context_window=config['training']['context_window'],
        forecast_horizon=config['training']['forecast_horizon']
    )
    
    start_time = time.time()
    last_metrics = {'loss': 0, 'mape': 0}
    status = 'completed'
    
    step = 0
    for batch_x, batch_target in lake_loader.stream_batched(batch_size=args.batch_size):
        if step >= args.steps_per_trial:
            break
            
        batch_x = batch_x.to(device)
        batch_target = batch_target.to(device)
        
        safety_monitor.heartbeat()
        loss_metrics = trainer.train_step_with_target(batch_x, batch_target)
        
        # NaN detection
        if np.isnan(loss_metrics['loss']) or np.isnan(loss_metrics.get('mape', 0)):
            logger.error(f"Trial {trial_id} encountered NaN at step {step}")
            last_metrics = loss_metrics
            status = 'failed_nan'
            break
            
        last_metrics = loss_metrics
        
        # Periodic step logging
        if step % 500 == 0:
            logger.info(f"Trial {trial_id} | Step {step}/{args.steps_per_trial} | Loss: {last_metrics['loss']:.4f}")
        
        # Safety check
        reasons = safety_monitor.check_all()
        if reasons:
            logger.warning(f"Trial {trial_id} triggered shutdown: {reasons}")
            status = 'shutdown'
            break
            
        step += 1
        
    duration = time.time() - start_time
    logger.info(f"Trial {trial_id} finished in {duration:.1f}s | Status: {status} | Final Loss: {last_metrics['loss']:.4f}")
    
    return TrialResult(
        trial_id=trial_id,
        final_loss=float(last_metrics['loss']),
        final_mape=float(last_metrics.get('mape', 0)),
        duration_s=duration,
        status=status
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trials', type=int, default=1000)
    parser.add_argument('--steps-per-trial', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-trial-hours', type=float, default=1.0)
    parser.add_argument('--parquet-path', type=str, default='lakehouse/gold/forecast_predictions')
    parser.add_argument('--output-file', type=str, default='sota_continual_learning/monte_carlo_results.json')
    args = parser.parse_args()
    
    setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'model': {'input_dim': 1, 'hidden_dim': 1024, 'num_experts': 32},
        'training': {'context_window': 168, 'forecast_horizon': 24}
    }
    
    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            results = [TrialResult(**r) for r in json.load(f)]
        logger.info(f"Resuming from trial {len(results)}")

    start_sim_time = time.time()
    
    try:
        for i in range(len(results), args.num_trials):
            result = run_trial(i, args, device, config)
            results.append(result)
            
            # Periodic save
            if i % 5 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                    
            # Check total sim time
            if time.time() - start_sim_time > 3600 * 12: # Auto-save and exit after 12h
                logger.info("Reached 12h limit for this session. Saving and exiting.")
                break
                
    except KeyboardInterrupt:
        logger.info("Simulation interrupted. Saving current results.")
    
    # Aggregate and report
    if results:
        losses = [r.final_loss for r in results]
        mapes = [r.final_mape for r in results]
        
        summary = {
            "num_trials": len(results),
            "loss": {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "min": float(np.min(losses)),
                "max": float(np.max(losses))
            },
            "mape": {
                "mean": float(np.mean(mapes)),
                "std": float(np.std(mapes)),
                "min": float(np.min(mapes)),
                "max": float(np.max(mapes))
            }
        }
        
        print("\n" + "="*40)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*40)
        print(json.dumps(summary, indent=2))
        print("="*40)
        
        with open(args.output_file.replace('.json', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
