#!/usr/bin/env python3
"""
Main entry point for SOTA Continual Learning System.

This system continuously learns and optimizes itself on RTX 5090.
It never stops - it's designed for lifelong learning.
"""

import argparse
import torch
import json
import logging
from pathlib import Path
import time
from datetime import datetime
from typing import List

from core import (
    ContinualLearner,
    DynamicMoE,
    ElasticWeightConsolidation,
    ExperienceReplayBuffer,
    SafetyMonitor
)
from trainer import Trainer, PerformanceMonitor
from data import (
    SyntheticContinualDataset,
    DataDriftGenerator,
    create_dataloaders,
    LakehouseDataLoader
)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


logger = logging.getLogger(__name__)


class ContinualLearningLoop:
    """
    Main loop that orchestrates continual learning.

    Never stops - continuously learns from new data, adapts architecture,
    and optimizes its own hyperparameters.

    SAFETY FEATURES:
    - VRAM monitoring with auto-shutdown at 80%
    - Time-based termination to prevent runaway training
    """

    def __init__(self, config: dict):
        self.config = config

        # Device setup
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")

        # Initialize model
        self.model = ContinualLearner(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_experts=config['model'].get('num_experts', 32)
        ).to(self.device)

        logger.info(f"Model initialized: hidden_dim={config['model']['hidden_dim']}, experts={config['model'].get('num_experts', 32)}")

        # Initialize trainer
        self.trainer = Trainer(
            self.model,
            device=self.device,
            mixed_precision=config.get('mixed_precision', True),
            grad_accum_steps=config.get('grad_accum_steps', 4)
        )

        # Initialize replay buffer
        self.replay_buffer = ExperienceReplayBuffer(
            max_size=config.get('replay_size', 1000)
        )

        # Initialize safety monitor (shutdown at 80% VRAM, max 72h runtime)
        self.safety_monitor = SafetyMonitor(
            vram_threshold=0.80,
            max_runtime_hours=config.get('max_runtime_hours', 72),
            idle_shutdown_minutes=config.get('idle_shutdown_minutes', 30)
        )

        # Metrics tracking
        self.metrics = {
            'train': [],
            'validation': [],
            'architectures_used': [],
            'tasks_seen': []
        }

        self.total_steps = 0

    def check_safety(self) -> List[str]:
        """Check all safety conditions and return shutdown reasons if any."""
        return self.safety_monitor.check_all()
    
    def continual_learning_step(
        self,
        batch: torch.Tensor,
        task_id: int
    ) -> dict:
        """Perform one step of continual learning with replay."""
        # Sample from replay buffer if available
        replay_metrics = {}
        
        if self.replay_buffer.size() > 0:
            replay_x, _, _ = self.replay_buffer.sample(batch_size=len(batch) // 2)
            
            if len(replay_x) > 0:
                combined_batch = torch.cat([batch.to(self.device), replay_x.to(self.device)], dim=0)
                
                # Train on both current and replayed data
                loss_metrics = self.trainer.train_step(combined_batch)
                
                # Add replayed samples to buffer
                for i, sample in enumerate(replay_x):
                    if i < len(batch) // 2:
                        self.replay_buffer.add(sample, torch.tensor(0), task_id)
        else:
            loss_metrics = self.trainer.train_step(batch)
        
        return loss_metrics
    
    def train(
        self,
        total_steps: int = 100000,
        eval_interval: int = 1000,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Run the continual learning loop.
        
        Args:
            total_steps: Total training steps
            eval_interval: Evaluation interval
            checkpoint_dir: Directory for checkpoints
        """
        logger.info(f"Starting continual learning for {total_steps} steps")

        # Initialize Lakehouse data loader (streams silver/forecast_splits)
        parquet_path = self.config.get('parquet_path', 'lakehouse/silver/forecast_splits')
        context_window = self.config.get('context_window', 720)  # 30 days
        forecast_horizon = self.config.get('forecast_horizon', 24)
        
        try:
            lake_loader = LakehouseDataLoader(
                parquet_path=parquet_path,
                context_window=context_window,
                forecast_horizon=forecast_horizon,
                min_rows_per_series=800
            )
            logger.info(f"Lakehouse data loaded: {len(lake_loader.valid_series)} valid series")
        except Exception as e:
            logger.warning(f"Failed to load LakehouseDataLoader: {e}, falling back to synthetic data")
            lake_loader = None

        start_time = time.time()

        # Handle bothLakehouse streaming and fallback data generator
        if lake_loader is not None:
            # Stream from actual lakehouse data
            batch_size = self.config.get('batch_size', 8)
            
            for step, (batch_x, batch_target) in enumerate(lake_loader.stream_batched(batch_size=batch_size)):
                self.total_steps = step + 1
                
                # Move to device (batch_x: (batch, seq_len=720, input_dim))
                batch_x = batch_x.to(self.device)
                batch_target = batch_target.to(self.device)
                
                # Perform learning step with target supervision
                metrics = self.continual_learning_step_with_target(batch_x, batch_target, task_id=step // 100)

                # Mark activity (reset idle timer)
                self.safety_monitor.heartbeat()

                # SAFETY CHECK - shutdown if threshold exceeded
                shutdown_reasons = self.check_safety()
                if shutdown_reasons:
                    logger.warning(f"[SAFETY] Shutdown triggered: {shutdown_reasons}")
                    break  # Gracefully exit training loop

                # Record metrics
                self.metrics['train'].append({
                    'step': step,
                    'loss': metrics['loss'],
                    'ewc_loss': metrics.get('ewc_loss', 0),
                    'expert_balance_std': metrics.get('metric_expert_balance_std', 0),
                    'reconstruction_loss': metrics.get('reconstruction_loss', 0)
                })

                # Progress update
                if step % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step + 1) / elapsed

                    logger.info(
                        f"Step {step+1}/{total_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"EWC: {metrics.get('ewc_loss', 0):.4f} | "
                        f"Expert balance std: {metrics.get('metric_expert_balance_std', 0):.3f} | "
                        f"Speed: {steps_per_sec:.2f} steps/sec"
                    )

                # Evaluation checkpoint
                if (step + 1) % eval_interval == 0:
                    val_loss = self.trainer.evaluate(self._create_eval_loader())
                
                    self.metrics['validation'].append({
                        'step': step,
                        'loss': val_loss
                    })
                
                    logger.info(f"Validation loss at step {step+1}: {val_loss:.4f}")
                
                    # Save checkpoint
                    checkpoint_path = Path(checkpoint_dir) / f'step_{step+1}.pt'
                    self.trainer.save_checkpoint(checkpoint_path)
                
                    # Update best model if improved
                    if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        best_path = Path(checkpoint_dir) / 'best.pt'
                        torch.save({
                            'model_state': self.model.state_dict(),
                            'total_steps': step + 1,
                            'metrics': self.metrics
                        }, best_path)
            
                # Early save checkpoint periodically
                if (step + 1) % 5000 == 0:
                    interim_path = Path(checkpoint_dir) / f'checkpoint_{step+1}.pt'
                    torch.save({
                        'model_state': self.model.state_dict(),
                        'total_steps': step + 1,
                        'metrics': self.metrics
                    }, interim_path)
        
        # Final save
        final_path = Path(checkpoint_dir) / 'final.pt'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'total_steps': total_steps,
            'metrics': self.metrics
        }, final_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete! Total time: {elapsed:.2f}s")
        logger.info(f"Final loss: {self.metrics['train'][-1]['loss']:.4f}")
    
    def _create_eval_loader(self):
        """Create evaluation data loader."""
        class DummyLoader:
            def __init__(self, gen, count=10):
                self.gen = gen
                self.count = count
            
            def __iter__(self):
                for _ in range(self.count):
                    yield self.gen.generate_batch(32)
        
        return DummyLoader(
            DataDriftGenerator(input_dim=self.config['model']['input_dim']),
            count=10
        )
    
    def save_state(self, path: str) -> None:
        """Save complete learning state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model.state_dict(),
            'metrics': self.metrics,
            'total_steps': self.total_steps
        }
        
        torch.save(state, path)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='SOTA Continual Learning System for RTX 5090'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'demo'],
        default='train',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to dataset (optional for demo)'
    )
    
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10000,
        help='Total training steps'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        default=None,
        help='Path to JSON config file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for resuming'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and metrics'
    )

    # Safety parameters
    parser.add_argument(
        '--max-vram-pct',
        type=float,
        default=80.0,
        help='Auto-shutdown at this percent VRAM usage (default: 80)'
    )

    parser.add_argument(
        '--max-runtime-hours',
        type=float,
        default=72.0,
        help='Max continuous runtime in hours (default: 72 hours)'
    )

    parser.add_argument(
        '--idle-shutdown-minutes',
        type=float,
        default=30.0,
        help='Shutdown after this many idle minutes (default: 30 minutes)'
    )

    args = parser.parse_args()
    
    # Load config
    config = {
        'model': {
            'input_dim': 256,
            'hidden_dim': 512,
            'num_experts': 16
        },
        'batch_size': 32,
        'mixed_precision': True,
        'grad_accum_steps': 4,
        'replay_size': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Safety defaults (can be overridden by config file)
        'max_runtime_hours': args.max_runtime_hours,
        'idle_shutdown_minutes': args.idle_shutdown_minutes
    }

    # VRAM threshold is passed as percentage, convert to 0-1 for config
    config['vram_threshold'] = args.max_vram_pct / 100.0

    if args.config:
        config = load_config(args.config)
    
    # Initialize learning loop
    cl_loop = ContinualLearningLoop(config)
    
    if args.checkpoint:
        cl_loop.trainer.load_checkpoint(args.checkpoint)
    
    if args.mode == 'train':
        cl_loop.train(
            total_steps=args.total_steps,
            checkpoint_dir=str(Path(args.output_dir) / 'checkpoints')
        )
    elif args.mode == 'eval':
        # evaluation mode would go here
        pass
    else:  # demo
        logger.info("Running demo - single batch update")
        
        data_gen = DataDriftGenerator(input_dim=config['model']['input_dim'])
        batch = data_gen.generate_batch(32)
        
        metrics = cl_loop.continual_learning_step(batch, task_id=0)
        logger.info(f"Demo step completed. Loss: {metrics['loss']:.4f}")


if __name__ == '__main__':
    main()
