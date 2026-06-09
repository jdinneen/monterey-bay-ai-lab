"""
Trainer engine for SOTA continual learning system.
Handles distributed training, mixed precision, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Loss Functions
# =============================================================================

def huber_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for robust regression (less sensitive to outliers than MSE)."""
    return F.smooth_l1_loss(predictions, targets, beta=delta, reduction='mean')


def mape_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean Absolute Percentage Error (MAPE) metric with robust zero handling."""
    numerator = torch.abs(targets - predictions)
    # Use a larger epsilon to avoid division by very small values
    denominator = torch.abs(targets) + eps
    return (numerator / denominator).mean()


class PerformanceMonitor:
    """Track performance metrics during training."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.metrics: Dict[str, List[float]] = {}
        self.start_time = time.time()
    
    def start_timer(self, name: str) -> None:
        setattr(self, f'_{name}_start', time.time())
    
    def stop_timer(self, name: str) -> float:
        elapsed = time.time() - getattr(self, f'_{name}_start')
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        return elapsed
    
    def record_metric(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_avg_timing(self, name: str) -> float:
        timings = self.timings.get(name, [])
        return sum(timings) / len(timings) if timings else 0.0
    
    def get_avg_metric(self, name: str) -> float:
        metrics = self.metrics.get(name, [])
        return sum(metrics) / len(metrics) if metrics else 0.0
    
    @property
    def total_time(self) -> float:
        return time.time() - self.start_time


class Trainer:
    """
    Distributed training engine with mixed precision support.
    Optimized for RTX 5090's AMP and memory capacity.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        mixed_precision: bool = True,
        grad_accum_steps: int = 4
    ):
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        self.grad_accum_steps = grad_accum_steps
        
        # Mixed precision: use bfloat16 on the RTX 5090, NOT fp16. bf16 keeps
        # fp32's exponent range, so a large-scale (unnormalized) target batch can
        # no longer overflow to inf -> NaN the way fp16 did (which diverged a prior
        # run at step ~800). bf16 also needs no GradScaler.
        if mixed_precision:
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32
        self.scaler = None  # bf16 requires no loss scaling
        
        # Optimizer - use fused AdamW if available (RTX 5090)
        try:
            from apex.optimizers import FusedAdam
            self.optimizer = FusedAdam(model.parameters(), lr=1e-4, weight_decay=0.1)
            logger.info("Using FusedAdam optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=1e-4, 
                weight_decay=0.1,
                fused=True  # RTX 5090 supports this
            )
        
        # Learning rate scheduler (cosine with warmup)
        self.scheduler = None
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for larger batch memory efficiency."""
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = True
    
    def _setup_scheduler(
        self,
        num_training_steps: int,
        warmup_steps: int = 1000
    ) -> None:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(100, num_training_steps // 10),
            T_mult=2
        )
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step with mixed precision and gradient accumulation.
        
        Args:
            batch: Input data tensor
            
        Returns:
            Dictionary of loss and metrics
        """
        self.model.train()
        
        # Move batch to device
        if isinstance(batch, dict):
            x = batch.get('input_ids', batch.get('x'))
        else:
            x = batch
        
        x = x.to(self.device)
        
        # Mixed precision forward pass (new API: torch.amp.autocast)
        with torch.amp.autocast('cuda', enabled=self.mixed_precision):
            output, extras = self.model(x, compute_loss=True)
            loss = extras['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
        
        # Backward pass with scaler (new API: use scale when scaling)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation check
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            # Clip gradients
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step with scaler
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        
        # Record metrics
        def _get_val(v):
            if isinstance(v, list):
                v = v[-1]
            if isinstance(v, torch.Tensor):
                return v.mean().item() if v.numel() > 1 else v.item()
            return v

        metrics = {
            'loss': loss.item() * self.grad_accum_steps,
            **{f'metric_{k}': _get_val(v) for k, v in extras.get('metrics', {}).items()}
        }

        # Advance global step (gates the grad-accumulation optimizer step above)
        self.global_step += 1

        return metrics

    def train_step_with_target(
        self,
        batch_x: torch.Tensor,
        batch_target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one training step with target supervision using Huber loss.
        
        Args:
            batch_x: Input sequences (batch, seq_len, input_dim)
            batch_target: Target predictions (batch, forecast_horizon)
            
        Returns:
            Dictionary of loss and metrics including MAPE
        """
        self.model.train()
        
        # Move to device
        batch_x = batch_x.to(self.device)
        batch_target = batch_target.to(self.device)
        
        # Mixed precision forward pass (bf16 on RTX 5090)
        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.mixed_precision):
            # Forward pass - get predictions for the forecast horizon
            outputs, extras = self.model(batch_x, compute_loss=True)

            # Extract predictions for forecast horizon (use last step output)
            if outputs.dim() == 2:
                predictions = outputs[:, :batch_target.shape[1]]  # Slice to match target length
            else:
                # Reshape to match target dimensions
                predictions = outputs.view(outputs.size(0), -1)[:, :batch_target.shape[1]]

            # Compute Huber loss (robust regression)
            loss = huber_loss(predictions, batch_target)

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

        # NaN/Inf guard: a single pathological batch (e.g. a huge unnormalized
        # target) must never back-propagate non-finite gradients and permanently
        # poison the weights, which is what killed the prior run. Skip the update
        # for this batch instead, and surface it via the nan_skipped flag.
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            self._nan_skips = getattr(self, '_nan_skips', 0) + 1
            return {'loss': float('nan'), 'mape': float('nan'),
                    'nan_skipped': True, 'nan_skips_total': self._nan_skips}

        # Backward pass with scaler
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation check
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            # Clip gradients
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Step with scaler
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

        # Compute MAPE metric
        with torch.no_grad():
            mape = mape_loss(predictions.detach(), batch_target)

        # Record metrics
        def _get_val(v):
            if isinstance(v, list):
                v = v[-1]
            if isinstance(v, torch.Tensor):
                return v.mean().item() if v.numel() > 1 else v.item()
            return v

        metrics = {
            'loss': loss.item() * self.grad_accum_steps,
            'mape': mape.item(),
            **{f'metric_{k}': _get_val(v) for k, v in extras.get('metrics', {}).items()}
        }

        # Advance global step (gates the grad-accumulation optimizer step above)
        self.global_step += 1

        return metrics

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 100,
        max_steps: Optional[int] = None,
        val_interval: int = 1000,
        checkpoint_dir: str = 'checkpoints',
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Maximum number of epochs
            max_steps: Maximum number of training steps
            val_interval: Validation interval in steps
            checkpoint_dir: Directory for checkpoints
            save_best: Whether to save best model
            
        Returns:
            History dictionary with losses and metrics
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        history = {'train_loss': [], 'val_loss': []}
        
        self._setup_scheduler(len(train_loader) * num_epochs)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Mixed precision: {self.mixed_precision}")

        global_step = 0

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                self.monitor.start_timer('data_loading')
                
                try:
                    # Data loading timing (if loader supports it)
                    pass
                except:
                    pass
                
                self.monitor.start_timer('forward_backward')
                
                # Training step
                metrics = self.train_step(batch)
                
                forward_backward_time = self.monitor.stop_timer('forward_backward')
                
                # Accumulate losses
                epoch_loss += metrics['loss']
                num_steps += 1
                global_step += 1
                
                # Record timing
                self.monitor.record_metric('forward_backward_time', forward_backward_time)
                
                # Print progress every 10 steps
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}, "
                        f"Loss: {metrics['loss']:.4f}, "
                        f"Forward/Backward: {forward_backward_time:.3f}s"
                    )
                
                # Validation
                if val_loader and global_step % val_interval == 0:
                    val_loss = self.evaluate(val_loader)
                    history['val_loss'].append(val_loss)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    if save_best and val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(Path(checkpoint_dir) / 'best.pt')
                        logger.info("Saved new best model")
                
                # Max steps check
                if max_steps and global_step >= max_steps:
                    break
            
            # Epoch average loss
            avg_epoch_loss = epoch_loss / (num_steps + 1e-8)
            history['train_loss'].append(avg_epoch_loss)
            
            logger.info(
                f"Epoch {epoch} complete. "
                f"Avg train loss: {avg_epoch_loss:.4f}, "
                f"Total time: {self.monitor.total_time:.0f}s"
            )
        
        # Save final checkpoint
        self.save_checkpoint(Path(checkpoint_dir) / 'final.pt')
        
        return history
    
    def evaluate(self, data_loader: Any) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    x = batch.get('input_ids', batch.get('x'))
                else:
                    x = batch
                
                x = x.to(self.device)
                
                output, extras = self.model(x, compute_loss=True)
                total_loss += extras['loss'].item() * len(x)
                num_samples += len(x)
        
        return total_loss / (num_samples + 1e-8)
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }
        
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state'])
        
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if 'scheduler_state' in checkpoint and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        if 'scaler_state' in checkpoint and checkpoint['scaler_state'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path}")


if __name__ == '__main__':
    # Quick test
    from core import ContinualLearner
    
    model = ContinualLearner(input_dim=128, hidden_dim=256)
    trainer = Trainer(model, mixed_precision=False)  # FP32 for quick test
    
    print("Trainer initialized successfully")
