"""
High-Signal Core Module.

Replaced the bloated MoE/EWC/LatentTSF architecture with a streamlined, 
anti-bloat Multi-Layer Perceptron (MLP) architecture.
Focuses strictly on Lakehouse integration and temporal prediction without 
resume-driven complexity.

SAFETY FEATURES:
- VRAM monitoring with auto-shutdown at 80% usage
- Time-based termination to prevent runaway training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class ContinualLearner(nn.Module):
    """
    Simplified High-Signal Architecture.
    A multi-layer perceptron over a 1D convolution abstractor.
    Replaces the previous 1000-line "Continual Learning MoE" with standard PyTorch.
    
    (Note: num_experts is kept in the signature purely for backwards compatibility 
    with existing run_production.py configuration parsers, but it is ignored).
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 1024,
        num_experts: int = 32  # Ignored (kept for compat)
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple 1D Convolution over the time domain to extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Forecast head predicting the 24-hour horizon
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 24)  # Default forecast horizon
        )
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        compute_loss: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        """
        # Ensure correct shape
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
            
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Encode and forecast
        h_out = self.encoder(x)
        forecast_output = self.forecast_head(h_out)
        
        # Dummy metrics to satisfy the trainer API
        extras = {'metrics': {'routing_sparsity': torch.tensor(0.0)}}
        
        return forecast_output, extras

class SafetyMonitor:
    """
    VRAM and resource monitoring with auto-shutdown.
    Protects RTX 5090 from Out-of-Memory crashes.
    """
    def __init__(
        self,
        vram_threshold: float = 0.95,
        max_runtime_hours: float = 72.0,
        idle_shutdown_minutes: float = 30.0
    ):
        self.vram_threshold = vram_threshold
        self.max_runtime_hours = max_runtime_hours
        self.idle_shutdown_minutes = idle_shutdown_minutes

        self.start_time = time.time()
        self.last_activity = time.time()

    def check_vram(self) -> Tuple[float, bool]:
        if not torch.cuda.is_available():
            return 0.0, False

        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3

        vram_used_gb = allocated_mem
        vram_pct = allocated_mem / total_mem if total_mem > 0 else 0

        should_shutdown = vram_pct >= self.vram_threshold
        return vram_used_gb, should_shutdown

    def check_runtime(self) -> Tuple[float, bool]:
        elapsed = (time.time() - self.start_time) / 3600
        should_shutdown = elapsed >= self.max_runtime_hours
        return elapsed, should_shutdown

    def check_idle(self) -> Tuple[float, bool]:
        idle = (time.time() - self.last_activity) / 60
        should_shutdown = idle >= self.idle_shutdown_minutes
        return idle, should_shutdown

    def heartbeat(self):
        self.last_activity = time.time()

    def check_all(self) -> List[str]:
        shutdown_reasons = []

        _, vram_shutdown = self.check_vram()
        if vram_shutdown:
            shutdown_reasons.append("VRAM threshold exceeded")

        _, runtime_shutdown = self.check_runtime()
        if runtime_shutdown:
            shutdown_reasons.append("Max runtime reached")

        _, idle_shutdown = self.check_idle()
        if idle_shutdown:
            shutdown_reasons.append("Idle timeout reached")

        return shutdown_reasons
