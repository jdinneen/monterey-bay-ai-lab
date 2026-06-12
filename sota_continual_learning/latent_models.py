"""
Latent Representation Models for Oceanographic Time Series.

Implements the "LatentTSF" paradigm:
1. Compresses noisy raw sensor data into an abstract latent vector.
2. Feeds the abstract vector to the forecasting model (MoE).

This is designed to be fully modular and additive. It wraps the existing 
ContinualLearner without requiring modifications to the core logic.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

class Conv1DEncoder(nn.Module):
    """Zero-Bloat 1D-CNN Encoder for extracting abstract physics from raw sensors."""
    def __init__(self, seq_len: int = 720, latent_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Fast, hierarchical temporal compression
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            flat_size = self.encoder(dummy).shape[1]
            
        self.fc_encode = nn.Linear(flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input from DataLoader: (Batch, SeqLen, 1)
        # Conv1d requires channels first: (Batch, 1, SeqLen)
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.fc_encode(h)

class LatentMoEPipeline(nn.Module):
    """
    Google-Quality Adapter Pattern.
    Wraps the Encoder and the MoE Learner into a single end-to-end module.
    """
    def __init__(self, encoder: nn.Module, moe_learner: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.moe_learner = moe_learner
        
    def forward(self, x: torch.Tensor, compute_loss: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 1. Project raw context window into the abstract latent space
        z = self.encoder(x)
        
        # 2. Pass the latent representation to the MoE Continual Learner
        # The MoE operates entirely on abstract signals to predict the physical horizon
        return self.moe_learner(z, compute_loss=compute_loss)

    def continual_step(self, *args, **kwargs):
        """Pass-through for continual learning methods."""
        return self.moe_learner.continual_step(*args, **kwargs)

    def register_new_task(self, *args, **kwargs):
        """Pass-through for continual learning methods."""
        return self.moe_learner.register_new_task(*args, **kwargs)
