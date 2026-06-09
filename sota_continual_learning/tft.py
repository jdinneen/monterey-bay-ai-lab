"""
Temporal Fusion Transformer components for Monterey Bay forecasting.

Replaces DynamicMoE with proper time-series attention mechanisms.
Handles multi-variate sequences, diurnal patterns, and seasonal trends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any


class TemporalAttention(nn.Module):
    """
    Multi-head attention with causal masking for temporal forecasting.
    Supports variable-length sequences with temporal encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Causal mask to prevent looking ahead

        Returns:
            output: Attention-transformed features
            attention_weights: For interpretability
        """
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, dim*3)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: (batch, seq, heads, head_dim)

        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Apply causal mask if provided
        if mask is not None:
           (mask == float("-inf")).expand_as(attn)
            
        # Softmax for attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        out = out.view(bsz, seq_len, self.dim)  # (batch, seq, dim)

        out = self.out_proj(out)
        return out, attn


class TemporalFusionBlock(nn.Module):
    """
    TFT block with temporal attention + feature gating.
    Combines historical patterns with static covariates.
    """

    def __init__(self, dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Temporal attention
        self.attn = TemporalAttention(dim, num_heads, dropout)

        # Feature gating network
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            output: Processed features
            attention_weights: For interpretability
        """
        # Temporal attention with residual
        attn_out, attn = self.attn(self.norm1(x))

        # Gating mechanism - learn which features to emphasize
        gate_out = self.gate(attn_out)
        x = attn_out + gate_out

        # Output projection with residual
        out = self.output_proj(self.norm2(x))

        return out + x, attn


class TemporalEmbedding(nn.Module):
    """
    Embed temporal information including time of day, day of week, etc.
    """

    def __init__(self, numeric_dim: int, embed_dim: int, max_time_values: int = 1000):
        super().__init__()
        self.numeric_proj = nn.Linear(numeric_dim, embed_dim)
        self.time_embed = nn.Parameter(torch.randn(max_time_values, embed_dim))

    def forward(
        self,
        numeric_features: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine numeric features with temporal embeddings.

        Args:
            numeric_features: (batch, seq, numeric_dim)
            time_indices: Optional (batch, seq) - indices for temporal embedding

        Returns:
            embedded: (batch, seq, embed_dim)
        """
        # Project numeric features
        numerical = self.numeric_proj(numeric_features)

        if time_indices is not None:
            # Add learned temporal encoding
            temporal = self.time_embed[time_indices.clamp(max=len(self.time_embed) - 1)]
            return numerical + temporal

        return numerical


class TFTBackbone(nn.Module):
    """
    Main Temporal Fusion Transformer backbone.
    Processes multi-variate time series with temporal attention.
    """

    def __init__(
        self,
        input_dim: int = 768,
        seq_len: int = 24,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # Initial embedding
        self.embed = nn.Linear(input_dim, hidden_dim)

        # Temporal fusion blocks
        self.blocks = nn.ModuleList([
            TemporalFusionBlock(hidden_dim, hidden_dim * 2, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through TFT backbone.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            output: Processed features
            metrics: Dictionary of attention statistics
        """
        batch_size, seq_len, _ = x.shape

        # Embed input
        x = self.embed(x)  # (batch, seq, hidden_dim)

        all_attn_weights = []
        for i, block in enumerate(self.blocks):
            x, attn = block(x)
            if attn is not None:
                all_attn_weights.append(attn.mean(dim=1))  # Average over heads

        # Normalized output
        out = self.norm(x)

        metrics = {
            'attn_weights': torch.stack(all_attn_weights) if all_attn_weights else None,
            'output_std': out.std().item()
        }

        return out, metrics


class TFTForecaster(nn.Module):
    """
    Complete Temporal Fusion Transformer for forecasting.
    Handles multi-variate sequences with temporal and static features.
    """

    def __init__(
        self,
        input_dim: int = 768,
        seq_len: int = 24,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        forecast_horizon: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        # TFT backbone
        self.backbone = TFTBackbone(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Forecast head - outputs predictions for future horizon
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )

    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = False,
        target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional loss computation.

        Args:
            x: Input sequences (batch, seq_len, input_dim)
            compute_loss: Whether to return loss
            target: Ground truth for next horizon (batch, forecast_horizon, ...)

        Returns:
            predictions: Forecasts for future horizon
            extras: Dictionary of outputs and metrics
        """
        batch_size = x.shape[0]

        # Process through backbone
        features, all_metrics = self.backbone(x)

        # Use last time step for forecasting
        last_feature = features[:, -1, :]  # (batch, hidden_dim)

        # Generate forecasts for future horizon
        predictions = self.forecast_head(last_feature)  # (batch, forecast_horizon)

        extras = {'metrics': all_metrics}

        if compute_loss and target is not None:
            extras['loss'] = F.mse_loss(predictions, target)
            extras['predictions'] = predictions

        return predictions, extras


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for robust regression."""
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return (0.5 * quadratic.pow(2) + delta * linear).mean()


def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean absolute percentage error."""
    diff = (pred - target).abs()
    denom = target.abs() + eps
    return (diff / denom).mean()
