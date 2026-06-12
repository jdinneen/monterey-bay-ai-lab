"""
MBAL SOTA Utilities for Continual Learning and Safety Monitoring.

Extracted and adapted from the experimental sota_continual_learning module.
Includes:
- Mixture of Experts (MoE) components
- Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
- Safety Monitoring for VRAM and runtime protection
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer."""
    num_experts: int = 16
    hidden_dim: int = 2048
    expert_hidden_dim: int = 4096
    num_regular_experts: int = 2  # Always-active experts for stability
    capacity_factor: float = 1.25
    routing_dropout: float = 0.1

@dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation."""
    elasticity: float = 1000.0
    fisher_samples: int = 100
    subset_fraction: float = 0.1

class DynamicRouter(nn.Module):
    """
    Learnable dynamic router that adapts routing weights based on input.
    Uses top-k routing with load balancing.
    """
    
    def __init__(self, dim: int, num_experts: int, capacity_factor: float):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        self.routing_proj = nn.Linear(dim, num_experts)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to top-k experts.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            routing_weights: Weights for each expert (batch, seq, num_experts)
            mask: Binary mask indicating which experts are selected
        """
        bsz, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.dim)
        scores = self.routing_proj(x_flat)
        
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # Route to top 25% of experts
        k = max(1, self.num_experts // 4)
        topk_weights, topk_indices = torch.topk(probs, k, dim=-1)
        
        mask = torch.zeros_like(probs).scatter_(
            dim=1, index=topk_indices, src=torch.ones_like(topk_weights)
        )
        
        routing_weights = probs * mask
        row_sums = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / (row_sums + 1e-8)
        
        return (
            routing_weights.view(bsz, seq_len, self.num_experts),
            mask.view(bsz, seq_len, self.num_experts)
        )

class ExpertFeedForward(nn.Module):
    """Expert-specific feed-forward network."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gelu(self.w1(x)) * self.w3(x))

class DynamicMoE(nn.Module):
    """
    Dynamic Mixture of Experts with adaptive routing.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.experts = nn.ModuleList([
            ExpertFeedForward(config.hidden_dim, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        ])
        
        self.regular_experts = nn.ModuleList([
            ExpertFeedForward(config.hidden_dim, config.expert_hidden_dim)
            for _ in range(config.num_regular_experts)
        ])
        
        self.router = DynamicRouter(
            config.hidden_dim,
            config.num_experts,
            config.capacity_factor
        )
        
        self.register_buffer("expert_counts", torch.zeros(config.num_experts, dtype=torch.long))
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        routing_weights, mask = self.router(x)
        
        x_regular = x.clone()
        for expert in self.regular_experts:
            x_regular = x_regular + expert(x_regular) * 0.5
        
        x_moe = torch.zeros_like(x)
        expert_usage = routing_weights.sum(dim=[0, 1])
        
        for i, expert in enumerate(self.experts):
            weights_i = routing_weights[:, :, i].unsqueeze(-1)
            if weights_i.any():
                x_moe = x_moe + weights_i * expert(x)
        
        output = x_regular + x_moe
        
        if self.training:
            self.expert_counts += expert_usage.long()
            self.total_tokens += bsz * seq_len
        
        metrics = {
            'expert_usage': expert_usage / (expert_usage.sum() + 1e-8),
            'routing_sparsity': (mask.sum() > 0).float().mean(),
            'expert_balance_std': expert_usage.std() / (expert_usage.mean() + 1e-8)
        }
        
        return output, metrics

class ElasticWeightConsolidation:
    """
    EWC implementation to prevent catastrophic forgetting.
    """
    
    def __init__(self, model: nn.Module, config: EWCConfig):
        self.model = model
        self.config = config
        self.prior_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.task_count = 0
    
    def compute_fisher(self, dataloader: Any, device: str = 'cuda') -> None:
        self.model.eval()
        self.fisher_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_info[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('x'))
            else:
                inputs = batch[0]
            
            inputs = inputs.to(device)
            outputs = self.model(inputs)
            
            if isinstance(outputs, tuple):
                loss = outputs[0].mean()
            else:
                loss = outputs.mean()
            
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += (param.grad.data ** 2)
            
            sample_count += 1
            if sample_count >= self.config.fisher_samples:
                break
        
        for name in self.fisher_info:
            self.fisher_info[name] /= (sample_count + 1e-8)
        
        self.model.train()
    
    def compute_consolidation_loss(self, device: str = 'cuda') -> torch.Tensor:
        if not self.prior_params or not self.fisher_info:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = torch.tensor(0.0, device=device)
        for name, param in self.model.named_parameters():
            if name in self.prior_params and name in self.fisher_info:
                diff = param - self.prior_params[name]
                loss += (self.config.elasticity * self.fisher_info[name] * diff.pow(2)).sum()
        
        return loss / (self.task_count + 1)
    
    def register_task(self, dataloader: Any, device: str = 'cuda') -> None:
        self.task_count += 1
        self.prior_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.compute_fisher(dataloader, device)

    def save_state(self, path: str | Path) -> None:
        torch.save({
            'prior_params': self.prior_params,
            'fisher_info': self.fisher_info,
            'task_count': self.task_count
        }, path)

    def load_state(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location='cpu')
        self.prior_params = checkpoint['prior_params']
        self.fisher_info = checkpoint['fisher_info']
        self.task_count = checkpoint['task_count']

class SafetyMonitor:
    """
    VRAM and resource monitoring with auto-shutdown.
    """

    def __init__(
        self,
        vram_threshold: float = 0.80,
        max_runtime_hours: float = 72.0,
        idle_shutdown_minutes: float = 30.0
    ):
        self.vram_threshold = vram_threshold
        self.max_runtime_hours = max_runtime_hours
        self.idle_shutdown_minutes = idle_shutdown_minutes

        self.start_time = time.time()
        self.last_activity = time.time()

    def check_vram(self) -> Tuple[float, bool]:
        """
        Check VRAM usage across the ENTIRE system.
        
        Returns:
            (vram_used_gb, should_shutdown)
        """
        if not torch.cuda.is_available():
            return 0.0, False

        # mem_get_info() returns (free_bytes, total_bytes) for the whole GPU
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        
        total_gb = total_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        used_gb = total_gb - free_gb
        
        vram_pct = used_gb / total_gb if total_gb > 0 else 0
        should_shutdown = vram_pct >= self.vram_threshold

        status_str = f"VRAM: {used_gb:.1f}/{total_gb:.1f} GB ({vram_pct*100:.1f}%)"
        if should_shutdown:
            logger.warning(f"{status_str} [SHUTDOWN THRESHOLD EXCEEDED]")
        else:
            logger.info(status_str)

        return used_gb, should_shutdown

    def check_runtime(self) -> Tuple[float, bool]:
        elapsed = (time.time() - self.start_time) / 3600
        should_shutdown = elapsed >= self.max_runtime_hours
        logger.info(f"Runtime: {elapsed:.1f}/{self.max_runtime_hours:.1f} hours")
        return elapsed, should_shutdown

    def check_idle(self) -> Tuple[float, bool]:
        idle = (time.time() - self.last_activity) / 60
        should_shutdown = idle >= self.idle_shutdown_minutes
        logger.info(f"Idle time: {idle:.1f}/{self.idle_shutdown_minutes:.1f} minutes")
        return idle, should_shutdown

    def heartbeat(self) -> None:
        self.last_activity = time.time()

    def check_all(self) -> List[str]:
        reasons = []
        if self.check_vram()[1]:
            reasons.append("VRAM threshold exceeded")
        if self.check_runtime()[1]:
            reasons.append("Max runtime reached")
        if self.check_idle()[1]:
            reasons.append("Idle timeout reached")
        return reasons
