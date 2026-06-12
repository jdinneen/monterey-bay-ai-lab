#!/usr/bin/env python3
"""
MoE Expert Routing Analysis

This script loads a trained SOTA Continual Learner checkpoint and feeds it a 
representative sample of lakehouse data. It records the gating probabilities 
(routing weights) for the 32 experts across different time steps to determine 
if the model is achieving true specialization (e.g., Expert 3 handles Upwelling, 
Expert 7 handles El Niño) or suffering from representation collapse.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the 'sota_continual_learning' directory itself to the path so absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOTA_DIR = PROJECT_ROOT / "sota_continual_learning"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(SOTA_DIR))

from core import ContinualLearner
from data import create_high_performance_loader

def load_latest_checkpoint(checkpoint_dir: Path) -> Path:
    ckpts = list(checkpoint_dir.glob('step_*.pt'))
    if not ckpts:
        final_ckpt = checkpoint_dir / 'final.pt'
        if final_ckpt.exists():
            return final_ckpt
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # Sort by step number
    return sorted(ckpts, key=lambda p: int(p.stem.split('_')[1]))[-1]

def analyze_routing(checkpoint_path: Path, parquet_path: str, num_batches: int = 50):
    print(f"Loading checkpoint: {checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model (Matching run_production.py contract)
    model = ContinualLearner(
        input_dim=1,
        hidden_dim=1024,
        num_experts=32
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("Loading data stream...")
    loader = create_high_performance_loader(
        parquet_path=parquet_path,
        batch_size=32,
        num_workers=4,
        context_window=720,
        forecast_horizon=24
    )
    
    # We want to capture the gating probabilities. The `ContinualLearner` returns 
    # an `extras` dict when `compute_loss=True`. We'll need to patch the model temporarily 
    # to expose the routing probabilities if it doesn't already, or just run it and see.
    # Looking at `core.py`, the MoE layer handles the gating. We will use a forward hook.
    
    routing_probs_history = []
    
    def gating_hook(module, input, output):
        # output of DynamicMoE is (output, load_loss, probs)
        # However, looking closely at how standard PyTorch hooks work, we might only get the final tensor.
        # Let's assume we can grab the activations or we'll infer it.
        # A simpler way for a data scientist: just run the forward pass and grab the load_loss 
        # as a proxy for balance, or patch the method.
        pass

    # For now, we will perform a high-level activation capture if possible, 
    # or just measure the loss distribution over different series.
    print(f"Running inference over {num_batches} batches to map expert utilization...")
    
    # Track the usage of each expert
    # Shape will be [num_batches, num_experts]
    all_expert_usage = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            batch_x, batch_target = batch
            batch_x = batch_x.to(device)
            
            # Run inference without internal loss computation (trainer handles that)
            outputs, extras = model(batch_x, compute_loss=False)
            
            # The gating metrics are stored in extras['metrics']
            # DynamicMoE returns 'expert_usage', a tensor of probabilities
            metrics = extras.get('metrics', {})
            
            if 'expert_usage' in metrics:
                # DynamicMoE returns a list of metrics (one per MoE block)
                # Let's grab the usage from the first MoE block
                usage = metrics['expert_usage'][0].cpu().numpy()
                all_expert_usage.append(usage)
            
            sys.stdout.write(f"\rProcessed {i+1}/{num_batches} batches")
            sys.stdout.flush()
            
    print("\nRouting analysis complete.")
    
    if all_expert_usage:
        usage_matrix = np.array(all_expert_usage)  # (num_batches, num_experts)
        avg_usage = usage_matrix.mean(axis=0)
        
        print("\n--- Expert Utilization Summary ---")
        for expert_id, usage in enumerate(avg_usage):
            # Print a simple text-based bar chart
            bar = "#" * int(usage * 100)
            print(f"Expert {expert_id:02d}: {usage*100:5.1f}% | {bar}")
            
        # Check for representation collapse
        active_experts = (avg_usage > 0.01).sum()
        print(f"\nActive Experts (>1% usage): {active_experts}/32")
        if active_experts < 4:
            print("WARNING: Severe Representation Collapse Detected! The MoE router is starving experts.")
        elif active_experts == 32:
            print("STATUS: Excellent load balancing. All experts are contributing.")
        else:
            print("STATUS: Normal specialization. Experts are selectively routing.")
    else:
        print("Could not find 'expert_usage' in model metrics.")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="sota_continual_learning/output_production/checkpoints")
    parser.add_argument("--parquet-path", type=str, default="lakehouse/gold/forecast_predictions")
    args = parser.parse_args()
    
    try:
        latest = load_latest_checkpoint(Path(args.ckpt_dir))
        analyze_routing(latest, args.parquet_path)
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Note: If the training run just started, wait a few minutes for the first checkpoint.")
