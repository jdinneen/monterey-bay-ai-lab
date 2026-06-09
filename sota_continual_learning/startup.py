#!/usr/bin/env python3
"""
Quick start guide and demonstration for the SOTA Continual Learning System.

This script:
1. Checks GPU availability (RTX 5090)
2. Runs a simple demo of continuous learning
3. Shows how to use all components

Usage: python startup.py [--duration SECONDS] [--demo]
"""

import torch
import sys
from pathlib import Path


def print_banner():
    """Print ASCII art banner."""
    print("\n" + "="*60)
    print("  SOTA CONTINUAL LEARNING SYSTEM FOR RTX 5090")
    print("="*60)
    print()


def check_gpu():
    """Check GPU availability and specs."""
    if not torch.cuda.is_available():
        print("[WARNING] No CUDA device found!")
        return None
    
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    sm = torch.cuda.get_device_capability(0)
    
    print(f"GPU: {device_name}")
    print(f"VRAM: {total_mem:.1f} GB")
    print(f"Compute Capability: {sm[0]}.{sm[1]}")
    
    # Check if this is an RTX 5090 (or similar high-end)
    if "RTX 5090" in device_name or "4090" in device_name:
        print("[OK] High-end GPU detected - optimal for MoE training!")
    elif total_mem >= 16:
        print(f"[OK] {total_mem:.0f}GB VRAM available for large models")
    
    return device_name


def quick_demo(duration_steps: int = 100):
    """Run a quick learning demo."""
    from core import ContinualLearner
    from data import DataDriftGenerator
    
    print("\n" + "-"*60)
    print("QUICK LEARNING DEMO")
    print("-"*60)
    
    # Initialize with smaller model for stability
    model = ContinualLearner(input_dim=256, hidden_dim=256, num_experts=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    gen = DataDriftGenerator(input_dim=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # Lower LR
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Running for {duration_steps} steps...\n")
    
    # Training loop
    losses = []
    for step in range(duration_steps):
        batch_size = 32
        batch = gen.generate_batch(batch_size).to(device)
        
        optimizer.zero_grad()
        out, extras = model(batch, compute_loss=True)
        loss = extras['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / len(losses[-20:])
            print(f"Step {step+1:4d}: Loss = {loss.item():.4f}, Avg(20) = {avg_loss:.4f}")
    
    print("\n[OK] Demo completed!")
    return model, losses


def main():
    """Run startup sequence."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SOTA Continual Learning System - Quick Start"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps in demo"
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Run full training with all features"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check GPU
    device_name = check_gpu()
    
    if not torch.cuda.is_available():
        print("\nRunning in CPU mode - performance will be limited.")
        print("For best results, use CUDA-capable GPU (RTX 5090 recommended).\n")
    
    # Show system capabilities
    print("\n" + "-"*60)
    print("SYSTEM CAPABILITIES:")
    print("-"*60)
    print("✓ Dynamic Mixture-of-Experts with adaptive routing")
    print("✓ Elastic Weight Consolidation (EWC) for zero-forgetting")
    print("✓ Experience replay buffer for continual learning")
    print("✓ Adaptive meta-learning for hyperparameter optimization")
    print("✓ Gradient checkpointing for memory efficiency")
    print("✓ Mixed precision (AMP) training support")
    
    # Run demo
    if args.full_training:
        print("\n" + "="*60)
        print("RUNNING FULL TRAINING...")
        print("="*60)
        
        from trainer import Trainer
        
        model = ContinualLearner(input_dim=256, hidden_dim=512, num_experts=8)
        trainer = Trainer(model, mixed_precision=torch.cuda.is_available())
        
        # Create dummy data
        class DummyDataset:
            def __len__(self): return 1000
            def __getitem__(self, _):
                import torch
                return torch.randn(32, 256)
        
        dataset = DummyDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        history = trainer.fit(loader, num_epochs=1, max_steps=100)
        print("\nTraining complete!")
    else:
        model, losses = quick_demo(args.steps)
        
        # Basic analysis
        if len(losses) >= 20:
            initial_loss = sum(losses[:5]) / 5
            final_loss = sum(losses[-5:]) / 5
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"\n{'='*60}")
            print("RESULTS:")
            print(f"{'='*60}")
            print(f"Initial loss: {initial_loss:.4f}")
            print(f"Final loss:   {final_loss:.4f}")
            print(f"Improvement:  {improvement:.1f}%")
        
        # Save demo model
        script_dir = Path(__file__).parent
        save_path = script_dir / "checkpoints" / "demo_model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nDemo model saved to: {save_path}")
    
    print("\n" + "="*60)
    print("USAGE:")
    print("="*60)
    print("To train continuously on your data:")
    print("  python main.py --mode train")
    print()
    print("For continuous learning (never stops):")
    print("  python example_continuous_learner.py --duration-seconds 3600")
    print()
    print("Monitor training progress:")
    print("  python monitor.py")
    print()


if __name__ == "__main__":
    main()
