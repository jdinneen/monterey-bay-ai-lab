"""
Monitoring and visualization dashboard for the continual learning system.

Real-time tracking of:
- Training metrics (loss, expert usage)
- Model architecture selection
- GPU memory utilization
- Learning rate schedules

Optimized for RTX 5090's monitor capabilities.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import json
import psutil


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for continual learning.
    
    Features:
    - Training loss and metrics plotting
    - GPU memory and utilization tracking
    - Expert usage distribution visualization
    - Architecture selection history
    
    Designed to run alongside training without blocking.
    """
    
    def __init__(self, save_dir: str = 'monitoring'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.step_times: List[float] = []
        self.expert_usage_history: List[np.ndarray] = []
        self.architecture_history: List[str] = []
        self.gpu_memory_used: List[float] = []
        
        # Timing
        self.start_time = time.time()
        self.last_update = 0
        
        plt.ion()  # Interactive mode
    
    def log_train_step(
        self,
        loss: float,
        expert_usage: Optional[torch.Tensor] = None,
        step_time: Optional[float] = None
    ) -> None:
        """Log a training step."""
        self.train_losses.append(loss)
        
        if expert_usage is not None:
            self.expert_usage_history.append(expert_usage.cpu().numpy())
        
        if step_time is not None:
            self.step_times.append(step_time)
    
    def log_validation(self, val_loss: float) -> None:
        """Log validation result."""
        self.val_losses.append(val_loss)
    
    def log_architecture(self, arch_name: str) -> None:
        """Log architecture selection."""
        self.architecture_history.append(arch_name)
    
    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory and utilization stats."""
        if not torch.cuda.is_available():
            return {}
        
        gpu_stats = {
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
            'utilization': torch.cuda.utilization() if hasattr(torch, 'cuda') and 
                          hasattr(torch.cuda, 'utilization') else 0
        }
        
        return gpu_stats
    
    def update(self, interval: int = 5) -> None:
        """Update dashboard display."""
        current_time = time.time()
        
        # Update at most every `interval` seconds
        if current_time - self.last_update < interval:
            return
        
        self.last_update = current_time
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Continual Learning Monitor | Step: {len(self.train_losses)} | '
                    f'Time: {time.time() - self.start_time:.0f}s', fontsize=14)
        
        # Plot 1: Training loss
        if len(self.train_losses) > 0:
            axes[0, 0].plot(self.train_losses)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
        
        # Plot 2: Validation loss
        if len(self.val_losses) > 0:
            axes[0, 1].plot(self.val_losses)
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
        
        # Plot 3: Step time
        if len(self.step_times) > 0:
            axes[0, 2].plot(self.step_times[-100:])  # Last 100 steps
            axes[0, 2].set_title('Step Time (recent)')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Time (s)')
        
        # Plot 4: Expert usage distribution
        if len(self.expert_usage_history) > 0:
            latest_usage = self.expert_usage_history[-1]
            n_experts = len(latest_usage)
            x = np.arange(n_experts)
            axes[1, 0].bar(x, latest_usage)
            axes[1, 0].set_title(f'Expert Usage (n={n_experts})')
            axes[1, 0].set_xlabel('Expert ID')
            axes[1, 0].set_ylabel('Usage Fraction')
        
        # Plot 5: Architecture history
        if len(self.architecture_history) > 0:
            arch_counts = {}
            for a in self.architecture_history:
                arch_counts[a] = arch_counts.get(a, 0) + 1
            
            x = list(arch_counts.keys())
            y = list(arch_counts.values())
            axes[1, 1].bar(x, y)
            axes[1, 1].set_title('Architecture Selection History')
            axes[1, 1].set_xlabel('Architecture')
            axes[1, 1].set_ylabel('Count')
        
        # Plot 6: GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            labels = ['Memory Allocated (GB)', 'Memory Reserved (GB)']
            values = [gpu_stats['memory_allocated'], gpu_stats['memory_reserved']]
            axes[1, 2].bar(labels, values)
            axes[1, 2].set_title('GPU Memory Usage')
            for i, v in enumerate(values):
                axes[1, 2].text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def save_snapshot(self, name: str = 'snapshot') -> None:
        """Save current dashboard state."""
        snapshot = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'expert_usage_history': self.expert_usage_history,
            'architecture_history': self.architecture_history,
            'total_steps': len(self.train_losses),
            'elapsed_time': time.time() - self.start_time
        }
        
        path = self.save_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        print(f"Saved dashboard snapshot to {path}")
    
    def close(self):
        """Clean up and save final state."""
        plt.ioff()
        self.save_snapshot('final')
        plt.close('all')


class GPUResourceMonitor:
    """
    Continuous monitoring of GPU resources.
    
    Tracks VRAM usage, temperature, and utilization for RTX 5090 optimization.
    """
    
    def __init__(self):
        self.usage_history: List[Dict] = []
        self.temperature_history: List[float] = []
    
    def sample(self) -> Dict:
        """Take a sample of GPU stats."""
        if not torch.cuda.is_available():
            return {}
        
        # Get memory info
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Sample temperature (if available via nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            temp = float(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            temp = 0.0
        
        stats = {
            'mem_allocated': mem_allocated,
            'mem_reserved': mem_reserved,
            'mem_total': mem_total,
            'mem_utilization': mem_allocated / mem_total if mem_total > 0 else 0,
            'temperature': temp,
            'timestamp': time.time()
        }
        
        self.usage_history.append(stats)
        self.temperature_history.append(temp)
        
        return stats
    
    def get_summary(self) -> Dict:
        """Get summary of GPU resource usage."""
        if not self.usage_history:
            return {}
        
        mem_allocs = [h['mem_allocated'] for h in self.usage_history]
        temps = [t for t in self.temperature_history if t > 0]
        
        return {
            'peak_memory_gb': max(mem_allocs),
            'avg_memory_gb': sum(mem_allocs) / len(mem_allocs),
            'current_memory_gb': mem_allocs[-1] if mem_allocs else 0,
            'peak_temperature': max(temps) if temps else 0,
            'avg_temperature': sum(temps) / len(temps) if temps else 0
        }


if __name__ == '__main__':
    # Quick demo
    dashboard = MonitoringDashboard()
    
    # Simulate some training
    for step in range(50):
        loss = np.exp(-step/10) + np.random.randn() * 0.05
        expert_usage = torch.softmax(torch.randn(8), dim=0)
        
        dashboard.log_train_step(loss, expert_usage)
        
        if step % 10 == 0:
            dashboard.update()
        
        time.sleep(0.1)
    
    print("Demo complete!")
    dashboard.close()
