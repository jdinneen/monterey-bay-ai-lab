# SOTA Continual Learning System for RTX 5090

> **Honest framing (read first).** On this project's actual data (M1/M2 hourly mooring
> physics) this continual-learning / MoE track is the **persistence-ceiling backbone — a
> documented negative result**, not the predictive headline. The series are
> persistence-dominated (acf_1h ≈ 0.99): the best models realize only single-digit %
> skill over a naive "same as last hour" baseline, and in the in-sample sanity eval this
> stack does not beat naive (see `evaluate.py` and `../research/model_lab/HONEST_SKILL_BASELINE.md`).
> The project's predictive headline is **statewide bacterial-exceedance prediction** (see the
> repo root `README.md` and `MODEL_SUITE.md`). The architecture below is described on its own
> engineering merits; it is not a claim of forecasting skill on this data.

A state-of-the-art, continuously learning ML architecture designed to leverage the full power of RTX 5090.

## Architecture Overview

### Core Components

1. **Dynamic Mixture-of-Experts (MoE)**
   - 16+ experts with dynamic routing
   - Expert specialization via clustering  
   - Load balancing and routing optimization

2. **Continual Learning Stack**
   - Elastic Weight Consolidation (EWC)
   - Experience replay buffer with Reservoir Sampling
   - Task-incremental learning (class/function separation)

3. **Optimization Engine**
   - Full FP16/bfloat16 support for RTX 5090
   - Gradient checkpointing for memory efficiency
   - Distributed training ready (DDP/FSDP)

### Learning Modes

- **Online Learning**: Stream data continuously, learning without pauses
- **Active Learning**: Query high-value samples for labels
- **Meta-Learning**: Optimize its own architecture over time
- **Self-Supervised**: Pre-train on unlabelled data, fine-tune on labelled

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run continual learning training
python main.py --mode online --data-path /path/to/data

# Monitor learning progress
python monitor.py  # TensorBoard dashboard

#Export optimized model
python export.py --checkpoint checkpoints/latest.pt
```

## Key Features

- **Zero-Catastrophic-Forgetting**: EWC + replay buffers preserve prior knowledge
- **Scalable to 5090**: Efficiently uses 24GB+ VRAM with MoE architecture
- **Self-Improving**: Meta-learning loop continuously optimizes hyperparameters
- **Multi-Task Ready**: Same architecture for seq prediction, classification, regression

## Research Background

Based on SOTA approaches:
- DeepMoE: Dynamic routing for large-scale experts (arXiv:2406.05397)
- i-DROPS: Incremental learning with drop-based regularization
- Self-Supervised Replay: Generating synthetic samples to preserve knowledge
- Adaptive Gradient Meta-Learning: Learning rate and architecture optimization
