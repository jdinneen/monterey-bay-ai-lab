# Continual-Learning MoE (Monterey Bay AI Lab)

A continuously-learning Mixture-of-Experts (MoE) time-series model with Elastic
Weight Consolidation (EWC) and experience replay, built to run on a single
RTX 5090.

> **Results, up front (see [RESULTS.md](RESULTS.md)).** On the 24 Monterey Bay
> hourly series this model **does not beat a seasonal-naive (same-hour-yesterday)
> baseline** — it wins on only 2/24 series, median skill -0.45. The low absolute
> MAPE (~0.65%) is misleading because naive already scores ~0.56%. This code is
> published for transparency and reuse, **not** as a state-of-the-art forecasting
> claim.

## Architecture
- **Dynamic Mixture-of-Experts** (`core.py`): expert feed-forward networks with a
  learned top-k router plus always-on "regular" experts for stability.
- **Continual-learning stack**: Elastic Weight Consolidation (EWC) and an
  experience-replay buffer (reservoir sampling), intended to reduce catastrophic
  forgetting across tasks.
- **Training** (`trainer.py`): mixed precision (bfloat16) and gradient
  accumulation, a launch-time VRAM admission guard (`ops/gpu_admission.py`), and a
  runtime `SafetyMonitor` (VRAM / runtime / idle limits).

## Quick start
```bash
pip install -r requirements.txt

# Smoke test on a few series (needs the gold parquet; see ../DATA.md)
python sota_continual_learning/smoke_test_production.py --steps 20 --context-window 168 --batch-size 2

# Production training run
python sota_continual_learning/run_production.py --total-steps 10000 --batch-size 2 --context-window 168

# Honest evaluation vs seasonal-naive (needs a trained checkpoint)
python sota_continual_learning/evaluate.py --checkpoint sota_continual_learning/output_production/checkpoints/final.pt
```

Data and trained checkpoints are **not** distributed with this repo (see
`../DATA.md`); train first, then evaluate.

## Methods
This combines established techniques rather than novel ones:
- **Sparse Mixture-of-Experts** routing (top-k expert selection).
- **Elastic Weight Consolidation** (EWC) for continual learning
  (Kirkpatrick et al., *PNAS* 2017).
- **Experience replay** via reservoir sampling.

See [RESULTS.md](RESULTS.md) for the measured outcome and reproduction notes.
