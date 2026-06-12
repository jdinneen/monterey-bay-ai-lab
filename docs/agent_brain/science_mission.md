# Science Mission

This is the Monterey Bay AI Lab science memory for agents. It explains what the
project is trying to discover and what kinds of automation are valuable. It does not
override `AGENTS.md`, the value gate, traffic control, file locks, tests, or human
approval rules.

## Identity Rule

The project and lab are **Monterey Bay AI Lab**. Agents must not describe the
project as MBAL, MBAL AI, MBAL AI Lab, or MBAL AI Forecasting. MBAL is a
source-data/provider reference only, for example when discussing M1/M2 mooring
datasets, source URLs, legacy module names, or compatibility environment variables.

## Director Model

The human user is the AI Director. Agents are execution engines and scientific critics:
they should move fast on data, math, and evidence, while pushing back on bloat,
unsupported claims, and unsafe automation.

Before implementation, explain the physics or math of the approach and cite the
relevant value-gate result. Follow the current repository approval policy in
`AGENTS.md`.

## Monterey Bay Moat

The mission is to forecast complex coastal dynamics in Monterey Bay, including
upwelling, hypoxia, harmful algal blooms, bacterial exceedances, and related regime
changes.

The moat is not generic regression on raw sensor data. The moat is discovering
second- and third-order signal combinations across dense local data sources:

- M1/M2 moorings.
- Acoustics.
- Satellite products.
- ADCP and flow/current data.
- Weather, waves, rainfall, discharge, and event drivers.

## Preferred Scientific Strategy

1. Ingest dense, multimodal local data.
2. Compress noisy raw windows into latent representations.
3. Search for hidden nonlinear relationships between latent states.
4. Forecast future ocean states in latent space where doing so beats honest baselines.
5. Promote only when the evidence is causal, tested, and reversible or a correctness fix.

## SOTA Blueprint

These are preferred hypotheses, not excuses to overbuild. Each must beat the relevant
baseline and pass the value gate.

- **Latent time-series forecasting:** use autoencoders, such as 1D-CNN encoders, to
  project noisy raw windows into lower-dimensional latent states before forecasting
  when that improves evidence or stability.
- **Mixture of Experts:** use expert specialization for distinct physical regimes such
  as diurnal cycles, storms, El Nino conditions, and anomalous upwelling. Keep expert
  count within measured VRAM limits.
- **Continuous learning / EWC:** learn new seasonal or regime patterns without
  catastrophically forgetting earlier regimes.

## Good Automation

Automate data and math:

- Data ingestion and chunking.
- Polars/vectorized scans.
- CUDA prefetching and pinned-memory loading.
- Gradient accumulation.
- Hyperparameter sweeps.
- Latent correlation hunting.
- Evidence reports and value-gate checks.

Do not automate away scientific anomalies. A loss spike may be real ocean physics, not
a code defect.

## Anti-Patterns

- Do not write agents that automatically rewrite model code when loss spikes.
- Do not hide real anomalies with auto-repair.
- Do not add Kubernetes, heavy governance ledgers, or databases unless a concrete
  value-gate result demands it.
- Do not use "SOTA" as a substitute for baseline evidence.
- Do not promote a model because it is sophisticated; promote it because it beats an
  honest baseline on the right split with the right normalization.

## Hard-Learned Lessons

- **Normalization mismatch:** training target scaling must match evaluator un-scaling.
  Do not train on global-average scaling if evaluation uses local context-window
  un-scaling. Prior fix dropped MAPE from 1.29% to 0.21%.
- **Target outliers:** ocean sensors can flatline. Near-zero variance can make
  normalized targets explode. Clip normalized targets, for example to `[-20, 20]`.
- **MoE sizing:** a 32-expert MoE can OOM a 32 GB RTX 5090. Prefer 8 experts and use
  gradient accumulation, for example `batch_size=2` with `accum_steps=4`, unless a
  measured value-gate result says otherwise.
- **PyTorch allocator:** safety monitors must not trigger shutdowns based on
  `torch.cuda.memory_reserved()`. PyTorch caches aggressively; use
  `torch.cuda.memory_allocated()` for true tensor footprint.
- **I/O starvation:** an RTX 5090 can starve behind single-threaded Pandas loading.
  Prefer vectorized scans, Polars streaming, pinned memory, and CUDA prefetching for
  heavy training input pipelines.
- **NOAA/MUR chunking:** NOAA MUR SST annual requests can time out. Chunk satellite
  fetches by quarter.
