# Continual-Learning MoE — Monterey Bay Ocean Forecasting

A streaming, continually-learning **Mixture-of-Experts (MoE)** time-series model for
Monterey Bay (MBARI) ocean data. It consumes a stream of hourly observations and
forecasts the next 24 hours, using **Elastic Weight Consolidation (EWC)** and an
**experience-replay buffer** to carry knowledge forward — designed to run on a single
RTX 5090.

> ### Results, up front
> On the 24 Monterey Bay hourly series, this model **does not beat a seasonal-naive
> (same-hour-yesterday) baseline**: it wins on **2 / 24** series, median skill **−0.45**.
> Its ~**0.65%** MAPE looks strong only until you notice naive scores ~**0.56%**.
> This is published for transparency, reuse, and as a worked example of *honest* model
> evaluation — **not** as a state-of-the-art forecasting result. Full numbers and method:
> **[RESULTS.md](RESULTS.md)**.

## Why it exists

Monterey Bay's moorings emit continuous hourly oceanographic series (temperature,
salinity, pressure, …). The goal here was a **lifelong learner**: one that ingests new
data as a stream and adapts online — without retraining from scratch and without
*catastrophically forgetting* earlier regimes. The MoE supplies capacity and per-pattern
specialization; EWC and replay are the anti-forgetting machinery.

It serves a second purpose too: a candid case study in how a large, modern-looking model
can still lose to one line of arithmetic — and the discipline needed to catch that.

## Architecture

```
 past window (default 168 h of y)
        │   per-series, per-window z-score
        ▼
 input projection ─► [ Dynamic-MoE block ] ×3 ─► mean-pool ─► forecast head ─► next 24 h (ŷ)
                         │  learned top-k router
                         │  + always-on "regular" experts (stability)
                         ▼
        EWC penalty  +  experience replay (reservoir, size 5000)   ← continual-learning stack
```

- **Dynamic MoE** (`core.py`) — 3 stacked blocks; each routes every token to the top
  ~25% of experts via a learned router, alongside two always-on "regular" experts for
  stability. Expert FFNs are SwiGLU-style (`w1/w2/w3` + GELU). Routing/expert-usage
  metrics are tracked for load balancing.
- **Elastic Weight Consolidation** (`core.py`) — a Fisher-information-weighted penalty on
  weight drift (default elasticity 1000) to resist catastrophic forgetting across tasks.
- **Experience replay** (`core.py`) — a 5,000-sample buffer using **reservoir sampling**
  (O(1) insertion), replacing an earlier O(N²) diversity-checked design.
- **SafetyMonitor** (`core.py`) — auto-shutdown at 80% VRAM, a max-runtime cap (72 h
  default) and an idle timeout, with a per-step heartbeat — built to keep long lifelong
  runs from OOM-ing or running away on the 32 GB card.
- **TFT backbone** (`tft.py`) — a Temporal Fusion Transformer (temporal attention, causal
  masking) is available as an alternative sequence encoder.

At the production settings (32 experts × 3 blocks, `hidden_dim` 1024) the model is
**~1.3 B parameters** — see *Known limitations* for why that number is larger than it
should be.

## How it trains

- **bfloat16, not fp16** (`trainer.py`). bf16 keeps fp32's exponent range, so a
  large-magnitude (unnormalized) target batch can't overflow to `inf → NaN` — fp16 did
  exactly that and diverged a prior run around step ~800. bf16 also needs no `GradScaler`.
- **NaN/Inf guard** — a pathological batch is skipped rather than allowed to poison the
  weights.
- **Gradient accumulation** with global-step–gated optimizer steps, gradient clipping, and
  periodic checkpointing.
- **VRAM-safe config matters.** The MoE is dense (below), so the documented "big" defaults
  (batch 8 / context 720) **OOM a 32 GB GPU**. The validated-safe configuration is
  `--batch-size 2 --context-window 168`, which peaks around **75%** VRAM.

## The data it learns from

Hourly Monterey Bay oceanographic series stored as partitioned Parquet (lakehouse gold
layer). The loader (`data.py`):

- builds a clean per-series timeline (sorted, timestamp-deduplicated),
- forecasts the **actual observed value** autoregressively — **past `y` → future `y`** —
  *not* timestamps and *not* another model's prediction column,
- z-scores each window using its own context statistics (and de-normalizes at evaluation),
- skips series shorter than `context_window + horizon`.

Default context window is 168 h (7 days); 720 h (30 days) is the design target but does not
fit at the current density. The data itself is **not** distributed with this repo (see
[`../DATA.md`](../DATA.md)).

## Quick start

```bash
pip install -r requirements.txt

# 1) Smoke test on a few series (needs the gold parquet; see ../DATA.md)
python sota_continual_learning/smoke_test_production.py --steps 20 --context-window 168 --batch-size 2

# 2) Production training run (VRAM-safe config)
python sota_continual_learning/run_production.py --total-steps 10000 --batch-size 2 --context-window 168

# 3) Honest evaluation vs seasonal-naive (needs a trained checkpoint)
python sota_continual_learning/evaluate.py \
    --checkpoint sota_continual_learning/output_production/checkpoints/final.pt
```

Trained checkpoints and data are not shipped; train first, then evaluate.

## Results

| Metric | Model | Seasonal-naive |
|---|---|---|
| Series won (of 24) | **2** | 22 |
| Median skill vs naive | **−0.45** | — |
| Median MAPE | 0.65% | **0.56%** |

The low absolute MAPE is the trap: on these near-stationary, diurnal series,
same-hour-yesterday is already very strong, and the model loses on 22 of 24 series. The
evaluation is even *in-sample-favorable* (training streamed these windows) and still loses,
so a strict holdout would be no kinder. See **[RESULTS.md](RESULTS.md)**.

## Engineering notes (what the build taught us)

These are the load-bearing lessons, kept here because they generalize:

1. **Benchmark against seasonal-naive before claiming anything.** A 0.65% MAPE felt like a
   win; the trivial baseline at 0.56% says otherwise. Absolute error is not skill.
2. **Verify the data the model actually receives** — not merely that the pipeline runs. An
   earlier version fed *normalized timestamps* as input and the *wrong target column*; the
   loss looked fine and meant nothing.
3. **Confirm the optimizer is really stepping.** A step counter that was never incremented
   silently disabled all learning while training "ran" cleanly.
4. **Prefer bf16 over fp16 for large-scale / unnormalized regression targets** — the wider
   exponent range avoids `inf → NaN` divergence and needs no loss scaling.
5. **"Dense MoE" is a budget trap** (see below).

## Known limitations

- **The MoE is dense, not sparse.** Every expert is currently evaluated for every token, so
  a nominally-sparse design behaves like a ~1.3 B-parameter dense model — that's what
  drives the VRAM/checkpoint cost and blocks the 720 h context window. Genuine top-k
  routing (compute only selected experts) is the main lever to make the design live up to
  its name.
- **EWC isn't wired into the production train step yet.** The regularizer exists, but the
  production loop logs `EWC: 0.0000` — so the model trains, but is not yet doing
  continual-learning consolidation on that path.
- **Checkpoints are large.** Model + optimizer + scaler state is ~14.7 GB each; use a
  generous `--eval-interval` and prune intermediates.

## Repository layout

| File | Purpose |
|---|---|
| `core.py` | MoE, router, EWC, replay buffer, `ContinualLearner`, `SafetyMonitor` |
| `trainer.py` | bf16 training loop, NaN guard, grad accumulation, checkpointing |
| `data.py` | `LakehouseDataLoader` — streaming past-`y` → future-`y` windows |
| `run_production.py` | production training entry point (CLI) |
| `evaluate.py` | forecast skill vs seasonal-naive, in physical units |
| `smoke_test_production.py` | quick end-to-end check on a few series |
| `main.py` / `startup.py` | online continual-learning loop / bootstrap |
| `tft.py`, `monitor.py` | TFT backbone; metrics/monitoring helpers |
| `RESULTS.md` | the honest evaluation writeup |

## License & data

Code is **Apache-2.0** (see repository `LICENSE`). Data has separate terms and is not
included — see [`../DATA.md`](../DATA.md). Part of the
[Monterey Bay AI Lab](../README.md) project, which enforces a best-of-naive promotion gate
in code (`AGENTS.md`).
