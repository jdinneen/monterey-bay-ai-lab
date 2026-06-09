# SOTA Continual Learning — Results on MBARI Data

An honest evaluation of the continual-learning Mixture-of-Experts model in
`sota_continual_learning/` on the Monterey Bay (MBARI) hourly series.

## Setup
- **Model:** dynamic MoE (32 experts x 3 blocks, hidden_dim=1024, ~1.3B params)
  with EWC + experience replay (`core.py`).
- **Task:** 24-hour-ahead forecast of the observed value `y`, autoregressive
  (past `y` -> future `y`), per-window z-scored, across 24 Monterey Bay series.
- **Data:** lakehouse gold `forecast_predictions` (~6.9M rows; not distributed).
- **Baseline:** seasonal-naive (same-hour-yesterday) — the bar required by AGENTS.md.
- **Reproduce:** train first with
  `python sota_continual_learning/run_production.py`, then
  `python sota_continual_learning/evaluate.py --checkpoint <final.pt>`.
  Requires a trained checkpoint (not distributed) and the gold parquet (see
  `../DATA.md`); it will not run out of the box without both.

## Result (honest)
| Metric | Value |
|---|---|
| Series evaluated | 24 |
| Beat seasonal-naive | 2 / 24 (8%) |
| Median skill vs naive | -0.45 |
| Median model MAPE | 0.65% |
| Median naive MAPE | 0.56% |

**The model does not beat the seasonal-naive baseline.** The low absolute MAPE is
misleading: same-hour-yesterday already scores ~0.56% on these near-stationary,
diurnal series, and the model loses on 22 of 24. This is an in-sample-favorable
eval (training streamed these windows), so a strict holdout would be no better.

## Takeaway
These series are persistence/seasonality-ceilinged; a 1.3B-parameter MoE does not
break that ceiling here. The architecture and procedures are published for
transparency and reuse, **not** as a state-of-the-art forecasting claim.
