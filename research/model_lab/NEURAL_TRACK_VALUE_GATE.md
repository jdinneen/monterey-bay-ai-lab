# Value-Gate verdict: neural / MoE / EWC / LatentTSF forecasting track

**Date:** 2026-06-10 · **Gate:** `docs/VALUE_GATE.md` · **Decision: REJECTED as a production capability — research-only.**

> Value gate: **REJECTED — vapor/premature.** The MoE/EWC/LatentTSF forecasting stack
> beats the best-naive baseline on **1 of 24 series (4.2%)** with **median skill −0.78 to
> −0.88** (i.e. 78–88% *worse* than same-hour-yesterday / persistence), and that is an
> **in-sample** eval, not a temporal holdout. It does not clear Q1 of the gate (beats a real
> baseline). Evidence: `sota_continual_learning/output_{idle_2500,production}/eval/eval_summary.json`.

## What was evaluated

The continual-learning stack (`sota_continual_learning/`: dense MoE transformer, EWC, replay
buffer) and the in-progress `LatentTSF` encoder (`mbal_neural_forecast.py`,
`research/model_lab/latent_tsf_value_gate.py`), aimed at M1/M2 mooring forecasting.

## The numbers (both trained checkpoints, identical verdict)

| metric | output_idle_2500 | output_production |
|---|--:|--:|
| series beating best-naive | **1 / 24 (4.2%)** | **1 / 24 (4.2%)** |
| median skill vs best-naive | **−0.877** | **−0.785** |
| median skill vs seasonal-naive | −0.566 | −0.417 |
| median skill vs persistence | −0.819 | −0.713 |
| median model MAPE | 1.29% | 0.68% |
| median persistence MAPE | **0.36%** | **0.36%** |
| eval type | in-sample sanity | in-sample sanity |

The model is 2–3.6× *worse* than predicting "same as last hour," and several series numerically
diverged (mean skill −9.8e7 on idle_2500). This is consistent with the independently established
**M1 persistence ceiling** (`reports/PERSISTENCE_CEILING.md`, `research/model_lab/HONEST_SKILL_BASELINE.md`):
hourly mooring physics is irreducible-error-dominated, so no amount of model sophistication helps.

## Why the existing "value gate" did not catch this

`research/model_lab/latent_tsf_value_gate.py` gates on whether a UMAP projection of the
autoencoder's latent space shows a temperature **color gradient**. That is circular — a latent of
temperature data will of course separate by temperature — and it measures **zero forecasting
skill**. A latent that "looks disentangled" is not a model that beats a baseline. The real gate is
skill-vs-best-naive on a **temporal holdout**, which this track fails.

## Decision (per `docs/VALUE_GATE.md` procedure)

1. **Not a capability.** Remove "MoE Advantage" / "Incremental SOTA (EWC)" from
   `PRODUCTION_READINESS.md`'s *Defensible Model Claims* — they beat no honest baseline. (Done
   2026-06-10.) Do not present this stack as a forecasting capability in README/pitch material.
2. **Research-only, shelved.** Keep the code under `sota_continual_learning/` and `research/` as a
   research track; stop spending GPU/maintenance on it for the M1 forecasting goal (the consultants'
   STOP-list, reframe Finding 10, already said this).
3. **One bounded, falsifiable re-entry condition.** The track may be re-promoted *only* if it
   clears a **pre-registered temporal-holdout** bar: beat the best-naive baseline (the better of
   persistence and seasonal-naive) on **≥ a majority of series** out-of-sample — not in-sample, not
   a UMAP picture. Until that test passes, it stays research-only.
4. **Higher-value home for the EWC idea, if any:** the one place a continual-learning/drift-aware
   model is *empirically motivated* is the **bacteria** regime-shift calibration problem (reframe
   Findings 9, 12: a pre-2022 isotonic map breaks post-break in San Diego). That is a real,
   baseline-beating target; M1 hourly forecasting is not.

## What stays

Deletion is not required — the gate rewards keeping honest negative results as deliverables. This
verdict **is** the deliverable: a documented, reproducible negative that stops sophistication-chasing
on a persistence-ceilinged target.
