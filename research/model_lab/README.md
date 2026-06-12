# Model Lab

Exploratory model work belongs here, not in the production root. This includes foundation
benchmarks, SOTA experiments, driver ablations, and read-only diagnostics.

Production promotion still runs through:

- `mbal_forecast_v2.py`
- `mbal_neural_forecast.py`
- `mbal_train.py`
- `release_gate/`
- `ops/run_xgb_on_candidate_splits.py`

Retained research utilities:

- `mbal_big_analysis.py`
- `mbal_data_intimacy.py`
- `generate_golden_era_splits.py`
- `mbal_neural_merge.py`
- `jobs.wave_ablation.json`

Latent discovery 2026:

- The current hidden-pattern candidate is the masked corpus learner in
  `unsupervised_corpus_run.py` plus the weak-label variant in
  `semi_supervised_corpus_run.py`.
- It is additive: source data is read-only and outputs land under `runs/`.
- Coordination, review gates, and full `ops/run_safe.py` commands live in
  `reports/model_hardening/latent_2026/`.
- Do not claim success from reconstruction loss alone. Use downstream probes against
  the honest bacteria and forecasting baselines, or keep the result as a documented
  negative/mixed finding.

Wave-driver status:

- `jobs.wave_ablation.json` is a bounded screening job, not a production promotion
  path.
- `ops/report_wave_driver_claim.py` regenerates the wave-driver claim report from
  matched baseline/wave ablation outputs.
- The 2026-06-08 bounded NDBC 46042 wave screen is
  `not_supported_as_general_claim`; keep wave-driver claims research-only unless
  repeated shared-split evidence clears the release/promotion gates.
