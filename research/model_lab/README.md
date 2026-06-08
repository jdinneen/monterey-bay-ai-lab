# Model Lab

Exploratory model work belongs here, not in the production root. This includes foundation
benchmarks, SOTA experiments, driver ablations, and read-only diagnostics.

Production promotion still runs through:

- `mbari_forecast_v2.py`
- `mbari_neural_forecast.py`
- `mbari_train.py`
- `release_gate/`
- `ops/run_xgb_on_candidate_splits.py`

Retained research utilities:

- `mbari_big_analysis.py`
- `mbari_data_intimacy.py`
- `generate_golden_era_splits.py`
- `mbari_neural_merge.py`
- `jobs.wave_ablation.json`

Wave-driver status:

- `jobs.wave_ablation.json` is a bounded screening job, not a production promotion
  path.
- `ops/report_wave_driver_claim.py` regenerates the wave-driver claim report from
  matched baseline/wave ablation outputs.
- The 2026-06-08 bounded NDBC 46042 wave screen is
  `not_supported_as_general_claim`; keep wave-driver claims research-only unless
  repeated shared-split evidence clears the release/promotion gates.
