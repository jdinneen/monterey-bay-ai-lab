# Model Suite — what exists, what has deps, what is claimable

Generated from `research/model_lab/model_registry.yaml` by `research/model_lab/model_suite.py`.
Fail-closed: a model is **claimable** only if the registry marks it so AND its evidence file exists.

- registry validation: PASS
- models registered: 13
- claimable now: 2

| model | family | task | neural? | deps available? | beats baseline? | calibrated? | claimable? | status | blocker |
|---|---|---|:-:|:-:|---|:-:|:-:|---|---|
| `bacteria_hgbt_isotonic` | tree | bacteria_classification | no | yes | yes — beats AB411, Virtual-Beach MLR, and station-memory | yes | **YES** | production_candidate | prospective forward-time pilot + partner available_at lab timestamps |
| `bacteria_hgbt_spatial` | tree | bacteria_classification | no | yes | no — raw coordinates and neighbour-lag are a wash (all spatial arms AP 0.497 = the no-spatial driver model); per-station/county/statewide rate features already encode location | yes | no | negative_result | raw-coordinate spatial surface does not beat the no-spatial driver model; physical covariates (outfall/river-mouth distance) untested |
| `bacteria_xgboost` | tree | bacteria_classification | no | yes | comparator — run via the suite to test the data ceiling vs HGBT | yes | no | benchmark | promote only if it beats the best operational baseline AND calibrates (it won't materially beat HGBT — data ceiling) |
| `xgboost_forecast_v2` | tree | m1_forecasting | no | yes | horizon-specific — gated by best-naive (ops/seasonal_naive.py); +72h tier rejected | no | no | benchmark | promotion is per target/horizon cell via release_gate/mbal_promotion_matrix.py |
| `patchtst` | neural | m1_forecasting | yes | yes | horizon-specific; best aggregate full-history neural run | no | no | benchmark | must clear the best-naive gate per cell |
| `nhits` | neural | m1_forecasting | yes | yes | not established | no | no | research_candidate | no validated win over best-naive |
| `early_warning_cusum` | detector | bacteria_event_detection | no | yes | NO — a NULL result; the SD-2022 break is a ~35-sigma step any detector flags | no | no | negative_result | n/a — documented null; do not revive as a contribution |
| `ab411_rain_rule` | baseline | bacteria_classification | no | yes | n/a — this is the deployed practice we must beat (AP ~0.18 SD-excluded) | no | no | baseline | n/a |
| `virtual_beach_mlr` | baseline | bacteria_classification | no | yes | n/a — the deployed tool we'd replace (AP ~0.38 SD-excluded) | no | no | baseline | n/a |
| `station_memory` | baseline | bacteria_classification | no | yes | n/a — strong baseline (AP ~0.28 SD-excluded); the bar drivers must beat | no | no | baseline | n/a |
| `da_forecast_hgbt` | tree | domoic_acid_forecasting | no | yes | yes — AP 0.232 vs best-naive 0.138; also beats NOAA C-HARM v3.1 nowcast 8/8 stations | yes | **YES** | production_candidate | small test-event count (52 events > 2020); weekly CalHABMAP sampling = ~1-week lead; needs a prospective forward-time pilot before an operational claim. |
| `hab_sota_hgbt` | tree | domoic_acid_forecasting | no | yes | no — adding the external-context signals does NOT beat the lean DA+precursor incumbent (gate is +0.005 AP) | yes | no | research_only_failed_gate | all_normalized_signals AP 0.181 and external_context AP 0.235 vs lean incumbent 0.232 (gap below the +0.005 gate) -- context signals are a WASH/HURT on this target. Kept as the honest 'more data did not help' ablation, not a promotion. |
| `charm_da_nowcast` | baseline | domoic_acid_forecasting | no | yes | n/a — this IS the operational incumbent our prior-visit forecast must beat (and does, 8/8) | no | no | baseline | n/a — operational comparator; badly miscalibrated on the CalHABMAP-pier overlap (mean p=0.69 vs 4% base) |

## Legend
- **neural?** — `neural`/`foundation`/`continual` families are neural; `tree`/`baseline` are not. The MoE/EWC/LatentTSF stack is `moe_ewc_latenttsf` (family `continual`).
- **deps available?** — whether the model's declared dependencies import in this environment. This is not an execution smoke test for the registry entrypoint.
- **beats baseline?** — against the honest baseline (best-naive for forecasting; AB411 / Virtual-Beach MLR / station-memory for bacteria), stated straight.
- **claimable?** — safe to put in a pitch/paper. Only `production_candidate` (and, case-by-case, a promoted `benchmark`) with on-disk evidence qualifies.

## The fundable headline
`bacteria_hgbt_isotonic` remains THE fundable, statewide headline: a calibrated enterococcus nowcast that beats deployed practice and — verified by leave-one-beach-out (AP 0.494 / ROC-AUC 0.852) — generalizes to beaches it never trained on. (A *learned spatial surface* from raw coordinates was tested and is a WASH, not a lift; see the `bacteria_hgbt_spatial` negative-result row.) A second, distinct claimable now exists on the marine-HAB frontier: `da_forecast_hgbt` (domoic-acid), which beats best-naive 8/8 LOSO stations and the operational NOAA C-HARM nowcast 8/8 piers (AP 0.271 vs 0.027; evidence `reports/hab/charm_comparison.json`, a frozen live-ERDDAP comparison). It reproduces from the shipped 304 KB CalHABMAP input, but rests on a small 52-event test set pending a prospective pilot, so it is a frontier result, not the statewide headline. Everything on the M1-forecasting / neural side remains a benchmark, research-only, or a documented negative.
