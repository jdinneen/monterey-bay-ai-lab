# Signal Catalog

Single source of truth for every signal the lab has touched. The catalog **records**; the gate (`research/bacteria/signal_lab.py`, ΔAP>=0.005 + leave-one-beach-out) **decides** promotion into any model. Nulls are logged on purpose.

`PRIMARY` = the validated model for that target · `KEEP` = passed the gate as a feature · `WASH` = tested, no lift · `REJECT` = tested, hurt · `UNTESTED` = not yet gated · `PLANNED` = not yet fetched.

## Targets (state of play)

- **bacteria_exceedance** — Will a beach's water exceed the AB411 enterococcus standard (>104 MPN/100mL)?
  - best model: statewide HGBT + isotonic (research/bacteria/operational_benchmark.py)  ·  EXCLUDE_SAN_DIEGO calibrated AP ~0.47 / ROC-AUC ~0.86; deploy-ready 8/9 counties
  - baseline to beat: seasonal-naive + station-memory + AB411 rain rule  ·  coverage: ~830 statewide beaches, 2005-2026
- **domoic_acid** — Will a pier's next sample exceed particulate domoic acid 500 ng/L (=0.5 ng/mL)?
  - best model: DA+precursor HGBT (research/hab/da_forecast.py), ~1-week lead  ·  AP 0.232 / ROC-AUC 0.831 time-holdout; beats seasonal-naive 8/8 and C-HARM 8/8 piers
  - baseline to beat: seasonal-naive + persistence; NOAA C-HARM (operational)  ·  coverage: 17 CalHABMAP piers only, weekly, 2005-2026

## Predictive value by target

| signal | modality | bacteria_exceedance | domoic_acid | coverage |
|---|---|--:|--:|---|
| **station_lab_history** | lab | PRIMARY | - | all ~830 statewide AB411 beaches; any station-da… |
| **rainfall** | meteorological | KEEP | - | statewide coastal-county ASOS airports, 2000-202… |
| **neighbour_state** | derived | KEEP | - | statewide; any beach with neighbours reporting i… |
| **river_discharge** | hydrological | WASH | - | beaches within ~30km of a data-bearing USGS gaug… |
| **cdip_waves** | oceanographic | WASH | - | beaches near a CDIP buoy; ~39% of statewide stat… |
| **tide_water_level** | oceanographic | WASH | UNTESTED | all beaches mapped to nearest CO-OPS gauge (100%… |
| **news_event_gdelt** | event | WASH | - | county-level, sparse; GDELT API currently IP-blo… |
| **domoic_acid_pda** | biological | UNTESTED | PRIMARY | 17 CalHABMAP piers, 2005-2026 (NOT statewide — ~… |
| **pseudo_nitzschia** | biological | - | KEEP | 17 CalHABMAP piers, 2005-2026. |
| **hab_physical_nutrient_drivers** | oceanographic | - | REJECT | 17 CalHABMAP piers, weekly. |
| **upwelling_index_beuti** | oceanographic | UNTESTED | REJECT | full CA coast, daily 1988-present (308,836 rows)… |
| **open_meteo_air_temp** | meteorological | KEEP | - | statewide CA beaches via nearest source location… |
| **ghcnd_precip** | meteorological | WASH | - | statewide CA beaches via nearest source location… |
| **usgs_turbidity** | hydrological | REJECT | - | statewide CA beaches via nearest source location… |
| **usgs_water_temp** | hydrological | WASH | - | statewide CA beaches via nearest source location… |
| **cdip_sst_buoy** | oceanographic | WASH | - | statewide CA beaches via nearest source location… |
| **cdip_wave_network** | oceanographic | WASH | - | statewide CA beaches via nearest source location… |
| **ndbc_buoy_wind** | meteorological | WASH | - | statewide CA beaches via nearest source location… |
| **ncei_isd_air_temp** | meteorological | REJECT | - | statewide CA beaches via nearest source location… |
| **nasa_power_air_temp** | meteorological | REJECT | - | statewide CA beaches via nearest source location… |
| **gridmet_precip** | meteorological | REJECT | - | statewide CA beaches via nearest source location… |

## Detail

### station_lab_history  (lab)
*Captures:* Per-station prior exceedance + analyte history — the "this beach was recently dirty" memory.
- **Does:** Carries most of the bacteria nowcast skill — the single strongest signal.
- **Does NOT:** Anticipate a NEW event at a clean beach with no prior signal (e.g. a fresh spill); it is memory, so it lags regime breaks.
- **Why:** Fecal exceedance is strongly autocorrelated at a site (chronic sources, slow-changing sanitary infrastructure).
- **Coverage:** all ~830 statewide AB411 beaches; any station-day with a prior sample. · cadence per beach-sample
- **Access:** derived in research/bacteria/operational_benchmark.py (add_causal_features)
- **bacteria_exceedance:** `PRIMARY` — dominant signal; station_memory baseline AP 0.28 on EXCLUDE_SAN_DIEGO — `research/bacteria/operational_benchmark.py`

### rainfall  (meteorological)
*Captures:* County-mapped ASOS daily precipitation + first-flush (rain after a dry spell).
- **Does:** Lift bacteria via first-flush — storm runoff mobilizes fecal load. The one external driver that clears the gate.
- **Does NOT:** Help in the dry season or for chronic non-storm exceedances; it is coarse (county-level, not per-beach).
- **Why:** Storm runoff is a genuine causal pathway for fecal indicator bacteria; first-flush concentrates it after dry spells.
- **Revisit when:** per-beach nearest-gauge rainfall (vs county-mapping) could earn more — retest if finer rain is added.
- **Coverage:** statewide coastal-county ASOS airports, 2000-2026. · cadence daily
- **Access:** Iowa Mesonet ASOS; cached bacteria_results/rainfall/
- **bacteria_exceedance:** `KEEP` — ΔAP +0.010 (pooled A/B); first-flush mechanism — `research/bacteria/operational_benchmark.py`

### neighbour_state  (derived)
*Captures:* Within-county / statewide recent exceedance context (cty_prev7, sw_prev7).
- **Does:** Add spatial context — a dirty neighbourhood predicts a dirty beach. Passes the gate as an L2 signal.
- **Does NOT:** Help isolated beaches with no reporting neighbours; it is a regional, not local, signal.
- **Why:** Nearby beaches share drivers (one storm or spill raises a whole stretch of coast at once).
- **Revisit when:** a distance-weighted neighbour graph could beat flat county-pooling.
- **Coverage:** statewide; any beach with neighbours reporting in the prior week. · cadence derived, ~weekly
- **Access:** derived in research/bacteria/ (signal_lab.py L2 candidate, spatial_autocorr)
- **bacteria_exceedance:** `KEEP` — L2 spatial signal; passes ΔAP + LOBO gate — `research/bacteria/signal_lab.py`

### river_discharge  (hydrological)
*Captures:* Nearest USGS gauge first-flush river-discharge features.
- **Does:** Little — physically plausible (river plumes carry load) but it does not clear the lift bar.
- **Does NOT:** Add anything over rainfall; gauge coverage is sparse.
- **Why:** Discharge is downstream of rain that is already in the model → redundant.
- **Revisit when:** could matter for specific river-mouth beaches in a stratified (not pooled) model.
- **Coverage:** beaches within ~30km of a data-bearing USGS gauge. · cadence daily
- **Access:** USGS; cached bacteria_results/discharge/
- **bacteria_exceedance:** `WASH` — marginal; does not clear the lift bar (honest negative/mixed) — `research/bacteria/operational_benchmark.py`

### cdip_waves  (oceanographic)
*Captures:* Nearest CDIP/NDBC buoy wave height / period / direction (Hs, Tp, Dm).
- **Does:** Essentially nothing for bacteria.
- **Does NOT:** Provide transferable lift; net-negative pooled; ~39% coverage.
- **Why:** Wave energy is not a primary fecal-bacteria driver — orthogonal to runoff.
- **Revisit when:** could matter for sediment-resuspension at specific surf-zone beaches; UNTESTED for DA.
- **Coverage:** beaches near a CDIP buoy; ~39% of statewide station-days. · cadence hourly -> daily
- **Access:** CDIP; cached bacteria_results/cdip_waves/
- **bacteria_exceedance:** `WASH` — pooled ΔAP -0.008; noise-band on Monterey (4th driver-null) — `reports/operational_benchmark/wave_tide_ablation.md`

### tide_water_level  (oceanographic)
*Captures:* NOAA CO-OPS daily-mean water level (a storm-surge proxy) at nearest gauge.
- **Does:** Appear to help pooled — but that gain is a San-Diego artifact, not real.
- **Does NOT:** Lift transferably — ΔAP sign-flips across strata (+0.044 pooled/SD but -0.010 Monterey).
- **Why:** Daily-MEAN level is mostly astronomical+seasonal (≈ calendar, already modeled) plus a weak surge term that ≈ rain (redundant).
- **Revisit when:** the de-tided SURGE residual (not raw level) might carry real storm signal — retest as residual; and test for DA (surge↔upwelling).
- **Coverage:** all beaches mapped to nearest CO-OPS gauge (100%, but distant for some). · cadence daily
- **Access:** NOAA CO-OPS; cached bacteria_results/tide_stages/
- **bacteria_exceedance:** `WASH` — ΔAP sign-flips across strata (+0.044 pooled/SD, -0.010 Monterey) = artifact — `reports/operational_benchmark/wave_tide_ablation.md`
- **domoic_acid:** `UNTESTED` — not gated for DA; plausible (surge ~ upwelling/storm coupling)

### news_event_gdelt  (event)
*Captures:* GDELT news/event mentions (spill / sewage / closure language).
- **Does:** Nothing additive — its only value is an operational human digest.
- **Does NOT:** Improve the model; a powered A/B slightly HURT it.
- **Why:** News of a spill lags or co-occurs with the rain signal already in the model → redundant.
- **Revisit when:** a STRUCTURED spill-report feed (SSO / CIWQS), not news, could be a real leading indicator — that is a different signal worth fetching.
- **Coverage:** county-level, sparse; GDELT API currently IP-blocked. · cadence intermittent
- **Access:** GDELT (blocked); see news-modality work.
- **bacteria_exceedance:** `WASH` — redundant with rainfall; powered A/B hurt slightly. Operational digest only. — `research/bacteria/operational_benchmark.py`

### domoic_acid_pda  (biological)
*Captures:* CalHABMAP particulate domoic acid (pDA, ng/mL) — the marine biotoxin label itself.
- **Does:** Drive the DA forecast (AP 0.232, AUC 0.831; beats C-HARM 8/8 piers). Richest in Monterey Bay (Monterey 17.8% / Santa Cruz 7.6% event rate).
- **Does NOT:** Inform the bacteria (fecal) target — a different hazard; coverage-limited; weekly cadence caps lead at ~1 week.
- **Why:** DA is an ocean-upwelling biotoxin, mechanistically unrelated to land-driven fecal contamination — and seasonally OFFSET (DA = dry upwelling season; bacteria = wet winter).
- **Revisit when:** a bacteria test is only meaningful on the ~8 co-located pier beaches, as an INTERACTION — gate it there, not statewide.
- **Coverage:** 17 CalHABMAP piers, 2005-2026 (NOT statewide — ~813/830 beaches have none). · cadence weekly
- **Access:** SCCOOS ERDDAP; cached data/external_curated/habmap_cdph/; ops/data_fetch adapters/habmap.py
- **domoic_acid:** `PRIMARY` — forecast AP 0.232, ROC-AUC 0.831 (time-holdout); beats C-HARM 8/8 piers — `research/hab/da_forecast.py`
- **bacteria_exceedance:** `UNTESTED` — COVERAGE-LIMITED: 17 piers can't inform 830 beaches. Test only on co-located sites.

### pseudo_nitzschia  (biological)
*Captures:* Pseudo-nitzschia seriata + delicatissima cell counts (cells/L) — the DA-producing diatom.
- **Does:** Improve the DA forecast — the precursor bloom precedes the toxin (best-AP feature group).
- **Does NOT:** Perfectly predict toxin — not all Pseudo-nitzschia is toxic; pier-only coverage.
- **Why:** The toxin is produced BY this diatom, so a rising cell count is a leading precursor.
- **Revisit when:** species/condition-resolved counts (toxic vs non-toxic strains) could sharpen it.
- **Coverage:** 17 CalHABMAP piers, 2005-2026. · cadence weekly
- **Access:** SCCOOS ERDDAP; cached data/external_curated/habmap_cdph/
- **domoic_acid:** `KEEP` — best-AP DA feature group (DA+precursor 0.232 vs DA-history 0.218) — `research/hab/da_forecast.py`

### hab_physical_nutrient_drivers  (oceanographic)
*Captures:* In-situ temperature / chlorophyll / nitrate / phosphate / silicate at the piers.
- **Does:** Nothing useful — adding them HURTS the DA model.
- **Does NOT:** Provide lift in additive form (AP 0.169 with drivers vs 0.218 without).
- **Why:** Same pattern as the bacteria driver-nulls — the bloom + precursor + season already capture the regime, so raw physical/nutrient drivers add only noise (5th driver-null).
- **Revisit when:** regime-aware use (upwelling vs stratification) or the Si:N ratio inside a regime-split model might rescue some; the raw additive form is rejected.
- **Coverage:** 17 CalHABMAP piers, weekly. · cadence weekly
- **Access:** SCCOOS ERDDAP; cached data/external_curated/habmap_cdph/
- **domoic_acid:** `REJECT` — HURT the DA model (AP 0.169 with drivers vs 0.218 without) = 5th driver-null — `research/hab/da_forecast.py`

### upwelling_index_beuti  (oceanographic)
*Captures:* NOAA BEUTI / CUTI biologically-effective upwelling transport (nutrient-flux proxy).
- **Does:** TESTED + REJECTED for DA. Windowed BEUTI/CUTI at the pier latitude band, swept over 7/14/21/28/45-day lead windows, all FAIL the gate (REJECT or WASH).
- **Does NOT:** Improve the DA forecast - every window regresses or washes; the 28d window gained on the temporal holdout (+0.0192) but FAILED leave-one-station-out (-0.0087) = sparse-panel overfit, not signal.
- **Why:** Mechanistically the root cause (upwelling -> nitrate -> Pseudo-nitzschia -> DA), but on the 52-event panel it adds noise the precursor-dominated model cannot use - the DA driver-null extends even to the textbook-correct driver. The dual temporal+LOSO gate caught the false 28d positive.
- **Revisit when:** DA panel grows materially (more events) or a non-linear/threshold upwelling formulation is tried; also testable for bacteria_exceedance (untested).
- **Coverage:** full CA coast, daily 1988-present (308,836 rows). FETCHED: source upwelling_indices. · cadence daily by latitude band (32N-42N)
- **Access:** https://mjacox.com/wp-content/uploads/{CUTI,BEUTI}_daily.csv -> data/external_curated/upwelling_indices/ (the erdCUTIdaily ERDDAP dataset is dead-404).
- **domoic_acid:** `REJECT` — swept 7-45d lead windows: all REJECT/WASH; 28d temporal +0.0192 but LOSO -0.0087 (overfit) — `reports/hab/da_upwelling_gate/verdict.json`
- **bacteria_exceedance:** `UNTESTED` — coastal physical driver; not yet gated for bacteria

### open_meteo_air_temp  (meteorological)
*Captures:* ERA5 reanalysis 2m air temperature at the nearest CA-coastal grid cell (open_meteo_archive), as-of the sample.
- **Does:** Clears the bacteria gate - temporal dAP +0.0131 AND survives leave-one-beach-out (+0.0050) over FEATS+rain.
- **Does NOT:** Add much beyond a seasonal proxy; lift is modest (~+0.013 AP).
- **Why:** Warmer antecedent air temp tracks bacterial survival/regrowth plus a finer seasonal signal than calendar features; dense ERA5 coverage makes the join clean.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `KEEP` — temporal dAP +0.0131, LOBO +0.0050 (SD-excluded, FEATS+rain base) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### ghcnd_precip  (meteorological)
*Captures:* GHCN-Daily statewide-mean station precipitation (ghcnd), as-of the sample.
- **Does:** Borderline wash - temporal dAP +0.0049 (under the 0.005 bar) though LOBO +0.0062.
- **Does NOT:** Beat the existing Open-Meteo rainfall driver it duplicates.
- **Why:** Precip is already the one KEEP driver; a second statewide-mean precip is near-redundant.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `WASH` — temporal dAP +0.0049 < 0.005 (redundant with rainfall) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### usgs_turbidity  (hydrological)
*Captures:* USGS NWIS daily turbidity at the nearest CA site (usgs_dv_statewide), as-of the sample.
- **Does:** Regresses the bacteria target - temporal dAP -0.0053 (LOBO +0.0076 sign-flips).
- **Does NOT:** Help statewide - turbidity sites are sparse/inland relative to beaches.
- **Why:** A buggy earlier join false-KEPT it (+0.0071); the corrected leakage-safe join shows it does not help.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `REJECT` — temporal dAP -0.0053 (sparse inland coverage; earlier KEEP was a join artifact) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### usgs_water_temp  (hydrological)
*Captures:* USGS NWIS daily water temperature at the nearest CA site (usgs_dv_statewide), as-of the sample.
- **Does:** Wash for bacteria - temporal dAP -0.0007.
- **Does NOT:** Carry independent signal beyond SST/air-temp already tested.
- **Why:** River water temp at sparse inland gauges is a weak, poorly-located proxy for beach conditions.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `WASH` — temporal dAP -0.0007 (noise) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### cdip_sst_buoy  (oceanographic)
*Captures:* In-situ sea-surface temperature at the nearest CDIP buoy (cdip_sst_network), as-of the sample.
- **Does:** Pure wash - temporal dAP 0.0000.
- **Does NOT:** Add anything over the SST signal already covered (MUR ocean driver).
- **Why:** SST is a smooth seasonal field already proxied by calendar + ocean features.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `WASH` — temporal dAP 0.0000 — `reports/bacteria/new_source_signal_gate/verdicts.json`

### cdip_wave_network  (oceanographic)
*Captures:* Significant wave height at the nearest CDIP network buoy (cdip_wave_network), as-of the sample.
- **Does:** Pure wash - temporal dAP 0.0000; confirms the wave driver-null at network scale.
- **Does NOT:** Change the standing wave/tide driver-null for bacteria.
- **Why:** Waves do not drive fecal-indicator exceedance here; the single-buoy wrapper washed and the network does too.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `WASH` — temporal dAP 0.0000 (confirms wave driver-null) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### ndbc_buoy_wind  (meteorological)
*Captures:* Statewide-mean NDBC continuous buoy wind speed (ndbc_cwind), as-of the sample.
- **Does:** Wash for bacteria - temporal dAP +0.0035 (LOBO +0.0062, temporal under the bar).
- **Does NOT:** Clear the gate as a mixing/upwelling proxy at statewide-mean resolution.
- **Why:** A statewide-mean wind (no per-buoy geo in cwind) is too coarse to move the target.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `WASH` — temporal dAP +0.0035 < 0.005 (coarse statewide-mean) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### ncei_isd_air_temp  (meteorological)
*Captures:* Observed hourly air temperature at the nearest NCEI ISD station (ncei_isd_hourly), as-of the sample.
- **Does:** Strongly regresses - temporal dAP -0.0489 (LOBO -0.0271).
- **Does NOT:** Help - sparse station coverage makes the nearest-station join noisy.
- **Why:** Same variable as the KEPT open_meteo_air_temp but from sparse stations; the noisy join adds harmful variance. Coverage/quality, not the variable, decides.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `REJECT` — temporal dAP -0.0489 (sparse-station noise; cf. dense ERA5 KEEP) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### nasa_power_air_temp  (meteorological)
*Captures:* NASA POWER daily 2m air temperature at the nearest CA grid cell (nasa_power_daily), as-of the sample.
- **Does:** Regresses - temporal dAP -0.0193 (LOBO -0.0107).
- **Does NOT:** Beat the dense ERA5 air-temp that did KEEP.
- **Why:** Coarser/daily reanalysis air temp adds noise where finer ERA5 hourly helped.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `REJECT` — temporal dAP -0.0193 (coarser than the ERA5 KEEP) — `reports/bacteria/new_source_signal_gate/verdicts.json`

### gridmet_precip  (meteorological)
*Captures:* gridMET 4km daily precipitation at the nearest CA grid cell (gridmet_daily), as-of the sample.
- **Does:** Regresses - temporal dAP -0.0163 (LOBO -0.0024).
- **Does NOT:** Beat the Open-Meteo rainfall driver it duplicates.
- **Why:** A second gridded precip is redundant with the established rainfall driver and adds noise.
- **Coverage:** statewide CA beaches via nearest source location. · cadence per beach-sample (nearest-location as-of, reveal-lagged)
- **Access:** research/bacteria/new_source_signal_gate.py (causal join over data/external_curated/)
- **bacteria_exceedance:** `REJECT` — temporal dAP -0.0163 (redundant with rainfall) — `reports/bacteria/new_source_signal_gate/verdicts.json`
