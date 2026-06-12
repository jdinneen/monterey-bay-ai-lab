# Operational beach-bacteria benchmark (honest, stratified, calibrated)

- availability mode: fixed_lag | reveal lag days: 2
- train events: 20,714 | train base rate: 0.0767
- AB411 rain rule: EVALUATED - rain_3d >= 2.54mm (0.1in) wet-weather advisory rule; gridded Open-Meteo daily rainfall at 0.1deg, joined causally by station cell.

## ALL
- Ranking: model BEATS the operational baseline - AP 0.7534 vs best-operational 0.6991 (delta +0.0543).
- AB411 regulatory rule: model BEATS it - model AP 0.7534 vs AB411 rain-rule AP 0.2454.
- Calibration: raw ECE 0.1396 -> calibrated ECE 0.0972 vs best-operational ECE 0.0936 => deploy-ready.

| model | n | events | base | AP | AUC | rec@20%FPR | Brier | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_global_rate | 115647 | 24917 | 0.2155 | 0.2155 | 0.5 | None | 0.1883 | 0.1388 |
| baseline_month_climatology | 115647 | 24917 | 0.2155 | 0.2444 | 0.5514 | 0.2922 | 0.18671 | 0.1369 |
| baseline_prior_lab | 115647 | 24917 | 0.2155 | 0.5668 | 0.8116 | 1.0 | 0.12405 | 0.124 |
| baseline_station_memory | 115647 | 24917 | 0.2155 | 0.6991 | 0.863 | 0.761 | 0.13008 | 0.0936 |
| baseline_ab411_rain | 115647 | 24917 | 0.2155 | 0.2454 | 0.555 | None | 0.25857 | 0.2586 |
| baseline_vb_mlr | 115647 | 24917 | 0.2155 | 0.525 | 0.779 | 0.6726 | 0.18199 | 0.2055 |
| model_hgbt | 115647 | 24917 | 0.2155 | 0.7534 | 0.9181 | 0.8856 | 0.11322 | 0.1396 |
| model_hgbt_calibrated | 115647 | 24917 | 0.2155 | 0.7453 | 0.9177 | 0.8934 | 0.11332 | 0.0972 |

## EXCLUDE_SAN_DIEGO
- Ranking: model BEATS the operational baseline - AP 0.5119 vs best-operational 0.3754 (delta +0.1365).
- AB411 regulatory rule: model BEATS it - model AP 0.5119 vs AB411 rain-rule AP 0.1779.
- Calibration: raw ECE 0.1915 -> calibrated ECE 0.0203 vs best-operational ECE 0.011 => deploy-ready.

| model | n | events | base | AP | AUC | rec@20%FPR | Brier | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_global_rate | 89321 | 8999 | 0.1007 | 0.1007 | 0.5 | None | 0.09118 | 0.0241 |
| baseline_month_climatology | 89321 | 8999 | 0.1007 | 0.1531 | 0.647 | 0.4108 | 0.08871 | 0.0222 |
| baseline_prior_lab | 89321 | 8999 | 0.1007 | 0.1852 | 0.6318 | 1.0 | 0.12768 | 0.1277 |
| baseline_station_memory | 89321 | 8999 | 0.1007 | 0.2782 | 0.7716 | 0.6006 | 0.08166 | 0.011 |
| baseline_ab411_rain | 89321 | 8999 | 0.1007 | 0.1779 | 0.6495 | None | 0.16285 | 0.1629 |
| baseline_vb_mlr | 89321 | 8999 | 0.1007 | 0.3754 | 0.7493 | 0.5835 | 0.17629 | 0.2805 |
| model_hgbt | 89321 | 8999 | 0.1007 | 0.5119 | 0.8584 | 0.7534 | 0.12166 | 0.1915 |
| model_hgbt_calibrated | 89321 | 8999 | 0.1007 | 0.4974 | 0.8578 | 0.768 | 0.0672 | 0.0203 |

## SAN_DIEGO_ONLY
- Ranking: model does NOT beat the operational baseline - AP 0.9314 vs best-operational 0.9389 (delta -0.0075).
- AB411 regulatory rule: model BEATS it - model AP 0.9314 vs AB411 rain-rule AP 0.6022.
- Calibration: raw ECE 0.0706 -> calibrated ECE 0.3611 vs best-operational ECE 0.1118 => NOT deploy-ready (recalibrate).

| model | n | events | base | AP | AUC | rec@20%FPR | Brier | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_global_rate | 26326 | 15918 | 0.6046 | 0.6046 | 0.5 | None | 0.51782 | 0.528 |
| baseline_month_climatology | 26326 | 15918 | 0.6046 | 0.5877 | 0.476 | 0.2252 | 0.51923 | 0.526 |
| baseline_prior_lab | 26326 | 15918 | 0.6046 | 0.881 | 0.8842 | 1.0 | 0.11175 | 0.1118 |
| baseline_station_memory | 26326 | 15918 | 0.6046 | 0.9389 | 0.9265 | 0.9025 | 0.29437 | 0.3761 |
| baseline_ab411_rain | 26326 | 15918 | 0.6046 | 0.6022 | 0.4948 | None | 0.58334 | 0.5833 |
| baseline_vb_mlr | 26326 | 15918 | 0.6046 | 0.7978 | 0.7647 | 0.6767 | 0.20132 | 0.1134 |
| model_hgbt | 26326 | 15918 | 0.6046 | 0.9314 | 0.9388 | 0.9599 | 0.08456 | 0.0706 |
| model_hgbt_calibrated | 26326 | 15918 | 0.6046 | 0.9309 | 0.9384 | 0.9643 | 0.26978 | 0.3611 |

## MONTEREY
- Ranking: model BEATS the operational baseline - AP 0.416 vs best-operational 0.2903 (delta +0.1257).
- AB411 regulatory rule: model BEATS it - model AP 0.416 vs AB411 rain-rule AP 0.1822.
- Calibration: raw ECE 0.2517 -> calibrated ECE 0.0123 vs best-operational ECE 0.0161 => deploy-ready.

| model | n | events | base | AP | AUC | rec@20%FPR | Brier | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline_global_rate | 5265 | 454 | 0.0862 | 0.0862 | 0.5 | None | 0.07889 | 0.0096 |
| baseline_month_climatology | 5265 | 454 | 0.0862 | 0.1204 | 0.6192 | 0.3678 | 0.07808 | 0.0156 |
| baseline_prior_lab | 5265 | 454 | 0.0862 | 0.1007 | 0.5465 | 1.0 | 0.14492 | 0.1449 |
| baseline_station_memory | 5265 | 454 | 0.0862 | 0.1571 | 0.6871 | 0.3987 | 0.07628 | 0.0161 |
| baseline_ab411_rain | 5265 | 454 | 0.0862 | 0.1822 | 0.695 | None | 0.16372 | 0.1637 |
| baseline_vb_mlr | 5265 | 454 | 0.0862 | 0.2903 | 0.7095 | 0.5 | 0.17883 | 0.3037 |
| model_hgbt | 5265 | 454 | 0.0862 | 0.416 | 0.7946 | 0.6123 | 0.15276 | 0.2517 |
| model_hgbt_calibrated | 5265 | 454 | 0.0862 | 0.3995 | 0.7936 | 0.6916 | 0.06366 | 0.0123 |
