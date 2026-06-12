# BEATING DEPLOYED PRACTICE IN BEACH-WATER QUALITY NOWCASTING: A LEAKAGE-CLEAN, CALIBRATED, STATEWIDE BENCHMARK ON THE CALIFORNIA MARINE ENTEROCOCCUS STANDARD

**Author:** Jon Dinneen
**Affiliation:** [to be completed]
**Correspondence:** [email]
**Draft for submission — not peer reviewed. Prepared 2026-06-09.**

> Reproduction materials for every number in this draft are in the same directory:
> see `REPRODUCE.md`. The headline configuration is
> `--label enterococcus --reveal-lag-days 2`; expected outputs are in `expected/`.

Suggested venues (any one; this is written venue-neutral as a short paper / extended abstract):
NeurIPS / ICLR "Tackling Climate Change with ML" workshop; ACM COMPASS; Environmental
Modelling & Software; Water Research; AGU Ocean Sciences; KDD Applied Data Science track.

---

## ABSTRACT

Beach advisories for fecal-indicator-bacteria (FIB) exceedance are issued from laboratory
cultures that return roughly 18-24 hours after a water sample is collected, so the same-day
public-health decision is effectively unguided or relies on a coarse wet-weather rule
(California AB411). We build a statewide California benchmark (1.32M assays, 802 monitoring
stations, 17 counties, 2005-2026) for NOWCASTING single-sample marine enterococcus
exceedance (>104 MPN/100 mL) using only information available at sampling time. With strictly
causal features — prior laboratory results enforced against the true lab-reveal lag, station
memory, seasonality, gridded rainfall, and USGS river first-flush discharge — a gradient-
boosted decision-tree model, isotonically calibrated, attains Average Precision (AP) 0.497 /
AUROC 0.858 on the honest statewide stratum. It beats, on the same data: the AB411 rain rule
(AP 0.178; ~2.8x), a Virtual-Beach-class per-site multiple logistic regression representing
DEPLOYED practice (AP 0.375; +0.12 AP, ~32%), and a station-memory baseline (AP 0.278) —
while remaining deploy-ready in calibration (Expected Calibration Error, ECE, 0.020). Under
leave-one-county-out (LOCO) spatial cross-validation, the model beats AB411 and station
memory in 9/9 held-out counties. We report negative and null results honestly: a 2022
San Diego / Tijuana-River sewage REGIME BREAK inflates pooled metrics and is excluded from
the headline; river discharge is largely rain-redundant; and a static probability calibrator
degrades under regime shift, which a prequential (online) recalibrator only partially repairs.
We release a leakage-controlled, reproducible benchmark and argue that the field's
publishable contribution is not a higher pooled AUC but a calibrated model that beats deployed
practice under honest, stratified, spatially-held-out evaluation.

---

## 1. INTRODUCTION

Recreational coastal waters are monitored for fecal-indicator bacteria; when a single-sample
standard is exceeded, an advisory or closure is posted. The operational bottleneck is time:
the culture-based laboratory assay returns ~18-24 hours after sampling, so on the day a
beachgoer is exposed, the manager either has yesterday's result or, under California's AB411
program, a wet-weather rule that posts an advisory when rainfall exceeds a threshold. AB411 is
known to be weak; the predictive tool actually deployed at many modern beach programs is EPA
Virtual Beach, a per-site multiple-linear/logistic regression on local hydrometeorological
drivers, or a USGS site-specific nowcast.

A large body of work reports machine-learning models for FIB nowcasting, often with
impressive pooled AUC. Two failure modes recur and motivate this paper. First, LEAKAGE: many
"prior result" features are constructed by sample index (the previous SAMPLE), not by reveal
time (the previous RESULT that had physically returned by prediction time); and spatially
pooled features can leak a held-out region into training. Second, ARTIFACTS: a model evaluated
across a measurement/program discontinuity can score the discontinuity rather than generalizable
skill, inflating a headline number that does not survive domain scrutiny.

Our contribution is an honest benchmark. We (i) define a single, standard-aligned target — the
AB411 marine enterococcus single-sample standard; (ii) construct strictly causal features with
an enforced lab-reveal lag; (iii) evaluate against a baseline ladder that includes a
Virtual-Beach-class model representing deployed practice, not only the weak AB411 rule;
(iv) stratify by region and era to surface, rather than hide, a regime break; (v) test spatial
generalization by leave-one-county-out; and (vi) treat calibration as an explicit deploy gate.
We find a model that beats deployed practice and generalizes — and we report where it does not.

---

## 2. DATA

We use the statewide California beach-water monitoring record (CEDEN / BeachWatch lineage):
1,319,296 analyte observations across 802 stations and 17 coastal counties, 2005-2026, for
four indicators (Enterococcus, E. coli, Fecal Coliforms, Total Coliforms). Observations are
aggregated worst-of-day per station/analyte. Station geolocation (lat/lon) for 822 stations is
joined to enable environmental-driver attribution.

**Target.** The headline label is the AB411 MARINE single-sample standard: enterococcus > 104
MPN/100 mL, evaluated on station-days that carry an enterococcus reading. This is a single,
era-stable, standard-aligned target. We deliberately do NOT use a pooled "any-analyte"
disjunction (enterococcus OR fecal OR total OR a ratio gate): that compound label is a moving
target as the reported analyte panel shifts across years and is precisely what a 2022 regime
break (Section 4.4) exploits. The non-enterococcus analytes are retained as predictive FEATURES.

**Drivers.** Daily precipitation is ingested from the Open-Meteo historical reanalysis (free, no
key) for the ~91 0.1-degree coastal grid cells onto which the 800 geolocated stations collapse,
2005-2026, and joined causally to each station-day. River first-flush is ingested from USGS
NWIS daily discharge for the nearest data-bearing stream gauge (California has 2,400+; 91% of
test station-days fall within range), with anomaly and days-since-high-flow features.

---

## 3. METHODS

**3.1 Causal features and the lab-reveal lag.**
All features use only information available at sampling time. Prior-laboratory and
station-memory features (previous result, rolling exceedance memory, expanding station rate,
days-since-prior) are constructed against an enforced REVEAL LAG: the "previous result" for a
sample at time t is the latest prior sample whose laboratory result had RETURNED by t
(sample_date <= t - lag), not simply the previous sample by index. With a default 2-day lag,
a same-day or next-day re-sample — common after a suspected exceedance — can no longer feed a
label that had not yet come back. We verify that lag = 0 reproduces the naive shift-by-one
behavior exactly, isolating the effect of the lag. Rainfall and discharge windows are
backward-looking and known at sampling time.

**3.2 Baseline ladder (what a model must beat).**
- Climatology / base rate and month-of-year seasonality.
- Prior-lab: "was it over the standard last time?" (the operational status quo).
- Station memory: the expanding prior exceedance rate at the station ("a recently dirty beach
  tends to stay dirty") — a strong, well-calibrated baseline.
- AB411 rain rule: advise if >= 0.1 inch (2.54 mm) fell in the prior ~3 days.
- VIRTUAL-BEACH-CLASS MLR: a per-station logistic regression on causal hydrometeorological
  predictors (rainfall windows, antecedent dry days, season, prior lab) with a pooled fallback
  for sparse stations — a faithful reimplementation of EPA Virtual Beach's METHOD (not its
  software). This represents deployed predictive practice.

**3.3 Model and calibration.**
A histogram gradient-boosted decision-tree classifier (class-weighted) is trained on data
through 2019. Because the raw probabilities are inflated, we fit an isotonic calibration map on
a held-out 2020-2021 validation era and apply it to the 2022+ test era. We additionally
implement a PREQUENTIAL (online) recalibrator that, for each forward time block, refits the
isotonic map using only labels already revealed by then (respecting the reveal lag) — the
realistic continual-learning setting in which laboratory results keep arriving.

**3.4 Evaluation.**
We report Average Precision (AP) and AUROC for ranking, recall at a fixed 20% false-alarm
budget for the operational decision, and Brier score and Expected Calibration Error (ECE) for
probability quality. Crucially, we STRATIFY the test set by region (all / excluding San Diego /
San Diego only / Monterey Bay) so that a regional regime break is visible, not absorbed into a
pooled number. Temporal split: train <= 2019, calibrate 2020-2021, test >= 2022.

**3.5 Spatial generalization.**
Temporal hold-out answers "predict a known beach's future." To answer the harder, deployment-
relevant question "predict a beach never seen in training," we run LEAVE-ONE-COUNTY-OUT:
train on all other counties (and calibrate on their 2020-2021 data), test on the held-out
county's 2022+ samples. The statewide prior-rate feature is recomputed per fold EXCLUDING the
held-out county to remove a cross-boundary leak.

---

## 4. RESULTS

**4.1 Headline (honest statewide; marine enterococcus; calibrated; reveal-lag enforced).**

```
Stratum: EXCLUDE_SAN_DIEGO (n=89,321; base rate 0.10)
Method                              AP       AUROC    ECE
---------------------------------   ------   ------   ------
AB411 rain rule                     0.178    0.650      -
Station memory                      0.278    0.772    0.011
Virtual-Beach-class MLR (deployed)  0.375      -        -
Model (HGBT, isotonic-calibrated)   0.497    0.858    0.020   <-- deploy-ready
```

The calibrated model beats the AB411 rule by ~2.8x AP, the deployed-practice Virtual-Beach-
class MLR by +0.12 AP (~32%), and station memory by +0.22 AP — and it is well calibrated
(ECE 0.020, near the best baseline). For the Monterey Bay region: AP 0.400 / AUROC 0.794 /
ECE 0.012, again beating all three baselines and deploy-ready. (Pooled over all regions the
model reaches AP 0.745 / AUROC 0.918, but we do NOT headline that number; see 4.4.)

**4.2 Calibration is the deploy gate.**
The raw class-weighted model RANKS well but is badly miscalibrated (ECE 0.21-0.30); its
probabilities cannot drive an advisory threshold as-is. A single isotonic map fit on the
validation era restores deploy-grade calibration (ECE -> ~0.020) while preserving ranking and
improving recall at a fixed false-alarm budget. The lesson: ranking lift without calibration is
not deploy-readiness — a point under-reported in prior FIB-ML work.

**4.3 Spatial generalization (leave-one-county-out).**
Trained on every OTHER county and tested on a held-out county's 2022+ samples, the calibrated
model beats the AB411 rule in 9/9 held-out counties and station memory in 9/9 (median held-out
AP 0.522; deploy-ready in 8/9; the lone non-deploy case is San Diego, the 2022 regime-broken
county where calibration fails, Section 4.4). The result is a transferable nowcasting pattern,
not memorized known beaches.

**4.4 Honest negative / artifact findings.**
- REGIME BREAK. The train-to-test base-rate jump that inflates pooled metrics is almost
  entirely enterococcus, concentrated in San Diego, stepping exactly at the 2022 split — the
  signature of the Tijuana-River trans-boundary sewage crisis plus a likely method/units
  change in the right tail. A pooled AUC measured across this discontinuity partly scores the
  artifact. We therefore exclude San Diego from the headline and report it as a separate
  stratum; the AB411 rule is a coin-flip there (AUROC ~0.49), because the contamination is
  chronic sewage, not rainfall-driven runoff.
- RIVER DISCHARGE IS RAIN-REDUNDANT. Adding USGS first-flush discharge yields only a marginal
  lift on the honest statewide stratum (~+0.018 AP) and slightly hurts the pooled number;
  rainfall already encodes most of the first-flush signal. We keep it as an optional feature
  but it is not the lever.
- ONLINE RECALIBRATION IS REGIME-LOCAL. A static calibrator degrades under the San Diego
  regime shift (ECE -> 0.32). Prequential recalibration substantially repairs San Diego
  (ECE -> 0.18) but DEGRADES already-stable strata (refitting adds variance where the static
  map was already good). Recalibration should therefore be drift-triggered, not global — an
  empirical motivation for continual-learning approaches keyed to detected distribution shift.

**4.5 Companion negative result (forecasting vs nowcasting).**
For contrast, single-station ocean-state FORECASTING at a fixed mooring is persistence-ceilinged
(hourly autocorrelation ~0.99); a battery of statistical, gradient-boosted, neural, and
foundation time-series models fails to beat the better of persistence and seasonal-naive at the
median across horizons, and the entire +72-hour tier is rejected. We report this null plainly:
it delimits where additional compute cannot help and underscores that the tractable, fundable
problem is the nowcasting DECISION gap, not mooring forecasting.

---

## 5. LIMITATIONS

(1) The lab-reveal lag is modeled as a fixed 2-day offset against sample date; the true
laboratory publication timestamp would be preferable and is the subject of partner data access.
(2) Rainfall is gridded at 0.1 degrees (~11 km), not at the surf zone. (3) Evaluation is
retrospective on >=2022 data; a prospective, forward-time pilot (freeze the model; score the
next season on beaches not used for tuning) is the deployment evidence and is future work.
(4) Spatial validation now includes **leave-one-BEACH-out** (GroupKFold over station id, with
the county/statewide prior-rate aggregates recomputed per fold to exclude the held-out beaches):
on beaches never seen in training, the San-Diego-excluded model holds at **AP 0.464 / ROC-AUC
0.850 / ECE 0.010** — within noise of the temporal-holdout headline — and still beats station
memory (AP 0.278) and AB411 (AP 0.178), so the skill is not an artifact of co-located stations
leaking across folds. **However**, Moran's I on the per-beach residuals is **0.25 (one-sided
p=0.002; expected ~0)** San-Diego-excluded (0.38 pooled), i.e. residuals remain spatially
clustered: there is genuine sub-county spatial structure the current features do not capture,
motivating explicit spatial / hierarchical features. Harness: `research/bacteria/spatial_autocorr.py`
(unit-tested in `tests/test_spatial_autocorr.py`).

**Acting on (4): a learned spatial risk surface.** Adding raw station coordinates (latitude,
longitude) to the gradient-boosted model lets it learn a spatial risk surface directly. On the
San-Diego-excluded enterococcus holdout this lifts **AP 0.461 -> 0.494** (+0.033; +nbr&latlon
0.497) and reduces residual Moran's I **0.240 -> 0.198**, with calibration preserved (ECE ~0.009).
Crucially the lift **generalizes to never-trained-on beaches** (leave-one-beach-out AP
0.464 -> 0.498, +0.034, positive and stable across 5 resampling seeds, +0.022..+0.034) and is
**not a county proxy** (it persists at +0.027 with `cty_prev7` removed) -- i.e. an interpolable
spatial field, not per-station memorization. This is the basis for a **risk estimate at
unmonitored beaches**. Honest bounds: the gain is **rainfall-conditioned** (smaller without rain
features); a causal k-nearest-beach *dynamic* lag (neighbours' recent exceedance) was a **wash**
(regional state is already captured by the county/statewide rate); and the residual clustering is
**reduced, not eliminated** (Moran's I 0.198 is still ~9 sigma above the permutation null, p=0.002),
so physical spatial covariates (outfall / river-mouth distance, embayment) remain the next lever.
Harness: `research/bacteria/spatial_drivers_experiment.py` (tests `tests/test_spatial_drivers.py`).

(5) Much of the achievable skill is carried by station memory; the fundable
scientific claim is precisely "beat per-site practice using environmental drivers," which the
present results support but which deserves driver-by-driver attribution.

---

## 6. REPRODUCIBILITY

All features are leakage-controlled by construction (reveal-lag-enforced prior labels;
per-fold exclusion of the held-out region from pooled features); the data layer is a portable
medallion (bronze/silver/gold) lakehouse with content-hash source provenance and deterministic
splits; and all environmental drivers come from free, public sources (Open-Meteo historical
reanalysis; USGS NWIS). The benchmark harness — label, baselines (including the Virtual-Beach-
class MLR), stratified evaluation, leave-one-county-out, and prequential recalibration — is
released with unit tests, including a causality test that corrupting the most-recent revealed
labels does not change earlier predictions.

A one-command reproduction (data, pinned environment, exact commands, and expected outputs) is
included alongside this draft in `REPRODUCE.md`.

---

## 7. CONCLUSION

On a statewide California benchmark, a calibrated nowcasting model beats deployed predictive
practice (a Virtual-Beach-class per-site regression) and the AB411 regulatory rule on the marine
enterococcus single-sample standard, and generalizes to unseen counties — under leakage-clean,
reveal-lag-enforced, region- and era-stratified, spatially-held-out evaluation. Equally
important, we report what does not work: a sewage regime break that must be excluded, a river-
discharge driver that is rain-redundant, and a static calibrator that fails under drift. We
argue that the next contribution for the field is not a higher pooled AUC but exactly this kind
of honest, deployment-aligned benchmark, and that the path to impact runs through a forward-time
pilot with a public-health partner rather than through additional model capacity.
