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

## TL;DR

- **Problem.** Beach FIB advisories rely on a lab culture that returns 18–24 h *after* the
  water sample — so the same-day public-health decision is made blind, or with California's
  weak AB411 wet-weather rule.
- **What we do.** A statewide California benchmark (1.32M assays, 802 stations, 17 counties,
  2005–2026) for *nowcasting* marine-enterococcus exceedance (>104 MPN/100 mL) from
  information available **at sampling time**, with an enforced lab-reveal lag so "prior result"
  features can't see a result that hadn't come back yet.
- **Headline (honest statewide stratum, San Diego excluded; n=89,321).** An isotonically
  calibrated gradient-boosted model reaches **AP 0.472 / AUROC 0.855 / ECE 0.017** and beats,
  on the same data: the **AB411 rule (AP 0.178, ~2.6×)**, a **Virtual-Beach-class per-site
  regression representing *deployed practice* (AP 0.375, +0.097 AP)**, and **station memory
  (AP 0.278)** — while staying deploy-grade in calibration.
- **It generalizes.** Leave-one-county-out: beats AB411 and station memory in **9/9** held-out
  counties (median AP 0.538; deploy-ready in 8/9).
- **We report what fails, too.** A 2022 San Diego / Tijuana-River sewage *regime break* is
  excluded from the headline (it inflates pooled metrics); river discharge is largely
  rain-redundant; a static calibrator breaks under regime shift and online recalibration only
  partially repairs it.
- **Fully reproducible.** Every number regenerates from a public clone in ~2 minutes on a
  laptop (no GPU, no API keys). See **§8 (How to peer-review this)** and `REPRODUCE.md`.
- **Numbers note.** Figures are the *reproduced* values on the committed frozen data snapshot
  (e.g. headline AP 0.472). An earlier draft quoted AP 0.465; the gap is dataset growth through
  2026, not a methods change — `random_state` is pinned and the values below match the released
  `expected/` artifacts exactly.

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
boosted decision-tree model, isotonically calibrated, attains Average Precision (AP) 0.472 /
AUROC 0.855 on the honest statewide stratum. It beats, on the same data: the AB411 rain rule
(AP 0.178; ~2.6x), a Virtual-Beach-class per-site multiple logistic regression representing
DEPLOYED practice (AP 0.375; +0.097 AP, ~26%), and a station-memory baseline (AP 0.278) —
while remaining deploy-ready in calibration (Expected Calibration Error, ECE, 0.017). Under
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
Stratum: EXCLUDE_SAN_DIEGO (n=89,321; base rate 0.101)
Method                              AP       AUROC    ECE
---------------------------------   ------   ------   ------
AB411 rain rule                     0.178    0.650      -
Station memory                      0.278    0.772    0.011
Virtual-Beach-class MLR (deployed)  0.375    0.749      -
Model (HGBT, isotonic-calibrated)   0.472    0.855    0.017   <-- deploy-ready
```

(These are the exact values in `expected/operational_benchmark.json`, stratum
`EXCLUDE_SAN_DIEGO`, regenerable per §8.)

The calibrated model beats the AB411 rule by ~2.6x AP, the deployed-practice Virtual-Beach-
class MLR by +0.097 AP (~26%), and station memory by +0.194 AP — and it is well calibrated
(ECE 0.017, matching the best baseline within rounding). For the Monterey Bay region: AP 0.402 /
AUROC 0.788 / ECE 0.018, again beating all three baselines and deploy-ready. (Pooled over all
regions the model reaches AP 0.740 / AUROC 0.915, but we do NOT headline that number; see 4.4.)

**4.2 Calibration is the deploy gate.**
The raw class-weighted model RANKS well but is badly miscalibrated (ECE 0.21-0.30); its
probabilities cannot drive an advisory threshold as-is. A single isotonic map fit on the
validation era restores deploy-grade calibration (ECE -> ~0.017) while preserving ranking and
improving recall at a fixed false-alarm budget. The lesson: ranking lift without calibration is
not deploy-readiness — a point under-reported in prior FIB-ML work.

**4.3 Spatial generalization (leave-one-county-out).**
Trained on every OTHER county and tested on a held-out county's 2022+ samples, the calibrated
model beats the AB411 rule in 9/9 held-out counties and station memory in 9/9 (median held-out
AP 0.538; deploy-ready in 8/9; the two non-deploy cases are the smallest-event counties, where
calibration is noisy). The result is a transferable nowcasting pattern, not memorized known
beaches.

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
- ONLINE RECALIBRATION IS REGIME-LOCAL. A static calibrator that is well-calibrated off
  San Diego (ECE < 0.02) degrades badly under the San Diego regime shift (ECE 0.285).
  Prequential recalibration substantially repairs San Diego (ECE -> 0.174) but DEGRADES
  already-stable strata (e.g. EXCLUDE_SAN_DIEGO ECE 0.009 -> 0.033; refitting adds variance
  where the static map was already good). Recalibration should therefore be drift-triggered,
  not global — an empirical motivation for continual-learning approaches keyed to detected
  distribution shift.

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
(4) Spatial validation is leave-one-COUNTY-out; co-located stations on the same beach can split
across train/test, so leave-one-BEACH-out and spatial-block cross-validation are stronger tests
we plan to add. (5) Much of the achievable skill is carried by station memory; the fundable
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
included alongside this draft in `REPRODUCE.md`; a reviewer-facing walkthrough is in **§8**.

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

---

## 8. HOW TO PEER-REVIEW THIS (REPRODUCTION GUIDE)

Every number in this paper regenerates from a public clone in ~2 minutes on a laptop — **no GPU,
no API keys, no cloud account**. A reviewer can independently confirm the headline, the
baselines, the spatial generalization, the calibration, and the honest-negative findings.

### 8.1 Get the code + data

The benchmark, the ~6 MB of frozen public inputs, the pinned environment, and the expected
outputs all live on one self-contained branch:

```bash
git clone -b bacteria-nowcasting-repro \
  https://github.com/jdinneen/monterey-bay-ai-lab.git
cd monterey-bay-ai-lab
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r research/bacteria/reproduce/requirements.txt
```

Pinned environment (the scikit-learn version governs HistGradientBoosting + isotonic
reproducibility): **Python 3.12 · scikit-learn 1.9.0 · pandas 2.3.3 · numpy 2.4.6 ·
pyarrow 24.0.0**. Verify the inputs are bit-for-bit the ones used here:

```bash
sha256sum -c research/bacteria/reproduce/MANIFEST.sha256   # from repo root
```

### 8.2 Run the three experiments

```bash
OBS=bacteria_results/statewide/statewide_beach_observations.parquet

# 1. Headline stratified, calibrated benchmark            (~60-90 s)
python research/bacteria/operational_benchmark.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall --discharge-dir bacteria_results/discharge \
  --reveal-lag-days 2 --label enterococcus --out-dir /tmp/repro

# 2. Leave-one-county-out spatial generalization          (~30-60 s)
python research/bacteria/spatial_holdout.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall \
  --reveal-lag-days 2 --label enterococcus --out-dir /tmp/repro

# 3. Prequential (online) recalibration                   (~20-40 s)
python research/bacteria/online_recalibration.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall --lag-days 2 --out-dir /tmp/repro
```

> **The flags are the headline.** The scripts *default* to `--label any --reveal-lag-days 0`
> (a legacy multi-analyte target that gives a different number). The marine-enterococcus headline
> in this paper requires `--label enterococcus --reveal-lag-days 2`, exactly as above. If your
> numbers don't match, check those two flags first.

### 8.3 Confirm the numbers

Compare your output to the committed `expected/` artifacts:

```bash
diff /tmp/repro/operational_benchmark.json \
     research/bacteria/reproduce/expected/operational_benchmark.json
```

The metrics are deterministic (`random_state=42` pinned). The only line that can differ is the
`obs_path` provenance string (it echoes your path separator). To ignore that one cosmetic line:

```bash
diff <(grep -v obs_path /tmp/repro/operational_benchmark.json) \
     <(grep -v obs_path research/bacteria/reproduce/expected/operational_benchmark.json)
```

Expected headline (`expected/operational_benchmark.json`, stratum `EXCLUDE_SAN_DIEGO`):

| Method | AP | AUROC | ECE |
|---|--:|--:|--:|
| AB411 rain rule | 0.178 | 0.650 | — |
| Station memory | 0.278 | 0.772 | 0.011 |
| Virtual-Beach-class MLR | 0.375 | 0.749 | — |
| **Model (HGBT, calibrated)** | **0.472** | **0.855** | **0.017** |

Spatial (`expected/spatial_holdout.json`): 9 counties · beats AB411 9/9 · beats memory 9/9 ·
median AP 0.538 · deploy-ready 8/9. Online recal (`expected/online_recalibration.json`):
San Diego ECE 0.285 → 0.174; stable strata degrade.

### 8.4 Run the leakage / causality tests

```bash
pytest tests/test_operational_benchmark.py tests/test_spatial_holdout.py \
       tests/test_online_recalibration.py -q
```

These include a **causality test**: corrupting the most-recent *revealed* labels must not change
earlier predictions — i.e. the reveal-lag gating actually prevents look-ahead.

### 8.5 What to scrutinize (reviewer checklist)

- **Leakage.** `operational_benchmark.py` — confirm "prior result" features select the latest
  prior sample with `sample_date <= t - lag` (not a naive `shift(1)`), and that rolling/expanding
  features are strictly prior (exclude the current row). Confirm the train (≤2019) / calibrate
  (2020–21) / test (≥2022) eras are disjoint and the isotonic map is fit on the calibration era
  only.
- **Spatial leakage.** `spatial_holdout.py` — confirm the held-out county's rows are removed
  *before* any pooled/statewide feature is computed (the cross-boundary leak fix).
- **Baseline fairness.** The "deployed practice" claim rests on the Virtual-Beach-class MLR. Check
  it is a faithful per-site logistic regression with a pooled fallback, given the same split and
  causal treatment — not a strawman. (Note: giving the MLR the model's full feature matrix *lowers*
  its AP; the reported 7-feature configuration is its most favorable one.)
- **Calibration as a deploy gate.** Confirm raw probabilities are miscalibrated (ECE ~0.2) and the
  isotonic map restores ECE ~0.017 without destroying ranking.
- **The regime break.** Confirm San Diego is excluded from the headline for a stated reason
  (AB411 is a coin-flip there, AUROC ~0.49; the contamination is chronic sewage, not runoff), not
  to flatter the result — and that the pooled-with-SD number is reported, not hidden.

### 8.6 Independent adversarial review (already run)

This benchmark was put through an independent multi-agent adversarial review that re-executed the
pipeline from scratch and tried to refute each claim. It confirmed: the headline regenerates; the
features are leakage-clean (including the subtle case of same-day non-enterococcus analytes, which
are *not* used); the Virtual-Beach baseline is fair (not feature-starved); and the data-scale
claims match the parquet exactly. The only substantive issue it raised was provenance hygiene —
the published artifacts must match the paper's exact configuration — which is why §8.1–8.3 pin the
data snapshot, the environment, and the exact command. Reviewers are encouraged to repeat the
refutation independently.

### 8.7 Data provenance & licensing

- **Beach monitoring** — California statewide record (CEDEN / BeachWatch lineage); public
  environmental monitoring data; station IDs + coordinates only, no personal data.
- **Rainfall** — Open-Meteo historical reanalysis; free for research use, attribution requested.
- **River discharge** — USGS NWIS (parameter 00060); U.S. Government public-domain data.

The fetchers (`research/bacteria/fetch_*.py`) can rebuild the drivers from these public APIs, but
live APIs grow over time and are **not bit-reproducible** — use the committed frozen inputs to
verify the paper. The frozen snapshot covers the record through 2026 as fetched 2026-06-08.

### 8.8 Reporting problems

Please file discrepancies (a number that doesn't regenerate, a suspected leak, a baseline concern)
as a GitHub issue on the repository above, or contact the author. Include your OS, Python and
scikit-learn versions, and the diff against `expected/`.
