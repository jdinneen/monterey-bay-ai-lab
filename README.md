# Monterey Bay AI Lab

**Predicting rare coastal water-quality events at statewide scale — anchored on two decades of Monterey Bay mooring physics.**

The Monterey Bay AI Lab (MBAI) predicts rare, high-consequence coastal events — starting with beach **bacterial exceedances** across California — by fusing large public-health, hydrology, and ocean-physics datasets under strictly causal, leakage-controlled contracts. The 21-year MBARI M1/M2 mooring record is the physical **backbone** that grounds and calibrates the work; the predictive **headline** is event forecasting, where a model can actually beat the baseline.

## Core Capabilities

### 1. Statewide Event-Driven Forecasting — the headline
A multimodal engine predicting rare, non-persistent events (bacterial exceedances) across 800+ California stations.
- **Verified margin:** on a statewide California enterococcus benchmark (1.32M lab **assays** → ~442k modeled station-days, of which ~89k are in the San-Diego-excluded test split; test events ≥2022), a calibrated gradient-boosted model reaches **ROC-AUC 0.858 / AP 0.497** on the **San-Diego-excluded** stratum — the honest headline, since the cross-border San Diego/Tijuana plume is a 2022 data-regime artifact and is reported separately, not pooled. It beats the deployed AB411 rain rule (AP 0.18) and a Virtual-Beach-class MLR (AP 0.38), is well-calibrated after isotonic calibration (ECE 0.020 at the operational 2-day lab-result reveal lag — the tracked, byte-reproducible `expected/` outputs; see `MODEL_SUITE.md`), and generalizes spatially: it beats both the regulatory rule and station-memory in **9/9 leave-one-county-out folds**. Decision-support-grade. (The older "AUC 0.888" headline was the artifact-inflated pooled/multi-analyte figure and has been retired — see `research/bacteria/reproduce/PAPER_DRAFT.md`, which derives the honest San-Diego-excluded stratum and the reveal-lag-enforced metrics above.)
- **The honest frontier:** site memory is a strong baseline (a recently-dirty beach tends to stay dirty; station-memory alone reaches AUC ~0.77 / AP 0.28 on the same San-Diego-excluded stratum). The model already beats it meaningfully (+0.22 AP) with environmental drivers, and — verified — **generalizes to beaches it was never trained on**: under leave-one-beach-out cross-validation the driver model holds **AP 0.494 / ROC-AUC 0.852** on never-seen beaches, still beating station-memory (AP 0.28) and the AB411 rule (AP 0.18), so the skill is a transferable environmental pattern, not per-station memorization. The open frontier is extending risk estimates to the beaches we **don't** monitor. A learned spatial surface from *raw* station coordinates or a nearest-neighbor exceedance lag turns out to be a **wash** — no measurable lift (all spatial arms AP 0.497) because the per-station / county / statewide rate features already encode location — while residual per-beach autocorrelation remains (Moran's I ≈ 0.20, one-sided p<0.01). So **physical spatial covariates (outfall / river-mouth distance, embayment)**, not raw coordinates, are the next lever. Both results reproduce from `research/bacteria/spatial_autocorr.py` (leave-one-beach-out + Moran's I) and `research/bacteria/spatial_drivers_experiment.py` (the spatial-arm wash).

### 2. Foundational Mooring Physics (M1 / M2) — the backbone
21.6 continuous years of hourly MBAL mooring data (temperature, salinity, synoptic met) — the longest, cleanest, QC'd, release-gate-blessed asset in the lab. It anchors the science and supplies physical drivers.
- **Honest scope:** M1/M2 hourly forecasting is **persistence-ceilinged** (acf_1h ≈ 0.99; best models realize only single-digit % skill over a naive baseline). We do **not** spend GPU compute squeezing sub-percent RMSE here — this record's value is calibrated ground-truth and a driver source, **not** a forecasting headline. See `research/model_lab/HONEST_SKILL_BASELINE.md` (skill vs the best-naive baseline, per horizon).

### 3. High-Performance Training (HPT)
We leverage a local **RTX 5090 workstation** for intensive iterative testing.
- **AI is HPC:** We use long-running training jobs to conduct exhaustive A/B testing across hundreds of model architectures and feature combinations.
- **Heuristic Fusion:** We combine pure regression with physical heuristics (like coastal upwelling indices and wave-energy transport) to drive predictive skill.

---

## Technical Achievements (June 2026)

- **Multimodal Connectivity Map:** Validated the additive power of Weather (Rain), Physics (Waves), and Events (News/Spills) against a statewide baseline.
- **Geolocation Unblock:** Successfully mapped **99.75% of statewide stations** to verified coordinates (`reports/station_geo.parquet`), enabling high-precision gridded joins.
- **Verified Skill:** Established a robust operational bar by benchmarking ML models against regulatory physical rules (AB411), ensuring the AI provides real decision-support value.

## The Philosophy: Iterate & Test
We don't settle for single metrics. Some signals may dilute, so we run with and without, keep testing, and refine the heuristics. AI is an experimental science of multivariate connections.

---

## Remote Research & Cost Control

To scale the Monterey Bay AI Lab while protecting local compute resources and managing Google Cloud costs, we utilize a **Push-Only Architecture**.

- **The Forge:** All high-performance training and data normalization happens on our local RTX 5090 workstation. This environment is private; no inbound traffic is allowed.
- **The Evidence Layer:** After local validation, normalized **Gold Artifacts** (Parquet files) are pushed to a public, low-cost Google Cloud Storage (GCS) layer.
- **The API Model:** Public research APIs query the pre-computed GCS artifacts directly. They **never** hit the raw BigQuery database or the local Forge, ensuring $0 database costs for researchers.

For details on how to contribute or recreate our results, see **`REMOTE_RESEARCH_CONTRACT.md`**.

---

## Current Status

The architecture follows a production-grade **Lakehouse Pattern** (Bronze/Silver/Gold) with deterministic split contracts and run manifests.

Authoritative docs:
- `PRODUCTION_READINESS.md`
- `MODEL_SUITE.md` — which models exist, run, beat baselines, and are claimable (generated from the fail-closed `research/model_lab/model_registry.yaml`)
- `MBAL_PRODUCTION_LAKEHOUSE_CONTRACTS.md`
- `docs/open_source_readiness.md`
- `docs/docker.md`

## Install

Create a local Python 3.12 environment, then install the project:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For broader research workflows:

```powershell
python -m pip install -e ".[all]"
```

## Local Verification

Run the stable local test suite:
```powershell
python .\ops\run_tests.py
```

## Docker Verification

Build and run the clean reproducibility image:

```powershell
docker build -t monterey-bay-ai-lab:dev .
docker run --rm monterey-bay-ai-lab:dev
```

The Docker image excludes private data, generated artifacts, local credentials, and workstation caches. See `docs/docker.md` for data mounts and GPU notes.

Launch long-running training:
```powershell
python .\ops\launch_mbal_train_detached.py --wait-lock --enforce-power
```

## Remote Portability
Compute portability is provided by environment variables (`MBAL_PROJECT_ROOT`, `MBAL_SOURCE_PARQUET`) and portable smoke jobs (`ops/jobs.smoke.json`).

## Open Source

This project is licensed under Apache-2.0. Contributors should review `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and `docs/open_source_readiness.md` before preparing public changes.
