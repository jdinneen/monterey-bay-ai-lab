# Monterey Bay AI Lab Production Readiness Register

Last verified: 2026-06-08

## Decision

Status: **local production gate passed; portable smoke validation passed in an isolated path.**

The project is now shaped as a production candidate: source control is initialized,
generated artifacts are ignored, unit tests pass, release-gate tooling exists, and the
portable CPU smoke job succeeded from `C:\mbal_portable_smoke`. It is not approved for
global SOTA claims; promotion remains target/horizon-specific and driver-enabled claims
must be rerun if they depend on the rebuilt driver cache.

## Verified Local State

- Editable install: `python -m pip install -e .[test]` succeeded.
- Test suite: `python ops\run_tests.py` passes in the current local verification.
- Release gate: `python release_gate\mbal_release_gate.py --project-root . --output-dir release_gate\reports` returns `PASS`.
- Senior gate: `python release_gate\mbal_sr_manager_gate.py --project-root . --output release_gate\reports\sr_manager_gate_report.json --no-exit-code` returns `PASS`.
- GPU stack: RTX 5090 CUDA/Torch/XGBoost checks pass in the release gate.
- Curated dataset: validation report matches 3,807,370 expected rows across 351 Parquet files.
- Promotion matrix: 4,950 rows with 39 `promote`, 12 `candidate_split_mismatch`, 717 `reject`, and 4,182 `insufficient_data` after the 2026-06-08 split-closure and Chronos shared-split reruns.
- Portable smoke: `python .\mbal_train.py --jobs .\ops\jobs.smoke.json --accel cpu --no-wake-lock --no-gpu-preflight --force` passed from `C:\mbal_portable_smoke\src` with `scored_rows=4`, `summary.json`, `leaderboard.csv`, `cv_predictions.parquet`, and lakehouse artifacts.
- Artifact audit: smoke outputs contain no absolute project/data paths from the local working tree. The raw process log contains the local Python installation path from a third-party warning; this is runtime-environment noise, not a project artifact dependency.
- **Hardware Safety**: Training is protected by `ops/mbal_sota_utils.py:SafetyMonitor`, enforcing an 80% VRAM ceiling, 72h runtime limit, and 30m idle shutdown.

## Fundable Headline vs Credibility Substrate

- The strongest fundable headline is **statewide bacteria classification (San-Diego-excluded
  ROC-AUC 0.858 / AP 0.497, calibrated ECE 0.020, 9/9 leave-one-county-out wins)**,
  not M1 hourly forecasting. (The retired "AUC ~0.888" was the artifact-inflated pooled
  /multi-analyte figure distorted by the San Diego/Tijuana 2022 regime break.) M1 hourly forecasting is **persistence-ceilinged**: most
  target/horizon cells lose to seasonal-naive (same-hour-yesterday), and the entire +72h
  tier was rejected because the models cannot beat the free diurnal baseline there. Frame
  M1 forecasting as the **credibility substrate** -- the rigorously gated, leakage-checked
  evidence base that earns trust -- and lead external messaging with the bacteria result.

## Lakehouse Hardening Updates (2026-06-08)

- **Content-addressed source fingerprint**: the source fingerprint now includes a
  `content_sha256` (stdlib `hashlib`) in addition to size/mtime (workstream A), so source
  identity no longer depends on filesystem metadata alone.
- **Versioned gold snapshots**: an additive, content-addressed versioned snapshot layer
  now exists for the gold tables (workstream B). It is interim and does not replace the
  documented Delta/Iceberg upgrade path for remote deployments.
- **GCS mirror + GCP portable smoke (PENDING EXECUTION)**: `ops/gcs_mirror.py`
  (dry-run by default) and `ops/gcp_portable_smoke_runbook.md` now exist to mirror the
  lakehouse to object storage and to run the existing portable smoke on a generic managed
  runtime. **Execution is still pending** -- no upload has been performed and the external
  object-storage smoke has not been run. The external-smoke blocker below is NOT resolved.

## Defensible Model Claims

- XGBoost forecast-v2 is the current leakage-corrected production-style baseline.
- PatchTST is the best aggregate full-history neural run in `nn_results`.
- **Neural / MoE / EWC / LatentTSF: NOT a defensible capability — research-only.** Per the
  Value-Gate verdict (`research/model_lab/NEURAL_TRACK_VALUE_GATE.md`, 2026-06-10) the
  continual-learning stack beats the best-naive baseline on only **1/24 series (4.2%)**, median
  skill **−0.78 to −0.88**, on an *in-sample* eval. It is consistent with the M1 persistence
  ceiling and must not be presented as "MoE Advantage" or "Incremental SOTA." Re-promotion
  requires a pre-registered temporal-holdout win over best-naive.
- Chronos/TimesFM are the strongest exploratory zero-shot foundation models at +6h in the current benchmark.
- The 39 promotion rows are **target/horizon-level** promotions, not a global model promotion.
  They collapse to 12 unique target/horizon cells and 34 unique model cells in the
  current matrix, so row counts must not be described as independent scientific wins.
- Promotions are now gated on the **best-naive** baseline (the better of persistence and
  seasonal-naive / same-hour-yesterday), computed per cell from gold prediction partitions
  by `ops/seasonal_naive.py` and surfaced as `seasonal_naive_rmse` /
  `skill_vs_best_naive_pct`. Every promoted cell sits at +1h/+6h/+24h; the entire +72h tier
  was rejected because the models lose to seasonal-naive there (persistence alone is a free
  lunch at diurnal horizons). This matches `research/model_lab/HONEST_SKILL_BASELINE.md`.
- **Strict ML Value Gate (Bacteria/Lakehouse):** Any new model attempting to predict bacteria
  exceedances (or any spatio-temporal target) must **never** use random `train_test_split`.
  It must be evaluated using strict chronological holdouts (e.g., train < 2022, test >= 2022) 
  or spatial leave-one-county-out CV. Furthermore, it must explicitly report the majority-class
  base rate and mathematically prove it beats a naive "always guess safe" baseline. Models failing
  this gate are considered invalid bloat and will be rejected.

## Claims Not Yet Approved

- No single global project SOTA model is established.
- Chronos foundation rows now have shared-split rerun support where exported split contracts exist; remaining foundation screening rows are TimesFM or internal-XGB cases that still need a dedicated rerun path.
- Driver-value claims are screening-level only unless they pass the promotion matrix.
  The driver cache was rebuilt with a 1-hour hourly observed-driver availability lag,
  1-day daily observed-driver lag, and bounded wave-driver cleaning/staleness policy.
- NDBC 46042 wave drivers were rerun under the rebuilt manifest on 2026-06-08.
  The bounded screen produced `not_supported_as_general_claim`: wave drivers won
  121/240 cells (50.4%) with mean RMSE delta +1.825% and mean skill delta -1.991 pp.
  Wave-driver language must stay research-only unless repeated shared-split evidence
  clears the claim gate.
- Bounded-history driver runs must not be compared as full-history equivalents without explicit approval.

## Production Blockers

1. The remaining 12 split-mismatch candidates need shared-split reruns or explicit rejection.
2. Driver-enabled analyses need promotion-gated shared-split evidence before stronger driver-value claims; wave-driver analyses are explicitly research-only after the 2026-06-08 bounded rerun failed the general-claim bar.
3. Driver-model production policy still needs explicit approval for bounded-history comparisons.
4. A true external runner should repeat the portable smoke with its own storage paths
   before external deployment, even though the isolated local-path smoke passed.
   `ops/gcp_portable_smoke_runbook.md` and `ops/gcs_mirror.py` now provide the procedure
   and tooling, but this blocker remains **OPEN** until the runbook is executed end-to-end
   on an external runner and its pass criteria are met.

## Required Production Commands

Local gate:

```powershell
python .\ops\run_tests.py
python .\release_gate\mbal_promotion_matrix.py --project-root . --output-dir lakehouse\gold\promotion_matrix
python .\release_gate\mbal_release_gate.py --project-root . --output-dir release_gate\reports
```

Portable external-environment gate:

```powershell
python .\mbal_train.py --jobs .\ops\jobs.smoke.json --accel cpu --no-wake-lock --no-gpu-preflight --force
```

## Source-Control Policy

- Track source, tests, runbooks, portable job configs, and small audit reports.
- Do not track generated Parquet, TensorBoard logs, neural result directories, lakehouse outputs, local caches, or unrelated scratch projects.
- Commit only after reviewing `git status --short --ignored` for accidental generated artifacts.

