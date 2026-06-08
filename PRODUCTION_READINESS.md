# MBARI Production Readiness Register

Last verified: 2026-06-08

## Decision

Status: **local production gate passed; portable smoke validation passed in an isolated path.**

The project is now shaped as a production candidate: source control is initialized,
generated artifacts are ignored, unit tests pass, release-gate tooling exists, and the
portable CPU smoke job succeeded from `C:\mbari_portable_smoke`. It is not approved for
global SOTA claims; promotion remains target/horizon-specific and driver-enabled claims
must be rerun if they depend on the rebuilt driver cache.

## Verified Local State

- Editable install: `python -m pip install -e .[test]` succeeded.
- Test suite: `python ops\run_tests.py` passes in the current local verification.
- Release gate: `python release_gate\mbari_release_gate.py --project-root . --output-dir release_gate\reports` returns `PASS`.
- Senior gate: `python release_gate\mbari_sr_manager_gate.py --project-root . --output release_gate\reports\sr_manager_gate_report.json --no-exit-code` returns `PASS`.
- GPU stack: RTX 5090 CUDA/Torch/XGBoost checks pass in the release gate.
- Curated dataset: validation report matches 3,807,370 expected rows across 351 Parquet files.
- Promotion matrix: 4,950 rows with 39 `promote`, 12 `candidate_split_mismatch`, 717 `reject`, and 4,182 `insufficient_data` after the 2026-06-08 split-closure and Chronos shared-split reruns.
- Portable smoke: `python .\mbari_train.py --jobs .\ops\jobs.smoke.json --accel cpu --no-wake-lock --no-gpu-preflight --force` passed from `C:\mbari_portable_smoke\src` with `scored_rows=4`, `summary.json`, `leaderboard.csv`, `cv_predictions.parquet`, and lakehouse artifacts.
- Artifact audit: smoke outputs contain no absolute project/data paths from the local working tree. The raw process log contains the local Python installation path from a third-party warning; this is runtime-environment noise, not a project artifact dependency.

## Defensible Model Claims

- XGBoost forecast-v2 is the current leakage-corrected production-style baseline.
- PatchTST is the best aggregate full-history neural run in `nn_results`.
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
4. A true external runner should repeat the portable smoke with its own storage paths before external deployment, even though the isolated local-path smoke passed.

## Required Production Commands

Local gate:

```powershell
python .\ops\run_tests.py
python .\release_gate\mbari_promotion_matrix.py --project-root . --output-dir lakehouse\gold\promotion_matrix
python .\release_gate\mbari_release_gate.py --project-root . --output-dir release_gate\reports
```

Portable external-environment gate:

```powershell
python .\mbari_train.py --jobs .\ops\jobs.smoke.json --accel cpu --no-wake-lock --no-gpu-preflight --force
```

## Source-Control Policy

- Track source, tests, runbooks, portable job configs, and small audit reports.
- Do not track generated Parquet, TensorBoard logs, neural result directories, lakehouse outputs, local caches, or unrelated scratch projects.
- Commit only after reviewing `git status --short --ignored` for accidental generated artifacts.
