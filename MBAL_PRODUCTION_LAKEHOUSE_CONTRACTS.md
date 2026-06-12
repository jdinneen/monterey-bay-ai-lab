# Monterey Bay AI Lab Production Lakehouse Contracts

Generated: 2026-06-07

This project uses a production-grade local lakehouse architecture. Neural runs emit
structured artifacts under `lakehouse/` ensuring the workflow remains portable to
generic remote compute environments (Spark, Kubernetes, object storage, or managed
runtime platforms) without changing the core modeling contract.

## Layers

- `lakehouse/silver/forecast_splits/`
  Canonical split contracts. Each `split_id` records the expanding-window cutoffs,
  train window bounds, evaluated horizons, source fingerprint, and run configuration.

- `lakehouse/gold/forecast_runs/`
  Run manifests. Each `run_id` records model, loss, driver configuration, source
  fingerprint, cache version, environment knobs, and artifact paths.

- `lakehouse/gold/forecast_metrics/`
  Normalized target/horizon metrics keyed by `run_id`, `split_id`, `unique_id`,
  and `horizon_h`. Carries the honest-baseline columns `seasonal_naive_rmse` and
  `skill_vs_best_naive_pct` (skill vs the better of persistence and seasonal-naive),
  backfilled by `ops/backfill_seasonal_naive.py` from the prediction partitions.

- `lakehouse/gold/forecast_predictions/`
  Per-run prediction partitions keyed by `run_id`. These preserve the original
  `cv_predictions.parquet` shape while adding `run_id`, `split_id`, and prediction
  column metadata.

## Architectural Tier Mapping

- **Bronze (Raw/Ingest):**
  Immutable copies of raw MBAL source data and external driver inputs.

- **Silver (Curated/Enriched):**
  QC-applied hourly matrix, observed mask, driver availability table, and
  `forecast_splits`. These are ready for modeling and feature engineering.

- **Gold (Aggregated/Insights):**
  `forecast_runs`, `forecast_metrics`, `forecast_predictions`, and final comparison
  leaderboards. These are consumer-ready artifacts for scientists and managers.

## Mentality Over Product

- **Compute Portability:**
  One build-data task, fan-out model/config tasks, then a merge/report task. The
  `split_id` and `cache_version` are task-invariant values shared across runs.

- **Experiment Tracking:**
  Each `run_id` maps to a standard experiment tracking entry. Manifests record
  params/tags, `forecast_metrics` record performance, and Parquet files record results.

## Current Guarantees

- Neural scoring only counts originally observed target timestamps and originally
  observed forecast-origin timestamps.
- Gap filling is past-only.
- Split contracts are deterministic for a given data/config/cutoff set.
- Run manifests include source file size and mtime, cache version, model config, loss,
  driver config, environment batch settings, and artifact paths.
- Local scripts honor these environment variables for environment portability:
  - `MBAL_PROJECT_ROOT`
  - `MBAL_SOURCE_PARQUET`
  - `MBAL_CACHE_DIR`
  - `MBAL_LAKEHOUSE_DIR`
  - `MBAL_PYTHON`
- Model comparison output is generated locally at `nn_results/MODEL_COMPARISON.md` (a
  gitignored artifact produced by `ops/report_model_comparison.py`, distributed via the data
  release per `DATA.md`; not committed to the public tree) and separates full-history runs
  from bounded-history driver runs.
- Promotion matrix output supports enforceable promotion only when XGBoost and neural
  rows share the same exported `split_id`.
- Forecast-metric consumers should use the aggregate metrics table or the shared
  `mbal_lakehouse.read_forecast_metrics` helper; naive recursive reads across the
  aggregate table and `run_id=*` partitions double-count evidence.
- Target/horizon champion selection is generated from promotion-gated rows, not
  from global model averages.
- Dependency specs are tracked under `ops/requirements-*.txt`; reproduce a local
  environment freeze with `pip freeze` as needed.
- Isolated portable smoke validation passed from `C:\mbal_portable_smoke`; a true
  external runner still needs to repeat the smoke with its own storage paths.
- Production readiness is tracked in `PRODUCTION_READINESS.md`.

## Remaining Production Gaps

- Source fingerprint now includes a `content_sha256` content hash (stdlib `hashlib`)
  alongside file size/mtime, so source identity no longer depends on filesystem metadata
  alone. Remaining gap: the content hash is computed over the supplied source file(s);
  multi-file / streaming source fingerprinting is still future work.
- Tables are local Parquet. An interim, additive content-addressed snapshot layer now
  provides versioned gold snapshots, but it is not a full transactional table format;
  remote implementations should still adopt a versioned format (e.g., Delta or Iceberg)
  along the documented upgrade path.
- XGBoost and neural runs do not yet share one externally supplied split table, so
  some promotion rows remain `candidate_split_mismatch`.
- Driver-enabled claims need promotion-gated shared-split evidence before stronger
  driver-value claims. NDBC 46042 wave-driver analyses were rerun under the rebuilt
  manifest on 2026-06-08 and remain research-only because the bounded screen did
  not support a general wave-driver claim.
- Remote-backed output path validation on a true external runner is still required
  before external deployment.

