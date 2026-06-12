# GCP Portable Smoke Runbook (PENDING EXECUTION)

Status: **runbook only -- this smoke has NOT been executed against object storage yet.**

This runbook describes how to run the *existing* portable CPU smoke
(`ops/jobs.smoke.json` via `mbal_train.py`) on a generic managed runtime or VM with
its inputs and outputs on object storage (`gs://` paths). It does not assert that any
external run has completed; it is the procedure a true external runner should follow to
clear the "external runner must repeat the portable smoke" production blocker.

Architecture note: this is product-neutral. The target is any managed compute that can
run Python and read/write object storage (a VM, a Kubernetes job, or a managed-runtime
notebook). No proprietary lakehouse product is required.

## What the smoke validates

`ops/jobs.smoke.json` runs a single tiny CPU-safe job (`smoke_patchtst`, `--smoke`,
`--max-steps 10`, `--n-windows 3`, `--series air_pressure`). A successful run must emit:

- `summary.json` (with a non-zero `scored_rows`)
- `leaderboard.csv`
- `cv_predictions.parquet`
- lakehouse artifacts under the configured lakehouse dir
  (`silver/forecast_splits/`, `gold/forecast_runs/`, `gold/forecast_metrics/`,
  `gold/forecast_predictions/`)

## Prerequisites (do NOT create cloud resources as part of this runbook)

1. A managed runtime or VM with Python 3 and the project's CPU dependencies installed
   (`pip install -e .[test]` from the project source).
2. An **existing** object-storage bucket you can read and write. This runbook never
   creates buckets or compute; provision those out of band per your org's policy.
3. GCP credentials available to the runtime (workload identity, attached service
   account, or `GOOGLE_APPLICATION_CREDENTIALS`). Set the project via
   `MBAL_GCP_PROJECT` -- never bake a project id into source.
4. Cloud Storage FUSE (gcsfuse) or an equivalent object-storage mount **if** you plan
   to point the `MBAL_*` path variables directly at `gs://` paths. The smoke reads and
   writes ordinary filesystem paths, so a mounted bucket is the simplest portable route;
   alternatively, stage inputs locally and use `ops/gcs_mirror.py` to push outputs.

## Step 1 -- Stage the source data to object storage (optional, dry-run first)

If your source Parquet and lakehouse seed live locally, preview and then push them:

```bash
# Dry-run: prints the planned object operations and total bytes, uploads nothing.
python ops/gcs_mirror.py --bucket "$MBAL_GCS_BUCKET" --prefix mbal

# When the plan looks right, perform the upload (requires credentials + existing bucket).
python ops/gcs_mirror.py --bucket "$MBAL_GCS_BUCKET" --prefix mbal --execute
```

## Step 2 -- Point the portable path variables at object storage

The smoke honors these environment variables (see
`MBAL_PRODUCTION_LAKEHOUSE_CONTRACTS.md`). Set them to mounted object-storage paths
(via gcsfuse) or to a local staging dir that you later mirror with `gcs_mirror.py`:

```bash
export MBAL_GCP_PROJECT="<your-project-id>"        # never hard-coded in source
export MBAL_PROJECT_ROOT="/gcs/<bucket>/mbal/src"
export MBAL_SOURCE_PARQUET="/gcs/<bucket>/mbal/bronze/<source>.parquet"
export MBAL_CACHE_DIR="/gcs/<bucket>/mbal/cache"
export MBAL_LAKEHOUSE_DIR="/gcs/<bucket>/mbal/lakehouse"
```

`/gcs/<bucket>/...` denotes a gcsfuse mount of `gs://<bucket>/...`. If you cannot mount
the bucket, set these to a local working dir and mirror the results afterward with
`ops/gcs_mirror.py`.

## Step 3 -- Run the existing portable smoke

```bash
python ./mbal_train.py \
  --jobs ./ops/jobs.smoke.json \
  --accel cpu \
  --no-wake-lock \
  --no-gpu-preflight \
  --force
```

This is the same invocation recorded in `PRODUCTION_READINESS.md`, unchanged. Only the
storage paths differ.

## Step 4 -- Validate artifacts

Confirm all required artifacts exist on object storage and that scoring happened:

```bash
# Adjust the smoke outdir/prefix to match jobs.smoke.json's outdir.
ls "$MBAL_LAKEHOUSE_DIR"/silver/forecast_splits
ls "$MBAL_LAKEHOUSE_DIR"/gold/forecast_runs
ls "$MBAL_LAKEHOUSE_DIR"/gold/forecast_metrics
ls "$MBAL_LAKEHOUSE_DIR"/gold/forecast_predictions

# Inspect the run summary: scored_rows must be > 0.
python -c "import json,sys; d=json.load(open(sys.argv[1])); print('scored_rows=', d.get('scored_rows')); assert d.get('scored_rows',0) > 0" <path-to>/summary.json

# Confirm leaderboard + predictions landed.
ls <path-to>/leaderboard.csv <path-to>/cv_predictions.parquet
```

Pass criteria: `summary.json` shows `scored_rows > 0`, `leaderboard.csv` and
`cv_predictions.parquet` are present, and the four lakehouse layers above contain the
new run's partitions.

## Step 5 -- Tear down

Stop the VM / job and remove any temporary mounts. Do not delete the bucket from this
runbook; bucket lifecycle is managed out of band.

## Reporting

Until this procedure has been executed end-to-end on a true external runner and the
pass criteria above are met, the external-smoke production blocker in
`PRODUCTION_READINESS.md` remains **OPEN**. Record the run id, the object-storage paths,
and the validated `scored_rows` when it is completed.
