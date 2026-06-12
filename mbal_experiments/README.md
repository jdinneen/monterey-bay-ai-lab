# Monterey Bay AI Lab Experiment Tracking

Lightweight local experiment tracking for Monterey Bay AI Lab forecasting and statistical
model runs. It writes append-only JSONL plus a per-run artifact directory and
requires only the Python standard library.

## What It Records

- Run metadata: run id, name, status, start/end time, duration, notes.
- Metrics, parameters, and tags.
- Git state when the workspace is inside a git repository.
- Dataset fingerprints for files, directories, or globs:
  - path, size, modified time
  - SHA-256 content hash
  - manifest hash across all included files
- Environment versions for Python and common ML/data packages.
- GPU metadata from PyTorch when available.
- `nvidia-smi` GPU status when available on `PATH`.

## Layout

By default, running from the repository root writes:

```text
mbal_experiments/
  experiments.jsonl
  runs/
    20260607T...-run-name-.../
      run.json
      metrics.json
      params.json
      tags.json
      datasets.json
      environment.json
      gpu.json
      git.json
      notes.txt
```

## CLI Usage

From the repository root:

```powershell
python -m mbal_experiments.cli record `
  --name "xgboost temp d100 24h" `
  --metric rmse=0.211 `
  --metric r2=0.542 `
  --param model=xgboost `
  --param horizon_hours=24 `
  --tag station=M1 `
  --tag target=temp_d100p0 `
  --dataset "mbal_history/opendap/*.parquet" `
  --notes "Walk-forward validation run"
```

Smoke-test the tracker:

```powershell
python -m mbal_experiments.cli smoke
```

## Python Usage

```python
from mbal_experiments import ExperimentTracker

tracker = ExperimentTracker()

with tracker.start_run(
    "xgboost temp d100 24h",
    params={"model": "xgboost", "horizon_hours": 24},
    tags={"station": "M1", "target": "temp_d100p0"},
    dataset_paths=["mbal_history/opendap/*.parquet"],
) as run:
    # Train and evaluate model here.
    run.log_metrics({"rmse": 0.211, "r2": 0.542})
```

## Smoke Example

```powershell
python mbal_experiments\examples\smoke_record_run.py
```

The smoke example writes a normal tracked run with illustrative metrics and
fingerprints any matching local MBAL parquet/result files.

## Notes

- Hashing is exact up to `512 MiB` per file by default. Larger files are hashed
  up to that limit and marked with `hash_truncated=true`.
- Missing dataset paths are recorded in `datasets.json` instead of failing the
  run. This makes scheduled jobs auditable even when upstream backfills are not
  finished yet.
- The tracker is intentionally separate from model scripts. Existing MBAL
  scripts can opt in later by importing `ExperimentTracker`, but no existing
  script needs to change.

