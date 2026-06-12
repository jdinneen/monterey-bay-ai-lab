# Monterey Bay AI Lab Environment Runbook

## Design

Use separate environments for separate failure domains:

- `mbal-gpu`: model training, GPU benchmarks, forecasting pipelines.
- `geo-ocean`: OPeNDAP, netCDF, zarr, ERDDAP, BigQuery, parquet curation.
- `llm-local`: local LLM inference, embeddings, model-assisted analysis.

This prevents a PyTorch/CUDA upgrade from breaking xarray/netCDF ingestion, and prevents LLM dependencies from destabilizing production forecasting runs.

## Prerequisites

Required:

- Windows 11.
- NVIDIA RTX 5090 driver visible to `nvidia-smi`.
- Python 3.11 or 3.12 via the Python Launcher `py`.
- PowerShell 7 or Windows PowerShell 5.1.
- Network access for `pip`.

Recommended:

- Git.
- WSL2 Ubuntu for Linux-native GPU tooling.
- Enough free disk space for CUDA wheels and model caches.

## Create Environments

From the workspace root:

```powershell
Set-Location <repo-root>
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ops\create_mbal_envs.ps1
```

Create only one environment:

```powershell
.\ops\create_mbal_envs.ps1 -Only mbal-gpu
```

Force recreation:

```powershell
.\ops\create_mbal_envs.ps1 -Only mbal-gpu -Recreate
```

Use Python 3.11 explicitly:

```powershell
.\ops\create_mbal_envs.ps1 -PythonVersion 3.11
```

## Activate Environments

```powershell
.\.venvs\mbal-gpu\Scripts\Activate.ps1
python -m pip list
deactivate
```

```powershell
.\.venvs\geo-ocean\Scripts\Activate.ps1
python -m pip list
deactivate
```

```powershell
.\.venvs\llm-local\Scripts\Activate.ps1
python -m pip list
deactivate
```

## Verify GPU

Run:

```powershell
.\ops\verify_gpu_stack.ps1 -EnvName mbal-gpu
```

Expected success signals:

- `nvidia-smi` shows the RTX 5090.
- `torch.cuda.is_available()` is `True`.
- `torch.cuda.get_device_name(0)` reports the RTX 5090.
- `torch.cuda.get_arch_list()` includes `sm_120`.
- A small PyTorch CUDA matrix multiply completes.
- A small XGBoost model trains with `device=cuda`.

## Run Tests

Use the stable local test entrypoint:

```powershell
python .\ops\run_tests.py --quiet
```

`pytest` is included in `ops\requirements-mbal-gpu.txt`. If the command reports that pytest is missing, install the tooling dependency:

```powershell
python -m pip install -r .\ops\requirements-mbal-gpu.txt
```

## Package Policy

Use these rules:

- Install CUDA PyTorch with the official CUDA 12.8 wheel index.
- Keep `xgboost` in `mbal-gpu`, not in global Python.
- Keep netCDF and xarray tooling in `geo-ocean`.
- Keep `transformers` and local LLM dependencies in `llm-local`.
- Freeze known-good environments after successful smoke tests:

```powershell
.\.venvs\mbal-gpu\Scripts\python.exe -m pip freeze | Set-Content .\ops\freeze-mbal-gpu.txt
```

Only commit/update freeze files after a verified run.

## Data Safety Notes

Before training models, run a data validation gate:

- Unique key check: `station,time,depth`.
- Time monotonicity by station and depth.
- Expected units: temperature in C, salinity in PSU, current in m/s.
- QC flags handled before physical range filters.
- Sentinel values converted to null before feature engineering.
- Train/test split by time, never by random row split.
- Baseline comparison against persistence for each horizon.

Model quality is only meaningful after these checks pass.

## Training Orchestrator

The `mbal_train.py` script is the current single source of truth for GPU training jobs. It replaces the fragile PowerShell-based loop with a robust Python-based manager for long RTX 5090 runs. It is designed to prevent recurring flow failures: idle-sleep freezes, concurrent GPU crashes/OOM, VRAM thrashing, and terminal-owned child processes being killed when Windows Terminal or PowerShell crashes.

### Terminal Ownership Policy

Do not run long GPU training jobs as foreground work owned by Windows Terminal, PowerShell, VS Code terminals, or IDE-integrated terminals. Use terminals only to launch detached jobs, check status, and tail logs.

Reason: on 2026-06-07, Windows Terminal and `pwsh.exe` crashed with .NET `OutOfMemoryException` while `nhits+drv` was running. The model log ended with `forrtl: error (200): program aborting due to window-CLOSE event`, which means the console close event killed the training process even though the GPU was not out of memory.

Default long-run launch:

```powershell
python .\ops\launch_mbal_train_detached.py --wait-lock --enforce-power
```

After this prints a PID, the launching terminal can be closed. The training process continues independently and writes:

```text
nn_results\train_detached.log
nn_results\train_detached.err.log
```

Check progress without owning the job:

```powershell
python .\mbal_train.py --status
Get-Content .\nn_results\train_detached.log -Tail 80
```

Foreground runs are allowed only for quick smoke tests, `--status`, `--validate`/read-only checks, or deliberate debugging where losing the process is acceptable.

### Features

- **GPU Resource Guarding**: Implements a global file-lock to ensure only one GPU-intensive job runs at a time. It also checks `nvidia-smi` for VRAM availability before starting.
- **Wake Lock and Power Guarding**: Holds the system awake during long runs and can enforce AC sleep/hibernate timeouts to Never with `--enforce-power`.
- **Batch Enforcement**: Uses stable RTX 5090 profiles. The safe profile is Batch=12, Windows=64, Infer=128, Valid=64.
- **Driver-Model Guarding**: Full-history exogenous windows for NHITS, TFT, and NBEATSx can request more than 100 GiB of host RAM. The orchestrator automatically runs those driver-enabled jobs with `--tail-weeks 104` unless `--allow-unsafe-driver-models` is set. TSMixerx driver jobs can run full-history.
- **Stall Detection and Backoff**: Kills stalled jobs, retries with smaller batch profiles, and can fall back to CPU after repeated CUDA/OOM/stall failures.
- **Queue Management**: Sequential execution of jobs with per-job raw logs and `nn_results/run_manifest.json`.
- **Resume Awareness**: Skips jobs that already have a valid `summary.json` with the current cache version, unless `--force` is used.

### Usage

Run the default job suite detached:

```powershell
python .\ops\launch_mbal_train_detached.py --wait-lock --enforce-power
```

Check status:

```powershell
python .\mbal_train.py --status
```

Run a quick CPU-only smoke job in the foreground:

```powershell
python .\mbal_train.py --jobs .\ops\jobs.smoke.json --accel cpu --no-wake-lock --no-gpu-preflight
```

Force rerun all jobs detached:

```powershell
python .\ops\launch_mbal_train_detached.py --force --wait-lock --enforce-power
```

Use a custom batch of jobs via JSON:

```powershell
python .\ops\launch_mbal_train_detached.py --jobs my_batch.json --wait-lock --enforce-power
```

Run only the bounded driver-model validation set:

```powershell
python .\ops\launch_mbal_train_detached.py --jobs .\ops\jobs.driver_guarded.json --force --wait-lock --enforce-power
```

### Jobs File Format (.json)

```json
[
  {
    "label": "nhits_v2",
    "model": "nhits",
    "outdir": "nhits_v2",
    "extra_args": ["--smoke", "--max-steps", "100"]
  },
  {
    "label": "tft_with_drivers",
    "model": "tft",
    "outdir": "tft_drv",
    "extra_args": ["--drivers-parquet", "nn_cache/drivers_hourly.parquet", "--drivers-manifest", "nn_cache/drivers_manifest.json"]
  }
]
```

## Production Expectations

For each modeling run, record:

- Environment name and `pip freeze`.
- GPU name, driver version, CUDA runtime.
- Source data paths and time ranges.
- Feature set version.
- Train/validation/test date boundaries.
- Baseline metrics and model metrics.
- Random seed.
- Output artifact path.

Prefer walk-forward validation for forecasting. A single random split is not acceptable for time-series claims.

