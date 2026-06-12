# Optional WSL2 GPU Guide

Native Windows PyTorch and XGBoost are useful for quick iteration. WSL2 Ubuntu is recommended for heavier GPU workflows such as RAPIDS, vLLM, Triton-based models, and Linux-first time-series libraries.

## Install WSL2 Ubuntu

Run PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
wsl --update
```

Restart if Windows requests it.

## Verify GPU In WSL2

Inside Ubuntu:

```bash
nvidia-smi
```

If `nvidia-smi` is unavailable, update the Windows NVIDIA driver. Do not install a separate Linux display driver inside WSL2.

## Create A Conda/Mamba Environment

Recommended for Linux GPU stacks:

```bash
mkdir -p ~/ai-envs
python3 -m venv ~/ai-envs/mbal-gpu
source ~/ai-envs/mbal-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install xgboost polars pyarrow duckdb pandas scikit-learn mlflow
```

Smoke test from a WSL checkout or mounted workspace:

```bash
cd /path/to/mbal-projects
python ops/smoke_test_gpu.py
```

## RAPIDS Option

RAPIDS is generally easier in WSL2 than native Windows. Use the current RAPIDS selector from NVIDIA for exact package versions matching the installed CUDA/runtime.

Keep RAPIDS separate from the Windows `.venvs` environments.

## File Location Guidance

For heavy training, put hot data on the Linux filesystem:

```bash
mkdir -p ~/mbal-data/curated
```

Accessing files under `/mnt/c/...` is convenient but can be slower for many small files.

Use Windows paths for shared artifacts and final reports; use Linux paths for high-throughput training caches.
