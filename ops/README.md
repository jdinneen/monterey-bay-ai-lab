# MBARI Local AI Environment Ops

This directory contains setup artifacts for isolated local MBARI AI analysis environments on the RTX 5090 workstation.

The goal is to keep GPU ML, ocean/scientific data tooling, and local LLM tooling separated so one stack can be upgraded without breaking the others.

## Environments

- `mbari-gpu`: PyTorch CUDA, XGBoost GPU, core tabular ML, experiment tracking.
- `geo-ocean`: xarray, netCDF, zarr, parquet, BigQuery, ocean data ingestion and validation.
- `llm-local`: local LLM and embedding experimentation with Transformers, Accelerate, sentence-transformers, and Ollama helpers.

## Files

- `create_mbari_envs.ps1`: creates or updates isolated Python virtual environments.
- `verify_gpu_stack.ps1`: verifies NVIDIA driver, PyTorch CUDA, and XGBoost GPU execution.
- `smoke_test_gpu.py`: Python smoke test used by the verification script.
- `requirements-mbari-gpu.txt`: packages for GPU forecasting and tabular ML.
- `requirements-geo-ocean.txt`: packages for ocean data ingestion and validation.
- `requirements-llm-local.txt`: packages for local LLM analysis.
- `ENVIRONMENT_RUNBOOK.md`: operating guide and safety notes.
- `WSL2_GPU_GUIDE.md`: optional WSL2 guidance for heavier Linux-native GPU workflows.
- `launch_mbari_train_detached.py`: starts long training runs detached from the terminal.
- `jobs.smoke.json`: portable CPU smoke job for remote environment/cluster validation.
- `jobs.driver_guarded.json`: bounded-history validation set for driver-heavy neural models.
- `evidence_gate_agent.py`: audits promotion evidence, metric identities, and unsupported claim language.
- `data_health_agent.py`: audits data inventory, target gaps, driver freshness, and metric identities.
- `split_closure_agent.py`: plans or runs shared-split closure for split-mismatch promotion candidates.
- `promotion_critic_agent.py`: grades promoted rows as strong, weak, rerun-needed, or not claimable.
- `report_consistency_agent.py`: checks tracked docs against current promotion/release truth.
- `build_champion_selector.py`: emits target/horizon champion selections from promotion-gated rows.
- `report_model_comparison.py`: generates `..\nn_results\MODEL_COMPARISON.md` from run summaries.
- `run_tests.py`: stable pytest entrypoint for local release/promotion tests.
- `requirements-mbari-gpu.txt` / `requirements-geo-ocean.txt` / `requirements-llm-local.txt`: dependency specs for the GPU, geo/ocean, and local-LLM paths.
- `autonomous_rate_limit_fetcher.py`: polite resumable NOAA CoastWatch MUR SST yearly cache fetcher.
- `report_wave_driver_claim.py`: bounded screening report for observed NDBC wave-driver claims; current verdict is research-only / not supported as a general claim.

## Quick Start

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ops\create_mbari_envs.ps1
.\ops\verify_gpu_stack.ps1 -EnvName mbari-gpu
```

By default the script creates environments under:

```text
<repo-root>\.venvs
```

Use a different location if desired:

```powershell
.\ops\create_mbari_envs.ps1 -EnvRoot D:\ai-envs
.\ops\verify_gpu_stack.ps1 -EnvRoot D:\ai-envs -EnvName mbari-gpu
```

## Safety

These scripts do not modify existing MBARI analysis scripts. They create or update only the selected virtual environments and install packages inside those environments.

Do not install experimental ML packages into the global Windows Python. Use these isolated envs instead.

For long GPU jobs, do not run training as foreground work owned by Windows Terminal, PowerShell, VS Code terminals, or IDE terminals. Launch detached, then watch status/logs:

```powershell
python .\ops\launch_mbari_train_detached.py --wait-lock --enforce-power
python .\mbari_train.py --status
Get-Content .\nn_results\train_detached.log -Tail 80
```

For the production handoff gate, see `..\PRODUCTION_READINESS.md`, `..\MBARI_PRODUCTION_LAKEHOUSE_CONTRACTS.md`, and the portable smoke configurations in this directory.

## Tests

Run the local pipeline tests from the workspace root:

```powershell
python .\ops\run_tests.py --quiet
```

If `pytest` is missing, install the MBARI GPU/tooling requirements or install `pytest` directly:

```powershell
python -m pip install -r .\ops\requirements-mbari-gpu.txt
python -m pip install pytest
```
