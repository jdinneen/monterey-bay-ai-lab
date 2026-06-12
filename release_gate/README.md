# Monterey Bay AI Lab Release Gate

This directory contains a standalone release gate for the local Monterey Bay AI Lab analysis setup. It is read-only with respect to the existing analysis code: the CLI writes reports under `release_gate/reports/` and does not edit Monterey Bay AI Lab scripts or data.

## What It Checks

- GPU stack: `nvidia-smi`, PyTorch CUDA availability, a real CUDA tensor kernel, and a small XGBoost CUDA training run.
- Package consistency: installed versions for PyTorch, XGBoost, pandas, PyArrow, and useful optional data packages.
- Historical Parquet integrity: schema, non-empty row counts, duplicate `station/time/depth_m` keys, timestamp validity, core signal coverage, QC values, and physical range checks.
- Modeling output sanity: parses legacy `mbal*_results/model_results.json` files, checks finite metrics, train/test or fold evidence, baseline/persistence evidence, CUDA evidence, weak records, and whether any forecast records beat persistence.

## Run

From the repository root:

```powershell
python .\release_gate\mbal_release_gate.py
```

Reports are written to `release_gate/reports/`. These generated artifacts are ignored by
git and should be regenerated instead of edited.

Common outputs include `release_gate_report.json` and `release_gate_report.md`.

The CLI exits with code `1` when any check fails. To write reports without failing the shell command:

```powershell
python .\release_gate\mbal_release_gate.py --no-exit-code
```

## Status Meanings

- `PASS`: the checked surface met the current production gate.
- `WARN`: usable, but there is a caveat that should be tracked.
- `FAIL`: do not treat the setup as production-grade until fixed.

## Notes

The gate intentionally avoids hard-coding one expected model score. Ocean forecasting should be judged against baselines and leakage-safe validation, not a single static threshold. The gate therefore fails on invalid structure or impossible values, and warns on poor skill so a reviewer can decide whether the model is acceptable for the intended horizon and target.

