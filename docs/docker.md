# Docker Reproducibility

This repository includes a Docker workflow for reproducible tests and lightweight open-source review. The image intentionally excludes local datasets, generated artifacts, credentials, caches, and training outputs.

## Build

Build the default test image:

```powershell
docker build -t monterey-bay-ai-lab:dev .
```

Build with a broader optional dependency set:

```powershell
docker build --build-arg EXTRAS=all -t monterey-bay-ai-lab:all .
```

## Run Tests

```powershell
docker run --rm monterey-bay-ai-lab:dev
```

The default command runs:

```powershell
python ops/run_tests.py
```

## Mount Local Data

Keep datasets outside the image and mount them at runtime:

```powershell
docker run --rm `
  -e MBAL_SOURCE_PARQUET=/data/source.parquet `
  -v C:\path\to\data:/data:ro `
  monterey-bay-ai-lab:dev `
  python mbal_forecast_v2.py
```

## GPU Notes

GPU training is expected to run on a host with NVIDIA drivers and the NVIDIA Container Toolkit. After that is installed, run with:

```powershell
docker run --rm --gpus all monterey-bay-ai-lab:dev python mbal_gpu_analysis.py
```

The default image is optimized for reproducible tests and review, not for shipping a full CUDA training environment. For production GPU training, pin the CUDA base image, PyTorch wheel index, and model artifact source in a dedicated release image.

## What Docker Proves

Docker is used here to prove that a clean machine can install the project, import the modules, and run the test gate without relying on hidden local state. It does not prove that private datasets, local GCP credentials, or workstation-only training artifacts are available.
