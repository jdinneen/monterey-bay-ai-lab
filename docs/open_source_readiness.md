# Open-Source Readiness

This checklist defines the minimum bar before publishing the repository publicly.

## Required

- `README.md` explains the project, setup, tests, Docker workflow, and data boundaries.
- `LICENSE` grants explicit open-source rights.
- `CONTRIBUTING.md` explains branch, test, and pull request expectations.
- `SECURITY.md` gives a private vulnerability reporting path.
- `CODE_OF_CONDUCT.md` sets collaboration expectations.
- `pyproject.toml` declares package metadata and dependency groups.
- `Dockerfile` and `.dockerignore` support clean reproducibility checks.
- Tests pass with `python ops/run_tests.py`.

## Data Boundary

Do not publish private credentials, local `.env` files, raw protected datasets, generated caches, or workstation-only artifacts. Public releases should include one of the following:

- Small synthetic fixtures.
- Publicly licensed sample data.
- Scripts that download public data from documented sources.
- Precomputed public artifacts with provenance and version notes.

## Release Gate

Before a public push:

```powershell
python ops/run_tests.py
docker build -t monterey-bay-ai-lab:dev .
docker run --rm monterey-bay-ai-lab:dev
git status --short
```

Review staged files carefully and confirm that no generated data, credentials, or local-only outputs are included.
