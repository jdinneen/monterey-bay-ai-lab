# Contributing to Monterey Bay AI Lab

We welcome contributions to the revolutionary coastal intelligence platform. To maintain our high standards for high-performance computing (HPC) and multivariate analysis, please follow these guidelines.

## 1. Our Engineering Standards
- **AI is HPC:** We value iterative multivariate regression, A/B testing, and rigorous baseline benchmarking.
- **Leakage Zero:** All predictive features must be strictly causal. No information from the future is allowed.
- **Truth over Hype:** We favor the "Honest Baseline" over inflated metrics.

## 2. Development Workflow
1.  **Claim a File:** Before editing, use the cooperative lock:
    `python ops/agent_lock.py claim <paths...> --agent <id> --task "<desc>"`
2.  **Pre-commit:** Install the pre-commit hooks to ensure linting and formatting standards:
    `pre-commit install`
3.  **Install Locally:** Use an editable Python 3.12 environment:
    `python -m pip install -e ".[test]"`
4.  **Conventional Commits:** Use standard prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.

## 3. Testing
All changes must pass the local test suite before promotion:
`python .\ops\run_tests.py`

For open-source reproducibility, also verify the Docker path when dependency, packaging, or environment files change:
`docker build -t monterey-bay-ai-lab:dev .`
`docker run --rm monterey-bay-ai-lab:dev`

## 4. Submitting Changes
- Create a feature branch: `feature/your-innovation`
- Ensure all CI gates pass (Ruff, Gitleaks, Unit Tests).
- Review `docs/open_source_readiness.md` before public release.
- Do not commit local `.env` files, credentials, raw private data, caches, or generated training outputs.
- Submit a Pull Request.

---
**Monterey Bay AI Lab — Building the Future of Coastal Intelligence.**
