# Contributing

Thanks for helping improve the Monterey Bay AI Lab forecasting stack. The project favors
**honest evaluation over headline numbers** — changes that touch modeling or promotion
must keep the evidence gates green.

## Workflow (trunk-based + PR)

1. Branch off `main` with a short-lived branch: `git switch -c feat/<short-name>`.
2. Make focused changes. Keep one logical change per PR.
3. Run the suite locally: `python ops/run_tests.py` (must pass).
4. Open a PR. CI must be green and one review is required before merge.
5. **Merges are squash-only** — your branch collapses to a single, revertable commit on a
   linear `main`. Don't worry about messy intermediate commits; write a clear PR title (it
   becomes the squash commit subject).

`main` is protected: no direct pushes, required CI, required review, linear history.

## Developer Certificate of Origin (DCO)

We use the [DCO](https://developercertificate.org/) instead of a CLA. Every commit must be
signed off, certifying you have the right to submit it under the project's license:

```
git commit -s -m "your message"
```

This appends `Signed-off-by: Your Name <you@example.com>` (use your real name and a real
email). CI checks for the sign-off; PRs without it can't merge.

## Conventions

- Python, std formatting; match the surrounding code's style and comment density.
- New behavior needs a test under `tests/` and registration in `ops/run_tests.py`.
- Never commit credentials, API keys, or local absolute paths. Configure cloud access via
  the `MBARI_*` environment variables (see README). CI runs a secret scan.
- Don't commit generated artifacts or large data (see `.gitignore` and `DATA.md`).
- Promotion-affecting changes must keep `ops/evidence_gate_agent.py` passing: a model is
  only promotable if it beats the **best-naive** baseline (better of persistence and
  seasonal-naive). See `AGENTS.md`.

## Reporting issues

Use GitHub Issues for bugs/ideas. For anything security-sensitive, see `SECURITY.md`.
