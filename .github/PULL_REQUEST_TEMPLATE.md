## What & why

<!-- One or two sentences. Link any issue. -->

## Checklist

- [ ] Commits are signed off (`git commit -s`) — DCO
- [ ] `python ops/run_tests.py` passes locally
- [ ] No secrets, API keys, or local absolute paths added
- [ ] New behavior has a test under `tests/` (and is registered in `ops/run_tests.py`)
- [ ] If this touches modeling/promotion: the best-naive evidence gate still passes
      (`python ops/evidence_gate_agent.py`)
- [ ] Docs/decisions updated if behavior or claims changed

## Notes for reviewers

<!-- Anything non-obvious, trade-offs, follow-ups. -->
