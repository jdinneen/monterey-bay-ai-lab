# Monterey Bay AI Lab — Ocean Forecasting

Local, portable ocean forecasting for the Monterey Bay (MBARI M1/M2 moorings) plus a
statewide California beach water-quality (bacteria exceedance) model. Built **like** a
production lakehouse (bronze/silver/gold, deterministic split contracts, run manifests) on
local/portable compute — no managed-product dependency.

What makes this different from a typical model dump: **honest evaluation is enforced in
code.** A model is only "promoted" if it beats the *better of persistence and seasonal-naive*
(same-hour-yesterday) — persistence alone is a free lunch at daily horizons. The promotion
matrix, champion selector, and evidence gate all enforce this; see `AGENTS.md` and
`research/model_lab/HONEST_SKILL_BASELINE.md`.

## Quickstart

```bash
pip install -e .[test]
python ops/run_tests.py                  # unit suite
```

Configure data locations via environment variables (nothing is hardcoded):

```
MBARI_PROJECT_ROOT   MBARI_SOURCE_PARQUET   MBARI_CACHE_DIR
MBARI_LAKEHOUSE_DIR  MBARI_GCP_PROJECT
```

Curated data + model artifacts ship as a separate data release (see `DATA.md`).

## The pipeline (gates & decisions)

```
build panel ─► train/benchmark ─► gold metrics ─► promotion matrix ─► champion selector
                                       │                  │
                              best-naive backfill   evidence gate (fails on dishonest promotion)
```

- `ops/seasonal_naive.py` — best-naive scorer (persistence vs seasonal-naive)
- `ops/backfill_seasonal_naive.py` — enriches gold metrics
- `release_gate/mbari_promotion_matrix.py` — gated promotion decisions
- `ops/build_champion_selector.py` — per target/horizon champions
- `ops/evidence_gate_agent.py`, `ops/data_health_agent.py` — auditable gates
- `ops/agent_lock.py` — cooperative file locks for multi-agent development

## Key docs

- `PRODUCTION_READINESS.md` — what's defensible vs not-yet-approved
- `MBARI_PRODUCTION_LAKEHOUSE_CONTRACTS.md` — layer contracts & guarantees
- `AGENTS.md` — binding invariants (incl. the best-naive gate)
- `DATA.md` — data provenance & license (code license ≠ data terms)

## Contributing

PRs welcome — trunk-based, squash-merge, DCO sign-off (`git commit -s`). See
`CONTRIBUTING.md`. Licensed under **Apache-2.0** (`LICENSE`).
