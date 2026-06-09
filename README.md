# Monterey Bay Ocean Forecasting Lab

Reproducible experiments for forecasting Monterey Bay ocean conditions from **public**
oceanographic and coastal water-quality data. The emphasis is on **honest baselines,
reproducible data splits, and transparent results — including negative ones.**

> **Not affiliated with MBARI.** This is an independent research/engineering project that
> *uses* publicly available oceanographic data (including MBARI M1/M2 mooring data and other
> public sources). It is **not affiliated with, sponsored by, or endorsed by** MBARI or any
> data provider. "MBARI" is used only to describe data sources.

## What this is
- A portable, lakehouse-style data + modeling pipeline (bronze/silver/gold layering,
  deterministic split contracts, run manifests) that runs on local/portable compute — with
  no managed-product dependency.
- Forecasting experiments (classical baselines, gradient-boosted and neural models, and a
  continual-learning experiment) for Monterey Bay hourly series and a statewide California
  beach water-quality (bacteria-exceedance) task.
- An **honest-evaluation harness**: a model is "promoted" only if it beats the *better of
  persistence and seasonal-naive* (same-hour-yesterday). This is enforced in code, not just
  documented.

## What this is *not*
- **Not** an operational or deployed forecasting service.
- **Not** a state-of-the-art result. The continual-learning experiment **does not beat** the
  seasonal-naive baseline (see its `RESULTS.md`); it is published as a transparent negative
  result.
- **Not** affiliated with MBARI or any data provider.
- **Not** a redistribution of any dataset — no third-party data ships with this repo.

## Datasets
Public oceanographic and coastal sources (e.g. MBARI M1/M2 mooring data, NOAA products, and
state water-quality data). Each provider's own terms apply — see **[DATA.md](DATA.md)**.
Data is **not** included in this repository.

## Reproducibility status
Research-grade and reproducible — *not* production. A portable smoke pipeline and a unit
suite validate the evaluation gates; some model claims are explicitly marked not-yet-approved.

## Honest baseline methodology
Forecast skill is measured against the **better of persistence and seasonal-naive**
(same-hour-yesterday), because persistence alone gets a free pass at daily horizons. The
gate is enforced in code:
- `ops/seasonal_naive.py` — best-naive scorer
- `release_gate/mbari_promotion_matrix.py` — gated promotion decisions
- `ops/evidence_gate_agent.py` — fails dishonest promotions

## Quickstart
```bash
pip install -e .[test]
python ops/run_tests.py        # unit / gate suite
```
The repo ships **no data or trained artifacts**, so training/evaluation require you to point
the pipeline at your own copy of the public datasets (below). A small synthetic/demo path so
the repo runs end-to-end without private data is planned.

## Use your own data
Everything is configured via environment variables (nothing is hardcoded):
```
MBARI_PROJECT_ROOT   MBARI_SOURCE_PARQUET   MBARI_CACHE_DIR
MBARI_LAKEHOUSE_DIR  MBARI_GCP_PROJECT
```

## Known limitations
- Models do **not** yet beat strong naive baselines on the headline series (honest negative
  result — that's the point).
- Tested on a single consumer NVIDIA GPU; not validated across hardware.
- No bundled demo dataset yet (see Quickstart).

## License & data terms
- **Code:** Apache-2.0 (see `LICENSE`).
- **Data, model outputs, and trained checkpoints:** **not** covered by the code license;
  each remains subject to its provider's terms. Verify before any redistribution or
  commercial use. See **[DATA.md](DATA.md)**.


## Disclaimer
Independent project; not affiliated with, sponsored by, or endorsed by MBARI or any data
provider. Provided "as is", without warranty of any kind.
