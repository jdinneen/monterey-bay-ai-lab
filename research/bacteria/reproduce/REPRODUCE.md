# Reproduce the marine-enterococcus nowcasting benchmark

This folder lets a reviewer regenerate **every number** in `PAPER_DRAFT.md` from a clean
clone, deterministically, in a couple of minutes. Inputs are ~6 MB of public data committed
to the repo; no API keys, no cloud, no GPU.

- **Paper draft:** [`PAPER_DRAFT.md`](PAPER_DRAFT.md)
- **Pinned environment:** [`requirements.txt`](requirements.txt)
- **Expected outputs to diff against:** [`expected/`](expected/)
- **Input data checksums:** [`MANIFEST.sha256`](MANIFEST.sha256)

---

## TL;DR — three commands

Run from the **repository root** (the dir that contains `research/` and `bacteria_results/`):

```bash
python -m venv .venv && . .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install -r research/bacteria/reproduce/requirements.txt

OBS=bacteria_results/statewide/statewide_beach_observations.parquet

# 1. Headline benchmark (stratified, calibrated)            ~60-90 s
python research/bacteria/operational_benchmark.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall \
  --discharge-dir bacteria_results/discharge \
  --reveal-lag-days 2 --label enterococcus \
  --out-dir /tmp/repro

# 2. Leave-one-county-out spatial generalization             ~30-60 s
python research/bacteria/spatial_holdout.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall \
  --reveal-lag-days 2 --label enterococcus \
  --out-dir /tmp/repro

# 3. Prequential (online) recalibration                      ~20-40 s
python research/bacteria/online_recalibration.py --obs "$OBS" \
  --rain-dir bacteria_results/rainfall \
  --lag-days 2 --out-dir /tmp/repro
```

Then compare your output against the committed expected results:

```bash
diff /tmp/repro/operational_benchmark.json research/bacteria/reproduce/expected/operational_benchmark.json
diff /tmp/repro/spatial_holdout.json        research/bacteria/reproduce/expected/spatial_holdout.json
diff /tmp/repro/online_recalibration.json   research/bacteria/reproduce/expected/online_recalibration.json
```

With the pinned environment and the committed (frozen) data, the metrics are **deterministic and
identical** (`random_state=42` is pinned in all three scripts). The expected files were generated
with the relative `--obs` above, so on a Unix host the diff should be empty; the only line that can
ever differ is the `obs_path` provenance string (it echoes your path separator — `\` on Windows,
`/` on Unix). To compare just the results and ignore that one cosmetic line:

```bash
diff <(grep -v obs_path /tmp/repro/operational_benchmark.json) \
     <(grep -v obs_path research/bacteria/reproduce/expected/operational_benchmark.json)
```

---

## ⚠️ One thing reviewers always trip on: the flags ARE the headline

The script's **default** is `--label any --reveal-lag-days 0`, which is the *legacy*
multi-analyte OR target — **a different, non-headline result**. The paper's headline requires
both `--label enterococcus` and `--reveal-lag-days 2`, exactly as in the commands above. If your
numbers don't match, check those two flags first.

---

## What the numbers should be (headline stratum)

From `expected/operational_benchmark.json`, stratum `EXCLUDE_SAN_DIEGO`
(n = 89,321; base rate 0.1007):

| Method | AP | AUROC | ECE |
|---|--:|--:|--:|
| AB411 rain rule | 0.1779 | 0.6495 | — |
| Station memory | 0.2782 | 0.7716 | 0.0110 |
| Virtual-Beach-class MLR (deployed practice) | 0.3754 | 0.7493 | — |
| **Model (HGBT, isotonic-calibrated)** | **0.4719** | **0.8551** | **0.0169** |

Comparative claims: **2.65×** AB411 · **+0.097 AP** vs Virtual-Beach · **+0.194 AP** vs station memory.

**LOCO** (`expected/spatial_holdout.json`): 9 counties held out · beats AB411 **9/9** · beats
station memory **9/9** · median model AP **0.5379** · deploy-ready **8/9** (San Francisco the lone exception).

**Prequential recalibration** (`expected/online_recalibration.json`): San Diego ECE
**0.285 → 0.174** (repaired) while stable strata degrade (EXCLUDE_SAN_DIEGO 0.009 → 0.033) —
confirming recalibration should be drift-triggered, not global.

> **Note on the abstract's "0.465".** The committed data extends through 2026; as the record
> grows the calibrated AP point estimate moves slightly (currently **0.4719**). AUROC (0.855)
> and ECE (0.017) and every baseline are stable to 3 decimals. None of the qualitative claims
> change. When finalizing, quote the reproduced value (≈0.47) and cite this frozen snapshot.

---

## Inputs (frozen, committed — ~6 MB)

| File | Bytes | Source |
|---|--:|---|
| `bacteria_results/statewide/statewide_beach_observations.parquet` | 3.8 M | CA beach monitoring (CEDEN / BeachWatch lineage) |
| `bacteria_results/rainfall/rainfall_grid.parquet` | 1.3 M | Open-Meteo historical reanalysis |
| `bacteria_results/rainfall/station_grid_map.parquet` | 31 K | station → 0.1° grid-cell map |
| `bacteria_results/discharge/discharge_gauge.parquet` | 615 K | USGS NWIS daily discharge (param 00060) |
| `bacteria_results/discharge/station_gauge_map.parquet` | 32 K | station → nearest gauge map |

Verify you have the exact inputs:

```bash
sha256sum -c research/bacteria/reproduce/MANIFEST.sha256   # run from repo root
```

---

## Optional: rebuild the inputs from public sources (provenance, not reproduction)

The fetchers regenerate the driver data from free public APIs (no key):

```bash
python research/bacteria/fetch_statewide_beachwatch.py   # CA beach monitoring record
python research/bacteria/fetch_rainfall.py               # Open-Meteo  (archive-api.open-meteo.com)
python research/bacteria/fetch_discharge.py              # USGS NWIS   (waterservices.usgs.gov)
```

**This is for provenance only.** Live APIs return more data over time and gauge availability
changes, so a from-scratch fetch is **not bit-reproducible** and may shift point estimates. For
verifying the paper, use the committed frozen inputs above.

---

## Tests

The harness ships with unit tests, including a causality test (corrupting the most-recent
revealed labels must not change earlier predictions):

```bash
pytest tests/test_operational_benchmark.py tests/test_spatial_holdout.py tests/test_online_recalibration.py -q
```

---

## Sources & licensing

- **CA beach-water monitoring** (CEDEN / BeachWatch lineage) — public California environmental
  monitoring data; station IDs + coordinates only, no personal data.
- **Open-Meteo historical reanalysis** — free for non-commercial and research use; attribution requested.
- **USGS NWIS** — U.S. Government public-domain data.

Please credit these sources in any redistribution.

---

## Environment that produced `expected/`

```
python       3.12.10
scikit-learn 1.9.0     # the version that governs HistGradientBoosting + isotonic reproducibility
pandas       2.3.3
numpy        2.4.6
pyarrow      24.0.0
```
