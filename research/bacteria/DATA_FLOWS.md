# Data Flow Architecture: From Raw Sources to Predictive Features

> **BINDING**: All data flows must follow this architecture. Deviations require manual approval.

---

## Overview: The Three-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA INGESTION (fetch_*.py)                                       │
│   → Public sources only (no API keys)                                      │
│   → Parquet output to bacteria_results/                                    │
│   → MANIFEST.sha256 for integrity                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: FEATURE ENGINEERING (add_*_features.py)                           │
│   → Causal joins on station_id + sample_date                               │
│   → Reveal-lag enforcement                                                  │
│   → Window operations (rolling, expanding)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: MODEL TRAINING (benchmark.py / predictor.py)                      │
│   → Train/val/test splits                                                  │
│   → Calibration (isotonic regression)                                      │
│   → Stratified evaluation                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data Ingestion

### Current Sources

| Source | Fetcher File | Output Path | Freshness |
|--------|--------------|-------------|-----------|
| CA BeachWatch (lab assays) | `fetch_statewide_beachwatch.py` | `bacteria_results/statewide/` | ~2 day lag (lab processing) |
| Rainfall (Open-Meteo) | `fetch_rainfall.py` | `bacteria_results/rainfall/` | Daily (next-day coverage) |
| River Discharge (USGS NWIS) | `fetch_discharge.py` | `bacteria_results/discharge/` | Real-time (15-min updates) |
| CDIP Waves (NDBC fallback) | `fetch_cdip_waves.py` | `bacteria_results/cdip_waves/` | Hourly (via pre-fetched NDBC parquet)

### Running Fetchers

```powershell
# Fetch all sources in order
python research/bacteria/fetch_statewide_beachwatch.py
python research/bacteria/fetch_rainfall.py
python research/bacteria/fetch_discharge.py
python research/bacteria/fetch_cdip_waves.py

# Or re-run a specific source (forces refresh)
python research/bacteria/fetch_statewide_beachwatch.py --force
```

### Verifying Data Integrity

```powershell
# Check all checksums
sha256sum -c research/bacteria/reproduce/MANIFEST.sha256

# Expected: All files should pass validation
bacteria_results/statewide/statewide_beach_observations.parquet: OK
bacteria_results/rainfall/rainfall_grid.parquet: OK
...
```

---

## Layer 2: Feature Engineering

### Standard Feature Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `add_causal_features()` | Prior lab results, station memory, seasonality | `exc_prev`, `exc_roll5_prev`, `station_prior_rate`, `cty_prev7`, etc. |
| `add_rain_features()` | Gridded rainfall from Open-Meteo | `rain_0d`, `rain_3d`, `rain_7d`, `days_since_rain` |
| `add_discharge_features()` | USGS first-flush proxy | `discharge_log`, `discharge_3d`, `discharge_anom`, `days_since_highq` |
| `add_knn_spatial_lag()` | Neighbor beach states (k=8) | `nbr_prev1`, `nbr_prev7` |

### Feature Engineering Pipeline

```python
from research.bacteria import operational_benchmark as ob
import pandas as pd

# Load raw observations
df = ob.load_station_days("bacteria_results/statewide/statewide_beach_observations.parquet")

# Apply reveal-lag (2 days means prior labs must have returned by then)
df = ob.add_causal_features(df, reveal_lag_days=2)

# Add rainfall features (if available)
df = ob.add_rain_features(df, Path("bacteria_results/rainfall"))

# Add discharge features (if available)  
df = ob.add_discharge_features(df, Path("bacteria_results/discharge"))

# Result: 40+ columns ready for modeling
print(df.columns.tolist())
```

### Causality Rules (NON-NEGOTIABLE)

| Rule | Example | Why |
|------|---------|-----|
| **Reveal-lag** | `exc_prev` uses sampleDate - 2days | Same-day resample can't predict itself |
| **Rainfall windows END on sample_date** | `rain_3d` = sum of prior 3 days | At sampling time, today's rain is known |
| **Discharge backward-filled only** | Use last reading at/before sample_time | Never forward-fill (would leak future) |

---

## Layer 3: Model Training & Evaluation

### Standard Benchmark Runner

```python
from research.bacteria import operational_benchmark as ob

result = ob.run(
    obs_path="bacteria_results/statewide/statewide_beach_observations.parquet",
    rain_dir="bacteria_results/rainfall",
    discharge_dir="bacteria_results/discharge",
    reveal_lag_days=2,
    label="enterococcus",  # or "any"
)

# Output includes:
#   - models/: HGBT, AB411, station-memory, Virtual-Beach-class MLR
#   - strata: ALL / EXCLUDE_SAN_DIEGO / SAN_DIEGO_ONLY / MONTEREY
#   - calibrated vs raw probabilities
```

### Key Evaluation Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **AP (Average Precision)** | Maximize | Ranking quality (primary focus) |
| **ROC-AUC** | Maximize | Overall separability |
| **ECE (Expected Calibration Error)** | Minimize → 0.02 | Trustworthy probabilities for thresholds |
| **Recall@20%FPR** | Maximize | Operational: how many exceedances do we catch? |

### Stratification Strategy

```
Test Era: ≥2022-01-01
Train Era: ≤2019-12-31
Calibration Era: 2020-01-01 to 2021-12-31

Strata:
  - ALL: Full test set (inflated by San Diego regime break)
  - EXCLUDE_SAN_DIEGO: Canonical "honest" stratum (our headline result)
  - SAN_DIEGO_ONLY: artifact region (for debugging)
  - MONTEREY: our home region
```

---

## Complete Example: From Fetch to Prediction

```python
#!/usr/bin/env python3
"""End-to-end demonstration of MBAL data pipeline."""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Fetch data (run once per dataset)
# ──────────────────────────────────────────────────────────────────────────────
print("[1] Fetching raw data...")
import subprocess
subprocess.run(["python", REPO_ROOT / "research" / "bacteria" / "fetch_statewide_beachwatch.py"], check=True)
subprocess.run(["python", REPO_ROOT / "research" / "bacteria" / "fetch_rainfall.py"], check=True)
subprocess.run(["python", REPO_ROOT / "research" / "bacteria" / "fetch_discharge.py"], check=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Load and feature engineer
# ──────────────────────────────────────────────────────────────────────────────
print("[2] Loading and feature engineering...")
from research.bacteria import operational_benchmark as ob

# Load beach observations (1.3M rows)
df = ob.load_station_days(
    REPO_ROOT / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet",
    label="enterococcus"
)

# Apply reveal-lag features (2 days)
df = ob.add_causal_features(df, reveal_lag_days=2)

# Join rainfall grid (0.1° cells)
df = ob.add_rain_features(df, REPO_ROOT / "bacteria_results" / "rainfall")

# Merge USGS discharge gauges
df = ob.add_discharge_features(df, REPO_ROOT / "bacteria_results" / "discharge")

print(f"[2] Feature matrix shape: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Train/test split with temporal discipline
# ──────────────────────────────────────────────────────────────────────────────
tr = df[df["sample_date"] <= "2019-12-31"]
te = df[df["sample_date"] >= "2022-01-01"].copy()

feats = list(ob.FEATS) + ob.RAIN_FEATS + ob.DISCHARGE_FEATS

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Train model (HistGradientBoostingClassifier)
# ──────────────────────────────────────────────────────────────────────────────
print("[3] Training HGBT model...")
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier(
    max_iter=600, learning_rate=0.05, l2_regularization=1.0,
    class_weight="balanced", early_stopping=True, random_state=42
)
clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Calibrate (isotonic regression on validation era)
# ──────────────────────────────────────────────────────────────────────────────
va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
te["p_raw"] = clf.predict_proba(te[feats].astype(float))[:, 1]

from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(clf.predict_proba(va[feats].astype(float))[:, 1], va["exceed"].to_numpy())
te["p_cal"] = iso.predict(te["p_raw"])

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Evaluate (San-Diego-excluded stratum)
# ──────────────────────────────────────────────────────────────────────────────
print("[4] Evaluating on EXCLUDE_SAN_DIEGO stratum...")
excl_san_diego = te[~te["county"].isin(["San Diego"])]

from research.bacteria.operational_benchmark import score

y = excl_san_diego["exceed"].to_numpy()
base_rate = float(tr["exceed"].mean())

print(f"  Test rows: {len(y):,}")
print(f"  Events: {y.sum():,} (base rate: {base_rate:.3f})")

for name, col in [("Raw", "p_raw"), ("Calibrated", "p_cal")]:
    s = score(y, te[col].to_numpy(), base_rate)
    print(f"\n  {name} Model:")
    print(f"    AP: {s['ap']}")
    print(f"    ROC-AUC: {s['roc_auc']}")
    print(f"    ECE: {s['ece']}")

# ──────────────────────────────────────────────────────────────────────────────
# Expected output (San-Diego-excluded, calibrated):
#   AP ≈ 0.497, ROC-AUC ≈ 0.858, ECE ≈ 0.020
#   Beats AB411 (AP 0.178) by 2.80× and station-memory (AP 0.278) by +0.219 AP
```

---

## Adding New Sources: Check the Flow

### Before: Adding CDIP Wave Data

1. **Fetch** → `bacteria_results/cdip_waves/cdip_waves.parquet`
   ```bash
   python research/bacteria/fetch_cdip_waves.py
   ```

2. **Feature adder** → New columns in feature matrix
   ```python
   # Add to operational_benchmark.py:
   CDIP_FEATS = ["Hs", "Tp", "Dm"]
   
   def add_cdip_wave_features(df, cdip_dir):
       # Load and join on station_id + sample_date
       pass  # Implement similar to add_rain_features()
   ```

3. **Training** → Update feature list
   ```python
   feats = list(ob.FEATS) + ob.RAIN_FEATS + ob.DISCHARGE_FEATS + CDIP_FEATS
   clf.fit(tr[feats], y)
   ```

---

## Data Lineage Table

| Output File | Source Files | Script | Last Updated |
|-------------|--------------|--------|---------------|
| `statewide/statewide_beach_observations.parquet` | CA BeachWatch / CEDEN via provider-gated BigQuery mirror (needs `MBAL_GCP_PROJECT`; underlying records are public — see `DATA.md`) | `fetch_statewide_beachwatch.py` | See MANIFEST.sha256 |
| `rainfall/rainfall_grid.parquet` | Open-Meteo API | `fetch_rainfall.py` | See MANIFEST.sha256 |
| `discharge/discharge_gauge.parquet` | USGS NWIS | `fetch_discharge.py` | See MANIFEST.sha256 |
| `cdip_waves/cdip_waves.parquet` | CDIP UCSD (template) | `fetch_cdip_waves.py` | To be implemented |

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| **Empty dataframe after merge** | Date range mismatch | Ensure all sources cover same time period |
| **NaN features everywhere** | Station-to-source map missing IDs | Verify `station_grid_map.parquet` has all stations |
| **Manifest checksum mismatch** | File was modified externally | Re-run fetcher with `--force` |
| **Reveal-lag violation** | Using sample index instead of reveal time | Use `_predecessor_idx()` with proper lag |

---

## Future Extensions

### High Priority
- [ ] Implement CDIP wave data fetch (see `fetch_cdip_waves.py` TODO)
- [ ] Add NDBC storm event log (for major weather events)
- [ ] Integrate CA DFW wildlife surveillance (bird die-offs → runoff proxy)

### Medium Priority  
- [ ] Weather service AWOS readings (higher-temporal rainfall than Open-Meteo)
- [ ] Satellite-derived SST fronts (monthly from NOAA)
- [ ] Urban land use data (impervious surfaces → runoff potential)

---

**Maintainer**: Monterey Bay AI Lab Engineering  
**Last updated**: 2026-06-10
