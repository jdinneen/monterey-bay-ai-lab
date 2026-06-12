# CDIP Wave Data Fetcher Implementation Guide

> **BINDING**: All CDIP wavefetching must follow this pattern. Deviations require manual approval.

---

## Overview

CDIP (Coastal Data Information Program) operates ~40+ moorings along the California coast with spectral wave measurements. This guide explains how to fetch and integrate CDIP wave data into the Monterey Bay AI Lab pipeline.

---

## Current State

### What We Have

| File | Status | Purpose |
|------|--------|---------|
| `fetch_cdip_waves.py` | Template (not implemented) | Fetcher skeleton following project conventions |
| `DATA_FLOWS.md` | Complete | Architecture doc: 3-layer pipeline (ingestion→features→training) |
| `mbal_drivers_build.py` | Active | Uses CDIP data via pre-fetched parquet files in `mbal_history/noaa/` |

### Current State (Updated 2026-06-10)

| File | Status | Purpose |
|------|--------|---------|
| `fetch_cdip_waves.py` | ✅ Implemented | Full fetch logic with local parquet fallback |
| `DATA_FLOWS.md` | Complete | Architecture doc: 3-layer pipeline (ingestion→features→training) |
| `mbal_drivers_build.py` | Active | Uses CDIP data via pre-fetched parquet files |

### What We Have

1. **Local NDBC parquet** (`noaa_ndbc46042.parquet`) — Pre-fetched from NDBC API
2. **Output format** — Standardized_columns: station_id, sample_date, Hs, Tp, Dm
3. **Checksums** — SHA-256 tracked in `reproduce/MANIFEST.sha256`

### What We Still Need for Full Integration

1. **Spatial mapping** — Map CDIP stations (e.g., "46042") to beach station UUIDs
2. **Feature join logic** — Add `add_cdip_wave_features()` to `operational_benchmark.py`
3. **Temporal alignment** — Aggregate hourly waves to daily for beach observations

---

## Existing CDIP Integration Pattern

The production pipeline (`mbal_drivers_build.py:320-327`) expects pre-fetched parquet files:

```python
# --- CDIP waves (dynamic discovery)
for p in sorted(NOAA_DIR.glob("noaa_cdip*.parquet")):
    sid = p.stem.replace("noaa_cdip", "")
    cdip = _load_hourly(p, grid)
    if cdip is not None:
        for col in cdip.columns:
            out[col] = cdip[col]
    notes[f"cdip{sid}"] = f"station {sid} discovered; waves"
```

### Expected Parquet Schema

For each CDIP station (`noaa_cdip041.parquet`, etc.):

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime64[ns, UTC] | Measurement timestamp (hourly) |
| `wave_height_m` | float32 | Significant wave height (Hs) |
| `dom_period_s` | float32 | Dominant/peak period (Tp) |
| `mean_dir_deg` | float32 | Mean wave direction (Dm) |

---

## CDIP Data Sources

### Option 1: THREDDS/OPENDAP (Recommended—full spectral data)

**Endpoint**: `https://thredds.cdip.ucsd.edu/thredds/dodsC/ndbc/{station_id}.nc`

**Advantages**:
- Full directional wave spectrum (not just Hs, Tp, Dm)
- Quality-controlled, metadata-rich
- No rate limits for academic use

**Example netCDF structure**:
```python
import xarray as xr
ds = xr.open_dataset("https://thredds.cdip.ucsd.edu/thredds/dodsC/ndbc/041.nc")
print(ds.variables.keys())  # View all available fields
# ['time', 'wave_height', 'dominant_period', 'mean_direction', 'spectral_data', ...]
```

### Option 2: NDBC API (Simplified—NDBC-compatible stations only)

**Endpoint**: `https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt`

**Note**: Not all CDIP stations are in NDBC-compatible format. Check first.

### Option 3: Pre-fetched Parquet (Fallback—no network needed)

If you have local `mbal_history/noaa/noaa_cdip*.parquet` files, the fetcher can skip the network layer entirely:

```python
path = Path("mbal_history/noaa") / f"noaa_cdip{station_id}.parquet"
if path.exists():
    df = pd.read_parquet(path)
else:
    # Fall back to THREDDS or NDBC fetch
```

---

## Implementation Plan

### Step 1: Identify Working CDIP Station IDs

Test which stations are accessible (public data):

```python
# Northern CA
CDIP_STATION_IDS = ["041", "043", "067"]
# Central CA (Monterey Bay area)
CDIP_STATION_IDS += ["065", "066", "068"]
# Southern CA
CDIP_STATION_IDS += ["141", "142", "147", "149"]
```

### Step 2: Implement `fetch_cdip_station()` in `fetch_cdip_waves.py`

**Template for THREDDS access**:

```python
import xarray as xr

def fetch_cdip_station(station_id: str) -> pd.DataFrame:
    """Fetch CDIP wave data from UCSD THREDDS server."""
    url = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/ndbc/{station_id}.nc"
    
    try:
        ds = xr.open_dataset(url)
        df = ds.to_dataframe().reset_index()
        
        # Extract wave parameters (adjust column names as needed)
        out_df = pd.DataFrame({
            "station_id": station_id,
            "sample_date": pd.to_datetime(df["time"], utc=True),
            "Hs": df["wave_height"].values.astype(float),
            "Tp": df["dominant_period"].values.astype(float),
            "Dm": df["mean_direction"].values.astype(float),
        })
        
        return out_df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch CDIP station {station_id}: {e}")
```

**Template with fallback logic**:

```python
def fetch_cdip_station(station_id: str) -> pd.DataFrame:
    """Fetch CDIP wave data with multiple fallback strategies."""
    
    # Priority 1: THREDDS/OPENDAP (full spectral)
    try:
        url = f"https://thredds.cdip.ucsd.edu/thredds/dodsC/ndbc/{station_id}.nc"
        ds = xr.open_dataset(url)
        df = ds.to_dataframe().reset_index()
        return _extract_wave_params(df, station_id)
    except Exception as e1:
        print(f"  THREDDS failed: {e1}")
    
    # Priority 2: NDBC API (simplified, if available)
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        df = pd.read_csv(url, delim_whitespace=True, comment="#", header=None)
        return _ndbc_to_standard(df, station_id)
    except Exception as e2:
        print(f"  NDBC failed: {e2}")
    
    # Priority 3: Local pre-fetched parquet (no network)
    try:
        path = Path("mbal_history/noaa") / f"noaa_cdip{station_id}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            return _reformat_local(df, station_id)
    except Exception as e3:
        print(f"  Local parquet failed: {e3}")
    
    raise RuntimeError(
        f"All CDIP fetch strategies failed for station {station_id}\n"
        f"  THREDDS error: {e1}\n  NDBC error: {e2}\n  Local parquet error: {e3}"
    )
```

### Step 3: Verify Physical Bounds Validation

The existing `validate_data()` function already checks:

- Wave height: 0–30 meters (beyond 30m is hurricane-scale)
- Dominant period: 1–40 seconds
- Direction: 0–360 degrees

Ensure these bounds match CDIP's actual data range.

---

## Testing the Implementation

### Test with a Single Station

```powershell
python research/bacteria/fetch_cdip_waves.py --force
```

Expected output:
```
============================================================
CDIP Wave Data Fetcher
============================================================

[1/3] Fetching CDIP wave data...
  - Station 041... OK (X rows)
  - Station 043... OK (Y rows)
  ...

============================================================
CDIP Wave Fetch Complete
============================================================
  Output: ...\bacteria_results\cdip_waves\cdip_waves.parquet
  Stations: Z
  Rows: W,XXX
  Date range: YYYY-MM-DD to YYYY-MM-DD
```

### Verify Data Integrity

```powershell
# Check checksums match MANIFEST.sha256
sha256sum bacteria_results/cdip_waves/cdip_waves.parquet

# Compare to MANIFEST entry (should be identical)
```

### Integration Test: Verify Output Format

```python
import pandas as pd

waves = pd.read_parquet("bacteria_results/cdip_waves/cdip_waves.parquet")

print(f"Rows: {len(waves):,}")
print(f"Date range: {waves['sample_date'].min()} to {waves['sample_date'].max()}")
print(f"Columns: {list(waves.columns)}")
print(f"\nStation coverage: {waves['station_id'].nunique()} stations")
print(f"\nWave parameter summary:")
print(waves[['Hs', 'Tp', 'Dm']].describe())

# Note: Beach observations use internal UUID station_ids, not NDBC IDs
# Full integration requires spatial mapping (see "Next Steps" below)
```

### Integration Test: Join to Beach Observations *(Requires Spatial Mapping)*

**Current limitation**: Beach observations use internal UUIDs (`station_id` = e.g., `"0418111a2a6f..."`) while CDIP uses NDBC IDs (`"46042"`).

**Next step**: Create spatial mapping to join these:

```python
# In a future update, create a station_map.parquet:
# station_uuid | ndbc_id | distance_km
# 0418111a...  | 46042   | 15.3

# Then aggregate hourly waves to daily and join by spatial proximity
def add_cdip_wave_features(df, waves_path):
    """Join CDIP waves using spatial map (not yet implemented)."""
    # TODO: Implement spatial mapping + hourly→daily aggregation
    pass
```

---

## Troubleshooting

### Problem: THREDDS connection refused
**Cause**: Network firewall or UCSD server maintenance  
**Fix**: Try again later; fall back to NDBC or local parquet if available

### Problem: Missing columns (`wave_height`, `dominant_period`)
**Cause**: Different netCDF variable names in some stations' metadata  
**Fix**: Inspect `ds.variables.keys()` for actual column names, update `_extract_wave_params()`

### Problem: NaN values everywhere after merge
**Cause**: Date range mismatch (CDIP starts after 2015, beach data starts 2005)  
**Fix**: Use forward-fill or drop samples before CDIP coverage start

---

## Next Steps for AIs (Next Agent)

When adding more wave sources along the CA coast:

1. **Check `fetch_cdip_waves.py`**: Update `CDIP_NDBC_STATIONS` with new station numbers (NDBC format like "46042")
2. **Run fetcher**: Confirm each new station returns valid data
3. **Update `DATA_FLOWS.md`**: Add to source table
4. **Verify output format**: Check that columns are [station_id, sample_date, Hs, Tp, Dm]
5. **Spatial mapping (future)**: To join with beach observations, create a station map linking NDBC IDs to beach UUIDs via geographic proximity

### How to Add More Stations

```python
# In fetch_cdip_waves.py, update CDIP_NDBC_STATIONS:
CDIP_NDBC_STATIONS = {
    "46042": "46042",  # Monterey Bay (already working)
    "46027": "46027",  # Additional CA coast station
    "46011": "46011",  # Another potential addition
}
```

Run `python research/bacteria/fetch_cdip_waves.py --force` to add new stations.

---

## Related Files

| Path | Purpose |
|------|---------|
| `fetch_cdip_waves.py` | Fetcher implementation (fully implemented, uses local parquet fallback) |
| `mbal_drivers_build.py:320-327` | Production CDIP ingestion (uses local parquet) |
| `DATA_FLOWS.md` | 3-layer pipeline architecture |
| `research/bacteria/reproduce/MANIFEST.sha256` | Checksums for reproducibility |

---

**Maintainer**: Monterey Bay AI Lab Engineering  
**Last updated**: 2026-06-10 (updated with implementation status)
