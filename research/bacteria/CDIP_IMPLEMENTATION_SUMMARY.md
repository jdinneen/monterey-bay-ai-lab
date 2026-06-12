# CDIP Wave Data Fetcher Implementation Summary

## What Was Done

### 1. Architecture Documentation (3 files)

| File | Purpose |
|------|---------|
| `FETCHING_DATA.md` | High-level overview: 3-layer pipeline, current sources, running fetchers |
| `DATA_FLOWS.md` | Detailed technical spec: feature engineering, causality rules, example code |
| `CDIP_WAVE_FETCH_GUIDE.md` | Implementation guide with code examples for CDIP integration |

### 2. Fetcher Templates (1 file)

| File | Status | Description |
|------|--------|-------------|
| `fetch_cdip_waves.py` | **Implemented** | Full fetch logic with THREDDS/OPENDAP fallback chain |

### 3. Implementation Details

The fetcher implements a **fallback strategy** for maximum reliability:

1. **THREDDS/OPENDAP (Primary)**: Direct access to UCSD netCDF files with full spectral data
2. **NDBC API (Fallback)**: If THREDDS unavailable, try NDBC-compatible format
3. **Local Parquet (Offline)**: Skip network if pre-fetched data exists

### 4. Expected Output Format

```python
# Columns in bacteria_results/cdip_waves/cdip_waves.parquet:
- station_id: string (e.g., "041", "065")
- sample_date: datetime64[ns, UTC] (hourly measurements)
- Hs: float32 (significant wave height, meters)
- Tp: float32 (dominant/peak period, seconds)
- Dm: float32 (mean wave direction, degrees 0-360)
```

---

## Next Steps to Activate

### Option A: Install xarray (Recommended)

```powershell
pip install xarray netCDF4
python research/bacteria/fetch_cdip_waves.py --force
```

### Option B: Use local pre-fetched data

If you have `mbal_history/noaa/noaa_cdip*.parquet` files, the fetcher will auto-detect and use them.

### Option C: Manual testing (single station)

```python
python -c "
import sys
sys.path.insert(0, 'research/bacteria')
from fetch_cdip_waves import fetch_cdip_station

df = fetch_cdip_station('065')  # Moss Landing
print(f'Fetched {len(df)} rows from station 065')
print(df.head())
"
```

---

## Validation Checklist

After running the fetcher, verify:

```powershell
# 1. Check file exists
dir bacteria_results\cdip_waves\cdip_waves.parquet

# 2. Verify SHA-256 checksum matches manifest
sha256sum bacteria_results\cdip_waves\cdip_waves.parquet

# 3. Review manifest entry (JSON lines format)
type research\bacteria\reproduce\MANIFEST.sha256 | findstr cdip_waves

# 4. Test join to beach observations
python -c "
import pandas as pd
beach = pd.read_parquet('bacteria_results/statewide/statewide_beach_observations.parquet')
waves = pd.read_parquet('bacteria_results/cdip_waves/cdip_waves.parquet')

# Align日期 (waves are hourly, beach are daily)
waves['sample_date'] = pd.to_datetime(waves['sample_date']).dt.normalize()
merged = beach.merge(waves, on=['station_id','sample_date'], how='left')
print(f'Merged {len(merged)} rows, CDIP columns: {[c for c in merged.columns if c in [\"Hs\",\"Tp\",\"Dm\"]]}')
"
```

---

## Integration with Feature Engineering

Once data is fetched, add wave features to the model:

```python
# In research/bacteria/feature_engineering.py (or operational_benchmark.py)
CDIP_FEATS = ["Hs", "Tp", "Dm"]

def add_cdip_wave_features(df, cdip_dir):
    """Join CDIP wave data on station_id + sample_date."""
    waves_path = Path(cdip_dir) / "cdip_waves.parquet"
    
    if not waves_path.exists():
        print(f"[!] CDIP waves not found: {waves_path}")
        return df
    
    waves = pd.read_parquet(waves_path)
    # Convert hourly to daily (e.g., take mean or latest reading before sample_date)
    waves["sample_date"] = pd.to_datetime(waves["sample_date"]).dt.normalize()
    
    # Merge
    merged = df.merge(waves, on=["station_id", "sample_date"], how="left")
    
    return merged
```

---

## Troubleshooting

### Problem: `xarray` not installed
**Solution**: `pip install xarray netCDF4`

### Problem: THREDDS connection timeout
**Cause**: Network firewall or UCSD server maintenance  
**Fix**: Wait and retry; fetcher will fall back to NDBC or local parquet

### Problem: Missing wave parameters in netCDF
**Cause**: Variable names differ between CDIP stations  
**Fix**: The code auto-detects columns by keyword matching (wave_height, period, direction)

---

## Related Files

| Path | Purpose |
|------|---------|
| `research/bacteria/fetch_cdip_waves.py` | Fetcher implementation |
| `mbal_drivers_build.py:320-327` | Production CDIP ingestion (already integrated) |
| `DATA_FLOWS.md` | Architecture overview |
| `CDIP_WAVE_FETCH_GUIDE.md` | Detailed implementation guide |

---

**Implementation Date**: 2026-06-10  
**Status**: ✅ Complete (ready for testing)
