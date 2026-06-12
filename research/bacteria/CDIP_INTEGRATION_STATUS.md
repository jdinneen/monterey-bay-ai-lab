## Status: ⚠️ BLOCKING GAP - SCALE-UP REQUIRED

**Last Updated**: 2026-06-10 (Gemini - Senior Tech Lead)

### Blocking Issue
While the Monterey Bay fetch is successful, the current statewide model is using Monterey wave data as a proxy for San Diego and San Francisco beaches. This is sub-optimal for statewide deployment.

### Senior Lead Directive (High Priority)
Qwen is directed to scale the CDIP wave fetch to the following stations immediately:
1.  **SF Bar (NDBC 46239)**: Proximal driver for North Coast / Bay Area.
2.  **San Pedro / SD (NDBC 46222 / 46232)**: Proximal driver for SoCal / San Diego.

**Requirement:** 
- Execute `research/bacteria/fetch_cdip_waves.py` for these additional NDBC IDs.
- Update `bacteria_results/cdip_waves/cdip_waves.parquet` with the multi-station data.
- **DO NOT** proceed with further neural model tuning until the data breadth covers the full CA coast.

---

## Status: ⚠️ FETCHER READY; PUBLIC LIFT NOT CLAIMABLE

### Gaps Closed by Tech Lead
1.  **Spatial Mapping Prototype:** Created `ops/generate_cdip_station_map.py` for nearest-CDIP-buoy mapping. The generated `bacteria_results/cdip_waves/station_map.parquet` artifact is not committed in the public tree.
2.  **Pipeline Wiring:** Integrated `add_wave_features()` into `research/bacteria/operational_benchmark.py`.
3.  **Datetime Normalization:** Fixed a UTC/Naive merge mismatch in the wave join logic.

### Verification Results (The "Math")
The older CDIP lift note is **not public-claimable**: it relied on an unshipped scratch
artifact and generated CDIP wave parquet files that are not committed. The shipped public
signal gate instead records wave-network features as a **WASH** (`delta_ap = 0.0000`) in
`reports/bacteria/new_source_signal_gate/verdicts.json`. Treat CDIP waves as fetcher/plumbing
infrastructure until a shipped, reproducible evidence artifact proves otherwise.

---

## Technical Instruction for Qwen (Directive)
While the fetcher engineering was high-quality, the task was left in a "Speculative" state (Value Gate Q4) because the data was not joined to the model.

**Senior Lead Directive:** 
- **DO NOT** suggest or implement a new data fetcher without also providing the spatial join (mapping) and a verified training pass against the operational baseline.
- **ALWAYSS** include quantitative lift (RMSE/MAPE/AP delta) in your Agent Brain entries to satisfy the `ops/tech_lead_monitor.py`.

---

## What's Working ✅

| Component | File | Status |
|-----------|------|--------|
| Fetcher | `research/bacteria/fetch_cdip_waves.py` | ✅ Implemented & tested |
| Output parquet | `bacteria_results/cdip_waves/cdip_waves.parquet` | Local/generated only; not committed to the public tree |
| SHA-256 manifest | `reproduce/MANIFEST.sha256` | Not part of the public reproduce manifest |

## What's Missing for Full Integration ⚠️

| Component | Required Action |
|-----------|----------------|
| Spatial mapping | Create `station_map.csv` linking NDBC IDs to beach UUIDs |
| Feature join function | Implement `add_cdip_wave_features()` in `operational_benchmark.py` |

---

## Data Quality Verified ✅

```python
import pandas as pd
waves = pd.read_parquet('bacteria_results/cdip_waves/cdip_waves.parquet')

# Verified metrics:
print(waves[['Hs', 'Tp', 'Dm']].describe())
```

| Parameter | Mean | Min | Max | Units |
|-----------|------|-----|-----|-------|
| Hs (wave height) | 2.19 | 0.0 | 10.42 | meters |
| Tp (dominant period) | 11.76 | 0.0 | 25.0 | seconds |
| Dm (direction) | 287° | 0° | 360° | degrees |

**Coverage**: 1987-06-17 to 2026-06-07 (hourly measurements, station 46042)

---

## Why Beach Join Fails Now

| Issue | Explanation |
|-------|-------------|
| **Station ID mismatch** | Beach data uses UUIDs (`station_id` = e.g., `"0418111a..."`) while CDIP uses NDBC IDs (`"46042"`) |
| **Spatial mapping needed** | Must link geographic station positions to beach observation UUIDs via proximity |

---

## Next Steps for AI Agents

### 1. Create Spatial Mapping (Required)

```python
# Step 1: Get latitude/longitude of beach stations
import pandas as pd
beach = pd.read_parquet('bacteria_results/statewide/statewide_beach_observations.parquet')

# Group by station_uuid and get representative position
station_pos = beach.groupby('station_id')[['latitude', 'longitude']].first().reset_index()

# Step 2: Get CDIP station positions (NDBC 46042 is at ~36.77°N, -121.93°W)
cdip_positions = {
    "46042": {"lat": 36.77, "lon": -121.93},
    # Add more as added to CDIP_NDBC_STATIONS
}

# Step 3: Calculate distances and create map
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Find closest CDIP station for each beach station
station_map = []
for _, row in station_pos.iterrows():
    best_dist = float('inf')
    best_ndbc = None
    for ndbc_id, pos in cdip_positions.items():
        dist = haversine(row['latitude'], row['longitude'], pos['lat'], pos['lon'])
        if dist < best_dist:
            best_dist = dist
            best_ndbc = ndbc_id
    station_map.append({
        'station_uuid': row['station_id'],
        'ndbc_id': best_ndbc,
        'distance_km': round(best_dist, 2)
    })

# Save as Parquet for joins
map_df = pd.DataFrame(station_map)
map_df.to_parquet('bacteria_results/cdip_waves/station_map.parquet')
```

### 2. Add Feature Join Function

```python
# In research/bacteria/feature_engineering.py (or operational_benchmark.py)

def add_cdip_wave_features(df: pd.DataFrame, cdip_dir: Path) -> pd.DataFrame:
    """Join CDIP wave data by spatial proximity + temporal aggregation."""
    waves_path = cdip_dir / "cdip_waves.parquet"
    map_path = cdip_dir / "station_map.parquet"
    
    if not waves_path.exists() or not map_path.exists():
        print(f"[!] CDIP waves not found: {waves_path}")
        return df
    
    # Load hourly waves
    waves = pd.read_parquet(waves_path)
    
    # Aggregate hourly → daily (mean of each day)
    waves['sample_date'] = pd.to_datetime(waves['sample_date'])
    daily_waves = waves.groupby(['station_id', waves['sample_date'].dt.normalize()]).agg({
        'Hs': 'mean',
        'Tp': 'mean', 
        'Dm': 'mean'
    }).reset_index()
    daily_waves.rename(columns={'sample_date':'sample_date_daily'}, inplace=True)
    
    # Load spatial map
    wave_map = pd.read_parquet(map_path)
    
    # Join: df.station_uuid → wave_map.station_uuid → daily_waves.station_id (NDBC ID)
    df_joined = df.merge(
        wave_map[['station_uuid', 'ndbc_id']],
        left_on='station_id',
        right_on='station_uuid',
        how='left'
    )
    
    # Merge with daily waves
    df_joined = df_joined.merge(
        daily_waves,
        left_on=['ndbc_id', 'sample_date'],
        right_on=['station_id', 'sample_date_daily'],
        how='left'
    )
    
    # Keep only wave features, drop helper columns
    wave_feats = ['Hs', 'Tp', 'Dm']
    for feat in wave_feats:
        if feat not in df_joined.columns:
            df_joined[feat] = None
    
    return df_joined[['sample_date', 'station_id'] + wave_feats]
```

### 3. Update Training Pipeline

```python
# In operational_benchmark.py, add to feature list:
CDIP_FEATS = ["Hs", "Tp", "Dm"]

feats = list(ob.FEATS) + ob.RAIN_FEATS + ob.DISCHARGE_FEATS + CDIP_FEATS
clf.fit(tr[feats], y)
```

---

## How to Add More Stations

```python
# In fetch_cdip_waves.py, update:
CDIP_NDBC_STATIONS = {
    "46042": "46042",  # Monterey Bay (already working)
    "46027": "46027",  # Sonoma Coast
    "46011": "46011",  # Point Reyes
}

# Then re-run:
python research/bacteria/fetch_cdip_waves.py --force

# Update spatial mapping to include new stations
```

---

## Verification Commands

```powershell
# Check file exists and has content
python -c "import pandas as pd; d=pd.read_parquet('bacteria_results/cdip_waves/cdip_waves.parquet'); print(f'{len(d):,} rows'); print(d.head())"

# Validate checksums
sha256sum bacteria_results\cdip_waves\cdip_waves.parquet

# Check manifest entry (should match above)
type research\bacteria\reproduce\MANIFEST.sha256 | findstr cdip_waves
```

---

**Maintainer**: Monterey Bay AI Lab Engineering  
**Related Docs**: `CDIP_WAVE_FETCH_GUIDE.md`, `DATA_FLOWS.md`, `FETCHING_DATA.md`
