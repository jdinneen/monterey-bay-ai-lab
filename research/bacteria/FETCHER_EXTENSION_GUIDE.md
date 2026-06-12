# Data Ingestion Framework: How to Extend

> **BINDING**: All new data sources must follow this pattern. Do not create custom fetchers—extend the existing framework.

---

## Current State (Last Updated: 2026-06-10)

| Fetcher | Status | Purpose |
|---------|--------|---------|
| `fetch_station_geo.py` | ✅ **EXTENSIONS COMPLETE** | Geographic coordinates for 822 stations, 16 counties |
| `fetch_statewide_beachwatch.py` | ✅ Active | CA BeachWatch lab assays (1.3M rows) |
| `fetch_rainfall.py` | ✅ Active | Open-Meteo rainfall grids (91 cells) |
| `fetch_discharge.py` | ✅ Active | USGS river discharge gauges (~2,400) |
| `fetch_cdip_waves.py` | ✅ Active | CDIP/NDBC wave data (station 46042) |

---

## The Extension Template

### Step 1: Add Station Configuration

**File**: `research/bacteria/fetch_your_data.py`

```python
# Define which stations/sources you need
YOUR_STATIONS = {
    "source_id_1": {"name": "Short Name", "type": "data_type"},
    "source_id_2": {"name": "Another Source", "type": "data_type"},
}
```

### Step 2: Implement `fetch_your_datastation(station_id)` 

```python
def fetch_your_data_station(station_id: str) -> pd.DataFrame:
    """Fetch data for one station/source."""
    # Strategy 1: Direct API call
    # Strategy 2: BigQuery query  
    # Strategy 3: Local pre-fetched parquet (fallback)
    
    df = ...  # Your fetch logic here
    
    # StandardOUTPUT FORMAT:
    return pd.DataFrame({
        "station_id": station_id,           # Match your YOUR_STATIONS keys
        "sample_date": ...,                 # datetime64[ns, UTC]
        "value": ...,                       # The measured quantity
        # [optional] latitude, longitude  # If geographic
        # [optional] other params
    })
```

### Step 3: Implement `fetch_all_your_data()`

```python
def fetch_all_your_data() -> pd.DataFrame:
    """Fetch all stations, return combined DataFrame."""
    successes = 0
    failures = 0
    all_data = []
    
    for station_id in YOUR_STATIONS.keys():
        try:
            df = fetch_your_data_station(station_id)
            if not df.empty:
                all_data.append(df)
                successes += 1
        except Exception as e:
            print(f"  {station_id}: FAILED - {e}")
            failures += 1
    
    total = successes + failures
    if total > 0:
        print(f"\n[success_rate] {successes}/{total} stations ({100*successes/total:.1f}%)")
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
```

### Step 4: Validate & Write

```python
def validate_data(df: pd.DataFrame) -> bool:
    """Check data integrity."""
    required_cols = ["station_id", "sample_date", "value"]
    for col in required_cols:
        if col not in df.columns:
            return False
    
    # Physical bounds, non-null checks, etc.
    return True

def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """Write Parquet and compute checksums."""
    df = df.sort_values(["station_id", "sample_date"])
    
    # Standardize types
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype("string")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    
    return {
        "path": str(out_path.relative_to(_PROJECT_ROOT)),
        "size_bytes": out_path.stat().st_size,
        "sha256_checksum": compute_sha256(out_path),
        "rows": len(df),
        "columns": list(df.columns),
    }
```

### Step 5: Update Manifest

```python
def update_manifest(new_entries: list[dict]) -> None:
    """Update MANIFEST.sha256 with new file checksums."""
    manifest = []
    if _MANIFEST_PATH.exists():
        # Read existing entries (sha256sum format)
        with open(_MANIFEST_PATH) as f:
            for line in f:
                parts = line.strip().split("*", 1)
                if len(parts) == 2:
                    manifest.append({
                        "sha256": parts[0].strip(),
                        "path": parts[1].strip()
                    })
    
    # Add/update
    for entry in new_entries:
        existing_idx = next(
            (i for i, e in enumerate(manifest) if e["path"] == entry["path"]),
            None,
        )
        
        checksum_entry = f"{entry['sha256_checksum']} *{entry['path']}"
        
        if existing_idx is not None:
            manifest[existing_idx] = {
                "sha256": entry["sha256_checksum"],
                "path": entry["path"]
            }
        else:
            manifest.append({
                "sha256": entry["sha256_checksum"],
                "path": entry["path"]
            })
    
    # Write back sorted
    manifest.sort(key=lambda x: x["path"])
    with open(_MANIFEST_PATH, "w") as f:
        for e in manifest:
            f.write(f"{e['sha256']} *{e['path']}\n")
```

---

## Key Patterns

### 1. Multi-Strategy Fetching (Fallback Chain)

```python
def fetch_from_source(station_id: str) -> pd.DataFrame:
    """Try multiple strategies in priority order."""
    
    # Strategy A: Direct API call
    try:
        return _fetch_api(station_id)
    except Exception as e1:
        print(f"  API failed: {e1}")
    
    # Strategy B: BigQuery table
    try:
        return _fetch_bq(station_id)
    except Exception as e2:
        print(f"  BigQuery failed: {e2}")
    
    # Strategy C: Pre-fetched parquet (no network needed)
    try:
        return _fetch_local(station_id)
    except Exception as e3:
        print(f"  Local parquet failed: {e3}")
    
    raise RuntimeError(
        f"All fetch strategies failed for {station_id}\n"
        f"  API error: {e1}\n  BigQuery error: {e2}\n  Local error: {e3}"
    )
```

### 2. Quantitative Success Reporting

```python
def fetch_all() -> pd.DataFrame:
    successes = 0
    failures = 0
    
    for station_id in STATIONS.keys():
        try:
            df = fetch_single(station_id)
            if not df.empty:
                all_data.append(df)
                successes += 1
        except Exception as e:
            print(f"  {station_id}: FAILED - {e}")
            failures += 1
    
    total = successes + failures
    rate = 100 * successes / total if total > 0 else 0.0
    print(f"\n[fetch_summary] {successes}/{total} stations succeeded ({rate:.1f}% success rate)")
```

### 3. Standard Output Format

```python
# REQUIRED columns (minimum):
return pd.DataFrame({
    "station_id": ...          # String-like identifier
    "sample_date": ...,        # datetime64[ns, UTC]
    "value": ...,              # Measured quantity
})

# SUGGESTED additional columns:
"latitude", "longitude"     # Geographic position
"source_url"                # Data provenance
```

---

## Integration with Downstream Scripts

### 1. Join to Beach Observations

```python
import pandas as pd

# Load beach data (has station_id, sample_date)
beach = pd.read_parquet("bacteria_results/statewide/statewide_beach_observations.parquet")

# Load your data
your_data = pd.read_parquet("bacteria_results/your_data/your_data.parquet")

# Match datetime types (your_data is likely hourly, beach is daily)
your_data["sample_date"] = pd.to_datetime(your_data["sample_date"]).dt.normalize()

# Merge
merged = beach.merge(
    your_data[["station_id", "sample_date", "value"]],
    on=["station_id", "sample_date"],
    how="left"
)

print(f"Merged: {len(merged):,} rows")
print(f"Non-null values: {merged['value'].notna().sum():,}")
```

### 2. Add to Feature Engineering

```python
# In operational_benchmark.py or your feature adder:

def add_your_data_features(df: pd.DataFrame, your_dir: Path) -> pd.DataFrame:
    """Merge your data onto beach observations."""
    path = your_dir / "your_data.parquet"
    
    if not path.exists():
        print(f"[!] Your data not found: {path}")
        return df
    
    data = pd.read_parquet(path)
    
    # Aggregate hourly → daily (if needed)
    if data["sample_date"].dtype == "datetime64[ns, UTC]":
        data["day"] = data["sample_date"].dt.normalize()
        data = data.groupby(["station_id", "day"]).agg({"value": "mean"}).reset_index()
        data.rename(columns={"day": "sample_date"}, inplace=True)
    
    # Merge
    df = df.merge(
        data[["station_id", "sample_date", "value"]],
        on=["station_id", "sample_date"],
        how="left"
    )
    
    return df
```

### 3. Update Feature List

```python
# In operational_benchmark.py:
YOUR_FEATS = ["your_value"]

feats = list(ob.FEATS) + ob.RAIN_FEATS + ob.DISCHARGE_FEATS + YOUR_FEATS

clf.fit(tr[feats], y)
```

---

## Verification Checklist

After implementing a new fetcher, run:

```powershell
# 1. Run fetcher (with --force to refresh)
python research/bacteria/fetch_your_data.py --force

# 2. Verify checksum matches manifest
sha256sum bacteria_results/your_data/your_data.parquet
type research\bacteria\reproduce\MANIFEST.sha256 | findstr your_data

# 3. Test data quality
python -c "import pandas as pd; d=pd.read_parquet('bacteria_results/your_data/your_data.parquet'); print(f'{len(d):,} rows'); print(d[['station_id','sample_date','value']].head())"

# 4. Test join to beach observations
python -c "import pandas as pd; d=pd.read_parquet('...'); b=pd.read_parquet('bacteria_results/statewide/...'); merged=b.merge(d[['station_id','sample_date','value']], on=['station_id','sample_date'], how='left'); print(f'Joined {merged[\"value\"].notna().sum():,} rows')"
```

---

## Troubleshooting

### Problem: "MANIFEST.sha256 mismatch"
**Solution**: Re-run fetcher with `--force`; checksums validated on every run

### Problem: "Empty dataframe returned"
**Cause**: API rate limit or date range error
**Fix**: Check `sample_date >= '2005-01-01'` in queries

### Problem: Station ID mismatch with beach observations
**Cause**: Different ID schemes (e.g., NDBC IDs vs UUIDs)
**Fix**: Create a station map linking the two ID systems via geographic proximity

---

## Next Steps for Adding More Data Sources

Following the **Value Gate** criteria, prioritize sources that:

1. ✅ **Beat a real baseline** - Add detectable signal to AUC/AP metrics
2. ✅ **Additive & reversible** - New file, new column, no breaking changes  
3. ✅ **Testable** - Concrete pass/fail metrics (e.g., +ΔAP on stratum)
4. ✅ **Valuable NOW** - Not speculative, actually improves current results
5. ✅ **Genuinely new** - Doesn't duplicate existing drivers

See `docs/VALUE_GATE.md` for full rubric.

---

## Related Files

| Path | Purpose |
|------|---------|
| `research/bacteria/fetch_station_geo.py` | Station coordinates (template for extensions) |
| `research/bacteria/fetch_statewide_beachwatch.py` | BeachWatch data ingestion |
| `research/bacteria/fetch_rainfall.py` | Open-Meteo rainfall grids |
| `research/bacteria/fetch_discharge.py` | USGS discharge gauges |
| `research/bacteria/operational_benchmark.py` | Feature join & training pipeline |
| `reports/station_geo.parquet` | Geolocation lookup (822 stations) |

---

**Maintainer**: Monterey Bay AI Lab Engineering  
**Last updated**: 2026-06-10
