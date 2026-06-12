# Data Fetching Strategy for Monterey Bay AI Lab

> **This is a BINDING ARCHITECTURE DECISION**: All new data sources must follow this pattern. Do not create custom fetchers—extend the existing framework.

---

## Philosophy: One Framework, Many Sources

We do NOT write custom fetchers per data provider. Instead:

1. **Extend** `research/bacteria/fetch_*.py` to support new sources
2. **Standardize** output format (Parquet in `bacteria_results/`)
3. **Automate** reproducibility via `MANIFEST.sha256` verification

This ensures:
- 🔒 **No API key leakage**: All credentials via environment variables
- 📦 **Immutable cache**: SHA-256 manifest validates input data integrity
- 🔄 **Reproducible runs**: Same inputs → same outputs (random_state=42 in all scripts)
- 👥 **Team collaboration**: Everyone uses the same fetcher code

---

## Current Sources (As of 2026-06-10)

| Source | Fetcher | Output File | Content |
|--------|---------|-------------|---------|
| CA BeachWatch | `fetch_statewide_beachwatch.py` | `statewide/statewide_beach_observations.parquet` | 1.3M lab assays, 802 stations |
| Rainfall (Open-Meteo) | `fetch_rainfall.py` | `rainfall/rainfall_grid.parquet` | 0.1° grid daily precip |
| River Discharge (USGS NWIS) | `fetch_discharge.py` | `discharge/discharge_gauge.parquet` | Daily flow at ~2,400 gauges |

---

## Adding a NEW Data Source: The Template

### Step 1: Create the Fetcher

**File**: `research/bacteria/fetch_NEW_SOURCE.py`

```python
#!/usr/bin/env python3
"""Fetch NEW SOURCE data along the California coast.

Follows MONTEREY BAY AI LAB fetch strategy:
- Read-only, public sources only (no API keys required)
- Output: Parquet in bacteria_results/NEW_SOURCE/
- Manifest checksum: MANIFEST.sha256 (git-tracked)

Usage:
    python research/bacteria/fetch_NEW_SOURCE.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (ALL environment-based for portability)
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "bacteria_results" / "NEW_SOURCE"
_MANIFEST_PATH = _PROJECT_ROOT / "research" / "bacteria" / "reproduce" / "MANIFEST.sha256"

# Source-specific configs
SOURCE_API_URL = "https://api.example.com/data"  # Replace with actual endpoint


def compute_sha256(path: Path) -> str:
    """Hash a file (streaming, handles large files)."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH LOGIC (Public data only!)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_new_source_data() -> pd.DataFrame:
    """
    Fetch NEW SOURCE data from the public API.
    
    RETURN FORMAT: pandas DataFrame with at minimum these columns:
        - station_id: str (unique identifier)
        - sample_date: datetime64[ns, UTC] or date
        - value: float (the measured quantity)
        - [optional] latitude, longitude
    
    Raises:
        RuntimeError: If fetch fails after retries
    """
    # TODO: Implement your actual fetch logic here
    raise NotImplementedError("Implement fetch for NEW SOURCE")


def validate_data(df: pd.DataFrame) -> bool:
    """
    Check data integrity BEFORE writing.
    
    Return True if valid, False otherwise.
    Common checks:
        - Non-empty dataframe
        - Required columns present
        - No NaN in critical fields (station_id, sample_date)
        - Values within physical bounds
    """
    required_cols = ["station_id", "sample_date", "value"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing required column: {col}")
            return False
    
    if df.empty:
        print("[!] Empty dataframe — nothing to save")
        return False
    
    # Physical bounds check (example: wave height can't be negative)
    if (df["value"] < 0).any():
        print(f"[!] Negative values found in 'value' column")
        return False
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITING (Standardized format)
# ─────────────────────────────────────────────────────────────────────────────

def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """
    Write Parquet and compute checksums.
    
    Returns metadata dict for manifest.
    """
    # Sort by station then date (deterministic ordering)
    df = df.sort_values(["station_id", "sample_date"])
    
    # Convert object columns to standard types
    for col in ["station_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Write Parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    
    return {
        "path": str(out_path.relative_to(_PROJECT_ROOT)),
        "size_bytes": out_path.stat().st_size,
        "sha256_checksum": compute_sha256(out_path),
        "rows": len(df),
        "columns": list(df.columns),
    }


def update_manifest(new_entries: list[dict]) -> None:
    """
    Update MANIFEST.sha256 with new file checksums.
    
    Manifest format (JSON lines):
        {"path": "...", "size_bytes": 1234, "sha256_checksum": "..."}
    """
    # Read existing manifest
    manifest = []
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    manifest.append(json.loads(line))
    
    # Add/update new entries
    for entry in new_entries:
        existing_idx = next(
            (i for i, e in enumerate(manifest) if e["path"] == entry["path"]),
            None,
        )
        if existing_idx is not None:
            manifest[existing_idx] = entry
        else:
            manifest.append(entry)
    
    # Write back sorted by path
    manifest.sort(key=lambda x: x["path"])
    with open(_MANIFEST_PATH, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"[✓] Updated MANIFEST.sha256 ({len(manifest)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help=f"Output directory (default: {_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch even if output exists"
    )
    args = parser.parse_args()
    
    # Setup
    OUTPUT_FILE = args.out_dir / "new_source_data.parquet"
    
    print(f"[NEW SOURCE fetcher] Starting...")
    
    # Skip if exists and not forced
    if OUTPUT_FILE.exists() and not args.force:
        print(f"[!] Output exists: {OUTPUT_FILE}")
        print("[!] Use --force to re-fetch")
        return 0
    
    # Fetch data
    print("[1/3] Fetching NEW SOURCE data from public API...")
    try:
        df = fetch_new_source_data()
    except Exception as e:
        print(f"[!] Fetch failed: {e}")
        return 1
    
    # Validate
    print("[2/3] Validating data integrity...")
    if not validate_data(df):
        print("[!] Validation FAILED — aborting")
        return 1
    
    # Write output
    print(f"[3/3] Writing to {OUTPUT_FILE}...")
    metadata = write_output(df, OUTPUT_FILE)
    
    # Update manifest
    update_manifest([metadata])
    
    # Summary
    print()
    print("=" * 60)
    print("NEW SOURCE Data Fetch Complete")
    print("=" * 60)
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Rows: {len(df):,}")
    print(f"  Stations: {df['station_id'].nunique() if 'station_id' in df.columns else 'N/A'}")
    print(f"  Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

### Step 2: Register in MANIFEST (after first run)

```bash
cd /path/to/repo

# Run the fetcher once
python research/bacteria/fetch_NEW_SOURCE.py

# Verify SHA-256 checksums match
sha256sum bacteria_results/NEW_SOURCE/new_source_data.parquet
# Compare to MANIFEST.sha256 entry
```

---

### Step 3: Use in Downstream Scripts

```python
from research.bacteria import fetch_new_source as fs

def load_and_process():
    df = pd.read_parquet("bacteria_results/NEW_SOURCE/new_source_data.parquet")
    
    # Merge with beach observations using standard keys
    merged = beach_obs.merge(
        df,
        left_on=["station_id", "sample_date"],
        right_on=["station_id", "sample_date"],
        how="left",
    )
    
    return merged
```

---

## Current Source Details (Reference Files)

### 1. BeachWatch (Statewide Monitoring)
**Fetcher**: `research/bacteria/fetch_statewide_beachwatch.py`  
**Output**: `bacteria_results/statewide/statewide_beach_observations.parquet`  
**Columns**: 
- `sample_date`, `county`, `beach_name`, `station_id`
- `source_parameter` (enterococcus/fecals/total_coliforms)
- `result_value_numeric` (MPN/100mL or CFU/100mL)

### 2. Rainfall (Open-Meteo Reanalysis)
**Fetcher**: `research/bacteria/fetch_rainfall.py`  
**Output**: `bacteria_results/rainfall/rainfall_grid.parquet`  
**Columns**: 
- `date`, `grid_lat`, `grid_lon`
- `precip_mm` (daily total)

### 3. River Discharge (USGS NWIS)
**Fetcher**: `research/bacteria/fetch_discharge.py`  
**Output**: `bacteria_results/discharge/discharge_gauge.parquet`  
**Columns**: 
- `date`, `gauge_id`
- `discharge_cfs` (cubic feet per second)

### 4. Station-to-Source Maps
**Fixed CSVs**: Committed to repo (no fetch needed)
- `bacteria_results/rainfall/station_grid_map.parquet`: station → 0.1° grid cell
- `bacteria_results/discharge/station_gauge_map.parquet`: station → nearest USGS gauge

---

## Prohibited Patterns (DO NOT DO)

| Pattern | Why | Example |
|---------|-----|---------|
| **Hardcoded API keys** | Security risk; breaks reproducibility | `"api_key": "12345"` |
| **Write to project root** | Breaks containerization | `open("data.csv", "w")` |
| **Insecure HTTP** | MITM vulnerability | `http://example.com/data` |
| **Manual CSV parsing** | encoding/line-ending bugs | `csv.reader(open(...))` |

---

## Troubleshooting

### Problem: "MANIFEST.sha256 mismatch"
**Solution**: Re-run fetcher; checksums are validated on every run

### Problem: "Empty dataframe returned"
**Cause**: API rate limit or date range error  
**Fix**: Check `sample_date >= '2005-01-01'` in SQL queries

### Problem: "Permission denied" on write
**Cause**: File lock from previous failed run  
**Fix**: `rm bacteria_results/NEW_SOURCE/*.parquet.lock` then retry

---

## Extension Checklist

When adding a new source:

- [ ] Created `fetch_NEW_SOURCE.py` following template above
- [ ] Uses only public data sources (no API keys)
- [ ] Outputs Parquet to `bacteria_results/NEW_SOURCE/`
- [ ] Validate function checks physical bounds
- [ ] MANIFEST.sha256 updated after first run
- [ ] Test script added: `tests/test_fetch_NEW_SOURCE.py`

---

## Related Files

| Path | Purpose |
|------|---------|
| `research/bacteria/fetch_statewide_beachwatch.py` | CA beach monitoring (BigQuery) |
| `research/bacteria/fetch_rainfall.py` | Open-Meteo rainfall grids |
| `research/bacteria/fetch_discharge.py` | USGS river discharge |
| `research/bacteria/reproduce/MANIFEST.sha256` | Data integrity checksums |
| `mbal_drivers_build.py` | Driver table assembly (uses fetched data) |

---

**Last updated**: 2026-06-10  
**Maintainer**: Monterey Bay AI Lab Engineering
