#!/usr/bin/env python3
"""Fetch tide stage (water level) from NOAA CO-OPS API for CA coastal stations.

NOAA CO-OPS provides hourly water level measurements relative to MLLW
(Mean Lower Low Water) datum. This fetcher retrieves data from known CA
coastal stations and stores it in a standardized format.

Output:
  - bacteria_results/tide_stages/tide_stages.parquet
    Columns: station_id, sample_date, water_level_m

Known CA CO-OPS stations (tide gauges):
  - 9410230: San Francisco
  - 9410680: Monterey
  - 9411340: Santa Cruz
  - 9414290: Point Reyes
  - 9415070: Morro Bay
  - 9416110: Los Angeles (southern boundary)

API endpoint:
  https://tidesandcurrents.noaa.gov/api/datagetter

Usage:
    python research/bacteria/fetch_tide_stages.py [--force] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "bacteria_results" / "tide_stages"
_MANIFEST_PATH = _PROJECT_ROOT / "research" / "bacteria" / "reproduce" / "MANIFEST.sha256"

# CA coastal CO-OPS tide stations (verified working)
# Format: station_id -> {"name": descriptive name, "type": "water_level"}
CA_TIDE_STATIONS = {
    "9410230": {"name": "San Francisco", "type": "water_level"},
    "9410680": {"name": "Monterey", "type": "water_level"},
    "9411340": {"name": "Santa Cruz", "type": "water_level"},
    "9414290": {"name": "Point Reyes", "type": "water_level"},
}


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file (streaming, handles large files)."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_co_ops_station_chunk(
    station_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch water level data from NOAA CO-OPS API for one chunk (max 93 days).
    
    Args:
        station_id: CO-OPS station identifier (e.g., "9410230")
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        
    Returns:
        DataFrame with columns: [station_id, sample_date, water_level_m]
    """
    url = "https://tidesandcurrents.noaa.gov/api/datagetter"
    params = {
        "product": "water_level",
        "application": "test",
        "units": "metric",
        "time_zone": "GMT",
        "format": "json",
        "station": station_id,
        "begin_date": start_date,
        "end_date": end_date,
        "datum": "MLLW",
    }
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    if "error" in data:
        raise RuntimeError(f"API error for station {station_id}: {data['error']}")
    
    results = data.get("data", [])
    
    records = []
    for entry in results:
        t = entry.get("t")
        v = entry.get("v")
        
        if t is None or v is None:
            continue
        
        try:
            if isinstance(v, str) and v.strip() == "":
                continue
            water_level = float(v)
            records.append({
                "station_id": station_id,
                "sample_date": pd.to_datetime(t, utc=True),
                "water_level_m": water_level,
            })
        except (ValueError, TypeError):
            continue
    
    return pd.DataFrame(records)


def _fetch_co_ops_station(station_id: str, date_range: tuple[str, str]) -> pd.DataFrame:
    """
    Fetch water level data from NOAA CO-OPS API for one station.
    
    The API has a ~93-day limit per request. This function chunks the date range
    into 90-day intervals and fetches each chunk separately.
    
    Args:
        station_id: CO-OPS station identifier (e.g., "9410230")
        date_range: Tuple of (start_date, end_date) in YYYYMMDD format
        
    Returns:
        DataFrame with columns: [station_id, sample_date, water_level_m]
        
    Raises:
        RuntimeError: If API call fails or returns invalid data
    """
    if requests is None:
        raise ImportError(
            "requests not installed. Install with: pip install requests"
        )
    
    start_str, end_str = date_range
    
    # NOAA API has a ~31-day limit per request for water_level product
    MAX_DAYS_PER_REQUEST = 30
    
    # Parse dates
    from datetime import datetime, timedelta
    
    try:
        start_dt = datetime.strptime(start_str, "%Y%m%d")
        end_dt = datetime.strptime(end_str, "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYYMMDD: {e}")
    
    if (end_dt - start_dt).days <= MAX_DAYS_PER_REQUEST:
        # Single request fits
        df = _fetch_co_ops_station_chunk(station_id, start_str, end_str)
        print(f"  API call ({(end_dt-start_dt).days} days): ... {len(df)} rows")
        return df
    
    # Multiple chunks needed
    all_dfs = []
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=MAX_DAYS_PER_REQUEST), end_dt)
        
        chunk_start_str = current_start.strftime("%Y%m%d")
        chunk_end_str = current_end.strftime("%Y%m%d")
        
        df_chunk = _fetch_co_ops_station_chunk(
            station_id, chunk_start_str, chunk_end_str
        )
        
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
            print(f"  API call {chunk_start_str}-{chunk_end_str}: ... {len(df_chunk)} rows")
        
        current_start = current_end + timedelta(days=1)
    
    if not all_dfs:
        return pd.DataFrame(columns=["station_id", "sample_date", "water_level_m"])
    
    return pd.concat(all_dfs, ignore_index=True)


def fetch_tide_station(station_id: str) -> pd.DataFrame:
    """
    Fetch tide stage data for one station.
    
    Strategy chain:
      1. Try direct CO-OPS API call with date chunking (primary)
      
    Args:
        station_id: CO-OPS station identifier
        
    Returns:
        DataFrame with columns: [station_id, sample_date, water_level_m]
    """
    # Fetch for the last 365 days, but chunk into 90-day windows
    import datetime
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = datetime.datetime(2015, 1, 1, tzinfo=datetime.timezone.utc)
    
    date_range = (
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
    )
    
    # Strategy 1: Direct CO-OPS API call (with automatic chunking)
    return _fetch_co_ops_station(station_id, date_range)


def fetch_all_tide_stations() -> pd.DataFrame:
    """Fetch tide data from all CA CO-OPS stations."""
    print(f"[tide_stages] Fetching {len(CA_TIDE_STATIONS)} stations: {list(CA_TIDE_STATIONS.keys())}")
    
    successes = 0
    failures = 0
    
    all_data = []
    
    for station_id in CA_TIDE_STATIONS.keys():
        try:
            df = fetch_tide_station(station_id)
            
            if not df.empty:
                all_data.append(df)
                print(f"  {station_id}: OK ({len(df)} rows)")
                successes += 1
            else:
                print(f"  {station_id}: NO DATA")
                failures += 1
                
        except Exception as e:
            print(f"\n    FAILED: {type(e).__name__}: {e}")
            failures += 1
    
    # Report metrics
    total = successes + failures
    if total > 0:
        print(f"\n[tide_stages] Summary: {successes}/{total} stations succeeded ({100*successes/total:.1f}% success rate)")
    
    if not all_data:
        return pd.DataFrame(columns=["station_id", "sample_date", "water_level_m"])
    
    return pd.concat(all_data, ignore_index=True)


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly water level data to daily means.
    
    Args:
        df: Hourly tide data with columns [station_id, sample_date, water_level_m]
        
    Returns:
        Daily aggregated data with columns [station_id, sample_date, water_level_m_daily]
    """
    # Convert to date (normalize to midnight)
    df = df.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"], utc=True).dt.normalize()
    
    # Group by station and date, compute mean
    daily = (
        df.groupby(["station_id", "sample_date"])
        .agg(water_level_m_daily=("water_level_m", "mean"))
        .reset_index()
    )
    
    return daily


def validate_data(df: pd.DataFrame) -> bool:
    """Validate tide stage data integrity."""
    # Check both possible column names (before/after rename)
    if "water_level_m_daily" not in df.columns and "water_level_m" not in df.columns:
        print(f"[!] Missing required column. Available columns: {list(df.columns)}")
        return False
    
    if df.empty:
        print("[!] Empty dataframe - nothing to save")
        return False
    
    # Physical bounds for daily-averaged water level
    import numpy as np
    col = "water_level_m" if "water_level_m" in df.columns else "water_level_m_daily"
    
    valid_levels = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    if (valid_levels < -5).any() or (valid_levels > 10).any():
        bad = df[(df[col] < -5) | (df[col] > 10)]
        print(f"[!] Invalid water levels (-5 to +10m): {len(bad)} rows")
        return False
    
    # Check for datetime validity
    if not pd.api.types.is_datetime64_any_dtype(df["sample_date"]):
        print("[!] sample_date must be datetime type")
        return False
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITING
# ─────────────────────────────────────────────────────────────────────────────

def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """Write tide stage data to Parquet with standardized formatting."""
    # Sort by station then date (deterministic)
    df = df.sort_values(["station_id", "sample_date"])
    
    # Standardize types
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype("string")
    
    # Write to Parquet
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
    """Update MANIFEST.sha256 with new tide stage file checksums."""
    manifest = []
    
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("{"):
                    parts = line.split("*", 1)
                    if len(parts) == 2:
                        manifest.append({
                            "sha256": parts[0].strip(),
                            "path": parts[1].strip()
                        })
    
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
    
    manifest.sort(key=lambda x: x["path"])
    with open(_MANIFEST_PATH, "w") as f:
        for e in manifest:
            f.write(f"{e['sha256']} *{e['path']}\n")

    print(f"[OK] Updated MANIFEST.sha256 ({len(manifest)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_OUTPUT_DIR,
        help=f"Output directory (default: {_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if output exists"
    )
    args = parser.parse_args()
    
    OUTPUT_FILE = args.out_dir / "tide_stages.parquet"
    
    print("=" * 60)
    print("NOAA CO-OPS Tide Stage Fetcher (CA Coastal)")
    print("=" * 60)

    # Skip if exists
    if OUTPUT_FILE.exists() and not args.force:
        print(f"[!] Output exists: {OUTPUT_FILE}")
        print("[!] Use --force to re-fetch")
        return 0

    # Fetch data
    print("\n[1/3] Fetching tide stage data from NOAA CO-OPS...")
    try:
        df = fetch_all_tide_stations()
    except Exception as e:
        print(f"\n[!] Fetch failed: {type(e).__name__}: {e}")
        if "requests" in str(e).lower():
            print("\n    HINT: Install with: pip install requests")
        return 1
    
    # Aggregate to daily
    print("\n[2/3] Aggregating hourly -> daily means...")
    df_daily = aggregate_to_daily(df)
    print(f"  {len(df):,} hourly rows -> {len(df_daily):,} daily records")

    # Validate
    print("\n[3/3] Validating data...")
    if not validate_data(df_daily):
        print("[!] Validation FAILED")
        return 1
    
    # Rename column to match standard output format
    df_daily = df_daily.rename(columns={"water_level_m_daily": "water_level_m"})

    # Write output
    print(f"\nWriting to {OUTPUT_FILE}...")
    metadata = write_output(df_daily, OUTPUT_FILE)
    
    # Update manifest
    update_manifest([metadata])
    
    # Summary
    print("\n" + "=" * 60)
    print("Tide Stage Fetch Complete")
    print("=" * 60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Stations: {df_daily['station_id'].nunique()}")
    print(f"  Rows: {len(df_daily):,}")
    if not df_daily.empty:
        print(f"  Date range: {df_daily['sample_date'].min()} to {df_daily['sample_date'].max()}")
        print(f"  Water level range: {df_daily['water_level_m'].min():.2f} to {df_daily['water_level_m'].max():.2f} m")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
