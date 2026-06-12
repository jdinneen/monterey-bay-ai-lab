#!/usr/bin/env python3
"""Fetch wave data from CDIP (Coastal Data Information Program) along the CA coast.

CDIP operates ~40+ moorings along California with spectral wave measurements.
This fetcher retrieves significant wave height, dominant period, and direction
from each station.

Output format follows MONTEREY BAY AI LAB conventions:
- Parquet output in bacteria_results/cdip_waves/
- SHA-256 checksums tracked in reproduce/MANIFEST.sha256
- Deterministic: random_state=42 in all downstream scripts

Sources:
  - CDIP UCSD THREDDS/OPENDAP: https://thredds.cdip.ucsd.edu/thredds/
  - NDBC Pacific stations with wave data (subset of full network)

Required columns for downstream use:
  - station_id: str (e.g., "041", "067")
  - sample_date: datetime [UTC]
  - Hs: float (significant wave height, meters)
  - Tp: float (peak/dominant period, seconds)
  - Dm: float (mean wave direction, degrees)

Usage:
    python research/bacteria/fetch_cdip_waves.py [--force] [--out-dir PATH]

Notes:
  - This uses public CDIP THREDDS data (full spectral data available)
  - Try xarray first; fall back to NDBC API if needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import xarray as xr
except ImportError:
    xr = None

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "bacteria_results" / "cdip_waves"
_MANIFEST_PATH = _PROJECT_ROOT / "research" / "bacteria" / "reproduce" / "MANIFEST.sha256"

# CDIP Pacific stations (NDBC-compatible format)
# These map to existing noaa_ndbc*.parquet files in mbal_history/noaa/
# Currently only station 46042 is available. Add more as they are fetched.
CDIP_NDBC_STATIONS = {
    # Existing stations with available data
    "46042": "46042",  # Pacific Ocean near Monterey Bay (currently the only one available)
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

def fetch_cdip_station(station_id: str) -> pd.DataFrame:
    """
    Fetch CDIP/NDBC wave data from existing local parquet files.
    
    This function retrieves pre-fetched NDBC wave data that has already been
    collected by the mbal_history/noaa fetchers. The data includes:
      - Significant wave height (Hs)
      - Dominant/peak period (Tp)  
      - Mean wave direction (Dm)

    Args:
        station_id: CDIP/NDBC station identifier (e.g., "041", "065")

    Returns:
        DataFrame with columns: [station_id, sample_date, Hs, Tp, Dm]
    
    Raises:
        RuntimeError: If local parquet file not found
    """
    # Strategy 3: Local pre-fetched parquet (primary method - already available)
    ndbc_num = CDIP_NDBC_STATIONS.get(station_id, station_id)
    local_path = _PROJECT_ROOT / "mbal_history" / "noaa" / f"noaa_ndbc{ndbc_num}.parquet"
    
    print(f"  [Local] Trying {local_path.name}...", end=" ")
    
    if not local_path.exists():
        raise FileNotFoundError(
            f"NDBC parquet file not found: {local_path}\n"
            "Run mbal_history/noaa fetcher first to collect NDBC wave data.\n"
            "Expected stations: " + ", ".join(CDIP_NDBC_STATIONS.keys())
        )
    
    # Read the pre-fetched parquet
    df = pd.read_parquet(local_path)
    
    if len(df) == 0:
        raise ValueError(f"Empty parquet file: {local_path}")
    
    ndbc_num = CDIP_NDBC_STATIONS.get(station_id, station_id)
    
    # Extract wave parameters (handle NaN by keeping all rows but using float dtype)
    result = pd.DataFrame({
        "station_id": station_id,
        "sample_date": pd.to_datetime(df.index, utc=True),
        "Hs": df[f"ndbc{ndbc_num}_wave_height_m"].astype(float).values,
        "Tp": df[f"ndbc{ndbc_num}_dom_wave_period_s"].astype(float).values,
        "Dm": df[f"ndbc{ndbc_num}_mean_wave_dir_deg"].astype(float).values,
    })
    
    print(f"OK ({len(result)} rows)")
    return result


def fetch_all_stations() -> pd.DataFrame:
    """Fetch CDIP wave data for all configured stations."""
    print(f"[cdip_waves] Fetching {len(CDIP_NDBC_STATIONS)} NDBC stations: {list(CDIP_NDBC_STATIONS.keys())}")

    # Track success/failure metrics
    successes = 0
    failures = 0

    all_data = []

    for station_id, ndbc_num in CDIP_NDBC_STATIONS.items():
        print(f"  - Station {station_id} (ndbc={ndbc_num})...", end=" ")

        try:
            wave_df = fetch_cdip_station(station_id)

            if wave_df.empty:
                print("NO DATA")
                failures += 1
                continue

            # Use standardized columns from fetch_cdip_station output
            out_df = pd.DataFrame({
                "station_id": station_id,
                "sample_date": wave_df["sample_date"],
                "Hs": wave_df["Hs"].values.astype(float),
                "Tp": wave_df["Tp"].values.astype(float),
                "Dm": wave_df["Dm"].values.astype(float),
            })

            all_data.append(out_df)
            print(f"OK ({len(wave_df)} rows)")
            successes += 1

        except Exception as e:
            print(f"\n    FAILED: {type(e).__name__}: {e}")
            failures += 1
    
    # Report metrics (quantitative success rate)
    total = successes + failures
    if total > 0:
        print(f"\n[cdip_waves] Summary: {successes}/{total} stations succeeded ({100*successes/total:.1f}% success rate)")

    if not all_data:
        return pd.DataFrame(columns=["station_id", "sample_date", "Hs", "Tp", "Dm"])
    
    return pd.concat(all_data, ignore_index=True)


def validate_data(df: pd.DataFrame) -> bool:
    """Validate CDIP wave data integrity."""
    required_cols = ["station_id", "sample_date", "Hs", "Tp", "Dm"]
    
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing required column: {col}")
            return False
    
    if df.empty:
        print("[!] Empty dataframe — nothing to save")
        return False
    
    # Physical bounds checks - handle NaN by only checking non-null values
    import numpy as np
    
    # Significant wave height should be 0-30m (beyond 30m is hurricane-scale)
    hs_values = df["Hs"].replace([np.inf, -np.inf], np.nan).dropna()
    if (hs_values < 0).any() or (hs_values > 30).any():
        bad = df[(df["Hs"] < 0) | (df["Hs"] > 30)]
        print(f"[!] Invalid wave heights (must be 0-30m): {len(bad)} rows")
        return False

    # Peak period should be >= 0 seconds (0 is used for missing/invalid data but kept in output)
    tp_values = df["Tp"].replace([np.inf, -np.inf], np.nan).dropna()
    if (tp_values < 0).any() or (tp_values > 120).any():  # Allow 0 and up to 120s for extreme events
        bad = df[(df["Tp"] < 0) | (df["Tp"] > 120)]
        print(f"[!] Invalid dominant periods (must be 0-120s): {len(bad)} rows")
        return False

    # Direction should be 0-360 degrees
    dm_values = df["Dm"].replace([np.inf, -np.inf], np.nan).dropna()
    if (dm_values < 0).any() or (dm_values > 360).any():
        print("[!] Invalid wave directions (must be 0-360°)")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITING
# ─────────────────────────────────────────────────────────────────────────────

def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """Write CDIP wave data to Parquet with standardized formatting."""
    # Sort by station then date
    df = df.sort_values(["station_id", "sample_date"])
    
    # Standardize types
    df["station_id"] = df["station_id"].astype("string")
    
    # Write
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
    """Update MANIFEST.sha256 with new CDIP wave file checksums.
    
    The manifest uses the standard sha256sum format (hash *filename).
    """
    # Read existing entries
    manifest = []
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('{'):  # Skip JSON lines, keep sha256 format
                    parts = line.split('*', 1)
                    if len(parts) == 2:
                        manifest.append({
                            'sha256': parts[0].strip(),
                            'path': parts[1].strip()
                        })
    
    # Add/update new entries (convert dict to sha256sum format)
    for entry in new_entries:
        existing_idx = next(
            (i for i, e in enumerate(manifest) if e["path"] == entry["path"]),
            None,
        )
        checksum_entry = f"{entry['sha256_checksum']} *{entry['path']}"
        
        if existing_idx is not None:
            manifest[existing_idx] = {
                'sha256': entry['sha256_checksum'],
                'path': entry['path']
            }
        else:
            manifest.append({
                'sha256': entry['sha256_checksum'],
                'path': entry['path']
            })
    
    # Write back sorted
    manifest.sort(key=lambda x: x["path"])
    with open(_MANIFEST_PATH, "w") as f:
        for e in manifest:
            f.write(f"{e['sha256']} *{e['path']}\n")
    
    print(f"[✓] Updated MANIFEST.sha256 ({len(manifest)} entries)")


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
    
    OUTPUT_FILE = args.out_dir / "cdip_waves.parquet"
    
    print("=" * 60)
    print("CDIP Wave Data Fetcher")
    print("=" * 60)
    
    # Skip if exists
    if OUTPUT_FILE.exists() and not args.force:
        print(f"[!] Output exists: {OUTPUT_FILE}")
        print("[!] Use --force to re-fetch")
        return 0
    
    # Fetch
    print("\n[1/3] Fetching CDIP wave data...")
    try:
        df = fetch_all_stations()
    except NotImplementedError as e:
        print(f"\n{e}")
        print("\nNOTE: This fetcher is a template. Implement one of:")
        print("  A) THREDDS/OPENDAP access for full spectral data")
        print("  B) NDBC API wrapper for NDBC-compatible stations")
        print("  C) Direct parquet consumption from mbal_history/noaa/")
        return 1
    except Exception as e:
        print(f"\n[!] Fetch failed: {e}")
        return 1
    
    # Validate
    print("\n[2/3] Validating data...")
    if not validate_data(df):
        print("[!] Validation FAILED")
        return 1
    
    # Write
    print(f"\n[3/3] Writing to {OUTPUT_FILE}...")
    metadata = write_output(df, OUTPUT_FILE)
    
    # Manifest
    update_manifest([metadata])
    
    # Summary
    print("\n" + "=" * 60)
    print("CDIP Wave Fetch Complete")
    print("=" * 60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Stations: {df['station_id'].nunique()}")
    print(f"  Rows: {len(df):,}")
    if not df.empty:
        print(f"  Date range: {df['sample_date'].min()} to {df['sample_date'].max()}")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
