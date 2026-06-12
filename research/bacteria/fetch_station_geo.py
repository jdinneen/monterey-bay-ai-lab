#!/usr/bin/env python3
"""Fetch station geographic coordinates (lat/lon) for statewide beach monitoring.

This fetcher pulls station-level geographic coordinates from the CA BeachWatch BigQuery
table to enable geophysical driver joins (upwelling bands, SST gradients, etc.).

Output (bacteria_results/, gitignored data):
  - station_geo.parquet: station_id, latitude, longitude, beach_name, county

Required by:
  - Physical spatial covariates (distance to gauges, monitoring density)
  - Geophysical driver joins (upwelling, SST, tide stage)
  - Spatial generalization tests (leave-one-county-out)

Note: This is a READ-ONLY public-record pull via BigQuery CLI or python client.
No API keys required.

Usage:
    python research/bacteria/fetch_station_geo.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Allow local imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force MBAL_GCP_PROJECT to be set (no default - auth must be configured)
import os

os.environ.setdefault("MBAL_GCP_PROJECT", "")

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

import pandas as pd


_OUTPUT_DIR = _PROJECT_ROOT / "reports"
_MANIFEST_PATH = _PROJECT_ROOT / "research" / "bacteria" / "reproduce" / "MANIFEST.sha256"

# BigQuery table coordinates are in the same table as beach observations
_PROJECT = os.environ.get("MBAL_GCP_PROJECT", "")
_DS = "blue_current_core_v2"
_OBS_TABLE = f"`{_PROJECT}.{_DS}.california_beach_sample_observations`"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file (streaming, handles large files)."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _bq_query_client(sql: str) -> pd.DataFrame:
    """Primary path: google-cloud-bigquery python client."""
    if bigquery is None:
        raise ImportError(
            "google-cloud-bigquery not installed. "
            "Install with: pip install google-cloud-bigquery"
        )
    
    project_id = os.environ.get("MBAL_GCP_PROJECT")
    if not project_id:
        raise RuntimeError(
            "MBAL_GCP_PROJECT environment variable not set.\n"
            "Set it to your GCP project ID (e.g., 'my-project-123456')"
        )
    
    client = bigquery.Client(project=project_id)
    rows = client.query(" ".join(sql.split())).result()
    try:
        return rows.to_dataframe(create_bqstorage_client=True)
    except Exception:
        return rows.to_dataframe(create_bqstorage_client=False)


def _bq_query_cli(sql: str, max_rows: int = 100_000) -> pd.DataFrame:
    """Fallback path: shell out to the `bq` CLI."""
    import subprocess
    import shutil
    import io
    
    bq = shutil.which("bq.cmd") or shutil.which("bq")
    if not bq:
        raise RuntimeError("bq CLI not found (need authenticated Google Cloud SDK).")
    
    cmd = [
        bq, "query",
        "--use_legacy_sql=false",
        "--format=csv",
        f"--max_rows={max_rows}",
        " ".join(sql.split())
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    
    return pd.read_csv(io.StringIO(r.stdout))


def bq_query(sql: str, max_rows: int = 100_000) -> pd.DataFrame:
    """Run a read-only query. Prefer python client; fall back to CLI."""
    try:
        return _bq_query_client(sql)
    except ImportError:
        return _bq_query_cli(sql, max_rows=max_rows)
    except Exception:
        # Auth/runtime failure in the client path: retry via the CLI
        return _bq_query_cli(sql, max_rows=max_rows)


def fetch_station_geo() -> pd.DataFrame:
    """
    Fetch station geographic coordinates from BigQuery.
    
    Returns:
        DataFrame with columns:
            - station_id: unique beach monitoring station identifier
            - latitude: decimal degrees (WGS84)
            - longitude: decimal degrees (WGS84)  
            - beach_name: human-readable beach name
            - county: county name
    
    Raises:
        RuntimeError: If query fails or no data returned
    """
    # Strategy: Try BigQuery first, fall back to existing parquet if needed
    project = os.environ.get("MBAL_GCP_PROJECT", "")
    
    if not project:
        # No GCP project set - try using existing parquet file
        fallback_path = _PROJECT_ROOT / "reports" / "station_geo.parquet"
        print(f"[!] MBAL_GCP_PROJECT not set, trying fallback: {fallback_path.name}")
        
        if fallback_path.exists():
            df = pd.read_parquet(fallback_path)
            print(f"  Loaded {len(df):,} rows from existing file")
            return _standardize_geo(df)
        else:
            raise RuntimeError(
                "MBAL_GCP_PROJECT not set and no existing station_geo.parquet found.\n"
                "Either:\n"
                "  1. Set MBAL_GCP_PROJECT environment variable, OR\n"
                "  2. Run fetch_statewide_beachwatch.py first to populate the parquet"
            )
    
    # Query BigQuery
    sql = f"""
        SELECT DISTINCT 
            station_id,
            latitude,
            longitude,
            beach_name,
            county
        FROM {_OBS_TABLE}
        WHERE station_id IS NOT NULL
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
        ORDER BY county, beach_name, station_id
    """
    
    df = bq_query(sql, max_rows=200_000)
    
    if df.empty:
        raise RuntimeError("Station geo query returned empty dataframe")
    
    return _standardize_geo(df)


def _standardize_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize geo DataFrame column types and validate."""
    # Standardize types
    for c in ["station_id", "beach_name", "county"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    
    # Validate geographic bounds
    if (df["latitude"] < -90).any() or (df["latitude"] > 90).any():
        bad = df[(df["latitude"] < -90) | (df["latitude"] > 90)]
        raise ValueError(f"Invalid latitude values: {len(bad)} rows out of bounds")
    
    if (df["longitude"] < -180).any() or (df["longitude"] > 180).any():
        bad = df[(df["longitude"] < -180) | (df["longitude"] > 180)]
        raise ValueError(f"Invalid longitude values: {len(bad)} rows out of bounds")
    
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["station_id"])
    if len(df) < before:
        print(f"[!] Dropped {before - len(df)} duplicate station_ids")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate station geo data integrity."""
    required_cols = ["station_id", "latitude", "longitude"]
    
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing required column: {col}")
            return False
    
    if df.empty:
        print("[!] Empty dataframe — nothing to save")
        return False
    
    # Check geographic coverage (California coast roughly)
    lat_min, lat_max = 32.0, 42.5  # CA coast bounds
    lon_min, lon_max = -125.0, -117.0
    
    if (df["latitude"] < lat_min).any() or (df["latitude"] > lat_max).any():
        print(f"[!] Latitude out of CA coast range ({lat_min}-{lat_max})")
        return False
    
    if (df["longitude"] < lon_min).any() or (df["longitude"] > lon_max).any():
        print(f"[!] Longitude out of CA coast range ({lon_min}-{lon_max})")
        return False
    
    # Check for NaN in critical fields
    if df[["latitude", "longitude"]].isna().any().any():
        bad_rows = df[["latitude", "longitude"]].isna().any(axis=1).sum()
        print(f"[!] {bad_rows} rows with NaN latitude/longitude")
        return False
    
    # Check reasonable number of stations
    if len(df) < 10:
        print(f"[!] Unusually few stations: {len(df)} (expected >200)")
        return False
    
    return True


def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """Write station geo data to Parquet and compute checksums."""
    # Sort by station_id (deterministic ordering)
    df = df.sort_values("station_id")
    
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
    """Update MANIFEST.sha256 with new file checksums."""
    # Read existing entries in sha256sum format
    manifest = []
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("{"):  # Skip JSON lines, keep sha256 format
                    parts = line.split("*", 1)
                    if len(parts) == 2:
                        manifest.append({
                            "sha256": parts[0].strip(),
                            "path": parts[1].strip()
                        })
    
    # Add/update new entries
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
    
    # Write back sorted by path
    manifest.sort(key=lambda x: x["path"])
    with open(_MANIFEST_PATH, "w") as f:
        for e in manifest:
            f.write(f"{e['sha256']} *{e['path']}\n")
    
    print(f"[✓] Updated MANIFEST.sha256 ({len(manifest)} entries)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if output exists"
    )
    args = parser.parse_args()
    
    OUTPUT_FILE = _OUTPUT_DIR / "station_geo.parquet"
    
    print("=" * 60)
    print("Station Geographic Coordinates Fetcher")
    print("=" * 60)
    
    # Check if exists (skip GCP check - can use existing file)
    if OUTPUT_FILE.exists() and not args.force:
        print(f"[!] Output exists: {OUTPUT_FILE}")
        print("[!] Use --force to re-fetch")
        return 0

    # Run fetch
    try:
        print("\n[1/2] Fetching station geo from BigQuery or fallback...")
        df = fetch_station_geo()
    except Exception as e:
        print(f"\n[!] Fetch failed: {type(e).__name__}: {e}")
        if "bigquery" in str(e).lower() or "client" in str(e).lower():
            print("\n    HINT: Install with: pip install google-cloud-bigquery")
        return 1
    
    # Validate
    print("\n[2/2] Validating data integrity...")
    if not validate_data(df):
        print("[!] Validation FAILED — aborting")
        return 1

    # Write output
    print(f"\nWriting to {OUTPUT_FILE}...")
    metadata = write_output(df, OUTPUT_FILE)
    
    # Update manifest
    update_manifest([metadata])
    
    # Summary
    print("\n" + "=" * 60)
    print("Station Geo Fetch Complete")
    print("=" * 60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Stations: {len(df):,}")
    print(f"  County coverage: {df['county'].nunique()} counties")
    print(f"  Lat range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"  Lon range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
