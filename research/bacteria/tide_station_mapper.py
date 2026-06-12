#!/usr/bin/env python3
"""Map CA CO-OPS tide stations to beach station UUIDs via geographic proximity.

This module:
  1. Loads CA CO-OPS tide station coordinates from the fetched parquet
  2. Loads beach station coordinates from station_geo.parquet  
  3. For each beach, finds the nearest CO-OPS tide station (haversine distance)
  4. Outputs a mapping DataFrame: beach_station_uuid -> nearest_tide_station_id

Output:
  - bacteria_results/tide_stations/tide_station_mapping.parquet
    Columns: beach_station_uuid, latitude, longitude,
             tide_station_id, tide_latitude, tide_longitude, distance_km
    
This mapping is used by `operational_benchmark.py` to join tide data onto
beach observations.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_OUTPUT_DIR = _PROJECT_ROOT / "bacteria_results" / "tide_stations"
_MANIFEST_PATH = _PROJECT_ROOT / "research" / "bacteria" / "reproduce" / "MANIFEST.sha256"


_EARTH_KM = 6371.0


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in km."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return _EARTH_KM * c


def load_tide_stations() -> pd.DataFrame:
    """Load CA CO-OPS tide station data from parquet."""
    path = _PROJECT_ROOT / "bacteria_results" / "tide_stages" / "tide_stages.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Tide stages not found: {path}\n"
            "Run fetch_tide_stages.py first."
        )

    df = pd.read_parquet(path)

    # Get unique stations with their coordinates
    stations = (
        df.groupby("station_id")
        .agg({"sample_date": ["min", "max"], "water_level_m": "count"})
        .reset_index()
    )
    stations.columns = ["station_id", "start_date", "end_date", "n_observations"]

    # We need coordinates - fetch from NOAA API for each station
    try:
        import requests

        coords = []
        for sid in df["station_id"].unique():
            url = "https://tidesandcurrents.noaa.gov/api/datagetter"
            params = {
                "product": "water_level",
                "application": "test",
                "units": "metric",
                "time_zone": "GMT",
                "format": "json",
                "station": sid,
                "begin_date": "20250101",
                "end_date": "20250102",
                "datum": "MLLW",
            }

            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    meta = data.get("metadata", {})
                    lat = meta.get("lat")
                    lon = meta.get("lon")

                    if lat and lon:
                        coords.append({
                            "station_id": sid,
                            "tide_latitude": float(lat),
                            "tide_longitude": float(lon),
                        })
                        print(f"  {sid}: ({lat}, {lon})")
                else:
                    print(f"  {sid}: HTTP {r.status_code} (no coordinates)")
            except Exception as e:
                print(f"  {sid}: API error - {e}")

        if coords:
            coords_df = pd.DataFrame(coords)
            stations = stations.merge(coords_df, on="station_id")
        else:
            raise RuntimeError("Could not fetch coordinates from any station")

    except ImportError:
        raise ImportError(
            "requests not installed. Install with: pip install requests"
        )

    return stations


def load_beach_stations() -> pd.DataFrame:
    """Load beach station geo data."""
    path = _PROJECT_ROOT / "reports" / "station_geo.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Beach stations not found: {path}")

    df = pd.read_parquet(path)

    return df[["station_id", "latitude", "longitude"]].rename(
        columns={"station_id": "beach_station_uuid"}
    )


def map_to_nearest_tide_station(
    beach_df: pd.DataFrame, tide_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map each beach station to its nearest CO-OPS tide station.
    
    Args:
        beach_df: Beach stations with columns [beach_station_uuid, latitude, longitude]
        tide_df: Tide stations with columns [station_id, tide_latitude, tide_longitude]
        
    Returns:
        Mapping DataFrame with columns:
          beach_station_uuid, latitude, longitude,
          tide_station_id, tide_latitude, tide_longitude, distance_km
    """
    mappings = []

    for _, beach in beach_df.iterrows():
        if pd.isna(beach["latitude"]) or pd.isna(beach["longitude"]):
            continue

        best_dist = float("inf")
        best_tide_sid = None
        best_tide_lat = None
        best_tide_lon = None

        for _, tide in tide_df.iterrows():
            if pd.isna(tide["tide_latitude"]) or pd.isna(tide["tide_longitude"]):
                continue

            dist = haversine_distance_km(
                beach["latitude"], beach["longitude"],
                tide["tide_latitude"], tide["tide_longitude"]
            )

            if dist < best_dist:
                best_dist = dist
                best_tide_sid = tide["station_id"]
                best_tide_lat = tide["tide_latitude"]
                best_tide_lon = tide["tide_longitude"]

        if best_tide_sid is not None:
            mappings.append({
                "beach_station_uuid": beach["beach_station_uuid"],
                "latitude": beach["latitude"],
                "longitude": beach["longitude"],
                "tide_station_id": best_tide_sid,
                "tide_latitude": best_tide_lat,
                "tide_longitude": best_tide_lon,
                "distance_km": round(best_dist, 3),
            })

    return pd.DataFrame(mappings)


def validate_mapping(df: pd.DataFrame) -> bool:
    """Validate tide station mapping integrity."""
    required_cols = [
        "beach_station_uuid", "latitude", "longitude",
        "tide_station_id", "tide_latitude", "tide_longitude", "distance_km"
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing column: {col}")
            return False

    if df.empty:
        print("[!] Empty mapping — nothing to save")
        return False

    # Check all beaches have a mapping
    beach_count = df["beach_station_uuid"].nunique()
    tide_count = df["tide_station_id"].nunique()

    print(f"  Beaches mapped: {beach_count}")
    print(f"  Tide stations used: {tide_count}")

    # Check distance bounds (should be reasonable)
    if (df["distance_km"] < 0).any():
        print("[!] Negative distances found")
        return False

    # Report max/max distances
    print(f"  Distance range: {df['distance_km'].min():.2f} - {df['distance_km'].max():.2f} km")

    return True


def write_output(df: pd.DataFrame, out_path: Path) -> dict:
    """Write mapping to Parquet."""
    df = df.sort_values(["beach_station_uuid", "tide_station_id"])

    # Standardize types
    if "beach_station_uuid" in df.columns:
        df["beach_station_uuid"] = df["beach_station_uuid"].astype("string")
    if "tide_station_id" in df.columns:
        df["tide_station_id"] = df["tide_station_id"].astype("string")

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
    """Update MANIFEST.sha256."""
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

    print(f"[✓] Updated MANIFEST.sha256 ({len(manifest)} entries)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-map even if output exists"
    )
    args = parser.parse_args()

    OUTPUT_FILE = _OUTPUT_DIR / "tide_station_mapping.parquet"

    print("=" * 60)
    print("CA CO-OPS Tide Station to Beach UUID Mapper")
    print("=" * 60)

    # Skip if exists
    if OUTPUT_FILE.exists() and not args.force:
        print(f"[!] Output exists: {OUTPUT_FILE}")
        print("[!] Use --force to re-map")
        return 0

    # Load sources
    print("\n[1/3] Loading CA CO-OPS tide stations...")
    try:
        tide_df = load_tide_stations()
        print(f"  Loaded {len(tide_df)} tide stations with coordinates")
    except Exception as e:
        print(f"[!] Failed to load tide stations: {e}")
        return 1

    print("\n[2/3] Loading beach station geo...")
    try:
        beach_df = load_beach_stations()
        print(f"  Loaded {len(beach_df):,} beach stations")
    except Exception as e:
        print(f"[!] Failed to load beach stations: {e}")
        return 1

    # Map
    print("\n[3/3] Mapping beaches to nearest tide stations...")
    mapping = map_to_nearest_tide_station(beach_df, tide_df)
    print(f"  Mapped {len(mapping):,} beach-tide pairs")

    # Validate
    if not validate_mapping(mapping):
        print("[!] Validation FAILED")
        return 1

    # Write
    print(f"\nWriting to {OUTPUT_FILE}...")
    metadata = write_output(mapping, OUTPUT_FILE)

    # Manifest
    update_manifest([metadata])

    # Summary
    print("\n" + "=" * 60)
    print("Tide Station Mapping Complete")
    print("=" * 60)
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Beaches mapped: {mapping['beach_station_uuid'].nunique()}")
    print(f"  Tide stations used: {mapping['tide_station_id'].nunique()}")
    print(
        f"  Max distance: {mapping['distance_km'].max():.2f} km "
        f"(San Francisco: {mapping[mapping['tide_station_id']=='9410230']['distance_km'].max():.2f} km)"
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
