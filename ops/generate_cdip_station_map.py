#!/usr/bin/env python3
"""
Generate CDIP Station Map (Spatial Join)

Maps beach observation UUIDs to the nearest CDIP/NDBC wave buoy.
Uses Haversine distance for geographic proximity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, cos, sin, sqrt, atan2

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_PATH = PROJECT_ROOT / "reports" / "station_geo.parquet"
OUTPUT_PATH = PROJECT_ROOT / "bacteria_results" / "cdip_waves" / "station_map.parquet"

# CDIP/NDBC buoy positions (CA Coast)
BUOY_POSITIONS = {
    "46042": {"lat": 36.759, "lon": -122.469}, # Monterey Bay
    "46239": {"lat": 37.778, "lon": -122.839}, # San Francisco Bar
    "46013": {"lat": 38.238, "lon": -123.307}, # Bodega Bay
    "46026": {"lat": 37.759, "lon": -122.841}, # San Francisco
    "46237": {"lat": 37.525, "lon": -122.868}, # San Francisco South
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def main():
    print(f"[*] Loading station geo data from {GEO_PATH.name}...")
    if not GEO_PATH.exists():
        print(f"[!] Station geo file not found at {GEO_PATH}")
        return 1
    
    geo = pd.read_parquet(GEO_PATH)
    
    # Ensure columns exist
    if not all(c in geo.columns for c in ['station_id', 'latitude', 'longitude']):
        print(f"[!] Missing required columns in {GEO_PATH.name}: {geo.columns.tolist()}")
        return 1

    print(f"[*] Found {len(geo)} unique beach stations with GPS coordinates.")
    
    # Map to nearest buoy
    mapping = []
    for _, s in geo.iterrows():
        best_dist = float('inf')
        best_buoy = None
        
        for buoy_id, pos in BUOY_POSITIONS.items():
            dist = haversine(s['latitude'], s['longitude'], pos['lat'], pos['lon'])
            if dist < best_dist:
                best_dist = dist
                best_buoy = buoy_id
        
        mapping.append({
            'station_id': s['station_id'],
            'ndbc_id': best_buoy,
            'distance_km': round(best_dist, 2)
        })
    
    map_df = pd.DataFrame(mapping)
    
    print(f"[*] Mapping complete. Average distance to buoy: {map_df['distance_km'].mean():.2f} km")
    print(f"[*] Saving map to {OUTPUT_PATH}...")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    map_df.to_parquet(OUTPUT_PATH, index=False)
    
    print("[✓] Spatial join successful.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
