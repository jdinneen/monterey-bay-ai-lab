#!/usr/bin/env python3
"""Ingest daily rainfall for CA beach stations to enable the AB411 rain-rule baseline.

The statewide obs table has no rainfall; the only existing rain was a single Lovers
Point ASOS gauge. This fetches gridded daily precipitation from the free Open-Meteo
historical archive (no API key) for the ~110 unique 0.1-degree cells that the 800
geolocated beach stations collapse to, caching one Parquet so it is a one-time cost.

Outputs (under bacteria_results/rainfall/):
- rainfall_grid.parquet : grid_lat, grid_lon, date, precip_mm
- station_grid_map.parquet : station_id, grid_lat, grid_lon

Resumable: re-running skips cells already in the cache. Polite: a short delay and
bounded retries between requests. Read-only over a public dataset; no credentials.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def grid_round(series: pd.Series, grid: float) -> pd.Series:
    return (series / grid).round() * grid


def station_grid_map(geo_path: Path, obs_path: Path, grid: float) -> pd.DataFrame:
    geo = pd.read_parquet(geo_path)
    obs_ids = set(pd.read_parquet(obs_path, columns=["station_id"])["station_id"].unique())
    g = geo[geo["station_id"].isin(obs_ids)].dropna(subset=["latitude", "longitude"]).copy()
    g["grid_lat"] = grid_round(g["latitude"], grid).round(4)
    g["grid_lon"] = grid_round(g["longitude"], grid).round(4)
    return g[["station_id", "grid_lat", "grid_lon"]].reset_index(drop=True)


def fetch_cell(lat: float, lon: float, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    url = (f"{ARCHIVE_URL}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily=precipitation_sum&timezone=America%2FLos_Angeles")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            daily = data.get("daily", {})
            return pd.DataFrame({
                "grid_lat": lat, "grid_lon": lon,
                "date": pd.to_datetime(daily.get("time", [])),
                "precip_mm": daily.get("precipitation_sum", []),
            })
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))  # backoff (handles transient + 429)
    print(f"[fetch_rainfall] WARN cell ({lat},{lon}) failed after {retries} tries: {last}", file=sys.stderr)
    return pd.DataFrame(columns=["grid_lat", "grid_lon", "date", "precip_mm"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geo", default="reports/station_geo.parquet")
    ap.add_argument("--obs", default="bacteria_results/statewide/statewide_beach_observations.parquet")
    ap.add_argument("--out-dir", default="bacteria_results/rainfall")
    ap.add_argument("--grid", type=float, default=0.1, help="cell size in degrees (~11km at 0.1)")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--delay", type=float, default=0.3, help="politeness delay between requests (s)")
    ap.add_argument("--max-cells", type=int, default=None, help="cap cells (for a quick smoke)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "rainfall_grid.parquet"
    map_path = out_dir / "station_grid_map.parquet"

    smap = station_grid_map(Path(args.geo), Path(args.obs), args.grid)
    smap.to_parquet(map_path, index=False)
    cells = smap[["grid_lat", "grid_lon"]].drop_duplicates().reset_index(drop=True)
    if args.max_cells:
        cells = cells.head(args.max_cells)

    cached = pd.read_parquet(grid_path) if grid_path.exists() else pd.DataFrame(
        columns=["grid_lat", "grid_lon", "date", "precip_mm"])
    done = set(map(tuple, cached[["grid_lat", "grid_lon"]].drop_duplicates().to_numpy())) if len(cached) else set()
    todo = [(r.grid_lat, r.grid_lon) for r in cells.itertuples() if (r.grid_lat, r.grid_lon) not in done]
    print(f"[fetch_rainfall] cells total={len(cells)} cached={len(done)} to_fetch={len(todo)}")

    frames = [cached] if len(cached) else []
    for i, (lat, lon) in enumerate(todo, 1):
        df = fetch_cell(lat, lon, args.start, args.end)
        frames.append(df)
        if i % 10 == 0 or i == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(grid_path, index=False)  # checkpoint
            print(f"[fetch_rainfall] {i}/{len(todo)} cells (rows so far ~{sum(len(f) for f in frames):,})")
        time.sleep(args.delay)

    out = pd.concat(frames, ignore_index=True) if frames else cached
    out = out.dropna(subset=["date"]).drop_duplicates(["grid_lat", "grid_lon", "date"])
    out.to_parquet(grid_path, index=False)
    print(f"[fetch_rainfall] DONE grid rows={len(out):,} cells={out[['grid_lat','grid_lon']].drop_duplicates().shape[0]} "
          f"date span {out['date'].min().date()}..{out['date'].max().date()}")
    print(f"[fetch_rainfall] wrote {grid_path} and {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
