#!/usr/bin/env python3
"""Ingest USGS daily river discharge to add a first-flush driver to the benchmark.

First flush -- the runoff spike after a dry period that flushes land-based bacteria to
the surf zone -- is the physical mechanism behind beach contamination, and AB411's
rain rule is a crude proxy for it. California is densely gauged (2,400+ daily-discharge
gauges; 99% of beaches within 20km of one), so this maps each beach station to its
nearest gauge and pulls daily discharge (USGS NWIS dv, parameter 00060), enabling
first-flush features in operational_benchmark.

Outputs (under bacteria_results/discharge/):
- discharge_gauge.parquet : gauge_id, date, discharge_cfs
- station_gauge_map.parquet : station_id, gauge_id, distance_km

Resumable (skips cached gauges); batched dv requests; read-only public data, no key.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SITE_URL = ("https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=ca"
            "&parameterCd=00060&siteType=ST&hasDataTypeCd=dv&siteStatus=all")
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"


def haversine_km(a, b, c, d):
    R = 6371.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def get_gauges() -> pd.DataFrame:
    with urllib.request.urlopen(SITE_URL, timeout=90) as r:
        txt = r.read().decode("utf-8", "replace")
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    hdr = lines[0].split("\t")
    g = pd.DataFrame([l.split("\t") for l in lines[2:]], columns=hdr)
    g["lat"] = pd.to_numeric(g["dec_lat_va"], errors="coerce")
    g["lon"] = pd.to_numeric(g["dec_long_va"], errors="coerce")
    return g.dropna(subset=["lat", "lon"])[["site_no", "lat", "lon"]].reset_index(drop=True)


def station_gauge_map(geo_path: Path, obs_path: Path, gauges: pd.DataFrame, max_km: float) -> pd.DataFrame:
    geo = pd.read_parquet(geo_path).dropna(subset=["latitude", "longitude"])
    obs_ids = set(pd.read_parquet(obs_path, columns=["station_id"])["station_id"].unique())
    geo = geo[geo["station_id"].isin(obs_ids)]
    gl = gauges[["lat", "lon"]].to_numpy()
    gid = gauges["site_no"].to_numpy()
    rows = []
    for s in geo.itertuples():
        d = np.array([haversine_km(s.latitude, s.longitude, gl[i, 0], gl[i, 1]) for i in range(len(gl))])
        j = int(d.argmin())
        if d[j] <= max_km:
            rows.append({"station_id": s.station_id, "gauge_id": gid[j], "distance_km": round(float(d[j]), 2)})
    return pd.DataFrame(rows)


def fetch_dv_batch(site_ids: list[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    url = (f"{DV_URL}?format=json&sites={','.join(site_ids)}&parameterCd=00060"
           f"&startDT={start}&endDT={end}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            frames = []
            for ts in data.get("value", {}).get("timeSeries", []):
                gid = ts.get("sourceInfo", {}).get("siteCode", [{}])[0].get("value")
                vals = ts.get("values", [{}])[0].get("value", [])
                if not gid or not vals:
                    continue
                v = pd.DataFrame(vals)
                v["discharge_cfs"] = pd.to_numeric(v["value"], errors="coerce")
                v = v[v["discharge_cfs"] > -999990]  # drop NoData sentinels
                frames.append(pd.DataFrame({
                    "gauge_id": gid, "date": pd.to_datetime(v["dateTime"]).dt.tz_localize(None).dt.normalize(),
                    "discharge_cfs": v["discharge_cfs"].to_numpy(),
                }))
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["gauge_id", "date", "discharge_cfs"])
        except Exception as exc:  # noqa: BLE001 - transient API/network
            last = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"[fetch_discharge] WARN batch failed ({len(site_ids)} sites): {last}", file=sys.stderr)
    return pd.DataFrame(columns=["gauge_id", "date", "discharge_cfs"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geo", default="reports/station_geo.parquet")
    ap.add_argument("--obs", default="bacteria_results/statewide/statewide_beach_observations.parquet")
    ap.add_argument("--out-dir", default="bacteria_results/discharge")
    ap.add_argument("--max-km", type=float, default=20.0, help="cap station->gauge distance")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--batch", type=int, default=40, help="gauges per dv request")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "discharge_gauge.parquet"
    map_path = out_dir / "station_gauge_map.parquet"

    gauges = get_gauges()
    print(f"[fetch_discharge] CA daily-discharge gauges: {len(gauges)}")
    smap = station_gauge_map(Path(args.geo), Path(args.obs), gauges, args.max_km)
    smap.to_parquet(map_path, index=False)
    need = sorted(smap["gauge_id"].unique())
    print(f"[fetch_discharge] stations mapped: {len(smap)} | unique gauges to fetch: {len(need)}")

    cached = pd.read_parquet(grid_path) if grid_path.exists() else pd.DataFrame(
        columns=["gauge_id", "date", "discharge_cfs"])
    done = set(cached["gauge_id"].unique()) if len(cached) else set()
    todo = [g for g in need if g not in done]
    print(f"[fetch_discharge] cached gauges={len(done)} to_fetch={len(todo)}")

    frames = [cached] if len(cached) else []
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        frames.append(fetch_dv_batch(batch, args.start, args.end))
        pd.concat(frames, ignore_index=True).to_parquet(grid_path, index=False)  # checkpoint
        print(f"[fetch_discharge] {min(i + args.batch, len(todo))}/{len(todo)} gauges "
              f"(rows ~{sum(len(f) for f in frames):,})")
        time.sleep(args.delay)

    out = pd.concat(frames, ignore_index=True) if frames else cached
    out = out.dropna(subset=["date"]).drop_duplicates(["gauge_id", "date"])
    out.to_parquet(grid_path, index=False)
    span = (out["date"].min(), out["date"].max()) if len(out) else (None, None)
    print(f"[fetch_discharge] DONE rows={len(out):,} gauges={out['gauge_id'].nunique()} span {span[0]}..{span[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
