"""NCEI Integrated Surface Database (ISD) — observed hourly surface met, California.

Source: ``https://www.ncei.noaa.gov/access/services/data/v1`` dataset ``global-hourly``
(open, keyless). Station-observed hourly air temperature, wind (dir/speed), sea-level
pressure, and dewpoint at California surface stations. Distinct from daily `ghcnd`
(station daily) and from `open_meteo_archive`/`nasa_power`/`gridmet` (model reanalysis):
this is *observed sub-daily ground truth*. Station list is discovered from the ISD
history inventory; chunked one (station, year) per resumable chunk.

Lane B (claude-fetch-2).
"""
from __future__ import annotations

import csv
import io
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from ..core import Adapter, get_with_backoff

_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"
_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def _ca_isd_stations(limit: int = 50) -> List[Tuple[str, float, float]]:
    """Long-record California ISD stations as (11-digit id, lat, lon)."""
    fallback = [
        ("72494023234", 37.62, -122.40), ("72295023174", 33.94, -118.41),
        ("72290023188", 32.73, -117.18), ("72389093193", 36.78, -119.72),
        ("72483023232", 38.51, -121.49), ("72594024257", 40.98, -124.11),
        ("72484523237", 38.70, -121.59), ("72286903171", 34.06, -117.65),
    ]
    try:
        resp = get_with_backoff(_HISTORY, timeout=120, headers={"User-Agent": "mbal-datafetch/0.1"})
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            if r.get("CTRY") == "US" and r.get("STATE") == "CA" and r.get("USAF") not in ("999999",):
                end = (r.get("END") or "0")
                begin = (r.get("BEGIN") or "99999999")
                try:
                    lat, lon = float(r["LAT"]), float(r["LON"])
                except (ValueError, KeyError):
                    continue
                if end >= "20230000" and begin <= "20050000":  # long, current record
                    rows.append((f"{r['USAF']}{r['WBAN']}", lat, lon, begin))
        rows.sort(key=lambda x: x[3])  # earliest-starting first (longest records)
        out = [(sid, lat, lon) for sid, lat, lon, _ in rows[:limit]]
        return out or fallback
    except Exception:
        return fallback


def _coded(val, scale=10.0, miss=("+9999", "9999", "99999", "+99999")):
    """Parse an NCEI comma-coded field's first numeric token -> scaled float/NaN."""
    if not val:
        return None
    tok = str(val).split(",")[0].strip()
    if tok in miss or tok.lstrip("+-") in ("9999", "99999"):
        return None
    try:
        return float(tok) / scale
    except ValueError:
        return None


def _clean_physical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bounds = {
        "air_temp_c": (-40.0, 55.0),
        "slp_hpa": (850.0, 1100.0),
        "wind_speed_ms": (0.0, 120.0),
    }
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce")
            out[col] = v.where(v.between(lo, hi))
    return out


class NceiIsdHourlyAdapter(Adapter):
    START_YEAR = 2005
    END_YEAR = 2024
    STATION_LIMIT = 50

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.START_YEAR
        y1 = int(end[:4]) if end else self.END_YEAR
        for sid, lat, lon in _ca_isd_stations(self.STATION_LIMIT):
            for year in range(y0, y1 + 1):
                yield {"key": f"{sid}_{year}", "station": sid, "lat": lat, "lon": lon, "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        sid, year = chunk["station"], chunk["year"]
        url = (f"{_DATA}?dataset=global-hourly&stations={sid}"
               f"&startDate={year}-01-01&endDate={year}-12-31"
               f"&dataTypes=TMP,WND,SLP,DEW&format=json")
        try:
            resp = get_with_backoff(url, timeout=180, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            recs = resp.json()
        except Exception:
            return pd.DataFrame()
        if not isinstance(recs, list) or not recs:
            return pd.DataFrame()
        out = []
        for r in recs:
            wnd = str(r.get("WND", "")).split(",")
            wind_dir = None
            try:
                if wnd and wnd[0] not in ("999", ""):
                    wind_dir = float(wnd[0])
            except ValueError:
                pass
            wind_spd = _coded(",".join(wnd[3:5]) if len(wnd) >= 5 else "", scale=10.0)
            out.append({
                "station_id": sid,
                "time": r.get("DATE"),
                "air_temp_c": _coded(r.get("TMP")),
                "dewpoint_c": _coded(r.get("DEW")),
                "slp_hpa": _coded(r.get("SLP")),
                "wind_dir_deg": wind_dir,
                "wind_speed_ms": wind_spd,
                "latitude": chunk["lat"],
                "longitude": chunk["lon"],
            })
        df = pd.DataFrame(out)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df[df["time"].notna()]
        # keep rows with at least one real observation
        obs = ["air_temp_c", "dewpoint_c", "slp_hpa", "wind_speed_ms"]
        df = _clean_physical(df)
        df = df[df[obs].notna().any(axis=1)]
        return df.reset_index(drop=True)

    def _consolidate(self):
        """Consolidate then re-apply physical cleaning to repair legacy raw chunks."""
        from ..core import _to_parquet_safe

        path = super()._consolidate()
        df = pd.read_parquet(path)
        before = df.copy()
        df = _clean_physical(df)
        if not df.equals(before):
            _to_parquet_safe(df, path)
        return path
