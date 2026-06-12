"""NOAA WCOFS circulation adapter.

WCOFS is the operational West Coast ROMS-based forecast/nowcast source that
continues past the discontinued SCCOOS `roms_ncst` archive. This adapter uses
CeNCOOS THREDDS OPeNDAP aggregation so we can request a bounded Monterey Bay
surface grid subset instead of downloading native NetCDF files.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..core import Adapter, _guard_write, _to_parquet_safe

WCOFS_OPENDAP = "https://thredds.cencoos.org/thredds/dodsC/AWS_WCOFS.nc"
LAT_MIN = 36.0
LAT_MAX = 38.0
LON_MIN = -124.4
LON_MAX = -121.0
GRID_STRIDE = 10
TIME_STRIDE = 8  # WCOFS ocean_time is 3-hourly; 8 steps gives daily driver samples.


class WcofsCirculationAdapter(Adapter):
    """Fetch bounded WCOFS surface circulation fields from OPeNDAP."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        cov_start, cov_end = _dataset_time_coverage()
        s_date = dt.date.fromisoformat(start[:10]) if start else cov_end.date().replace(day=1)
        e_date = dt.date.fromisoformat(end[:10]) if end else cov_end.date()
        s_date = max(s_date, cov_start.date())
        e_date = min(e_date, cov_end.date())
        if s_date > e_date:
            return

        curr = s_date.replace(day=1)
        last_month = e_date.replace(day=1)
        while curr <= last_month:
            month_end = _month_end(curr)
            c0 = max(dt.datetime.combine(curr, dt.time.min, tzinfo=dt.timezone.utc), cov_start)
            c1 = min(dt.datetime.combine(month_end, dt.time.max, tzinfo=dt.timezone.utc), cov_end)
            if c0.date() <= e_date and c1.date() >= s_date:
                c0 = max(c0, dt.datetime.combine(s_date, dt.time.min, tzinfo=dt.timezone.utc))
                c1 = min(c1, dt.datetime.combine(e_date, dt.time.max, tzinfo=dt.timezone.utc))
                yield {
                    "key": curr.strftime("%Y-%m"),
                    "start": c0.isoformat().replace("+00:00", "Z"),
                    "end": c1.isoformat().replace("+00:00", "Z"),
                }
            curr = _next_month(curr)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        import xarray as xr

        with xr.open_dataset(WCOFS_OPENDAP, decode_times=True) as ds:
            eta_slice, xi_slice = _bbox_slices(ds["lat_rho"], ds["lon_rho"])
            start = _to_xarray_time(chunk["start"])
            end = _to_xarray_time(chunk["end"])
            sub = ds[["zeta", "temp", "salt", "urot", "vrot"]].sel(
                ocean_time=slice(start, end)
            ).isel(
                ocean_time=slice(None, None, TIME_STRIDE)
            ).isel(
                s_rho=-1,
                eta_rho=eta_slice,
                xi_rho=xi_slice,
            ).load()
        return _normalize_wcofs_frame(sub.to_dataframe().reset_index(), chunk)

    def _consolidate(self):
        frames = []
        required = set(self.spec.required_columns)
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[wcofs_circulation] skipping unreadable chunk {p.name}: {exc}")
                continue
            if len(d) and required.issubset(d.columns):
                frames.append(d)
        out_path = self.curated_path
        _guard_write(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            _to_parquet_safe(pd.DataFrame(columns=self.spec.required_columns), out_path)
            return out_path
        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        numeric_cols = [
            "latitude",
            "longitude",
            "surface_sigma",
            "u",
            "v",
            "temp",
            "salt",
            "sea_surface_height_m",
            "current_speed_m_s",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _clean_physical_bounds(df)
        df = df.dropna(subset=["time", "latitude", "longitude"])
        df = df.dropna(subset=["u", "v", "temp", "salt", "sea_surface_height_m"], how="all")
        keys = [k for k in self.spec.dedup_keys if k in df.columns]
        if keys:
            df = df.drop_duplicates(keys)
        df = df.sort_values(["time", "latitude", "longitude"])
        _to_parquet_safe(df, out_path)
        return out_path


def _dataset_time_coverage() -> tuple[dt.datetime, dt.datetime]:
    import xarray as xr

    with xr.open_dataset(WCOFS_OPENDAP, decode_times=True) as ds:
        times = pd.to_datetime(ds["ocean_time"].values)
    if len(times) == 0:
        raise RuntimeError("AWS_WCOFS has no ocean_time values")
    return (
        times.min().to_pydatetime().replace(tzinfo=dt.timezone.utc),
        times.max().to_pydatetime().replace(tzinfo=dt.timezone.utc),
    )


def _bbox_slices(lat, lon) -> tuple[slice, slice]:
    lat_v = lat.load().values
    lon_v = lon.load().values
    mask = (lat_v >= LAT_MIN) & (lat_v <= LAT_MAX) & (lon_v >= LON_MIN) & (lon_v <= LON_MAX)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise RuntimeError("WCOFS Monterey Bay bbox selected no grid cells")
    return (
        slice(int(ys.min()), int(ys.max()) + 1, GRID_STRIDE),
        slice(int(xs.min()), int(xs.max()) + 1, GRID_STRIDE),
    )


def _normalize_wcofs_frame(df: pd.DataFrame, chunk: dict) -> pd.DataFrame:
    rename = {
        "ocean_time": "time",
        "lat_rho": "latitude",
        "lon_rho": "longitude",
        "s_rho": "surface_sigma",
        "urot": "u",
        "vrot": "v",
        "zeta": "sea_surface_height_m",
    }
    out = df.rename(columns=rename).copy()
    for col in ["time", "latitude", "longitude", "surface_sigma", "u", "v", "temp", "salt", "sea_surface_height_m"]:
        if col not in out.columns:
            out[col] = np.nan
    out["time"] = pd.to_datetime(out["time"], errors="coerce", utc=True)
    for col in ["latitude", "longitude", "surface_sigma", "u", "v", "temp", "salt", "sea_surface_height_m"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = _clean_physical_bounds(out)
    out["current_speed_m_s"] = (out["u"] ** 2 + out["v"] ** 2) ** 0.5
    out["source_dataset"] = "CeNCOOS THREDDS AWS_WCOFS"
    out["source_url"] = WCOFS_OPENDAP
    out["time_stride_hours"] = 24
    out["chunk_start"] = chunk["start"]
    out["chunk_end"] = chunk["end"]
    keep = [
        "time",
        "latitude",
        "longitude",
        "surface_sigma",
        "u",
        "v",
        "temp",
        "salt",
        "sea_surface_height_m",
        "current_speed_m_s",
        "source_dataset",
        "source_url",
        "time_stride_hours",
        "chunk_start",
        "chunk_end",
    ]
    out = out[keep].dropna(subset=["time", "latitude", "longitude"])
    return out.dropna(subset=["u", "v", "temp", "salt", "sea_surface_height_m"], how="all")


def _to_xarray_time(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _clean_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["u", "v", "temp", "salt", "sea_surface_height_m"]:
        if col in out.columns:
            out.loc[out[col] <= -9990, col] = np.nan
    if "salt" in out.columns:
        out.loc[(out["salt"] < 0) | (out["salt"] > 45), "salt"] = np.nan
    if "temp" in out.columns:
        out.loc[(out["temp"] < -5) | (out["temp"] > 40), "temp"] = np.nan
    if "sea_surface_height_m" in out.columns:
        out.loc[(out["sea_surface_height_m"] < -10) | (out["sea_surface_height_m"] > 10), "sea_surface_height_m"] = np.nan
    return out


def _month_end(month_start: dt.date) -> dt.date:
    return _next_month(month_start) - dt.timedelta(days=1)


def _next_month(month_start: dt.date) -> dt.date:
    if month_start.month == 12:
        return dt.date(month_start.year + 1, 1, 1)
    return dt.date(month_start.year, month_start.month + 1, 1)
