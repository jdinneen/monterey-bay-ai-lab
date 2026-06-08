#!/usr/bin/env python
"""
Build a leakage-safe hourly exogenous-driver table for the MBARI M1 forecasting harness.

The neural harness (mbari_neural_forecast.py) currently feeds the models only CALENDAR
futures (hour/doy sin/cos). Our experiments show the models learn real physics, so the
biggest accuracy lever is feeding PHYSICAL DRIVERS. This script assembles those drivers,
aligned to the exact hourly-UTC grid the harness builds from m1_history.parquet, split
into two leakage categories the harness must respect:

  futr  -- DETERMINISTIC, known into the future at any forecast horizon. These need NO
           external data: tidal harmonic regressors (cos/sin at the principal constituent
           angular frequencies) and an astronomical solar-elevation / daylight term. The
           model fits their amplitudes; their values at horizon h are computable in advance.

  hist  -- OBSERVED-ONLY, NOT known in the future (wind, wind stress, upwelling indices,
           buoy/harbor obs). At forecast origin t these are known only up to t. The harness
           must treat them as historical exogenous inputs (hist_exog), never futr_exog.
           Gaps here are filled PAST-ONLY (forward-fill) so no future value ever leaks.

Output contract (nn_cache/):
  drivers_hourly.parquet   WIDE table indexed by `ds` (hourly UTC, covering m1 span).
  drivers_manifest.json    {"futr":[...], "hist":[...], "coverage":{col:frac}, "ds_min",
                            "ds_max", "notes"}

CPU/IO only -- no GPU. Re-runnable. Reads local NOAA driver parquets first; an optional
network fetch for blended winds is attempted by noaa_drivers_fetch.py separately and its
result (noaa_winds.parquet) is consumed here if present.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths & geometry
# ----------------------------------------------------------------------------
PROJ = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
M1_PARQUET = PROJ / "mbari_history" / "opendap" / "m1_history.parquet"
NOAA_DIR = PROJ / "mbari_history" / "noaa"
CACHE = PROJ / "nn_cache"
OUT_PARQUET = CACHE / "drivers_hourly.parquet"
OUT_MANIFEST = CACHE / "drivers_manifest.json"
OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS = 1
DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS = 1
MAX_HIST_FFILL_HOURS = 168

M1_LAT = 36.7511        # median latitude from m1_history.parquet (matches noaa fetcher)
M1_LON = -122.0292      # deg east, -180..180

# Central-CA coastline runs roughly NW-SE; the local "alongshore" (upwelling-favourable,
# equatorward) direction is ~320 deg True. We project wind onto this axis so the model gets
# a single physically meaningful upwelling-driver instead of raw u/v.
COASTLINE_ANGLE_DEG = 320.0

# Principal tidal constituents: name -> period in hours. (futr harmonic basis.)
TIDAL_CONSTITUENTS = {
    "M2": 12.4206, "S2": 12.0000, "N2": 12.6583, "K2": 11.9672,
    "K1": 23.9345, "O1": 25.8193, "P1": 24.0659, "Q1": 26.8684,
    "Mf": 327.86, "Mm": 661.31,
}


# ----------------------------------------------------------------------------
# 1) Canonical hourly grid -- must match the harness exactly.
# ----------------------------------------------------------------------------
def build_grid() -> pd.DatetimeIndex:
    """Replicate the harness grid: resample('1h') of the m1 timestamps, then a dense
    hourly date_range over [min, max]. (mbari_forecast_v2.build_hourly_matrix line 273
    does matrix.resample('1h'), and mbari_neural_forecast reindexes to date_range(min,max,
    freq='1h').) We only need the unique timestamps to find the bounds."""
    t = pd.read_parquet(M1_PARQUET, columns=["time"])["time"]
    t = pd.to_datetime(t, utc=True)
    bounds = pd.Series(1, index=t).resample("1h").mean()
    grid = pd.date_range(bounds.index.min(), bounds.index.max(), freq="1h", tz="UTC")
    grid.name = "ds"
    return grid


# ----------------------------------------------------------------------------
# 2) FUTR drivers -- deterministic, known into the future.
# ----------------------------------------------------------------------------
def build_futr(grid: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=grid)
    # Hours since an absolute epoch so the harmonic phases are continuous & reproducible
    # regardless of where the grid starts.
    epoch = pd.Timestamp("2000-01-01T00:00:00Z")
    t_h = (grid - epoch).total_seconds().to_numpy() / 3600.0

    # Tidal harmonic basis: cos/sin pair per constituent. Unit-amplitude regressors;
    # the network learns each constituent's amplitude & phase via the linear combination.
    for name, period in TIDAL_CONSTITUENTS.items():
        w = 2.0 * math.pi / period
        df[f"tide_{name}_cos"] = np.cos(w * t_h)
        df[f"tide_{name}_sin"] = np.sin(w * t_h)

    # Astronomical solar geometry at M1: solar elevation (deg) and a daylight indicator.
    elev = solar_elevation_deg(grid, M1_LAT, M1_LON)
    df["solar_elev_deg"] = elev
    df["solar_elev_pos"] = np.clip(elev, 0.0, None)   # insolation proxy (>=0)
    df["is_daylight"] = (elev > 0).astype("float64")
    return df


def solar_elevation_deg(idx: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Solar elevation angle (degrees above horizon) via a standard NOAA-style algorithm.
    Deterministic from clock time + geolocation, hence a valid known-future regressor."""
    dt = idx.tz_convert("UTC")
    # Fractional day-of-year and time
    doy = dt.dayofyear.to_numpy().astype(float)
    hour = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0).to_numpy()

    # Fractional year (radians)
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    # Equation of time (minutes) and solar declination (radians)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    time_offset = eqtime + 4.0 * lon            # minutes (lon in deg east)
    tst = hour * 60.0 + time_offset             # true solar time (minutes)
    ha = np.radians(tst / 4.0 - 180.0)          # hour angle (radians)

    latr = math.radians(lat)
    cos_zen = (np.sin(latr) * np.sin(decl)
               + np.cos(latr) * np.cos(decl) * np.cos(ha))
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    elev = 90.0 - np.degrees(np.arccos(cos_zen))
    return elev


# ----------------------------------------------------------------------------
# 3) HIST drivers -- observed-only. Past-only gap fill.
# ----------------------------------------------------------------------------
def _load_hourly(
    path: Path,
    grid: pd.DatetimeIndex,
    availability_lag_hours: int = OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    observed_at = pd.to_datetime(d.index, utc=True)
    available_at = observed_at.ceil("h") + pd.Timedelta(hours=int(availability_lag_hours))
    d.index = available_at
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.reindex(grid)


def _load_daily_available(
    daily: pd.DataFrame,
    grid: pd.DatetimeIndex,
    availability_lag_days: int = DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS,
) -> pd.DataFrame:
    """Place daily observed products on the hourly grid only once that day is available.

    A source row dated D is treated as visible at D + availability_lag_days, 00:00 UTC.
    The caller's later past-only ffill carries that visible value forward.
    """
    available = daily.copy()
    available.index = pd.to_datetime(available.index, utc=True).normalize() + pd.Timedelta(
        days=int(availability_lag_days)
    )
    available = available[~available.index.duplicated(keep="last")].sort_index()
    return available.reindex(grid)


def _wind_components(speed: pd.Series, from_dir_deg: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Meteorological wind: from_dir is the direction the wind blows FROM (deg clockwise
    from N). Return eastward(u)/northward(v) wind-velocity components plus the alongshore
    projection onto the COASTLINE_ANGLE_DEG axis. With this axis a NW wind (the classic
    summer upwelling-favourable case) projects NEGATIVE; the model learns the sign, so the
    convention only needs to be consistent (verified: NW->negative, SE->positive)."""
    # Convert "from" direction to the vector the wind blows TOWARD.
    to_rad = np.radians((from_dir_deg + 180.0) % 360.0)
    u = speed * np.sin(to_rad)   # eastward
    v = speed * np.cos(to_rad)   # northward
    ax = np.radians(COASTLINE_ANGLE_DEG)
    alongshore = u * math.sin(ax) + v * math.cos(ax)
    return u, v, alongshore


def _clean_wave_series(ndbc: pd.DataFrame) -> dict[str, pd.Series]:
    """Return physically bounded wave drivers from NDBC observations.

    NDBC historical files can contain placeholder zeros in period/direction fields.
    Keep wave height as non-negative, require positive periods, and encode direction
    with sin/cos so models do not see a false jump between 359 and 0 degrees.
    """
    out: dict[str, pd.Series] = {}
    if "ndbc46042_wave_height_m" in ndbc:
        out["ndbc46042_wave_height_m"] = ndbc["ndbc46042_wave_height_m"].where(
            ndbc["ndbc46042_wave_height_m"].between(0.0, 30.0)
        )
    period_cols = ["ndbc46042_dom_wave_period_s", "ndbc46042_avg_wave_period_s"]
    for col in period_cols:
        if col in ndbc:
            out[col] = ndbc[col].where(ndbc[col].between(1.0, 40.0))
    if "ndbc46042_mean_wave_dir_deg" in ndbc:
        deg = ndbc["ndbc46042_mean_wave_dir_deg"].where(
            ndbc["ndbc46042_mean_wave_dir_deg"].between(0.0, 360.0)
        )
        rad = np.radians(deg)
        out["ndbc46042_mean_wave_dir_sin"] = np.sin(rad)
        out["ndbc46042_mean_wave_dir_cos"] = np.cos(rad)
    return out


def _hist_quality(raw_hist: pd.DataFrame, filled_hist: pd.DataFrame) -> dict:
    raw_coverage: dict[str, float] = {}
    max_staleness_hours: dict[str, float | None] = {}
    for col in raw_hist.columns:
        observed = raw_hist[col].notna()
        raw_coverage[col] = round(float(observed.mean()), 4)
        if not observed.any():
            max_staleness_hours[col] = None
            continue
        observed_times = pd.Series(pd.NaT, index=raw_hist.index, dtype="datetime64[ns, UTC]")
        observed_times.loc[observed] = raw_hist.index[observed]
        last_observed = observed_times.ffill()
        usable = filled_hist[col].notna() & last_observed.notna()
        if not usable.any():
            max_staleness_hours[col] = None
            continue
        age_hours = (raw_hist.index.to_series() - last_observed).dt.total_seconds() / 3600.0
        max_staleness_hours[col] = round(float(age_hours[usable].max()), 2)
    return {
        "observed_driver_availability_lag_hours": OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS,
        "daily_observed_driver_availability_lag_days": DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS,
        "max_hist_ffill_hours": MAX_HIST_FFILL_HOURS,
        "raw_coverage": raw_coverage,
        "max_staleness_hours": max_staleness_hours,
    }


def _cap_hist_staleness(
    raw_hist: pd.DataFrame,
    filled_hist: pd.DataFrame,
    max_age_hours: int = MAX_HIST_FFILL_HOURS,
) -> pd.DataFrame:
    capped = filled_hist.copy()
    for col in raw_hist.columns:
        observed = raw_hist[col].notna()
        if not observed.any():
            capped[col] = np.nan
            continue
        observed_times = pd.Series(pd.NaT, index=raw_hist.index, dtype="datetime64[ns, UTC]")
        observed_times.loc[observed] = raw_hist.index[observed]
        last_observed = observed_times.ffill()
        age_hours = (raw_hist.index.to_series() - last_observed).dt.total_seconds() / 3600.0
        capped.loc[age_hours > float(max_age_hours), col] = np.nan
    return capped


def build_hist(grid: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=grid)
    notes = {}

    # --- NDBC 46042 (offshore Monterey Bay buoy, ~36.79N/122.40W): best in-situ wind near M1.
    ndbc = _load_hourly(NOAA_DIR / "noaa_ndbc46042.parquet", grid)
    if ndbc is not None:
        sp = ndbc["ndbc46042_wind_speed_ms"]
        wd = ndbc["ndbc46042_wind_dir_deg"]
        u, v, along = _wind_components(sp, wd)
        out["ndbc46042_wind_speed_ms"] = sp
        out["ndbc46042_wind_u_ms"] = u
        out["ndbc46042_wind_v_ms"] = v
        out["ndbc46042_wind_alongshore_ms"] = along
        # Wind stress magnitude via quadratic bulk formula tau = rho_air * Cd * |U|^2.
        rho_air, cd = 1.22, 1.3e-3
        out["ndbc46042_wind_stress_pa"] = rho_air * cd * sp * sp
        out["ndbc46042_wind_stress_alongshore_pa"] = rho_air * cd * sp.abs() * along
        out["ndbc46042_pressure_mb"] = ndbc["ndbc46042_pressure_mb"]
        out["ndbc46042_air_temp_c"] = ndbc["ndbc46042_air_temp_c"]
        out["ndbc46042_water_temp_c"] = ndbc["ndbc46042_water_temp_c"]
        for col, values in _clean_wave_series(ndbc).items():
            out[col] = values
        notes["ndbc46042"] = (
            "offshore buoy ~36.79N/122.40W; wind rotated to alongshore@320T; "
            "wave height/period/direction are observed-only hist drivers; "
            "wave direction is encoded as sin/cos"
        )
    else:
        notes["ndbc46042"] = "MISSING local parquet"

    # --- CO-OPS 9413450 (Monterey harbor, 36.61N/121.89W): good water-temp & pressure.
    coops = _load_hourly(NOAA_DIR / "noaa_coops.parquet", grid)
    if coops is not None:
        out["coops_water_temp_c"] = coops["coops_water_temp_c"]
        out["coops_air_pressure_mb"] = coops["coops_air_pressure_mb"]
        sp = coops["coops_wind_speed_ms"]
        wd = coops["coops_wind_dir_deg"]
        _, _, along = _wind_components(sp, wd)
        out["coops_wind_speed_ms"] = sp
        out["coops_wind_alongshore_ms"] = along
        notes["coops"] = "Monterey harbor 36.61N/121.89W; sheltered -> wind less open-ocean"
    else:
        notes["coops"] = "MISSING local parquet"

    # --- Upwelling indices CUTI/BEUTI at 37N (closest band to M1). Daily -> hourly by
    #     forward-fill (past-only: each hour gets the most recent prior daily value).
    upw = NOAA_DIR / "noaa_upwelling.parquet"
    if upw.exists():
        u = pd.read_parquet(upw)
        daily = _load_daily_available(u, grid)
        for col in ("cuti_37n", "beuti_37n"):
            if col in daily.columns:
                out[col] = daily[col]
        notes["upwelling"] = (
            "CUTI/BEUTI@37N (closest band); daily observed values become visible "
            f"after +{DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS}d, then past-only ffill"
        )
    else:
        notes["upwelling"] = "MISSING local parquet"

    # --- Optional: NCEI blended winds at M1 (network fetch). Daily product -> hourly ffill.
    bw = NOAA_DIR / "noaa_winds.parquet"
    if bw.exists():
        w = pd.read_parquet(bw)
        keep = [c for c in w.columns if c.endswith("_m1")]
        if keep:
            wg = _load_daily_available(w[keep], grid)
            for c in keep:
                out[c] = wg[c]
            notes["blended_winds"] = (
                "NCEI blended seawinds@M1 daily observed product becomes visible "
                f"after +{DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS}d, then past-only ffill"
            )
    else:
        notes["blended_winds"] = "not fetched (network skip/fail) -- NDBC wind is the in-situ fallback"

    # Past-only gap fill: forward-fill only. NEVER bfill (that would leak future obs into
    # the past). Leading NaNs (before first obs) are left as NaN -- documented in coverage.
    raw_hist = out.copy()
    out = _cap_hist_staleness(raw_hist, out.ffill())
    notes["hist_quality"] = _hist_quality(raw_hist, out)
    return out, notes


# ----------------------------------------------------------------------------
# Assemble + write
# ----------------------------------------------------------------------------
def main():
    CACHE.mkdir(exist_ok=True)
    print("Building canonical hourly grid from m1_history...", flush=True)
    grid = build_grid()
    print(f"  grid: {len(grid)} hours  {grid.min()} -> {grid.max()}", flush=True)

    print("Building FUTR (deterministic) drivers...", flush=True)
    futr = build_futr(grid)
    print(f"  {futr.shape[1]} futr columns", flush=True)

    print("Building HIST (observed-only) drivers...", flush=True)
    hist, notes = build_hist(grid)
    print(f"  {hist.shape[1]} hist columns", flush=True)

    wide = pd.concat([futr, hist], axis=1)
    wide.index.name = "ds"

    futr_cols = list(futr.columns)
    hist_cols = list(hist.columns)
    coverage = {c: float(wide[c].notna().mean()) for c in wide.columns}

    wide.to_parquet(OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET}  ({wide.shape[0]} rows x {wide.shape[1]} cols)", flush=True)

    manifest = {
        "futr": futr_cols,
        "hist": hist_cols,
        "coverage": {c: round(coverage[c], 4) for c in wide.columns},
        "input_coverage": {c: round(coverage[c], 4) for c in wide.columns},
        "hist_raw_coverage": notes.get("hist_quality", {}).get("raw_coverage", {}),
        "hist_max_staleness_hours": notes.get("hist_quality", {}).get("max_staleness_hours", {}),
        "observed_driver_availability_lag_hours": OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS,
        "daily_observed_driver_availability_lag_days": DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS,
        "max_hist_ffill_hours": MAX_HIST_FFILL_HOURS,
        "ds_min": str(wide.index.min()),
        "ds_max": str(wide.index.max()),
        "notes": (
            "futr = deterministic, known into the future (tidal harmonic cos/sin pairs at the "
            "M2/S2/N2/K2/K1/O1/P1/Q1/Mf/Mm periods + solar-elevation/daylight at M1); valid as "
            "futr_exog -- the model fits amplitudes. hist = observed-only, must be hist_exog "
            "(NOT known in the future); gaps filled past-only (ffill, never bfill). Hourly UTC "
            "grid matches mbari_forecast_v2.build_hourly_matrix resample('1h'). Observed hourly "
            f"drivers are visible only after ceil(observed_time, 1h)+{OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS}h. "
            f"Daily observed products are visible only after date+{DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS}d. "
            f"Forward-filled hist values are masked after {MAX_HIST_FFILL_HOURS}h of staleness. "
            "Source-specific notes: "
            + "; ".join(f"{k}: {v}" for k, v in notes.items() if k != "hist_quality")
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_MANIFEST}", flush=True)

    print("\nCoverage summary (% non-null over m1 span):")
    for c in wide.columns:
        tag = "futr" if c in futr_cols else "hist"
        print(f"  [{tag}] {c}: {coverage[c]*100:.1f}%")


if __name__ == "__main__":
    main()
