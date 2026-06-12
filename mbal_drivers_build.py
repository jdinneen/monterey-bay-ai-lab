#!/usr/bin/env python
"""
Build a leakage-safe hourly exogenous-driver table for the MBAL M1 forecasting harness.

The neural harness (mbal_neural_forecast.py) currently feeds the models only CALENDAR
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
PROJ = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
LAKEHOUSE_DIR = Path(os.environ.get("MBAL_LAKEHOUSE_DIR", PROJ / "lakehouse"))
M1_PARQUET = PROJ / "mbal_history" / "opendap" / "m1_history.parquet"
NOAA_DIR = PROJ / "mbal_history" / "noaa"
CACHE = PROJ / "nn_cache"
OUT_PARQUET = CACHE / "drivers_hourly.parquet"
OUT_MANIFEST = CACHE / "drivers_manifest.json"
SILVER_DRIVER_DIR = LAKEHOUSE_DIR / "silver" / "external_drivers"
SILVER_OUT_PARQUET = SILVER_DRIVER_DIR / "drivers_hourly.parquet"
SILVER_OUT_MANIFEST = SILVER_DRIVER_DIR / "drivers_manifest.json"
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
    hourly date_range over [min, max]. (mbal_forecast_v2.build_hourly_matrix line 273
    does matrix.resample('1h'), and mbal_neural_forecast reindexes to date_range(min,max,
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


def _curated_path(source: str) -> Path:
    return PROJ / "data" / "external_curated" / source / f"{source}.parquet"


def _daily_point_series(
    path: Path,
    time_col: str,
    value_map: dict[str, str],
    *,
    filters: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    if d.empty or time_col not in d.columns:
        return None
    keep = [time_col] + [src for src in value_map if src in d.columns]
    d = d[keep].copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col])
    if filters:
        for col, (lo, hi) in filters.items():
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").where(
                    pd.to_numeric(d[col], errors="coerce").between(lo, hi)
                )
    for src in value_map:
        if src in d.columns:
            d[src] = pd.to_numeric(d[src], errors="coerce")
    d["date"] = d[time_col].dt.normalize()
    daily = d.groupby("date")[[src for src in value_map if src in d.columns]].mean()
    return daily.rename(columns=value_map)


def _add_daily_frame(
    out: pd.DataFrame,
    grid: pd.DatetimeIndex,
    daily: pd.DataFrame | None,
    source_name: str,
    notes: dict,
    *,
    lag_days: int = DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS,
) -> None:
    if daily is None or daily.empty:
        notes[source_name] = "MISSING or empty curated parquet"
        return
    hg = _load_daily_available(daily, grid, availability_lag_days=lag_days)
    added = []
    for col in hg.columns:
        out[col] = hg[col]
        added.append(col)
    notes[source_name] = (
        f"{len(added)} daily observed columns become visible after +{lag_days}d, "
        "then past-only ffill/staleness masking"
    )


def _add_cencoos_mooring_drivers(out: pd.DataFrame, grid: pd.DatetimeIndex, notes: dict) -> None:
    path = _curated_path("cencoos_mbal_moorings")
    if not path.exists():
        notes["cencoos_mbal_moorings"] = "MISSING curated parquet"
        return
    d = pd.read_parquet(path)
    if d.empty or "time" not in d.columns:
        notes["cencoos_mbal_moorings"] = "empty curated parquet"
        return
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["depth_m"] = pd.to_numeric(d.get("z"), errors="coerce").abs()
    d["sea_water_temperature"] = pd.to_numeric(d.get("sea_water_temperature"), errors="coerce").where(
        pd.to_numeric(d.get("sea_water_temperature"), errors="coerce").between(-2.5, 35.0)
    )
    d["sea_water_practical_salinity"] = pd.to_numeric(
        d.get("sea_water_practical_salinity"), errors="coerce"
    ).where(pd.to_numeric(d.get("sea_water_practical_salinity"), errors="coerce").between(0.0, 42.0))
    d["air_temperature"] = pd.to_numeric(d.get("air_temperature"), errors="coerce").where(
        pd.to_numeric(d.get("air_temperature"), errors="coerce").between(-20.0, 45.0)
    )
    bins = {
        "surface": (0, 10),
        "mid": (10, 75),
        "deep": (75, 500),
    }
    frames = []
    for label, (lo, hi) in bins.items():
        g = d[d["depth_m"].between(lo, hi, inclusive="left")].copy()
        if g.empty:
            continue
        g["date"] = g["time"].dt.normalize()
        daily = g.groupby("date").agg(
            **{
                f"cencoos_m1_temp_{label}_c": ("sea_water_temperature", "mean"),
                f"cencoos_m1_salinity_{label}_psu": ("sea_water_practical_salinity", "mean"),
            }
        )
        frames.append(daily)
    if "air_temperature" in d:
        air = d.dropna(subset=["time"]).copy()
        air["date"] = air["time"].dt.normalize()
        frames.append(air.groupby("date")["air_temperature"].mean().to_frame("cencoos_m1_air_temp_c"))
    daily = pd.concat(frames, axis=1) if frames else None
    _add_daily_frame(out, grid, daily, "cencoos_mbal_moorings", notes)


def _add_caloes_event_drivers(out: pd.DataFrame, grid: pd.DatetimeIndex, notes: dict) -> None:
    path = _curated_path("caloes_spills")
    if not path.exists():
        notes["caloes_spills"] = "MISSING curated parquet"
        return
    d = pd.read_parquet(path)
    if d.empty or "event_time" not in d.columns:
        notes["caloes_spills"] = "empty curated parquet"
        return
    d["date"] = pd.to_datetime(d["event_time"], utc=True, errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"])
    water = d.get("water", pd.Series("", index=d.index)).astype(str).str.lower().str.contains("yes")
    coast = d.get("county", pd.Series("", index=d.index)).astype(str).str.contains(
        "Monterey|Santa Cruz|San Mateo|San Luis Obispo", case=False, regex=True, na=False
    )
    daily = pd.DataFrame(index=pd.date_range(d["date"].min(), d["date"].max(), freq="D", tz="UTC"))
    counts = d.groupby("date").size().reindex(daily.index, fill_value=0).astype(float)
    water_counts = d[water].groupby("date").size().reindex(daily.index, fill_value=0).astype(float)
    coast_water_counts = d[water & coast].groupby("date").size().reindex(daily.index, fill_value=0).astype(float)
    daily["caloes_spill_cnt_7d"] = counts.rolling(7, min_periods=1).sum()
    daily["caloes_spill_cnt_30d"] = counts.rolling(30, min_periods=1).sum()
    daily["caloes_water_spill_cnt_7d"] = water_counts.rolling(7, min_periods=1).sum()
    daily["caloes_water_spill_cnt_30d"] = water_counts.rolling(30, min_periods=1).sum()
    daily["caloes_central_coast_water_spill_cnt_30d"] = coast_water_counts.rolling(30, min_periods=1).sum()
    _add_daily_frame(out, grid, daily, "caloes_spills", notes)


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


def _clean_wave_series(ndbc: pd.DataFrame, station: str) -> dict[str, pd.Series]:
    """Return physically bounded wave drivers from NDBC observations.

    NDBC historical files can contain placeholder zeros in period/direction fields.
    Keep wave height as non-negative, require positive periods, and encode direction
    with sin/cos so models do not see a false jump between 359 and 0 degrees.
    """
    out: dict[str, pd.Series] = {}
    h_col = f"ndbc{station}_wave_height_m"
    if h_col in ndbc:
        out[h_col] = ndbc[h_col].where(
            ndbc[h_col].between(0.0, 30.0)
        )
    period_cols = [f"ndbc{station}_dom_wave_period_s", f"ndbc{station}_avg_wave_period_s"]
    for col in period_cols:
        if col in ndbc:
            out[col] = ndbc[col].where(ndbc[col].between(1.0, 40.0))
    d_col = f"ndbc{station}_mean_wave_dir_deg"
    if d_col in ndbc:
        deg = ndbc[d_col].where(
            ndbc[d_col].between(0.0, 360.0)
        )
        rad = np.radians(deg)
        out[f"ndbc{station}_mean_wave_dir_sin"] = np.sin(rad)
        out[f"ndbc{station}_mean_wave_dir_cos"] = np.cos(rad)
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

    # --- NDBC stations (dynamic discovery)
    for p in sorted(NOAA_DIR.glob("noaa_ndbc*.parquet")):
        sid = p.stem.replace("noaa_ndbc", "")
        ndbc = _load_hourly(p, grid)
        if ndbc is not None:
            sp_col, dir_col = f"ndbc{sid}_wind_speed_ms", f"ndbc{sid}_wind_dir_deg"
            if sp_col in ndbc and dir_col in ndbc:
                sp, wd = ndbc[sp_col], ndbc[dir_col]
                u, v, along = _wind_components(sp, wd)
                out[sp_col] = sp
                out[f"ndbc{sid}_wind_u_ms"] = u
                out[f"ndbc{sid}_wind_v_ms"] = v
                out[f"ndbc{sid}_wind_alongshore_ms"] = along
                # Wind stress magnitude via quadratic bulk formula tau = rho_air * Cd * |U|^2.
                rho_air, cd = 1.22, 1.3e-3
                out[f"ndbc{sid}_wind_stress_pa"] = rho_air * cd * sp * sp
                out[f"ndbc{sid}_wind_stress_alongshore_pa"] = rho_air * cd * sp.abs() * along
            
            for base in ["pressure_mb", "air_temp_c", "water_temp_c"]:
                col = f"ndbc{sid}_{base}"
                if col in ndbc:
                    out[col] = ndbc[col]
            
            for col, values in _clean_wave_series(ndbc, sid).items():
                out[col] = values
            notes[f"ndbc{sid}"] = f"station {sid} discovered; alongshore@320T"

    # --- CO-OPS stations (dynamic discovery)
    for p in sorted(NOAA_DIR.glob("noaa_coops*.parquet")):
        if p.stem == "noaa_coops": continue # skip the legacy compatibility link
        sid = p.stem.replace("noaa_coops", "")
        coops = _load_hourly(p, grid)
        if coops is not None:
            for col in coops.columns:
                out[col] = coops[col]
                if "wind_speed_ms" in col:
                    sp = coops[col]
                    wd_col = col.replace("wind_speed_ms", "wind_dir_deg")
                    if wd_col in coops:
                        wd = coops[wd_col]
                        _, _, along = _wind_components(sp, wd)
                        out[col.replace("wind_speed_ms", "wind_alongshore_ms")] = along
            notes[f"coops{sid}"] = f"station {sid} discovered"

    # --- USGS discharge (dynamic discovery)
    for p in sorted(NOAA_DIR.glob("noaa_usgs*.parquet")):
        sid = p.stem.replace("noaa_usgs", "")
        usgs = pd.read_parquet(p)
        daily = _load_daily_available(usgs.set_index("time"), grid)
        for col in daily.columns:
            out[col] = daily[col]
        notes[f"usgs{sid}"] = f"station {sid} discovered; daily discharge ffilled"

    # --- CDIP waves (dynamic discovery)
    for p in sorted(NOAA_DIR.glob("noaa_cdip*.parquet")):
        sid = p.stem.replace("noaa_cdip", "")
        cdip = _load_hourly(p, grid)
        if cdip is not None:
            for col in cdip.columns:
                out[col] = cdip[col]
        notes[f"cdip{sid}"] = f"station {sid} discovered; waves"

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

    # --- Statewide Bacteria Signals (from silver layer)
    silver_bact = LAKEHOUSE_DIR / "silver" / "bacteria_drivers" / "imputed_signals.parquet"
    if silver_bact.exists():
        b_df = pd.read_parquet(silver_bact)
        if "imputed_exceed_prob" in b_df.columns:
            b_daily = b_df.groupby("sample_date")["imputed_exceed_prob"].mean().to_frame("statewide_bact_exceed_prob")
            b_daily.index = pd.to_datetime(b_daily.index, utc=True)
            bg = _load_daily_available(b_daily, grid)
            for c in bg.columns:
                out[c] = bg[c]
            notes["bacteria_signals"] = (
                "Statewide imputed bacteria exceedance prob becomes visible "
                f"after +{DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS}d, then past-only ffill"
            )
    else:
        notes["bacteria_signals"] = "MISSING silver lakehouse parquet"

    # --- Curated external products from ops/data_fetch.py. These are observed-only
    # source products, so they enter the hist bucket with an availability lag.
    _add_daily_frame(
        out,
        grid,
        _daily_point_series(_curated_path("mur_sst"), "time", {"sst_c": "mur_sst_c"}),
        "mur_sst",
        notes,
    )
    
    # --- WQP Nutrients
    wqp_path = _curated_path("wqp_nutrients")
    if wqp_path.exists():
        wd = pd.read_parquet(wqp_path)
        if not wd.empty and "ActivityStartDate" in wd.columns:
            wd["date"] = pd.to_datetime(wd["ActivityStartDate"], utc=True, errors="coerce").dt.normalize()
            wd["val"] = pd.to_numeric(wd.get("ResultMeasureValue"), errors="coerce")
            wd = wd.dropna(subset=["date", "val"])
            if not wd.empty:
                daily_wqp = wd.groupby(["date", "CharacteristicName"])["val"].mean().unstack("CharacteristicName")
                daily_wqp.columns = [f"wqp_{c.lower().replace(' ', '_').replace(',', '')}" for c in daily_wqp.columns]
                _add_daily_frame(out, grid, daily_wqp, "wqp_nutrients", notes, lag_days=30)
            else:
                notes["wqp_nutrients"] = "curated parquet contained no valid numerics"
        else:
            notes["wqp_nutrients"] = "empty curated parquet"
    else:
        notes["wqp_nutrients"] = "MISSING curated parquet"

    # --- ROMS Circulation
    roms_path = _curated_path("roms_circulation")
    if roms_path.exists():
        rd = pd.read_parquet(roms_path)
        if not rd.empty and "time" in rd.columns:
            rd["date"] = pd.to_datetime(rd["time"], utc=True, errors="coerce").dt.normalize()
            rd = rd.dropna(subset=["date"])
            keep_cols = ["u", "v", "temp", "salt"]
            rd_sub = rd[["date"] + [c for c in keep_cols if c in rd.columns]].copy()
            for c in keep_cols:
                if c in rd_sub.columns:
                    rd_sub[c] = pd.to_numeric(rd_sub[c], errors="coerce")
            daily_roms = rd_sub.groupby("date").mean()
            daily_roms.columns = [f"roms_{c}" for c in daily_roms.columns]
            _add_daily_frame(out, grid, daily_roms, "roms_circulation", notes)
        else:
            notes["roms_circulation"] = "empty curated parquet"
    else:
        notes["roms_circulation"] = "MISSING curated parquet"
    _add_daily_frame(
        out,
        grid,
        _daily_point_series(_curated_path("viirs_chl"), "time", {"chlor_a": "viirs_chlor_a"}),
        "viirs_chl",
        notes,
    )
    _add_daily_frame(
        out,
        grid,
        _daily_point_series(
            _curated_path("cencoos_ocean_acidification"),
            "time",
            {
                "sea_water_temperature": "cencoos_oa_temp_c",
                "sea_water_practical_salinity": "cencoos_oa_salinity_psu",
                "mass_concentration_of_chlorophyll_in_sea_water": "cencoos_oa_chlorophyll",
                "moles_of_oxygen_per_unit_mass_in_sea_water": "cencoos_oa_oxygen_umol_kg",
                "sea_water_ph_reported_on_total_scale": "cencoos_oa_ph_total",
                "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": "cencoos_oa_pco2_uatm",
            },
            filters={
                "sea_water_temperature": (-2.5, 35.0),
                "sea_water_practical_salinity": (0.0, 42.0),
                "mass_concentration_of_chlorophyll_in_sea_water": (0.0, 500.0),
                "moles_of_oxygen_per_unit_mass_in_sea_water": (0.0, 600.0),
                "sea_water_ph_reported_on_total_scale": (6.5, 9.5),
                "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": (0.0, 5000.0),
            },
        ),
        "cencoos_ocean_acidification",
        notes,
        lag_days=30,
    )
    _add_cencoos_mooring_drivers(out, grid, notes)
    _add_caloes_event_drivers(out, grid, notes)

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
            "grid matches mbal_forecast_v2.build_hourly_matrix resample('1h'). Observed hourly "
            f"drivers are visible only after ceil(observed_time, 1h)+{OBSERVED_DRIVER_AVAILABILITY_LAG_HOURS}h. "
            f"Daily observed products are visible only after date+{DAILY_OBSERVED_DRIVER_AVAILABILITY_LAG_DAYS}d. "
            f"Forward-filled hist values are masked after {MAX_HIST_FFILL_HOURS}h of staleness. "
            "Source-specific notes: "
            + "; ".join(f"{k}: {v}" for k, v in notes.items() if k != "hist_quality")
        ),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_MANIFEST}", flush=True)

    SILVER_DRIVER_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(SILVER_OUT_PARQUET)
    SILVER_OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {SILVER_OUT_PARQUET}", flush=True)
    print(f"Wrote {SILVER_OUT_MANIFEST}", flush=True)

    print("\nCoverage summary (% non-null over m1 span):")
    for c in wide.columns:
        tag = "futr" if c in futr_cols else "hist"
        print(f"  [{tag}] {c}: {coverage[c]*100:.1f}%")


if __name__ == "__main__":
    main()
