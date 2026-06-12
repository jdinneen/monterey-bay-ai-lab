import pandas as pd
import numpy as np
import os
import json

CORE_COLS = ['source', 'station_id', 'latitude', 'longitude', 'sample_date', 'bacteria_result']
OPTIONAL_COLS = ['county']

def _select_output_cols(df):
    cols = CORE_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    return df[cols]

def value_gate(df):
    if len(df) == 0:
        raise ValueError("CRITIC GATE FAILED: DataFrame is empty. Expected a non-empty unified dataset.")
    
    missing_counts = df[CORE_COLS].isnull().sum()
    if missing_counts.sum() > 0:
        raise ValueError(f"CRITIC GATE FAILED: Missing values detected in core columns.\n{missing_counts[missing_counts > 0]}")
    
    today = pd.Timestamp.today().normalize()
    if (df['sample_date'] > today).any():
        future_leak = df[df['sample_date'] > today]
        raise ValueError(f"CRITIC GATE FAILED: Future leakage detected! {len(future_leak)} rows have dates > today.")

def load_surfrider():
    path = 'data/external_curated/surfrider_bwtf/surfrider_bwtf.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)

    if 'sample_date' in df.columns:
        dates = df['sample_date']
    else:
        date_parts = []
        if 'collection date' in df.columns:
            date_parts.append(df['collection date'].replace('Invalid Date', pd.NA))
        if 'stored collectionTime' in df.columns:
            date_parts.append(df['stored collectionTime'])
        if 'publication_time' in df.columns:
            date_parts.append(df['publication_time'])
        dates = date_parts[0] if date_parts else pd.Series(pd.NaT, index=df.index)
        for fallback in date_parts[1:]:
            dates = dates.fillna(fallback)
    df['sample_date'] = pd.to_datetime(dates, utc=True, errors='coerce').dt.tz_localize(None)

    df['source'] = 'surfrider'

    if 'bacteria_result' in df.columns:
        df['bacteria_result'] = pd.to_numeric(df['bacteria_result'], errors='coerce')
    elif 'Enterococcus (mpn/100mL)' in df.columns and 'Ecoli (mpn/100mL)' in df.columns:
        df['bacteria_result'] = df['Enterococcus (mpn/100mL)'].fillna(df['Ecoli (mpn/100mL)'])
    elif 'Enterococcus (mpn/100mL)' in df.columns:
        df['bacteria_result'] = df['Enterococcus (mpn/100mL)']
    elif 'Ecoli (mpn/100mL)' in df.columns:
        df['bacteria_result'] = df['Ecoli (mpn/100mL)']
    else:
        df['bacteria_result'] = np.nan

    df = df.rename(columns={'site id': 'station_id', 'site_id': 'station_id'})
    
    for col in ['latitude', 'longitude']:
        if col not in df.columns:
            df[col] = np.nan
            
    return _select_output_cols(df)

def load_wqp():
    path = 'data/external_curated/wqp/wqp.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['sample_date'] = pd.to_datetime(df['ActivityStartDate'], errors='coerce').dt.tz_localize(None)
    df['source'] = 'wqp'
    df = df.rename(columns={
        'MonitoringLocationIdentifier': 'station_id',
        'lat': 'latitude',
        'lon': 'longitude',
        'ResultMeasureValue': 'bacteria_result'
    })
    
    if 'bacteria_result' in df.columns:
        df['bacteria_result'] = pd.to_numeric(df['bacteria_result'], errors='coerce')
    else:
        df['bacteria_result'] = np.nan
        
    for col in ['latitude', 'longitude']:
        if col not in df.columns:
            df[col] = np.nan
            
    return _select_output_cols(df)

def load_legacy():
    path = 'bacteria_results/statewide/statewide_beach_observations.parquet'
    if not os.path.exists(path):
        path = 'bacteria_results/statewide/ca_beach_observations.csv'
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    
    df['source'] = 'legacy'
    if 'sample_date' in df.columns:
        df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce').dt.tz_localize(None)
    else:
        df['sample_date'] = pd.NaT
        
    if 'result_value_numeric' in df.columns:
        df = df.rename(columns={'result_value_numeric': 'bacteria_result'})
    else:
        df['bacteria_result'] = np.nan
        
    # Attempt to merge lat/lon from station_geo if missing
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        geo_path = 'reports/station_geo.parquet'
        if os.path.exists(geo_path):
            geo = pd.read_parquet(geo_path)[['station_id', 'latitude', 'longitude']]
            df = df.merge(geo, on='station_id', how='left')
    
    for col in ['station_id', 'latitude', 'longitude']:
        if col not in df.columns:
            df[col] = np.nan
            
    return _select_output_cols(df)

# --- Environmental driver enrichment (Monterey Bay point/bbox sources) -----------
# mur_sst, viirs_chl, hf_radar describe Monterey Bay (M1 point / bay bbox). They are
# attached ONLY to observations within the greater Monterey region; a San Diego sample
# cannot be described by M1 SST, so those rows stay NaN (honest, not broadcast).
M1_LAT, M1_LON = 36.7511, -122.0292
MONTEREY_RADIUS_DEG = 0.75  # ~greater Monterey Bay region (~75 km)
DRIVER_COLS = [
    'sst_c', 'chlor_a', 'current_speed_ms',
    'cencoos_oa_temp_c', 'cencoos_oa_salinity_psu', 'cencoos_oa_chlorophyll',
    'cencoos_oa_oxygen_umol_kg', 'cencoos_oa_ph_total', 'cencoos_oa_pco2_uatm',
    'cencoos_m1_temp_surface_c', 'cencoos_m1_salinity_surface_psu',
    'cencoos_m1_temp_mid_c', 'cencoos_m1_salinity_mid_psu',
    'cencoos_m1_temp_deep_c', 'cencoos_m1_salinity_deep_psu',
    'cencoos_m1_air_temp_c',
]
EVENT_COLS = [
    'caloes_spill_cnt_7d_county', 'caloes_spill_cnt_30d_county',
    'caloes_water_spill_cnt_7d_county', 'caloes_water_spill_cnt_30d_county',
    'caloes_spill_observed',
]

def _daily_point(path, value_col, out_col):
    """Daily mean of a point time series -> columns [date, out_col]."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['date', out_col])
    df = pd.read_parquet(path)
    if df.empty or value_col not in df.columns or 'time' not in df.columns:
        return pd.DataFrame(columns=['date', out_col])
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    g = df.dropna(subset=['date']).groupby('date')[value_col].mean().reset_index()
    return g.rename(columns={value_col: out_col})

def _daily_hf_radar(path):
    """Daily mean surface-current speed over the bay bbox -> [date, current_speed_ms]."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['date', 'current_speed_ms'])
    df = pd.read_parquet(path)
    if df.empty or not {'time', 'u', 'v'}.issubset(df.columns):
        return pd.DataFrame(columns=['date', 'current_speed_ms'])
    df['speed'] = np.sqrt(df['u'].astype(float) ** 2 + df['v'].astype(float) ** 2)
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    g = df.dropna(subset=['date']).groupby('date')['speed'].mean().reset_index()
    return g.rename(columns={'speed': 'current_speed_ms'})

def _daily_cencoos_oa(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            'date', 'cencoos_oa_temp_c', 'cencoos_oa_salinity_psu',
            'cencoos_oa_chlorophyll', 'cencoos_oa_oxygen_umol_kg',
            'cencoos_oa_ph_total', 'cencoos_oa_pco2_uatm',
        ])
    df = pd.read_parquet(path)
    if df.empty or 'time' not in df.columns:
        return pd.DataFrame(columns=['date'])
    rename = {
        'sea_water_temperature': 'cencoos_oa_temp_c',
        'sea_water_practical_salinity': 'cencoos_oa_salinity_psu',
        'mass_concentration_of_chlorophyll_in_sea_water': 'cencoos_oa_chlorophyll',
        'moles_of_oxygen_per_unit_mass_in_sea_water': 'cencoos_oa_oxygen_umol_kg',
        'sea_water_ph_reported_on_total_scale': 'cencoos_oa_ph_total',
        'mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas': 'cencoos_oa_pco2_uatm',
    }
    keep = ['time'] + [c for c in rename if c in df.columns]
    df = df[keep].copy()
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    for c in rename:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    out = df.dropna(subset=['date']).groupby('date')[[c for c in rename if c in df.columns]].mean().reset_index()
    return out.rename(columns=rename)

def _daily_cencoos_moorings(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['date'])
    df = pd.read_parquet(path)
    if df.empty or 'time' not in df.columns:
        return pd.DataFrame(columns=['date'])
    df = df.copy()
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    df['depth_m'] = pd.to_numeric(df.get('z'), errors='coerce').abs()
    temp = pd.to_numeric(df.get('sea_water_temperature'), errors='coerce')
    salt = pd.to_numeric(df.get('sea_water_practical_salinity'), errors='coerce')
    df['sea_water_temperature'] = temp.where(temp.between(-2.5, 35.0))
    df['sea_water_practical_salinity'] = salt.where(salt.between(0.0, 42.0))
    air = pd.to_numeric(df.get('air_temperature'), errors='coerce')
    df['air_temperature'] = air.where(air.between(-20.0, 45.0))
    frames = []
    for label, lo, hi in [('surface', 0, 10), ('mid', 10, 75), ('deep', 75, 500)]:
        g = df[df['depth_m'].between(lo, hi, inclusive='left')]
        if g.empty:
            continue
        frames.append(g.groupby('date').agg(**{
            f'cencoos_m1_temp_{label}_c': ('sea_water_temperature', 'mean'),
            f'cencoos_m1_salinity_{label}_psu': ('sea_water_practical_salinity', 'mean'),
        }))
    frames.append(df.groupby('date')['air_temperature'].mean().to_frame('cencoos_m1_air_temp_c'))
    return pd.concat(frames, axis=1).reset_index()

def _clean_county(s):
    return (s.astype(str)
            .str.replace(r'\s+County$', '', regex=True)
            .str.replace(r'\s+City$', '', regex=True)
            .str.strip())

def _caloes_county_daily(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['county', 'date'] + EVENT_COLS)
    df = pd.read_parquet(path)
    if df.empty or 'event_time' not in df.columns or 'county' not in df.columns:
        return pd.DataFrame(columns=['county', 'date'] + EVENT_COLS)
    df = df.copy()
    df['date'] = pd.to_datetime(df['event_time'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    df['county'] = _clean_county(df['county'])
    df = df.dropna(subset=['date', 'county'])
    df = df[df['county'].ne('') & df['county'].ne('<NA>')]
    df['water_flag'] = df.get('water', '').astype(str).str.lower().str.contains('yes')
    parts = []
    for county, g in df.groupby('county'):
        idx = pd.date_range(g['date'].min(), g['date'].max(), freq='D')
        daily = pd.DataFrame(index=idx)
        cnt = g.groupby('date').size().reindex(idx, fill_value=0).astype(float)
        water = g[g['water_flag']].groupby('date').size().reindex(idx, fill_value=0).astype(float)
        daily['county'] = county
        daily['date'] = daily.index
        prior_cnt = cnt.shift(1).fillna(0.0)
        prior_water = water.shift(1).fillna(0.0)
        daily['caloes_spill_cnt_7d_county'] = prior_cnt.rolling(7, min_periods=1).sum()
        daily['caloes_spill_cnt_30d_county'] = prior_cnt.rolling(30, min_periods=1).sum()
        daily['caloes_water_spill_cnt_7d_county'] = prior_water.rolling(7, min_periods=1).sum()
        daily['caloes_water_spill_cnt_30d_county'] = prior_water.rolling(30, min_periods=1).sum()
        daily['caloes_spill_observed'] = 1.0
        parts.append(daily.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['county', 'date'] + EVENT_COLS)

def enrich_with_drivers(df):
    """Left-join daily Monterey drivers onto Monterey-region observations only."""
    sst = _daily_point('data/external_curated/mur_sst/mur_sst.parquet', 'sst_c', 'sst_c')
    chl = _daily_point('data/external_curated/viirs_chl/viirs_chl.parquet', 'chlor_a', 'chlor_a')
    cur = _daily_hf_radar('data/external_curated/hf_radar/hf_radar.parquet')
    oa = _daily_cencoos_oa('data/external_curated/cencoos_ocean_acidification/cencoos_ocean_acidification.parquet')
    m1 = _daily_cencoos_moorings('data/external_curated/cencoos_mbal_moorings/cencoos_mbal_moorings.parquet')
    drivers = sst.merge(chl, on='date', how='outer').merge(cur, on='date', how='outer')
    drivers = drivers.merge(oa, on='date', how='outer').merge(m1, on='date', how='outer')

    dist = np.sqrt((df['latitude'] - M1_LAT) ** 2 + (df['longitude'] - M1_LON) ** 2)
    df['in_monterey_region'] = dist <= MONTEREY_RADIUS_DEG

    if drivers.empty:
        for c in DRIVER_COLS:
            df[c] = np.nan
        return df

    df['_date'] = df['sample_date'].dt.normalize()
    df = df.merge(drivers, left_on='_date', right_on='date', how='left').drop(columns=['_date', 'date'])
    # drivers are only meaningful for Monterey-region rows; null them elsewhere
    for c in DRIVER_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df.loc[~df['in_monterey_region'], c] = np.nan
    return df

def enrich_with_events(df):
    """Left-join strictly prior county-level Cal OES spill pressure features."""
    events = _caloes_county_daily('data/external_curated/caloes_spills/caloes_spills.parquet')
    df = df.copy()
    df['_date'] = df['sample_date'].dt.normalize()
    if 'county' in df.columns:
        df['_county'] = _clean_county(df['county'])
    else:
        df['_county'] = ''
    if events.empty:
        for c in EVENT_COLS:
            df[c] = 0.0
        return df.drop(columns=['_date', '_county'], errors='ignore')
    events = events.rename(columns={'county': '_event_county'})
    out = df.merge(events, left_on=['_county', '_date'], right_on=['_event_county', 'date'], how='left')
    for c in EVENT_COLS:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return out.drop(columns=['_date', '_county', '_event_county', 'date'], errors='ignore')

def main():
    dfs = []
    
    surfrider = load_surfrider()
    if not surfrider.empty:
        dfs.append(surfrider)
        
    wqp = load_wqp()
    if not wqp.empty:
        dfs.append(wqp)
        
    legacy = load_legacy()
    if not legacy.empty:
        dfs.append(legacy)
        
    if not dfs:
        raise ValueError("No data found!")
        
    final_df = pd.concat(dfs, ignore_index=True)
    
    final_df = final_df.dropna(subset=CORE_COLS)
    
    final_df['station_id'] = final_df['station_id'].astype(str)
    final_df['bacteria_result'] = final_df['bacteria_result'].astype(float)
    final_df['latitude'] = final_df['latitude'].astype(float)
    final_df['longitude'] = final_df['longitude'].astype(float)
    
    # Check for leakage strictly before passing to value_gate
    # Wait, value_gate will raise if dates are > today. We just let value_gate do it.
    
    value_gate(final_df)

    # Enrich with Monterey Bay environmental drivers (added after the gate so core
    # columns are untouched; driver columns are nullable by design).
    final_df = enrich_with_drivers(final_df)
    final_df = enrich_with_events(final_df)

    os.makedirs('data/lakehouse', exist_ok=True)
    final_df.to_parquet('data/lakehouse/master_bacteria.parquet', index=False)
    silver_dir = 'lakehouse/silver/bacteria_observations'
    os.makedirs(silver_dir, exist_ok=True)
    final_df.to_parquet(os.path.join(silver_dir, 'master_bacteria.parquet'), index=False)
    manifest = {
        'schema_version': 1,
        'table': 'bacteria_observations',
        'rows': int(len(final_df)),
        'core_columns': CORE_COLS,
        'driver_columns': DRIVER_COLS,
        'event_columns': EVENT_COLS,
        'source_tables': [
            'data/external_curated/surfrider_bwtf/surfrider_bwtf.parquet',
            'data/external_curated/wqp/wqp.parquet',
            'bacteria_results/statewide/statewide_beach_observations.parquet',
            'data/external_curated/mur_sst/mur_sst.parquet',
            'data/external_curated/viirs_chl/viirs_chl.parquet',
            'data/external_curated/hf_radar/hf_radar.parquet',
            'data/external_curated/cencoos_ocean_acidification/cencoos_ocean_acidification.parquet',
            'data/external_curated/cencoos_mbal_moorings/cencoos_mbal_moorings.parquet',
            'data/external_curated/caloes_spills/caloes_spills.parquet',
        ],
        'notes': (
            'Core lab observations pass value_gate before enrichment. Monterey Bay '
            'environmental drivers are attached only to samples within the M1 regional '
            'radius. Cal OES county spill features use rolling counts through the prior '
            'calendar day and carry caloes_spill_observed as a coverage mask.'
        ),
    }
    with open(os.path.join(silver_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Successfully built lakehouse! Rows: {len(final_df)}")
    print(f"Wrote silver bacteria table: {os.path.join(silver_dir, 'master_bacteria.parquet')}")
    region = int(final_df['in_monterey_region'].sum())
    print(f"Monterey-region rows (driver-eligible): {region:,} / {len(final_df):,}")
    for c in DRIVER_COLS:
        cov = int(final_df[c].notna().sum())
        print(f"  driver {c}: {cov:,} rows populated")

if __name__ == "__main__":
    main()
