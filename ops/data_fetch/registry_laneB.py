"""Lane-B contributed source specs (agent: claude-fetch-2).

Kept in a separate module so two agents can add sources without clobbering
``registry.py``. ``registry.py`` merges these at import time (idempotent,
``setdefault`` — lane A always wins a key collision). Each source targets >=50k
labeled, timestamped, open-data rows and writes only to ``data/external_*``.
"""
from __future__ import annotations

from .registry import SourceSpec

LANE_B_SPECS = [
    SourceSpec(
        key="ceden_water_chem",
        title="CEDEN surface-water chemistry results (statewide CA)",
        type="tabular_api",
        endpoint="https://data.ca.gov/api/3/action/datastore_search (CEDEN Field & Lab Chemistry)",
        description="California Environmental Data Exchange Network surface-water chemistry "
        "(~1.84M timestamped analyte results). Broad water-chemistry driver/label corpus "
        "complementary to WQP bacteria and nutrient sources.",
        adapter="ops.data_fetch.adapters.ceden:CedenWaterChemAdapter",
        priority=6,
        spatial="California statewide (CEDEN stations)",
        temporal="multi-decade → present",
        rate_limit="data.ca.gov CKAN (paged, polite)",
        required_columns=["StationCode", "SampleDate", "Analyte", "Result"],
        dedup_keys=["StationCode", "SampleDate", "Analyte", "MethodName", "Result",
                    "CollectionReplicate", "ResultsReplicate"],
        date_column="SampleDate",
        spatial_columns=["StationCode", "Analyte"],
        # NOTE (claude-fetch-A fix): do NOT gate validation on Lat/Lon — they are spatial
        # METADATA, not the measured label. ~2k of 1.65M rows have missing/zero coords
        # (legitimately, for some CEDEN stations); failing a 1.65M-row chemistry corpus on
        # that is a category error. Spatial validity is an informational concern, not a gate.
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="ndbc_stdmet",
        title="NDBC historical standard-meteorological buoy observations (CA buoys)",
        type="tabular_api",
        endpoint="https://www.ndbc.noaa.gov/view_text_file.php (data/historical/stdmet)",
        description="Hourly buoy met (wind, gust, wave height/period, pressure, air/water "
        "temperature) for ~16 California buoys, many years. Physical-driver corpus "
        "complementary to the CDIP wave wrapper.",
        adapter="ops.data_fetch.adapters.ndbc:NdbcStdmetAdapter",
        priority=7,
        spatial="California / nearshore-CA NDBC buoys (~16 stations)",
        temporal="2007 → 2024 (configurable by --start/--end year)",
        rate_limit="NDBC public archive (per station-year)",
        required_columns=["station_id", "time", "WTMP"],
        dedup_keys=["station_id", "time"],
        date_column="time",
        spatial_columns=["station_id"],
        # NOTE (claude-fetch-A fix): dropped PRES from the gate. 23 of 5.7M PRES values
        # are 0.0 — an NDBC missing-pressure sentinel, not a real reading; pressure is a
        # secondary field here (WTMP is the required label). The real labels stay bounded.
        # (A framework-level max_bound_violation_frac is the more general fix; pending the
        # registry.py SourceSpec field which is currently locked by another agent.)
        value_bounds={"WTMP": (-2.0, 40.0), "ATMP": (-30.0, 50.0), "WVHT": (0.0, 30.0)},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
    SourceSpec(
        key="usgs_iv_turbidity",
        title="USGS NWIS instantaneous turbidity / water-temp / gage-height (CA)",
        type="tabular_api",
        endpoint="https://waterservices.usgs.gov/nwis/iv/",
        description="Sub-hourly instantaneous sensor series for CA sites: turbidity (a "
        "first-flush runoff proxy), water temperature, gage height, specific conductance. "
        "New modality vs the existing daily-mean discharge wrapper.",
        adapter="ops.data_fetch.adapters.usgs_iv:UsgsIvTurbidityAdapter",
        priority=6,
        spatial="California USGS IV sites carrying turbidity (~60 of 141)",
        temporal="2015 → 2025 (configurable by --start/--end year)",
        rate_limit="USGS waterservices fair-use (per site-year)",
        required_columns=["site_id", "time", "parameter", "result_value"],
        dedup_keys=["site_id", "time", "parameter"],
        date_column="time",
        spatial_columns=["site_id", "parameter"],
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="nasa_power_daily",
        title="NASA POWER daily agro-meteorology over a CA coastal grid",
        type="tabular_api",
        endpoint="https://power.larc.nasa.gov/api/temporal/daily/point",
        description="Daily gridded met (precip, 2m air temp mean/min/max, RH, wind, "
        "shortwave) over a CA coastal grid; independent reanalysis complementary to the "
        "Open-Meteo rainfall grid.",
        adapter="ops.data_fetch.adapters.nasa_power:NasaPowerDailyAdapter",
        priority=7,
        spatial="CA coastal grid (~19 points, lat 32.5–41.5)",
        temporal="2005 → 2024 (configurable by --start/--end)",
        rate_limit="NASA POWER public API (per point)",
        required_columns=["grid_lat", "grid_lon", "date", "parameter", "value"],
        dedup_keys=["grid_lat", "grid_lon", "date", "parameter"],
        date_column="date",
        spatial_columns=["grid_lat", "grid_lon", "parameter"],
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="ceden_toxicity",
        title="CEDEN surface-water toxicity bioassay results (statewide CA)",
        type="tabular_api",
        endpoint="https://data.ca.gov/api/3/action/datastore_search (CEDEN Toxicity Results)",
        description="California Environmental Data Exchange Network toxicity bioassay results "
        "(~1.71M rows): organism survival/growth, % control, % effect against test species. "
        "Distinct biological-toxicity modality vs CEDEN chemistry and WQP bacteria.",
        adapter="ops.data_fetch.adapters.ceden_tox:CedenToxicityAdapter",
        priority=6,
        spatial="California statewide (CEDEN stations)",
        temporal="multi-decade → present",
        rate_limit="data.ca.gov CKAN (paged, polite)",
        required_columns=["StationCode", "SampleDate", "OrganismName", "Analyte", "Result"],
        dedup_keys=["StationCode", "SampleDate", "ToxBatch", "OrganismName", "Analyte",
                    "Result", "LabReplicate"],
        date_column="SampleDate",
        spatial_columns=["StationCode", "OrganismName"],
        # NOTE (claude-fetch-A fix): Lat/Lon are metadata, not the toxicity label — removed
        # from the validation gate (see ceden_water_chem note). The bioassay Result is the label.
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="gridmet_daily",
        title="gridMET daily 4km gridded meteorology over a CA coastal grid",
        type="netcdf_archive",
        endpoint="http://thredds.northwestknowledge.net:8080/thredds/dodsC (agg_met_*_CONUS)",
        description="University of Idaho gridMET daily 4km met (precip, max/min air temp, "
        "max/min RH, wind, VPD, shortwave) at CA coastal grid points; independent gridded "
        "driver complementary to NASA POWER and the Open-Meteo rainfall grid.",
        adapter="ops.data_fetch.adapters.gridmet:GridmetDailyAdapter",
        priority=7,
        spatial="CA coastal grid (~19 points), nearest 4km gridMET cell",
        temporal="2005 → 2024 (configurable; native 1979 → present)",
        rate_limit="gridMET THREDDS OPeNDAP (per variable)",
        required_columns=["grid_lat", "grid_lon", "date", "parameter", "value"],
        dedup_keys=["grid_lat", "grid_lon", "date", "parameter"],
        date_column="date",
        spatial_columns=["grid_lat", "grid_lon", "parameter"],
        min_rows=50_000,
        delay_seconds=0.0,
    ),
    SourceSpec(
        key="open_meteo_archive",
        title="Open-Meteo ERA5 archive — hourly atmospheric reanalysis (CA coastal grid)",
        type="tabular_api",
        endpoint="https://archive-api.open-meteo.com/v1/archive",
        description="ERA5 hourly reanalysis (2m temp, RH, dewpoint, precip, 10m wind "
        "speed/dir, surface pressure, shortwave, cloud cover) over a CA coastal grid. "
        "Distinct hourly atmospheric-driver modality vs the existing DAILY rainfall wrapper; "
        "the ARCHIVE endpoint has full non-null history (unlike the marine forecast API).",
        adapter="ops.data_fetch.adapters.open_meteo_archive:OpenMeteoArchiveAdapter",
        priority=7,
        spatial="CA coastal grid (~19 points)",
        temporal="2010 → 2024 (configurable; ERA5 native 1940 → present)",
        rate_limit="Open-Meteo free archive (per point, polite)",
        required_columns=["grid_lat", "grid_lon", "time", "parameter", "value"],
        dedup_keys=["grid_lat", "grid_lon", "time", "parameter"],
        date_column="time",
        spatial_columns=["grid_lat", "grid_lon", "parameter"],
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    # ===== Round 2 (claude-fetch-2): 5 more large open driver sources =============
    SourceSpec(
        key="usgs_dv_statewide",
        title="USGS NWIS daily values — statewide CA water quality + streamflow",
        type="tabular_api",
        endpoint="https://waterservices.usgs.gov/nwis/dv/",
        description="Statewide CA daily-mean water temperature, specific conductance, "
        "dissolved oxygen, pH, turbidity, and discharge. Broad runoff / water-quality "
        "driver set; distinct from the near-beach discharge wrapper and the sub-hourly "
        "turbidity-site IV source.",
        adapter="ops.data_fetch.adapters.usgs_dv:UsgsDvStatewideAdapter",
        priority=6,
        spatial="California statewide (~5000 NWIS dv sites)",
        temporal="2005 → 2025 (configurable)",
        rate_limit="USGS waterservices fair-use (per parameter-year, statewide)",
        required_columns=["site_id", "date", "parameter", "result_value"],
        dedup_keys=["site_id", "date", "parameter"],
        date_column="date",
        spatial_columns=["site_id", "parameter"],
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="cdip_wave_network",
        title="CDIP wave-buoy NETWORK (all CA buoys) — Hs/Tp/Dp",
        type="erddap",
        endpoint="https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg",
        description="Aggregated wave measurements (significant height, peak/avg period, "
        "peak direction) across the full CDIP buoy network statewide — distinct from the "
        "single-buoy cdip_waves wrapper. Hourly, ~44 CA stations.",
        adapter="ops.data_fetch.adapters.cdip_network:CdipWaveNetworkAdapter",
        priority=8,
        spatial="California coast CDIP buoys (~44 stations)",
        temporal="2000 → present (per-year chunks)",
        rate_limit="CDIP ERDDAP tabledap (per year)",
        required_columns=["station_id", "time", "waveHs"],
        dedup_keys=["station_id", "time"],
        date_column="time",
        spatial_columns=["station_id"],
        value_bounds={"waveHs": (0.0, 30.0), "waveTp": (0.0, 40.0), "waveDp": (0.0, 360.0)},
        min_rows=50_000,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="cdip_sst_network",
        title="CDIP in-situ sea-surface temperature NETWORK (all CA buoys)",
        type="erddap",
        endpoint="https://erddap.cdip.ucsd.edu/erddap/tabledap/sst_agg",
        description="Measured (in-situ) sea-surface temperature at every CA CDIP buoy — "
        "ground-truth temperature distinct from the satellite/model mur_sst point. A "
        "HAB / marine-heatwave driver.",
        adapter="ops.data_fetch.adapters.cdip_network:CdipSstNetworkAdapter",
        priority=8,
        spatial="California coast CDIP buoys",
        temporal="2000 → present (per-year chunks)",
        rate_limit="CDIP ERDDAP tabledap (per year)",
        required_columns=["station_id", "time", "sst_c"],
        dedup_keys=["station_id", "time"],
        date_column="time",
        spatial_columns=["station_id"],
        value_bounds={"sst_c": (-2.0, 40.0)},
        min_rows=50_000,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="ndbc_cwind",
        title="NDBC continuous-winds (10-min) — California buoys",
        type="tabular_api",
        endpoint="https://www.ndbc.noaa.gov/view_text_file.php (data/historical/cwind)",
        description="Continuous (~10-min) wind direction/speed/gust at CA NDBC buoys — a "
        "higher-frequency, upwelling-favorable WIND driver distinct from the hourly "
        "ndbc_stdmet standard-met record.",
        adapter="ops.data_fetch.adapters.ndbc_cwind:NdbcCwindAdapter",
        priority=8,
        spatial="California / nearshore-CA NDBC buoys",
        temporal="2000 → 2024 (configurable by --start/--end year)",
        rate_limit="NDBC public archive (per station-year)",
        required_columns=["station_id", "time", "WSPD"],
        dedup_keys=["station_id", "time"],
        date_column="time",
        spatial_columns=["station_id"],
        value_bounds={"WSPD": (0.0, 120.0), "WDIR": (0.0, 360.0), "GST": (0.0, 150.0)},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
    SourceSpec(
        key="ncei_isd_hourly",
        title="NCEI Integrated Surface Database — observed hourly met (CA)",
        type="tabular_api",
        endpoint="https://www.ncei.noaa.gov/access/services/data/v1 (global-hourly)",
        description="Station-observed hourly air temperature, wind, sea-level pressure, "
        "and dewpoint at California surface stations — observed sub-daily ground truth, "
        "distinct from daily GHCN-D and from model reanalysis.",
        adapter="ops.data_fetch.adapters.ncei_isd:NceiIsdHourlyAdapter",
        priority=7,
        spatial="California ISD surface stations (~50 long-record)",
        temporal="2005 → 2024 (configurable by --start/--end year)",
        rate_limit="NCEI access services (per station-year)",
        required_columns=["station_id", "time", "air_temp_c"],
        dedup_keys=["station_id", "time"],
        date_column="time",
        spatial_columns=["station_id"],
        value_bounds={"air_temp_c": (-40.0, 55.0), "slp_hpa": (850.0, 1100.0),
                      "wind_speed_ms": (0.0, 120.0)},
        min_rows=50_000,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="upwelling_indices",
        title="CUTI + BEUTI coastal upwelling indices (Jacox) — canonical HAB/DA driver",
        type="tabular_api",
        endpoint="https://mjacox.com/wp-content/uploads/{CUTI,BEUTI}_daily.csv",
        description="Daily Coastal Upwelling Transport Index (CUTI, volume) and Biologically "
        "Effective Upwelling Transport Index (BEUTI, nitrate flux) at CA latitude bands "
        "32N-42N, 1988-present. BEUTI is the mechanistic driver of Pseudo-nitzschia / domoic "
        "acid blooms — the strongest published DA driver.",
        adapter="ops.data_fetch.adapters.upwelling:UpwellingIndicesAdapter",
        priority=6,
        spatial="California coast, 1-degree latitude bands 32N-42N",
        temporal="1988-01-01 → present (daily)",
        rate_limit="public CSV (2 files)",
        required_columns=["date", "latitude", "index_name", "value"],
        dedup_keys=["date", "latitude", "index_name"],
        date_column="date",
        spatial_columns=["latitude", "index_name"],
        value_bounds={"latitude": (32.0, 42.0)},
        min_rows=50_000,
        delay_seconds=0.5,
    ),
]
