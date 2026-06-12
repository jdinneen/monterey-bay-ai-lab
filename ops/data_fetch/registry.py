"""Source registry. Declares every supported source: type, endpoint, coverage,
rate limit, output schema, validation rules, priority, and the adapter class.

The adapter is referenced by a dotted "module:Class" string and resolved lazily so
importing the registry never forces importing every adapter (or its heavy deps).
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceSpec:
    key: str
    title: str
    type: str  # 'tabular_api' | 'bigquery' | 'erddap' | 'grib_archive' | 'raster' | ...
    endpoint: str
    description: str
    adapter: str  # "module.path:ClassName"
    priority: int = 99

    # coverage (human-readable)
    spatial: str = ""
    temporal: str = ""
    rate_limit: str = "unspecified"

    # output schema / validation rules
    required_columns: list[str] = field(default_factory=list)
    dedup_keys: list[str] = field(default_factory=list)
    date_column: Optional[str] = None
    spatial_columns: list[str] = field(default_factory=list)
    value_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_rows: int = 1

    # wrapping / credentials
    wraps_trusted: Optional[str] = None  # repo-relative path to a trusted output (read-only)
    needs_credentials: bool = False
    credentials_doc: str = ""
    credentials_env: list[str] = field(default_factory=list)

    delay_seconds: float = 0.0
    default_status: str = "NOT_STARTED"
    bounded_sample: bool = False  # True => deliberately partial; never auto-promote to READY

    def credentials_present(self) -> bool:
        if not self.needs_credentials:
            return True
        return all(os.environ.get(v) for v in self.credentials_env)

    def build_adapter(self):
        mod_name, _, cls_name = self.adapter.partition(":")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(self)


# ── registry ──────────────────────────────────────────────────────────────────
_SPECS: list[SourceSpec] = [
    # ===== wrappers around existing TRUSTED outputs (read-only inventory+validate)
    SourceSpec(
        key="statewide_beachwatch",
        title="CA BeachWatch statewide beach water-quality observations",
        type="bigquery",
        endpoint="BigQuery blue_current_core_v2.california_beach_sample_observations",
        description="Statewide CA beach bacteria observations (long format). Wraps the "
        "trusted output of research/bacteria/fetch_statewide_beachwatch.py.",
        adapter="ops.data_fetch.adapters.wrap_trusted:WrapTrustedAdapter",
        priority=0,
        spatial="California statewide (~830 beach stations)",
        temporal="2005-01-01 → present",
        rate_limit="BigQuery project quota",
        required_columns=["sample_date", "county", "station_id", "property_id", "result_value_numeric"],
        dedup_keys=["station_id", "sample_date", "property_id"],
        date_column="sample_date",
        spatial_columns=["county", "station_id"],
        wraps_trusted="bacteria_results/statewide/statewide_beach_observations.parquet",
    ),
    SourceSpec(
        key="discharge_usgs",
        title="USGS NWIS daily river discharge (first-flush driver)",
        type="tabular_api",
        endpoint="https://waterservices.usgs.gov/nwis/dv/",
        description="Daily discharge (param 00060) for gauges near CA beaches. Wraps the "
        "trusted output of research/bacteria/fetch_discharge.py.",
        adapter="ops.data_fetch.adapters.wrap_trusted:WrapTrustedAdapter",
        priority=0,
        spatial="CA coastal gauges within 20km of a beach station",
        temporal="2005 → present",
        rate_limit="USGS NWIS fair-use (batched dv)",
        required_columns=["gauge_id", "date", "discharge_cfs"],
        dedup_keys=["gauge_id", "date"],
        date_column="date",
        spatial_columns=["gauge_id"],
        value_bounds={"discharge_cfs": (-1.0, 5_000_000.0)},
        wraps_trusted="bacteria_results/discharge/discharge_gauge.parquet",
    ),
    SourceSpec(
        key="rainfall_openmeteo",
        title="Open-Meteo gridded daily rainfall (AB411 rain-rule driver)",
        type="tabular_api",
        endpoint="https://archive-api.open-meteo.com/v1/archive",
        description="Gridded daily precipitation over CA beach grid cells. Wraps the trusted "
        "output of research/bacteria/fetch_rainfall.py.",
        adapter="ops.data_fetch.adapters.wrap_trusted:WrapTrustedAdapter",
        priority=0,
        spatial="~110 0.1° cells covering CA beach stations",
        temporal="2005 → present",
        rate_limit="Open-Meteo free archive (polite delay)",
        required_columns=["grid_lat", "grid_lon", "date", "precip_mm"],
        dedup_keys=["grid_lat", "grid_lon", "date"],
        date_column="date",
        spatial_columns=["grid_lat", "grid_lon"],
        value_bounds={"precip_mm": (0.0, 2000.0)},
        wraps_trusted="bacteria_results/rainfall/rainfall_grid.parquet",
    ),
    SourceSpec(
        key="cdip_waves",
        title="CDIP/NDBC wave observations",
        type="erddap",
        endpoint="https://thredds.cdip.ucsd.edu/thredds/",
        description="Significant wave height/period/direction. Wraps the trusted output of "
        "research/bacteria/fetch_cdip_waves.py.",
        adapter="ops.data_fetch.adapters.wrap_trusted:WrapTrustedAdapter",
        priority=12,
        spatial="CA coast buoys (currently NDBC 46042 Monterey Bay)",
        temporal="1987 → present",
        rate_limit="CDIP THREDDS public",
        required_columns=["station_id", "sample_date", "Hs", "Tp", "Dm"],
        dedup_keys=["station_id", "sample_date"],
        date_column="sample_date",
        spatial_columns=["station_id"],
        value_bounds={"Hs": (0.0, 30.0), "Tp": (0.0, 120.0), "Dm": (0.0, 360.0)},
        wraps_trusted="bacteria_results/cdip_waves/cdip_waves.parquet",
    ),
    SourceSpec(
        key="tide_stages",
        title="NOAA CO-OPS tide stage (water level)",
        type="tabular_api",
        endpoint="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        description="Hourly water level for CA CO-OPS stations. Wraps the trusted output of "
        "research/bacteria/fetch_tide_stages.py. NOTE: trusted file currently ~1yr only.",
        adapter="ops.data_fetch.adapters.wrap_trusted:WrapTrustedAdapter",
        priority=12,
        spatial="CA CO-OPS tide gauges",
        temporal="~2025-06 → present (gap: needs backfill)",
        rate_limit="CO-OPS API public",
        required_columns=["station_id", "sample_date", "water_level_m"],
        dedup_keys=["station_id", "sample_date"],
        date_column="sample_date",
        spatial_columns=["station_id"],
        wraps_trusted="bacteria_results/tide_stages/tide_stages.parquet",
    ),

    # ===== NEW staged adapters (real fetch into data/external_*) ================
    SourceSpec(
        key="ciwqs_sso",
        title="CIWQS Sanitary Sewer Overflow (sewage spills)",
        type="tabular_api",
        endpoint="https://data.ca.gov/api (CKAN datastore_search)",
        description="California Integrated Water Quality System sanitary-sewer-overflow "
        "spill events published on data.ca.gov (CKAN). A direct land-based contamination "
        "driver for the bacteria model.",
        adapter="ops.data_fetch.adapters.ciwqs_sso:CiwqsSsoAdapter",
        priority=1,
        spatial="California statewide",
        temporal="varies by published resource",
        rate_limit="data.ca.gov CKAN (polite)",
        required_columns=["spill_id", "spill_date"],
        dedup_keys=["spill_id"],
        date_column="spill_date",
        min_rows=1,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="caloes_spills",
        title="Cal OES HazMat spill-release archive",
        type="tabular_api",
        endpoint="https://www.caloes.ca.gov/.../spill-release-reporting/ year Excel archives",
        description="Official Cal OES year-specific HazMat spill/release archive files. "
        "Fetches bounded Excel archive years into staged Parquet with source year and URL.",
        adapter="ops.data_fetch.adapters.caloes:CalOesSpillArchiveAdapter",
        priority=1,
        spatial="California statewide",
        temporal="1993 → present archive; default 2020 → latest available",
        rate_limit="Cal OES public website; yearly Excel files",
        required_columns=["source_year", "source_url", "source_record_id"],
        dedup_keys=["source_year", "source_record_id"],
        date_column="event_time",
        min_rows=1,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="wqp",
        title="Water Quality Portal — expanded statewide CA bacteria results",
        type="tabular_api",
        endpoint="https://www.waterqualitydata.us/data/Result/search",
        description="EPA/USGS/STORET Water Quality Portal results for fecal indicator "
        "bacteria (E. coli, Enterococcus, fecal coliform) statewide CA. Chunked by year.",
        adapter="ops.data_fetch.adapters.wqp:WqpAdapter",
        priority=6,
        spatial="California statewide (statecode US:06)",
        temporal="2018 → present (configurable)",
        rate_limit="WQP web services (chunk by year, polite)",
        required_columns=["MonitoringLocationIdentifier", "ActivityStartDate", "CharacteristicName"],
        dedup_keys=["ResultIdentifier"],
        date_column="ActivityStartDate",
        spatial_columns=["MonitoringLocationIdentifier", "CharacteristicName"],
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="wqp_nutrients",
        title="Water Quality Portal — CA agricultural nutrient loading",
        type="tabular_api",
        endpoint="https://www.waterqualitydata.us/data/Result/search",
        description="EPA/USGS/STORET Water Quality Portal results for agricultural "
        "nutrients (Nitrate, Phosphate, Total Nitrogen, etc.) statewide CA. Chunked by year.",
        adapter="ops.data_fetch.adapters.wqp_nutrients:WqpNutrientsAdapter",
        priority=6,
        spatial="California statewide (statecode US:06)",
        temporal="2018 → present (configurable)",
        rate_limit="WQP web services (chunk by year, polite)",
        required_columns=["MonitoringLocationIdentifier", "ActivityStartDate", "CharacteristicName"],
        dedup_keys=["ResultIdentifier"],
        date_column="ActivityStartDate",
        spatial_columns=["MonitoringLocationIdentifier", "CharacteristicName"],
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="mur_sst",
        title="MUR SST backfill 2020-2026 (M1 point)",
        type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap (jplMURSST41 / MUR)",
        description="JPL MUR sea-surface temperature at the M1 point. Delegates to the "
        "existing resumable MUR fetcher (ops/autonomous_rate_limit_fetcher.py) for a STAGED "
        "backfill of 2020-2026 — does not touch the trusted mbal_history cache.",
        adapter="ops.data_fetch.adapters.mur:MurAdapter",
        priority=5,
        spatial="M1 mooring point (36.7511, -122.0292)",
        temporal="2020-01-01 → present",
        rate_limit="CoastWatch ERDDAP (per-year, Retry-After backoff)",
        required_columns=["time", "sst_c"],
        dedup_keys=["time"],
        date_column="time",
        value_bounds={"sst_c": (-2.0, 40.0)},
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="viirs_chl",
        title="VIIRS ocean color / chlorophyll-a (M1 point)",
        type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap (erdVHNchla1day)",
        description="VIIRS chlorophyll-a proxy at the M1 point. Delegates to the existing "
        "ops/fetch_chlorophyll.py resumable per-year/per-chunk design, STAGED.",
        adapter="ops.data_fetch.adapters.viirs_chl:ViirsChlAdapter",
        priority=8,
        spatial="M1 point (configurable)",
        temporal="2015-02-25 → present",
        rate_limit="CoastWatch ERDDAP (per-chunk, cooldown on 429)",
        required_columns=["time", "chlor_a"],
        dedup_keys=["time"],
        date_column="time",
        value_bounds={"chlor_a": (0.0, 1000.0)},
        min_rows=1,
        delay_seconds=2.0,
    ),

    SourceSpec(
        key="c_harm",
        title="C-HARM statewide marine HAB predictions",
        type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap (wvcharmV3_0day_LonPM180)",
        description="California Harmful Algae Risk Mapping (C-HARM) Nowcast V3. "
        "Provides statewide 3km gridded probabilities for pseudo_nitzschia, "
        "particulate_domoic, and cellular_domoic acid. Chunked by month.",
        adapter="ops.data_fetch.adapters.charm:CharmAdapter",
        priority=8,
        spatial="California coast 3km grid (32.5N to 42.0N)",
        temporal="2018-09 → present",
        rate_limit="CoastWatch ERDDAP (chunk by month, polite)",
        required_columns=["time", "pseudo_nitzschia", "particulate_domoic", "cellular_domoic"],
        date_column="time",
        spatial_columns=["latitude", "longitude"],
        min_rows=1,
        delay_seconds=2.0,
    ),

    # ===== implemented-but-heavy / credentialed adapters =======================
    SourceSpec(
        key="hrrr",
        title="HRRR weather-driver GRIB2 index",
        type="grib_archive",
        endpoint="https://noaa-hrrr-bdp-pds.s3.amazonaws.com (AWS Open Data, GRIB2 .idx)",
        description="NOAA HRRR 3km forecast message index for rain, temperature, and wind "
        "driver extraction. This fetches real public GRIB2 byte-range metadata; downstream "
        "station/catchment subsetting can use the indexed GRIB URLs without a blind bulk pull.",
        adapter="ops.data_fetch.adapters.heavy:HrrrAdapter",
        priority=2,
        spatial="CONUS 3km index rows; downstream subset to California stations/catchments",
        temporal="2014-07 → present",
        rate_limit="AWS S3 anonymous HTTPS (chunk by run hour/day)",
        required_columns=["time", "run_time", "forecast_hour", "variable", "driver", "byte_start", "grib_url"],
        dedup_keys=["run_time", "forecast_hour", "variable", "level", "message"],
        date_column="time",
        spatial_columns=["variable", "driver"],
        min_rows=1,
    ),
    SourceSpec(
        key="nwm",
        title="National Water Model NetCDF object index",
        type="netcdf_archive",
        endpoint="https://noaa-nwm-pds.s3.amazonaws.com (AWS Open Data, NetCDF)",
        description="NOAA National Water Model public NetCDF object inventory for hydrology "
        "driver extraction. This fetches real S3 object metadata for analysis-assimilation "
        "and short-range products; downstream code can byte/download only the needed files.",
        adapter="ops.data_fetch.adapters.heavy:NwmAdapter",
        priority=3,
        spatial="CONUS NWM products; downstream subset to California reaches/catchments",
        temporal="operational daily object inventory",
        rate_limit="AWS S3 anonymous HTTPS ListObjectsV2",
        required_columns=["run_time", "valid_time", "product", "variable_group", "object_key", "url", "size_bytes"],
        dedup_keys=["object_key"],
        date_column="valid_time",
        spatial_columns=["product", "variable_group"],
        min_rows=1,
    ),
    SourceSpec(
        key="echo_dmr",
        title="Permitted wastewater effluent (CA eSMR; ECHO DMR equivalent)",
        type="tabular_api",
        endpoint="https://data.ca.gov CKAN eSMR (federal echodata.epa.gov returned HTTP 500)",
        description="Permitted-discharger effluent monitoring. EPA ECHO DMR REST was down at "
        "build time; adapter targets California's authoritative 2024 eSMR analytical data "
        "on data.ca.gov (the CA slice of ECHO DMR-style effluent reporting). Chunked by "
        "CKAN page; default cap covers the complete 2024 annual resource.",
        adapter="ops.data_fetch.adapters.echo_dmr:EchoDmrAdapter",
        priority=7,
        spatial="California NPDES permittees",
        temporal="2024 annual resource",
        rate_limit="ECHO web services fair-use",
        required_columns=["npdes_id", "monitoring_period_end_date"],
        dedup_keys=["npdes_id", "monitoring_period_end_date", "parameter_code"],
        date_column="monitoring_period_end_date",
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="hf_radar",
        title="HF radar surface currents",
        type="erddap",
        endpoint="https://hfrnet-tds.ucsd.edu/thredds (HFRADAR US West Coast)",
        description="HFRNet hourly surface-current u/v near Monterey Bay. ERDDAP/THREDDS; "
        "chunked by month; staged.",
        adapter="ops.data_fetch.adapters.hf_radar:HfRadarAdapter",
        priority=10,
        spatial="Monterey Bay bbox subset",
        temporal="2012 → present",
        rate_limit="HFRNet THREDDS public",
        required_columns=["time", "lat", "lon", "u", "v"],
        date_column="time",
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="surfrider_bwtf",
        title="Surfrider Blue Water Task Force bacteria samples",
        type="tabular_api",
        endpoint="https://bwtf.surfrider.org (no documented public bulk API)",
        description="Surfrider volunteer bacteria sampling from the public annual CSV "
        "endpoint, normalized to California enterococcus sample rows.",
        adapter="ops.data_fetch.adapters.surfrider:SurfriderAdapter",
        priority=11,
        spatial="CA chapters",
        temporal="varies",
        rate_limit="n/a",
        required_columns=["sample_id", "station", "sample_date", "bacteria_result"],
        dedup_keys=["sample_id"],
        date_column="sample_date",
        spatial_columns=["station"],
        min_rows=1,
    ),
    SourceSpec(
        key="habmap_cdph",
        title="HABMAP / CDPH marine biotoxin & domoic acid labels",
        type="tabular_api",
        endpoint="https://erddap.sccoos.org/erddap (HABs) / CDPH biotoxin reports",
        description="Marine HAB / domoic-acid / biotoxin observations from SCCOOS HABMAP "
        "ERDDAP (and CDPH biotoxin bulletins). Chunked; staged.",
        adapter="ops.data_fetch.adapters.habmap:HabmapAdapter",
        priority=12,
        spatial="California coast HAB sampling piers",
        temporal="2008 → present",
        rate_limit="SCCOOS ERDDAP public",
        required_columns=["time", "latitude", "longitude"],
        date_column="time",
        min_rows=1,
        delay_seconds=1.0,
    ),
    SourceSpec(
        key="cencoos_mbal_moorings",
        title="CenCOOS MBARI M1 public mooring slices",
        type="tabular_api",
        endpoint="https://erddap.cencoos.org/erddap (org_mbari_m1 + M1 historic)",
        description="Bounded, checkpointed ERDDAP tabledap slices for MBARI M1 real-time "
        "and historic datasets. Staged source-health/backfill support; does not overwrite "
        "trusted mbal_history outputs.",
        adapter="ops.data_fetch.adapters.cencoos:CencoosMbalMooringsAdapter",
        priority=13,
        spatial="M1 mooring point, Monterey Bay",
        temporal="2024 → present by default; configurable by --start/--end",
        rate_limit="CenCOOS ERDDAP public tabledap; yearly chunks",
        required_columns=["dataset_id", "time", "latitude", "longitude"],
        dedup_keys=["dataset_id", "time", "z"],
        date_column="time",
        spatial_columns=["dataset_id", "latitude", "longitude", "z"],
        value_bounds={
            "sea_water_temperature": (-2.5, 35.0),
            "sea_water_practical_salinity": (0.0, 42.0),
            "air_temperature": (-20.0, 45.0),
        },
        min_rows=1,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="cencoos_ocean_acidification",
        title="CenCOOS MBARI OA1/OA2 ocean-acidification buoys",
        type="tabular_api",
        endpoint="https://erddap.cencoos.org/erddap (oa1-mbari-buoy-1 + oa2-mbari-buoy)",
        description="Bounded, checkpointed ERDDAP tabledap slices for MBARI ocean-acidification "
        "buoys. OA1 may be intermittently unavailable upstream and is skipped gracefully.",
        adapter="ops.data_fetch.adapters.cencoos:CencoosOceanAcidificationAdapter",
        priority=14,
        spatial="OA1 Hopkins/Monterey and OA2 Santa Cruz/Ano Nuevo area buoys",
        temporal="2024 → present by default; configurable by --start/--end",
        rate_limit="CenCOOS ERDDAP public tabledap; yearly chunks",
        required_columns=["dataset_id", "time", "latitude", "longitude"],
        dedup_keys=["dataset_id", "time", "z"],
        date_column="time",
        spatial_columns=["dataset_id", "latitude", "longitude", "z"],
        value_bounds={
            "sea_water_temperature": (-2.5, 35.0),
            "sea_water_practical_salinity": (0.0, 42.0),
            "mass_concentration_of_chlorophyll_in_sea_water": (0.0, 500.0),
            "moles_of_oxygen_per_unit_mass_in_sea_water": (0.0, 600.0),
            "sea_water_ph_reported_on_total_scale": (6.5, 9.5),
            "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": (0.0, 5000.0),
        },
        min_rows=1,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="beacon_advisories",
        title="BEACON beach advisories/closures normalization",
        type="derived",
        endpoint="(normalizes bacteria_results/statewide advisories)",
        description="Normalizes the trusted statewide advisories into a clean BEACON-style "
        "open/close event table (read trusted, write staged — never overwrites trusted).",
        adapter="ops.data_fetch.adapters.beacon:BeaconAdapter",
        priority=9,
        spatial="California statewide",
        temporal="2005 → present",
        rate_limit="n/a (local transform)",
        required_columns=["county", "beach_name", "advisory_date", "advisory_type"],
        dedup_keys=["county", "beach_name", "advisory_date", "advisory_type"],
        date_column="advisory_date",
        spatial_columns=["county"],
        min_rows=1,
    ),
    SourceSpec(
        key="ccap_nhd_nlcd",
        title="C-CAP / NHDPlus / NLCD static watershed & land-use features",
        type="raster",
        endpoint="TNM NHDPlus HR products API; NOAA C-CAP bulk index; MRLC NLCD services page",
        description="Static land-cover/hydrography source catalog for California beach "
        "catchment features. Fetches real public product/download/service metadata for "
        "NHDPlus HR, NOAA C-CAP, and MRLC/NLCD into staged Parquet; full raster/GDB "
        "subsetting remains an explicit downstream extraction step.",
        adapter="ops.data_fetch.adapters.heavy:CcapNhdNlcdAdapter",
        priority=4,
        spatial="California watersheds",
        temporal="static (epoch snapshots)",
        rate_limit="bulk file servers",
        required_columns=[
            "provider",
            "product_family",
            "product_id",
            "title",
            "url",
            "format",
            "california_relevant",
        ],
        dedup_keys=["provider", "product_id", "url"],
        spatial_columns=["provider", "product_family"],
        min_rows=10,
    ),
    SourceSpec(
        key="station_static_features",
        title="Station-level static physical source/proximity features",
        type="derived",
        endpoint="Local transform of station_geo, ccap_nhd_nlcd, USGS gauge map, eSMR, and CIWQS SSO",
        description="Per-station static covariates for beach bacteria nowcasting: "
        "monitoring density, nearest USGS discharge gauge, nearest geocoded permitted "
        "wastewater facility, nearest geocoded sanitary-sewer-overflow record, and "
        "provenance pointers to the NHDPlus/C-CAP/NLCD catalog.",
        adapter="ops.data_fetch.adapters.station_static:StationStaticFeaturesAdapter",
        priority=4,
        spatial="California beach monitoring stations",
        temporal="static",
        rate_limit="n/a (local transform)",
        required_columns=[
            "station_id",
            "latitude",
            "longitude",
            "station_density_10km",
            "dist_to_usgs_gauge_km",
            "dist_to_npdes_km",
            "dist_to_sso_km",
            "static_catalog_rows",
        ],
        dedup_keys=["station_id"],
        spatial_columns=["county"],
        min_rows=800,
    ),
    SourceSpec(
        key="coops_tide_staged",
        title="NOAA CO-OPS staged water levels",
        type="tabular_api",
        endpoint="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        description="Staged NOAA CO-OPS verified/preliminary water levels, commonly at 6-minute cadence, for California "
        "tide stations used as additive bacteria drivers. Writes only to data/external_curated; "
        "trusted bacteria tide-stage outputs remain read-only.",
        adapter="ops.data_fetch.adapters.coops_tide:CoopsTideStagedAdapter",
        priority=5,
        spatial="California coastal NOAA CO-OPS stations",
        temporal="1996 → present, chunked by station and <=30 day API windows",
        rate_limit="NOAA CO-OPS API; 30-day chunks, polite delay",
        required_columns=["station_id", "sample_date", "water_level_m"],
        dedup_keys=["station_id", "sample_date"],
        date_column="sample_date",
        spatial_columns=["station_id"],
        value_bounds={"water_level_m": (-10.0, 10.0)},
        min_rows=1,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="roms_circulation",
        title="SCCOOS ROMS Circulation (u, v, temp, salt)",
        type="erddap",
        endpoint="https://erddap.sccoos.org/erddap (roms_ncst)",
        description="SCCOOS ROMS gridded circulation archive for California coastal waters. "
        "Provides bounded surface-grid samples of currents (u, v), temperature, and salinity; "
        "adapter uses ERDDAP metadata because this dataset is historical rather than current.",
        adapter="ops.data_fetch.adapters.roms:RomsAdapter",
        priority=8,
        spatial="California coastal ROMS grid; current adapter samples Monterey Bay surface window",
        temporal="2015-03-19 → 2022-12-01 in SCCOOS roms_ncst metadata",
        rate_limit="SCCOOS ERDDAP (polite)",
        required_columns=[
            "time",
            "depth",
            "latitude",
            "longitude",
            "u",
            "v",
            "temp",
            "salt",
            "longitude_180",
            "current_speed_m_s",
            "source",
        ],
        dedup_keys=["time", "depth", "latitude", "longitude"],
        date_column="time",
        spatial_columns=["depth", "latitude", "longitude"],
        value_bounds={"temp": (-5.0, 40.0), "salt": (0.0, 45.0), "current_speed_m_s": (0.0, 5.0)},
        min_rows=1,
        delay_seconds=2.0,
    ),
    SourceSpec(
        key="wcofs_circulation",
        title="NOAA WCOFS operational circulation",
        type="opendap",
        endpoint="https://thredds.cencoos.org/thredds/dodsC/AWS_WCOFS.nc",
        description="Operational NOAA West Coast Operational Forecast System, the ROMS-based "
        "continuation for current West Coast circulation after the discontinued California ROMS "
        "stream. Adapter fetches daily samples from a bounded Monterey Bay surface-grid subset of currents, "
        "temperature, salinity, and sea-surface height from the CeNCOOS THREDDS AWS aggregate.",
        adapter="ops.data_fetch.adapters.wcofs:WcofsCirculationAdapter",
        priority=8,
        spatial="WCOFS West Coast grid; adapter samples Monterey Bay surface window",
        temporal="2022-01 → present/forthcoming forecast horizon in AWS_WCOFS aggregate",
        rate_limit="CeNCOOS THREDDS OPeNDAP; monthly chunks",
        required_columns=[
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
        ],
        dedup_keys=["time", "latitude", "longitude"],
        date_column="time",
        spatial_columns=["latitude", "longitude"],
        value_bounds={
            "temp": (-5.0, 40.0),
            "salt": (0.0, 45.0),
            "sea_surface_height_m": (-10.0, 10.0),
            "current_speed_m_s": (0.0, 5.0),
        },
        min_rows=1,
        delay_seconds=2.0,
    ),

    # ===== marine-mammal mortality thread (whale die-off investigation) =========
    SourceSpec(
        key="marine_mammal_mortality",
        title="Marine-mammal mortality labels (iNaturalist cetaceans, CA)",
        type="tabular_api",
        endpoint="https://api.inaturalist.org/v1/observations",
        description="Open marine-mammal mortality labels for the California coast: iNaturalist "
        "cetacean observations carrying the 'Alive or Dead' annotation (term 17, value 19=Dead). "
        "Fetches BOTH dead and not-dead observations so the table carries its own effort "
        "denominator — mortality rates can be growth-normalized (raw dead counts are confounded "
        "by iNaturalist user growth). Best OPEN substitute for NOAA's request-walled National "
        "Stranding Database. Deliberate capped slice (bounded_sample).",
        adapter="ops.data_fetch.adapters.marine_mammal_mortality:MarineMammalMortalityAdapter",
        priority=6,
        spatial="California coastal bbox (32-42N, -124 to -117W)",
        temporal="2013 → present (year-chunked)",
        rate_limit="iNaturalist API (polite, deep-paged by id_above)",
        required_columns=["observation_id", "time", "taxon_name", "latitude", "longitude", "dead_flag"],
        dedup_keys=["observation_id"],
        date_column="time",
        spatial_columns=["taxon_name", "common_name"],
        value_bounds={"latitude": (32.0, 42.0), "longitude": (-124.0, -117.0), "dead_flag": (0.0, 1.0)},
        min_rows=1,
        delay_seconds=1.0,
        bounded_sample=True,
    ),
    SourceSpec(
        key="cciea_forage",
        title="CCIEA Forage Biomass index — California Current Central",
        type="erddap",
        endpoint="https://oceanview.pfeg.noaa.gov/erddap (cciea_EI_FBC)",
        description="NOAA CCIEA annual forage-biomass anomaly index (RREAS survey) for the "
        "central California Current — Total Krill, Anchovy, YOY Rockfish, Market Squid, etc. The "
        "mechanistic prey/malnutrition driver for the marine-mammal mortality work (NOAA tied the "
        "2019-2023 gray-whale UME to feeding-ground prey decline). Complete published index "
        "(~500 annual rows, 1990→present), long format.",
        adapter="ops.data_fetch.adapters.cciea_forage:CcieaForageAdapter",
        priority=8,
        spatial="California Current — Central region (incl. Monterey Bay foraging grounds)",
        temporal="1990 → present (annual)",
        rate_limit="CoastWatch/CCIEA ERDDAP public",
        required_columns=["time", "species_group", "mean_cpue"],
        dedup_keys=["time", "species_group"],
        date_column="time",
        spatial_columns=["species_group", "region"],
        value_bounds={"mean_cpue": (-10.0, 10.0)},
        min_rows=1,
        delay_seconds=1.0,
    ),
]

REGISTRY: dict[str, SourceSpec] = {s.key: s for s in _SPECS}

# Lane-A contributed sources (agent: claude-fetch-A), kept in a separate module so the
# two Claude agents don't clobber this file. This block runs BEFORE the lane-B block so
# that, with setdefault semantics, Lane A wins any key collision (as agreed in
# reports/data_fetch/COORDINATION.md).
try:
    from .registry_laneA import LANE_A_SPECS as _LANE_A_SPECS

    for _s in _LANE_A_SPECS:
        REGISTRY.setdefault(_s.key, _s)
except Exception as _exc:  # pragma: no cover - defensive
    import sys as _sys

    print(f"[registry] lane-A contrib not loaded: {_exc}", file=_sys.stderr)

# Lane-B contributed sources (agent: claude-fetch-2), kept in a separate module so
# concurrent agents don't clobber this file. Idempotent + additive: lane A wins any
# key collision (setdefault), and a contrib import error never breaks the core registry.
try:
    from .registry_laneB import LANE_B_SPECS as _LANE_B_SPECS

    for _s in _LANE_B_SPECS:
        REGISTRY.setdefault(_s.key, _s)
except Exception as _exc:  # pragma: no cover - defensive
    import sys as _sys

    print(f"[registry] lane-B contrib not loaded: {_exc}", file=_sys.stderr)


def get_spec(key: str) -> SourceSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown source '{key}'. known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def validate_registry() -> list[str]:
    """Return a list of problems with the registry (empty == valid)."""
    problems: list[str] = []
    seen = set()
    for key, spec in REGISTRY.items():
        if spec.key != key:
            problems.append(f"{key}: spec.key mismatch ({spec.key})")
        if spec.key in seen:
            problems.append(f"{key}: duplicate key")
        seen.add(spec.key)
        for fieldname in ("title", "type", "endpoint", "description", "adapter"):
            if not getattr(spec, fieldname):
                problems.append(f"{key}: missing required field '{fieldname}'")
        if ":" not in spec.adapter:
            problems.append(f"{key}: adapter must be 'module:Class', got '{spec.adapter}'")
        if not spec.required_columns:
            problems.append(f"{key}: no required_columns declared")
        if spec.wraps_trusted and spec.needs_credentials:
            problems.append(f"{key}: wraps_trusted and needs_credentials are mutually exclusive")
    return problems
