"""Lane-A contributed source specs (agent: claude-fetch-A).

Mirrors the Lane-B pattern: Lane A keeps its new specs in this separate module so the
two Claude agents never clobber `registry.py`'s `_SPECS`. `registry.py` merges these
via setdefault. Lane A is intended to WIN any key collision, so the laneA merge must
run BEFORE the laneB merge in registry.py (see the note left for claude-fetch-B in
reports/data_fetch/fetch_learnings.jsonl).

Each spec here has a landed/validated curated parquet under data/external_curated/.
"""
from __future__ import annotations

from .registry import SourceSpec

LANE_A_SPECS: list[SourceSpec] = [
    SourceSpec(
        key="cdec",
        title="California Data Exchange Center (CDEC) daily hydrology sensors",
        type="tabular_api",
        endpoint="https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet",
        description="CA DWR operational network: daily reservoir storage / incremental "
        "precipitation / reservoir outflow at ~15 long-record stations across California "
        "basins (Shasta, Oroville, Folsom, Trinity, San Luis, Pine Flat, Cachuma, San "
        "Antonio, …). Runoff / antecedent-hydrology driver complementary to the existing "
        "USGS daily discharge source. Long format: one row per station-sensor-date.",
        adapter="ops.data_fetch.adapters.cdec:CdecAdapter",
        priority=7,
        spatial="~15 California CDEC hydrology stations (statewide basins incl. Central Coast)",
        temporal="1990-01-01 → present (daily)",
        rate_limit="CDEC CSVDataServlet public (one request per station-sensor)",
        required_columns=["station_id", "date", "sensor_label", "value", "units"],
        dedup_keys=["station_id", "date", "sensor_num"],
        date_column="date",
        spatial_columns=["station_id", "sensor_label"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="ghcnd",
        title="GHCN-Daily climate stations (California subset)",
        type="tabular_api",
        endpoint="https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/",
        description="NCEI Global Historical Climatology Network — Daily. Ground-truth daily "
        "precipitation and temperature at ~12 long-record California stations (one ~380k-row "
        "gzip CSV per station). Station-resolution first-flush / antecedent-dryness driver, "
        "finer than the existing gridded rainfall source. QC-failed values dropped; values kept "
        "raw + converted to SI (mm, °C). Long format: one row per station-date-element.",
        adapter="ops.data_fetch.adapters.ghcnd:GhcndAdapter",
        priority=6,
        spatial="~12 California GHCN-D stations (SF, LAX, San Diego, Santa Maria, Oakland, "
        "Sacramento, Fresno, Santa Barbara, Eureka/Arcata, Bakersfield)",
        temporal="station record start (1940s for airports) → present",
        rate_limit="NCEI public file server (one file per station)",
        required_columns=["station_id", "date", "element", "value", "units"],
        dedup_keys=["station_id", "date", "element"],
        date_column="date",
        spatial_columns=["station_id", "element"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.3,
    ),
    SourceSpec(
        key="obis_hab",
        title="OBIS Pseudo-nitzschia (domoic-acid HAB) occurrences — 1902-2009 slice",
        type="tabular_api",
        endpoint="https://api.obis.org/v3/occurrence",
        description="OBIS occurrence records for Pseudo-nitzschia — the domoic-acid-producing "
        "diatom that is the lab's stated marine-HAB target (~77k timestamped, geolocated "
        "occurrences). A labeled biological-occurrence corpus for the domoic-acid forecasting "
        "frontier. NOTE: this is a DELIBERATE temporal SLICE covering 1902-2009 — the 2010+ "
        "decades are data-heavy and OBIS API rate-limiting made them slow to land in-session; "
        "they are RESUMABLE (extend DECADES in the adapter and re-run `fetch`). Not a complete "
        "census; documented as such so it never masquerades as full coverage. The adapter also "
        "generalizes to additional HAB genera if expanded later.",
        adapter="ops.data_fetch.adapters.obis_hab:ObisHabAdapter",
        priority=8,
        spatial="Global OBIS records for Pseudo-nitzschia (incl. CA Current)",
        temporal="1902-2009 slice (recent decades resumable; OBIS API rate-limited)",
        rate_limit="OBIS API public (cursor-paged per genus)",
        required_columns=["time", "scientificName", "latitude", "longitude"],
        dedup_keys=["id"],
        date_column="time",
        spatial_columns=["query_genus", "scientificName"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="obis_marine_mammals",
        title="OBIS marine-mammal occurrences + strandings — US West Coast (Project Leviathan labels)",
        type="tabular_api",
        endpoint="https://api.obis.org/v3/occurrence",
        description="OBIS cetacean + pinniped (Cetacea / Otariidae / Phocidae) occurrence records on "
        "the US West Coast (California Current, lon -130..-116 / lat 32..49) — both live sightings and "
        "dead/stranded specimens, each timestamped + geolocated. The MORTALITY/STRANDING LABEL side of "
        "the whale-mortality investigation (the driver side — domoic acid via habmap_cdph/c_harm, ocean "
        "conditions via mur_sst/roms/wcofs — already exists). Keeps basisOfRecord/datasetName/"
        "occurrenceStatus + an is_stranding_candidate flag so the modeling lane can separate strandings "
        "(the mortality signal) from sightings. Probe 2026-06: ~159k Cetacea + ~13k Otariidae + ~18k "
        "Phocidae on the West Coast; includes the gray-whale UME species and the CA sea lion of the "
        "published pDA-stranding link.",
        adapter="ops.data_fetch.adapters.obis_marine_mammals:ObisMarineMammalsAdapter",
        priority=9,
        spatial="US West Coast polygon (CA/OR/WA; California Current)",
        temporal="1950-2029 (decade-windowed, resumable)",
        rate_limit="OBIS API public (cursor-paged per taxon)",
        required_columns=["time", "scientificName", "latitude", "longitude"],
        dedup_keys=["id"],
        date_column="time",
        spatial_columns=["query_taxon", "scientificName", "family"],
        value_bounds={},
        min_rows=20_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="inat_mammal_mortality",
        title="iNaturalist marine-mammal MORTALITY (dead-annotation) — CA coast (Project Leviathan)",
        type="tabular_api",
        endpoint="https://api.inaturalist.org/v1/observations",
        description="Community-verified DEAD marine-mammal observations (iNaturalist 'Alive or Dead' "
        "annotation: attribute 17, value 19=Dead) for Cetacea + Otariidae + Phocidae on the CA coast, "
        "each dated/geolocated/species-tagged, with the per-year observation EFFORT total attached so "
        "the effort-normalized dead-FRACTION (dead/effort) is computable — that fraction reconstructs "
        "the 2019-2023 gray-whale UME from open data. The open MORTALITY label for the whale-death "
        "investigation (the per-animal NOAA cause-of-death DB is request-walled). Honest caveats: "
        "voluntary annotation + observer-effort/COVID confounds; a citizen-science PROXY, not a census. "
        "Probe 2026-06: ~1157 dead cetaceans + pinniped dead on the CA coast.",
        adapter="ops.data_fetch.adapters.inaturalist_mammal_mortality:INatMammalMortalityAdapter",
        priority=9,
        spatial="California coast bbox (32-42N, -124..-117)",
        temporal="2010-2026 (per-year, deep-paged)",
        rate_limit="iNaturalist API public (id_above deep-paging)",
        required_columns=["time", "taxon_name", "latitude", "longitude", "is_dead"],
        dedup_keys=["observation_id"],
        date_column="time",
        spatial_columns=["query_taxon", "taxon_name", "place_guess"],
        value_bounds={},
        min_rows=1000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="coops_met",
        title="NOAA CO-OPS meteorology (water/air temp, wind) at California stations",
        type="tabular_api",
        endpoint="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        description="Hourly water temperature, air temperature, and wind at ~12 California "
        "CO-OPS stations — nearshore mixing/temperature drivers complementary to the existing "
        "tide water-level source (distinct product set). Long format: station-product-time.",
        adapter="ops.data_fetch.adapters.coops_met:CoopsMetAdapter",
        priority=8,
        spatial="~12 California CO-OPS stations (SF, LA, San Diego, Monterey, Humboldt, …)",
        temporal="2010 → present (hourly)",
        rate_limit="NOAA CO-OPS API public (yearly hourly per station-product)",
        required_columns=["station_id", "time", "product", "value"],
        dedup_keys=["station_id", "product", "time"],
        date_column="time",
        spatial_columns=["station_id", "product"],
        value_bounds={"value": (-50.0, 100.0)},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
    SourceSpec(
        key="inaturalist_ca",
        title="iNaturalist research-grade observations, California coast",
        type="tabular_api",
        endpoint="https://api.inaturalist.org/v1/observations",
        description="Community-verified (research-grade) species occurrences in the CA coastal "
        "bbox — a labeled, timestamped, geolocated citizen-science biodiversity corpus. NOTE: a "
        "DELIBERATE representative slice (per-year page cap), not a full census of the 13.5M-record "
        "universe; documented as such (not bounded_sample because it is a self-contained >=50k "
        "labeled dataset, but coverage is intentionally partial — see adapter).",
        adapter="ops.data_fetch.adapters.inaturalist:INaturalistAdapter",
        priority=11,
        spatial="California coastal bbox (32–42N, 117–124W)",
        temporal="2015 → present",
        rate_limit="iNaturalist API public (id_above paging, capped per year)",
        required_columns=["observation_id", "time", "taxon_name", "latitude", "longitude"],
        dedup_keys=["observation_id"],
        date_column="time",
        spatial_columns=["iconic_taxon", "taxon_rank"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.5,
    ),
    SourceSpec(
        key="dwr_groundwater",
        title="DWR periodic groundwater-level measurements (statewide CA)",
        type="ckan",
        endpoint="https://data.ca.gov/api/3/action/datastore_search (Periodic GWL Measurements)",
        description="California DWR periodic groundwater-level measurements (~6.25M timestamped "
        "well records statewide): groundwater elevation (gwe) and depth-to-water (gse_gwe). A "
        "genuinely new modality for the lab — no existing source covers groundwater — relevant "
        "as a baseflow / seawater-intrusion context driver for coastal water quality.",
        adapter="ops.data_fetch.adapters.dwr_groundwater:DwrGroundwaterAdapter",
        priority=9,
        spatial="California statewide groundwater wells",
        temporal="multi-decade → present (per msmt_date)",
        rate_limit="data.ca.gov CKAN (offset-paged, full coverage)",
        required_columns=["site_code", "msmt_date", "gwe"],
        dedup_keys=["site_code", "msmt_date"],
        date_column="msmt_date",
        spatial_columns=["county_name", "basin_code"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
    SourceSpec(
        key="ca_snow_swc",
        title="California monthly snow water content (snow courses, statewide)",
        type="ckan",
        endpoint="https://data.ca.gov/api/3/action/datastore_search (DWR Monthly Snow Water Content)",
        description="CA DWR monthly snow-water-equivalent at mountain snow courses since 1930 "
        "(~191k measurements). New snowpack modality — snowmelt drives spring streamflow timing "
        "that conditions coastal runoff/water-quality. Single complete CKAN resource.",
        adapter="ops.data_fetch.adapters.ca_snow:CaSnowAdapter",
        priority=10,
        spatial="California mountain snow courses",
        temporal="1930 → present (monthly)",
        rate_limit="data.ca.gov CKAN (offset-paged, full coverage)",
        required_columns=["station_id", "date", "value"],
        dedup_keys=["station_id", "date", "sensor_num"],
        date_column="date",
        spatial_columns=["station_id", "sensor_type"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
    SourceSpec(
        key="dwr_gw_continuous",
        title="DWR continuous groundwater-level daily measurements (statewide CA)",
        type="ckan",
        endpoint="https://data.ca.gov/api/3/action/datastore_search (Continuous GWL — Daily)",
        description="CA DWR continuous groundwater-level daily records (~3.13M) from automated "
        "sensor loggers with QC flags — a different network and cadence than the periodic manual "
        "measurements (dwr_groundwater). Groundwater surface elevation / depth-to-water; baseflow "
        "and seawater-intrusion context driver. Full native coverage.",
        adapter="ops.data_fetch.adapters.dwr_gw_continuous:DwrGwContinuousAdapter",
        priority=10,
        spatial="California statewide continuous groundwater logger stations",
        temporal="multi-year → present (daily)",
        rate_limit="data.ca.gov CKAN (offset-paged, full coverage)",
        required_columns=["station_id", "date", "wse"],
        dedup_keys=["station_id", "date"],
        date_column="date",
        spatial_columns=["station_id"],
        value_bounds={},
        min_rows=50_000,
        delay_seconds=0.2,
    ),
]
