"""Target catalog — the 30-candidate funnel for landing 15 NEW integrated sources.

This is the *discovery input* for the fetcher buildout. Each candidate is an
open-source, timestamped, labeled dataset that is NOT already in `registry.py`. The
funnel is deliberately oversized (~30) because some candidates will fail the bar:

    PROMOTION GATE (a candidate becomes a registry source only if):
      - reachable (endpoint responds), and
      - yields >= MIN_LABELED_ROWS rows, each with a real measured/observed value
        (a "label") and a timestamp.

A candidate that fails the gate is recorded as `rejected` (with the reason) in
`reports/data_fetch/agent_source_intents.json` and in `fetch_learnings.jsonl` — a
NULL result is a kept learning, not a silent drop.

Run:
    python -m ops.data_fetch.target_catalog list
    python -m ops.data_fetch.target_catalog lane A
    python -m ops.data_fetch.target_catalog probe --candidate open_meteo_marine
    python -m ops.data_fetch.target_catalog probe --lane A        # probe a whole lane

`probe` only does a cheap HEAD/small-GET reachability check — it does NOT fetch. The
real >=50k gate is enforced by the adapter's `validate()` once a source is built.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict

MIN_LABELED_ROWS = 50_000


@dataclass(frozen=True)
class Candidate:
    key: str
    title: str
    modality: str          # what kind of signal (ocean_color, met, bacteria, ...)
    type: str              # adapter family: tabular_api | erddap | bulk_text | ckan | netcdf_archive
    endpoint: str
    probe_url: str         # a cheap URL to test reachability (small response)
    expected_rows: str     # honest estimate / basis for >=50k confidence
    lane: str              # "A" | "B" | "pool"
    relevance: str         # why it matters to the lab's models
    label_columns: list[str] = field(default_factory=list)  # the measured value(s)
    date_basis: str = ""   # how each row is timestamped
    notes: str = ""


# ── the funnel ────────────────────────────────────────────────────────────────
# Lane A is owned by claude-fetch-A (launched first); Lane B by claude-fetch-B.
# `pool` candidates are backups either Claude may claim if a lane source fails.
CANDIDATES: list[Candidate] = [
    # ===== Lane A ===============================================================
    Candidate(
        key="open_meteo_marine",
        title="Open-Meteo Marine hourly archive (waves/swell/SST) at CA beach grid",
        modality="wave_swell_sst",
        type="tabular_api",
        endpoint="https://marine-api.open-meteo.com/v1/marine",
        probe_url="https://marine-api.open-meteo.com/v1/marine?latitude=36.75&longitude=-122.0&hourly=wave_height&start_date=2024-01-01&end_date=2024-01-02",
        expected_rows="1 cell x 1yr hourly = 8.7k; ~12 CA cells x 5yr >> 500k",
        lane="A",
        relevance="hourly wave/swell/SST driver at every CA beach cell — denser than the single CDIP buoy; surf/mixing driver for bacteria + HAB transport.",
        label_columns=["wave_height", "wave_period", "wave_direction", "swell_wave_height", "sea_surface_temperature"],
        date_basis="hourly UTC `time` per grid cell",
        notes="Same vendor family as the existing rainfall_openmeteo source; free archive, no creds.",
    ),
    Candidate(
        key="ndbc_stdmet",
        title="NDBC historical standard meteorological (all CA buoys, hourly)",
        modality="met_ocean",
        type="bulk_text",
        endpoint="https://www.ndbc.noaa.gov/data/historical/stdmet/",
        probe_url="https://www.ndbc.noaa.gov/data/historical/stdmet/",
        expected_rows="~10 CA buoys x ~10-30yr x hourly = millions",
        lane="A",
        relevance="wind/pressure/air+water temp/wave at offshore buoys — primary physical drivers; distinct from CDIP (which we only have for 46042 waves).",
        label_columns=["WSPD", "GST", "WVHT", "DPD", "APD", "PRES", "ATMP", "WTMP", "DEWP"],
        date_basis="YYYY MM DD hh mm columns -> UTC timestamp",
        notes="Static yearly .txt.gz files per station; extremely reliable, no creds.",
    ),
    Candidate(
        key="modis_chl",
        title="MODIS-Aqua chlorophyll-a daily (erdMH1chla1day) at CA HAB points",
        modality="ocean_color",
        type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla1day",
        probe_url="https://coastwatch.pfeg.noaa.gov/erddap/info/erdMH1chla1day/index.json",
        expected_rows="2003->present daily x multiple CA pier/HAB points >> 50k",
        lane="A",
        relevance="longer chlorophyll record than VIIRS (2003 vs 2015) — bloom precursor for domoic-acid/HAB model; pairs with c_harm.",
        label_columns=["chlorophyll"],
        date_basis="daily `time` per lat/lon point",
        notes="ERDDAP griddap; reuse the existing viirs_chl adapter pattern (per-year chunks, .json query).",
    ),
    Candidate(
        key="oisst",
        title="NOAA OISST v2.1 daily SST (ncdcOisst21Agg) at CA points",
        modality="sst",
        type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180",
        probe_url="https://coastwatch.pfeg.noaa.gov/erddap/info/ncdcOisst21Agg_LonPM180/index.json",
        expected_rows="1981->present daily x several CA points >> 50k",
        lane="A",
        relevance="gap-free daily SST back to 1981 + anomaly — broader temporal anchor than MUR (2002); upwelling/marine-heatwave signal.",
        label_columns=["sst", "anom"],
        date_basis="daily `time` per lat/lon point",
        notes="ERDDAP griddap; same chunking as modis_chl/viirs_chl.",
    ),
    Candidate(
        key="ghcnd",
        title="GHCN-Daily climate stations (California subset)",
        modality="land_weather",
        type="bulk_text",
        endpoint="https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/",
        probe_url="https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt",
        expected_rows="~1000 CA stations x decades daily (PRCP/TMAX/TMIN) = millions",
        lane="A",
        relevance="dense ground-truth precip/temp network — finer than the gridded rainfall source; first-flush + antecedent-dryness drivers at station resolution.",
        label_columns=["PRCP", "TMAX", "TMIN", "SNOW"],
        date_basis="one row per station-day (.dly fixed-width or .csv by_station)",
        notes="by_station CSVs avoid the giant superghcnd; filter station list to CA (state field).",
    ),
    Candidate(
        key="cdec",
        title="CA Data Exchange Center (CDEC) hydrology/precip/reservoir sensors",
        modality="hydrology",
        type="tabular_api",
        endpoint="https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet",
        probe_url="https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=SHA&SensorNums=15&dur_code=D&Start=2024-01-01&End=2024-01-10",
        expected_rows="100s of CA stations x daily/hourly x many sensors = millions",
        lane="A",
        relevance="CA-operated precip/flow/reservoir/snow network feeding coastal watersheds — runoff driver complementary to USGS discharge.",
        label_columns=["value"],
        date_basis="per station-sensor-time CSV row",
        notes="CSVDataServlet returns CSV; chunk by station+sensor+date window.",
    ),
    Candidate(
        key="usgs_iv_wq",
        title="USGS NWIS instantaneous water-quality (turbidity/temp/SpC/gage height)",
        modality="water_quality",
        type="tabular_api",
        endpoint="https://waterservices.usgs.gov/nwis/iv/",
        probe_url="https://waterservices.usgs.gov/nwis/iv/?format=json&stateCd=ca&parameterCd=00010&startDT=2024-01-01&endDT=2024-01-02&siteType=ST",
        expected_rows="CA coastal gauges x 15-min IV x params = millions",
        lane="A",
        relevance="continuous in-stream turbidity/temp/conductance at gauges near beaches — finer contamination proxy than daily discharge; distinct param set.",
        label_columns=["value"],
        date_basis="instantaneous-value dateTime per site-parameter",
        notes="Reuse discharge_usgs idiom but iv (instantaneous) + WQ params 00010/00095/00300/63680/00065.",
    ),
    Candidate(
        key="obis",
        title="OBIS marine biodiversity occurrences (CA coast bbox)",
        modality="biodiversity",
        type="tabular_api",
        endpoint="https://api.obis.org/v3/occurrence",
        probe_url="https://api.obis.org/v3/occurrence?geometry=POLYGON((-124%2032,-117%2032,-117%2042,-124%2042,-124%2032))&size=1",
        expected_rows="CA coastal bbox -> hundreds of thousands of occurrences",
        lane="A",
        relevance="species occurrence (incl. HAB taxa, indicator organisms) timestamped+geolocated — ecological corpus + potential HAB co-occurrence labels.",
        label_columns=["scientificName", "individualCount"],
        date_basis="eventDate per occurrence record",
        notes="Paged JSON API (size+after); filter to bbox + non-null eventDate.",
    ),

    # ===== Lane B ===============================================================
    Candidate(
        key="noaa_coops_met",
        title="NOAA CO-OPS meteorology (wind/air-temp/water-temp) at CA stations",
        modality="met_ocean",
        type="tabular_api",
        endpoint="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        probe_url="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_temperature&application=mbal&begin_date=20240101&end_date=20240107&datum=MLLW&station=9413450&time_zone=gmt&units=metric&format=json",
        expected_rows="CA CO-OPS stations x 6-min/hourly x products = millions",
        lane="B",
        relevance="station wind/air-temp/water-temp at the coast — pairs with the existing tide source; nearshore mixing/temperature drivers.",
        label_columns=["v"],
        date_basis="per station-product time row",
        notes="Reuse coops_tide adapter pattern; products: water_temperature, air_temperature, wind. 31-day windows.",
    ),
    Candidate(
        key="cimis",
        title="CIMIS agricultural weather stations (ET/soil-temp/solar, CA)",
        modality="ag_weather",
        type="tabular_api",
        endpoint="https://et.water.ca.gov/api/data",
        probe_url="https://et.water.ca.gov/api/station",
        expected_rows="~150 active stations x daily/hourly x many items = millions",
        lane="B",
        relevance="evapotranspiration/soil-temp/solar over ag land — antecedent-moisture + ag-runoff driver for bacteria and nutrient loading.",
        label_columns=["DayAirTmpAvg", "DayEto", "DaySoilTmpAvg", "DaySolRadAvg"],
        date_basis="per station-day JSON record",
        notes="Needs a free appKey (CIMIS_APP_KEY env). Station list endpoint is keyless for probe.",
    ),
    Candidate(
        key="gbif",
        title="GBIF species occurrences (CA coastal)",
        modality="biodiversity",
        type="tabular_api",
        endpoint="https://api.gbif.org/v1/occurrence/search",
        probe_url="https://api.gbif.org/v1/occurrence/search?decimalLatitude=32,42&decimalLongitude=-124,-117&limit=1",
        expected_rows="CA coastal bbox -> millions of occurrences",
        lane="B",
        relevance="broad biodiversity corpus incl. phytoplankton/indicator taxa; complements OBIS with terrestrial-adjacent + research-grade records.",
        label_columns=["species", "individualCount"],
        date_basis="eventDate per occurrence",
        notes="Paged JSON (limit+offset, cap 100k via offset; use facets/year chunks for more).",
    ),
    Candidate(
        key="ceden",
        title="CEDEN CA environmental water-quality results (data.ca.gov)",
        modality="water_quality",
        type="ckan",
        endpoint="https://data.ca.gov/api/3/action/datastore_search",
        probe_url="https://data.ca.gov/api/3/action/package_search?q=CEDEN%20water%20quality&rows=1",
        expected_rows="CEDEN chemistry/toxicity resources = millions of result rows",
        lane="B",
        relevance="statewide ambient water-quality chemistry/toxicity (nutrients, metals, pesticides) — contamination drivers beyond WQP coverage.",
        label_columns=["Result", "Analyte"],
        date_basis="SampleDate per result row",
        notes="Reuse the ckan.py adapter (datastore_search paging). Pick the right resource_id from package_search.",
    ),
    Candidate(
        key="gridmet",
        title="gridMET daily gridded meteorology (CA cells)",
        modality="land_weather",
        type="erddap",
        endpoint="https://www.northwestknowledge.net/metdata/data/",
        probe_url="https://thredds.northwestknowledge.net:8443/thredds/catalog/MET/catalog.html",
        expected_rows="CA cells x 1979->present daily x vars = millions",
        lane="B",
        relevance="4km gridded precip/temp/humidity/wind/VPD — consistent gridded driver stack; complements station GHCN-D.",
        label_columns=["precipitation_amount", "air_temperature", "relative_humidity"],
        date_basis="daily per grid cell",
        notes="THREDDS/OPeNDAP NetCDF; subset by lat/lon. Heavier — may need xarray/netCDF.",
    ),
    Candidate(
        key="calcofi",
        title="CalCOFI hydrographic bottle/cast data (CA Current)",
        modality="ocean_profile",
        type="erddap",
        endpoint="https://erddap.cencoos.org/erddap/tabledap/",
        probe_url="https://coastwatch.pfeg.noaa.gov/erddap/search/index.json?searchFor=calcofi",
        expected_rows="decades of bottle samples x depths = hundreds of thousands",
        lane="B",
        relevance="long temperature/salinity/oxygen/nutrient profile record in the CA Current — deep-water context for HAB + upwelling.",
        label_columns=["temperature", "salinity", "oxygen", "nitrate"],
        date_basis="cast time per bottle/depth row",
        notes="Find the live CalCOFI ERDDAP tabledap id via ERDDAP search first (logged for both lanes).",
    ),
    Candidate(
        key="epa_aqs",
        title="EPA AQS air-quality daily (PM2.5/ozone, CA)",
        modality="air_quality",
        type="tabular_api",
        endpoint="https://aqs.epa.gov/data/api",
        probe_url="https://aqs.epa.gov/data/api/list/states?email=test@aqs.api&key=test",
        expected_rows="CA monitors x daily x params = millions",
        lane="B",
        relevance="coastal air quality corpus; weaker model relevance — keep only if it clears 50k easily and adds a genuinely new modality.",
        label_columns=["arithmetic_mean", "aqi"],
        date_basis="date_local per monitor-parameter",
        notes="Needs a free email+key. Lower priority — value-gate carefully (corpus, not a proven driver).",
    ),
    Candidate(
        key="hycom",
        title="HYCOM global ocean reanalysis (currents/temp/salinity) CA subset",
        modality="ocean_3d",
        type="erddap",
        endpoint="https://tds.hycom.org/thredds/",
        probe_url="https://coastwatch.pfeg.noaa.gov/erddap/search/index.json?searchFor=hycom",
        expected_rows="CA box x daily x depths x vars = millions",
        lane="B",
        relevance="3D currents/T/S reanalysis — transport driver for HAB advection; complements ROMS with a global-consistent product.",
        label_columns=["water_temp", "salinity", "water_u", "water_v"],
        date_basis="daily per lat/lon/depth",
        notes="Heavy NetCDF; prefer an ERDDAP mirror if available (search first).",
    ),

    # ===== Shared backup pool (claim before taking) =============================
    Candidate(
        key="argo_floats", title="Argo profiling floats (T/S profiles, CA Current)",
        modality="ocean_profile", type="erddap",
        endpoint="https://www.ifremer.fr/erddap/tabledap/ArgoFloats",
        probe_url="https://www.ifremer.fr/erddap/info/ArgoFloats/index.json",
        expected_rows="CA-Current bbox profiles x depths >> 50k", lane="pool",
        relevance="autonomous T/S/oxygen profiles — open-ocean context for upwelling/HAB.",
        label_columns=["temp", "psal"], date_basis="profile time per level",
    ),
    Candidate(
        key="cdph_biotoxin", title="CDPH shellfish biotoxin / domoic-acid timeseries",
        modality="hab_label", type="tabular_api",
        endpoint="https://www.cdph.ca.gov (biotoxin reports / data.ca.gov mirror)",
        probe_url="https://data.ca.gov/api/3/action/package_search?q=biotoxin&rows=1",
        expected_rows="may be <50k — verify; valuable as HAB labels even if small", lane="pool",
        relevance="direct domoic-acid/PSP labels — the HAB target itself; check row count honestly.",
        label_columns=["domoic_acid", "result"], date_basis="sample date",
    ),
    Candidate(
        key="noaa_coops_currents", title="NOAA CO-OPS observed currents (CA stations)",
        modality="currents", type="tabular_api",
        endpoint="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
        probe_url="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=currents&application=mbal&begin_date=20240101&end_date=20240102&station=PCT0101&time_zone=gmt&units=metric&format=json",
        expected_rows="current stations x 6-min x bins — verify >=50k", lane="pool",
        relevance="measured nearshore currents — transport driver.",
        label_columns=["s", "d"], date_basis="per station-time bin",
    ),
    Candidate(
        key="smap_sss", title="SMAP sea-surface salinity (CA coast)",
        modality="salinity", type="erddap",
        endpoint="https://coastwatch.pfeg.noaa.gov/erddap (SMAP SSS)",
        probe_url="https://coastwatch.pfeg.noaa.gov/erddap/search/index.json?searchFor=SMAP+salinity",
        expected_rows="2015->present x CA points — verify", lane="pool",
        relevance="freshwater-plume/runoff signal at the surface.",
        label_columns=["sss"], date_basis="daily/8-day per point",
    ),
    Candidate(
        key="usgs_gw", title="USGS NWIS groundwater levels (CA coastal)",
        modality="hydrology", type="tabular_api",
        endpoint="https://waterservices.usgs.gov/nwis/gwlevels/",
        probe_url="https://waterservices.usgs.gov/nwis/gwlevels/?format=json&stateCd=ca&startDT=2020-01-01&endDT=2020-02-01",
        expected_rows="CA wells x periodic levels — verify >=50k", lane="pool",
        relevance="baseflow/seawater-intrusion context; weaker — verify before committing.",
        label_columns=["value"], date_basis="measurement date per site",
    ),
    Candidate(
        key="inaturalist", title="iNaturalist research-grade marine observations (CA coast)",
        modality="biodiversity", type="tabular_api",
        endpoint="https://api.inaturalist.org/v1/observations",
        probe_url="https://api.inaturalist.org/v1/observations?nelat=42&nelng=-117&swlat=32&swlng=-124&quality_grade=research&per_page=1",
        expected_rows="CA coastal research-grade obs = hundreds of thousands", lane="pool",
        relevance="citizen-science occurrences incl. HAB/wildlife-mortality events.",
        label_columns=["taxon_name"], date_basis="observed_on per record",
    ),
    Candidate(
        key="calhabmap_piers", title="CalHABMAP/SCCOOS full pier timeseries (all piers)",
        modality="hab_label", type="erddap",
        endpoint="https://erddap.sccoos.org/erddap/tabledap/",
        probe_url="https://erddap.sccoos.org/erddap/search/index.json?searchFor=HABs",
        expected_rows="all CA piers x weekly x params — verify vs existing habmap_cdph", lane="pool",
        relevance="expands the existing habmap_cdph beyond current pier coverage — HAB labels + chl/nutrients.",
        label_columns=["Chlorophyll", "Pseudo_nitzschia", "Domoic_Acid"], date_basis="sample time per pier",
        notes="Must beat the existing habmap_cdph (don't re-fetch the same rows).",
    ),
    Candidate(
        key="prism", title="PRISM daily precip/temp (CA cells)",
        modality="land_weather", type="bulk_text",
        endpoint="https://www.prism.oregonstate.edu/",
        probe_url="https://www.prism.oregonstate.edu/",
        expected_rows="CA cells x daily — verify vs gridmet/GHCN-D overlap", lane="pool",
        relevance="alt gridded precip; only if it adds over gridmet+GHCN-D (avoid redundancy).",
        label_columns=["ppt", "tmean"], date_basis="daily per cell",
    ),
]

BY_KEY = {c.key: c for c in CANDIDATES}


def lane(name: str) -> list[Candidate]:
    return [c for c in CANDIDATES if c.lane.upper() == name.upper()]


def probe_one(c: Candidate, timeout: int = 20) -> dict:
    """Cheap reachability check — NOT a fetch. Returns status dict."""
    import urllib.request
    out = {"key": c.key, "probe_url": c.probe_url, "lane": c.lane}
    try:
        req = urllib.request.Request(c.probe_url, headers={"User-Agent": "mbal-datafetch/0.1"})
        r = urllib.request.urlopen(req, timeout=timeout)
        body = r.read(2000)
        out.update(reachable=True, http_status=getattr(r, "status", 200), bytes_seen=len(body))
    except Exception as exc:  # noqa: BLE001
        out.update(reachable=False, error=f"{type(exc).__name__}: {str(exc)[:120]}")
    return out


def _cmd_list(args) -> int:
    rows = CANDIDATES if not args.lane else lane(args.lane)
    print(f"{'lane':4} {'key':22} {'type':14} {'modality':16} title")
    print("-" * 110)
    for c in rows:
        print(f"{c.lane:4} {c.key:22} {c.type:14} {c.modality:16} {c.title}")
    print(f"\n{len(rows)} candidates · MIN_LABELED_ROWS={MIN_LABELED_ROWS:,} · target=15 landed")
    return 0


def _cmd_lane(args) -> int:
    for c in lane(args.name):
        print(json.dumps(asdict(c), indent=2))
    return 0


def _cmd_probe(args) -> int:
    if args.candidate:
        targets = [BY_KEY[args.candidate]]
    elif args.lane:
        targets = lane(args.lane)
    else:
        targets = CANDIDATES
    results = [probe_one(c) for c in targets]
    for r in results:
        flag = "OK " if r.get("reachable") else "DEAD"
        print(f"[{flag}] {r['key']:22} {r.get('http_status', r.get('error',''))}")
    print(json.dumps(results, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="target_catalog", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list"); pl.add_argument("--lane", default=None); pl.set_defaults(func=_cmd_list)
    pa = sub.add_parser("lane"); pa.add_argument("name"); pa.set_defaults(func=_cmd_lane)
    pp = sub.add_parser("probe")
    pp.add_argument("--candidate", default=None)
    pp.add_argument("--lane", default=None)
    pp.set_defaults(func=_cmd_probe)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
