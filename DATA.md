# Data Provenance & License

**The Apache-2.0 LICENSE covers the code only.** Third-party data retrieved or redistributed
by this project remains under its providers' terms. Verify each source's terms before
redistribution or commercial use.

## Sources

| Dataset | Provider | Access | Terms |
|---|---|---|---|
| M1 / M2 mooring time series | MBARI | OPeNDAP `https://dods.mbari.org/opendap/...` | MBARI data use terms |
| Sea-surface temperature (MUR SST) | NASA JPL PO.DAAC | NOAA MUR cache | NASA open data |
| Buoy waves/wind (NDBC 46042) | NOAA NDBC | `ndbc.noaa.gov` | U.S. Gov public domain |
| Tides / water level (CO-OPS) | NOAA CO-OPS | `tidesandcurrents.noaa.gov` | U.S. Gov public domain |
| Upwelling indices | NOAA | public feeds | U.S. Gov public domain |
| CA beach water-quality (BeachWatch) | CA SWRCB / CDPH | `waterqualitydata.us` / state records | CA public records |
| CalHABMAP pier domoic acid (particulate DA) | CalHABMAP / SCCOOS | ERDDAP `erddap.sccoos.org` | public research data (shipped: `data/external_curated/habmap_cdph/habmap_cdph.parquet`, 304 KB) |

## How the pipeline reads data

Code never hardcodes a path or cloud project. Configure via environment variables:

- `MBAL_PROJECT_ROOT` — repo / working root
- `MBAL_SOURCE_PARQUET` — curated source panel
- `MBAL_CACHE_DIR` — feature/driver cache (`nn_cache`)
- `MBAL_LAKEHOUSE_DIR` — lakehouse outputs (`lakehouse`)
- `MBAL_GCP_PROJECT` — your own GCP project, if pulling from BigQuery

## Curated data + results release

The curated panel, lakehouse gold tables, and model results are **not** committed to git
(too large, and provider-licensed). They are distributed as a tagged **data release**
(GitHub Release asset or a Zenodo DOI). Download it and point `MBAL_SOURCE_PARQUET` /
`MBAL_LAKEHOUSE_DIR` at your local copy. See the repo's Releases page.

To reproduce from scratch instead, use the fetchers under `research/` and the build steps
in `README.md` against your own credentials/quota.
