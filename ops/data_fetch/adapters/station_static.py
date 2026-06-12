"""Station-level static physical covariates for beach bacteria nowcasting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..core import Adapter, PROJECT_ROOT

STATION_GEO = PROJECT_ROOT / "reports" / "station_geo.parquet"
GAUGE_MAP_PRIMARY = PROJECT_ROOT / "bacteria_results" / "discharge" / "station_gauge_map.parquet"
GAUGE_MAP_FALLBACK = PROJECT_ROOT / "reports" / "station_usgs_mapping.parquet"
ECHO_DMR = PROJECT_ROOT / "data" / "external_curated" / "echo_dmr" / "echo_dmr.parquet"
CIWQS_SSO = PROJECT_ROOT / "data" / "external_curated" / "ciwqs_sso" / "ciwqs_sso.parquet"
STATIC_CATALOG = PROJECT_ROOT / "data" / "external_curated" / "ccap_nhd_nlcd" / "ccap_nhd_nlcd.parquet"
EARTH_KM = 6371.0088


class StationStaticFeaturesAdapter(Adapter):
    """Materialize additive per-station physical source/proximity features."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        yield {"key": "station_static_features"}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        geo = _load_station_geo(STATION_GEO)
        out = geo.copy()
        out = _add_station_density(out)
        out = _add_gauge_features(out, GAUGE_MAP_PRIMARY, GAUGE_MAP_FALLBACK)
        out = _add_nearest_points(
            out,
            ECHO_DMR,
            lat_col="latitude",
            lon_col="longitude",
            id_col="npdes_id",
            name_col="facility_name",
            prefix="npdes",
        )
        out = _add_nearest_points(
            out,
            CIWQS_SSO,
            lat_col="latitude",
            lon_col="longitude",
            id_col="spill_id",
            name_col="agency_name",
            prefix="sso",
        )
        out = _add_catalog_provenance(out, STATIC_CATALOG)
        out = _add_ccap_raster_sample(out)
        return out.sort_values("station_id").reset_index(drop=True)


def _load_station_geo(path: Path) -> pd.DataFrame:
    geo = pd.read_parquet(path)
    required = ["station_id", "latitude", "longitude"]
    missing = [c for c in required if c not in geo.columns]
    if missing:
        raise ValueError(f"station_geo missing columns: {missing}")
    keep = [c for c in ["station_id", "beach_name", "county", "latitude", "longitude"] if c in geo.columns]
    out = geo[keep].dropna(subset=["station_id", "latitude", "longitude"]).drop_duplicates("station_id").copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    return out.dropna(subset=["latitude", "longitude"])


def _haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _add_station_density(df: pd.DataFrame) -> pd.DataFrame:
    dist = _haversine_matrix(df["latitude"], df["longitude"], df["latitude"], df["longitude"])
    out = df.copy()
    for radius in (5, 10, 25):
        out[f"station_density_{radius}km"] = ((dist <= radius).sum(axis=1) - 1).astype(int)
    return out


def _add_gauge_features(df: pd.DataFrame, primary: Path, fallback: Path) -> pd.DataFrame:
    out = df.copy()
    if primary.exists():
        gm = pd.read_parquet(primary).rename(columns={
            "gauge_id": "nearest_usgs_gauge_id",
            "distance_km": "dist_to_usgs_gauge_km",
        })
        cols = [c for c in ["station_id", "nearest_usgs_gauge_id", "dist_to_usgs_gauge_km"] if c in gm.columns]
        return out.merge(gm[cols], on="station_id", how="left")
    if fallback.exists():
        gm = pd.read_parquet(fallback).rename(columns={
            "nearest_gauge_id": "nearest_usgs_gauge_id",
            "dist_km": "dist_to_usgs_gauge_km",
        })
        cols = [c for c in ["station_id", "nearest_usgs_gauge_id", "dist_to_usgs_gauge_km"] if c in gm.columns]
        return out.merge(gm[cols], on="station_id", how="left")
    out["nearest_usgs_gauge_id"] = pd.NA
    out["dist_to_usgs_gauge_km"] = np.nan
    return out


def _add_nearest_points(
    stations: pd.DataFrame,
    path: Path,
    *,
    lat_col: str,
    lon_col: str,
    id_col: str,
    name_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = stations.copy()
    default_cols = {
        f"nearest_{prefix}_id": pd.NA,
        f"nearest_{prefix}_name": pd.NA,
        f"dist_to_{prefix}_km": np.nan,
        f"{prefix}_count_5km": 0,
        f"{prefix}_count_25km": 0,
    }
    if not path.exists():
        return out.assign(**default_cols)

    points = pd.read_parquet(path)
    required = [lat_col, lon_col, id_col]
    if any(c not in points.columns for c in required):
        return out.assign(**default_cols)
    points = points.dropna(subset=[lat_col, lon_col, id_col]).copy()
    points[lat_col] = pd.to_numeric(points[lat_col], errors="coerce")
    points[lon_col] = pd.to_numeric(points[lon_col], errors="coerce")
    points = points.dropna(subset=[lat_col, lon_col])
    if points.empty:
        return out.assign(**default_cols)

    point_identity = [id_col] + ([name_col] if name_col in points.columns else [])
    points = points.drop_duplicates(point_identity)
    dist = _haversine_matrix(out["latitude"], out["longitude"], points[lat_col], points[lon_col])
    nearest_idx = dist.argmin(axis=1)
    nearest = points.iloc[nearest_idx].reset_index(drop=True)
    out[f"nearest_{prefix}_id"] = nearest[id_col].astype("string").to_numpy()
    out[f"nearest_{prefix}_name"] = (
        nearest[name_col].astype("string").to_numpy() if name_col in nearest.columns else pd.NA
    )
    out[f"dist_to_{prefix}_km"] = dist.min(axis=1).round(3)
    out[f"{prefix}_count_5km"] = (dist <= 5.0).sum(axis=1).astype(int)
    out[f"{prefix}_count_25km"] = (dist <= 25.0).sum(axis=1).astype(int)
    return out


def _add_catalog_provenance(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.copy()
    out["static_catalog_rows"] = 0
    out["nhdplus_hr_product_count"] = 0
    out["ccap_latest_year"] = np.nan
    out["ccap_latest_geotiff_url"] = pd.NA
    out["nlcd_impervious_wcs_url"] = pd.NA
    if not path.exists():
        return out
    cat = pd.read_parquet(path)
    out["static_catalog_rows"] = int(len(cat))
    out["nhdplus_hr_product_count"] = int((cat.get("provider") == "USGS").sum()) if "provider" in cat else 0
    if {"provider", "format", "year", "url"}.issubset(cat.columns):
        ccap = cat[(cat["provider"] == "NOAA") & (cat["format"] == "GeoTIFF")].dropna(subset=["year"])
        if len(ccap):
            latest = ccap.sort_values("year").iloc[-1]
            out["ccap_latest_year"] = int(latest["year"])
            out["ccap_latest_geotiff_url"] = latest["url"]
    if {"provider", "product_id", "format", "url"}.issubset(cat.columns):
        nlcd = cat[
            (cat["provider"] == "MRLC")
            & (cat["format"] == "WCS")
            & (cat["product_id"].astype("string").str.contains("Impervious", case=False, na=False))
        ]
        if len(nlcd):
            out["nlcd_impervious_wcs_url"] = nlcd.iloc[0]["url"]
    return out


def _add_ccap_raster_sample(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ccap_landcover_code"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")
    out["ccap_developed"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="boolean")
    if "ccap_latest_geotiff_url" not in out.columns or out.empty:
        return out
    url = out["ccap_latest_geotiff_url"].dropna().astype("string")
    if url.empty:
        return out

    try:
        import rasterio
        from pyproj import Transformer

        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(str(url.iloc[0])) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs, ys = transformer.transform(
                    out["longitude"].astype(float).to_numpy(),
                    out["latitude"].astype(float).to_numpy(),
                )
                codes = []
                nodata = src.nodata
                for val in src.sample(list(zip(xs, ys))):
                    code = val[0] if len(val) else np.nan
                    if np.isnan(code) or (nodata is not None and code == nodata):
                        codes.append(pd.NA)
                    else:
                        codes.append(int(code))
        out["ccap_landcover_code"] = pd.Series(codes, index=out.index, dtype="Int64")
        out["ccap_developed"] = out["ccap_landcover_code"].isin(_CCAP_DEVELOPED_CODES).astype("boolean")
    except Exception as exc:  # noqa: BLE001
        print(f"[station_static_features] C-CAP raster sample unavailable: {exc}")
    return out


_CCAP_DEVELOPED_CODES = {2, 3, 4, 5}
