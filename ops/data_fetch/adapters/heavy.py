"""Heavy gridded archives (HRRR, NWM, CCAP/NHD/NLCD).

The static watershed/land-use source intentionally fetches provider catalogs
and direct product URLs, not the multi-GB rasters/geodatabases themselves. That
keeps the framework honest: the curated table is a real, reproducible inventory
of public inputs to subset later, while bulk extraction remains an explicit
operator action.
"""
from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET
from typing import Iterator, Optional
from urllib.parse import urlencode, urljoin

import pandas as pd
import numpy as np
from ..core import Adapter, _guard_write, _to_parquet_safe, get_with_backoff

CA_BBOX = {"min_lat": 32.5, "max_lat": 42.0, "min_lon": -124.4, "max_lon": -114.1}
CA_BBOX_TNM = (
    f"{CA_BBOX['min_lon']},{CA_BBOX['min_lat']},"
    f"{CA_BBOX['max_lon']},{CA_BBOX['max_lat']}"
)
TNM_PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
NHDPLUS_HR_DATASET = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
CCAP_CONUS_INDEX = (
    "https://ocmgeodatastor1.blob.core.windows.net/ccap/bulk_download/"
    "C-CAP_Regional_30-meter_Data/C-CAP_Regional_Land_Cover_Classification/"
    "CONUS/index.html"
)
MRLC_SERVICES_PAGE = "https://www.mrlc.gov/data-services-page"
NHDPLUS_FLOWLINE_QUERY = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/3/query"
HRRR_S3_HTTP = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
NWM_S3_HTTP = "https://noaa-nwm-pds.s3.amazonaws.com"
HRRR_VARIABLES = {
    "APCP": "precipitation_accumulation",
    "PRATE": "precipitation_rate",
    "TMP": "temperature",
    "UGRD": "u_wind",
    "VGRD": "v_wind",
    "GUST": "wind_gust",
}
FETCHED_AT = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

class _HeavyBase(Adapter):
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = dt.date.fromisoformat(start) if start else dt.date.today() - dt.timedelta(days=2)
        e = dt.date.fromisoformat(end) if end else dt.date.today()
        d = s
        while d <= e:
            yield {"key": d.isoformat(), "date": d.isoformat()}
            d += dt.timedelta(days=1)

class HrrrAdapter(_HeavyBase):
    """NOAA HRRR public GRIB2 message index rows for weather-driver extraction."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = dt.date.fromisoformat(start) if start else dt.date.today() - dt.timedelta(days=2)
        e = dt.date.fromisoformat(end) if end else dt.date.today() - dt.timedelta(days=1)
        d = s
        while d <= e:
            for hour in (0, 6, 12, 18):
                yield {
                    "key": f"{d.isoformat()}_{hour:02d}_f01",
                    "date": d.isoformat(),
                    "hour": hour,
                    "forecast_hour": 1,
                }
            d += dt.timedelta(days=1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        run = dt.datetime.fromisoformat(chunk["date"]).replace(hour=int(chunk["hour"]), tzinfo=dt.timezone.utc)
        fxx = int(chunk["forecast_hour"])
        stem = f"hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfsfcf{fxx:02d}.grib2"
        grib_url = f"{HRRR_S3_HTTP}/{stem}"
        idx_url = f"{grib_url}.idx"
        try:
            text = get_with_backoff(idx_url, timeout=60, retries=3, headers={"User-Agent": "mbal-datafetch/0.1"}).text
        except Exception as exc:  # noqa: BLE001
            print(f"[hrrr] index unavailable {idx_url}: {exc}")
            return pd.DataFrame()

        parsed = [_parse_hrrr_idx_line(line) for line in text.splitlines() if line.strip()]
        parsed = [p for p in parsed if p is not None]
        records = []
        for i, row in enumerate(parsed):
            variable = row["variable"]
            level = row["level"]
            if variable not in HRRR_VARIABLES:
                continue
            if variable in {"TMP", "UGRD", "VGRD"} and not level.startswith(("2 m", "10 m")):
                continue
            next_start = parsed[i + 1]["byte_start"] if i + 1 < len(parsed) else None
            records.append({
                "time": run + dt.timedelta(hours=fxx),
                "run_time": run,
                "forecast_hour": fxx,
                "variable": variable,
                "driver": HRRR_VARIABLES[variable],
                "level": level,
                "message": row["message"],
                "byte_start": row["byte_start"],
                "byte_end": int(next_start - 1) if next_start is not None else pd.NA,
                "grib_url": grib_url,
                "idx_url": idx_url,
                "source": "noaa-hrrr-bdp-pds",
            })
        return pd.DataFrame.from_records(records)

    def _consolidate(self):
        frames = []
        required = set(self.spec.required_columns)
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[hrrr] skipping unreadable chunk {p.name}: {exc}")
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
        for col in ("time", "run_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        keys = [k for k in self.spec.dedup_keys if k in df.columns]
        if keys:
            df = df.drop_duplicates(keys)
        df = df.sort_values(["time", "run_time", "forecast_hour", "message"])
        _to_parquet_safe(df, out_path)
        return out_path


def _parse_hrrr_idx_line(line: str) -> dict | None:
    parts = line.split(":")
    if len(parts) < 6:
        return None
    try:
        return {
            "message": int(parts[0]),
            "byte_start": int(parts[1]),
            "variable": parts[3],
            "level": parts[4],
            "forecast_label": parts[5],
        }
    except ValueError:
        return None

class NwmAdapter(_HeavyBase):
    """NOAA NWM public NetCDF object index for hydrology-driver extraction."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = dt.date.fromisoformat(start) if start else dt.date.today() - dt.timedelta(days=2)
        e = dt.date.fromisoformat(end) if end else dt.date.today() - dt.timedelta(days=1)
        d = s
        while d <= e:
            for family in ("analysis_assim", "short_range"):
                yield {"key": f"{d.isoformat()}_{family}", "date": d.isoformat(), "family": family}
            d += dt.timedelta(days=1)

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        prefix = f"nwm.{chunk['date'].replace('-', '')}/{chunk['family']}/"
        rows = []
        token = None
        while True:
            params = {
                "list-type": "2",
                "prefix": prefix,
                "max-keys": "1000",
            }
            if token:
                params["continuation-token"] = token
            resp = get_with_backoff(f"{NWM_S3_HTTP}/?{urlencode(params)}", timeout=60, retries=3)
            root = ET.fromstring(resp.content)
            ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for item in root.findall("s3:Contents", ns):
                key = (item.findtext("s3:Key", default="", namespaces=ns) or "").strip()
                if not key.endswith(".nc"):
                    continue
                meta = _parse_nwm_key(key)
                if meta is None:
                    continue
                rows.append(meta | {
                    "object_key": key,
                    "url": f"{NWM_S3_HTTP}/{key}",
                    "size_bytes": int(item.findtext("s3:Size", default="0", namespaces=ns) or 0),
                    "last_modified": item.findtext("s3:LastModified", default="", namespaces=ns),
                    "source": "noaa-nwm-pds",
                })
            truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns) or "").lower() == "true"
            token = root.findtext("s3:NextContinuationToken", default="", namespaces=ns)
            if not truncated or not token:
                break
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _consolidate(self):
        frames = []
        required = set(self.spec.required_columns)
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[nwm] skipping unreadable chunk {p.name}: {exc}")
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
        for col in ("run_time", "valid_time", "last_modified"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        keys = [k for k in self.spec.dedup_keys if k in df.columns]
        if keys:
            df = df.drop_duplicates(keys)
        df = df.sort_values(["run_time", "product", "member", "variable_group", "offset_hour", "object_key"])
        _to_parquet_safe(df, out_path)
        return out_path


def _parse_nwm_key(key: str) -> dict | None:
    name = key.rsplit("/", 1)[-1]
    m = re.match(
        r"nwm\.t(?P<hour>\d{2})z\.(?P<product>[a-z0-9_]+)\."
        r"(?P<group>[a-z0-9_]+)\.(?P<offset>(?:tm|f)\d{2,3})(?:\.(?P<member>[^.]+))?\.conus\.nc$",
        name,
    )
    if not m:
        return None
    date_m = re.match(r"nwm\.(\d{8})/", key)
    if not date_m:
        return None
    run_date = dt.datetime.strptime(date_m.group(1), "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    run_time = run_date.replace(hour=int(m.group("hour")))
    offset_token = m.group("offset")
    sign = -1 if offset_token.startswith("tm") else 1
    offset_hour = sign * int(offset_token[2:])
    return {
        "run_time": run_time,
        "valid_time": run_time + dt.timedelta(hours=offset_hour),
        "product": m.group("product"),
        "variable_group": m.group("group"),
        "offset_hour": offset_hour,
        "member": m.group("member") or "deterministic",
    }


def _geometry_centroid(geom: dict) -> tuple[float, float]:
    paths = geom.get("paths") or []
    pts = []
    for path in paths:
        pts.extend(path)
    if not pts:
        return np.nan, np.nan
    arr = np.asarray(pts, dtype=float)
    return float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))

class CcapNhdNlcdAdapter(Adapter):
    """C-CAP / NHDPlus / NLCD public static watershed-data catalog."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        yield {"key": "nhdplus_hr", "provider": "USGS", "product_family": "NHDPlus HR"}
        yield {"key": "ccap_regional", "provider": "NOAA", "product_family": "C-CAP Regional"}
        yield {"key": "nlcd_services", "provider": "MRLC", "product_family": "NLCD"}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        key = chunk["key"]
        if key == "nhdplus_hr":
            return self._fetch_nhdplus_hr_catalog()
        if key == "ccap_regional":
            return self._fetch_ccap_regional_catalog()
        if key == "nlcd_services":
            return self._fetch_nlcd_service_catalog()
        raise ValueError(f"unknown static catalog chunk: {key}")

    def _consolidate(self):
        """Consolidate only current provider chunks, ignoring legacy placeholders."""
        frames = []
        for c in self.iter_chunks(None, None):
            p = self._chunk_path(c["key"])
            if p.exists() and p.stat().st_size > 0:
                d = pd.read_parquet(p)
                if len(d):
                    frames.append(d)
        out_path = self.curated_path
        _guard_write(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            _to_parquet_safe(pd.DataFrame(columns=self.spec.required_columns), out_path)
            return out_path
        df = pd.concat(frames, ignore_index=True)
        keys = [k for k in self.spec.dedup_keys if k in df.columns]
        if keys:
            df = df.drop_duplicates(keys)
        df = df.sort_values(["provider", "product_family", "product_id", "url"])
        _to_parquet_safe(df, out_path)
        return out_path

    def _base_row(self, *, provider: str, product_family: str, product_id: str, title: str,
                  url: str, fmt: str, year: int | None = None,
                  is_bulk_download: bool = False) -> dict:
        return {
            "provider": provider,
            "product_family": product_family,
            "product_id": product_id,
            "title": title,
            "url": url,
            "format": fmt,
            "year": year,
            "california_relevant": True,
            "is_bulk_download": bool(is_bulk_download),
            "fetched_at_utc": FETCHED_AT,
        }

    def _fetch_nhdplus_hr_catalog(self) -> pd.DataFrame:
        params = {
            "datasets": NHDPLUS_HR_DATASET,
            "bbox": CA_BBOX_TNM,
            "prodFormats": "FileGDB",
            "outputFormat": "JSON",
            "max": "100",
        }
        # requests-style clients do not let get_with_backoff pass params, so use
        # a query URL to keep urllib and requests behavior identical.
        try:
            import requests

            prepared = requests.Request("GET", TNM_PRODUCTS_URL, params=params).prepare()
            resp = get_with_backoff(prepared.url, timeout=60, retries=4)
        except ImportError:
            from urllib.parse import urlencode

            resp = get_with_backoff(f"{TNM_PRODUCTS_URL}?{urlencode(params)}", timeout=60, retries=4)
        payload = resp.json()
        rows = []
        for item in payload.get("items", []):
            url = item.get("downloadURL") or item.get("downloadUrl") or item.get("url") or item.get("moreInfoUrl")
            title = item.get("title") or item.get("sourceId") or "NHDPlus HR product"
            product_id = item.get("sourceId") or item.get("id") or title
            rows.append(self._base_row(
                provider="USGS",
                product_family="NHDPlus HR",
                product_id=str(product_id),
                title=title,
                url=str(url or ""),
                fmt=str(item.get("format") or "FileGDB"),
                year=_extract_year(title),
                is_bulk_download=True,
            ) | {
                "source_url": TNM_PRODUCTS_URL,
                "source_total": int(payload.get("total", len(rows))),
                "bbox": CA_BBOX_TNM,
            })
        return pd.DataFrame(rows)

    def _fetch_ccap_regional_catalog(self) -> pd.DataFrame:
        html = get_with_backoff(CCAP_CONUS_INDEX, timeout=60, retries=4).text
        rows = []
        for href in re.findall(r'href="([^"]+)"', html):
            if not href.lower().endswith((".tif", ".xml")):
                continue
            url = urljoin(CCAP_CONUS_INDEX, href)
            year = _extract_year(href)
            fmt = "GeoTIFF" if href.lower().endswith(".tif") else "FGDC XML"
            rows.append(self._base_row(
                provider="NOAA",
                product_family="C-CAP Regional Land Cover Classification",
                product_id=f"ccap_conus_{year}_{fmt.lower().replace(' ', '_')}",
                title=f"C-CAP CONUS land-cover {year} {fmt}",
                url=url,
                fmt=fmt,
                year=year,
                is_bulk_download=href.lower().endswith(".tif"),
            ) | {
                "source_url": CCAP_CONUS_INDEX,
                "bbox": "CONUS coastal; subset required for California beach catchments",
            })
        return pd.DataFrame(rows)

    def _fetch_nlcd_service_catalog(self) -> pd.DataFrame:
        html = get_with_backoff(MRLC_SERVICES_PAGE, timeout=60, retries=4).text
        urls = sorted(set(re.findall(r'https?://[^"\']+', html)))
        rows = []
        for url in urls:
            lower = url.lower()
            if "dmsdata.cr.usgs.gov/geoserver" not in lower:
                continue
            if not any(token in lower for token in ("land-cover", "impervious", "canopy", "descriptor")):
                continue
            fmt = "WCS" if "wcs?" in lower else ("WMS" if "wms?" in lower else "service")
            service_name = url.rstrip("?").split("/")[-2] if "/" in url else url
            rows.append(self._base_row(
                provider="MRLC",
                product_family="NLCD map service",
                product_id=service_name,
                title=service_name.replace("_", " "),
                url=url,
                fmt=fmt,
                year=_extract_year(service_name),
                is_bulk_download=False,
            ) | {
                "source_url": MRLC_SERVICES_PAGE,
                "bbox": "CONUS service; request California beach catchment windows via WCS/WMS",
            })
        return pd.DataFrame(rows)


def _extract_year(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", str(text))
    return int(m.group(0)) if m else None
