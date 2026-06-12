"""USGS NWIS instantaneous-values: turbidity / water-temperature / gage-height (CA).

Source: ``https://waterservices.usgs.gov/nwis/iv/`` (keyless). Instantaneous (sub-
hourly, typically 15-min) sensor series — a *new modality* vs the existing daily-mean
discharge wrapper. Turbidity (param 63680) is a direct first-flush / runoff
contamination proxy; water temperature (00010) and gage height (00065) are physical
drivers. CA has ~141 IV turbidity sites; this fetches turbidity + temp + gage-height
for the configured site set, chunked one (site, year) per resumable chunk.

Lane B (claude-fetch-2).
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

_BASE = "https://waterservices.usgs.gov/nwis/iv/"
_PARAMS = "00010,63680,00065,00095"  # water temp, turbidity, gage height, sp. conductance
_PARAM_LABEL = {
    "00010": "water_temp_c", "63680": "turbidity_fnu",
    "00065": "gage_height_ft", "00095": "sp_conductance_uscm",
}


def _ca_iv_turbidity_sites(limit: int = 1000) -> List[str]:
    """Live-list CA sites that carry instantaneous turbidity. Falls back to a
    curated coastal/estuary set if the site service is unreachable."""
    fallback = [
        "11447650", "11303500", "11455420", "11458000", "11460400", "11162630",
        "11169025", "11447905", "375607122264701", "374811122235001",
    ]
    url = (f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=ca"
           f"&parameterCd=63680&hasDataType=iv&siteStatus=all")
    try:
        resp = get_with_backoff(url, timeout=90, headers={"User-Agent": "mbal-datafetch/0.1"})
        sites = []
        for ln in getattr(resp, "text", "").splitlines():
            if not ln or ln.startswith("#") or ln.startswith("agency_cd") or ln.startswith("5s"):
                continue
            parts = ln.split("\t")
            if len(parts) > 1 and parts[1].strip():
                sites.append(parts[1].strip())
        sites = sorted(set(sites))
        return sites[:limit] if sites else fallback
    except Exception:
        return fallback


class UsgsIvTurbidityAdapter(Adapter):
    # Full native coverage: every CA IV turbidity site, all years the API serves.
    START_YEAR = 2007
    END_YEAR = 2025
    SITE_LIMIT = 1000

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.START_YEAR
        y1 = int(end[:4]) if end else self.END_YEAR
        sites = _ca_iv_turbidity_sites(self.SITE_LIMIT)
        for site in sites:
            for year in range(y0, y1 + 1):
                yield {"key": f"{site}_{year}", "site": site, "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        site, year = chunk["site"], chunk["year"]
        url = (f"{_BASE}?format=json&sites={site}&parameterCd={_PARAMS}"
               f"&startDT={year}-01-01&endDT={year}-12-31&siteStatus=all")
        try:
            resp = get_with_backoff(url, timeout=120, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            ts_list = resp.json().get("value", {}).get("timeSeries", [])
        except Exception:
            return pd.DataFrame()
        frames = []
        for ts in ts_list:
            pcode = ts.get("variable", {}).get("variableCode", [{}])[0].get("value", "")
            label = _PARAM_LABEL.get(pcode, pcode)
            sinfo = ts.get("sourceInfo", {})
            site_no = (sinfo.get("siteCode", [{}])[0].get("value", site))
            geo = sinfo.get("geoLocation", {}).get("geogLocation", {})
            for block in ts.get("values", []):
                recs = block.get("value", [])
                if not recs:
                    continue
                d = pd.DataFrame.from_records(recs)
                d = d.rename(columns={"dateTime": "time", "value": "result_value"})
                d["site_id"] = str(site_no)
                d["parameter"] = label
                d["result_value"] = pd.to_numeric(d["result_value"], errors="coerce")
                d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
                d["latitude"] = geo.get("latitude")
                d["longitude"] = geo.get("longitude")
                frames.append(d[["site_id", "time", "parameter", "result_value",
                                 "latitude", "longitude"]])
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out[out["time"].notna() & (out["result_value"] > -999990)]
        return out.reset_index(drop=True)
