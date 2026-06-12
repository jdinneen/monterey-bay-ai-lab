"""USGS NWIS Daily Values — statewide CA water quality + streamflow.

Source: ``https://waterservices.usgs.gov/nwis/dv/`` (keyless). Daily-mean sensor
series across ALL California sites for six driver parameters: water temperature,
specific conductance, dissolved oxygen, pH, turbidity, and discharge. Distinct from
the existing `discharge_usgs` wrapper (near-beach daily discharge only) and from
`usgs_iv_turbidity` (sub-hourly, turbidity sites only): this is the broad *statewide
daily* water-quality + flow corpus — a runoff / first-flush / water-quality driver set
for the bacteria model. One statewide call per (parameter, year) — efficient.

Lane B (claude-fetch-2).
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

_BASE = "https://waterservices.usgs.gov/nwis/dv/"
_PARAMS = {
    "00010": "water_temp_c", "00095": "sp_conductance_uscm", "00300": "dissolved_oxygen_mgl",
    "00400": "ph", "63680": "turbidity_fnu", "00060": "discharge_cfs",
}


class UsgsDvStatewideAdapter(Adapter):
    START_YEAR = 2005
    END_YEAR = 2025

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.START_YEAR
        y1 = int(end[:4]) if end else self.END_YEAR
        for pcode, label in _PARAMS.items():
            for year in range(y0, y1 + 1):
                yield {"key": f"{label}_{year}", "pcode": pcode, "label": label, "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        url = (f"{_BASE}?format=json&stateCd=ca&parameterCd={chunk['pcode']}"
               f"&startDT={chunk['year']}-01-01&endDT={chunk['year']}-12-31&siteStatus=all")
        try:
            resp = get_with_backoff(url, timeout=180, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
            ts_list = resp.json().get("value", {}).get("timeSeries", [])
        except Exception:
            return pd.DataFrame()
        frames = []
        for ts in ts_list:
            sinfo = ts.get("sourceInfo", {})
            site = sinfo.get("siteCode", [{}])[0].get("value", "")
            geo = sinfo.get("geoLocation", {}).get("geogLocation", {})
            for block in ts.get("values", []):
                recs = block.get("value", [])
                if not recs:
                    continue
                d = pd.DataFrame.from_records(recs)
                d = d.rename(columns={"dateTime": "date", "value": "result_value"})
                d["site_id"] = str(site)
                d["parameter"] = chunk["label"]
                d["result_value"] = pd.to_numeric(d["result_value"], errors="coerce")
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
                d["latitude"] = geo.get("latitude")
                d["longitude"] = geo.get("longitude")
                frames.append(d[["site_id", "date", "parameter", "result_value",
                                 "latitude", "longitude"]])
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out = out[out["date"].notna() & (out["result_value"] > -999990)]
        return out.reset_index(drop=True)
