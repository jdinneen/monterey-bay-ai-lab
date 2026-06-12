"""CalHABMAP marine HAB / domoic-acid labels from the SCCOOS ERDDAP (tabledap).

Pulls the weekly shore-monitoring panel for ALL CalHABMAP pier/buoy stations (domoic acid
pDA/tDA/dDA, the two Pseudo-nitzschia groups, plus the in-situ drivers nutrients / temp /
chlorophyll). This is the human-health marine-biotoxin label — NOT the freshwater
cyanobacteria in `california_hab_bloom_reports`, which is irrelevant to coastal beaches.

Robust to per-station schema differences: each station's available variables are read once
from the ERDDAP `info` endpoint and intersected with the wanted set, so a station that lacks
(say) nutrients still fetches its DA + cell counts instead of erroring the whole chunk. On any
network/parse failure a chunk degrades to empty and the source is honestly classified
IMPLEMENTED_NOT_FETCHED.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import urllib.request
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

ERDDAP = "https://erddap.sccoos.org/erddap"
START_YEAR = 2005  # CalHABMAP series begin 2005-06; default earlier bound

# Wanted columns, superset across stations. Intersected per-station with what ERDDAP reports.
WANT = [
    "time", "latitude", "longitude", "Location_Code",
    "pDA", "tDA", "dDA",                                    # domoic acid (ng/mL) -- the label
    "Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group",  # cells/L
    "Temp", "Salinity", "Avg_Chloro",                      # physical / chl drivers
    "Phosphate", "Silicate", "Nitrate", "Nitrite_Nitrate", "Ammonium",  # nutrient drivers
    "Akashiwo_sanguinea", "Alexandrium_spp", "Dinophysis_spp",
    "Lingulodinium_polyedra", "Total_Phytoplankton",       # other-bloom context
]

# Fallback station list (verified 2026-06 on the SCCOOS ERDDAP) if discovery fails.
FALLBACK_STATIONS = [
    "HABs-BodegaMarineLabBuoy", "HABs-BodegaMarineLab", "HABs-CalPolyPier",
    "HABs-HumboldtSouthBay", "HABs-Humboldt", "HABs-InnerTomalesBay",
    "HABs-MontereyWharf", "HABs-MorroBayBackBay", "HABs-MorroBayFrontBay",
    "HABs-NewportBeachPier", "HABs-SantaCruzWharf", "HABs-SantaMonicaPier",
    "HABs-ScrippsPier", "HABs-StearnsWharf", "HABs-TomalesBayMid-ChannelBuoy",
    "HABs-TomalesBayMouth", "HABs-TrinidadPier",
]


def _http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "mbal-datafetch/0.1"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace"))


def discover_stations() -> list[str]:
    try:
        d = _http_json(f"{ERDDAP}/search/index.json?searchFor=HABs&itemsPerPage=1000")
        cols = d["table"]["columnNames"]; di = cols.index("Dataset ID")
        ids = [r[di] for r in d["table"]["rows"] if str(r[di]).startswith("HAB")]
        return ids or list(FALLBACK_STATIONS)
    except Exception:
        return list(FALLBACK_STATIONS)


class HabmapAdapter(Adapter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._vars_cache: dict[str, list[str]] = {}

    def _station_vars(self, ds: str) -> list[str]:
        """Variables this station actually exposes, intersected with WANT (order-preserved)."""
        if ds not in self._vars_cache:
            try:
                d = _http_json(f"{ERDDAP}/info/{ds}/index.json")
                cols = d["table"]["columnNames"]
                ri, vi = cols.index("Row Type"), cols.index("Variable Name")
                have = {r[vi] for r in d["table"]["rows"] if r[ri] == "variable"}
                self._vars_cache[ds] = [v for v in WANT if v in have]
            except Exception:
                self._vars_cache[ds] = []
        return self._vars_cache[ds]

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else START_YEAR
        y1 = int(end[:4]) if end else dt.date.today().year
        for ds in discover_stations():
            for yr in range(y0, y1 + 1):
                yield {"key": f"{ds}_{yr}", "station": ds, "year": yr}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        ds, yr = chunk["station"], chunk["year"]
        cols = self._station_vars(ds)
        if "time" not in cols:  # station unreadable / no usable vars
            return pd.DataFrame()
        url = (f"{ERDDAP}/tabledap/{ds}.csv?{','.join(cols)}"
               f"&time>={yr}-01-01&time<={yr}-12-31")
        try:
            resp = get_with_backoff(url, timeout=90, retries=2, base_sleep=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
        except Exception:
            return pd.DataFrame()
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        # ERDDAP returns an HTML/JSON error (not CSV) when a station has no rows in range.
        if not text.strip() or text.lstrip().startswith("<") or "time" not in text[:200]:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])  # row 1 = units
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df.get("time"), utc=True, errors="coerce")
        for c in df.columns:
            if c not in ("time", "Location_Code"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["station"] = ds
        return df.dropna(subset=["time"])
