"""GHCN-Daily climate stations (California subset).

NCEI Global Historical Climatology Network — Daily. One gzipped CSV per station at
https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/<ID>.csv.gz with no header;
columns: ID, YYYYMMDD, ELEMENT, VALUE, M-flag, Q-flag, S-flag, OBS-TIME.

A single long-record CA airport station is ~380k element-rows, so a handful of
stations clears 50k labeled rows easily. Provides ground-truth daily precipitation
and temperature at station resolution — finer than the existing gridded rainfall
source, for first-flush / antecedent-dryness drivers.

Staged + resumable: one chunk per station. Writes only to data/external_*.
Values are kept raw (`value_raw`, GHCN integer units) plus converted to SI
(`value`, mm for PRCP/SNOW/SNWD, °C for TMAX/TMIN/TAVG).
"""
from __future__ import annotations

import gzip
import io
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station"

# Long-record California stations (major airports / coastal sites).
CA_STATIONS = {
    "USW00023234": "San Francisco Intl",
    "USW00023174": "Los Angeles Intl",
    "USW00023188": "San Diego Lindbergh",
    "USW00023190": "Santa Maria",
    "USW00023230": "Oakland",
    "USW00023271": "Sacramento Executive",
    "USW00093193": "Fresno Yosemite Intl",
    "USW00023232": "Sacramento Metro",
    "USW00023273": "Santa Barbara Muni",
    "USW00023161": "Daggett/Barstow",
    "USW00024213": "Eureka/Arcata",
    "USW00023157": "Bakersfield Meadows",
}

# Elements we keep, with conversion factor to SI and unit label.
# GHCN: PRCP/SNOW/SNWD in tenths of mm or mm; TMAX/TMIN/TAVG in tenths of °C.
_ELEMENTS = {
    "PRCP": (0.1, "mm"),    # tenths of mm -> mm
    "SNOW": (1.0, "mm"),    # mm
    "SNWD": (1.0, "mm"),    # mm
    "TMAX": (0.1, "degC"),  # tenths of degC -> degC
    "TMIN": (0.1, "degC"),
    "TAVG": (0.1, "degC"),
}
_COLS = ["station_id", "yyyymmdd", "element", "value_raw", "mflag", "qflag", "sflag", "obstime"]


class GhcndAdapter(Adapter):
    """Fetch GHCN-Daily by-station CSVs for California, one station per chunk."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        for station_id, name in CA_STATIONS.items():
            yield {
                "key": station_id,
                "station_id": station_id,
                "station_name": name,
                "url": f"{BASE}/{station_id}.csv.gz",
                "start": start,
                "end": end,
            }

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        try:
            resp = get_with_backoff(
                chunk["url"], timeout=120, retries=3,
                headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
            )
        except Exception:
            return pd.DataFrame()  # station may not exist — non-fatal
        raw = resp.content
        if not raw:
            return pd.DataFrame()
        try:
            text = gzip.decompress(raw).decode("utf-8", "replace")
        except OSError:
            text = raw.decode("utf-8", "replace")
        try:
            df = pd.read_csv(io.StringIO(text), header=None, names=_COLS, dtype=str)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
        df = df[df["element"].isin(_ELEMENTS)]
        if df.empty:
            return pd.DataFrame()

        date = pd.to_datetime(df["yyyymmdd"], format="%Y%m%d", errors="coerce", utc=True)
        raw_val = pd.to_numeric(df["value_raw"], errors="coerce")
        factor = df["element"].map(lambda e: _ELEMENTS[e][0])
        out = pd.DataFrame({
            "station_id": df["station_id"].values,
            "station_name": chunk["station_name"],
            "date": date.values,
            "element": df["element"].values,
            "value": (raw_val * factor).values,
            "value_raw": raw_val.values,
            "units": df["element"].map(lambda e: _ELEMENTS[e][1]).values,
            "qflag": df["qflag"].fillna("").values,
        })
        out = out.dropna(subset=["date", "value"])
        # Drop values that failed GHCN quality control (non-blank qflag).
        out = out[out["qflag"].str.strip() == ""]
        # Optional date filter.
        if chunk.get("start"):
            out = out[out["date"] >= pd.Timestamp(chunk["start"], tz="UTC")]
        if chunk.get("end"):
            out = out[out["date"] <= pd.Timestamp(chunk["end"], tz="UTC")]
        return out.reset_index(drop=True)
