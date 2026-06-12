"""California Data Exchange Center (CDEC) daily hydrology sensors.

CA Department of Water Resources operational network. CSV endpoint:
https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=<ID>&SensorNums=<N>&dur_code=D&Start=<d>&End=<d>
returns columns: STATION_ID, DURATION, SENSOR_NUMBER, SENSOR_TYPE, DATE TIME, OBS DATE,
VALUE, DATA_FLAG, UNITS.

Reservoir storage / precipitation / river flow at long-record stations feeding
California coastal watersheds — a runoff / antecedent-hydrology driver complementary
to the existing USGS daily discharge source. One station-sensor over ~35 years daily
is ~12k rows; a dozen station-sensors clears 50k easily.

Staged + resumable: one chunk per (station, sensor). Writes only to data/external_*.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterator, Optional
from urllib.parse import urlencode

import pandas as pd

from ..core import Adapter, get_with_backoff

CSV_API = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"

# Long-record CDEC stations (major reservoirs / hydrology sites across CA basins).
STATIONS = {
    "SHA": "Shasta Lake", "ORO": "Oroville", "FOL": "Folsom Lake", "CLE": "Trinity Lake",
    "NML": "New Melones", "SNL": "San Luis", "BUL": "Bullards Bar", "MIL": "Millerton Lake",
    "DNP": "Don Pedro", "EXC": "Exchequer/McClure", "PNF": "Pine Flat", "CMN": "Camanche",
    "BER": "Lake Berryessa", "CCH": "Cachuma (Santa Barbara)", "SNB": "San Antonio (Monterey)",
}
# Daily sensors: 15=reservoir storage (AF), 45=precip incremental (in), 41=full natural flow,
# 23=reservoir outflow. Not every station has every sensor — empty responses are skipped.
SENSORS = {15: "reservoir_storage_af", 45: "precip_incremental_in", 23: "reservoir_outflow_cfs"}


class CdecAdapter(Adapter):
    """Fetch CDEC daily sensor series, one (station, sensor) chunk at a time."""

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        s = start or "1990-01-01"
        e = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
        for station_id, name in STATIONS.items():
            for sensor_num, label in SENSORS.items():
                yield {
                    "key": f"{station_id}_{sensor_num}",
                    "station_id": station_id,
                    "station_name": name,
                    "sensor_num": sensor_num,
                    "sensor_label": label,
                    "start": s,
                    "end": e,
                }

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        params = {
            "Stations": chunk["station_id"],
            "SensorNums": chunk["sensor_num"],
            "dur_code": "D",
            "Start": chunk["start"],
            "End": chunk["end"],
        }
        url = f"{CSV_API}?{urlencode(params)}"
        try:
            resp = get_with_backoff(
                url, timeout=120, retries=3,
                headers={"User-Agent": "Monterey-Bay-AI-Lab-data-fetcher/0.1"},
            )
        except Exception:
            return pd.DataFrame()
        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        if "STATION_ID" not in text:
            return pd.DataFrame()
        import io
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            return pd.DataFrame()
        if df.empty or "VALUE" not in df.columns or "DATE TIME" not in df.columns:
            return pd.DataFrame()
        ts = pd.to_datetime(df["DATE TIME"], format="%Y%m%d %H%M", errors="coerce", utc=True)
        val = pd.to_numeric(df["VALUE"], errors="coerce")
        out = pd.DataFrame({
            "station_id": df["STATION_ID"].astype(str).values,
            "station_name": chunk["station_name"],
            "date": ts.values,
            "sensor_num": chunk["sensor_num"],
            "sensor_label": chunk["sensor_label"],
            "value": val.values,
            "units": df.get("UNITS", pd.Series([""] * len(df))).astype(str).values,
        })
        # CDEC encodes missing as large negatives (e.g. -9999/-9998) or blanks.
        out = out[(out["value"] > -9000) & out["value"].notna()]
        out = out.dropna(subset=["date"])
        return out.reset_index(drop=True)
