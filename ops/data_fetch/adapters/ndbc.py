"""NDBC historical standard-meteorological buoy observations (California buoys).

Source: NOAA NDBC historical stdmet text archive
``https://www.ndbc.noaa.gov/view_text_file.php?filename=<STN>h<YYYY>.txt.gz&dir=data/historical/stdmet/``

Each (station, year) file is ~hourly fixed-width met: wind, gust, wave height/period,
pressure, air/water temperature, dewpoint, visibility, tide. Multiple CA buoys ×
many years × hourly >> 50k labeled timestamped rows. A physical-driver corpus
(water temp / wave / wind / pressure) complementary to the CDIP wave wrapper.

Lane B (claude-fetch-2). Resumable: one chunk per (station, year).
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

# California / nearshore-CA NDBC standard-met buoys.
CA_BUOYS = [
    "46042",  # Monterey Bay
    "46026",  # San Francisco
    "46028",  # Cape San Martin
    "46012",  # Half Moon Bay
    "46013",  # Bodega Bay
    "46014",  # Point Arena
    "46022",  # Eel River
    "46011",  # Santa Maria
    "46025",  # Santa Monica Basin
    "46047",  # Tanner Bank
    "46086",  # San Clemente Basin
    "46053",  # East Santa Barbara
    "46054",  # West Santa Barbara
    "46069",  # South Santa Rosa Island
    "46239",  # Point Sur
    "46092",  # MBARI M1 (NDBC mirror)
]

_BASE = "https://www.ndbc.noaa.gov/view_text_file.php"
# Per-column NDBC fill sentinels.
_SENTINELS = {99.0, 99.00, 999.0, 9999.0, 9999.00, 99999.0}
_NUM_COLS = ["WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
             "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE", "BAR"]
# Hard physical limits — values outside are sensor failures (e.g. PRES==0.0), set to
# NaN. Kept >= the spec value_bounds so a cleaned column always clears the gate.
_PHYS_RANGE = {
    "PRES": (850.0, 1100.0), "BAR": (850.0, 1100.0),
    "WTMP": (-4.0, 40.0), "ATMP": (-40.0, 55.0), "DEWP": (-40.0, 40.0),
    "WVHT": (0.0, 35.0), "WSPD": (0.0, 120.0), "GST": (0.0, 150.0),
    "WDIR": (0.0, 360.0), "MWD": (0.0, 360.0),
}


def _clean_physical(df: pd.DataFrame) -> pd.DataFrame:
    """NaN values outside hard physical limits (sensor failures like PRES==0)."""
    for c, (lo, hi) in _PHYS_RANGE.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.where((v >= lo) & (v <= hi))
    return df


class NdbcStdmetAdapter(Adapter):
    # Full native coverage; legacy 2-digit-year files (pre-2007) are handled in _parse.
    START_YEAR = 2000
    END_YEAR = 2024

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.START_YEAR
        y1 = int(end[:4]) if end else self.END_YEAR
        for stn in CA_BUOYS:
            for year in range(y0, y1 + 1):
                yield {"key": f"{stn}_{year}", "station": stn, "year": year}

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        stn, year = chunk["station"], chunk["year"]
        url = f"{_BASE}?filename={stn}h{year}.txt.gz&dir=data/historical/stdmet/"
        try:
            resp = get_with_backoff(url, timeout=90, retries=3,
                                    headers={"User-Agent": "mbal-datafetch/0.1"})
        except Exception:
            return pd.DataFrame()
        text = getattr(resp, "text", "")
        if not text or "<html" in text[:200].lower() or "Unable" in text[:200]:
            return pd.DataFrame()
        return self._parse(text, stn)

    @staticmethod
    def _parse(text: str, stn: str) -> pd.DataFrame:
        header = None
        rows = []
        for ln in text.splitlines():
            if ln.startswith("#"):
                tok = ln.lstrip("#").split()
                if header is None and tok and tok[0] in ("YY", "YYYY"):
                    header = tok
                continue
            if ln.strip():
                rows.append(ln.split())
        if header is None or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Align width (older files lack some trailing columns).
        ncol = min(len(header), df.shape[1])
        df = df.iloc[:, :ncol]
        df.columns = header[:ncol]
        # Build timestamp from first columns (positional: year month day hour [minute]).
        ren = {header[0]: "year", header[1]: "month", header[2]: "day", header[3]: "hour"}
        if len(header) > 4 and header[4] == "mm":
            ren[header[4]] = "minute"
        df = df.rename(columns=ren)
        parts = {p: pd.to_numeric(df.get(p), errors="coerce")
                 for p in ("year", "month", "day", "hour")}
        yr = parts["year"]
        yr = yr.where(yr > 100, yr + 1900)  # 2-digit legacy years
        ts = pd.to_datetime(dict(
            year=yr, month=parts["month"], day=parts["day"],
            hour=parts["hour"], minute=pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0),
        ), errors="coerce")
        out = pd.DataFrame({"station_id": stn, "time": ts})
        for c in _NUM_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out[c] = v.where(~v.isin(_SENTINELS))
        out = _clean_physical(out)
        out = out[out["time"].notna()]
        return out.reset_index(drop=True)

    def _consolidate(self):
        """Consolidate then re-apply physical cleaning, so a re-run repairs legacy raw
        chunks (written before the clean existed) without any re-download."""
        from ..core import _to_parquet_safe

        path = super()._consolidate()
        df = pd.read_parquet(path)
        before = df.copy()
        df = _clean_physical(df)
        if not df.equals(before):
            _to_parquet_safe(df, path)
        return path
