"""NDBC continuous-winds (cwind) historical — 10-minute winds at California buoys.

Source: NOAA NDBC historical ``data/historical/cwind`` text archive (same access
pattern as stdmet). Continuous (≈10-min) wind: direction, speed, gust direction, gust
speed — a higher-frequency, upwelling-favorable WIND driver distinct from the hourly
`ndbc_stdmet` standard-met record. Reuses the NDBC fixed-width parser + physical clean.

Lane B (claude-fetch-2). Resumable: one chunk per (station, year).
"""
from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff
from .ndbc import CA_BUOYS, _clean_physical

_BASE = "https://www.ndbc.noaa.gov/view_text_file.php"
_SENTINELS = {99.0, 999.0, 9999.0, 99.00}
_NUM_COLS = ["WDIR", "WSPD", "GDR", "GST"]  # GTIME (hhmm) dropped


class NdbcCwindAdapter(Adapter):
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
        url = f"{_BASE}?filename={stn}c{year}.txt.gz&dir=data/historical/cwind/"
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
        header, rows = None, []
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
        ncol = min(len(header), df.shape[1])
        df = df.iloc[:, :ncol]
        df.columns = header[:ncol]
        ren = {header[0]: "year", header[1]: "month", header[2]: "day", header[3]: "hour"}
        if len(header) > 4 and header[4] == "mm":
            ren[header[4]] = "minute"
        df = df.rename(columns=ren)
        parts = {p: pd.to_numeric(df.get(p), errors="coerce") for p in ("year", "month", "day", "hour")}
        yr = parts["year"].where(parts["year"] > 100, parts["year"] + 1900)
        ts = pd.to_datetime(dict(
            year=yr, month=parts["month"], day=parts["day"], hour=parts["hour"],
            minute=pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0),
        ), errors="coerce")
        out = pd.DataFrame({"station_id": stn, "time": ts})
        for c in _NUM_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                out[c] = v.where(~v.isin(_SENTINELS))
        out = _clean_physical(out)  # WDIR/WSPD/GST physical ranges (GDR not in dict -> untouched)
        out = out[out["time"].notna()]
        return out.reset_index(drop=True)
