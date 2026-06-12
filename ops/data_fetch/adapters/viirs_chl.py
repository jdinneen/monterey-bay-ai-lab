"""VIIRS ocean color / chlorophyll-a (M1 point), staged.

CoastWatch ERDDAP science-quality VIIRS chlorophyll (`nesdisVHNSQchlaDaily`),
extracted at the M1 point into data/external_* (the trusted chlorophyll_cache is
empty). Chunked/resumable by year.

Note: the legacy `erdVHNchla1day` dataset was retired (HTTP 404). The current
science-quality dataset carries an altitude dimension, so HAS_ALTITUDE is set.
"""
from __future__ import annotations

import calendar
import datetime as dt
from typing import Iterator, Optional

from .erddap import ErddapPointAdapter


class ViirsChlAdapter(ErddapPointAdapter):
    DATASET_ID = "nesdisVHNSQchlaDaily"
    VARIABLE = "chlor_a"
    OUT_TIME = "time"
    OUT_VALUE = "chlor_a"
    FIRST_YEAR = 2012
    HAS_ALTITUDE = True

    # This dataset is global 4km daily; a full-year extraction 502s (proxy timeout),
    # so chunk MONTHLY instead of the base adapter's yearly granularity.
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = max(int(start[:4]) if start else self.FIRST_YEAR, self.FIRST_YEAR)
        m0 = int(start[5:7]) if start and len(start) >= 7 else 1
        today = dt.date.today()
        y1 = int(end[:4]) if end else today.year
        m1 = int(end[5:7]) if end and len(end) >= 7 else 12
        for yr in range(y0, y1 + 1):
            for m in range(1, 13):
                if (yr, m) < (y0, m0) or (yr, m) > (y1, m1):
                    continue
                if (yr, m) > (today.year, today.month):
                    break
                yield {"key": f"{yr}-{m:02d}", "year": yr, "month": m}

    def _bounds(self, chunk: dict) -> tuple[str, str]:
        yr, m = chunk["year"], chunk["month"]
        last = calendar.monthrange(yr, m)[1]
        s = f"{yr}-{m:02d}-01"
        e = f"{yr}-{m:02d}-{last:02d}"
        today = dt.date.today()
        if (yr, m) == (today.year, today.month):
            e = (today - dt.timedelta(days=1)).isoformat()
        return s, e
