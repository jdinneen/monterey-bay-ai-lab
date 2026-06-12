"""Bounded CenCOOS ERDDAP tabledap pulls for MBARI-linked moorings.

These adapters intentionally fetch small, date-bounded tabledap slices into the
staged data-fetch area. They are not a replacement for the trusted MBAL history
pipeline; they add reproducible public-ERDDAP checkpoints for missing source
coverage and source-health checks.
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Iterator, Optional

import pandas as pd

from ..core import Adapter, get_with_backoff

CENCOOS_ERDDAP = "https://erddap.cencoos.org/erddap"


def _year_bounds(year: int, first_year: int) -> tuple[str, str]:
    start = f"{year}-01-01"
    if year == first_year:
        start = f"{first_year}-01-01"
    end = f"{year}-12-31"
    today = dt.date.today()
    if year == today.year:
        end = (today - dt.timedelta(days=1)).isoformat()
    return start, end


class _CencoosTabledapBase(Adapter):
    DATASETS: dict[str, list[str]] = {}
    FIRST_YEAR: int = 2024
    MAX_ROWS_PER_CHUNK: int = 500_000
    QUALITY_BOUNDS: dict[str, tuple[float, float]] = {}

    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        y0 = int(start[:4]) if start else self.FIRST_YEAR
        y0 = max(y0, self.FIRST_YEAR)
        y1 = int(end[:4]) if end else dt.date.today().year
        for dataset_id in self.DATASETS:
            for year in range(y0, y1 + 1):
                yield {"key": f"{dataset_id}_{year}", "dataset_id": dataset_id, "year": year}

    def _fetch_dataset_year(self, dataset_id: str, columns: list[str], year: int) -> pd.DataFrame:
        start, end = _year_bounds(year, self.FIRST_YEAR)
        query = ",".join(columns)
        url = (
            f"{CENCOOS_ERDDAP}/tabledap/{dataset_id}.csv?{query}"
            f"&time>={start}&time<={end}"
        )
        try:
            resp = get_with_backoff(
                url,
                timeout=180,
                retries=2,
                base_sleep=3,
                headers={"User-Agent": "mbal-datafetch/0.1"},
            )
        except Exception as exc:  # noqa: BLE001 - unavailable datasets should not kill a multi-source fetch.
            print(f"[{self.source}] {dataset_id} {year} unavailable: {exc}")
            return pd.DataFrame(columns=["dataset_id", *columns])

        text = resp.text if hasattr(resp, "text") else resp.content.decode("utf-8", "replace")
        if not text.strip() or text.lstrip().startswith("<") or "Error {" in text[:200]:
            return pd.DataFrame(columns=["dataset_id", *columns])
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=[1])
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.source}] {dataset_id} {year} parse failed: {exc}")
            return pd.DataFrame(columns=["dataset_id", *columns])
        if len(df) > self.MAX_ROWS_PER_CHUNK:
            df = df.head(self.MAX_ROWS_PER_CHUNK).copy()
            df["bounded_sample"] = True
        df.insert(0, "dataset_id", dataset_id)
        return df

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        dataset_id = chunk["dataset_id"]
        columns = self.DATASETS[dataset_id]
        out = self._fetch_dataset_year(dataset_id, columns, int(chunk["year"]))
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        numeric_cols = [c for c in out.columns if c not in {"dataset_id", "time", "station"}]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col, (lo, hi) in self.QUALITY_BOUNDS.items():
            if col in out.columns:
                out[col] = out[col].where(out[col].between(lo, hi))
        return out.dropna(subset=["time"]) if "time" in out.columns else out


class CencoosMbalMooringsAdapter(_CencoosTabledapBase):
    """M1 public ERDDAP real-time and historic source slices."""

    FIRST_YEAR = 2024
    QUALITY_BOUNDS = {
        "sea_water_temperature": (-2.5, 35.0),
        "sea_water_practical_salinity": (0.0, 42.0),
        "air_temperature": (-20.0, 45.0),
    }
    DATASETS = {
        "org_mbari_m1": [
            "time",
            "latitude",
            "longitude",
            "z",
            "sea_water_temperature",
            "sea_water_practical_salinity",
            "air_temperature",
        ],
        "m1-mooring-historic-data-montere": [
            "time",
            "latitude",
            "longitude",
            "z",
            "sea_water_temperature",
            "sea_water_practical_salinity",
            "air_temperature",
        ],
    }


class CencoosOceanAcidificationAdapter(_CencoosTabledapBase):
    """OA1/OA2 chemistry buoy slices. OA1 is kept in the plan but may be offline."""

    FIRST_YEAR = 2024
    QUALITY_BOUNDS = {
        "sea_water_temperature": (-2.5, 35.0),
        "sea_water_practical_salinity": (0.0, 42.0),
        "mass_concentration_of_chlorophyll_in_sea_water": (0.0, 500.0),
        "moles_of_oxygen_per_unit_mass_in_sea_water": (0.0, 600.0),
        "sea_water_ph_reported_on_total_scale": (6.5, 9.5),
        "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas": (0.0, 5000.0),
    }
    DATASETS = {
        "oa1-mbari-buoy-1": [
            "time",
            "latitude",
            "longitude",
            "z",
            "sea_water_temperature",
            "sea_water_practical_salinity",
            "mass_concentration_of_chlorophyll_in_sea_water",
            "moles_of_oxygen_per_unit_mass_in_sea_water",
            "sea_water_ph_reported_on_total_scale",
            "mole_fraction_of_carbon_dioxide_in_sea_water_in_wet_gas",
        ],
        "oa2-mbari-buoy": [
            "time",
            "latitude",
            "longitude",
            "z",
            "sea_water_temperature",
            "sea_water_practical_salinity",
            "moles_of_oxygen_per_unit_mass_in_sea_water",
        ],
    }
