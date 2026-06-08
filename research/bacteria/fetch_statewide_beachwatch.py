#!/usr/bin/env python3
"""Pull the STATEWIDE California beach water-quality record (not just Monterey).

The original lovers_point predictor queried WHERE county='Monterey' -> 15 sites / 244
exceedance events, which is too few to validate. The same BigQuery table
(<MBARI_GCP_PROJECT>.blue_current_core_v2.california_beach_sample_observations) is statewide:
~830 sites / ~60,616 events. This pulls the full long-format record (dates >= 2005) so a
hierarchical / pooled model with an HONEST spatial holdout (Lovers Point held out, learning
from every other site) becomes possible.

Output (bacteria_results/statewide/, gitignored data):
  - statewide_beach_observations.parquet  (long format, one row per property per sample)
  - statewide_advisories.parquet
  - statewide_event_inventory.csv         (per-county sites / sample-days / events)
Read-only public-record pull via the `bq` CLI against the configured GCP project.
"""
from __future__ import annotations
import io, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

# Set MBARI_GCP_PROJECT; no specific cloud project is baked into the published source.
PROJECT = os.environ.get("MBARI_GCP_PROJECT", "your-gcp-project")
DS = "blue_current_core_v2"
OBS = f"`{PROJECT}.{DS}.california_beach_sample_observations`"
ADV = f"`{PROJECT}.{DS}.california_beach_advisory_events`"
PROPS = "('prop_enterococcus','prop_total_coliform','prop_fecal_coliform','prop_e_coli')"
START = "2005-01-01"
OUT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parents[2])) / "bacteria_results" / "statewide"


def bq_query(sql: str, max_rows: int = 100_000, timeout: int = 600) -> pd.DataFrame:
    bq = shutil.which("bq.cmd") or shutil.which("bq")
    if not bq:
        raise RuntimeError("bq CLI not found (need authenticated Google Cloud SDK).")
    cmd = [bq, "query", "--use_legacy_sql=false", "--format=csv",
           f"--max_rows={max_rows}", " ".join(sql.split())]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return pd.read_csv(io.StringIO(r.stdout))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/3] statewide long-format observations (dates >= %s)..." % START, flush=True)
    obs = bq_query(f"""
        SELECT sample_date, county, beach_name, station_name, station_id,
               source_parameter, property_id, result_comparator, result_value_numeric
        FROM {OBS}
        WHERE result_value_numeric IS NOT NULL
          AND property_id IN {PROPS}
          AND sample_date >= '{START}'
        ORDER BY county, beach_name, station_name, sample_date, property_id
    """, max_rows=2_000_000)
    obs["sample_date"] = pd.to_datetime(obs["sample_date"], errors="coerce")
    # Some station_name/station_id values are numeric -> mixed-type object columns that break
    # parquet. Coerce all identifier/text columns to pandas nullable string.
    for c in ["county", "beach_name", "station_name", "station_id", "source_parameter",
              "property_id", "result_comparator"]:
        if c in obs.columns:
            obs[c] = obs[c].astype("string")
    obs.to_parquet(OUT / "statewide_beach_observations.parquet", index=False)
    print(f"      {len(obs):,} rows, {obs['county'].nunique()} counties -> statewide_beach_observations.parquet")

    print("[2/3] statewide advisories...", flush=True)
    try:
        adv = bq_query(f"""
            SELECT advisory_date, opened_date, county, beach_name, station_name,
                   advisory_type, advisory_cause
            FROM {ADV}
            WHERE advisory_date >= '{START}'
            ORDER BY county, advisory_date
        """, max_rows=500_000)
        for c in ["county", "beach_name", "station_name", "advisory_type", "advisory_cause"]:
            if c in adv.columns:
                adv[c] = adv[c].astype("string")
        for c in ["advisory_date", "opened_date"]:
            if c in adv.columns:
                adv[c] = pd.to_datetime(adv[c], errors="coerce")
        adv.to_parquet(OUT / "statewide_advisories.parquet", index=False)
        print(f"      {len(adv):,} advisory rows -> statewide_advisories.parquet")
    except Exception as e:
        print(f"      advisories skipped: {e}")

    print("[3/3] per-county event inventory...", flush=True)
    inv = bq_query(f"""
        WITH piv AS (
          SELECT county, beach_name, station_name, sample_date,
            MAX(IF(property_id='prop_enterococcus', result_value_numeric, NULL)) ent,
            MAX(IF(property_id='prop_fecal_coliform', result_value_numeric, NULL)) fc,
            MAX(IF(property_id='prop_total_coliform', result_value_numeric, NULL)) tc
          FROM {OBS}
          WHERE result_value_numeric IS NOT NULL AND property_id IN {PROPS}
            AND sample_date >= '{START}'
          GROUP BY county, beach_name, station_name, sample_date
        )
        SELECT county,
          COUNT(DISTINCT FORMAT('%s|%s', IFNULL(beach_name,''), IFNULL(station_name,''))) sites,
          COUNT(*) sample_days,
          COUNTIF((ent>=104) OR (fc>=400) OR (tc>=10000) OR (tc>=1000 AND fc>0 AND tc>0 AND fc/tc>=0.1)) events
        FROM piv GROUP BY county ORDER BY events DESC
    """)
    inv.to_csv(OUT / "statewide_event_inventory.csv", index=False)
    print(f"      {inv['sites'].sum()} sites / {inv['sample_days'].sum():,} sample-days / "
          f"{inv['events'].sum():,} events across {len(inv)} counties")
    print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
