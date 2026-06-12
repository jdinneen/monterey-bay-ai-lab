#!/usr/bin/env python3
"""Turn the new CIWQS sanitary-sewer-overflow (SSO) record into a causal, leakage-safe
driver for the beach-bacteria benchmark — WITH the coverage mitigations baked in.

Why this exists
---------------
`ops/data_fetch.py fetch --source ciwqs_sso` produced 43,546 SSO spill events
(data/external_curated/ciwqs_sso/ciwqs_sso.parquet). Two facts force how it can be used
honestly (see reports/data_fetch/CRITIC_PROOF_REPORT.md):

  * Coverage ends in 2015 (dense 2007-2015). The operational benchmark tests on >=2022,
    where SSO is entirely absent — so SSO can only be evaluated *inside* its coverage
    window. We therefore (a) emit a per-region daily feature table, and (b) attach a
    `sso_observed` coverage MASK so a downstream join never imputes "no spill record" as
    "zero spills".
  * 57% of rows lack lat/lon and WDID self-join recovery is 0%. But the RWQCB `region`
    code is present on 99.9% of rows, so we aggregate to **RWQCB region x day** — a robust
    spatial unit that uses every row — and join beaches by a coastal county -> region map.

Causality
---------
Every feature for a sample on date D uses only spills with `spill_date < D`
(merge_asof backward with allow_exact_matches=False), so a spill reported the same day as
(but after) a water sample cannot feed that sample's label.

Outputs
-------
- data/external_curated/ciwqs_sso/sso_region_daily_features.parquet
- reports/data_fetch/ciwqs_sso/sso_features_coverage.json

`add_sso_features(df, sso_path)` is importable by the A/B experiment.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SSO = PROJECT_ROOT / "data" / "external_curated" / "ciwqs_sso" / "ciwqs_sso.parquet"
OUT_FEATURES = PROJECT_ROOT / "data" / "external_curated" / "ciwqs_sso" / "sso_region_daily_features.parquet"
OUT_COVERAGE = PROJECT_ROOT / "reports" / "data_fetch" / "ciwqs_sso" / "sso_features_coverage.json"

# Coastal CA county -> RWQCB region code (matches the SSO `region` field).
# Approximate where a county spans boards (Santa Barbara mostly 3; Orange mostly 8);
# documented so a reviewer can see the assumption. Inland-only counties are omitted.
COASTAL_COUNTY_RWQCB = {
    "Del Norte": "1", "Humboldt": "1", "Mendocino": "1", "Sonoma": "1",
    "Marin": "2", "San Francisco": "2", "San Mateo": "2", "Alameda": "2",
    "East Bay Parks District": "2", "Contra Costa": "2", "Solano": "2", "Napa": "2",
    "Santa Clara": "2",
    "Santa Cruz": "3", "San Benito": "3", "Monterey": "3", "San Luis Obispo": "3",
    "Santa Barbara": "3",
    "Ventura": "4", "Los Angeles": "4", "Long Beach City": "4",
    "Orange": "8",
    "San Diego": "9",
}

SSO_FEATS = ["sso_cnt_7d", "sso_cnt_30d", "sso_gal_30d_log", "days_since_sso", "sso_observed"]

_GAL_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*gallons", re.IGNORECASE)


def parse_gallons(desc) -> float:
    if not isinstance(desc, str):
        return np.nan
    m = _GAL_RE.search(desc)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def load_sso(sso_path: Path) -> pd.DataFrame:
    sso = pd.read_parquet(sso_path)
    sso["spill_date"] = pd.to_datetime(sso["spill_date"], errors="coerce", utc=True).dt.tz_localize(None)
    sso = sso.dropna(subset=["spill_date", "region"]).copy()
    sso["region"] = sso["region"].astype(str)
    sso["gallons"] = sso["description"].map(parse_gallons)
    return sso


def build_region_daily(sso: pd.DataFrame) -> pd.DataFrame:
    """Per RWQCB-region daily spill series with rolling causal features.

    Resampled to a continuous daily grid per region so every calendar day carries the
    correct as-of-that-day rolling counts and an incrementing days-since-last-spill.
    """
    daily = (sso.groupby(["region", sso["spill_date"].dt.normalize()])
                .agg(cnt=("spill_id", "size"), gal=("gallons", "sum"))
                .rename_axis(["region", "date"]).reset_index())
    parts = []
    for reg, g in daily.groupby("region"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        s = g.set_index("date").reindex(idx)
        s["cnt"] = s["cnt"].fillna(0.0)
        s["gal"] = s["gal"].fillna(0.0)
        s["region"] = reg
        s["sso_cnt_7d"] = s["cnt"].rolling(7, min_periods=1).sum()
        s["sso_cnt_30d"] = s["cnt"].rolling(30, min_periods=1).sum()
        s["sso_gal_30d_log"] = np.log1p(s["gal"].rolling(30, min_periods=1).sum())
        # days since the last day that actually had a spill
        last = np.where(s["cnt"].to_numpy() > 0)[0]
        dss = np.full(len(s), np.nan)
        if len(last):
            ptr = -1
            cur = np.nan
            li = set(last.tolist())
            for i in range(len(s)):
                if i in li:
                    cur = 0.0
                elif not np.isnan(cur):
                    cur += 1.0
                dss[i] = cur
        s["days_since_sso"] = dss
        s["date"] = s.index
        parts.append(s.reset_index(drop=True))
    out = pd.concat(parts, ignore_index=True)
    return out[["region", "date", *[c for c in SSO_FEATS if c != "sso_observed"]]]


def _coverage_by_region(sso: pd.DataFrame) -> dict:
    cov = {}
    for reg, g in sso.groupby("region"):
        cov[reg] = {"min": str(g["spill_date"].min().date()), "max": str(g["spill_date"].max().date())}
    return cov


def add_sso_features(df: pd.DataFrame, sso_path: Path = DEFAULT_SSO) -> pd.DataFrame:
    """Join causal SSO region-day features to beach station-days.

    `df` must have columns: station_id, county, sample_date, exceed. Returns df with the
    SSO_FEATS columns added (NaN/0 where a county has no RWQCB mapping or is outside the
    region's coverage window — with `sso_observed` flagging the latter).
    """
    sso = load_sso(sso_path)
    region_daily = build_region_daily(sso)
    coverage = _coverage_by_region(sso)

    work = df.copy()
    work["sample_date"] = pd.to_datetime(work["sample_date"]).dt.tz_localize(None)
    work["region_rwqcb"] = work["county"].map(COASTAL_COUNTY_RWQCB)

    left = work.dropna(subset=["region_rwqcb"]).sort_values("sample_date")
    rd = region_daily.sort_values("date")
    # strictly-prior join: allow_exact_matches=False => last region-day BEFORE sample_date
    merged = pd.merge_asof(
        left, rd,
        left_on="sample_date", right_on="date",
        left_by="region_rwqcb", right_by="region",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.drop(columns=["date", "region"], errors="ignore")

    # coverage mask: 1 if sample_date is within the region's observed spill window
    def _observed(row):
        c = coverage.get(row["region_rwqcb"])
        if not c:
            return 0
        return int(str(c["min"]) <= str(pd.Timestamp(row["sample_date"]).date()) <= str(c["max"]))

    merged["sso_observed"] = merged.apply(_observed, axis=1)
    # where unobserved, zero the count features (mask carries the "unknown" signal)
    for c in ["sso_cnt_7d", "sso_cnt_30d", "sso_gal_30d_log"]:
        merged.loc[merged["sso_observed"] == 0, c] = 0.0
    merged["days_since_sso"] = merged["days_since_sso"].fillna(-1.0)

    # re-attach rows whose county had no RWQCB mapping (SSO feats absent)
    unmapped = work[work["region_rwqcb"].isna()].copy()
    for c in SSO_FEATS:
        unmapped[c] = 0.0 if c != "days_since_sso" else -1.0
    return pd.concat([merged, unmapped], ignore_index=True).sort_values(
        ["station_id", "sample_date"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sso", default=str(DEFAULT_SSO))
    ap.add_argument("--out", default=str(OUT_FEATURES))
    args = ap.parse_args(argv)

    sso = load_sso(Path(args.sso))
    region_daily = build_region_daily(sso)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    region_daily.to_parquet(args.out, index=False)

    coverage = {
        "source": "ciwqs_sso",
        "rows_in": int(len(sso)),
        "regions": sorted(sso["region"].unique().tolist()),
        "region_coverage": _coverage_by_region(sso),
        "gallons_parse_rate": round(float(sso["gallons"].notna().mean()), 4),
        "region_daily_rows": int(len(region_daily)),
        "features": SSO_FEATS,
        "coastal_county_rwqcb_map": COASTAL_COUNTY_RWQCB,
        "note": "Coverage ends 2015; use only inside the coverage window and carry "
                "sso_observed as a mask. Region x day aggregation uses every row "
                "(region present on 99.9%), unlike the 43%-present lat/lon.",
    }
    OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    OUT_COVERAGE.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"[build_sso_features] wrote {args.out}  ({len(region_daily):,} region-days)")
    print(f"[build_sso_features] coverage -> {OUT_COVERAGE}")
    print(f"  regions: {coverage['regions']}")
    print(f"  gallons parse rate: {coverage['gallons_parse_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
