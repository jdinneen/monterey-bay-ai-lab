#!/usr/bin/env python3
"""Build the marine-mammal MORTALITY PANEL for Project Leviathan — the hand-off to whale-2.

Monthly marine-mammal death counts by California region + taxon group (from the iNaturalist
dead-annotation labels), effort-normalized, joined to the domoic-acid exposure (pier pDA + C-HARM
nowcast) and SST. This is the table whale-2's driver->mortality evidence harness plugs into.

Honesty: deaths are a citizen-science PROXY (iNat dead annotation), effort-normalized by the year's
total observation effort for that taxon (monthly effort not fetched -> within-year effort assumed
roughly uniform; stated, not hidden). DA exposure is provided same-month AND prior-month so the
modeler can respect that exposure precedes death by days-to-weeks (no future leak). Correlational,
not causal; vessel-strike / entanglement / prey-shift confounders are NOT in this panel.

Usage:  python research/whale_mortality/build_mortality_panel.py [--output-dir reports/whale_mortality]
Outputs: data/whale_mortality/mortality_panel.parquet + reports/whale_mortality/mortality_panel.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CUR = ROOT / "data" / "external_curated"
OUT_DATA = ROOT / "data" / "whale_mortality"
OUT_REP = ROOT / "reports" / "whale_mortality"

# California latitude bands -> region label (S = SoCal, C = Central, N = NorCal).
def _region(lat: pd.Series) -> pd.Series:
    return pd.cut(lat, [-90, 34.5, 37.5, 90], labels=["S_SoCal", "C_Central", "N_NorCal"])


def _monthly_region_mean(pq: Path, tcol: str, latcol: str, valcol: str, out_name: str) -> pd.DataFrame:
    """Monthly, per-region mean of a driver variable from a curated source with lat + time."""
    if not pq.exists():
        return pd.DataFrame(columns=["ym", "region", out_name])
    d = pd.read_parquet(pq, columns=[tcol, latcol, valcol])
    d[valcol] = pd.to_numeric(d[valcol], errors="coerce")
    d["lat"] = pd.to_numeric(d[latcol], errors="coerce")
    t = pd.to_datetime(d[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    d = d.assign(ym=t.dt.to_period("M"), region=_region(d["lat"])).dropna(subset=["ym", "region", valcol])
    g = d.groupby(["ym", "region"], observed=True)[valcol].mean().reset_index(name=out_name)
    return g


def build() -> pd.DataFrame:
    # --- mortality (numerator) ---
    dead = pd.read_parquet(CUR / "inat_mammal_mortality" / "inat_mammal_mortality.parquet")
    dead["ym"] = pd.to_datetime(dead["time"], utc=True).dt.tz_localize(None).dt.to_period("M")
    dead["region"] = _region(pd.to_numeric(dead["latitude"], errors="coerce"))
    dead["taxon_group"] = np.where(dead["query_taxon"] == "Cetacea", "cetacean", "pinniped")
    dead = dead.dropna(subset=["ym", "region"])
    panel = (dead.groupby(["ym", "region", "taxon_group"], observed=True)
             .agg(dead_count=("observation_id", "size"),
                  year_effort=("year_effort_total", "first")).reset_index())
    # within-year-uniform effort approximation -> a monthly rate per 1k effort
    panel["dead_per_1k_effort"] = (1000 * panel["dead_count"] / (panel["year_effort"] / 12.0)).round(2)

    # --- domoic-acid exposure + SST drivers (same-month, per region) ---
    pda = _monthly_region_mean(CUR / "habmap_cdph" / "habmap_cdph.parquet",
                               "time", "latitude", "pDA", "pda_habmap")
    charm = _monthly_region_mean(CUR / "c_harm" / "c_harm.parquet",
                                 "time", "latitude", "particulate_domoic", "charm_pda")
    # SST is statewide (mur_sst has no per-record lat) -> one series joined to all regions
    sstpq = CUR / "mur_sst" / "mur_sst.parquet"
    sst = pd.DataFrame(columns=["ym", "sst"])
    if sstpq.exists():
        s = pd.read_parquet(sstpq, columns=["time", "sst_c"])
        st = pd.to_datetime(s["time"], utc=True, errors="coerce").dt.tz_localize(None)
        sst = (s.assign(ym=st.dt.to_period("M"))
               .groupby("ym")["sst_c"].mean().reset_index(name="sst"))

    for drv in (pda, charm):
        panel = panel.merge(drv, on=["ym", "region"], how="left")
    panel = panel.merge(sst, on="ym", how="left")

    # prior-month exposure (exposure precedes death; lets the modeler avoid same-month ambiguity)
    panel = panel.sort_values(["region", "taxon_group", "ym"])
    for col in ["pda_habmap", "charm_pda", "sst"]:
        panel[f"{col}_prev"] = panel.groupby(["region", "taxon_group"], observed=True)[col].shift(1)

    # --- prey / STARVATION driver: NOAA CCIEA central-CA forage anomalies (annual) ---
    # Total Krill is the gray-whale food whose crash drives the malnutrition mechanism; anchovy is
    # a warm-regime marker (not whale food). Annual index -> joined to every month of its year, and
    # applied to all regions (it is a central-CA prey index, stated as such).
    fpq = CUR / "cciea_forage" / "cciea_forage.parquet"
    panel["year"] = panel["ym"].dt.year
    if fpq.exists():
        f = pd.read_parquet(fpq, columns=["time", "species_group", "mean_cpue"])
        f["year"] = pd.to_datetime(f["time"], utc=True, errors="coerce").dt.year
        keep = {"Total Krill": "krill_anom", "Adult Anchovy": "anchovy_anom", "YOY Rockfish": "rockfish_anom"}
        fw = (f[f["species_group"].isin(keep)]
              .pivot_table(index="year", columns="species_group", values="mean_cpue", aggfunc="mean")
              .rename(columns=keep).reset_index())
        panel = panel.merge(fw, on="year", how="left")

    panel["ym"] = panel["ym"].astype(str)
    return panel.reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUT_REP))
    args = ap.parse_args(argv)
    panel = build()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_DATA / "mortality_panel.parquet", index=False)

    yr = panel.copy()
    yr["year"] = yr["ym"].str[:4].astype(int)
    cet = yr[yr["taxon_group"] == "cetacean"].groupby("year")["dead_count"].sum()
    rep = Path(args.output_dir); rep.mkdir(parents=True, exist_ok=True)
    md = ["# Marine-mammal mortality panel (Project Leviathan -> whale-2)", "",
          f"Monthly death counts x {panel['region'].nunique()} CA regions x 2 taxon groups, "
          f"joined to domoic-acid (pier pDA + C-HARM) + SST, same-month and prior-month. "
          f"{len(panel)} rows, {panel['ym'].min()}..{panel['ym'].max()}.", "",
          "Columns: ym, region (S/C/N CA), taxon_group, dead_count, year_effort, "
          "dead_per_1k_effort, pda_habmap(_prev), charm_pda(_prev), sst(_prev).", "",
          "## Cetacean deaths by year (open-data UME reconstruction)", "",
          "| year | cetacean deaths |", "|---|--:|"]
    for y, n in cet.items():
        md.append(f"| {y} | {int(n)} |")
    md += ["", "_Caveats: iNat dead-annotation PROXY; effort-normalized within-year; correlational; "
           "vessel-strike/entanglement/prey confounders not included. whale-2 owns the modeling._"]
    (rep / "mortality_panel.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_DATA / 'mortality_panel.parquet'} ({len(panel)} rows) + {rep / 'mortality_panel.md'}")
    print(panel.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
