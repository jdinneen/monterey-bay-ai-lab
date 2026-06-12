"""Whale-mortality thread — does the open mortality signal track prey + ocean state?

Joins three staged sources at ANNUAL resolution (the native cadence of the forage index):
  * marine_mammal_mortality  -> effort-normalized cetacean dead-fraction (dead / all obs)
  * cciea_forage             -> Total Krill + Adult Anchovy biomass anomaly (central CA)
  * mur_sst                  -> annual-mean SST + anomaly at the M1 point

Tests NOAA's gray-whale-UME hypothesis (feeding-ground prey decline -> malnutrition):
prior-year forage collapse and SST warm anomalies should precede higher mortality.

Honest by construction: the dead-fraction is effort-normalized (robust to iNaturalist
growth); N is small (~13 yrs) so we report Spearman rho + N, not p-hacked claims; the
2020-21 COVID observer-behavior confound is flagged, not hidden.

Run:  PYTHONPATH=. python research/data_lab/whale_mortality_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CUR = Path("data/external_curated")


def _year(s):
    return pd.to_datetime(s, utc=True).dt.year


def mortality_by_year() -> pd.DataFrame:
    df = pd.read_parquet(CUR / "marine_mammal_mortality" / "marine_mammal_mortality.parquet")
    df["year"] = _year(df["time"])
    g = df.groupby("year").agg(
        all_obs=("observation_id", "nunique"),
        dead=("dead_flag", lambda s: int((s == 1).sum())),
    )
    g["dead_frac"] = g["dead"] / g["all_obs"]
    return g.reset_index()


def forage_by_year(groups=("Total Krill", "Adult Anchovy")) -> pd.DataFrame:
    df = pd.read_parquet(CUR / "cciea_forage" / "cciea_forage.parquet")
    df["year"] = _year(df["time"])
    sub = df[df["species_group"].isin(groups)]
    return (sub.pivot_table(index="year", columns="species_group", values="mean_cpue")
            .add_prefix("forage_").reset_index())


def sst_by_year() -> pd.DataFrame:
    df = pd.read_parquet(CUR / "mur_sst" / "mur_sst.parquet")
    df["year"] = _year(df["time"])
    g = df.groupby("year")["sst_c"].mean().rename("sst_mean").reset_index()
    g["sst_anom"] = g["sst_mean"] - g["sst_mean"].mean()
    return g


def main() -> None:
    mort = mortality_by_year()
    forage = forage_by_year()
    sst = sst_by_year()

    panel = mort.merge(forage, on="year", how="left").merge(sst, on="year", how="left")
    panel = panel.sort_values("year").reset_index(drop=True)

    # lag-1 prey/ocean -> this-year mortality (the malnutrition mechanism is delayed)
    for c in [c for c in panel.columns if c.startswith("forage_")] + ["sst_anom"]:
        panel[f"{c}_lag1"] = panel[c].shift(1)

    print("\n=== ANNUAL PANEL: mortality vs prey vs ocean (central CA) ===")
    show = ["year", "all_obs", "dead", "dead_frac", "forage_Total Krill",
            "forage_Adult Anchovy", "sst_anom"]
    show = [c for c in show if c in panel.columns]
    with pd.option_context("display.float_format", lambda v: f"{v:6.3f}", "display.width", 140):
        print(panel[show].to_string(index=False))

    print("\n=== Spearman rank corr with dead_frac (negative prey / positive warm = hypothesis) ===")
    drivers = [c for c in panel.columns
               if c.startswith("forage_") or c.startswith("sst")]
    for c in drivers:
        d = panel[["dead_frac", c]].dropna()
        if len(d) >= 5:
            rho = d["dead_frac"].corr(d[c], method="spearman")
            print(f"  dead_frac ~ {c:28s}  rho={rho:+.3f}  N={len(d)}")

    print("\nCAVEATS: N~13yr (small); 2020-21 overlaps COVID observer-behavior shift; "
          "iNat 'dead' annotation is voluntary. Effort-normalization controls platform growth, "
          "not reporting-behavior change. Treat as corroborating, not calibrated.")


if __name__ == "__main__":
    main()
