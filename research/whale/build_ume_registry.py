#!/usr/bin/env python
"""Build the authoritative whale / large-cetacean mortality registry.

This is the *label* backbone for the Cetacean Mortality Initiative: the
cause-attributed record of significant whale die-offs, curated from NOAA Fisheries'
Marine Mammal Unusual Mortality Event (UME) program plus documented (not-yet-formally-
declared) mortality events such as the 2024-2026 domoic-acid bloom whale deaths.

WHY a hand-curated table and not an API pull: UME *declarations* are authoritative,
low-volume (72 total since 1991; ~11 involve large whales), and carry the one thing the
occurrence APIs do NOT — an investigated **cause of death**. That cause attribution is the
entire point of "why are whales dying," so it is encoded here with per-row source URLs and
an honest `formally_declared_ume` flag. The high-volume spatiotemporal corpus (where/when
whales are) comes from the OBIS/GBIF adapters; this table says *why they died*.

Honesty contract (see research/whale/MISSION.md):
- `confirmed_whale_deaths` counts WHALES only. The 2024-2026 DA event killed hundreds of
  sea lions/dolphins; only the 2 confirmed *whale* necropsies are counted here.
- `formally_declared_ume=False` marks documented-but-not-declared events — do not let them
  inflate "declared UME" statistics.

Writes only to research/whale/data/whale_ume_registry.{csv,parquet}.

Sources (per row in `source_url`):
- NOAA Active & Closed UMEs: fisheries.noaa.gov/national/marine-life-distress/active-and-closed-unusual-mortality-events
- 2019-2023 Gray Whale UME: fisheries.noaa.gov/national/marine-life-distress/2019-2021-gray-whale-unusual-mortality-event-along-west-coast-and
- 2024-2026 CA domoic-acid bloom whale deaths: marinemammalcenter.org / NOAA Fisheries feature stories (Jan & Apr 2025 necropsies)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data"

# Cause taxonomy used across the initiative (keep stable; join key for attribution work).
#   DA            = domoic-acid / Pseudo-nitzschia HAB biotoxin
#   VESSEL        = vessel/ship strike
#   ENTANGLE      = fishing-gear entanglement
#   STARVATION    = malnutrition / prey-access / ecological factors
#   DISEASE       = infectious disease (e.g. morbillivirus)
#   UNDETERMINED  = investigated, cause not established
NOAA_UME = "https://www.fisheries.noaa.gov/national/marine-life-distress/active-and-closed-unusual-mortality-events"
GRAYWHALE = "https://www.fisheries.noaa.gov/national/marine-life-distress/2019-2021-gray-whale-unusual-mortality-event-along-west-coast-and"
DA_BLOOM = "https://www.fisheries.noaa.gov/feature-story/toxic-algal-bloom-suspected-dolphin-and-sea-lion-deaths-southern-california"

RECORDS = [
    # --- Pacific / West Coast / California (the in-scope events) ---
    dict(event="West Coast Gray Whale UME", species="gray whale", start_year=2019, end_year=2023,
         region="Pacific (CA/OR/WA + AK)", ca_relevant=True, status="closed",
         formally_declared_ume=True, confirmed_whale_deaths=690,
         primary_cause="STARVATION", cause_note="NOAA: 'ecological factors' — malnutrition / "
         "reduced prey access; coincident with Arctic feeding-ground change.", source_url=GRAYWHALE),
    dict(event="Pacific Gray Whale UME", species="gray whale", start_year=1999, end_year=2000,
         region="California / Oregon / Washington", ca_relevant=True, status="closed",
         formally_declared_ume=True, confirmed_whale_deaths=651,
         primary_cause="UNDETERMINED", cause_note="Undetermined; many emaciated (malnutrition "
         "suspected). Prior eastern-Pacific gray-whale die-off.", source_url=NOAA_UME),
    dict(event="California Blue Whale UME", species="blue whale", start_year=2007, end_year=2007,
         region="California (Pacific)", ca_relevant=True, status="closed",
         formally_declared_ume=True, confirmed_whale_deaths=6,
         primary_cause="VESSEL", cause_note="NOAA: human interaction — ship strikes in the "
         "Santa Barbara Channel shipping lanes.", source_url=NOAA_UME),
    dict(event="Alaska & BC Large Whale UME", species="fin & humpback whale", start_year=2015,
         end_year=2016, region="Gulf of Alaska / British Columbia", ca_relevant=False,
         status="closed", formally_declared_ume=True, confirmed_whale_deaths=52,
         primary_cause="UNDETERMINED", cause_note="Undetermined; secondary ecological factors — "
         "coincided with NE-Pacific marine heatwave ('the Blob') and a major Pseudo-nitzschia "
         "HAB. Partial relevance to CA via the same warm-anomaly/HAB regime.", source_url=NOAA_UME),
    # --- Documented but NOT a formally declared *whale* UME (honest flag = False) ---
    dict(event="2024-2026 CA Domoic-Acid Bloom (whale deaths)", species="humpback & minke whale",
         start_year=2024, end_year=2026, region="Southern → Central California", ca_relevant=True,
         status="active", formally_declared_ume=False, confirmed_whale_deaths=2,
         primary_cause="DA", cause_note="4th consecutive Pseudo-nitzschia bloom year. Confirmed "
         "whale necropsies: humpback (Huntington Beach, Jan 2025), minke (Long Beach, Apr 2025). "
         "Hundreds of sea lions/dolphins also killed (not counted here). THIS is the event the "
         "lab's existing DA/C-HARM model directly forecasts.", source_url=DA_BLOOM),
    # --- Atlantic large-whale UMEs (out of CA scope; kept for completeness/baseline) ---
    dict(event="North Atlantic Right Whale UME", species="North Atlantic right whale",
         start_year=2017, end_year=2026, region="Atlantic (US & Canada)", ca_relevant=False,
         status="active", formally_declared_ume=True, confirmed_whale_deaths=None,
         primary_cause="VESSEL", cause_note="Vessel strike + rope entanglement (human "
         "interaction). Critically endangered population.", source_url=NOAA_UME),
    dict(event="Atlantic Humpback Whale UME", species="humpback whale", start_year=2016,
         end_year=2026, region="Atlantic coast", ca_relevant=False, status="active",
         formally_declared_ume=True, confirmed_whale_deaths=None,
         primary_cause="VESSEL", cause_note="Suspect human interaction (vessel strike).",
         source_url=NOAA_UME),
    dict(event="Atlantic Minke Whale UME", species="minke whale", start_year=2017, end_year=2026,
         region="Atlantic coast", ca_relevant=False, status="active",
         formally_declared_ume=True, confirmed_whale_deaths=None,
         primary_cause="ENTANGLE", cause_note="Suspect human interaction (entanglement) / "
         "infectious disease.", source_url=NOAA_UME),
    # --- SENTINEL pinniped/dolphin events (added at Whale 1's request) -------------------
    # Sea lions are the DA SENTINEL species — the published pDA↔stranding model is on them.
    # These anchor the DA→mortality mechanism that the whale deaths are too few to establish alone.
    dict(event="CA Sea-Lion UME (emaciated pups)", species="California sea lion", taxon_group="pinniped",
         start_year=2013, end_year=2016, region="California (Pacific)", ca_relevant=True, status="closed",
         formally_declared_ume=True, confirmed_whale_deaths=None,
         primary_cause="STARVATION", cause_note="NOAA 'ecological factors': emaciated PUPS from prey "
         "collapse / warm-water (Blob) — NOT primarily domoic acid (DA was a co-factor in 2015 adults "
         "only). Kept as a SECOND starvation negative-control, honestly labeled. n≈8,122.",
         source_url=NOAA_UME),
    dict(event="SoCal Domoic-Acid Mass Stranding 2023", species="sea lions + dolphins", taxon_group="mixed",
         start_year=2023, end_year=2023, region="Southern California", ca_relevant=True, status="closed",
         formally_declared_ume=False, confirmed_whale_deaths=None,
         primary_cause="DA", cause_note="DA-confirmed mass marine-mammal stranding (~1,000 sea lions + "
         "dolphins) during a severe Pseudo-nitzschia bloom. The DA sentinel signal in pinnipeds/odontocetes.",
         source_url=DA_BLOOM),
    dict(event="SoCal Domoic-Acid Event 2024-2025", species="sea lions + dolphins", taxon_group="mixed",
         start_year=2024, end_year=2025, region="Southern California", ca_relevant=True, status="active",
         formally_declared_ume=False, confirmed_whale_deaths=None,
         primary_cause="DA", cause_note="Continuation of the 2024-2026 bloom in the sentinel species "
         "(hundreds of sea lions/dolphins) — same event window as the confirmed whale necropsies.",
         source_url=DA_BLOOM),
]


def build() -> pd.DataFrame:
    df = pd.DataFrame(RECORDS)
    # taxon_group defaults to 'whale' for the original whale-only rows; sentinel rows set their own.
    if "taxon_group" not in df.columns:
        df["taxon_group"] = "whale"
    df["taxon_group"] = df["taxon_group"].fillna("whale")
    df["duration_years"] = df["end_year"] - df["start_year"] + 1
    # Stable column order for the join contract.
    cols = ["event", "species", "taxon_group", "start_year", "end_year", "duration_years", "region",
            "ca_relevant", "status", "formally_declared_ume", "confirmed_whale_deaths",
            "primary_cause", "cause_note", "source_url"]
    return df[cols].sort_values(["ca_relevant", "start_year"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT_DIR / "whale_ume_registry.csv", index=False)
    try:
        df.to_parquet(OUT_DIR / "whale_ume_registry.parquet", index=False)
    except Exception as e:  # parquet engine optional
        print(f"[warn] parquet skipped: {e}")
    ca = df[df["ca_relevant"]]
    print(f"Wrote {len(df)} whale mortality events ({len(ca)} California-relevant) to {OUT_DIR}")
    print("\nCalifornia-relevant events by primary cause:")
    print(ca.groupby("primary_cause")["event"].count().to_string())
    print("\nIn-scope (CA) events:")
    for _, r in ca.iterrows():
        decl = "declared" if r.formally_declared_ume else "DOCUMENTED-not-declared"
        n = "" if pd.isna(r.confirmed_whale_deaths) else f", n={int(r.confirmed_whale_deaths)}"
        print(f"  {r.start_year}-{r.end_year}  {r.species:24s} {r.primary_cause:12s} [{decl}{n}]")


if __name__ == "__main__":
    main()
