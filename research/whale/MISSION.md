# Cetacean Mortality Initiative — *why are the whales dying?*

A research thread **inside Monterey Bay AI Lab** (per the identity invariant in `AGENTS.md`,
the project is Monterey Bay AI Lab; this is a thread within it, not a new project/brand).

**Question:** *Why are California whales dying — and how much of it does the lab's existing
domoic-acid (DA) bloom forecasting actually explain?*

## The thesis (why this is value-gated, not bloat)

California whale mortality has **multiple documented, separable drivers**, and this lab already
owns infrastructure for one of them:

1. **Domoic-acid (DA) biotoxin poisoning** — *Pseudo-nitzschia* harmful algal blooms →
   toxin accumulates in anchovies/sardines → marine mammals. **Confirmed killing whales in
   2025** (humpback, Huntington Beach; minke, Long Beach), the 4th consecutive bloom year.
   **The lab already forecasts exactly this bloom** (`research/hab/`, `research/algae/`,
   C-HARM comparison). This is the fundable bridge.
2. **Vessel strikes** — a leading cause of large-whale death off CA (e.g. the 2007 CA blue
   whale UME). 521 West Coast whale strandings 2020–2025; 2026 on track to be among deadliest.
3. **Entanglement** in fishing gear — chronic, well-documented.
4. **Prey collapse / starvation** tied to marine heatwaves — the 2018–2023 Eastern North
   Pacific Gray Whale UME (690 animals, NOAA cause = "ecological factors").

**Value gate: DO_NOW.** (1) New target — whale mortality is downstream-impact, not redundant
with the bacteria/DA-bloom forecasting we already claim. (2) Additive & reversible — every
artifact is a new file under `research/whale/` or a new fetch adapter writing only to
`data/external_*`; nothing overwrites gold. (3) Tested — adapters + join logic ship with
tests. (4) Valuable now — UME registry, OBIS/GBIF occurrences, and our DA drivers all exist
today. (5) Genuinely new — it *answers a question that was actually asked.*
**Rejected interpretation:** "redo everything we have" — fails Q2 (in-place destruction of
working gold) and Q5 (redundant). We **extend**, we do not rebuild.

## Honesty guardrails (do not regress)

- **Sightings ≠ mortality.** OBIS/GBIF cetacean occurrences are mostly *presence*
  (distribution/exposure). Mortality labels come from the **UME registry** and
  **stranding** records. Never let an occurrence-density model masquerade as a death model.
- **"DA predicts blooms" ≠ "DA explains whale deaths."** We must quantify the
  *DA-attributable fraction* of strandings against a seasonal/climatological baseline, and
  test whether our DA signal *leads* stranding spikes — an honest attribution, with the
  vessel-strike and entanglement fractions reported alongside, not buried.
- Region/era stratified. No pooled number that hides a regime break.

## What ships in this thread

- **UME cause-registry builder** (`build_ume_registry.py`) — assembles the Unusual Mortality
  Event table with an honest `formally_declared_ume` flag and whale-only death counts.
- **DA↔mortality evidence harness** (`da_mortality_evidence.py`, tested) — pier pDA is
  **2.09× the seasonal norm (z=10)** in the 2024–2026 DA window and **below baseline in the
  2019–2023 starvation control**; independent *Pseudo-nitzschia* cell counts corroborate
  directionally (1.17×). C-HARM cannot corroborate (record too short → no pre-bloom baseline).
  **Honest finding: multi-cause — the largest recent die-off (gray-whale UME) was starvation,
  the newest deaths are DA-linked.** A sentinel cross-check finds the 2013–2016 sea-lion
  "starvation" UME overlaps the real 2015 Blob DA bloom.
- **Driver→mortality model harness** (`mortality_model.py`, tested) — leakage-safe
  lagged-pDA features; beats per-region seasonal climatology on a synthetic DA-driven panel and
  reports an honest null when no link exists. Runs against a dated mortality panel:
  `python research/whale/mortality_model.py --panel <path>`.

## Open work (honest status)

- Run the model on a real dated dead-cetacean panel (counts by region/species joined to pDA +
  C-HARM + SST); reproduce the sea-lion↔pDA link as a positive control; calibrate; and **name
  the confounders** (vessel strike / entanglement / marine-heatwave starvation) alongside the
  DA-attributable fraction rather than burying them.
