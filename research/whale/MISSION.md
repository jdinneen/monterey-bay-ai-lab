# Cetacean Mortality Initiative — *why are the whales dying?*

A research thread **inside Monterey Bay AI Lab** (per the identity invariant in `AGENTS.md`,
the project is Monterey Bay AI Lab; this is a thread within it, not a new project/brand).

> **Question (stakeholder, Jon, 2026-06-11):** *Find out why the whales are dying.*
> Two agents run this overnight: **Whale 1** and **Whale 2**, coordinating through this file
> and the `ops/agent_lock.py` cooperative locks.

## The thesis (why this is value-gated, not bloat)

California whale mortality has **three documented, separable drivers**, and this lab already
owns infrastructure for the first one:

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
today. (5) Genuinely new and it *answers the question a stakeholder actually asked.*
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

## Lane split — RECONCILED with Project Leviathan (single source of truth)

> **Canonical charter:** `reports/whale_mortality/CHARTER.md` (Whale 1's "Project Leviathan").
> Whale 1 launched concurrently and proposed the opposite cut from this file's first draft;
> I (Whale 2) **accepted Whale 1's split** to avoid a lane war. This doc now mirrors it.
> Coordination channel: `reports/data_fetch/fetch_learnings.jsonl` (source `_WHALE`).

### Whale 1 — **DATA ACQUISITION + LABELS** (owns `reports/whale_mortality/`, OBIS adapter, panel)
- [~] OBIS marine-mammal adapter (`obis_marine_mammals.py`) — strandings + sightings *(in progress)*
- [x] **UME events table — already built by Whale 2 as a gift:** `research/whale/data/whale_ume_registry.csv`
  (reuse, don't rebuild — honest `formally_declared_ume` flag, whale-only death counts)
- [ ] Dated mortality **panel** (counts by region/species) joined to pDA + C-HARM + SST → `data/external_curated/`

### Whale 2 — **MODELING + EVIDENCE** (this agent; owns `research/whale/` analysis files)
Answer *why*, honestly, leakage-safe, against an honest baseline.
- [x] UME cause-registry (built; handed to Whale 1's label lane)
- [x] **DA↔mortality evidence harness** (`da_mortality_evidence.py`, 6 tests green) — pier pDA
  is **2.09× the seasonal norm (z=10)** in the 2024–2026 DA window, **below baseline in the
  2019–2023 starvation control**; independent Pseudo-nitzschia cell counts corroborate
  directionally (1.17×). C-HARM can't corroborate (record too short → no pre-bloom baseline).
  Report: `reports/WHY_WHALES_ARE_DYING.md`. **Honest finding: multi-cause — largest recent
  die-off (gray UME) was starvation, newest deaths are DA-linked.**
- [x] Panel contract posted to `_WHALE` channel (so Whale 1's panel lands model-ready)
- [x] Sentinel pinniped/dolphin DA events added to registry (taxon_group) + per-window pDA test;
  honest finding that 2013–2016 sea-lion "starvation" UME overlaps the real 2015 Blob DA bloom
- [x] **Driver→mortality model harness BUILT + tested** (`mortality_model.py`, 5 tests) — leakage-safe
  lagged-pDA features, beats per-region seasonal climatology on synthetic DA-driven panel, reports an
  honest null when no link. **Runs the moment Whale 1's panel lands:** `python research/whale/mortality_model.py --panel <path>`
- [ ] *(blocked on panel)* Run model on Whale 1's real iNat dead-cetacean panel; reproduce the
  sea-lion↔pDA link as a positive control; calibrate; **name confounders (vessel/entanglement/blob)**
- [ ] Honest report → `research/whale/reports/WHY_WHALES_ARE_DYING.md` (effect size + uncertainty)

### Dropped (anti-redundancy, value gate Q5)
- `obis_cetacea.py` (Whale 1 owns OBIS occurrence acquisition) — not built.

## Coordination log (append; newest last)
- 2026-06-11 20:30 — Whale 2: mission opened; OBIS/GBIF/NOAA-UME sources verified live; UME
  registry built.
- 2026-06-11 20:43 — Whale 2: discovered Whale 1 live on the same data lane; **accepted Whale 1's
  split**, took MODELING+EVIDENCE, dropped redundant OBIS adapter, gifted the UME registry to the
  label lane, reconciled this doc to defer to `CHARTER.md`. Starting the DA↔mortality evidence
  harness on habmap/C-HARM + UME windows.
