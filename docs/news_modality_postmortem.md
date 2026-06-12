# News/Event Modality — Postmortem (2026-06-10)

**Decision: removed from the working tree.** The Monterey Bay "news agent" did not
improve the bacteria exceedance model. This document is the durable record of what
was built, what was tested, and why it was retired, so the investigation does not
get repeated. Tracked code remains recoverable from git history; the
never-committed config reconstructed during this investigation is embedded below.

## What was built (removed)

| File | Role |
|---|---|
| `ops/fetch_news_events.py` | GDELT DOC API query planner + executor (dry-run-first) |
| `ops/fetch_seed_news_documents.py` | direct URL → bronze document fetcher |
| `ops/curate_news_events.py` | rules classifier: bronze docs → silver events (simple schema) |
| `ops/curate_news_event_evidence.py` | evidence-JSON → silver articles/events (rich schema; produced the on-disk `events.parquet`) |
| `ops/build_news_event_features.py` | silver events → leakage-safe hourly driver features |
| `ops/audit_news_event_drift.py` | drift auditor |
| `tests/test_*news*` | unit tests for the above |
| `research/news_events_taxonomy.json`, `research/news_events_sources.json` | config (was missing; reconstructed this session — see appendix) |
| `lakehouse/silver/news_events/`, `news_articles/`, `nn_cache/news_drivers_*`, `reports/news_events_*` | artifacts/data |

## Why it was retired (evidence)

1. **Nothing consumed its output.** `nn_cache/news_drivers_hourly.parquet` was read
   by no model — not the forecast, not `merge_driver_tables`, not the bacteria
   predictor. Dead end.
2. **Too sparse.** Only **30 hand-seeded events across 13 years**; the most-covered
   hourly feature was non-zero in ~2% of hours.
3. **GDELT DOC API is IP-blocked from this network.** A 2017→2026 chunked backfill
   returned **0 documents**; every query hit HTTP 429 through full exponential
   backoff. (If news breadth is ever needed: use the GDELT bulk GKG CSV exports at
   data.gdeltproject.org, not the rate-limited DOC API.)
4. **The authoritative structured equivalent did not help either.** The news
   taxonomy's causal content (sewage spills, line breaks, rain runoff) already lands
   in BigQuery `california_beach_advisory_events.advisory_cause`, already fetched by
   the bacteria model. Tried as leakage-safe cause-typed rolling features
   (spill/rain, shift(1), 7/30/90d). **Powered A/B on the full multi-site 2024+
   holdout (691 samples, 24 positive events):**

   | metric | baseline | +cause-typed | delta |
   |---|---|---|---|
   | avg_precision | 0.1230 | 0.0863 | **−0.037** |
   | roc_auc | 0.6808 | 0.6672 | **−0.014** |

   It slightly *hurt*. The rain-runoff signal is already captured by the rich ASOS
   rainfall features (first-flush, wet-days, multi-window sums); the unique residual
   (≈25 sewage-spill events total) is too rare to register. **Redundant** → fails the
   `docs/VALUE_GATE.md` "must beat a real baseline / reject redundant work" test.

## If revisited

The only defensible use is **operational** (a real-time coastal-incident digest for
situational awareness), which does not need to beat a model metric — not as an ML
feature for the bacteria or forecast models.

## Appendix — reconstructed config (so the work is not lost)

Taxonomy `2026-06-08.v1` — geofence + per-event-type terms recovered from the
dry-run `planned_queries.json`. Feature windows: `[24, 72, 168, 336, 720]` hours.

Geofence primary places: Monterey Bay, Monterey, Pacific Grove, Carmel, Seaside,
Marina, Moss Landing, Elkhorn Slough, Salinas River, Santa Cruz, Watsonville,
Capitola, Big Sur, Monterey Canyon, Monterey Bay National Marine Sanctuary.

Event classes (positive terms):
- **spill_release**: spill, oil/petroleum/chemical spill, hazardous material, hazmat, sewage/wastewater spill, sanitary sewer overflow, discharge, oil sheen, Cal OES, National Response Center
- **beach_closure**: beach closure/closed, water quality advisory, unsafe for contact, bacteria, enterococcus, fecal indicator, runoff advisory, sewer overflow
- **hab_biotoxin**: harmful algal bloom, HAB, red tide, domoic acid, Pseudo-nitzschia, shellfish advisory/closure, mussel quarantine, crab closure, marine biotoxin, ASP, PSP
- **wildlife_mortality**: dead whale, whale carcass, marine mammal stranding, sea lion poisoning, fish kill, dead fish, unusual mortality, stranded dolphin/seal
- **shark_public_safety**: shark attack/bite/warning/advisory, beach warning
- **storm_runoff**: atmospheric river, coastal flood, high surf, storm runoff, flood advisory, harbor closure, storm drain, debris flow, large swell
- **ocean_climate_anomaly**: marine heatwave, the Blob, ocean heatwave, SST anomaly, warm water anomaly, upwelling failure, hypoxia, low dissolved oxygen, ocean acidification, El Nino, La Nina

Source registry `2026-06-08.v1` (only `gdelt_doc_gkg_event` had an implemented
fetcher; the rest are authoritative agency sources needing dedicated fetchers):
gdelt_doc_gkg_event (news_search), caloes_spill_release_reporting (state),
cdfw_news_and_reporting (state), cdph_marine_biotoxin (state),
monterey_county_beach_water_quality (county), noaa_coastwatch_charm (federal),
noaa_ecoforecasting (federal).
