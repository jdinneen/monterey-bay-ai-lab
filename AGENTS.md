# Agent Directive — MBARI AI Forecasting

> Read this before editing the lakehouse/Databricks-related files. It is binding for
> all agents (Codex, Claude, Gemini, Qwen) operating in this directory.

## CONCURRENCY (binding): claim a file before you edit it

Multiple agents share this one working tree. To avoid clobbering each other, this repo
uses a lightweight cooperative lock: `ops/agent_lock.py`. The rule is simple — **do not
edit a file another live agent has claimed.**

Protocol (every agent, every vendor):
1. Before editing files, claim them:
   `python ops/agent_lock.py claim <paths...> --agent <your-id> --task "<short desc>"`
   Exit code `2` means another agent holds an overlapping file — pick different work or wait.
2. When done (success or abort), release:
   `python ops/agent_lock.py release-mine --agent <your-id>`
3. To see who is working on what: `python ops/agent_lock.py status`

Notes:
- Claims auto-expire after 30 min and refresh while you keep editing, so a crashed agent
  never wedges the tree. If a claim is genuinely dead, reclaim with
  `python ops/agent_lock.py release <path> --force`.
- Claude Code agents are enforced automatically via a `PreToolUse` hook
  (`.claude/settings.json`) that auto-claims on edit and blocks on conflict — no manual
  step needed. **Codex / Gemini / Qwen MUST run the `claim`/`release-mine` commands
  themselves** (wire them into your launcher's pre/post-edit step); the hook does not bind you.
- Locks live under `.agent_locks/` (gitignored). They gate edits, not reads.

## ARCHITECTURE DECISION (binding): build LIKE a lakehouse, do NOT use the Databricks product

The goal is to apply production **lakehouse patterns** on local/portable compute — NOT to
integrate with, deploy to, or depend on the Databricks managed product.

**DO**
- Keep bronze/silver/gold layering, deterministic split contracts, run manifests, and
  portable env vars (`MBARI_PROJECT_ROOT`, `MBARI_SOURCE_PARQUET`, `MBARI_CACHE_DIR`,
  `MBARI_LAKEHOUSE_DIR`).
- Use the portable smoke job `ops/jobs.smoke.json` for remote-environment validation.
- Describe remote targets generically (Spark / Kubernetes / object storage / any managed
  runtime) — never product-specifically.

**DO NOT** (treat any of these as a *defect to remove, not restore*)
- Create or restore `databricks.yml`, `resources/mbari_smoke.job.yml`, or
  `ops/jobs.databricks_smoke.json`.
- Add Databricks CLI / bundle / Unity Catalog / DBR-runtime dependencies, or any
  `DATABRICKS_*` env requirement.
- Frame "Databricks cloud execution" as a release blocker. The only remote gap is a
  product-agnostic portable smoke run.

If an audit finds these product files present or referenced, the correct action is to
**DELETE** them and fix the doc reference — never to re-add them.

Canonical contract doc: `MBARI_PRODUCTION_LAKEHOUSE_CONTRACTS.md`
(supersedes the removed `MBARI_DATABRICKS_LOCAL_CONTRACTS.md`).

## Data/results correctness invariants (audit 2026-06-08 — do not regress)
- Lakehouse metric dedup key MUST include `model` (and `split_id`). Foundation runs pack
  multiple models (chronos, timesfm) under one `run_id`; a `(run_id,unique_id,horizon_h)` key
  lets a global `drop_duplicates` silently drop all but one model (this dropped chronos — the
  strongest 6h model — from selection). Fixed in `mbari_neural_forecast.py` append.
- Aggregate skill across series with the **median**, not the mean. `skill = 1 - rmse/persistence`
  has near-zero denominators that produce ~-2000% per-cell outliers; the arithmetic mean is
  meaningless (it published `dlinear -1993.9%`). Keep mean only as a labeled reference.
- NEVER pool smoke runs (`summary.json smoke=true`, n=3-8) into production leaderboards, and
  NEVER average full-history with bounded-history (`tail_weeks`) runs. Report classes separately.
- Rebuild aggregate `forecast_metrics` from the per-run partitions (not by mutation) and rebuild
  the promotion matrix in the same pass so aggregate + matrix stay consistent.
- Persistence is NOT a sufficient baseline at diurnal horizons (it gets a free lunch at 24/72/168h);
  compare against the better of persistence and seasonal-naive (same-hour-yesterday). This is now
  ENFORCED, not advisory: `ops/seasonal_naive.py` recomputes `seasonal_naive_rmse` /
  `skill_vs_best_naive_pct` per cell from the gold prediction partitions; the promotion matrix gates
  `promote` on beating best-naive and marks diurnal cells `insufficient_data` when seasonal evidence
  is missing; `ops/evidence_gate_agent.py` FAILs any promoted row that does not beat best-naive. Do
  NOT reintroduce persistence-only promotion or drop the seasonal columns from the gold metrics.
