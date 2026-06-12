# Agent Directive — Monterey Bay AI Lab

> Read this before editing the lakehouse/Databricks-related files. It is binding for
> all agents (Codex, Claude, Gemini, Qwen) operating in this directory.

## IDENTITY INVARIANT (binding): this project is Monterey Bay AI Lab

The lab/project/repository identity is **Monterey Bay AI Lab**.

**Do not call this project "MBAL", "MBAL AI", "MBAL AI Lab", or "MBAL AI Forecasting".**
MBAL may appear only when it is factually a data provider/source, URL/domain,
legacy compatibility module/file name, or existing compatibility environment variable
such as `MBAL_LAKEHOUSE_DIR`. User-facing dashboards, GitHub metadata, package
metadata, public docs, generated reports, and agent summaries must say
**Monterey Bay AI Lab**.

## APPROVAL BOUNDARY (binding): autonomous by default, confirm the irreversible

This machine runs agents in YOLO / no-approval mode **on purpose** for local edits and runs.
**DO NOT CREATE GIT COMMITS OR PUSH TO GITHUB** until explicitly instructed otherwise. Everything must remain as uncommitted local file changes on the machine for now.
Safety comes from *mechanical* guards, not per-write prompts: agent-locks
(`ops/agent_lock.py`), the precious-file guard (`ops/precious_guard.py`), traffic control +
watchdog halts (`ops/run_safe.py`), the value gate, and tests. A lack of explicit approval is
**NOT** a blanket block on local edits.

Still **confirm with the user first** before these specific actions — hard to reverse or
outward-facing:
- **Leaves the machine:** git commits, pushing to a remote, opening PRs, publishing/releasing, sending
  mail, posting to external APIs or services.
- **Destroys data you did not create:** deleting/overwriting datasets, gold snapshots,
  `release/` artifacts; history rewrites; force-push.
- **Spends real money/quota at scale:** large cloud or multi-hour GPU jobs beyond the local box.
- **Blind full-file overwrite of a standing rules/memory file** (any `*.md` charter,
  `MEMORY.md`, `.qwenrules`): use a *targeted* edit instead. The precious-file guard enforces
  this for Claude automatically; Codex/Gemini/Qwen must self-enforce.

Everything else: proceed autonomously and report what you did.

---

## DATA CORPUS STEWARDSHIP (binding): preserve good labeled data

The lab is building a large corpus of high-quality labeled data. A record, source,
partition, or feature that does not matter for the current metric may matter later.

**DO NOT discard, overwrite, downsample away, or "clean up" good labeled data merely
because it is not useful to today's model or leaderboard.** Prefer additive retention:
keep the raw/bronze copy, preserve labels and provenance, and mark current exclusions
with metadata, manifests, split filters, or quarantine paths instead of destructive
deletion.

When data appears bad, duplicated, leaked, out-of-contract, or out-of-scope:
- separate quality defects from "not useful right now";
- quarantine or flag suspect data before deletion whenever feasible;
- keep enough provenance to revisit the decision later;
- confirm first before deleting or overwriting datasets, gold snapshots, or release
  artifacts you did not create.

Agents must treat high-quality labeled data as future research inventory, not clutter.

---

## TRAFFIC CONTROL & GPU GUARD (binding): Gate every heavy task

Multiple agents and long-running production jobs share a single RTX 5090. To prevent GPU oversubscription and redundant task execution, every agent and script MUST pass through the **Traffic Controller**.

### PROTOCOL: The "Traffic Gate"
Before starting any significant task (especially those requiring GPU), you must obtain clearance.

1.  **Request Permission:**
    `python ops/traffic_controller.py request --task "<task-name>" --agent <your-id> --gpu-mib <size>`
2.  **Execute:** Only proceed if the command returns exit code `0`.
3.  **Release:** Release the lock as soon as the task is finished:
    `python ops/traffic_controller.py release --task "<task-name>" --agent <your-id>`

### THE MANDATORY WAY: `ops/run_safe.py` (NO RAW NEURAL SCRIPTS)
To automate the request/release cycle and safely monitor long runs, you MUST use the safe launcher for any training or GPU tasks.
**DO NOT use `start /b python ...` or raw `python` for any heavy neural script (like `mbal_neural_forecast.py`). Doing so bypasses VRAM safety limits and WILL hard-freeze the machine!**
`python ops/run_safe.py --task "<name>" --agent <id> --gpu-mib <SIZE> [--brain-preflight warn|enforce] [--watchdog <metrics-path>] -- <your-command>`

**Antigravity CLI (TUI):**
To launch the interactive Antigravity TUI safely, use:
`pwsh ops/agy.ps1`
This wraps `agy` in the safe launcher with traffic coordination.


### Local command center
For browser-based oversight of Agent Brain memories, traffic locks, GPU state, value
gate preflights, and reflection logging, run:
`python ops/command_center.py`

The command center is local-only and does not replace this charter, the value gate,
traffic control, tests, or human approval.

**Rules:**
- **Observational Watchdog:** You may use `--watchdog` to monitor for VRAM or Loss explosions. If triggered, the run will **HALT** for human review. Agents are FORBIDDEN from automatically "repairing" training code when the loss spikes, as this often gaslights real physical anomalies (e.g., upwelling events) as "bugs."
- **Zero-Catastrophic-Collision:** If the Traffic Controller denies a request (exit code 2 or 3), you MUST NOT proceed. Pick a different task or wait.
- **Estimated Footprint:** Always provide an honest `--gpu-mib` estimate. If unknown, use 8192 (8GB) as a safe default for neural runs.
- **Task Identity:** Tasks are keyed by name (e.g., `TASK:daily-forecast`). Two agents cannot run the same task simultaneously.

---

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
  the existing legacy compatibility env vars (`MBAL_PROJECT_ROOT`,
  `MBAL_SOURCE_PARQUET`, `MBAL_CACHE_DIR`, `MBAL_LAKEHOUSE_DIR`). These env var
  names are compatibility tokens, not project identity.
- Use the portable smoke job `ops/jobs.smoke.json` for remote-environment validation.
- Describe remote targets generically (Spark / Kubernetes / object storage / any managed
  runtime) — never product-specifically.

**DO NOT** (treat any of these as a *defect to remove, not restore*)
- Create or restore `databricks.yml`, `resources/mbal_smoke.job.yml`, or
  `ops/jobs.databricks_smoke.json`.
- Add Databricks CLI / bundle / Unity Catalog / DBR-runtime dependencies, or any
  `DATABRICKS_*` env requirement.
- Frame "Databricks cloud execution" as a release blocker. The only remote gap is a
  product-agnostic portable smoke run.

If an audit finds these product files present or referenced, the correct action is to
**DELETE** them and fix the doc reference — never to re-add them.

Canonical contract doc: `MBAL_PRODUCTION_LAKEHOUSE_CONTRACTS.md` (legacy filename;
the project identity remains Monterey Bay AI Lab)
(supersedes the removed `MBAL_DATABRICKS_LOCAL_CONTRACTS.md`).

## Framing invariant (do not regress): event-prediction is the headline, M1/M2 is the backbone

The lab's public/pitch framing MUST lead with the **statewide bacterial-exceedance** result
(the problem where a model beats the baseline) and cast
the **M1/M2 mooring record as the physical backbone / driver source, NOT the forecasting
headline**. M1/M2 hourly forecasting is persistence-ceilinged (`reports/PERSISTENCE_CEILING.md`,
acf_1h≈0.99, single-digit % skill over naive) — do not re-promote it to "capability #1" or
imply GPU compute meaningfully improves it. State the bacteria result honestly: site memory is
a strong baseline (AUC ~0.77 / AP 0.28 on the San-Diego-excluded stratum); the model beats it
(+0.22 AP) and the fundable frontier is **widening that margin with environmental
drivers** (`reports/BACTERIA_REFRAME.md`). Use the doc-grounded honest numbers (the
reproduce kit's byte-reproducible `expected/`, reveal_lag_days=2 operational config)
(**San-Diego-excluded ROC-AUC 0.858 / AP 0.497, calibrated ECE 0.020, 9/9 LOCO wins** —
`research/bacteria/reproduce/expected/`). The "AUC 0.888 / AP 0.753" figure is **retired**: it
was the artifact-inflated pooled/multi-analyte number (San Diego/Tijuana 2022 regime break).
Never cite 0.888 or the stale "AUC 0.79+".

## Data/results correctness invariants (audit 2026-06-08 — do not regress)
- Lakehouse metric dedup key MUST include `model` (and `split_id`). Foundation runs pack
  multiple models (chronos, timesfm) under one `run_id`; a `(run_id,unique_id,horizon_h)` key
  lets a global `drop_duplicates` silently drop all but one model (this dropped chronos — the
  strongest 6h model — from selection). Fixed in `mbal_neural_forecast.py` append.
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

## VALUE GATE (binding): is this worth building, or is it bloat?

Before you **suggest OR execute** any new feature, module, abstraction, driver, data
source, model, metric, doc, or "phase," run the 5-point gate in `docs/VALUE_GATE.md` and
**cite the result** ("Value gate: DO_NOW because …" / "rejected: premature/redundant/…").
All five must be YES to build:
1. beats a real/honest baseline (seasonal-naive, persistence, AB411, prior-lab,
   station-memory, the current metric) or answers a question a stakeholder actually asked
   — "more sophisticated/SOTA/general" is NOT a yes;
2. additive & reversible (default-off flag or new file) — EXCEPT correctness fixes
   (leakage/dedup/median/causality), which may be in-place/one-way and are mandatory;
3. testable, and you add the test in the same change;
4. valuable NOW, not gated on a burn-in/evidence/data step that hasn't happened;
5. genuinely new and improves a result we actually claim — not redundant, and not
   polishing a case we deliberately exclude (e.g. San Diego in the bacteria work).

If any is NO, do **not** build it — name the kill-category (bloat / vapor / premature /
ops-toggle / redundant / cosmetic / excluded-case polish). The gate is **not a freeze**:
correctness fixes, reviewer-demanded validations, baseline-beating features with tests,
and honest negative results all pass. Report results straight (a wash is a wash; ranking
without calibration is not deploy-ready). Prefer the cheaper validation over the bigger
build; when unsure, ask. Full rubric + examples: `docs/VALUE_GATE.md`.


