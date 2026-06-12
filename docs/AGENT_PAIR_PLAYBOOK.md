# Agent-Pair Playbook — how two (or more) agents do coordinated work, fast and safe

A **reusable, task-agnostic** protocol for running 2+ autonomous agents (Claude, Codex,
Gemini, Qwen) on the **same working tree at the same time** without clobbering each
other, duplicating work, or shipping something wrong. It was distilled from a live
two-Claude data-source buildout (see `reports/data_fetch/COORDINATION.md` for that
concrete instance) and is meant to be copied for any future parallel effort: code
migrations, audits, multi-source fetches, refactors, doc sweeps.

> Use this with `AGENTS.md` (binding rules: approval boundary, traffic/GPU gate, data
> stewardship) and `docs/agent_brain/operating_loop.md` (single-task discipline). This
> file is the **how-two-agents-share-the-tree** layer.

---

## 0. The one-paragraph version

Pick distinct **agent IDs** and split the work into disjoint **lanes**. Coordinate through
three append/claim files, never by talking: **file locks** (`ops/agent_lock.py`, auto for
Claude) stop two agents editing one file; a **claim ledger** (one JSON per work-item)
stops two agents building the same thing; an **append-only learnings log** lets each
agent learn from the other's dead-ends and fixes. Edit shared files with **targeted
edits, never blind overwrites**. Put each agent's contributions in **separate files**
merged by a tiny idempotent hook, so the shared index is touched rarely and briefly.
**Check in** on a cadence. Nothing is "done" until the **other** agent reviews it and both
**co-sign a final review**. No commits/push without the human.

---

## 1. Set up (first agent launching does this)

1. **Choose stable agent IDs.** e.g. `claude-fetch-A` / `claude-fetch-B`, or
   `<vendor>-<task>-<n>`. Use the ID for ledger entries, the learnings log, the
   per-agent TASK marker, and deliberate cross-agent file claims.
2. **Create the three coordination files** for this effort (copy the data-fetch instance):
   - `…/COORDINATION.md` — the task-specific charter (lanes, done-definition, status).
   - `…/<effort>_intents.json` — the claim ledger (one entry per work-item).
   - `…/<effort>_learnings.jsonl` — append-only issues/fixes/check-ins.
3. **Split into lanes** so work is disjoint. Assign by *file ownership* where possible
   (each agent owns its own new files). List a shared **backup pool** for when a lane stalls.
4. **Drop a TASK marker** so others see you: `python ops/agent_lock.py claim
   TASK:<effort>-laneA --agent <id> --task "<what>"`.
5. **Post a `hello`** line to the learnings log and start.

The second agent: read `COORDINATION.md` + the ledger + the learnings log, claim the
other lane, post a `hello` + `checkin`, and go.

---

## 2. The three coordination channels (all file-based, no daemon)

### a) File locks — stop two agents editing one file (`ops/agent_lock.py`)
- For Claude this is **automatic**: the `PreToolUse` hook claims every file you
  Edit/Write under your **session_id**, and *blocks* you from a file another live agent
  holds. That block is the flag. Non-Claude agents must call `claim`/`release` themselves.
- Commands: `claim <paths…> --agent <id> --task "…"` (exit 2 = conflict, pick other work),
  `release-mine --agent <id>`, `status`. Claims auto-expire (30 min) so a dead agent
  never wedges the tree; reclaim a genuinely dead lock with `release <path> --force`.
- **GOTCHA (learned the hard way):** the hook claims under your *session_id*, not a custom
  `--agent` name. So **do not manually `claim` a file you will then Edit/Write yourself**
  under a different `--agent` id — you lock yourself out. Use the custom ID only for (i)
  the TASK marker and (ii) deliberate claims on a *shared* file you're coordinating
  hand-off on; let the hook own your normal edits.

### b) Claim ledger — stop two agents building the same item (`<effort>_intents.json`)
- One entry per work-item with `agent` + `stage`
  (`queued → claude_building → built_awaiting_review → reviewed_ok | review_failed →
  landed`, plus `rejected`). **Read it before starting any item.** Claim by flipping the
  entry's `agent`/`stage` with a **targeted edit** (see §3).

### c) Learnings log — learn from each other (`<effort>_learnings.jsonl`, append-only)
One JSON object per line:
```json
{"ts":"<iso>","agent":"<id>","source":"<item>","kind":"hello|checkin|issue|fix|note","msg":"<what>","reuse_when":"<when this matters again>"}
```
- Hit a wall → log `issue` immediately so the other agent doesn't repeat it.
- Solve it → log `fix` with the actual remedy.
- **Before starting an item, grep this log for its name and its category** and apply prior fixes.
- A NULL result (dead endpoint, doesn't meet the bar, doesn't help) is a *kept* learning —
  log it and mark the item `rejected` with the reason. Don't silently drop it.
- Promote durable, proven lessons to Agent Brain (`docs/agent_brain/learnings.jsonl`).

---

## 3. Editing shared files — targeted edits, NEVER blind overwrites

The whole pair stays sane only if nobody clobbers a shared file. Rules:
- **Append-only logs** (`*_learnings.jsonl`): only append (a `>>` append or an Edit that
  adds lines). Never rewrite.
- **Ledgers / status JSON**: change one entry with a **targeted Edit** of that object.
  Never `Write` the whole file after the initial seed — you'd drop the other agent's
  just-made claims. **Re-Read before editing** so you build on their latest state.
- **A shared index file** both must extend (a registry, a manifest, a router): use the
  **separate-contrib-file + idempotent merge** pattern in §4 instead of both editing it.
- **Precious files** (`*.md` charters, `MEMORY.md`, rules): targeted Edit only; the
  `ops/precious_guard.py` hook blocks blind overwrites for Claude. Others self-enforce.
- If you genuinely must replace a block in a shared file, **announce it** in the learnings
  log (`kind:"note"`) so the other agent re-syncs.

---

## 4. The collision-free shared-index pattern (key trick)

When both agents must register into **one** index file (e.g. `registry.py`,
`routes.py`, a `__init__` list), do **not** both edit it. Instead:
1. Each agent puts its contributions in its **own** module: `index_laneA.py`,
   `index_laneB.py`, each exposing a list (`LANE_A_SPECS`, …).
2. The core index imports and merges them **idempotently**, once, with a precedence rule:
   ```python
   for _s in LANE_A_SPECS: REGISTRY.setdefault(_s.key, _s)   # lane A wins collisions
   for _s in LANE_B_SPECS: REGISTRY.setdefault(_s.key, _s)
   ```
   Order matters with `setdefault` (first wins). A contrib import error is caught so it
   never breaks the core. Now each agent edits only its own lane file — zero contention —
   and the core index is touched once to wire the merge.
3. If the core index *is* briefly locked by the other agent, **don't force** — build your
   lane file and adapters meanwhile, validate via an inline/throwaway spec, and wire the
   merge when the lock frees.

---

## 5. Check-in cadence (stay on the same page)

Post a `checkin` to the learnings log: (a) on launch, (b) after each work-item lands,
and (c) at least every ~15 min — saying where you are, what's next, any blocker. Before
starting new work, read the other agent's latest `checkin`/`issue` lines. This is what
catches "we're both about to build X" before the wasted work happens.

---

## 6. Review gates — "we can't get this wrong"

Nothing counts as done on the builder's own word.

- **Gate 1 — cross-review (every item, by the *other* agent).** Builder sets stage
  `built_awaiting_review`. The other agent independently re-runs the item's checks,
  inspects the output for real (not just exit codes), confirms it meets the
  done-definition, and sets `reviewed_ok` or `review_failed` (with reasons, logged). A
  failed item goes **back to its builder** — the reviewer does not silently fix it
  (avoids both agents editing one file). Builder fixes, re-submits.
- **Gate 2 — joint final review (once all items are in).** Both agents, together, re-run
  the full check suite + tests, re-verify every item against the done-definition, and
  **co-sign** a `…_FINAL_REVIEW.md` with both IDs. Optionally bring in a **third
  adversarial critic agent** to try to *refute* each item (sub-bar, leaked, wrong-path,
  not actually wired in). Only after co-sign: update Agent Brain and tell the human.
- **No commits/push** until the human says so (`AGENTS.md` approval boundary). Safety is
  mechanical (locks, guards, tests, review), not per-write prompts.

---

## 7. The work loop (generic funnel)

For "produce N good items from a larger candidate pool":
1. **Funnel:** a catalog of ~2N candidates with a cheap **probe** (reachability) and an
   explicit **promotion gate** (the bar each item must clear). Oversize it — some fail.
2. **Per item:** claim → build in your own files → run it for real → **validate against
   the gate** → emit the same artifacts every existing item has → mark stage.
3. **Honesty:** the authoritative status is the tool's own report, not your memory of
   what you fetched. **Verify each item via the report before claiming it "landed."**
   In-flight/available counts ≠ validated output.
4. **Repeatability:** wire the funnel into one command (`… buildout`/`… run`) so re-running
   the whole thing is one invocation, and so a third party can reproduce it.

---

## 8. Done-definition (fill in per effort; example = data-fetch)

An item counts only when ALL hold (adapt the specifics):
1. Builds/runs via the standard entry point; writes only to allowed paths (no trusted-root
   writes; guard intact).
2. Produces real output meeting the **gate** (e.g. ≥50k labeled timestamped rows).
3. Passes validation (schema, coverage, bounds, no dup keys).
4. Shows green in the tool's own status report.
5. Test suite still green (index valid, guards intact).
6. Ledger → `landed`; one `fix`/`note` in the learnings log; cross-reviewed by the other agent.

---

## 9. Quick reference — copy/paste

```bash
# who's working on what
python ops/agent_lock.py status
# claim a TASK marker / a shared file you're coordinating handoff on
python ops/agent_lock.py claim TASK:<effort>-lane<X> --agent <id> --task "<what>"
# release everything you hold when done/aborting
python ops/agent_lock.py release-mine --agent <id>
# append a learning (issue/fix/checkin) — never rewrite the log
cat >> <effort>_learnings.jsonl <<<'{"ts":"…","agent":"<id>","kind":"issue","source":"…","msg":"…","reuse_when":"…"}'
```

**Anti-patterns that bite (all observed):** manually claiming a file you'll edit yourself
(self-lock); blind-`Write` over a shared ledger (drops the other's claims); both editing
the core index (use lane files); trusting "I fetched 1.8M rows" over the validated report
(it was 8k); forcing a lock another live agent holds; bounding/validating on *metadata*
instead of the actual label; declaring done without the other agent's review.
