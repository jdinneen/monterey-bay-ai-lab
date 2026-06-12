# Operating Loop

Use this loop when directing agents in the Monterey Bay AI Lab. The goal is to
turn agent energy into better science without letting parallel work become bloat.

## Daily Loop

1. **Observe.** Open the command center. Check GPU, sentinel, traffic locks, git
   dirtiness, and the next safe action.
2. **Search memory.** Query Agent Brain for the topic before asking an agent to
   build. Look for invariants, hard-learned lessons, and rejected ideas.
3. **Preflight.** Run the value gate. A task needs a baseline or stakeholder
   question, reversibility or correctness-fix status, a test/evidence path, value
   now, and a genuinely new contribution.
4. **Assign one owner.** Pick one lead/executor for implementation. Use other
   agents as critics: value critic, evidence critic, or ops guard.
5. **Run safely.** Use file claims for edits and `ops/run_safe.py` for heavy or
   long-running jobs. Use specific task names.
6. **Record only durable memory.** Append lessons or rejected ideas only when they
   include evidence and a clear reuse condition.

## Agent Roles

- **Lead Synthesizer:** chooses the next meaningful task and integrates critique.
- **Executor:** implements one bounded change with a clear evidence path.
- **Value Critic:** kills bloat, vapor, redundant phases, and unsupported SOTA.
- **Evidence Critic:** checks baselines, leakage, split contracts, and claims.
- **Ops Guard:** checks GPU, traffic locks, file claims, sentinel, and tests.

## Launch Rule

Do not ask agents to "improve the project." Give them a narrow task:

```text
Role: Evidence Critic
Task: Review whether quarterly NOAA MUR ingestion avoids the prior HTTP 408 failure.
Evidence path: mbal_history/noaa/fetch_run.log
Brain query: NOAA MUR HTTP 408 quarterly chunk
Output: pass/fail recommendation, no edits.
```

## Good Memory

Good:

```text
Lesson: NOAA MUR annual fetch times out; chunk by quarter.
Evidence: HTTP 408 on 2026-06-09.
Reuse when: any satellite fetch task.
```

Bad:

```text
Lesson: be careful.
```

## Failure Modes To Watch

- Dashboard bloat: adding panels before the next-safe-action signal is trusted.
- Agent drift: agents acting without Brain query, value gate, or file claims.
- Context bloat: loading all docs instead of querying relevant Brain memory.
- False authority: trusting stale files instead of the lock/status tools.
- Duplicate work: vague task names such as `analysis` or `forecast`.
- Scientific over-automation: treating anomalies as bugs and auto-repairing them.
- Memory pollution: recording weak lessons without evidence or reuse conditions.
- Bloat-by-phase: adding a new phase/framework before evidence says it matters.
