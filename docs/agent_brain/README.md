# Agent Brain

The Agent Brain is the shared memory and pushback layer for agents working in this
repository. It is intentionally small: agents should retrieve the relevant facts,
lessons, and rejected ideas just in time instead of loading every project artifact into
context.

This directory is not deployed. It is a local/project memory substrate for Codex,
Claude, Gemini, Qwen, and future agents.

## Memory Layers

- `project_state.json` stores stable facts, current claims, baselines, and known
  project boundaries.
- `science_mission.md` stores the scientific strategy, good automation targets, and
  hard-learned modeling lessons.
- `operating_loop.md` stores the daily AI Director workflow for using agents without
  drifting into bloat.
- `invariants.json` stores binding rules that must not regress.
- `agent_roles.json` stores recommended agent roles and critic responsibilities.
- `learnings.jsonl` stores episodic lessons from completed work.
- `rejected_ideas.jsonl` stores ideas that failed the value gate so they do not return
  as bloat under a new name.
- `decision_log.jsonl` stores higher-level decisions that should be cited later.

## Required Agent Loop

1. Read `AGENTS.md`.
2. Read `docs/VALUE_GATE.md`.
3. Read `docs/agent_brain/science_mission.md` for modeling/data tasks.
4. Read `docs/agent_brain/operating_loop.md` when coordinating multiple agents.
5. Query this directory for relevant memories.
6. Run `python ops/agent_preflight.py ...` before proposing new work.
7. Build only if the preflight verdict is `DO_NOW`.
8. After the task, run `python ops/agent_reflect.py ...` to record the lesson or
   rejection.

## Non-Goals

- This is not a vector database.
- This is not an autonomous deployment controller.
- This is not a substitute for tests, release gates, or human approval.
- This does not let agents override `AGENTS.md`.
