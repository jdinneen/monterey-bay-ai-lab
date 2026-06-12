# Value Gate — is this worth building, or is it bloat?

Binding for all agents (Codex, Claude, Gemini, Qwen). Run this gate **before you
suggest OR execute** any new feature, module, abstraction, driver, data source, model,
metric, doc, or "phase." Its job: keep the project continually improving **without**
accreting bloat. Cite the result with a **falsifiable payload** — a bare "beats baseline"
is itself a gate failure (it's how cargo-cult citation gets caught):
*"Value gate: DO_NOW — beats <baseline> by <Δmetric>, test <path::name>"* or
*"rejected: <kill-category> — <one-line reason>"*.

This gate is itself held to its own bar: it is one short checklist, not a manifesto.

## The 30-second gate — all five must be YES to build

1. **Beats a real baseline / answers a real question.** Does it measurably improve a
   result against an *honest* baseline (seasonal-naive, persistence, AB411, prior-lab,
   station-memory, the current metric) — or answer a question a stakeholder actually
   asked? "It's more sophisticated / more SOTA / more general" is **not** a yes.
2. **Additive & reversible — OR a correctness fix.** New *behavior* lands behind a
   default-off flag or as a new file and is removable cleanly (Delta phases, shadow
   dual-write off-by-default: yes; an in-place rewrite of the gold layer for a new
   feature: no). **Correctness fixes** (leakage, dedup key, median-not-mean, causality)
   are EXEMPT — they may be in-place and one-way, and fixing them is mandatory
   regardless of reversibility (see the data-correctness invariants in `AGENTS.md`).
3. **Testable, and you will test it.** Is there a concrete pass/fail test or a
   reproducible number you will add in the same change? If you can't test it, you can't
   claim it.
4. **Valuable NOW, not speculative.** Its value must be demonstrable now, not unprovable
   until some future step lands. Sequencing is fine — building a holdout harness *before*
   the validation it enables passes; building a feature whose worth can't be shown until
   a burn-in/dataset/evidence step that hasn't happened does not. When in doubt, do the
   cheaper enabling step first.
5. **Genuinely new, and improves a result we actually claim.** Not redundant with an
   existing path, and not *polishing* a case we deliberately **exclude** (e.g. tuning San
   Diego's calibration) or a stratum we don't deploy. **Carve-out:** work that would
   *change* the exclusion decision — expand coverage, or fix the reason a region was
   excluded — is a baseline-beating feature and PASSES.

If any answer is NO → **do not build it.** State which kill-category it is.

## Kill list — name the reason when you reject something

- **Bloat** — needless complexity/abstraction; recovers what a simpler thing already gives.
- **Vapor** — scaffolding not wired to real work; a "framework" with no caller.
- **Premature** — depends on a burn-in / evidence / data step not yet done.
- **Ops-toggle** — a config/env flip or one-line invocation, not an engineering deliverable.
- **Redundant** — an existing path already covers it.
- **Cosmetic** — renames, reformatting, micro-metrics that change no decision.
- **Excluded-case polish** — optimizes a region/stratum/path we deliberately set aside
  *without changing the exclusion decision* (tuning excluded San Diego's calibration).
  NB: work that makes an excluded case *includable* is NOT this — it passes (see Q5).

## The gate is NOT a freeze — these PASS

The point is to keep improving, just honestly. Genuine improvement looks like:

- A feature/driver that **lifts a metric against a real baseline**, with a test
  (e.g. rainfall → beat the AB411 rule).
- A **validation a reviewer will demand** — *especially* one that might puncture a claim
  (e.g. leave-one-county-out spatial holdout).
- A **correctness fix** (leakage, dedup key, calibration, causality) — always in scope,
  reversibility waived.
- **Tech-debt paydown / an enabling refactor** that removes a named footgun or unblocks
  specific future value — even if it beats no metric (state the footgun/value).
- **Deleting** dead code, unreferenced files, or forbidden artifacts (e.g. Databricks
  product files) — deletion is the purest anti-bloat act; always passes.
- **Security / dependency hygiene / reproducibility** fixes (a CVE bump, a pin, a content
  hash) — pass even though they beat no forecasting baseline.
- An **honest negative / mixed result** that changes the next step (e.g. discharge is
  marginal; online recal must be drift-gated). Report it straight; never overclaim.
- A **negative/null result that is itself the deliverable** — publishable or roadmap-
  changing (e.g. the M1 persistence-ceiling). Ship it as output, don't just mention it.
- A **reversibility / safety** improvement that closes a real gap (a two-key upload gate,
  a content hash).

## Honesty rules — so "better" stays real

- Compare against the **honest baseline**, **stratified by region/era**; never a pooled
  number that hides an artifact.
- Report the result **straight**: +0.01 is +0.01, a wash is a wash, a regression is a
  regression. A model that *ranks* better but isn't *calibrated* is not deploy-ready.
- If a sharpening is **redundant** with a signal you already have, say so and weigh
  dropping it.
- **Let the validation gate decide** — don't build the sharpening before the validation
  that justifies it has held.

## Procedure

1. Run the 30-second gate.
2. Fails → stop; state the kill-category and (if useful) the cheaper alternative.
3. Passes → say *"Value gate: DO_NOW because …"*, build the **smallest** version that
   delivers it, add the test, report the honest number.
4. Unsure or it's a big build → prefer the **cheaper validation** first, and **ask**.

> Default bias: when a step would mainly add sophistication, recover what a simpler
> thing already gives, or polish an excluded case — it's bloat. When it beats a real
> baseline, is reversible, and is tested — it's improvement. Most "let's also add…"
> ideas are the former; the gate exists to catch them before they ship.
