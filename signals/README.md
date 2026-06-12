# Signal Catalog

A ledger of every **signal** the lab has touched — a measurable input that may carry predictive
value for one or more targets (bacteria exceedance, domoic acid, …). The catalog exists to serve
the multi-signal strategy: *orthogonal signals may combine to mean something*, so we accumulate
candidates **honestly** instead of forgetting them — while not hoarding.

- **Source of truth:** [`catalog.yaml`](catalog.yaml)
- **Human view (generated):** [`SIGNALS.md`](SIGNALS.md) — `python signals/catalog.py`
- **Validator/renderer:** [`catalog.py`](catalog.py) (fail-closed; tested in `tests/test_signal_catalog.py`)

## What this is — and is NOT

- It is **metadata / provenance**, not a materialized lakehouse row. Logging a signal here costs
  nothing and commits us to nothing.
- It **records**; it does not **decide**. Promotion of a signal into any model is governed by the
  honest gate (`research/bacteria/signal_lab.py`: temporal **ΔAP ≥ 0.005 AND survives
  leave-one-beach-out**). The catalog just remembers the verdict.

## Two claims that must never be conflated

| claim | meaning | how it's earned |
|---|---|---|
| **predicts X** | measured skill for target X | a validated model / passing the gate → `PRIMARY` / `KEEP` |
| **improves Y** | lifts a *different* target when combined | a SEPARATE gated test → default `UNTESTED` until run |

A signal that is genuinely predictive for its own target is **not** automatically a useful
feature for another target. Example: `domoic_acid_pda` is `PRIMARY` for domoic acid but
`UNTESTED` for bacteria — and **coverage-limited** (17 piers can't inform 830 beaches), so any
bacteria test must be restricted to co-located sites.

## Log the nulls

"Tested, washed" (`WASH`) and "tested, hurt" (`REJECT`) entries are **as valuable as hits** —
they stop us re-litigating settled negatives (the wave/tide and news nulls are recorded for
exactly this reason). Coverage is first-class: record it, because a real signal with thin
coverage can't move a broad target no matter how good the combiner.

## What each entry records

Plain-language, so the log is readable at a glance:

| field | meaning |
|---|---|
| `captures` | the physical/operational quantity the signal measures |
| `does` | what it is genuinely good for |
| `does_not` | its limits — what it can't or doesn't do |
| `why` | the mechanism behind the verdict (especially *why* a null failed) |
| `revisit_when` | *(optional)* the specific condition that would change a verdict — the forward-looking, actionable part |
| `coverage` / `cadence` | spatial+temporal extent and sampling frequency (first-class — thin coverage caps usefulness) |
| `predictive_value` | per target → `status` + measured `metric` + `evidence` file |

The top of `catalog.yaml` also carries `meta.targets`: a **state-of-play** line per target (the
question, current best model + headline metric, and the baseline to beat) so the catalog opens
with where each problem stands.

## Status enum (per target)

`PRIMARY` · `KEEP` · `WASH` · `REJECT` · `UNTESTED` · `PLANNED`
(`PRIMARY`/`KEEP`/`REJECT` assert measured skill → must cite an evidence file that exists.)

## Adding / updating an entry

1. Edit `catalog.yaml` (one block per signal; `predictive_value` maps each target → status +
   metric + evidence).
2. Run `python signals/catalog.py` to validate and re-render `SIGNALS.md`.
3. When you gate a signal against a new target, update its status from `UNTESTED` and cite the
   evidence — pass or fail. Both directions are worth logging.
