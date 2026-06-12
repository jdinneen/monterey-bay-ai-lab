# Signal Discovery — how we keep finding level-2 (and level-3) signals

A repeatable machine for the loop: **find where the nowcast is wrong → build a derived
signal → gate it honestly → compose level-3 only from complementary level-2 signals.**
Run it again whenever new data lands.

## The three pieces

| Stage | Module | What it does |
|---|---|---|
| Discover | `residual_diagnostics.py` | Fits the current model, scores *structured residual* by axis (spatial / temporal / station / county / driver-decile) + an irreducible-error floor → a ranked candidate backlog. |
| Gate L2 | `signal_lab.py` | Each candidate is A/B'd vs a **pure-causal baseline** on the 2022+ holdout **and** leave-one-beach-out. KEEP only if temporal ΔAP ≥ 0.005 **and** it survives on unseen beaches (else REJECT = memorisation). |
| Compose L3 | `signal_lab.propose_l3` | From the KEPT L2 set, ranks pairs by **residual independence** and registers the most-independent pair's interaction as an L3 candidate, gated identically. |

Orchestrate all three: **`python -m research.bacteria.run_signal_discovery`**
→ writes `reports/operational_benchmark/SIGNAL_DISCOVERY.md`.

## "How much L2 do we need for L3?"

Not a quota. You need **≥2 KEPT L2 signals whose holdout residuals are independent**.
Interacting two signals that already capture the same structure (high residual correlation)
adds nothing — `propose_l3` measures this and picks the pair worth composing. Most L3
interactions are washes; that's expected and reported straight.

## Re-running as data arrives

```bash
# full loop on the local 802-station parquet
python -m research.bacteria.run_signal_discovery

# just the discovery scan (ranked backlog + headroom floor)
python -m research.bacteria.residual_diagnostics

# gate a specific new candidate you registered
python -m research.bacteria.signal_lab --candidates <name>
```

Add a new signal by registering a `Candidate` in `signal_lab.py` (≈10–15 lines: name, level,
feature columns, a leakage-safe builder, optional `requires=[...]`). It then gets the full
honest gate for free. A level-3 signal is just a candidate that `requires` level-2 candidates.

## Densifying the graph (gated — not run here)

More stations ⇒ denser neighbour graph ⇒ stronger spatial signal. The statewide BigQuery
pull (incl. the unused HAB tables) refreshes the obs parquet:

```bash
# needs MBAL_GCP_PROJECT + gcloud auth; execution-gated, run deliberately
python -m research.bacteria.fetch_statewide_beachwatch
```

## Next frontier (separate build)

Marine-HAB / **domoic-acid** prediction is a *new label* (CalHABMAP / SCCOOS ERDDAP source,
baseline to beat = C-HARM), not a new feature on this label. It gets its own fetcher + plan;
it is intentionally out of scope for this signal-discovery machine.
