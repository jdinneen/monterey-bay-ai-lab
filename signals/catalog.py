#!/usr/bin/env python3
"""Signal Catalog — validate signals/catalog.yaml and render signals/SIGNALS.md.

The catalog is the single source of truth for every signal the lab has touched and its measured
predictive value per target. This module FAIL-CLOSES: an entry with an unknown status, a missing
required field, or a KEEP/PRIMARY claim lacking evidence is a validation error. It also renders a
human-readable SIGNALS.md so the ledger is reviewable.

Usage:
    python signals/catalog.py            # validate + (re)render signals/SIGNALS.md
    python signals/catalog.py --check    # validate only (CI); non-zero exit on any problem
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.yaml"
RENDER = HERE / "SIGNALS.md"

STATUSES = {"PRIMARY", "KEEP", "WASH", "REJECT", "UNTESTED", "PLANNED"}
MODALITIES = {"meteorological", "hydrological", "oceanographic", "biological",
              "event", "derived", "lab"}
# `captures/does/does_not/why` are the plain-language clarity fields; `predictive_value` the
# per-target verdicts. `cadence`/`revisit_when` are optional (revisit_when is most useful on
# WASH/REJECT/UNTESTED/PLANNED entries).
REQUIRED = {"name", "modality", "captures", "coverage", "access",
            "does", "does_not", "why", "predictive_value"}
# statuses that ASSERT measured skill -> must cite an evidence file that exists
EVIDENCE_REQUIRED = {"PRIMARY", "KEEP", "REJECT"}


def load(path: Path = CATALOG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate(doc: dict, root: Path | None = None) -> list[str]:
    """Return a list of problems (empty = valid). Fail-closed semantics."""
    problems: list[str] = []
    root = root or HERE.parent
    signals = doc.get("signals") or []
    if not signals:
        return ["catalog has no signals"]
    seen = set()
    for i, s in enumerate(signals):
        tag = s.get("name", f"#{i}")
        missing = REQUIRED - set(s)
        if missing:
            problems.append(f"{tag}: missing fields {sorted(missing)}")
            continue
        if s["name"] in seen:
            problems.append(f"{tag}: duplicate name")
        seen.add(s["name"])
        if s["modality"] not in MODALITIES:
            problems.append(f"{tag}: bad modality '{s['modality']}'")
        pv = s.get("predictive_value") or {}
        if not pv:
            problems.append(f"{tag}: no predictive_value targets (every signal predicts something)")
        for target, rec in pv.items():
            st = rec.get("status")
            if st not in STATUSES:
                problems.append(f"{tag}/{target}: bad status '{st}'")
            ev = rec.get("evidence")
            if st in EVIDENCE_REQUIRED:
                if not ev:
                    problems.append(f"{tag}/{target}: status {st} requires an evidence path")
                elif not (root / ev).exists():
                    problems.append(f"{tag}/{target}: evidence not found: {ev}")
    return problems


def render(doc: dict) -> str:
    signals = doc["signals"]
    targets = sorted({t for s in signals for t in (s.get("predictive_value") or {})})
    L = ["# Signal Catalog", "",
         "Single source of truth for every signal the lab has touched. The catalog **records**; "
         "the gate (`research/bacteria/signal_lab.py`, ΔAP>=0.005 + leave-one-beach-out) "
         "**decides** promotion into any model. Nulls are logged on purpose.", "",
         "`PRIMARY` = the validated model for that target · `KEEP` = passed the gate as a feature "
         "· `WASH` = tested, no lift · `REJECT` = tested, hurt · `UNTESTED` = not yet gated · "
         "`PLANNED` = not yet fetched.", ""]

    # ---- topline: state of play per target ----
    tmeta = (doc.get("meta") or {}).get("targets") or {}
    if tmeta:
        L += ["## Targets (state of play)", ""]
        for t, m in tmeta.items():
            L.append(f"- **{t}** — {m.get('what','')}")
            L.append(f"  - best model: {m.get('best_model','')}  ·  {m.get('headline','')}")
            L.append(f"  - baseline to beat: {m.get('baseline_to_beat','')}  ·  coverage: {m.get('coverage','')}")
        L.append("")

    L += ["## Predictive value by target", "",
          "| signal | modality | " + " | ".join(targets) + " | coverage |",
          "|---|---|" + "|".join("--:" for _ in targets) + "|---|"]
    for s in signals:
        pv = s.get("predictive_value") or {}
        cells = [pv.get(t, {}).get("status", "-") for t in targets]
        cov = s["coverage"][:48] + ("…" if len(s["coverage"]) > 48 else "")
        L.append(f"| **{s['name']}** | {s['modality']} | " + " | ".join(cells) + f" | {cov} |")

    L += ["", "## Detail", ""]
    for s in signals:
        L.append(f"### {s['name']}  ({s['modality']})")
        L.append(f"*Captures:* {s['captures']}")
        L.append(f"- **Does:** {s['does']}")
        L.append(f"- **Does NOT:** {s['does_not']}")
        L.append(f"- **Why:** {s['why']}")
        if s.get("revisit_when"):
            L.append(f"- **Revisit when:** {s['revisit_when']}")
        cad = f" · cadence {s['cadence']}" if s.get("cadence") else ""
        L.append(f"- **Coverage:** {s['coverage']}{cad}")
        L.append(f"- **Access:** {s['access']}")
        for t, rec in (s.get("predictive_value") or {}).items():
            ev = f" — `{rec['evidence']}`" if rec.get("evidence") else ""
            L.append(f"- **{t}:** `{rec.get('status')}` — {rec.get('metric', '')}{ev}")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not render")
    args = ap.parse_args(argv)
    doc = load()
    problems = validate(doc)
    if problems:
        print("CATALOG INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 1
    n_sig = len(doc["signals"])
    n_pv = sum(len(s.get("predictive_value") or {}) for s in doc["signals"])
    print(f"catalog OK: {n_sig} signals, {n_pv} signal/target verdicts")
    if not args.check:
        RENDER.write_text(render(doc), encoding="utf-8")
        print(f"rendered {RENDER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
