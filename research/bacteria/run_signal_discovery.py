#!/usr/bin/env python3
"""Signal discovery orchestrator — run the whole loop in one shot.

Chains the three pieces built for the signal-discovery machine:

    residual_diagnostics  →  signal_lab.evaluate (gate L2)  →  signal_lab.propose_l3 (gate L3)

and writes ONE consolidated report so you can see, in order: where the model is wrong
(the backlog + headroom floor), which level-2 candidates survived the honest gate, and —
from the KEPT L2 signals — which level-3 interaction is worth composing and whether it held.

This is the entry point to re-run as new data arrives: `python -m research.bacteria.run_signal_discovery`.
Everything runs on the local 802-station parquet; nothing is materialised to the gold layer
(KEEP just means "earned a place in the model", a later, separate step).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob  # noqa: E402
from research.bacteria import residual_diagnostics as rd  # noqa: E402
from research.bacteria import signal_lab as sl  # noqa: E402

# The L2 candidates gated by default, all over the FEATS+rainfall baseline: the new-driver
# regression-tests (discharge/waves/tide, expected WASH), the proven spatial signals, and the
# EB region prior. Rainfall is in the baseline, not a candidate. (The L3 example nbr7_x_rain3 is
# intentionally left to propose_l3.)
DEFAULT_L2 = ["discharge", "waves", "tide", "latlon", "nbr_lag", "nbr_and_latlon"]


def run(obs_path: Path, *, l2=None, dirs=None, reveal_lag_days: int = 2, label: str = "enterococcus",
        exclude_counties=("San Diego",), k: int = 8, n_perm: int = 199, do_lobo: bool = True) -> dict:
    l2 = l2 or DEFAULT_L2
    dirs = dirs or {}
    kw = dict(reveal_lag_days=reveal_lag_days, label=label, exclude_counties=exclude_counties, k=k)

    diag = rd.run(obs_path, rain_dir=dirs.get("rain_dir"), discharge_dir=dirs.get("discharge_dir"),
                  cdip_dir=dirs.get("cdip_dir"), tide_dir=dirs.get("tide_dir"), n_perm=n_perm, **kw)

    l2_res = sl.evaluate(obs_path, l2, rain_dir=dirs.get("rain_dir"),
                         discharge_dir=dirs.get("discharge_dir"), cdip_dir=dirs.get("cdip_dir"),
                         tide_dir=dirs.get("tide_dir"), n_perm=n_perm, do_lobo=do_lobo, **kw)

    kept = [c["name"] for c in l2_res.get("candidates", [])
            if c["verdict"] == "KEEP" and c["level"] == 2]
    prop = sl.propose_l3(obs_path, kept, rain_dir=dirs.get("rain_dir"),
                         discharge_dir=dirs.get("discharge_dir"), cdip_dir=dirs.get("cdip_dir"),
                         tide_dir=dirs.get("tide_dir"), **kw)

    # Gate the auto-proposed L3 IF propose_l3 found a genuinely independent, buildable pair.
    # If it didn't, we report that honestly rather than gating a curated example to manufacture
    # a complete-looking verdict (the registry L3 examples remain runnable via signal_lab directly).
    l3_res = None
    if prop.get("registered"):
        l3_res = sl.evaluate(obs_path, [prop["registered"]], rain_dir=dirs.get("rain_dir"),
                             discharge_dir=dirs.get("discharge_dir"), cdip_dir=dirs.get("cdip_dir"),
                             tide_dir=dirs.get("tide_dir"), n_perm=n_perm, do_lobo=do_lobo, **kw)

    return {"diagnostics": diag, "l2": l2_res, "l3_proposal": prop, "l3_gate": l3_res,
            "kept_l2": kept}


def to_markdown(res: dict) -> str:
    diag, l2, prop, l3 = res["diagnostics"], res["l2"], res["l3_proposal"], res["l3_gate"]
    lines = ["# Signal Discovery — consolidated report", ""]

    # 1. headroom + backlog
    fl = diag.get("irreducible_floor", {})
    mbrier = diag.get("model", {}).get("brier")
    floor = fl.get("bayes_brier_floor")
    gap = (round(mbrier - floor, 5) if (mbrier is not None and floor is not None) else None)
    lines += [
        "## 1. Where is the model wrong? (residual diagnostics)",
        f"- current model AP **{diag.get('model', {}).get('ap')}** | Brier {mbrier} vs "
        f"Bayes-Brier floor {floor} → headroom **{gap}** (base {fl.get('base_rate')})",
        "- flagged axes → candidate backlog: "
        + (", ".join(dict.fromkeys(diag.get("backlog", []))) or "(none cleared threshold)"),
        "",
        "## 2. Level-2 gate (honest A/B over the established baseline)",
        f"- baseline = {l2.get('baseline', {}).get('feats')}: AP **{l2.get('baseline', {}).get('ap')}** "
        f"(LOBO {l2.get('baseline', {}).get('lobo_ap')})",
        "",
        "| candidate | lvl | ΔAP (temporal) | ΔAP (LOBO) | verdict |",
        "|---|:-:|--:|--:|:--|",
    ]
    for c in l2.get("candidates", []):
        lobo = "-" if c["lobo_delta_ap"] is None else f"{c['lobo_delta_ap']:+}"
        lines.append(f"| {c['name']} | {c['level']} | {c['delta_ap']:+} | {lobo} | **{c['verdict']}** |")

    # 3. L3
    lines += ["", "## 3. Level-3 composition (only between complementary KEPT L2 signals)",
              f"- KEPT L2 signals: {res['kept_l2'] or '(none)'}"]
    if prop.get("pairs"):
        lines += ["", "| L2 pair | residual corr | independence |", "|---|--:|--:|"]
        for p in prop["pairs"]:
            lines.append(f"| {p['pair'][0]} × {p['pair'][1]} | {p['residual_corr']:+} | {p['independence']} |")
        lines.append("")
        lines.append(f"- {prop.get('reason', '')} (threshold {prop.get('min_independence')})")
    else:
        lines.append(f"- {prop.get('reason', prop.get('note', 'no pairs'))}")
    if l3:
        lines.append("")
        for c in l3["candidates"]:
            lobo = "-" if c["lobo_delta_ap"] is None else f"{c['lobo_delta_ap']:+}"
            lines.append(f"- L3 **{c['name']}** gated: ΔAP {c['delta_ap']:+} (LOBO {lobo}) → "
                         f"**{c['verdict']}** — {c['reason']}")

    lines += ["", "## Takeaway",
              "- Level-3 needs ≥2 KEPT, residual-INDEPENDENT level-2 signals; the table above picks the pair.",
              "- Nothing here is materialised to gold — KEEP only means a signal earned its place in the model."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default="bacteria_results/rainfall")
    ap.add_argument("--discharge-dir", default="bacteria_results/discharge")
    ap.add_argument("--cdip-dir", default="bacteria_results/cdip_waves")
    ap.add_argument("--tide-dir", default="bacteria_results/tide_stages")
    ap.add_argument("--l2", default=",".join(DEFAULT_L2), help="comma-separated L2 candidates to gate")
    ap.add_argument("--reveal-lag-days", type=int, default=2)
    ap.add_argument("--label", default="enterococcus", choices=["any", "enterococcus"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--include-san-diego", action="store_true")
    ap.add_argument("--no-lobo", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[run_signal_discovery] obs not found: {obs}")
        return 2

    dirs = {"rain_dir": sl.existing_dir(args.rain_dir), "discharge_dir": sl.existing_dir(args.discharge_dir),
            "cdip_dir": sl.existing_dir(args.cdip_dir), "tide_dir": sl.existing_dir(args.tide_dir)}
    res = run(obs, l2=[c.strip() for c in args.l2.split(",") if c.strip()], dirs=dirs,
              reveal_lag_days=args.reveal_lag_days, label=args.label,
              exclude_counties=() if args.include_san_diego else ("San Diego",),
              k=args.k, n_perm=args.n_perm, do_lobo=not args.no_lobo)

    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signal_discovery.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (out_dir / "SIGNAL_DISCOVERY.md").write_text(md, encoding="utf-8")
    sl.echo_markdown(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
