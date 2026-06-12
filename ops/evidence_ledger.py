#!/usr/bin/env python3
"""Build a cross-target evidence ledger for the Monterey Bay AI Lab.

The command center needs one place that separates:
  * what data exists,
  * what models/results were run,
  * what is claimable,
  * what is evidence-stage,
  * what is blocked or unsafe.

This script reads local manifests/reports only. It does not train models or fetch data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reports" / "evidence_ledger"
LEDGER_JSON = OUTDIR / "evidence_ledger.json"
LEDGER_MD = OUTDIR / "evidence_ledger.md"
SOURCES_CSV = OUTDIR / "sources.csv"
RESULTS_CSV = OUTDIR / "results.csv"
CLAIMS_CSV = OUTDIR / "claims.csv"
AUDIT_CSV = OUTDIR / "audit_findings.csv"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_rel(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def artifact_state(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"exists": False, "path": ""}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {"exists": False, "path": safe_rel(p)}
    return {
        "exists": True,
        "path": safe_rel(p),
        "size_mb": round(p.stat().st_size / 1_000_000, 3) if p.is_file() else None,
        "mtime": pd.Timestamp.utcfromtimestamp(p.stat().st_mtime).isoformat(),
    }


def collect_sources() -> list[dict[str, Any]]:
    fetch_status = load_json(ROOT / "reports" / "data_fetch" / "fetch_status_matrix.json")
    coverage_rows = load_json(ROOT / "reports" / "data_fetch" / "latest_coverage_audit.json") or []
    coverage_by_source = {row.get("source"): row for row in coverage_rows if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for src in fetch_status.get("sources", []):
        source = src.get("source", "")
        seen.add(source)
        coverage = coverage_by_source.get(source, {})
        curated_path = src.get("curated_path", "")
        rows.append({
            "source": source,
            "title": src.get("title", ""),
            "layer": "registered_curated",
            "status": src.get("status", ""),
            "claim_status": "model_ready" if src.get("status") == "READY_FOR_MODELING" else "needs_review",
            "rows": int(src.get("rows") or 0),
            "columns": int(src.get("columns") or 0),
            "date_min": src.get("date_min", ""),
            "date_max": src.get("date_max", ""),
            "curated_path": curated_path,
            "curated_exists": artifact_state(curated_path)["exists"],
            "raw_chunk_count": raw_chunk_count(source),
            "coverage_finding": "; ".join(coverage.get("gaps", [])) if coverage else "",
            "notes": "",
        })

    data_fetch_dir = ROOT / "reports" / "data_fetch"
    if data_fetch_dir.exists():
        for source_dir in sorted(p for p in data_fetch_dir.iterdir() if p.is_dir()):
            source = source_dir.name
            if source in seen:
                continue
            manifest = load_json(source_dir / "manifest.json")
            validation = load_json(source_dir / "validation.json")
            coverage = load_json(source_dir / "coverage.json")
            if not (manifest or validation or coverage):
                continue
            source_name = manifest.get("source") or validation.get("source") or coverage.get("source") or source
            curated_path = manifest.get("curated_path") or validation.get("path") or manifest.get("hourly_path") or ""
            rows.append({
                "source": source_name,
                "title": manifest.get("title", ""),
                "layer": "triad_or_derived",
                "status": "READY_FOR_MODELING" if validation.get("passed") else "FETCHED_NEEDS_REVIEW",
                "claim_status": "derived_model_ready" if validation.get("passed") else "needs_review",
                "rows": int(manifest.get("rows") or validation.get("rows") or coverage.get("rows") or 0),
                "columns": int(manifest.get("columns") or validation.get("columns") or coverage.get("columns") or 0),
                "date_min": coverage.get("date_min") or validation.get("date_min") or manifest.get("date_min") or "",
                "date_max": coverage.get("date_max") or validation.get("date_max") or manifest.get("date_max") or "",
                "curated_path": curated_path,
                "curated_exists": artifact_state(curated_path)["exists"],
                "raw_chunk_count": raw_chunk_count(source_name),
                "coverage_finding": "; ".join(coverage.get("gaps", [])) if coverage else "",
                "notes": "Not in fetch_status_matrix; surfaced from manifest/coverage/validation triad.",
            })

    return sorted(rows, key=lambda r: (r["claim_status"], r["source"]))


def raw_chunk_count(source: str) -> int:
    raw_dir = ROOT / "data" / "external_raw" / source
    if not raw_dir.exists():
        return 0
    return len(list(raw_dir.glob("chunk_*.parquet")))


def metric_tuple(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and len(value) >= 4:
        return {"ap": value[0], "auc": value[1], "ece": value[2], "deploy_ready": value[3]}
    return {}


def collect_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    add = rows.append

    # Bacteria benchmark / champion upgrade
    champ = load_json(ROOT / "reports" / "operational_benchmark" / "champion_bakeoff.json")
    if champ:
        for stratum, old_vals in champ.get("champion", {}).items():
            new_vals = champ.get("best_catboost_all_air_warmspell", {}).get(stratum)
            old_m = metric_tuple(old_vals)
            new_m = metric_tuple(new_vals)
            add({
                "target": "bacteria",
                "artifact": "reports/operational_benchmark/champion_bakeoff.json",
                "result": f"catboost_all_air_vs_champion_{stratum}",
                "status": "promotion_candidate",
                "primary_metric": "AP",
                "baseline": old_m.get("ap"),
                "candidate": new_m.get("ap"),
                "delta": round((new_m.get("ap", 0) - old_m.get("ap", 0)), 4) if new_m else None,
                "caveat": "Measured upgrade candidate; needs normal promotion/release gate before public champion claim.",
            })

    gate = load_json(ROOT / "reports" / "bacteria" / "new_source_signal_gate" / "verdicts.json")
    for cand in gate.get("candidates", []):
        add({
            "target": "bacteria",
            "artifact": "reports/bacteria/new_source_signal_gate/verdicts.json",
            "result": cand.get("name"),
            "status": cand.get("verdict"),
            "primary_metric": "delta_ap",
            "baseline": gate.get("baseline", {}).get("ap"),
            "candidate": cand.get("ap"),
            "delta": cand.get("delta_ap"),
            "caveat": cand.get("reason", ""),
        })

    # HAB / DA
    da = load_json(ROOT / "reports" / "hab" / "da_forecast.json")
    if da:
        test = da.get("test", {}) or da.get("summary", {}) or da
        add({
            "target": "hab_da",
            "artifact": "reports/hab/da_forecast.json",
            "result": "da_forecast_hgbt",
            "status": "claimable_registry_model",
            "primary_metric": "AP",
            "baseline": test.get("best_naive_ap") or test.get("naive_ap"),
            "candidate": test.get("avg_precision") or test.get("ap") or test.get("test_ap"),
            "delta": None,
            "caveat": "Claimable per project_status/model registry; prospective pilot still needed for operational use.",
        })

    da_sources = load_json(ROOT / "reports" / "hab" / "da_new_source_signals.json")
    for name, item in da_sources.get("sources_tested", {}).items():
        add({
            "target": "hab_da",
            "artifact": "reports/hab/da_new_source_signals.json",
            "result": name,
            "status": item.get("verdict"),
            "primary_metric": "delta_ap_vs_lean",
            "baseline": da_sources.get("lean_test_ap"),
            "candidate": item.get("ap_lean_plus_source"),
            "delta": item.get("delta_ap_vs_lean"),
            "caveat": item.get("reason", ""),
        })

    upw = load_json(ROOT / "reports" / "hab" / "da_upwelling_gate" / "verdict.json")
    if upw:
        add({
            "target": "hab_da",
            "artifact": "reports/hab/da_upwelling_gate/verdict.json",
            "result": "CUTI_BEUTI_upwelling",
            "status": upw.get("verdict"),
            "primary_metric": "delta_ap",
            "baseline": upw.get("base_ap"),
            "candidate": upw.get("cand_ap"),
            "delta": upw.get("delta_ap"),
            "caveat": f"LOSO delta {upw.get('loso_delta_ap')}; mechanistic but failed generalization gate.",
        })

    # Whale / mortality
    whale = load_json(ROOT / "research" / "whale" / "reports" / "da_mortality_evidence.json")
    if whale:
        verdict = whale.get("verdict", {})
        add({
            "target": "whale_mortality",
            "artifact": "research/whale/reports/da_mortality_evidence.json",
            "result": "DA_window_vs_starvation_control",
            "status": "evidence_stage",
            "primary_metric": "pDA_ratio_and_robust_z",
            "baseline": verdict.get("PRIMARY_clean_control_gray_whale_2019_2023", {}).get("pDA_ratio"),
            "candidate": verdict.get("PRIMARY_headline_DA_window_2024_2026", {}).get("pDA_ratio"),
            "delta": None,
            "caveat": verdict.get("caveat", "Descriptive/correlational; not necropsy-level causal proof."),
        })

    panel = artifact_state("data/whale_mortality/mortality_panel.parquet")
    if panel["exists"]:
        add({
            "target": "whale_mortality",
            "artifact": panel["path"],
            "result": "monthly_mortality_panel",
            "status": "derived_panel",
            "primary_metric": "rows",
            "baseline": None,
            "candidate": parquet_rows(ROOT / panel["path"]),
            "delta": None,
            "caveat": "Derived panel; modeling report not present unless mortality_model.json is generated.",
        })

    # Data understanding / QA
    corr = load_json(ROOT / "reports" / "data_lab" / "correlation_discovery.json")
    if corr:
        add({
            "target": "data_understanding",
            "artifact": "reports/data_lab/correlation_discovery.json",
            "result": "cross_source_correlation_discovery",
            "status": "evidence_stage",
            "primary_metric": "variables",
            "baseline": None,
            "candidate": corr.get("matrix", {}).get("n_variables"),
            "delta": None,
            "caveat": "Correlation/lead-lag/coherence, not causal model evidence.",
        })
    consistency = load_json(ROOT / "reports" / "data_lab" / "source_consistency.json")
    if consistency:
        findings = consistency.get("findings") or consistency.get("issues") or []
        add({
            "target": "data_understanding",
            "artifact": "reports/data_lab/source_consistency.json",
            "result": "source_unit_consistency_gate",
            "status": "qa_gate",
            "primary_metric": "findings",
            "baseline": None,
            "candidate": len(findings) if isinstance(findings, list) else None,
            "delta": None,
            "caveat": "Use to catch unit/source normalization bugs before model claims.",
        })

    return rows


def parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def collect_claims(results: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    project_status = load_json(ROOT / "reports" / "project_status" / "project_status.json")
    claims: list[dict[str, Any]] = []
    for item in project_status.get("public_claims", {}).get("claimable_models", []):
        claims.append({
            "claim": item.get("model_id") or item.get("name") or str(item),
            "claim_status": "safe_to_claim",
            "evidence": item.get("evidence", ""),
            "caveat": item.get("reason", ""),
        })
    for row in results:
        if row["status"] in {"promotion_candidate", "evidence_stage", "derived_panel"}:
            claims.append({
                "claim": f"{row['target']}:{row['result']}",
                "claim_status": "not_public_yet",
                "evidence": row["artifact"],
                "caveat": row["caveat"],
            })
    for row in sources:
        if row["claim_status"] == "needs_review":
            claims.append({
                "claim": f"source:{row['source']}",
                "claim_status": "blocked_or_review",
                "evidence": row["curated_path"],
                "caveat": row["coverage_finding"] or row["notes"],
            })
    return claims


def collect_audit_findings(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    fetch_sources = {row["source"] for row in sources if row.get("layer") == "registered_curated"}

    inventory = load_json(ROOT / "lakehouse" / "silver" / "source_inventory" / "source_inventory.json")
    inventory_rows = inventory.get("sources", []) if isinstance(inventory, dict) else inventory
    inventory_sources = {row.get("source") for row in inventory_rows if isinstance(row, dict)}
    if fetch_sources and inventory_sources:
        missing = sorted(fetch_sources - inventory_sources)
        if missing:
            findings.append({
                "severity": "review",
                "finding": "fetch_matrix_sources_missing_from_silver_inventory",
                "detail": ", ".join(missing),
                "artifact": "lakehouse/silver/source_inventory/source_inventory.json",
            })

    tide_fetch = next((row for row in sources if row.get("source") == "tide_stages"), None)
    tide_actual = ROOT / "bacteria_results" / "tide_stages" / "tide_stages.parquet"
    if tide_fetch and tide_actual.exists():
        actual_rows = parquet_rows(tide_actual)
        if actual_rows and int(tide_fetch.get("rows") or 0) != actual_rows:
            findings.append({
                "severity": "review",
                "finding": "tide_stages_registry_row_count_differs_from_file",
                "detail": f"fetch matrix rows={tide_fetch.get('rows')}; bacteria_results rows={actual_rows}",
                "artifact": "bacteria_results/tide_stages/tide_stages.parquet",
            })

    raw_only = []
    for source in ["charm", "noaa_strandings"]:
        raw_dir = ROOT / "data" / "external_raw" / source
        if raw_dir.exists() and source not in fetch_sources:
            raw_only.append(source)
    if raw_only:
        findings.append({
            "severity": "review",
            "finding": "raw_sources_not_first_class_fetch_status_sources",
            "detail": ", ".join(raw_only),
            "artifact": "data/external_raw",
        })

    source_consistency = load_json(ROOT / "reports" / "data_lab" / "source_consistency.json")
    if str(source_consistency.get("overall_status", "")).upper() == "FAIL":
        findings.append({
            "severity": "blocker_for_unit_claims",
            "finding": "source_consistency_gate_failed",
            "detail": "Wind/solar absolute-unit comparisons need resolution before absolute cross-source claims.",
            "artifact": "reports/data_lab/source_consistency.json",
        })

    suite_md = artifact_state("reports/model_hardening/MODEL_SUITE.md")
    suite_matrix = load_json(ROOT / "reports" / "model_hardening" / "model_suite_status_matrix.json")
    if suite_md["exists"] and suite_matrix:
        findings.append({
            "severity": "review",
            "finding": "model_suite_artifacts_may_be_stale_or_scope_mismatched",
            "detail": "MODEL_SUITE.md and model_suite_status_matrix.json report different total/claimable model counts; reconcile before using as public registry.",
            "artifact": "reports/model_hardening",
        })

    findings.append({
        "severity": "needed",
        "finding": "alias_resolution_needed_for_production_ledger",
        "detail": "Correlation/discovery aliases such as hfradar, mursst, viirs, tide, usgs, nasapower should map to canonical fetch/source IDs.",
        "artifact": "reports/data_lab/correlation_discovery.json",
    })
    return findings


def build() -> dict[str, Any]:
    sources = collect_sources()
    results = collect_results()
    claims = collect_claims(results, sources)
    audit_findings = collect_audit_findings(sources)
    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sources_total": len(sources),
        "sources_ready": sum(1 for s in sources if s["status"] == "READY_FOR_MODELING"),
        "source_rows_total": int(sum(int(s.get("rows") or 0) for s in sources)),
        "results_total": len(results),
        "safe_claims": sum(1 for c in claims if c["claim_status"] == "safe_to_claim"),
        "not_public_claims": sum(1 for c in claims if c["claim_status"] == "not_public_yet"),
        "blocked_or_review": sum(1 for c in claims if c["claim_status"] == "blocked_or_review"),
        "audit_findings": len(audit_findings),
    }
    ledger = {
        "summary": summary,
        "sources": sources,
        "results": results,
        "claims": claims,
        "audit_findings": audit_findings,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LEDGER_JSON.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(sources).to_csv(SOURCES_CSV, index=False)
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(claims).to_csv(CLAIMS_CSV, index=False)
    pd.DataFrame(audit_findings).to_csv(AUDIT_CSV, index=False)
    write_markdown(ledger)
    return ledger


def write_markdown(ledger: dict[str, Any]) -> None:
    s = ledger["summary"]
    lines = [
        "# Evidence Ledger",
        "",
        f"- generated_at: `{s['generated_at']}`",
        f"- sources: **{s['sources_ready']}/{s['sources_total']} ready**",
        f"- source rows: **{s['source_rows_total']:,}**",
        f"- result records: **{s['results_total']}**",
        f"- safe public claims: **{s['safe_claims']}**",
        f"- evidence-stage / not-public claims: **{s['not_public_claims']}**",
        f"- blocked/review items: **{s['blocked_or_review']}**",
        f"- audit findings: **{s['audit_findings']}**",
        "",
        "## Audit Findings",
        "",
        "| severity | finding | detail | artifact |",
        "|---|---|---|---|",
    ]
    for row in ledger["audit_findings"]:
        lines.append(
            f"| {row['severity']} | {row['finding']} | {str(row['detail']).replace('|', '/')} | {row['artifact']} |"
        )
    lines += [
        "",
        "## High-Signal Results",
        "",
        "| target | result | status | metric | baseline | candidate | delta | caveat |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in ledger["results"]:
        lines.append(
            f"| {row['target']} | {row['result']} | {row['status']} | {row['primary_metric']} | "
            f"{row.get('baseline', '')} | {row.get('candidate', '')} | {row.get('delta', '')} | "
            f"{str(row.get('caveat', '')).replace('|', '/')} |"
        )
    lines += ["", "## Source Layers", "", "| source | layer | status | rows | date_min | date_max | caveat |", "|---|---|---|--:|---|---|---|"]
    for row in ledger["sources"]:
        lines.append(
            f"| {row['source']} | {row['layer']} | {row['status']} | {row['rows']} | "
            f"{str(row['date_min'])[:10]} | {str(row['date_max'])[:10]} | "
            f"{str(row.get('coverage_finding') or row.get('notes') or '').replace('|', '/')} |"
        )
    LEDGER_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ledger = build()
    print(json.dumps(ledger["summary"], indent=2))
    print(f"wrote {LEDGER_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
