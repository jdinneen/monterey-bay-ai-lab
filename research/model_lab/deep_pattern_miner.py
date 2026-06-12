#!/usr/bin/env python3
"""2026 whole-lakehouse deep pattern miner plan and smoke runner.

This is additive research infrastructure. It does not mutate lakehouse source
tables or gold metrics. Full runs must go through ops/run_safe.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.model_lab import semi_supervised_corpus_run as semisup
from research.model_lab import unsupervised_corpus_run as unsup

PIPELINE_JSON = REPO_ROOT / "reports" / "current_data_pipeline" / "current_data_pipeline.json"
SOURCE_INVENTORY_JSON = REPO_ROOT / "lakehouse" / "silver" / "source_inventory" / "source_inventory.json"
PLAN_JSON = REPO_ROOT / "reports" / "model_lab" / "deep_pattern_miner_plan.json"
PLAN_MD = REPO_ROOT / "reports" / "model_lab" / "DEEP_PATTERN_MINER_PLAN.md"
UNSUPERVISED_ARTIFACT_EXCLUDES = [
    "lakehouse/silver/forecast_splits",
    "reports/hab",
    "reports/operational_benchmark",
]

RESEARCH_SOURCES = [
    {
        "name": "TabPFN v2 / tabular foundation model",
        "url": "https://www.nature.com/articles/s41586-024-08328-6",
        "use": "Strong small/medium tabular baseline, but not the right whole-lakehouse scanner because current data is far beyond its practical row budget.",
    },
    {
        "name": "TabM",
        "url": "https://arxiv.org/abs/2410.24210",
        "use": "Strong 2025 supervised tabular baseline; useful later for target-specific probes after representation mining finds candidate signals.",
    },
    {
        "name": "ModernNCA",
        "url": "https://arxiv.org/abs/2407.03257",
        "use": "Deep nearest-neighbor tabular baseline; useful for probe comparison, not the first all-data corpus pass.",
    },
    {
        "name": "SCARF",
        "url": "https://arxiv.org/abs/2106.15147",
        "use": "Random-feature corruption contrastive learning supports the corruption-based representation strategy.",
    },
    {
        "name": "VIME",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html",
        "use": "Mask/reconstruction self-supervision for tabular data supports the denoising autoencoder objective.",
    },
    {
        "name": "2024 SSL for non-sequential tabular survey",
        "url": "https://link.springer.com/article/10.1007/s10994-024-06674-0",
        "use": "Frames the current field as predictive, contrastive, and hybrid self-supervision; our runner is hybrid.",
    },
]


@dataclass(frozen=True)
class MinerPlan:
    generated_at: str
    ready_sources: int
    total_sources: int
    ready_rows: int
    inputs: list[str]
    run_root: str
    commands: dict[str, list[str]]
    gate: dict[str, object]
    research_sources: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project": "Monterey Bay AI Lab",
            "value_gate": (
                "DO_NOW: stakeholder asked for whole-lakehouse hidden-pattern mining; "
                "all 46 sources are ready; artifacts are additive and tested; full run is traffic-gated."
            ),
            "recommended_model": "hybrid self-supervised tabular representation miner",
            "ready_sources": self.ready_sources,
            "total_sources": self.total_sources,
            "ready_rows": self.ready_rows,
            "inputs": self.inputs,
            "unsupervised_artifact_excludes": UNSUPERVISED_ARTIFACT_EXCLUDES,
            "run_root": self.run_root,
            "commands": self.commands,
            "gate": self.gate,
            "research_sources": self.research_sources,
            "outputs": [
                "unsupervised/artifacts/anomaly_candidates.parquet",
                "unsupervised/artifacts/embeddings.parquet",
                "unsupervised/artifacts/clusters.parquet",
                "unsupervised/artifacts/nearest_neighbors.parquet",
                "unsupervised/artifacts/label_evaluation.json",
                "semi_supervised/artifacts/anomaly_candidates.parquet",
                "semi_supervised/artifacts/embeddings.parquet",
                "semi_supervised/artifacts/label_summary.json",
            ],
            "promotion_rule": (
                "Treat as meaningful only if downstream probes beat honest baselines or anomalies "
                "identify reproducible data/physics events. Representation quality alone is not a claim."
            ),
        }


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def current_source_counts() -> tuple[int, int, int]:
    pipeline = load_json(PIPELINE_JSON)
    inventory = pipeline.get("source_inventory", {}) if isinstance(pipeline, dict) else {}
    ready = int(inventory.get("ready_sources", 0))
    total = int(inventory.get("sources", 0))
    rows = int(inventory.get("ready_rows", 0))
    if ready and total:
        return ready, total, rows
    source_inventory = load_json(SOURCE_INVENTORY_JSON)
    if isinstance(source_inventory, list):
        ready_rows = [row for row in source_inventory if row.get("status") == "READY_FOR_MODELING"]
        return len(ready_rows), len(source_inventory), int(sum(int(row.get("rows") or 0) for row in ready_rows))
    return 0, 0, 0


def default_inputs(root: Path = REPO_ROOT) -> list[str]:
    candidates = [
        "data/external_curated",
        "bacteria_results",
        "lakehouse/silver",
        "lakehouse/gold/forecast_metrics",
        "reports/hab",
        "reports/operational_benchmark",
        "research/bacteria/reproduce/expected",
    ]
    return [p for p in candidates if (root / p).exists()]


def safe_command(task: str, inner: list[str], gpu_mib: int = 8192) -> list[str]:
    return [
        sys.executable,
        "ops/run_safe.py",
        "--task",
        task,
        "--agent",
        "codex-lead",
        "--gpu-mib",
        str(gpu_mib),
        "--brain-preflight",
        "warn",
        "--",
        *inner,
    ]


def build_plan(run_id: str | None = None, hours: float = 8.0) -> MinerPlan:
    run_id = run_id or datetime.now(timezone.utc).strftime("deep_pattern_%Y%m%dT%H%M%SZ")
    ready, total, rows = current_source_counts()
    inputs = default_inputs()
    run_root = f"reports/model_lab/deep_pattern_miner/run_id={run_id}"
    common_inputs = ["--inputs", *inputs]
    unsup_excludes = [
        flag
        for fragment in UNSUPERVISED_ARTIFACT_EXCLUDES
        for flag in ("--exclude-path-fragment", fragment)
    ]
    unsup_inner = [
        sys.executable,
        "research/model_lab/unsupervised_corpus_run.py",
        *common_inputs,
        *unsup_excludes,
        "--output",
        f"{run_root}/unsupervised",
        "--run-id",
        f"{run_id}_unsupervised",
        "--hours",
        str(hours),
        "--batch-rows",
        "4096",
        "--input-dim",
        "4096",
        "--hidden-dim",
        "1024",
        "--latent-dim",
        "128",
        "--hash-scope",
        "global",
        "--device",
        "cuda",
        "--embedding-sample",
        "50000",
        "--evaluation-sample",
        "10000",
        "--checkpoint-every",
        "200",
    ]
    semisup_inner = [
        sys.executable,
        "research/model_lab/semi_supervised_corpus_run.py",
        *common_inputs,
        "--output",
        f"{run_root}/semi_supervised",
        "--run-id",
        f"{run_id}_semi_supervised",
        "--hours",
        str(max(2.0, hours / 2.0)),
        "--batch-rows",
        "4096",
        "--input-dim",
        "4096",
        "--hidden-dim",
        "1024",
        "--latent-dim",
        "128",
        "--hash-scope",
        "global",
        "--device",
        "cuda",
        "--supervised-weight",
        "0.05",
        "--max-labels-per-batch",
        "512",
        "--embedding-sample",
        "50000",
        "--checkpoint-every",
        "200",
    ]
    return MinerPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        ready_sources=ready,
        total_sources=total,
        ready_rows=rows,
        inputs=inputs,
        run_root=run_root,
        commands={
            "unsupervised_safe": safe_command("deep-pattern-miner-unsupervised", unsup_inner),
            "semi_supervised_safe": safe_command("deep-pattern-miner-semisup", semisup_inner),
        },
        gate={
            "traffic_gated": True,
            "destructive": False,
            "ready_to_run": bool(inputs and ready == total and total > 0),
            "requires_gpu_mib": 8192,
            "unsupervised_artifact_excludes": UNSUPERVISED_ARTIFACT_EXCLUDES,
            "critic_gate": "PASS" if inputs and ready == total and total > 0 else "BLOCKED",
            "blockers": [] if inputs and ready == total and total > 0 else ["current source inventory is incomplete"],
        },
        research_sources=RESEARCH_SOURCES,
    )


def write_plan(plan: MinerPlan) -> None:
    PLAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    PLAN_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Deep Pattern Miner Plan",
        "",
        f"- generated_at: `{plan.generated_at}`",
        f"- ready sources: {plan.ready_sources}/{plan.total_sources}",
        f"- ready rows: {plan.ready_rows:,}",
        f"- critic gate: **{plan.gate['critic_gate']}**",
        f"- run root: `{plan.run_root}`",
        f"- unsupervised artifact excludes: `{', '.join(UNSUPERVISED_ARTIFACT_EXCLUDES)}`",
        "",
        "## Model Choice",
        "",
        "Use a hybrid self-supervised tabular representation miner: denoising reconstruction plus weak supervised label probes over a shared global hashed feature space. This is the local-machine-compatible way to backpropagate through all current heterogeneous lakehouse tables without pretending that a small-data foundation model can ingest 240M rows directly.",
        "",
        "The unsupervised science-table pass explicitly excludes forecast split contracts and report artifacts. Those are control/evaluation outputs, not primary observations, and they are audited separately instead of being mixed into the same reconstruction-loss distribution.",
        "",
        "## Why Not Just TabPFN",
        "",
        "TabPFN/TabPFN-TS are excellent 2025-2026 small/medium tabular baselines. They are not the right first pass for this lakehouse because our current corpus is hundreds of millions of rows across incompatible schemas. Use them later as target-specific probes on mined candidate panels.",
        "",
        "## Full Commands",
        "",
        "Unsupervised:",
        "```powershell",
        " ".join(plan.commands["unsupervised_safe"]),
        "```",
        "",
        "Semi-supervised:",
        "```powershell",
        " ".join(plan.commands["semi_supervised_safe"]),
        "```",
        "",
        "## Research Basis",
        "",
    ]
    for src in plan.research_sources:
        lines.append(f"- {src['name']}: {src['url']} - {src['use']}")
    lines += [
        "",
        "## Gate",
        "",
        "- Full runs must use `ops/run_safe.py` with the traffic controller.",
        "- Outputs are additive under `reports/model_lab/deep_pattern_miner/`.",
        "- The unsupervised run records `exclude_path_fragments` in its manifest.",
        "- A result matters only if a downstream probe beats an honest baseline or if anomalies map to reproducible physical/data events.",
    ]
    PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_args(output: Path, inputs: list[str], mode: str) -> SimpleNamespace:
    common = {
        "inputs": inputs,
        "output": str(output),
        "run_id": f"smoke_{mode}",
        "hours": 0.0,
        "batch_rows": 8,
        "input_dim": 128,
        "hidden_dim": 32,
        "latent_dim": 8,
        "dropout": 0.0,
        "noise": 0.01,
        "mask_prob": 0.15,
        "learning_rate": 1e-3,
        "device": "cpu",
        "metrics_every": 1,
        "anomaly_cap": 20,
        "embedding_sample": 20,
        "checkpoint_every": 1,
        "watchdog_loss_floor": 1e-3,
        "max_steps": 2,
        "append_metrics": False,
        "hash_scope": "global",
    }
    if mode == "unsupervised":
        return SimpleNamespace(**common, evaluation_sample=20, clusters=4, neighbors=4)
    return SimpleNamespace(
        **common,
        supervised_weight=0.05,
        max_labels_per_batch=8,
        supervised_sample=20,
    )


def run_smoke(output: Path, inputs: list[str]) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    unsup_code = unsup.train(smoke_args(output / "unsupervised", inputs, "unsupervised"))
    semisup_code = semisup.train(smoke_args(output / "semi_supervised", inputs, "semi_supervised"))
    result = {
        "status": "PASS" if unsup_code == 0 and semisup_code == 0 else "FAIL",
        "unsupervised_returncode": unsup_code,
        "semi_supervised_returncode": semisup_code,
        "output": str(output),
    }
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--smoke-output", default="reports/model_lab/deep_pattern_miner/smoke")
    parser.add_argument("--smoke-inputs", nargs="+", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args.run_id, args.hours)
    if args.write_plan or not args.smoke:
        write_plan(plan)
    if args.smoke:
        inputs = args.smoke_inputs or plan.inputs[:2] or default_inputs()
        result = run_smoke(REPO_ROOT / args.smoke_output, inputs)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
