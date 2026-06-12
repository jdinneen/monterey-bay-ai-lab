#!/usr/bin/env python3
"""Local browser command center for Monterey Bay AI Lab agent operations.

This server is deliberately small and local-only. It exposes repository state,
Agent Brain query/preflight/reflect actions, traffic locks, and GPU status without
adding a framework or remote deployment surface.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "ops" / "command_center"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ops.agent_brain_core import (  # noqa: E402
    BRAIN_DIR,
    GateInput,
    append_jsonl,
    brain_snapshot,
    parse_bool,
    query_brain,
    reflection_payload,
    rejection_payload,
    run_value_gate,
)
from ops.gpu_admission import query_gpu_state, query_vram_processes  # noqa: E402


def _read_jsonl_tail(path: Path, limit: int = 8) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            rows.append({"malformed": text})
    return rows[-limit:]


def _read_markdown_sections(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    sections: list[dict[str, str]] = []
    title = ""
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title:
                sections.append({"title": title, "body": "\n".join(body).strip()})
            title = line[3:].strip()
            body = []
        elif title:
            body.append(line)
    if title:
        sections.append({"title": title, "body": "\n".join(body).strip()})
    return sections


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _stack_watch_state() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/state", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _layer_for_process(proc: dict[str, Any]) -> str:
    stage = str(proc.get("stage") or "").lower()
    kind = str(proc.get("kind") or "").lower()
    command = str(proc.get("command") or proc.get("command_line") or "").lower()
    if kind == "assistant" or stage == "agent":
        return "agents"
    if stage in {"training", "model_server"} or kind in {"continual", "neuralforecast", "orchestrator", "llm"}:
        return "training"
    if stage in {"bronze", "lakehouse"} or any(
        token in command
        for token in [
            "autonomous_rate_limit_fetcher",
            "bacteria_ingestion",
            "mbal_drivers_build",
            "noaa",
            "mur",
        ]
    ):
        return "bronze"
    if stage in {"evaluation", "uncertainty"} or "evaluate.py" in command or "monte_carlo" in command:
        return "evaluation"
    if "promotion" in command or "evidence_gate" in command or "release_gate" in command:
        return "claims"
    return "telemetry"


def _layer_for_lock(line: str) -> str:
    text = line.lower()
    if "agent_brain" in text or "value_gate" in text:
        return "brain"
    if "command_center" in text or "agent_lock" in text or "traffic" in text or "run_safe" in text:
        return "traffic"
    if "fetch" in text or "noaa" in text or "source" in text:
        return "bronze"
    if "train" in text or "moe" in text or "forecast" in text:
        return "training"
    if "eval" in text or "metric" in text:
        return "evaluation"
    if "promotion" in text or "release" in text or "claim" in text:
        return "claims"
    return "agents"


def _sentinel_task() -> dict[str, Any] | None:
    state = _read_json(REPO_ROOT / "ops" / "sentinel" / "state.json", {})
    if not state or not state.get("active"):
        return None
    return {
        "layer": "sentinel",
        "kind": "sentinel",
        "label": str(state.get("task") or "background task"),
        "detail": f"PID {state.get('pid')} | {state.get('log_path') or 'no log'}",
        "source": "ops/sentinel/state.json",
    }


def _in_flight_by_layer(stack_state: dict[str, Any] | None, locks: list[str]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if stack_state:
        for proc in stack_state.get("processes", [])[:80]:
            layer = _layer_for_process(proc)
            label = proc.get("model") or proc.get("kind") or proc.get("name") or f"pid {proc.get('pid')}"
            detail = proc.get("command") or proc.get("command_line") or ""
            items.append(
                {
                    "layer": layer,
                    "kind": str(proc.get("kind") or proc.get("stage") or "process"),
                    "label": str(label),
                    "detail": str(detail)[:220],
                    "source": f"pid {proc.get('pid')}",
                    "vram_mib": proc.get("vram_mib"),
                }
            )
    for line in locks:
        if not line or line == "no active claims":
            continue
        items.append(
            {
                "layer": _layer_for_lock(line),
                "kind": "lock",
                "label": line.split("  ")[0].strip(),
                "detail": line,
                "source": "ops/agent_lock.py status",
            }
        )
    sentinel = _sentinel_task()
    if sentinel:
        items.append(sentinel)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["layer"]), []).append(item)
    return {
        "items": items,
        "by_layer": grouped,
        "count": len(items),
    }


def stack_coverage() -> dict[str, Any]:
    registry = _read_json(REPO_ROOT / "ops" / "stack_registry.json", {"required_layers": []})
    stack_state = _stack_watch_state()
    live_nodes = {}
    if stack_state:
        for node in (stack_state.get("stack") or {}).get("nodes", []):
            live_nodes[str(node.get("id"))] = node

    command_center_layers = {
        "agents": True,
        "brain": True,
        "traffic": True,
        "sentinel": (REPO_ROOT / "ops" / "sentinel" / "state.json").exists(),
        "telemetry": (REPO_ROOT / "ops" / "telemetry" / "latest_system_telemetry.json").exists(),
    }
    rows = []
    for layer in registry.get("required_layers", []):
        layer_id = str(layer.get("id"))
        live = layer_id in live_nodes or command_center_layers.get(layer_id, False)
        rows.append(
            {
                **layer,
                "live": bool(live),
                "live_status": live_nodes.get(layer_id, {}).get("status") or ("covered" if live else "missing"),
                "meter": live_nodes.get(layer_id, {}).get("meter"),
            }
        )
    missing = [row for row in rows if row.get("must_show") and not row.get("live")]
    return {
        "registry_source": registry.get("source"),
        "anti_bloat_rule": registry.get("anti_bloat_rule"),
        "stack_watch_online": stack_state is not None,
        "rows": rows,
        "missing": missing,
        "covered_count": sum(1 for row in rows if row.get("live")),
        "required_count": len(rows),
    }


def _git_status() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=8,
    )
    if proc.returncode != 0:
        return [proc.stderr.strip() or "git status unavailable"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _lock_status() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "ops/agent_lock.py", "status"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=8,
    )
    text = proc.stdout.strip() or proc.stderr.strip()
    return [line for line in text.splitlines() if line.strip()]


def _gpu_payload() -> dict[str, Any]:
    state = query_gpu_state()
    processes = query_vram_processes(limit=6)
    return {
        "state": asdict(state) if state else None,
        "processes": [asdict(proc) for proc in processes],
    }


def _next_safe_action(*, git: list[str], locks: list[str], gpu: dict[str, Any]) -> dict[str, str]:
    live_locks = [line for line in locks if line and line != "no active claims" and not line.startswith("STALE")]
    state = gpu.get("state")
    if live_locks:
        return {
            "label": "Inspect active locks",
            "why": "Another agent or task has claimed work. Review lock holders before starting new changes.",
            "command": "python ops\\agent_lock.py status",
        }
    if state and state.get("used_pct", 0) > 0.85:
        return {
            "label": "Wait for GPU headroom",
            "why": "GPU used memory is above the admission comfort threshold.",
            "command": "python ops\\traffic_controller.py status",
        }
    if git:
        return {
            "label": "Run local tests before more edits",
            "why": "The working tree has changes; verify the current state before expanding scope.",
            "command": "python .\\ops\\run_tests.py",
        }
    return {
        "label": "Plan with Agent Brain preflight",
        "why": "No active lock blocker was detected. Start new work by proving value before taking a traffic slot.",
        "command": "python ops\\agent_preflight.py --idea \"...\" --agent codex --task \"...\" --stakeholder-question \"...\" --test tests/test_agent_brain.py --reversible true --valuable-now true --genuinely-new true",
    }


def command_center_state() -> dict[str, Any]:
    snapshot = brain_snapshot()
    locks = _lock_status()
    gpu = _gpu_payload()
    git = _git_status()
    stack_state = _stack_watch_state()
    return {
        "project": snapshot["project_state"].get("project", "Monterey Bay AI Lab"),
        "brain_status": snapshot["project_state"].get("brain_status", "unknown"),
        "deployment_boundary": snapshot["project_state"].get("deployment_boundary", ""),
        "contracts": snapshot["project_state"].get("primary_contracts", []),
        "baselines": snapshot["project_state"].get("current_baselines", {}),
        "invariants": snapshot["invariants"].get("invariants", []),
        "roles": snapshot["agent_roles"].get("roles", []),
        "operating_loop": _read_markdown_sections(BRAIN_DIR / "operating_loop.md"),
        "stack_coverage": stack_coverage(),
        "in_flight": _in_flight_by_layer(stack_state, locks),
        "learnings": _read_jsonl_tail(BRAIN_DIR / "learnings.jsonl", limit=8),
        "rejections": _read_jsonl_tail(BRAIN_DIR / "rejected_ideas.jsonl", limit=8),
        "decisions": _read_jsonl_tail(BRAIN_DIR / "decision_log.jsonl", limit=5),
        "locks": locks,
        "gpu": gpu,
        "git": git,
        "next_safe_action": _next_safe_action(git=git, locks=locks, gpu=gpu),
        "commands": {
            "preflight": "python ops\\agent_preflight.py --idea \"...\" --agent codex --task \"...\" --stakeholder-question \"...\" --test tests/test_agent_brain.py --reversible true --valuable-now true --genuinely-new true",
            "safe_run": "python ops\\run_safe.py --task \"...\" --agent codex --brain-preflight enforce --preflight-idea \"...\" --preflight-stakeholder-question \"...\" --preflight-test tests/test_agent_brain.py --preflight-reversible true --preflight-valuable-now true --preflight-genuinely-new true -- python your_script.py",
            "tests": "python .\\ops\\run_tests.py",
        },
    }


def _gate_from_payload(payload: dict[str, Any]) -> GateInput:
    return GateInput(
        idea=str(payload.get("idea", "")),
        agent=str(payload.get("agent", "")),
        task=str(payload.get("task", "")),
        baseline=str(payload.get("baseline", "")),
        test=str(payload.get("test", "")),
        reversible=parse_bool(payload.get("reversible", False)),
        valuable_now=parse_bool(payload.get("valuable_now", False)),
        genuinely_new=parse_bool(payload.get("genuinely_new", False)),
        stakeholder_question=str(payload.get("stakeholder_question", "")),
        correctness_fix=parse_bool(payload.get("correctness_fix", False)),
        kill_category=str(payload.get("kill_category", "bloat")),
    )


class CommandCenterHandler(BaseHTTPRequestHandler):
    server_version = "MBALCommandCenter/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/api/state", "/api/state/"}:
            self._json(command_center_state())
            return
        if self.path.startswith("/api/query"):
            from urllib.parse import parse_qs, urlparse

            params = parse_qs(urlparse(self.path).query)
            query = params.get("q", [""])[0]
            self._json({"query": query, "hits": query_brain(query, limit=12)})
            return
        self._serve_static()

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_payload()
        except json.JSONDecodeError as exc:
            self._json({"error": f"invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/preflight":
            gate = _gate_from_payload(payload)
            result = run_value_gate(gate)
            self._json({"result": asdict(result), "hits": query_brain(gate.idea, limit=8)})
            return

        if self.path == "/api/reflect/learning":
            entry = reflection_payload(
                agent=str(payload.get("agent", "")),
                task=str(payload.get("task", "")),
                lesson=str(payload.get("lesson", "")),
                evidence=str(payload.get("evidence", "")),
                reuse_when=str(payload.get("reuse_when", "")),
            )
            append_jsonl(BRAIN_DIR / "learnings.jsonl", entry)
            self._json({"written": "learnings.jsonl", "entry": entry})
            return

        if self.path == "/api/reflect/rejection":
            entry = rejection_payload(
                idea=str(payload.get("idea", "")),
                verdict="REJECT",
                category=str(payload.get("category", "")),
                reason=str(payload.get("reason", "")),
                source=str(payload.get("source", "command_center")),
            )
            append_jsonl(BRAIN_DIR / "rejected_ideas.jsonl", entry)
            self._json({"written": "rejected_ideas.jsonl", "entry": entry})
            return

        self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _serve_static(self) -> None:
        target = "index.html" if self.path in {"/", ""} else self.path.lstrip("/")
        path = (STATIC_DIR / target).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Monterey Bay AI Lab browser command center.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CommandCenterHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Monterey Bay AI Lab command center: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down command center.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
