#!/usr/bin/env python3
"""
The Safe Launcher for Monterey Bay AI Lab.
The "Traffic Gate" that enforces resource coordination.

Usage:
  python ops/run_safe.py --task <NAME> --agent <ID> --gpu-mib <SIZE> [--brain-preflight off|warn|enforce] -- <COMMAND...>
"""

import argparse
import subprocess
import sys
import os
import time
import json
import threading
import logging
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ops.agent_brain_core import GateInput, parse_bool, query_brain, run_value_gate

logging.basicConfig(level=logging.INFO, format='[*] %(message)s')


def build_preflight_gate(args) -> GateInput | None:
    proposal = {}
    if args.preflight:
        try:
            proposal = json.loads(args.preflight.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"FAILED: preflight file not found: {args.preflight}", file=sys.stderr)
            return None
        except json.JSONDecodeError as exc:
            print(f"FAILED: preflight file is not valid JSON: {args.preflight}: {exc}", file=sys.stderr)
            return None

    idea = args.preflight_idea or str(proposal.get("idea", ""))
    if not idea.strip():
        return None

    return GateInput(
        idea=idea,
        agent=str(proposal.get("agent") or args.agent),
        task=str(proposal.get("task") or args.task),
        baseline=args.preflight_baseline or str(proposal.get("baseline", "")),
        test=args.preflight_test or str(proposal.get("test", "")),
        reversible=parse_bool(args.preflight_reversible or proposal.get("reversible", False)),
        valuable_now=parse_bool(args.preflight_valuable_now or proposal.get("valuable_now", False)),
        genuinely_new=parse_bool(args.preflight_genuinely_new or proposal.get("genuinely_new", False)),
        stakeholder_question=args.preflight_stakeholder_question or str(proposal.get("stakeholder_question", "")),
        correctness_fix=parse_bool(args.preflight_correctness_fix or proposal.get("correctness_fix", False)),
        kill_category=args.preflight_kill_category or str(proposal.get("kill_category", "bloat")),
    )


def run_agent_preflight(args) -> int:
    """Validate an optional Agent Brain preflight ticket.

    This is intentionally opt-in. It gives heavy/long-running jobs a way to prove
    they passed the value gate without making every status command or tiny test run
    pay the ceremony cost.
    """
    gate = build_preflight_gate(args)
    if gate is None:
        print("Agent Brain preflight skipped: no preflight idea was provided.", file=sys.stderr, flush=True)
        return 2 if args.brain_preflight == "enforce" else 0

    result = run_value_gate(gate)
    print(result.summary, flush=True)
    hits = (
        query_brain(gate.idea, brain_dir=args.preflight_brain_dir, limit=5)
        if args.preflight_brain_dir
        else query_brain(gate.idea, limit=5)
    )
    if hits:
        print("Relevant brain hits:", flush=True)
        for hit in hits:
            print(f"- {hit['path']}:{hit['line']} {hit['text'][:180]}", flush=True)
    if result.verdict != "DO_NOW":
        if args.brain_preflight == "warn":
            print("WARNING: Agent Brain preflight rejected this task; continuing because mode=warn.", file=sys.stderr, flush=True)
            return 0
        print("FAILED: Agent Brain preflight rejected this task before traffic admission.", file=sys.stderr)
        return 2
    return 0

class ObservationalWatchdog:
    """
    A purely observational monitor. It looks for anomalies in metrics.jsonl
    and kills the process to protect the GPU, but it NEVER modifies code.
    It simply alerts the human scientist to investigate the physics.
    """
    def __init__(self, metrics_path, stop_event):
        self.metrics_path = Path(metrics_path)
        self.stop_event = stop_event
        self.anomaly_detected = False
        self.anomaly_info = None

    def monitor(self):
        logging.info(f"Observational Watchdog started for {self.metrics_path}")
        
        while not self.metrics_path.exists() and not self.stop_event.is_set():
            time.sleep(1)
            
        if self.stop_event.is_set():
            return

        baseline_throughput = None
        baseline_loss = None

        try:
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                while not self.stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(1)
                        continue

                    try:
                        record = json.loads(line.strip())
                        if not isinstance(record, dict): continue
                    except (json.JSONDecodeError, ValueError):
                        continue

                    # 1. Safety Shutdown
                    if record.get('event') == 'safety_shutdown':
                        self.anomaly_detected = True
                        self.anomaly_info = f"Safety Shutdown: {record.get('reasons')}"
                        break

                    # 2. NaN / Loss Explosion
                    loss = record.get('loss')
                    step = record.get('step', 0)
                    if loss is not None and isinstance(loss, (int, float)):
                        if str(loss).lower() == 'nan' or record.get('nan_skipped'):
                            self.anomaly_detected = True
                            self.anomaly_info = "Gradient NaN/Divergence"
                            break
                        
                        if baseline_loss is None:
                            baseline_loss = loss
                        elif loss > baseline_loss * 5.0 and step > 100:
                            self.anomaly_detected = True
                            self.anomaly_info = f"Loss Explosion ({baseline_loss:.2f} -> {loss:.2f})"
                            break
                        else:
                            baseline_loss = 0.9 * baseline_loss + 0.1 * loss

                    # 3. Throughput Collapse
                    speed = record.get('speed_bps') or record.get('speed')
                    if speed is not None and isinstance(speed, (int, float)) and step > 500:
                        if baseline_throughput is None:
                            baseline_throughput = speed
                        elif speed < (baseline_throughput * 0.2):
                            self.anomaly_detected = True
                            self.anomaly_info = f"Throughput Collapse ({baseline_throughput:.1f} -> {speed:.1f} b/s)"
                            break
                        else:
                            baseline_throughput = 0.95 * baseline_throughput + 0.05 * speed
        except Exception as e:
            logging.error(f"Watchdog monitor loop failed: {e}")

        if self.anomaly_detected:
            logging.error(f"WATCHDOG ALERT: {self.anomaly_info}")
            logging.error("HALTING EXECUTION FOR HUMAN SCIENTIST REVIEW.")

def kill_process_tree(proc):
    if not proc: return
    logging.info(f"Sending SIGTERM to process {proc.pid}...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        logging.warning(f"Process {proc.pid} ignored SIGTERM. Sending SIGKILL...")
        proc.kill()
        proc.wait()

def main():
    if "--" not in sys.argv:
        print("Usage: python ops/run_safe.py --task <NAME> --agent <ID> --gpu-mib <SIZE> [--watchdog <METRICS_PATH>] [--brain-preflight off|warn|enforce] -- <COMMAND...>")
        sys.exit(1)
    
    divider_idx = sys.argv.index("--")
    launcher_args = sys.argv[1:divider_idx]
    cmd_to_run = sys.argv[divider_idx + 1:]

    parser = argparse.ArgumentParser(description="Safe Task Launcher")
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--gpu-mib", type=int, default=0)
    parser.add_argument("--watchdog", help="Path to metrics.jsonl to monitor for anomalies (Observational only)")
    parser.add_argument("--brain-preflight", choices=["off", "warn", "enforce"], default="off", help="Optional Agent Brain value-gate mode")
    parser.add_argument("--preflight", type=Path, help="Optional Agent Brain proposal JSON")
    parser.add_argument("--preflight-brain-dir", type=Path, default=None, help="Optional alternate Agent Brain directory")
    parser.add_argument("--preflight-idea", default="", help="Proposed work to value-gate")
    parser.add_argument("--preflight-baseline", default="", help="Baseline or metric this work improves")
    parser.add_argument("--preflight-stakeholder-question", default="", help="Stakeholder question this work answers")
    parser.add_argument("--preflight-test", default="", help="Test/evidence path for the proposed work")
    parser.add_argument("--preflight-reversible", default="")
    parser.add_argument("--preflight-valuable-now", default="")
    parser.add_argument("--preflight-genuinely-new", default="")
    parser.add_argument("--preflight-correctness-fix", default="")
    parser.add_argument("--preflight-kill-category", default="")
    args = parser.parse_args(launcher_args)

    effective_preflight = "enforce" if args.preflight and args.brain_preflight == "off" else args.brain_preflight
    args.brain_preflight = effective_preflight
    if args.brain_preflight != "off":
        preflight_code = run_agent_preflight(args)
        if preflight_code != 0:
            sys.exit(preflight_code)

    print(f"[*] Requesting traffic clearance for '{args.task}'...")
    req_cmd = [sys.executable, "ops/traffic_controller.py", "request", "--task", args.task, "--agent", args.agent, "--gpu-mib", str(args.gpu_mib)]
    if subprocess.run(req_cmd).returncode != 0:
        sys.exit(1)

    stop_event = threading.Event()
    watchdog = None
    if args.watchdog:
        watchdog = ObservationalWatchdog(args.watchdog, stop_event)
        w_thread = threading.Thread(target=watchdog.monitor, daemon=True)
        w_thread.start()

    print(f"[*] ADMITTED. Executing: {' '.join(cmd_to_run)}")
    main_proc = None
    try:
        main_proc = subprocess.Popen(cmd_to_run)
        
        while main_proc.poll() is None:
            if watchdog and watchdog.anomaly_detected:
                print(f"[*] ANOMALY DETECTED. Halting process to preserve state for analysis...")
                kill_process_tree(main_proc)
                break
            time.sleep(2)
        
        exit_code = main_proc.wait()
    except KeyboardInterrupt:
        if main_proc: kill_process_tree(main_proc)
        exit_code = 130
    finally:
        stop_event.set()
        print(f"[*] Task finished. Releasing lock...")
        subprocess.run([sys.executable, "ops/traffic_controller.py", "release", "--task", args.task, "--agent", args.agent])

    if watchdog and watchdog.anomaly_detected:
        print("\n" + "="*60)
        print("SCIENTIFIC INTERVENTION REQUIRED")
        print("="*60)
        print(f"The run was halted due to: {watchdog.anomaly_info}")
        print("Please review the metrics and the physical data. DO NOT automatically rewrite the code.")
        sys.exit(5)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

