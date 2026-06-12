#!/usr/bin/env python3
"""
Lightweight Traffic Controller for Monterey Bay AI Lab.
Coordination without bloat.

Functions as a 'Central Agent' for resource management:
1. Prevents redundant agent tasks (via ops/agent_lock.py).
2. Prevents GPU oversubscription (via ops/gpu_admission.py).
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path so we can import ops
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ops.agent_lock import claim, release, status as lock_status, check as lock_check
from ops.gpu_admission import check_gpu_admission, query_gpu_state

def main():
    parser = argparse.ArgumentParser(description="MBAL Traffic Controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'request' command: Check if it's safe to run a task
    p_req = subparsers.add_parser("request", help="Request permission to run a task")
    p_req.add_argument("--task", required=True, help="Unique name for the task/job")
    p_req.add_argument("--agent", required=True, help="Agent ID (e.g., gemini, claude)")
    p_req.add_argument("--gpu-mib", type=int, default=0, help="Estimated GPU VRAM requirement in MiB")
    p_req.add_argument("--reserve-mib", type=int, default=4096, help="GPU VRAM reserve headroom")
    p_req.add_argument("--ttl", type=int, default=1800, help="Lock TTL in seconds")

    # 'release' command: Finish a task
    p_rel = subparsers.add_parser("release", help="Release a task lock")
    p_rel.add_argument("--task", required=True)
    p_rel.add_argument("--agent", required=True)

    # 'status' command: Show what's happening
    p_stat = subparsers.add_parser("status", help="Show current resource status")

    args = parser.parse_args()

    if args.command == "request":
        # 1. Check Task Lock (Prevent duplication)
        # We use the task name as the 'path' to lock
        task_id = f"TASK:{args.task}"
        lock_code = claim([task_id], args.agent, f"Running {args.task}", args.ttl)
        if lock_code != 0:
            print(f"FAILED: Task '{args.task}' is already being handled by another agent.", file=sys.stderr)
            sys.exit(2)

        # 2. Check GPU (Prevent oversubscription)
        if args.gpu_mib > 0:
            denial = check_gpu_admission(
                label=args.task,
                request_mib=args.gpu_mib,
                reserve_mib=args.reserve_mib
            )
            if denial:
                # If GPU check fails, we must release the task lock we just grabbed
                release([task_id], args.agent, force=False)
                print(denial, file=sys.stderr)
                sys.exit(3)
        
        print(f"ADMITTED: Task '{args.task}' granted to agent '{args.agent}'.")
        sys.exit(0)

    elif args.command == "release":
        task_id = f"TASK:{args.task}"
        sys.exit(release([task_id], args.agent, force=False))

    elif args.command == "status":
        print("--- ACTIVE TASK LOCKS ---")
        lock_status(show_all=False)
        
        print("\n--- GPU STATUS ---")
        gpu = query_gpu_state()
        if gpu:
            print(f"GPU: {gpu.name}")
            print(f"VRAM: {gpu.used_mib}/{gpu.total_mib} MiB used ({gpu.used_pct*100:.1f}%)")
            print(f"Free: {gpu.free_mib} MiB")
        else:
            print("GPU info unavailable (nvidia-smi not found).")

if __name__ == "__main__":
    main()

