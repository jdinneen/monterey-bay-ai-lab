#!/usr/bin/env python3
"""Launch the Monterey Bay AI Lab training orchestrator detached from the terminal.

Use this when Windows Terminal / PowerShell is unstable under memory pressure.
The training process keeps running after the launching shell closes, with logs in
``nn_results/train_detached.log`` and ``nn_results/train_detached.err.log``.

Preflight memory guard: refuses to start a GPU training run while a large local
model (e.g. ``qwen-next-deep:128k``, ~51GB weights + KV cache) is resident in
Ollama, because the two together overrun the system commit limit and OOM-crash
the terminals. Override with ``--skip-mem-guard`` or tune ``--mem-guard-gb``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
ORCHESTRATOR = ROOT / "mbal_train.py"
RESULTS = ROOT / "nn_results"
PYTHON = Path(os.environ.get("MBAL_PYTHON", sys.executable))
OLLAMA_PS_URL = "http://localhost:11434/api/ps"


def resident_big_models(threshold_bytes: int) -> list[tuple[str, int]]:
    """Return [(name, bytes)] for Ollama models resident above the threshold.

    Returns [] if Ollama is not running / unreachable (no guard in that case).
    """
    try:
        with urllib.request.urlopen(OLLAMA_PS_URL, timeout=3) as resp:
            data = json.load(resp)
    except Exception:
        return []
    big = []
    for m in data.get("models", []):
        size = m.get("size_vram") or m.get("size") or 0
        if size >= threshold_bytes:
            big.append((m.get("name", "?"), int(size)))
    return big


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start mbal_train.py detached, writing output under nn_results."
    )
    parser.add_argument(
        "--log",
        default=None,
        help="stdout log path",
    )
    parser.add_argument(
        "--err-log",
        default=None,
        help="stderr log path",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="project root to run from; also exported as MBAL_PROJECT_ROOT",
    )
    parser.add_argument(
        "--mem-guard-gb",
        type=float,
        default=40.0,
        help="Refuse to launch if an Ollama model >= this many GB is resident "
        "(default 40; blocks the 51GB-class models, allows the 18GB ones).",
    )
    parser.add_argument(
        "--skip-mem-guard",
        action="store_true",
        help="Bypass the resident-model memory guard.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    root = Path(args.project_root).resolve()
    orchestrator = root / "mbal_train.py"
    results = root / "nn_results"
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else results / "train_detached.log"
    err_log_path = Path(args.err_log) if args.err_log else results / "train_detached.err.log"

    # Preflight: don't stack a GPU training run on top of a big resident model.
    if not args.skip_mem_guard:
        threshold = int(args.mem_guard_gb * (1024 ** 3))
        big = resident_big_models(threshold)
        if big:
            print(
                "REFUSING to launch: large Ollama model(s) resident — running a GPU "
                "training job alongside these risks a commit-limit OOM that crashes "
                "the terminals:",
                file=sys.stderr,
            )
            for name, size in big:
                print(f"  - {name}: {size / 1024 ** 3:.1f} GB resident", file=sys.stderr)
            print(
                "\nFree it first:  ollama stop <name>   "
                "(or wait for OLLAMA_KEEP_ALIVE to unload it),\n"
                "then re-run. To override deliberately: --skip-mem-guard",
                file=sys.stderr,
            )
            return 2

    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = [str(python), str(orchestrator), *forwarded]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    env = os.environ.copy()
    env["MBAL_PROJECT_ROOT"] = str(root)
    with open(log_path, "ab", buffering=0) as stdout, open(err_log_path, "ab", buffering=0) as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=creationflags,
            env=env,
        )

    print(f"started pid={proc.pid}")
    print(f"stdout={log_path}")
    print(f"stderr={err_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
