"""Production-oriented local experiment tracking for MBAL model runs.

The tracker intentionally has no required third-party dependencies. Optional
packages such as torch, xgboost, pandas, and numpy are detected when present.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from typing import Any, Iterable


DEFAULT_TRACKING_DIR = Path("mbal_experiments") / "runs"
DEFAULT_INDEX_PATH = Path("mbal_experiments") / "experiments.jsonl"
DEFAULT_HASH_LIMIT_BYTES = 512 * 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def safe_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "run"


def json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        with contextlib.suppress(Exception):
            return value.item()
    if hasattr(value, "tolist"):
        with contextlib.suppress(Exception):
            return value.tolist()
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True, default=json_default)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=json_default)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_command(args: list[str], cwd: Path, timeout: int = 10) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return 1, str(exc)
    output = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if stderr:
        output = f"{output}\n{stderr}".strip()
    return completed.returncode, output


def collect_git_state(cwd: Path) -> dict[str, Any]:
    rc, root = run_command(["git", "rev-parse", "--show-toplevel"], cwd)
    if rc != 0:
        return {"available": False, "reason": root or "not a git repository"}

    root_path = Path(root.splitlines()[0])
    _, commit = run_command(["git", "rev-parse", "HEAD"], root_path)
    _, branch = run_command(["git", "branch", "--show-current"], root_path)
    _, status = run_command(["git", "status", "--short"], root_path)
    _, diff_stat = run_command(["git", "diff", "--stat"], root_path)
    _, untracked = run_command(["git", "ls-files", "--others", "--exclude-standard"], root_path)
    return {
        "available": True,
        "root": str(root_path),
        "commit": commit.splitlines()[0] if commit else None,
        "branch": branch or None,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
        "diff_stat": diff_stat.splitlines() if diff_stat else [],
        "untracked_files": untracked.splitlines() if untracked else [],
    }


def package_version(name: str) -> str | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    return None


def collect_environment() -> dict[str, Any]:
    packages = [
        "python",
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "duckdb",
        "scikit-learn",
        "scipy",
        "xarray",
        "netCDF4",
        "dask",
        "mlflow",
    ]
    versions = {"python": platform.python_version()}
    versions.update({name: package_version(name) for name in packages if name != "python"})
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "package_versions": versions,
    }


def collect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"captured_at": utc_now(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}

    with contextlib.suppress(Exception):
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        }
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                torch_info["devices"].append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                        "total_memory_bytes": getattr(props, "total_memory", None),
                    }
                )
            with contextlib.suppress(Exception):
                torch_info["arch_list"] = torch.cuda.get_arch_list()
        info["torch"] = torch_info

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        rc, output = run_command(
            [
                nvidia_smi,
                "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            Path.cwd(),
        )
        info["nvidia_smi"] = {
            "available": rc == 0,
            "raw": output.splitlines() if output else [],
        }
    else:
        info["nvidia_smi"] = {"available": False, "reason": "nvidia-smi not found on PATH"}

    return info


def sha256_file(path: Path, max_bytes: int = DEFAULT_HASH_LIMIT_BYTES) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    truncated = False
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if total + len(chunk) > max_bytes:
                digest.update(chunk[: max_bytes - total])
                truncated = True
                total = max_bytes
                break
            digest.update(chunk)
            total += len(chunk)
    return {
        "sha256": digest.hexdigest(),
        "hashed_bytes": total,
        "hash_truncated": truncated,
        "hash_limit_bytes": max_bytes,
    }


def fingerprint_path(path: Path, max_hash_bytes: int = DEFAULT_HASH_LIMIT_BYTES) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    payload = {
        "path": str(resolved),
        "name": resolved.name,
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).replace(microsecond=0).isoformat(),
    }
    payload.update(sha256_file(resolved, max_hash_bytes))
    return payload


def collect_dataset_fingerprints(
    paths: Iterable[str | Path],
    max_hash_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[Path] = set()

    for raw_path in paths:
        candidate = Path(raw_path)
        matches = list(Path.cwd().glob(str(candidate))) if any(ch in str(candidate) for ch in "*?[]") else [candidate]
        if not matches:
            missing.append(str(raw_path))
            continue
        for match in matches:
            if not match.exists():
                missing.append(str(match))
                continue
            expanded = sorted(match.rglob("*")) if match.is_dir() else [match]
            for file_path in expanded:
                if not file_path.is_file():
                    continue
                resolved = file_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(fingerprint_path(resolved, max_hash_bytes=max_hash_bytes))

    manifest_digest = hashlib.sha256()
    for item in sorted(files, key=lambda row: row["path"]):
        manifest_digest.update(item["path"].encode("utf-8"))
        manifest_digest.update(str(item["size_bytes"]).encode("utf-8"))
        manifest_digest.update(item["sha256"].encode("utf-8"))

    return {
        "captured_at": utc_now(),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "manifest_sha256": manifest_digest.hexdigest(),
        "files": files,
        "missing": missing,
    }


@dataclasses.dataclass
class RunRecord:
    run_id: str
    name: str
    status: str
    started_at: str
    ended_at: str | None
    duration_seconds: float | None
    run_dir: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    tags: dict[str, str]
    notes: str | None = None
    error: str | None = None


class ExperimentTracker:
    def __init__(
        self,
        tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        workspace: str | Path | None = None,
    ) -> None:
        self.tracking_dir = Path(tracking_dir)
        self.index_path = Path(index_path)
        self.workspace = Path(workspace or Path.cwd())
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        name: str,
        *,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        notes: str | None = None,
        dataset_paths: Iterable[str | Path] = (),
        max_hash_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
    ) -> "ActiveRun":
        return ActiveRun(
            tracker=self,
            name=name,
            params=params or {},
            tags=tags or {},
            notes=notes,
            dataset_paths=list(dataset_paths),
            max_hash_bytes=max_hash_bytes,
        )

    def record_run(
        self,
        *,
        name: str,
        metrics: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        notes: str | None = None,
        dataset_paths: Iterable[str | Path] = (),
        status: str = "completed",
        max_hash_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
    ) -> RunRecord:
        run = self.start_run(
            name,
            params=params,
            tags=tags,
            notes=notes,
            dataset_paths=dataset_paths,
            max_hash_bytes=max_hash_bytes,
        )
        run.__enter__()
        if metrics:
            run.log_metrics(metrics)
        return run.finish(status=status)


class ActiveRun:
    def __init__(
        self,
        *,
        tracker: ExperimentTracker,
        name: str,
        params: dict[str, Any],
        tags: dict[str, str],
        notes: str | None,
        dataset_paths: list[str | Path],
        max_hash_bytes: int,
    ) -> None:
        self.tracker = tracker
        self.name = name
        self.params = params
        self.tags = tags
        self.notes = notes
        self.dataset_paths = dataset_paths
        self.max_hash_bytes = max_hash_bytes
        self.metrics: dict[str, Any] = {}
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.run_id = f"{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}-{safe_slug(name)}-{uuid.uuid4().hex[:8]}"
        self.run_dir = self.tracker.tracking_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._finished = False

    def __enter__(self) -> "ActiveRun":
        self._write_static_artifacts()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if self._finished:
            return
        if exc is None:
            self.finish(status="completed")
        else:
            self.finish(status="failed", error="".join(traceback.format_exception(exc_type, exc, tb)))

    def log_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        atomic_write_json(self.run_dir / "metrics.json", self.metrics)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)
        atomic_write_json(self.run_dir / "metrics.json", self.metrics)

    def _write_static_artifacts(self) -> None:
        atomic_write_json(self.run_dir / "params.json", self.params)
        atomic_write_json(self.run_dir / "tags.json", self.tags)
        atomic_write_json(self.run_dir / "environment.json", collect_environment())
        atomic_write_json(self.run_dir / "gpu.json", collect_gpu_info())
        atomic_write_json(self.run_dir / "git.json", collect_git_state(self.tracker.workspace))
        atomic_write_json(
            self.run_dir / "datasets.json",
            collect_dataset_fingerprints(self.dataset_paths, max_hash_bytes=self.max_hash_bytes),
        )
        if self.notes:
            (self.run_dir / "notes.txt").write_text(self.notes + "\n", encoding="utf-8")
        atomic_write_json(self.run_dir / "metrics.json", self.metrics)

    def finish(self, *, status: str = "completed", error: str | None = None) -> RunRecord:
        self._finished = True
        ended_at = utc_now()
        duration = round(time.monotonic() - self.started_monotonic, 3)
        record = RunRecord(
            run_id=self.run_id,
            name=self.name,
            status=status,
            started_at=self.started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            run_dir=str(self.run_dir.resolve()),
            metrics=self.metrics,
            params=self.params,
            tags=self.tags,
            notes=self.notes,
            error=error,
        )
        payload = dataclasses.asdict(record)
        atomic_write_json(self.run_dir / "run.json", payload)
        append_jsonl(self.tracker.index_path, payload)
        if error:
            (self.run_dir / "error.txt").write_text(error, encoding="utf-8")
        return record


def parse_key_values(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in {item!r}")
        parsed[key] = parse_scalar(value.strip())
    return parsed


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


def create_run(
    name: str,
    *,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    notes: str | None = None,
    dataset_paths: Iterable[str | Path] = (),
) -> RunRecord:
    return ExperimentTracker().record_run(
        name=name,
        metrics=metrics,
        params=params,
        tags=tags,
        notes=notes,
        dataset_paths=dataset_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Monterey Bay AI Lab model experiment metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record a completed run from CLI key-value inputs.")
    record.add_argument("--name", required=True, help="Human-readable run name.")
    record.add_argument("--metric", action="append", default=[], help="Metric KEY=VALUE. Repeatable.")
    record.add_argument("--param", action="append", default=[], help="Parameter KEY=VALUE. Repeatable.")
    record.add_argument("--tag", action="append", default=[], help="Tag KEY=VALUE. Repeatable.")
    record.add_argument("--dataset", action="append", default=[], help="Dataset file, directory, or glob. Repeatable.")
    record.add_argument("--notes", default=None, help="Optional run notes.")
    record.add_argument("--status", default="completed", choices=["completed", "failed", "running", "aborted"])
    record.add_argument("--tracking-dir", default=str(DEFAULT_TRACKING_DIR))
    record.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    record.add_argument("--max-hash-bytes", type=int, default=DEFAULT_HASH_LIMIT_BYTES)

    smoke = subparsers.add_parser("smoke", help="Write a small smoke-test run.")
    smoke.add_argument("--tracking-dir", default=str(DEFAULT_TRACKING_DIR))
    smoke.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tracker = ExperimentTracker(tracking_dir=args.tracking_dir, index_path=args.index_path)
    if args.command == "record":
        record = tracker.record_run(
            name=args.name,
            metrics=parse_key_values(args.metric),
            params=parse_key_values(args.param),
            tags={str(k): str(v) for k, v in parse_key_values(args.tag).items()},
            notes=args.notes,
            dataset_paths=args.dataset,
            status=args.status,
            max_hash_bytes=args.max_hash_bytes,
        )
    elif args.command == "smoke":
        record = tracker.record_run(
            name="smoke-example",
            metrics={"rmse": 0.123, "r2": 0.987},
            params={"model": "xgboost", "horizon_hours": 24},
            tags={"station": "M1", "target": "temp_d100p0"},
            notes="Synthetic smoke record created by mbal_experiments.",
            dataset_paths=[],
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(dataclasses.asdict(record), indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


