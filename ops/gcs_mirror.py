#!/usr/bin/env python3
"""Mirror the local lakehouse (and an optional bronze/raw dir) to object storage.

DRY-RUN BY DEFAULT. This tool walks the local source trees and prints the planned
object operations and total bytes; it performs NO uploads unless ``--execute`` is
passed explicitly. Even with ``--execute`` it only PUTs objects into an existing
bucket -- it never creates buckets and never spins up compute.

Architecture note: this is a product-neutral object-storage mirror. The destination
is ``gs://<bucket>/<prefix>`` and the upload path prefers the ``google-cloud-storage``
client when importable, otherwise it shells to ``gsutil``. Both are imported/invoked
lazily inside the execute path so this module stays importable (and ``--help``-able)
on a machine without those dependencies and without any network access.

Environment:
  - ``MBAL_GCS_BUCKET``  default destination bucket (overridden by ``--bucket``)
  - ``MBAL_GCP_PROJECT`` GCP project id (never baked into source)

Read-only over the local tree. No mutation of local data, no bucket creation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional


class PlannedObject(NamedTuple):
    """One planned local->object-storage upload."""

    local_path: Path
    object_name: str  # key under the bucket, e.g. "<prefix>/lakehouse/gold/..."
    size_bytes: int


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield every regular file under ``root`` (recursively), sorted for stability."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _join_object_name(prefix: str, parts: Iterable[str]) -> str:
    """Join an object key from ``prefix`` and path parts using forward slashes.

    Object-storage keys always use ``/`` regardless of the local OS separator, so we
    never rely on ``os.path`` / ``PurePath`` joining here.
    """
    segments: List[str] = []
    if prefix:
        segments.append(prefix.strip("/"))
    segments.extend(p for p in parts if p)
    return "/".join(s for s in segments if s)


def plan_mirror(
    lakehouse_dir: Path,
    prefix: str,
    bronze_dir: Optional[Path] = None,
) -> List[PlannedObject]:
    """Build the list of planned uploads for the configured source trees.

    Each source tree is mirrored under a stable top-level namespace within the
    prefix (``lakehouse/...`` and, when supplied, ``bronze/...``) so the destination
    layout is self-describing.
    """
    planned: List[PlannedObject] = []

    sources = [("lakehouse", lakehouse_dir)]
    if bronze_dir is not None:
        sources.append(("bronze", bronze_dir))

    for namespace, root in sources:
        root = root.resolve()
        for path in _iter_files(root):
            rel = path.relative_to(root)
            object_name = _join_object_name(prefix, [namespace, *rel.parts])
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            planned.append(PlannedObject(path, object_name, size))

    return planned


def _human_bytes(n: int) -> str:
    """Render a byte count in human-friendly units."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{n} B"


def print_plan(planned: List[PlannedObject], bucket: str, execute: bool) -> int:
    """Print the planned operations and return the total byte count."""
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"[gcs_mirror] mode={mode} destination=gs://{bucket}")
    total = 0
    for obj in planned:
        total += obj.size_bytes
        print(
            f"  PUT gs://{bucket}/{obj.object_name}  "
            f"({_human_bytes(obj.size_bytes)})  <- {obj.local_path}"
        )
    print(
        f"[gcs_mirror] planned objects={len(planned)} "
        f"total_bytes={total} ({_human_bytes(total)})"
    )
    if not execute:
        print("[gcs_mirror] DRY-RUN: no uploads performed. Re-run with --execute to upload.")
    return total


def _credentials_present() -> bool:
    """Best-effort check that some GCP credential source is configured.

    We do not validate the credentials against the network here; we only confirm a
    credential source exists so ``--execute`` fails fast with a clear message instead
    of an obscure auth error deep in the client library.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        cred_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        return cred_path.exists()
    # Application Default Credentials well-known file (gcloud auth application-default login).
    if os.name == "nt":
        adc = Path(os.environ.get("APPDATA", "")) / "gcloud" / "application_default_credentials.json"
    else:
        adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc.exists()


def _execute_with_client(
    planned: List[PlannedObject], bucket: str, project: Optional[str]
) -> bool:
    """Upload via google-cloud-storage if importable. Returns True if it ran."""
    try:
        from google.cloud import storage  # type: ignore  # lazy: optional dep
    except Exception:  # pragma: no cover - exercised only when dep is present
        return False

    client = storage.Client(project=project) if project else storage.Client()
    bucket_obj = client.bucket(bucket)  # reference only; never .create()
    for obj in planned:
        blob = bucket_obj.blob(obj.object_name)
        print(f"[gcs_mirror] uploading -> gs://{bucket}/{obj.object_name}")
        blob.upload_from_filename(str(obj.local_path))
    print(f"[gcs_mirror] client upload complete: {len(planned)} objects.")
    return True


def _execute_with_gsutil(planned: List[PlannedObject], bucket: str) -> bool:
    """Upload via the gsutil CLI. Returns True if gsutil was found and invoked."""
    import shutil
    import subprocess

    gsutil = shutil.which("gsutil")
    if not gsutil:
        return False

    for obj in planned:
        dest = f"gs://{bucket}/{obj.object_name}"
        print(f"[gcs_mirror] gsutil cp -> {dest}")
        # No -m / no bucket creation; one object at a time, fail loud.
        subprocess.run([gsutil, "cp", str(obj.local_path), dest], check=True)
    print(f"[gcs_mirror] gsutil upload complete: {len(planned)} objects.")
    return True


def execute_mirror(
    planned: List[PlannedObject], bucket: str, project: Optional[str]
) -> int:
    """Perform the uploads. Prefer the client library, fall back to gsutil.

    Returns a process exit code (0 on success).
    """
    if _execute_with_client(planned, bucket, project):
        return 0
    print(
        "[gcs_mirror] google-cloud-storage not importable; falling back to gsutil CLI.",
        file=sys.stderr,
    )
    if _execute_with_gsutil(planned, bucket):
        return 0
    print(
        "[gcs_mirror] ERROR: neither google-cloud-storage nor gsutil is available. "
        "Install google-cloud-storage or the Cloud SDK to upload.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror the local lakehouse (and an optional bronze/raw dir) to "
            "gs://<bucket>/<prefix>. DRY-RUN by default; --execute to upload."
        )
    )
    parser.add_argument(
        "--lakehouse-dir",
        default=os.environ.get("MBAL_LAKEHOUSE_DIR", "lakehouse"),
        help="Local lakehouse directory to mirror (default: $MBAL_LAKEHOUSE_DIR or ./lakehouse).",
    )
    parser.add_argument(
        "--bronze-dir",
        default=None,
        help="Optional local bronze/raw directory to also mirror under bronze/.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MBAL_GCS_BUCKET"),
        help="Destination bucket name (default: $MBAL_GCS_BUCKET). Bucket must already exist.",
    )
    parser.add_argument(
        "--prefix",
        default="mbal",
        help="Object key prefix under the bucket (default: mbal).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually upload. Without this flag the tool only prints the plan.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Required second gate for real uploads. --execute alone is NOT enough: "
            "you must also pass --confirm. This prevents ambient Application Default "
            "Credentials from ever turning a stray --execute into a network PUT."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    lakehouse_dir = Path(args.lakehouse_dir)
    bronze_dir = Path(args.bronze_dir) if args.bronze_dir else None
    project = os.environ.get("MBAL_GCP_PROJECT")

    if not lakehouse_dir.exists():
        print(
            f"[gcs_mirror] ERROR: lakehouse dir not found: {lakehouse_dir}",
            file=sys.stderr,
        )
        return 2
    if bronze_dir is not None and not bronze_dir.exists():
        print(
            f"[gcs_mirror] ERROR: bronze dir not found: {bronze_dir}",
            file=sys.stderr,
        )
        return 2

    # --execute guard: fail fast with a clear message rather than an obscure error.
    if args.execute:
        if not args.confirm:
            print(
                "[gcs_mirror] ERROR: --execute also requires --confirm. This is a "
                "deliberate two-key gate so that ambient credentials cannot turn a "
                "stray --execute into a real network upload. No upload performed.",
                file=sys.stderr,
            )
            return 2
        if not args.bucket:
            print(
                "[gcs_mirror] ERROR: --execute requires a bucket. Pass --bucket or set "
                "MBAL_GCS_BUCKET.",
                file=sys.stderr,
            )
            return 2
        if not _credentials_present():
            print(
                "[gcs_mirror] ERROR: --execute requires GCP credentials. Set "
                "GOOGLE_APPLICATION_CREDENTIALS to a key file, or run "
                "'gcloud auth application-default login' first.",
                file=sys.stderr,
            )
            return 2

    bucket_for_plan = args.bucket or "<bucket-unset>"
    planned = plan_mirror(lakehouse_dir, args.prefix, bronze_dir)
    print_plan(planned, bucket_for_plan, args.execute)

    if not planned:
        print("[gcs_mirror] nothing to mirror; source trees are empty.")
        return 0

    if not args.execute:
        return 0

    return execute_mirror(planned, args.bucket, project)


if __name__ == "__main__":
    raise SystemExit(main())
