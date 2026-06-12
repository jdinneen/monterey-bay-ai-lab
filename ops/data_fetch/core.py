"""Core framework: status enums, staging paths, safety guard, HTTP backoff,
the Adapter base class, and the manifest/coverage/validation artifact writers.

Design goals (from the inventory of existing fetchers):
  * Generic orchestration here; source-specific weirdness stays in adapters.
  * All new/experimental output lands under data/external_{raw,curated}/<source>/
    and reports/data_fetch/<source>/ — NEVER inside a trusted production path.
  * Every fetch is chunked + resumable: each chunk checkpoints to its own raw
    parquet, so a re-run does only the outstanding work.
  * Every successful source emits manifest.json + coverage.json + validation.json
    + a source note, with a sha256 checksum on the curated parquet.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

# ── repo geography ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Trusted production roots that experimental fetches must NEVER write into.
TRUSTED_ROOTS = [
    PROJECT_ROOT / "bacteria_results",
    PROJECT_ROOT / "mbal_history",
    PROJECT_ROOT / "lakehouse",
    PROJECT_ROOT / "mbal_pipeline" / "curated_history",
]

_EXTERNAL_RAW = PROJECT_ROOT / "data" / "external_raw"
_EXTERNAL_CURATED = PROJECT_ROOT / "data" / "external_curated"
_REPORTS = PROJECT_ROOT / "reports" / "data_fetch"


def is_trusted_path(path: str | Path) -> bool:
    """True if `path` is inside a trusted production root (read-only for us)."""
    p = Path(path).resolve()
    for root in TRUSTED_ROOTS:
        try:
            p.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _guard_write(path: str | Path) -> Path:
    """Raise if a write would land in a trusted production path. Returns Path."""
    p = Path(path)
    if is_trusted_path(p):
        raise PermissionError(
            f"refusing to write trusted production path: {p}\n"
            "experimental fetches must use data/external_* or reports/data_fetch/*"
        )
    return p


def external_raw_dir(source: str) -> Path:
    return _EXTERNAL_RAW / source


def external_curated_dir(source: str) -> Path:
    return _EXTERNAL_CURATED / source


def report_dir(source: str) -> Path:
    return _REPORTS / source


def _rel(path: Path) -> str:
    """Repo-relative string when possible, else the absolute path (test staging
    can live outside PROJECT_ROOT)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class Status:
    """Source lifecycle classifications used across reports + the status matrix."""

    READY_FOR_MODELING = "READY_FOR_MODELING"
    FETCHED_NEEDS_REVIEW = "FETCHED_NEEDS_REVIEW"
    IMPLEMENTED_NOT_FETCHED = "IMPLEMENTED_NOT_FETCHED"
    FAILED = "FAILED"
    NOT_STARTED = "NOT_STARTED"

    ALL = (
        READY_FOR_MODELING,
        FETCHED_NEEDS_REVIEW,
        IMPLEMENTED_NOT_FETCHED,
        FAILED,
        NOT_STARTED,
    )


# ── HTTP with retry/backoff (idiom lifted from chlorophyll + MUR fetchers) ────
def get_with_backoff(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 4,
    base_sleep: float = 1.5,
    headers: Optional[dict] = None,
    session: Any = None,
):
    """GET with bounded exponential backoff; honors Retry-After on 429/503.

    Returns the response object (requests.Response if `requests` available, else a
    small shim exposing .status_code / .text / .content). Raises on final failure.
    """
    try:
        import requests  # noqa: PLC0415

        sess = session or requests
        last = None
        for attempt in range(retries):
            try:
                resp = sess.get(url, timeout=timeout, headers=headers or {})
                if resp.status_code in (429, 503):
                    ra = resp.headers.get("Retry-After")
                    wait = float(ra) if (ra and ra.isdigit()) else base_sleep * (2 ** attempt)
                    time.sleep(min(wait, 120))
                    last = RuntimeError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(min(base_sleep * (2 ** attempt), 120))
        raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")
    except ImportError:
        return _get_with_backoff_urllib(
            url, timeout=timeout, retries=retries, base_sleep=base_sleep, headers=headers
        )


@dataclass
class _UrllibResp:
    status_code: int
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _get_with_backoff_urllib(url, *, timeout, retries, base_sleep, headers):
    import urllib.error
    import urllib.request

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "mbal-datafetch/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _UrllibResp(r.status, r.read())
        except urllib.error.HTTPError as exc:
            last = exc
            ra = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(ra) if (ra and str(ra).isdigit()) else base_sleep * (2 ** attempt)
            time.sleep(min(wait, 120))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(base_sleep * (2 ** attempt), 120))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


# ── checksum ──────────────────────────────────────────────────────────────────
def _to_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """Write parquet; if a mixed-type object column breaks Arrow, coerce object
    (non-datetime) columns to pandas nullable string and retry once."""
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        d = df.copy()
        for c in d.columns:
            if d[c].dtype == object:
                d[c] = d[c].astype("string")
        d.to_parquet(path, index=False)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ── result object ─────────────────────────────────────────────────────────────
@dataclass
class AdapterResult:
    source: str
    status: str
    rows: int = 0
    columns: int = 0
    curated_path: Optional[str] = None
    coverage: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "status": self.status,
            "rows": self.rows,
            "columns": self.columns,
            "curated_path": self.curated_path,
            "coverage": self.coverage,
            "validation": self.validation,
            "message": self.message,
        }


# ── base adapter ──────────────────────────────────────────────────────────────
class Adapter:
    """Base class. Subclasses implement `iter_chunks` + `fetch_chunk` for staged
    fetches, OR set `wraps_trusted` to inventory/validate an existing trusted file
    read-only. The base provides resumable orchestration and artifact writers.
    """

    def __init__(self, spec):
        self.spec = spec
        self.source = spec.key

    # paths -------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return external_raw_dir(self.source)

    @property
    def curated_path(self) -> Path:
        if self.spec.wraps_trusted:
            return PROJECT_ROOT / self.spec.wraps_trusted
        return external_curated_dir(self.source) / f"{self.source}.parquet"

    @property
    def reports(self) -> Path:
        return report_dir(self.source)

    # subclass hooks (staged fetch) ------------------------------------------
    def iter_chunks(self, start: Optional[str], end: Optional[str]) -> Iterator[dict]:
        """Yield chunk descriptors, each a dict with at least a 'key' (filesystem-safe)."""
        raise NotImplementedError(f"{self.source}: iter_chunks not implemented")

    def fetch_chunk(self, chunk: dict) -> pd.DataFrame:
        """Fetch ONE chunk; return a DataFrame (may be empty)."""
        raise NotImplementedError(f"{self.source}: fetch_chunk not implemented")

    # discover ----------------------------------------------------------------
    def discover(self, start: Optional[str] = None, end: Optional[str] = None) -> dict:
        """Describe what a fetch would do, without fetching. Safe + read-only."""
        info = {
            "source": self.source,
            "title": self.spec.title,
            "type": self.spec.type,
            "endpoint": self.spec.endpoint,
            "spatial": self.spec.spatial,
            "temporal": self.spec.temporal,
            "rate_limit": self.spec.rate_limit,
            "wraps_trusted": self.spec.wraps_trusted,
            "needs_credentials": self.spec.needs_credentials,
        }
        if self.spec.wraps_trusted:
            p = self.curated_path
            info["mode"] = "wrap_trusted"
            info["trusted_exists"] = p.exists()
            info["trusted_path"] = _rel(p) if p.exists() else str(self.spec.wraps_trusted)
        else:
            info["mode"] = "staged"
            try:
                chunks = list(self.iter_chunks(start, end))
                info["planned_chunks"] = len(chunks)
                info["chunk_sample"] = [c.get("key") for c in chunks[:5]]
            except NotImplementedError as exc:
                info["planned_chunks"] = None
                info["note"] = str(exc)
            done = self._completed_chunk_keys()
            info["checkpointed_chunks"] = len(done)
        return info

    # dry-run -----------------------------------------------------------------
    def dry_run(self, start: Optional[str] = None, end: Optional[str] = None) -> dict:
        """Plan a fetch and write a plan file to reports/. Writes NO curated/raw data."""
        plan = self.discover(start, end)
        plan["dry_run"] = True
        self.reports.mkdir(parents=True, exist_ok=True)
        (self.reports / "dry_run_plan.json").write_text(
            json.dumps(plan, indent=2, default=str), encoding="utf-8")
        return plan

    # checkpoint helpers ------------------------------------------------------
    def _chunk_path(self, key: str) -> Path:
        safe = str(key).replace("/", "_").replace(":", "_")
        return self.raw_dir / f"chunk_{safe}.parquet"

    def _completed_chunk_keys(self) -> set[str]:
        if not self.raw_dir.exists():
            return set()
        out = set()
        for p in self.raw_dir.glob("chunk_*.parquet"):
            if p.stat().st_size > 0:
                out.add(p.stem[len("chunk_"):])
        return out

    # fetch -------------------------------------------------------------------
    def fetch(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        *,
        resume: bool = True,
        limit_chunks: Optional[int] = None,
    ) -> AdapterResult:
        """Run a resumable, chunked, staged fetch and emit all artifacts."""
        if self.spec.needs_credentials and not self.spec.credentials_present():
            return self._artifacts_for_missing_creds()

        if self.spec.wraps_trusted:
            # Read-only: inventory + validate the trusted file, emit artifacts.
            return self._finalize(self.curated_path, read_only=True)

        _guard_write(self.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        done = self._completed_chunk_keys() if resume else set()

        chunks = list(self.iter_chunks(start, end))
        if limit_chunks is not None:
            chunks = chunks[:limit_chunks]
        todo = [c for c in chunks if str(c["key"]).replace("/", "_").replace(":", "_") not in done]

        fetched = 0
        for c in todo:
            key = str(c["key"]).replace("/", "_").replace(":", "_")
            try:
                df = self.fetch_chunk(c)
            except Exception as exc:  # noqa: BLE001 — one bad chunk shouldn't kill the run
                print(f"[{self.source}] chunk {key} failed: {exc}")
                continue
            cp = self._chunk_path(key)
            _guard_write(cp)
            if df is None:
                df = pd.DataFrame()
            _to_parquet_safe(df, cp)  # checkpoint immediately (even if empty marker)
            fetched += 1
            if self.spec.delay_seconds:
                time.sleep(self.spec.delay_seconds)

        curated = self._consolidate()
        return self._finalize(curated, read_only=False, fetched_chunks=fetched, total_chunks=len(chunks))

    def _consolidate(self) -> Path:
        """Concatenate all raw chunk parquets -> dedup -> curated parquet."""
        frames = []
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
                if len(d):
                    frames.append(d)
            except Exception as exc:  # noqa: BLE001
                print(f"[{self.source}] skipping unreadable chunk {p.name}: {exc}")
        out_path = self.curated_path
        _guard_write(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            _to_parquet_safe(pd.DataFrame(columns=self.spec.required_columns), out_path)
            return out_path
        df = pd.concat(frames, ignore_index=True)
        if self.spec.dedup_keys:
            keys = [k for k in self.spec.dedup_keys if k in df.columns]
            if keys:
                # Dedup on a STRING-normalized view of the keys. Chunks fetched at
                # different times can give a key column inconsistent dtypes (e.g. a
                # numeric `Result` read as float in one chunk and object in another);
                # raw drop_duplicates then treats 100.0 and "100.0" as distinct and
                # silently leaves duplicates that only collapse after a parquet round
                # trip. Comparing as strings makes dedup deterministic and complete
                # while preserving the original row values in the output.
                norm = df[keys].astype("string")
                df = df[~norm.duplicated().to_numpy()].reset_index(drop=True)
        if self.spec.date_column and self.spec.date_column in df.columns:
            # Normalize the date column to timezone-aware UTC so EVERY curated source
            # shares one temporal contract. Adapters that parsed without utc=True left a
            # tz-naive column; mixing tz-naive and tz-aware timestamps across sources
            # breaks time joins in the downstream lakehouse/drivers build. Naive values
            # are already UTC-intended, so localize (not convert).
            s = df[self.spec.date_column]
            if pd.api.types.is_datetime64_any_dtype(s):
                if getattr(s.dt, "tz", None) is None:
                    df[self.spec.date_column] = s.dt.tz_localize("UTC")
                else:
                    df[self.spec.date_column] = s.dt.tz_convert("UTC")
            df = df.sort_values(self.spec.date_column)
        _to_parquet_safe(df, out_path)
        return out_path

    # validation --------------------------------------------------------------
    def validate(self, write: bool = True) -> dict:
        """Validate the curated/trusted output against the spec rules."""
        p = self.curated_path
        result: dict[str, Any] = {"source": self.source, "path": str(p), "checks": {}, "passed": False}
        if not p.exists():
            result["checks"]["exists"] = False
            result["error"] = "output file does not exist"
            if write:
                self._write_json("validation.json", result)
            return result
        result["checks"]["exists"] = True
        df = pd.read_parquet(p)
        result["rows"] = int(len(df))
        result["columns"] = int(df.shape[1])
        result["column_names"] = list(df.columns)

        checks = result["checks"]
        checks["non_empty"] = len(df) >= max(1, self.spec.min_rows)
        missing = [c for c in self.spec.required_columns if c not in df.columns]
        checks["required_columns_present"] = (len(missing) == 0)
        result["missing_columns"] = missing

        # date coverage
        dc = self.spec.date_column
        if dc and dc in df.columns and len(df):
            s = pd.to_datetime(df[dc], errors="coerce", utc=True)
            result["date_min"] = str(s.min())
            result["date_max"] = str(s.max())
            checks["has_date_coverage"] = bool(s.notna().any())
        else:
            checks["has_date_coverage"] = (dc is None)

        # null rates
        if len(df):
            null_rates = (df.isna().mean()).round(4).to_dict()
            result["null_rates"] = {k: float(v) for k, v in null_rates.items()}
        # duplicate keys
        if self.spec.dedup_keys and len(df):
            keys = [k for k in self.spec.dedup_keys if k in df.columns]
            if keys:
                result["duplicate_key_count"] = int(df.duplicated(keys).sum())
                checks["no_duplicate_keys"] = (result["duplicate_key_count"] == 0)
        # value bounds
        bound_violations = {}
        for col, (lo, hi) in (self.spec.value_bounds or {}).items():
            if col in df.columns and len(df):
                vals = pd.to_numeric(df[col], errors="coerce")
                bad = int(((vals < lo) | (vals > hi)).sum())
                if bad:
                    bound_violations[col] = bad
        result["bound_violations"] = bound_violations
        checks["within_bounds"] = (len(bound_violations) == 0)

        # hard pass = the gating checks
        hard = ["exists", "non_empty", "required_columns_present", "has_date_coverage", "within_bounds"]
        result["passed"] = all(checks.get(k, False) for k in hard)
        result["bounded_sample"] = bool(self.spec.bounded_sample)
        if write:
            self._write_json("validation.json", result)
        return result

    # artifact writers --------------------------------------------------------
    def _write_json(self, name: str, obj: dict) -> Path:
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / name
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return path

    def _coverage(self, df: pd.DataFrame) -> dict:
        cov: dict[str, Any] = {
            "source": self.source,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }
        dc = self.spec.date_column
        if dc and dc in df.columns and len(df):
            s = pd.to_datetime(df[dc], errors="coerce", utc=True)
            cov["date_min"] = str(s.min())
            cov["date_max"] = str(s.max())
        for sc in self.spec.spatial_columns or []:
            if sc in df.columns and len(df):
                cov[f"distinct_{sc}"] = int(df[sc].nunique())
        return cov

    def _write_manifest(self, curated: Path, df: pd.DataFrame) -> dict:
        chunk_files = sorted(self.raw_dir.glob("chunk_*.parquet")) if self.raw_dir.exists() else []
        manifest = {
            "source": self.source,
            "title": self.spec.title,
            "curated_path": _rel(curated) if curated.exists() else None,
            "curated_sha256": sha256_file(curated) if curated.exists() and not self.spec.wraps_trusted else (
                sha256_file(curated) if curated.exists() else None
            ),
            "size_bytes": curated.stat().st_size if curated.exists() else 0,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "wraps_trusted": bool(self.spec.wraps_trusted),
            "chunk_count": len(chunk_files),
            "chunk_files": [c.name for c in chunk_files],
            "endpoint": self.spec.endpoint,
        }
        self._write_json("manifest.json", manifest)
        return manifest

    def _write_source_note(self, status: str, validation: dict) -> None:
        note = (
            f"# {self.spec.title} ({self.source})\n\n"
            f"- **Type:** {self.spec.type}\n"
            f"- **Endpoint:** {self.spec.endpoint}\n"
            f"- **Mode:** {'wrap_trusted (read-only)' if self.spec.wraps_trusted else 'staged'}\n"
            f"- **Status:** {status}\n"
            f"- **Rows:** {validation.get('rows', 0):,} · **Columns:** {validation.get('columns', 0)}\n"
            f"- **Date range:** {validation.get('date_min', 'n/a')} → {validation.get('date_max', 'n/a')}\n"
            f"- **Curated path:** {validation.get('path')}\n\n"
            f"{self.spec.description}\n\n"
            f"Artifacts: manifest.json · coverage.json · validation.json (this folder).\n"
        )
        self.reports.mkdir(parents=True, exist_ok=True)
        (self.reports / "README.md").write_text(note, encoding="utf-8")

    def _finalize(
        self,
        curated: Path,
        *,
        read_only: bool,
        fetched_chunks: int = 0,
        total_chunks: int = 0,
    ) -> AdapterResult:
        validation = self.validate(write=True)
        df = pd.read_parquet(curated) if curated.exists() else pd.DataFrame()
        coverage = self._coverage(df)
        self._write_json("coverage.json", coverage)
        self._write_manifest(curated, df)

        if not curated.exists() or len(df) == 0:
            status = Status.IMPLEMENTED_NOT_FETCHED if not read_only else Status.FAILED
        elif validation["passed"] and not self.spec.bounded_sample:
            status = Status.READY_FOR_MODELING
        else:
            # validation failed OR it is a deliberately bounded/partial sample
            status = Status.FETCHED_NEEDS_REVIEW
        validation["bounded_sample"] = bool(self.spec.bounded_sample)
        self._write_json("validation.json", validation)
        self._write_source_note(status, validation)

        msg = (
            f"wrapped trusted output ({len(df):,} rows)" if read_only
            else f"fetched {fetched_chunks}/{total_chunks} chunks → {len(df):,} rows"
        )
        return AdapterResult(
            source=self.source,
            status=status,
            rows=int(len(df)),
            columns=int(df.shape[1]),
            curated_path=_rel(curated) if curated.exists() else None,
            coverage=coverage,
            validation=validation,
            message=msg,
        )

    def _artifacts_for_missing_creds(self) -> AdapterResult:
        self.reports.mkdir(parents=True, exist_ok=True)
        info = {
            "source": self.source,
            "status": Status.IMPLEMENTED_NOT_FETCHED,
            "reason": "missing credentials/config",
            "needs": self.spec.credentials_doc,
        }
        self._write_json("validation.json", info)
        self._write_source_note(Status.IMPLEMENTED_NOT_FETCHED, {"rows": 0, "columns": 0, "path": None})
        return AdapterResult(
            source=self.source,
            status=Status.IMPLEMENTED_NOT_FETCHED,
            message=f"missing credentials: {self.spec.credentials_doc}",
        )
