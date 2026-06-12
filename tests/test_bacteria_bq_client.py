#!/usr/bin/env python3
"""Workstream C: verify the bacteria BigQuery wrappers prefer the python client
(no 100k row cap, native types, no subprocess) and fall back to the bq CLI only
when the client library is missing or its query fails.

No network is touched: the bigquery client and subprocess are monkeypatched to
return known DataFrames.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "bacteria"))

import fetch_statewide_beachwatch as statewide  # noqa: E402
import lovers_point_bacteria_predict as lovers  # noqa: E402


# ---- fakes -----------------------------------------------------------------

class _FakeJob:
    def __init__(self, df: pd.DataFrame, recorder: dict):
        self._df = df
        self._recorder = recorder

    def result(self):
        return self

    def to_dataframe(self, create_bqstorage_client=False):
        self._recorder["bqstorage"] = create_bqstorage_client
        return self._df


class _FakeClient:
    def __init__(self, df: pd.DataFrame, recorder: dict):
        self._df = df
        self._recorder = recorder

    def query(self, sql):
        self._recorder["sql"] = sql
        return _FakeJob(self._df, self._recorder)


def _install_fake_bigquery(monkeypatch, df: pd.DataFrame, recorder: dict):
    """Install a fake google.cloud.bigquery module whose Client returns df."""
    fake_bq = types.ModuleType("google.cloud.bigquery")

    def _client_factory(project=None):
        recorder["project"] = project
        return _FakeClient(df, recorder)

    fake_bq.Client = _client_factory
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    # Ensure `from google.cloud import bigquery` resolves the attribute too.
    google_cloud = sys.modules.get("google.cloud")
    if google_cloud is not None:
        monkeypatch.setattr(google_cloud, "bigquery", fake_bq, raising=False)


# ---- tests: python-client primary path -------------------------------------

@pytest.mark.parametrize(
    "module, func_name",
    [(statewide, "bq_query"), (lovers, "run_bq_query")],
)
def test_python_client_path_returns_df_without_subprocess(monkeypatch, module, func_name):
    expected = pd.DataFrame({"county": ["Monterey"], "events": [244]})
    recorder: dict = {}
    _install_fake_bigquery(monkeypatch, expected, recorder)
    monkeypatch.setenv("MBAL_GCP_PROJECT", "unit-test-project")

    # Any call to subprocess.run in this path is a bug.
    def _boom(*a, **k):  # pragma: no cover - only runs on failure
        raise AssertionError("subprocess must not be invoked on the python-client path")

    monkeypatch.setattr(module.subprocess, "run", _boom)

    out = getattr(module, func_name)("SELECT county, events FROM t")

    pd.testing.assert_frame_equal(out, expected)
    assert recorder["project"] == "unit-test-project"
    # No 100k cap surfaces in the client path; full result returned as-is.
    assert "max_rows" not in recorder.get("sql", "")
    # bqstorage acceleration is requested first.
    assert recorder["bqstorage"] is True


# ---- tests: fallback to CLI when client import/use fails --------------------

@pytest.mark.parametrize(
    "module, func_name",
    [(statewide, "bq_query"), (lovers, "run_bq_query")],
)
def test_falls_back_to_cli_on_import_error(monkeypatch, module, func_name):
    # Simulate google-cloud-bigquery being absent: the local import raises.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "google.cloud" and a and "bigquery" in (a[2] or ()):
            raise ImportError("no google-cloud-bigquery")
        if name.startswith("google.cloud.bigquery"):
            raise ImportError("no google-cloud-bigquery")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    cli_df = pd.DataFrame({"county": ["Statewide"], "events": [60616]})
    called: dict = {}

    def _fake_cli(sql, *a, **k):
        called["sql"] = sql
        return cli_df

    # Patch the private CLI helper so we exercise the dispatch, not a real shell-out.
    cli_name = "_bq_query_cli" if module is statewide else "_run_bq_query_cli"
    monkeypatch.setattr(module, cli_name, _fake_cli)

    out = getattr(module, func_name)("SELECT county, events FROM t")
    pd.testing.assert_frame_equal(out, cli_df)
    assert "SELECT" in called["sql"]


@pytest.mark.parametrize(
    "module, func_name",
    [(statewide, "bq_query"), (lovers, "run_bq_query")],
)
def test_falls_back_to_cli_on_client_runtime_error(monkeypatch, module, func_name):
    # Client imports fine but raises at query time (e.g. auth/credentials error).
    fake_bq = types.ModuleType("google.cloud.bigquery")

    def _client_factory(project=None):
        raise RuntimeError("could not determine default credentials")

    fake_bq.Client = _client_factory
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    google_cloud = sys.modules.get("google.cloud")
    if google_cloud is not None:
        monkeypatch.setattr(google_cloud, "bigquery", fake_bq, raising=False)

    cli_df = pd.DataFrame({"ok": [1]})

    def _fake_cli(sql, *a, **k):
        return cli_df

    cli_name = "_bq_query_cli" if module is statewide else "_run_bq_query_cli"
    monkeypatch.setattr(module, cli_name, _fake_cli)

    out = getattr(module, func_name)("SELECT 1")
    pd.testing.assert_frame_equal(out, cli_df)


@pytest.mark.parametrize(
    "module, func_name",
    [(statewide, "bq_query"), (lovers, "run_bq_query")],
)
def test_client_retries_without_bqstorage(monkeypatch, module, func_name):
    # First to_dataframe(create_bqstorage_client=True) raises; wrapper must retry
    # with create_bqstorage_client=False and return the result.
    expected = pd.DataFrame({"v": [1, 2, 3]})

    class _RetryJob:
        def result(self):
            return self

        def to_dataframe(self, create_bqstorage_client=False):
            if create_bqstorage_client:
                raise RuntimeError("bqstorage client not installed")
            return expected

    class _RetryClient:
        def __init__(self, project=None):
            pass

        def query(self, sql):
            return _RetryJob()

    fake_bq = types.ModuleType("google.cloud.bigquery")
    fake_bq.Client = _RetryClient
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    google_cloud = sys.modules.get("google.cloud")
    if google_cloud is not None:
        monkeypatch.setattr(google_cloud, "bigquery", fake_bq, raising=False)

    out = getattr(module, func_name)("SELECT v FROM t")
    pd.testing.assert_frame_equal(out, expected)
