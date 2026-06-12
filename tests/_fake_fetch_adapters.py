"""Importable fake adapters for data-fetch tests (need a real dotted import path
so registry.build_adapter() can resolve them)."""
from __future__ import annotations

import pandas as pd

from ops.data_fetch.core import Adapter


class FakeOkAdapter(Adapter):
    """Deterministic 3-chunk staged adapter; counts fetch_chunk calls per source."""

    _calls: dict[str, int] = {}

    def iter_chunks(self, start, end):
        for i in range(3):
            yield {"key": f"c{i}", "i": i}

    def fetch_chunk(self, chunk):
        FakeOkAdapter._calls[self.source] = FakeOkAdapter._calls.get(self.source, 0) + 1
        return pd.DataFrame({
            "id": [f"{self.source}-{chunk['key']}"],
            "t": [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=chunk["i"])],
            "val": [float(chunk["i"])],
        })


class FakeEmptyAdapter(Adapter):
    """Always returns empty -> exercises empty-output rejection."""

    def iter_chunks(self, start, end):
        yield {"key": "only"}

    def fetch_chunk(self, chunk):
        return pd.DataFrame()


class FakeBoomAdapter(Adapter):
    """Raises during planning -> exercises fetch-all resilience."""

    def iter_chunks(self, start, end):
        raise RuntimeError("boom: simulated source failure")

    def fetch_chunk(self, chunk):  # pragma: no cover
        raise RuntimeError("boom")
