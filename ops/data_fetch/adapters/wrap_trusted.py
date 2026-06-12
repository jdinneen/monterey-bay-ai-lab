"""Read-only wrapper for an existing TRUSTED output.

Does not fetch or overwrite anything. The base Adapter.fetch() detects
`spec.wraps_trusted` and routes to a validate+inventory path that reads the
trusted parquet and emits manifest/coverage/validation into reports/data_fetch/.
"""
from __future__ import annotations

from ..core import Adapter


class WrapTrustedAdapter(Adapter):
    """Inventory + validate an existing trusted production file (never writes it)."""

    # All behavior is inherited; the base handles wraps_trusted read-only mode.
    pass
