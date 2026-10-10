"""Test-suite guard: no real network access unless a test is marked `network`.

Every test runs offline by default. A test that genuinely needs the network is marked
`@pytest.mark.network`; those tests are deselected by default (see `addopts` in
pyproject.toml) and run on purpose with `pytest -m network`.

The guard refuses outbound `connect` and `connect_ex`, which every TCP client goes through,
including `socket.create_connection`, urllib and asyncio. It does not cover DNS lookups or
UDP `sendto`; tests must not rely on those either.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest


class NetworkAccessError(RuntimeError):
    """Raised on a socket connect inside a test that did not opt in.

    Deliberately not an OSError: code under test that retries or falls back on OSError
    must not be able to swallow the block and let the test pass without the data.
    """


def _refuse(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> None:
    raise NetworkAccessError(
        f"network access is blocked in tests (connect to {address!r}); "
        "mark the test with @pytest.mark.network to allow it"
    )


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network") is not None:
        return
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
