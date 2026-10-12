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
from pathlib import Path
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


@pytest.fixture
def trust_test_tsa(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Trust the throwaway test authority (`tsa_factory`) instead of the pinned public one.

    The product verifies a timestamp against one pair of pinned certificates, FreeTSA's, and has
    no way to add trust from outside (no environment variable, path or flag). The responses the
    tests fake are really signed, but by a test authority, so a test that seals or verifies one
    swaps the module default for the authority's pair. This is the only place that is done. It
    is never autouse: a test that claims to check the real pins must not request it, and the
    fixtures below that seal request it themselves. Returns the `Trust` in force.

    Imported here, not at the top, for the same reason as in `ledger_dir`.
    """
    import tsa_factory

    from lastlight.seal import trust

    pair = tsa_factory.authority()
    swapped = trust.Trust(root=pair.root_certificate, tsa=pair.tsa_certificate)
    monkeypatch.setattr(trust, "DEFAULT_TRUST", swapped)
    return swapped


@pytest.fixture
def ledger_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trust_test_tsa: Any) -> Path:
    """A ledger of three sealed evenings (`ledger_factory.DATES`), each sealed with its painting.

    `ledger_factory` is imported here, not at the top, because the network-guard tests run this
    file on its own in a throwaway project that does not have the other test helpers. The test
    authority stays trusted for the whole test, so the ledger still verifies.
    """
    import ledger_factory

    ledger = tmp_path / "ledger"
    ledger_factory.seal_dates(ledger, monkeypatch)
    return ledger


@pytest.fixture
def shuffled_ledger_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trust_test_tsa: Any
) -> Path:
    """The same three evenings and paintings as `ledger_dir`, sealed out of date order.

    The index lists 2026-10-13, then 2026-10-11, then 2026-10-12: neither ascending nor
    descending, so only sorting by date can put them newest first.
    """
    import ledger_factory

    ledger = tmp_path / "ledger"
    ledger_factory.seal_dates_out_of_date_order(ledger, monkeypatch)
    return ledger
