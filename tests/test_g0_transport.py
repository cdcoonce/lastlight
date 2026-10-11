"""The polite transport the G0 run uses: a pause between requests and retries on transient errors.

A year of archived inputs is a few thousand requests to public buckets. The existing transport
deliberately retries nothing; the G0 run wraps it so that it never exceeds a couple of requests
per second, retries a flaky answer a few times with a growing pause, and never retries a 404
(a file that is not in the archive is an answer, not a fault).
"""

from __future__ import annotations

import pytest
from hrrr_support import FakeTransport

from lastlight.g0.transport import PoliteTransport, is_not_found
from lastlight.inputs.http import ByteRange, FetchError


class Clock:
    """A clock and a sleep that move together, so the tests take no real time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Flaky:
    """Fails with `errors` in turn on every call, then answers."""

    def __init__(self, errors: list[str], body: bytes = b"ok") -> None:
        self.errors = list(errors)
        self.body = body
        self.calls = 0

    def exists(self, url: str) -> bool:
        self.calls += 1
        if self.errors:
            raise FetchError(f"{url}: {self.errors.pop(0)}")
        return True

    def get(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        self.calls += 1
        if self.errors:
            raise FetchError(f"{url}: {self.errors.pop(0)}")
        return self.body


def polite(inner, clock: Clock, **kwargs) -> PoliteTransport:
    return PoliteTransport(
        inner, sleep=clock.sleep, clock=clock.monotonic, **{"min_interval_s": 0.5, **kwargs}
    )


def test_requests_are_spaced_at_least_the_minimum_interval_apart() -> None:
    clock = Clock()
    transport = polite(FakeTransport({"u": b"abc"}), clock)
    for _ in range(4):
        assert transport.get("u") == b"abc"
    assert clock.slept == pytest.approx([0.5, 0.5, 0.5])  # none before the first request
    assert transport.requests == 4


def test_time_already_spent_counts_toward_the_interval() -> None:
    clock = Clock()
    transport = polite(FakeTransport({"u": b"abc"}), clock)
    transport.get("u")
    clock.now += 0.3
    transport.get("u")
    assert clock.slept == pytest.approx([0.2])
    clock.now += 2.0
    transport.get("u")
    assert clock.slept == pytest.approx([0.2])  # nothing to wait for


def test_exists_and_get_share_the_same_pacing() -> None:
    clock = Clock()
    transport = polite(FakeTransport({"u": b"abc"}), clock)
    assert transport.exists("u")
    transport.get("u")
    assert clock.slept == pytest.approx([0.5])


def test_a_transient_failure_is_retried_with_a_growing_pause() -> None:
    clock = Clock()
    inner = Flaky(["timed out", "HTTP 503 Slow Down"])
    transport = polite(inner, clock, retries=3, backoff_s=4.0)
    assert transport.get("u") == b"ok"
    assert inner.calls == 3
    assert 4.0 in clock.slept and 8.0 in clock.slept


def test_a_failure_that_keeps_happening_is_raised_after_the_retries() -> None:
    clock = Clock()
    inner = Flaky(["timed out"] * 10)
    transport = polite(inner, clock, retries=2, backoff_s=1.0)
    with pytest.raises(FetchError, match="timed out"):
        transport.get("u")
    assert inner.calls == 3  # the first try and two retries


def test_a_404_is_never_retried() -> None:
    clock = Clock()
    inner = Flaky(["HTTP 404 Not Found"] * 5)
    transport = polite(inner, clock, retries=3)
    with pytest.raises(FetchError, match="404"):
        transport.get("u")
    assert inner.calls == 1


def test_is_not_found_reads_the_http_status_in_the_message() -> None:
    assert is_not_found(FetchError("https://x/y: HTTP 404 Not Found"))
    assert is_not_found(FetchError("https://x/y: HTTP 404"))
    assert not is_not_found(FetchError("https://x/y: HTTP 403 Forbidden"))
    assert not is_not_found(FetchError("https://x/y: HTTP 5040 odd"))
    assert not is_not_found(FetchError("https://x/y: timed out"))


def test_the_defaults_are_polite() -> None:
    transport = PoliteTransport(FakeTransport({}))
    assert transport.min_interval_s >= 0.4  # at most about two requests per second
    assert transport.retries >= 1
