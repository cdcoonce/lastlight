"""The polite transport the G0 run uses: paced requests, and retries for transient faults.

A year of archived inputs is a few thousand requests to two public buckets. The shared
`UrllibTransport` deliberately retries nothing, so this wrapper adds what a long unattended run
needs and nothing more: at most one request every `min_interval_s` (two a second by default,
never parallel), and up to `retries` more tries after a failure, pausing `backoff_s` times the
number of failures so far. A 404 is never retried: a file that is not in the archive is an
answer. Whatever still fails after the retries is raised unchanged, for the caller to treat as a
fault of the network rather than a fact about the date.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Final, TypeVar

from lastlight.inputs.http import ByteRange, FetchError, Transport

DEFAULT_MIN_INTERVAL_S: Final = 0.5
DEFAULT_RETRIES: Final = 3
DEFAULT_BACKOFF_S: Final = 5.0

_NOT_FOUND = re.compile(r"HTTP 404(?!\d)")
_Result = TypeVar("_Result")


def is_not_found(error: FetchError) -> bool:
    """True when the failure is the server saying the object does not exist (HTTP 404)."""
    return _NOT_FOUND.search(str(error)) is not None


class PoliteTransport:
    """A `Transport` that paces and retries another one. `requests` counts every attempt made."""

    def __init__(
        self,
        inner: Transport,
        *,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        retries: int = DEFAULT_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.requests = 0
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def _wait_turn(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (self._clock() - self._last)
            if remaining > 0.0:
                self._sleep(remaining)
        self._last = self._clock()
        self.requests += 1

    def _attempt(self, call: Callable[[], _Result]) -> _Result:
        for failures in range(self.retries + 1):
            self._wait_turn()
            try:
                return call()
            except FetchError as error:
                if is_not_found(error) or failures == self.retries:
                    raise
                self._sleep(self.backoff_s * (failures + 1))
        raise AssertionError("unreachable: the last attempt returns or raises")

    def exists(self, url: str) -> bool:
        return self._attempt(lambda: self.inner.exists(url))

    def get(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        return self._attempt(lambda: self.inner.get(url, byte_range))
