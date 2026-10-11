"""The HTTP transport the input modules share: existence checks and byte-range reads.

Input modules take a `Transport` argument so tests can serve recorded files and count requests;
`UrllibTransport` is the real one. Every failure surfaces as `FetchError` with the URL in the
message, so a caller never sees a bare `URLError`, and nothing is retried or swallowed here:
a flaky bucket should fail the run loudly, not hide behind a retry.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final, Protocol

from lastlight import __version__

USER_AGENT: Final = f"lastlight/{__version__} (public-data client; github.com/cdcoonce/lastlight)"
DEFAULT_TIMEOUT_S: Final = 60.0


class FetchError(RuntimeError):
    """A request failed: the network, the server, or a response that is not what was asked for."""


@dataclass(frozen=True)
class ByteRange:
    """An inclusive byte range of a file; `end=None` reads to the end of the file."""

    start: int
    end: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or (self.end is not None and self.end < self.start):
            raise ValueError(f"invalid byte range: start={self.start}, end={self.end}")

    def header(self) -> str:
        """The value of the HTTP `Range` header."""
        return f"bytes={self.start}-" if self.end is None else f"bytes={self.start}-{self.end}"

    @property
    def length(self) -> int | None:
        return None if self.end is None else self.end - self.start + 1


class Transport(Protocol):
    def exists(self, url: str) -> bool:
        """True if the URL names an object; False only for a 404. Anything else raises."""
        ...

    def get(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        """The whole object, or exactly the requested bytes of it."""
        ...


class UrllibTransport:
    """`Transport` over the standard library: no extra dependency, no connection pooling."""

    def __init__(self, user_agent: str = USER_AGENT, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    def exists(self, url: str) -> bool:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": self.user_agent}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s):
                return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise FetchError(f"{url}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise FetchError(f"{url}: {_reason(error)}") from error

    def get(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        headers = {"User-Agent": self.user_agent}
        if byte_range is not None:
            headers["Range"] = byte_range.header()
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if byte_range is not None and response.status != 206:
                    # The server ignored the range: the body is the whole object (~140 MB for an
                    # HRRR file). Refuse before reading a byte of it.
                    raise FetchError(
                        f"{url}: requested byte range {byte_range.header()} but the server "
                        f"answered HTTP {response.status} instead of 206"
                    )
                body = response.read()
        except urllib.error.HTTPError as error:
            raise FetchError(f"{url}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise FetchError(f"{url}: {_reason(error)}") from error
        if byte_range is not None and byte_range.length not in (None, len(body)):
            raise FetchError(
                f"{url}: asked for {byte_range.length} bytes ({byte_range.header()}), "
                f"got {len(body)}"
            )
        return body


def _reason(error: BaseException) -> str:
    return str(error.reason) if isinstance(error, urllib.error.URLError) else str(error)
