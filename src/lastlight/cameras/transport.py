"""The HTTP layer of the collector: one injectable function and a stdlib implementation.

The collector never opens a connection itself. It calls a `Fetch`, a function from
(url, headers, timeout) to a `Response`, so tests substitute a scripted fake and no socket is
ever opened. `urllib_fetch` is the real one. A 304 and any 4xx or 5xx status come back as a
`Response` for the collector to judge; only a failure to get any response (no connection, a
timeout, a body that is too large) raises `FetchError`.
"""

from __future__ import annotations

import http.client
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

# A 800x300 JPEG is tens of kilobytes. A body far beyond that is not a camera frame, and an
# unbounded read would let a misbehaving server fill the capture host's memory.
MAX_BODY_BYTES: Final = 10 * 1024 * 1024


class FetchError(Exception):
    """No HTTP response was obtained: refused or reset connection, timeout, oversize body."""


@dataclass(frozen=True, slots=True)
class Response:
    """An HTTP response: status, headers as the server sent them, and the body."""

    status: int
    headers: Mapping[str, str]
    body: bytes = b""

    def header(self, name: str) -> str | None:
        """The value of a header, matched without regard to case, or None if absent."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


Fetch = Callable[[str, Mapping[str, str], float], Response]
"""fetch(url, request_headers, timeout_seconds) -> Response; raises FetchError."""


def urllib_fetch(url: str, headers: Mapping[str, str], timeout_s: float) -> Response:
    """GET `url` with `urllib`, returning every HTTP status (including 304) as a `Response`."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as reply:
            body = reply.read(MAX_BODY_BYTES + 1)
            if len(body) > MAX_BODY_BYTES:
                raise FetchError(f"response from {url} is larger than {MAX_BODY_BYTES} bytes")
            return Response(status=reply.status, headers=dict(reply.headers.items()), body=body)
    except urllib.error.HTTPError as error:
        # urllib raises for every non-2xx status, 304 included.
        with error:
            headers_sent = error.headers.items() if error.headers is not None else []
            return Response(status=error.code, headers=dict(headers_sent), body=b"")
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        raise FetchError(f"{type(error).__name__}: {error}") from error
