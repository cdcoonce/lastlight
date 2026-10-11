"""The real HTTP layer (urllib), exercised by replacing `urlopen`: no sockets."""

from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
from typing import Any

import pytest

from lastlight.cameras.transport import MAX_BODY_BYTES, FetchError, Response, urllib_fetch

URL = "https://example.test/frame.jpg"


class _Reply:
    """What `urlopen` returns: a context manager with `status`, `headers` and `read`."""

    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = email.message.Message()
        for key, value in headers.items():
            self.headers[key] = value
        self._body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(request: urllib.request.Request, timeout: float | None = None) -> Any:
        seen.append({"request": request, "timeout": timeout})
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def _http_error(status: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError(URL, status, "reason", message, io.BytesIO(b"ignored"))


def test_a_200_becomes_a_response_with_body_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, _Reply(200, {"ETag": '"v1"', "Content-Length": "3"}, b"abc"))
    response = urllib_fetch(URL, {}, 60.0)
    assert response.status == 200
    assert response.body == b"abc"
    assert response.header("etag") == '"v1"'
    assert response.header("CONTENT-LENGTH") == "3"


def test_the_request_carries_the_given_headers_and_timeout_and_is_a_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_urlopen(monkeypatch, _Reply(200, {}, b""))
    urllib_fetch(URL, {"User-Agent": "probe/1", "If-None-Match": '"v1"'}, 42.0)
    (call,) = seen
    request = call["request"]
    assert request.full_url == URL
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == "probe/1"
    assert request.get_header("If-none-match") == '"v1"'
    assert call["timeout"] == 42.0


def test_a_304_arrives_as_an_exception_and_is_returned_as_a_normal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_urlopen(monkeypatch, _http_error(304, {"ETag": '"v1"'}))
    response = urllib_fetch(URL, {}, 60.0)
    assert response.status == 304
    assert response.body == b""
    assert response.header("ETag") == '"v1"'


@pytest.mark.parametrize("status", [404, 429, 503])
def test_error_statuses_are_returned_not_raised(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _patch_urlopen(monkeypatch, _http_error(status))
    assert urllib_fetch(URL, {}, 60.0).status == status


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("name resolution failed"),
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
    ],
)
def test_transport_failures_become_fetch_errors(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    _patch_urlopen(monkeypatch, failure)
    with pytest.raises(FetchError):
        urllib_fetch(URL, {}, 60.0)


def test_a_body_larger_than_the_cap_is_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, _Reply(200, {}, b"x" * (MAX_BODY_BYTES + 1)))
    with pytest.raises(FetchError, match="larger"):
        urllib_fetch(URL, {}, 60.0)


# --- Response ------------------------------------------------------------------------------


def test_header_lookup_ignores_case_and_reports_absence() -> None:
    response = Response(status=200, headers={"Last-Modified": "x"}, body=b"")
    assert response.header("last-modified") == "x"
    assert response.header("ETag") is None
