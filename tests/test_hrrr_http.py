"""The default HTTP transport: range requests, existence checks, and loud failures.

`urllib.request.urlopen` is replaced with a recorder, so no socket is ever opened; the transport's
real behaviour against the bucket is covered by the one `network` test in `test_hrrr_network.py`.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from lastlight.inputs import http
from lastlight.inputs.http import ByteRange, FetchError, UrllibTransport

URL = "https://example.invalid/file.grib2"


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200) -> None:
        self._body = body
        self.status = status
        self.read_called = False

    def read(self) -> bytes:
        self.read_called = True
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Recorder:
    """Stands in for `urlopen`: remembers the request and returns or raises what it is told."""

    def __init__(self, outcome: FakeResponse | BaseException) -> None:
        self.outcome = outcome
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[Any] = []

    def __call__(self, request: urllib.request.Request, timeout: Any = None) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _install(monkeypatch: pytest.MonkeyPatch, outcome: FakeResponse | BaseException) -> Recorder:
    recorder = Recorder(outcome)
    monkeypatch.setattr(http.urllib.request, "urlopen", recorder)
    return recorder


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, "reason", Message(), None)


# --- ByteRange ------------------------------------------------------------------------------------


def test_a_closed_range_renders_as_an_inclusive_range_header() -> None:
    assert ByteRange(10, 19).header() == "bytes=10-19"


def test_an_open_range_renders_as_a_header_to_the_end_of_the_file() -> None:
    assert ByteRange(10, None).header() == "bytes=10-"


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (10, 9)])
def test_a_range_must_be_ordered_and_not_negative(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="range"):
        ByteRange(start, end)


# --- get ------------------------------------------------------------------------------------------


def test_get_with_a_range_sends_one_range_header_and_returns_the_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _install(monkeypatch, FakeResponse(b"0123456789", status=206))

    body = UrllibTransport().get(URL, ByteRange(100, 109))

    assert body == b"0123456789"
    (request,) = recorder.requests
    assert request.full_url == URL
    assert request.get_method() == "GET"
    assert request.get_header("Range") == "bytes=100-109"


def test_get_identifies_itself_and_sets_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install(monkeypatch, FakeResponse(b"x", status=200))

    UrllibTransport().get(URL)

    assert recorder.requests[0].get_header("User-agent", "").startswith("lastlight/")
    assert recorder.timeouts[0] is not None


def test_get_without_a_range_returns_the_whole_body(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install(monkeypatch, FakeResponse(b"whole", status=200))

    assert UrllibTransport().get(URL) == b"whole"
    assert recorder.requests[0].get_header("Range") is None


def test_get_refuses_a_server_that_ignores_the_range_without_reading_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 200 for a range request means the whole object (about 140 MB) is on its way.
    response = FakeResponse(b"the whole file", status=200)
    _install(monkeypatch, response)

    with pytest.raises(FetchError, match="range"):
        UrllibTransport().get(URL, ByteRange(0, 9))

    assert not response.read_called


def test_get_refuses_a_body_of_the_wrong_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, FakeResponse(b"short", status=206))

    with pytest.raises(FetchError, match="10 bytes"):
        UrllibTransport().get(URL, ByteRange(0, 9))


def test_get_does_not_length_check_an_open_ended_range(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, FakeResponse(b"tail of the file", status=206))

    assert UrllibTransport().get(URL, ByteRange(500, None)) == b"tail of the file"


def test_get_turns_an_http_error_into_a_fetch_error_naming_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _http_error(404))

    with pytest.raises(FetchError, match="404") as caught:
        UrllibTransport().get(URL)

    assert URL in str(caught.value)


def test_get_turns_a_network_failure_into_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, urllib.error.URLError("name resolution failed"))

    with pytest.raises(FetchError, match="name resolution failed"):
        UrllibTransport().get(URL)


def test_get_turns_a_timeout_into_a_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, TimeoutError("timed out"))

    with pytest.raises(FetchError, match="timed out"):
        UrllibTransport().get(URL)


# --- exists ---------------------------------------------------------------------------------------


def test_exists_sends_a_head_request(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install(monkeypatch, FakeResponse(status=200))

    assert UrllibTransport().exists(URL) is True
    assert recorder.requests[0].get_method() == "HEAD"


def test_exists_is_false_for_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _http_error(404))

    assert UrllibTransport().exists(URL) is False


@pytest.mark.parametrize("code", [403, 500, 503])
def test_exists_raises_for_any_other_http_error_instead_of_guessing(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    _install(monkeypatch, _http_error(code))

    with pytest.raises(FetchError, match=str(code)):
        UrllibTransport().exists(URL)


def test_exists_turns_a_network_failure_into_a_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, urllib.error.URLError("unreachable"))

    with pytest.raises(FetchError, match="unreachable"):
        UrllibTransport().exists(URL)
