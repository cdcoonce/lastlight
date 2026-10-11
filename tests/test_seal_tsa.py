"""The RFC 3161 client: request, response parsing, imprint checks and the one network call."""

from __future__ import annotations

import datetime as dt
import io
import urllib.error
import urllib.request
from typing import Any

import pytest
import seal_helpers as helpers

from lastlight.seal import tsa
from lastlight.seal.tsa import DEFAULT_TSA_URL, Timestamp, TsaError

NOT_GRANTED = bytes.fromhex("30053003020102")  # status 2 (rejection), no token


# --- the request --------------------------------------------------------------------------


def test_the_default_tsa_is_freetsa() -> None:
    assert DEFAULT_TSA_URL == "https://freetsa.org/tsr"


def test_the_request_asks_for_a_sha256_imprint_of_the_data_without_a_nonce() -> None:
    # These exact bytes were accepted by the public TSA when the fixture was recorded.
    assert tsa.build_request(helpers.STUB_CANONICAL).hex() == helpers.STUB_REQUEST_HEX


def test_the_request_is_deterministic() -> None:
    assert tsa.build_request(b"abc") == tsa.build_request(b"abc")
    assert tsa.build_request(b"abc") != tsa.build_request(b"abd")


# --- the recorded response ----------------------------------------------------------------


def test_the_recorded_response_parses_to_the_stub_digest_and_a_utc_time() -> None:
    stamp = tsa.parse_response(helpers.recorded_tsr())
    assert stamp.algorithm == "sha256"
    assert stamp.imprint.hex() == helpers.STUB_DIGEST
    assert stamp.gen_time.utcoffset() == dt.timedelta(0)
    assert stamp.gen_time.date() == dt.date(2026, 10, 11)


def test_the_recorded_response_precedes_sunset_on_the_stub_date() -> None:
    stamp = tsa.parse_response(helpers.recorded_tsr())
    assert stamp.gen_time < helpers.sunset(helpers.STUB_DATE)


def test_a_re_dated_copy_reports_the_new_time_and_digest() -> None:
    when = helpers.sunset(helpers.STUB_DATE) - dt.timedelta(hours=1, seconds=3)
    other = "ab" * 32
    stamp = tsa.parse_response(helpers.make_tsr(other, when))
    assert stamp.imprint.hex() == other
    assert stamp.gen_time == when.replace(microsecond=0)


@pytest.mark.parametrize(
    "bad",
    [
        b"",
        b"not a timestamp response",
        b"\x30\x00",
        NOT_GRANTED,
    ],
)
def test_a_response_that_is_not_a_granted_timestamp_is_an_error(bad: bytes) -> None:
    with pytest.raises(TsaError):
        tsa.parse_response(bad)


def test_a_truncated_response_is_an_error() -> None:
    with pytest.raises(TsaError):
        tsa.parse_response(helpers.recorded_tsr()[:-40])


def test_a_response_with_a_rejection_status_but_a_token_is_an_error() -> None:
    template = helpers.recorded_tsr()
    patched = bytearray(template)
    marker = bytes.fromhex("3003020100")  # PKIStatusInfo { status granted }
    at = template.index(marker)
    patched[at + 4] = 2
    with pytest.raises(TsaError):
        tsa.parse_response(bytes(patched))


# --- asking the TSA -----------------------------------------------------------------------


def test_request_timestamp_posts_the_request_and_returns_a_matching_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = helpers.FakeTsa(helpers.recorded_tsr())
    monkeypatch.setattr(tsa, "post_request", fake)
    reply = tsa.request_timestamp(helpers.STUB_CANONICAL, "https://tsa.example/tsr")
    assert reply == helpers.recorded_tsr()
    assert fake.calls == [("https://tsa.example/tsr", tsa.build_request(helpers.STUB_CANONICAL))]


def test_request_timestamp_refuses_a_response_for_other_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(helpers.recorded_tsr()))
    with pytest.raises(TsaError, match="imprint"):
        tsa.request_timestamp(b"some other forecast", "https://tsa.example/tsr")


def test_request_timestamp_refuses_a_response_that_is_not_granted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(NOT_GRANTED))
    with pytest.raises(TsaError):
        tsa.request_timestamp(helpers.STUB_CANONICAL, "https://tsa.example/tsr")


# --- the one network function -------------------------------------------------------------


class FakeHttpResponse(io.BytesIO):
    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def fake(request: urllib.request.Request, timeout: float | None = None) -> FakeHttpResponse:
        seen.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data,
                "headers": {key.lower(): value for key, value in request.header_items()},
                "timeout": timeout,
            }
        )
        if isinstance(outcome, Exception):
            raise outcome
        return FakeHttpResponse(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def test_post_request_sends_the_query_as_a_post_and_returns_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = patch_urlopen(monkeypatch, b"reply-bytes")
    assert tsa.post_request("https://tsa.example/tsr", b"query-bytes") == b"reply-bytes"
    (call,) = seen
    assert call["url"] == "https://tsa.example/tsr"
    assert call["method"] == "POST"
    assert call["data"] == b"query-bytes"
    assert call["headers"]["content-type"] == "application/timestamp-query"
    assert "lastlight" in call["headers"]["user-agent"].lower()
    assert call["timeout"] is not None and 0 < call["timeout"] <= 120


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://tsa.example/tsr", 503, "unavailable", {}, None),  # type: ignore[arg-type]
        urllib.error.URLError("no route"),
        TimeoutError("slow"),
        ConnectionResetError("reset"),
    ],
)
def test_transport_failures_become_tsa_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    patch_urlopen(monkeypatch, error)
    with pytest.raises(TsaError):
        tsa.post_request("https://tsa.example/tsr", b"q")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://localhost/etc/passwd",
        "ftp://tsa.example/tsr",
        "tsa.example/tsr",
        "",
        "//tsa.example/tsr",
        "https:///tsr",
    ],
)
def test_only_http_and_https_urls_are_used(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    seen = patch_urlopen(monkeypatch, b"never")
    with pytest.raises(TsaError, match="http"):
        tsa.post_request(url, b"q")
    assert seen == []


def test_the_response_is_read_with_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []

    class Endless(FakeHttpResponse):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and size > 0, "the read must be bounded"
            sizes.append(size)
            return b"x" * min(size, 2 * 1024 * 1024)

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: Endless())
    with pytest.raises(TsaError, match="large"):
        tsa.post_request("https://tsa.example/tsr", b"q")
    assert sizes == [tsa.MAX_RESPONSE_BYTES + 1]


def test_an_oversized_response_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_urlopen(monkeypatch, b"x" * (1024 * 1024 + 1))
    with pytest.raises(TsaError, match="large"):
        tsa.post_request("https://tsa.example/tsr", b"q")


def test_the_timestamp_record_is_immutable() -> None:
    stamp = Timestamp(gen_time=dt.datetime(2026, 10, 11, tzinfo=dt.UTC), imprint=b"", algorithm="x")
    with pytest.raises(AttributeError):
        stamp.algorithm = "y"  # type: ignore[misc]
