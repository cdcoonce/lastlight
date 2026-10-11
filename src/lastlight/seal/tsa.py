"""RFC 3161 time-stamping: build a request, make the one network call, read the response.

Library choice: `rfc3161-client` (Trail of Bits, also what sigstore uses), which parses and,
in the follow-up slice, verifies timestamp responses. The slice verifies the message imprint
and the time only; validating the signature chain against the TSA certificate is a follow-up,
so until then `verify` trusts the TSA's response as delivered.

The request asks for a SHA-256 imprint of the data (the canonical forecast bytes, so the
imprint equals the forecast's digest), with `certReq` set so the response carries the TSA's
certificates for that later check. It carries no nonce: the imprint is already unique to the
forecast, a replayed response can only ever carry the same digest, and a request without a
nonce is deterministic, which keeps the recorded test fixture exact.

`post_request` is the only function in lastlight that touches the network. Tests replace it
(`monkeypatch.setattr(lastlight.seal.tsa, "post_request", fake)`) with a recorded response.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from rfc3161_client import HashAlgorithm, TimestampRequestBuilder, decode_timestamp_response

from lastlight import __version__

DEFAULT_TSA_URL = "https://freetsa.org/tsr"
USER_AGENT = f"lastlight/{__version__} (+https://github.com/cdcoonce/lastlight)"
TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 1024 * 1024

_SHA256_OID = "2.16.840.1.101.3.4.2.1"
_GRANTED = (0, 1)  # granted, granted with modifications


class TsaError(Exception):
    """The timestamp authority could not be reached, refused, or sent something unusable."""


@dataclass(frozen=True)
class Timestamp:
    """What a response says: when it was issued, and which hash it covers."""

    gen_time: dt.datetime
    imprint: bytes
    algorithm: str


def build_request(data: bytes) -> bytes:
    """The DER `TimeStampReq` asking for a SHA-256 timestamp over `data`."""
    request = (
        TimestampRequestBuilder()
        .data(data)
        .hash_algorithm(HashAlgorithm.SHA256)
        .nonce(nonce=False)
        .cert_request(cert_request=True)
        .build()
    )
    return request.as_bytes()


def parse_response(der: bytes) -> Timestamp:
    """Read a DER `TimeStampResp`; anything but a granted, well-formed token is a TsaError."""
    try:
        response = decode_timestamp_response(der)
        if response.status not in _GRANTED:
            raise TsaError(
                f"the timestamp authority did not grant the request (status {response.status})"
            )
        info = response.tst_info
        oid = info.message_imprint.hash_algorithm.dotted_string
        imprint = info.message_imprint.message
        gen_time = info.gen_time
    except TsaError:
        raise
    except ValueError as error:  # the parser reports every malformed encoding as a ValueError
        raise TsaError(f"unreadable timestamp response: {error}") from error
    if oid != _SHA256_OID:
        raise TsaError(f"the timestamp covers an unsupported hash (OID {oid}); expected SHA-256")
    return Timestamp(gen_time=gen_time, imprint=imprint, algorithm="sha256")


def post_request(url: str, request: bytes, *, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """POST a DER request to the TSA at `url` and return the DER response body."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise TsaError(f"the timestamp authority URL must be an http or https URL, got {url!r}")
    http_request = urllib.request.Request(
        url,
        data=request,
        method="POST",
        headers={
            "Content-Type": "application/timestamp-query",
            "Accept": "application/timestamp-reply",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            body: bytes = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise TsaError(f"the timestamp authority answered HTTP {error.code}") from error
    except OSError as error:  # URLError, timeouts and connection failures are all OSErrors
        raise TsaError(f"could not reach the timestamp authority: {error}") from error
    if len(body) > MAX_RESPONSE_BYTES:
        raise TsaError("the timestamp authority's response is too large")
    return body


def request_timestamp(data: bytes, url: str = DEFAULT_TSA_URL) -> bytes:
    """Get a timestamp response for `data` and check it covers exactly that data.

    Returns the DER response unchanged, ready to be written as a `.tsr` file.
    """
    reply = post_request(url, build_request(data))
    stamp = parse_response(reply)
    if stamp.algorithm != "sha256" or stamp.imprint != hashlib.sha256(data).digest():
        raise TsaError("the response's message imprint is not the digest that was sent")
    return reply
