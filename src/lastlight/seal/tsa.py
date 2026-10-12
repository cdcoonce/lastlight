"""RFC 3161 time-stamping: build a request, make the one network call, read and check the response.

Library choice: `rfc3161-client` (Trail of Bits, also what sigstore uses), which parses and
verifies timestamp responses. `parse_response` only reads one (the message imprint, the time and
the hash algorithm; status 0 is the only status it accepts, as in the library). `authenticate`
proves the token is genuine: the pinned TSA's certificate signed it, the certificate chains to the
pinned root, the certificate is a timestamping one, and it was valid at the time the token names.
It judges the certificates at that time, not today (RFC 3161: a timestamp issued while the
certificate was valid stays valid after it expires), and trusts only the pair in
`lastlight.seal.trust`, never the certificates a response happens to carry.

The library reports a failed chain, signature or key usage as one exception, so `authenticate`
checks them in a fixed order before it asks the library for the verdict, and the reason names the
first check that fails: status, certificate chain, validity, extended key usage, signature. What
the token covers (the imprint) is compared separately by the caller, after authenticity, so a
forged token is named for the forgery and not for the bytes it claims to cover.

The request asks for a SHA-256 imprint of the data (the canonical forecast bytes, so the
imprint equals the forecast's digest), with `certReq` set so the response carries the TSA's
certificates. It carries no nonce: the imprint is already unique to the forecast, a replayed
response can only ever carry the same digest, and a request without a nonce is deterministic,
which keeps the recorded test fixture exact.

`post_request` is the only function in lastlight that touches the network. Tests replace it
(`monkeypatch.setattr(lastlight.seal.tsa, "post_request", fake)`) with a canned response.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from cryptography import x509
from rfc3161_client import (
    HashAlgorithm,
    TimestampRequestBuilder,
    VerificationError,
    VerifierBuilder,
    decode_timestamp_response,
)
from rfc3161_client.tsp import TimeStampResponse

from lastlight import __version__
from lastlight.seal import trust as trust_module
from lastlight.seal.trust import Trust, fingerprint

DEFAULT_TSA_URL = "https://freetsa.org/tsr"
USER_AGENT = f"lastlight/{__version__} (+https://github.com/cdcoonce/lastlight)"
TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 1024 * 1024

_SHA256_OID = "2.16.840.1.101.3.4.2.1"
_GRANTED = 0  # status 1 (granted with modifications) is refused, as the library refuses it

_CHECK_NAMES = {
    "chain": "certificate chain",
    "validity": "certificate validity",
    "eku": "extended key usage (EKU)",
    "signature": "signature",
}


class TsaError(Exception):
    """The timestamp authority could not be reached, refused, or sent something unusable."""


class TokenRejected(TsaError):
    """A well-formed, granted token that is not genuine; `check` names the check that failed.

    `check` is one of "chain", "validity", "eku" or "signature", and the message starts with the
    check's name, so the reason a person reads says which one it was.
    """

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{_CHECK_NAMES[check]}: {detail}")
        self.check = check


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


def _read(der: bytes) -> tuple[TimeStampResponse, Timestamp]:
    """Decode a DER `TimeStampResp`; only a granted, well-formed SHA-256 token is accepted."""
    try:
        response = decode_timestamp_response(der)
        if response.status != _GRANTED:
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
    return response, Timestamp(gen_time=gen_time, imprint=imprint, algorithm="sha256")


def parse_response(der: bytes) -> Timestamp:
    """Read a DER `TimeStampResp`; anything but a granted, well-formed token is a TsaError.

    This reads what the token says and nothing more: it does not check the signature. Status 0
    is the only status accepted (1, granted with modifications, is refused as the library does).
    Use `authenticate` to know the token is genuine.
    """
    return _read(der)[1]


def _signer_is_pinned(response: TimeStampResponse, trust: Trust) -> None:
    """The certificate that signed the token must be the pinned TSA certificate (the chain)."""
    signed = response.signed_data
    signers = list(signed.signer_infos)
    if len(signers) != 1:
        raise TokenRejected("signature", f"expected exactly one signer, found {len(signers)}")
    (signer,) = signers
    try:
        carried = [x509.load_der_x509_certificate(raw) for raw in signed.certificates]
    except ValueError as error:
        raise TokenRejected("chain", "a certificate carried in the token cannot be read") from error
    if not carried:  # no certificates in the token: the pinned one is the only candidate
        return
    named = [
        c for c in carried if c.issuer == signer.issuer and c.serial_number == signer.serial_number
    ]
    if not named:
        raise TokenRejected(
            "chain", "none of the certificates carried in the token is the one that signed it"
        )
    if named[0] != trust.tsa:
        raise TokenRejected(
            "chain",
            "the token was signed by a certificate other than the pinned timestamp authority's "
            f"(signer SHA-256 {fingerprint(named[0])}), so it does not chain to the pinned root",
        )


def _valid_at(when: dt.datetime, trust: Trust) -> None:
    """Both pinned certificates must have been valid when the token says it was issued."""
    for name, certificate in (("TSA", trust.tsa), ("root", trust.root)):
        if when < certificate.not_valid_before_utc:
            began = certificate.not_valid_before_utc.date().isoformat()
            raise TokenRejected(
                "validity",
                f"the token's time is before the pinned {name} certificate was valid "
                f"(not yet valid until {began})",
            )
        if when > certificate.not_valid_after_utc:
            ended = certificate.not_valid_after_utc.date().isoformat()
            raise TokenRejected(
                "validity",
                f"the token's time is after the pinned {name} certificate expired ({ended})",
            )


def _is_timestamping(certificate: x509.Certificate) -> None:
    """The TSA certificate needs a critical extended key usage that includes id-kp-timeStamping."""
    try:
        usage = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    except x509.ExtensionNotFound:
        raise TokenRejected(
            "eku", "the pinned TSA certificate has no extended key usage extension"
        ) from None
    if not usage.critical:
        raise TokenRejected(
            "eku", "the pinned TSA certificate's extended key usage is not critical"
        )
    if x509.ExtendedKeyUsageOID.TIME_STAMPING not in usage.value:
        raise TokenRejected(
            "eku",
            "the pinned TSA certificate is not allowed to sign timestamps (id-kp-timeStamping)",
        )


_OPENSSL_REASON = re.compile(r"Verify error: ([A-Za-z0-9 ,.'-]{1,120})")


def _library_rejection(error: Exception) -> TokenRejected:
    """Name what the library's single "could not verify" exception means.

    By the time it is called the earlier checks have passed, so what is left is the CMS signature
    (or the signed attributes it covers) or, rarely, a chain failure OpenSSL found that they did
    not. The library's text is a debug dump; only a short, fixed-alphabet reason is passed on.
    """
    text = str(error)
    found = _OPENSSL_REASON.search(text)
    if found or "certificate verify error" in text or "signer certificate not found" in text:
        why = f" ({found.group(1).strip()})" if found else ""
        return TokenRejected("chain", f"the certificate chain was rejected{why}")
    return TokenRejected(
        "signature", "the token's signature does not verify against the pinned TSA certificate"
    )


def authenticate(der: bytes, *, trust: Trust | None = None) -> Timestamp:
    """Prove a response is a genuine token from the pinned authority, and say what it states.

    Checks, in this order, and raises `TokenRejected` naming the first that fails (a malformed or
    refused response is a plain `TsaError` from the parser): the signing certificate is the pinned
    TSA certificate (so it chains to the pinned root); both pinned certificates were valid at the
    token's own time; the TSA certificate carries a critical timestamping key usage; the signature
    verifies. `trust` defaults to `lastlight.seal.trust.DEFAULT_TRUST`, FreeTSA's pinned pair.
    It does not compare the imprint with anything: the caller does, after this.
    """
    anchors = trust if trust is not None else trust_module.DEFAULT_TRUST
    response, stamp = _read(der)
    _signer_is_pinned(response, anchors)
    _valid_at(stamp.gen_time, anchors)
    _is_timestamping(anchors.tsa)
    verifier = VerifierBuilder().add_root_certificate(anchors.root).tsa_certificate(anchors.tsa)
    try:
        verifier.build().verify(response, stamp.imprint)
    except (VerificationError, ValueError) as error:
        raise _library_rejection(error) from error
    return stamp


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


def request_timestamp(
    data: bytes, url: str = DEFAULT_TSA_URL, *, trust: Trust | None = None
) -> bytes:
    """Get a timestamp response for `data`, check it is genuine and covers exactly that data.

    The response must verify against the pinned authority (`authenticate`; `trust` replaces the
    pinned pair) before anything else is believed, so a bad response never reaches the caller.
    Returns the DER response unchanged, ready to be written as a `.tsr` file.
    """
    reply = post_request(url, build_request(data))
    stamp = authenticate(reply, trust=trust)
    if stamp.algorithm != "sha256" or stamp.imprint != hashlib.sha256(data).digest():
        raise TsaError("the response's message imprint is not the digest that was sent")
    return reply
