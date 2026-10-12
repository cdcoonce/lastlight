"""The certificates a timestamp is trusted through: FreeTSA's, pinned in the repository.

A timestamp is evidence only because an independent authority signed it, so `verify` and `seal`
check that signature against two certificates committed here as PEM files under `certs/`
(`certs/README.md` says where each came from): the root CA certificate and the TSA certificate
that signs the responses `DEFAULT_TSA_URL` returns. Their SHA-256 fingerprints are also written
down below, and `load_freetsa` refuses a file that does not match, so changing a pin takes a
visible edit in two places. Trusting whatever certificates a response carries would be
self-trust and prove nothing, which is why the response's own certificates are only ever
compared with these.

There is deliberately no way to add trust from outside. Nothing here (or anywhere in `seal`)
reads an environment variable, a path or a command-line option. The verifying functions take an
explicit `trust=` argument whose default is `DEFAULT_TRUST`, resolved when they are called, so
the tests can replace the default through exactly one fixture (`trust_test_tsa` in
`tests/conftest.py`) and nothing in production can.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import Encoding

CERTS_DIR: Traversable = resources.files("lastlight.seal") / "certs"
ROOT_FILE = "freetsa-root.pem"
TSA_FILE = "freetsa-tsa.pem"

# SHA-256 of each certificate's DER encoding (what `openssl x509 -fingerprint -sha256` prints),
# taken from https://freetsa.org/files/cacert.pem and https://freetsa.org/files/tsa.crt.
FREETSA_ROOT_SHA256 = "a6379e7cecc05faa3cbf076013d745e327bbbaa38c0b9af22469d4701d18aabc"
FREETSA_TSA_SHA256 = "32e841a95cc1164101ffde41298ef2fc75c1c4372ef095e88a6bbd47dfb191fc"


class TrustError(Exception):
    """A pinned certificate is missing, unreadable, swapped, or not issued by its root."""


def fingerprint(certificate: x509.Certificate) -> str:
    """Lowercase hex SHA-256 of the certificate's DER encoding."""
    return hashlib.sha256(certificate.public_bytes(Encoding.DER)).hexdigest()


@dataclass(frozen=True)
class Trust:
    """A root CA certificate and the one TSA certificate it issued: the pair a token must match."""

    root: x509.Certificate
    tsa: x509.Certificate

    def __post_init__(self) -> None:
        try:
            self.tsa.verify_directly_issued_by(self.root)
        except (ValueError, TypeError, InvalidSignature) as error:
            raise TrustError(
                f"the TSA certificate (SHA-256 {fingerprint(self.tsa)}) was not issued by the "
                f"root (SHA-256 {fingerprint(self.root)})"
            ) from error


def load_pinned(pem: bytes, expected_sha256: str, *, label: str) -> x509.Certificate:
    """The one certificate in `pem`, refused unless its fingerprint is `expected_sha256`."""
    try:
        certificates = x509.load_pem_x509_certificates(pem)
    except ValueError as error:
        raise TrustError(f"{label} is not a PEM certificate: {error}") from error
    if len(certificates) != 1:
        raise TrustError(f"{label} must hold exactly one certificate, found {len(certificates)}")
    (certificate,) = certificates
    actual = fingerprint(certificate)
    if actual != expected_sha256:
        raise TrustError(
            f"{label} is not the pinned certificate: its SHA-256 fingerprint is {actual} "
            f"but {expected_sha256} was expected"
        )
    return certificate


def _read(directory: Traversable, name: str) -> bytes:
    try:
        return (directory / name).read_bytes()
    except OSError as error:
        raise TrustError(f"cannot read the pinned certificate {name}: {error}") from error


def load_freetsa(directory: Traversable = CERTS_DIR) -> Trust:
    """FreeTSA's pinned pair from `directory`, refusing a file that is not the pinned one."""
    root = load_pinned(_read(directory, ROOT_FILE), FREETSA_ROOT_SHA256, label=ROOT_FILE)
    tsa = load_pinned(_read(directory, TSA_FILE), FREETSA_TSA_SHA256, label=TSA_FILE)
    return Trust(root=root, tsa=tsa)


DEFAULT_TRUST: Trust = load_freetsa()
"""The trust every verification uses unless it is handed another: FreeTSA's pinned pair."""
