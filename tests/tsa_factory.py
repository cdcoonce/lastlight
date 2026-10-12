"""A throwaway timestamp authority for the tests: a root CA, a TSA certificate, signed responses.

The tests need RFC 3161 responses for any bytes and any `gen_time`, and a real response cannot
be made on demand (the public authority is the network, and its signature covers exactly one
time and one digest). So this module plays the authority: it generates a root and a TSA
certificate with `cryptography` and builds genuinely signed responses with a small DER encoder.
The signature is real: a fixture that is edited after signing (its time, its imprint, its
signature) no longer verifies, exactly as a forged reply would not.

The certificates follow the profile of the public authority's: a self-signed CA root, and a TSA
certificate issued by it with a critical `id-kp-timeStamping` extended key usage. The response
follows the RFC 3161 shape the public authority sends: a `TimeStampResp` whose token is a CMS
`SignedData` over a `TSTInfo`, with the signed attributes content type, message digest and
signing-certificate-v2, the TSA certificate and the root in the certificate set, and an ECDSA
signature by the TSA key.

Nothing here is trusted by the product: the product trusts only the pinned public authority
(`lastlight.seal.trust`). The `trust_test_tsa` fixture in `conftest.py` is the one place a test
swaps that for the authority built here.

Times are whole seconds, the only precision the encoder writes. Signing is deterministic: the
same imprint, time and options give the same bytes within a test run.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

SHA256_OID = "2.16.840.1.101.3.4.2.1"
ECDSA_SHA256_OID = "1.2.840.10045.4.3.2"
SIGNED_DATA_OID = "1.2.840.113549.1.7.2"
TST_INFO_OID = "1.2.840.113549.1.9.16.1.4"
CONTENT_TYPE_OID = "1.2.840.113549.1.9.3"
MESSAGE_DIGEST_OID = "1.2.840.113549.1.9.4"
SIGNING_CERTIFICATE_V2_OID = "1.2.840.113549.1.9.16.2.47"
POLICY_OID = "1.2.3.4.1"  # the policy arc the public authority uses; the tests do not look at it

ROOT_VALID = (dt.datetime(2020, 1, 1, tzinfo=dt.UTC), dt.datetime(2040, 1, 1, tzinfo=dt.UTC))
TSA_VALID = (dt.datetime(2026, 1, 1, tzinfo=dt.UTC), dt.datetime(2030, 1, 1, tzinfo=dt.UTC))
"""The default validity windows. The TSA certificate is valid for four years from the start of
2026, comfortably around every time the fixtures use."""

EXPIRED_TSA_VALID = (dt.datetime(2026, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
"""A TSA certificate that ended before the evenings under test, for the expiry tests."""


# --- DER ----------------------------------------------------------------------------------


def _tlv(tag: int, body: bytes) -> bytes:
    size = len(body)
    if size < 0x80:
        length = bytes([size])
    else:
        raw = size.to_bytes((size.bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(raw)]) + raw
    return bytes([tag]) + length + body


def sequence(*items: bytes) -> bytes:
    return _tlv(0x30, b"".join(items))


def set_of(*items: bytes) -> bytes:
    return _tlv(0x31, b"".join(sorted(items)))  # DER: the members of a SET OF in encoded order


def integer(value: int) -> bytes:
    size = max(1, (value.bit_length() + 8) // 8)  # room for the sign bit
    return _tlv(0x02, value.to_bytes(size, "big", signed=True))


def octet_string(data: bytes) -> bytes:
    return _tlv(0x04, data)


def null() -> bytes:
    return b"\x05\x00"


def boolean(value: bool) -> bytes:
    return _tlv(0x01, b"\xff" if value else b"\x00")


def explicit(number: int, body: bytes) -> bytes:
    """`[number] EXPLICIT`, or `[number] IMPLICIT` over a constructed type: the body is wrapped."""
    return _tlv(0xA0 | number, body)


def oid(dotted: str) -> bytes:
    arcs = [int(part) for part in dotted.split(".")]
    body = bytearray([40 * arcs[0] + arcs[1]])
    for arc in arcs[2:]:
        chunk = [arc & 0x7F]
        arc >>= 7
        while arc:
            chunk.append(0x80 | (arc & 0x7F))
            arc >>= 7
        body.extend(reversed(chunk))
    return _tlv(0x06, bytes(body))


def generalized_time(moment: dt.datetime) -> bytes:
    text = moment.astimezone(dt.UTC).strftime("%Y%m%d%H%M%SZ")
    return _tlv(0x18, text.encode("ascii"))


def algorithm(identifier: str, *, parameters: bool = True) -> bytes:
    return sequence(oid(identifier), *([null()] if parameters else []))


# --- the authority ------------------------------------------------------------------------


@dataclass(frozen=True)
class Authority:
    """A root CA and the TSA certificate it issued, with both private keys."""

    root_key: ec.EllipticCurvePrivateKey
    root_certificate: x509.Certificate
    tsa_key: ec.EllipticCurvePrivateKey
    tsa_certificate: x509.Certificate


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "lastlight tests"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def build_authority(
    *,
    name: str = "lastlight test",
    root_valid: tuple[dt.datetime, dt.datetime] = ROOT_VALID,
    tsa_valid: tuple[dt.datetime, dt.datetime] = TSA_VALID,
    key_purposes: tuple[x509.ObjectIdentifier, ...] | None = (ExtendedKeyUsageOID.TIME_STAMPING,),
    key_purposes_critical: bool = True,
    tsa_serial: int | None = None,
) -> Authority:
    """A fresh root and TSA certificate; the arguments make the certificate wrong in one way.

    The same `name` gives the same subject and issuer names, and `tsa_serial` fixes the TSA
    certificate's serial number, so an authority can impersonate another one's names while its
    keys, and so its certificates, are different. `root_valid` and `tsa_valid` are the two
    validity windows, and `key_purposes=None` leaves the extended key usage extension out of the
    TSA certificate altogether.
    """
    root_key = ec.generate_private_key(ec.SECP256R1())
    tsa_key = ec.generate_private_key(ec.SECP256R1())
    root_name, tsa_name = _name(f"{name} root"), _name(f"{name} tsa")
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(root_valid[0])
        .not_valid_after(root_valid[1])
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), False)
        .sign(root_key, hashes.SHA256())
    )
    tsa_builder = (
        x509.CertificateBuilder()
        .subject_name(tsa_name)
        .issuer_name(root_name)
        .public_key(tsa_key.public_key())
        .serial_number(tsa_serial or x509.random_serial_number())
        .not_valid_before(tsa_valid[0])
        .not_valid_after(tsa_valid[1])
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=False,
        )
    )
    if key_purposes is not None:
        tsa_builder = tsa_builder.add_extension(
            x509.ExtendedKeyUsage(list(key_purposes)), critical=key_purposes_critical
        )
    tsa = (
        tsa_builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(tsa_key.public_key()), False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), False
        )
        .sign(root_key, hashes.SHA256())
    )
    return Authority(root_key=root_key, root_certificate=root, tsa_key=tsa_key, tsa_certificate=tsa)


@functools.cache
def authority() -> Authority:
    """The authority the shared fixtures use, generated once per test run."""
    return build_authority()


@functools.cache
def other_authority() -> Authority:
    """A second authority nobody trusts: a response it signs chains to a root not pinned."""
    return build_authority(name="lastlight other")


# --- responses ----------------------------------------------------------------------------


def _timestamp_info(imprint: bytes, gen_time: dt.datetime, serial: int, hash_oid: str) -> bytes:
    return sequence(
        integer(1),
        oid(POLICY_OID),
        sequence(algorithm(hash_oid), octet_string(imprint)),
        integer(serial),
        generalized_time(gen_time),
    )


def _signed_attributes(info: bytes, signer: x509.Certificate) -> list[bytes]:
    certificate_hash = hashlib.sha256(signer.public_bytes(Encoding.DER)).digest()
    return [
        sequence(oid(CONTENT_TYPE_OID), set_of(oid(TST_INFO_OID))),
        sequence(oid(MESSAGE_DIGEST_OID), set_of(octet_string(hashlib.sha256(info).digest()))),
        sequence(
            oid(SIGNING_CERTIFICATE_V2_OID),
            # SigningCertificateV2 { certs { ESSCertIDv2 { certHash } } }; sha256 is the default
            set_of(sequence(sequence(sequence(octet_string(certificate_hash))))),
        ),
    ]


def sign_response(
    imprint: bytes,
    gen_time: dt.datetime,
    *,
    signer: Authority | None = None,
    signing_key: ec.EllipticCurvePrivateKey | None = None,
    include_certificates: bool = True,
    carried: tuple[x509.Certificate, ...] | None = None,
    status: int = 0,
    serial: int = 1,
    hash_oid: str = SHA256_OID,
) -> bytes:
    """A DER `TimeStampResp` for `imprint`, issued at `gen_time` by `signer`.

    `signing_key` signs in place of the authority's TSA key, which makes a token with a valid
    structure and certificates whose signature belongs to someone else. `carried` replaces the
    certificates the token carries (the signer's TSA certificate and root by default), and
    `include_certificates=False` carries none. `status` other than 0
    and 1 gives a refusal with no token; 1 gives a token the authority "granted with
    modifications".
    """
    signer = signer or authority()
    if status not in (0, 1):
        return sequence(sequence(integer(status)))
    info = _timestamp_info(imprint, gen_time, serial, hash_oid)
    attributes = _signed_attributes(info, signer.tsa_certificate)
    signed_set = set_of(*attributes)
    # Deterministic (RFC 6979), so the same arguments give the same bytes: tests compare a
    # written `.tsr` with the reply it came from.
    scheme = ec.ECDSA(hashes.SHA256(), deterministic_signing=True)
    signature = (signing_key or signer.tsa_key).sign(signed_set, scheme)
    carried = carried or (signer.tsa_certificate, signer.root_certificate)
    certificates = b"".join(  # a SET OF: DER orders the members by their encoding
        sorted(c.public_bytes(Encoding.DER) for c in carried)
    )
    signer_info = sequence(
        integer(1),
        sequence(
            signer.tsa_certificate.issuer.public_bytes(),
            integer(signer.tsa_certificate.serial_number),
        ),
        algorithm(SHA256_OID),
        explicit(0, b"".join(sorted(attributes))),  # [0] IMPLICIT SET OF: same bytes, other tag
        algorithm(ECDSA_SHA256_OID, parameters=False),
        octet_string(signature),
    )
    signed_data = sequence(
        integer(3),
        set_of(algorithm(SHA256_OID)),
        sequence(oid(TST_INFO_OID), explicit(0, octet_string(info))),
        *([explicit(0, certificates)] if include_certificates else []),
        set_of(signer_info),
    )
    token = sequence(oid(SIGNED_DATA_OID), explicit(0, signed_data))
    return sequence(sequence(integer(status)), token)


def make_tsr(digest_hex: str, gen_time: dt.datetime, **options: object) -> bytes:
    """A genuinely signed response (by the shared test authority) covering `digest_hex`.

    `options` go to `sign_response`.
    """
    return sign_response(bytes.fromhex(digest_hex), gen_time, **options)  # type: ignore[arg-type]
