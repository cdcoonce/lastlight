"""The pinned FreeTSA certificates: what they are, where they came from, and the loader (issue #25).

The pins are two PEM files under `src/lastlight/seal/certs/`, FreeTSA's root CA and its TSA
certificate, with a README that says where each was fetched from. The SHA-256 fingerprints below
were taken from those downloads on the day they were fetched, and are written here a second time
(the code holds them as constants) so that changing a pin takes a visible edit in two places. The
certificates are also byte-identical to the two that FreeTSA embedded in the response recorded in
`tests/fixtures/seal/stub-forecast.tsr`, which was recorded from the service itself.

No test here requests `trust_test_tsa`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

import pytest
import seal_helpers as helpers
import tsa_factory
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from rfc3161_client import decode_timestamp_response

from lastlight.privacy import contains_clock_time
from lastlight.privacy.rules import text_hits
from lastlight.seal import trust
from lastlight.seal.trust import Trust, TrustError

ROOT_SHA256 = "a6379e7cecc05faa3cbf076013d745e327bbbaa38c0b9af22469d4701d18aabc"
TSA_SHA256 = "32e841a95cc1164101ffde41298ef2fc75c1c4372ef095e88a6bbd47dfb191fc"
ROOT_URL = "https://freetsa.org/files/cacert.pem"
TSA_URL = "https://freetsa.org/files/tsa.crt"
ROOT_VALID = (dt.date(2016, 3, 13), dt.date(2041, 3, 7))
TSA_VALID = (dt.date(2026, 2, 15), dt.date(2040, 2, 2))


@pytest.fixture(autouse=True)
def only_the_real_pins(request: pytest.FixtureRequest) -> None:
    assert "trust_test_tsa" not in request.fixturenames


def der(certificate: x509.Certificate) -> bytes:
    return certificate.public_bytes(Encoding.DER)


def sha256(certificate: x509.Certificate) -> str:
    return hashlib.sha256(der(certificate)).hexdigest()


def embedded_in_the_recorded_response() -> list[x509.Certificate]:
    signed = decode_timestamp_response(helpers.recorded_response()).signed_data
    return [x509.load_der_x509_certificate(raw) for raw in signed.certificates]


def readme() -> str:
    return (trust.CERTS_DIR / "README.md").read_text()


# --- what is pinned -----------------------------------------------------------------------


def test_the_default_trust_is_the_pinned_freetsa_pair() -> None:
    assert sha256(trust.DEFAULT_TRUST.root) == ROOT_SHA256
    assert sha256(trust.DEFAULT_TRUST.tsa) == TSA_SHA256


def test_the_fingerprints_in_code_are_the_ones_written_down_at_fetch_time() -> None:
    assert trust.FREETSA_ROOT_SHA256 == ROOT_SHA256
    assert trust.FREETSA_TSA_SHA256 == TSA_SHA256


def test_the_pinned_certificates_are_byte_identical_to_those_in_the_recorded_response() -> None:
    embedded = {der(certificate) for certificate in embedded_in_the_recorded_response()}
    assert len(embedded) == 2
    assert {der(trust.DEFAULT_TRUST.root), der(trust.DEFAULT_TRUST.tsa)} == embedded


def test_the_tsa_certificate_was_issued_by_the_pinned_root() -> None:
    trust.DEFAULT_TRUST.tsa.verify_directly_issued_by(trust.DEFAULT_TRUST.root)
    assert trust.DEFAULT_TRUST.tsa.issuer == trust.DEFAULT_TRUST.root.subject
    assert trust.DEFAULT_TRUST.root.issuer == trust.DEFAULT_TRUST.root.subject, "self-signed"


def test_the_pinned_tsa_certificate_carries_a_critical_timestamping_purpose() -> None:
    extension = trust.DEFAULT_TRUST.tsa.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    assert extension.critical
    assert list(extension.value) == [x509.ExtendedKeyUsageOID.TIME_STAMPING]


def test_the_pinned_validity_windows_are_the_ones_the_readme_reports() -> None:
    root, pinned_tsa = trust.DEFAULT_TRUST.root, trust.DEFAULT_TRUST.tsa
    assert (root.not_valid_before_utc.date(), root.not_valid_after_utc.date()) == ROOT_VALID
    assert (pinned_tsa.not_valid_before_utc.date(), pinned_tsa.not_valid_after_utc.date()) == (
        TSA_VALID
    )


def test_the_recorded_response_was_issued_inside_both_validity_windows() -> None:
    issued = helpers.recorded_gen_time()
    for certificate in (trust.DEFAULT_TRUST.root, trust.DEFAULT_TRUST.tsa):
        assert certificate.not_valid_before_utc <= issued <= certificate.not_valid_after_utc


def test_each_pem_file_holds_exactly_one_certificate() -> None:
    for name in (trust.ROOT_FILE, trust.TSA_FILE):
        text = (trust.CERTS_DIR / name).read_text()
        assert text.count("-----BEGIN CERTIFICATE-----") == 1, name
        assert text.count("-----END CERTIFICATE-----") == 1, name
        assert text.isascii(), name
        assert len(x509.load_pem_x509_certificates(text.encode())) == 1, name


# --- the README says where they came from -------------------------------------------------


def test_the_readme_gives_the_source_urls_and_both_fingerprints() -> None:
    text = readme()
    for needle in (ROOT_URL, TSA_URL, ROOT_SHA256, TSA_SHA256):
        assert needle in text, needle


def test_the_readme_gives_both_validity_windows_as_dates() -> None:
    text = readme()
    for date in (*ROOT_VALID, *TSA_VALID):
        assert date.isoformat() in text, date


def test_the_readme_gives_the_fetch_date() -> None:
    assert re.search(r"[Ff]etched\b.*\b2026-10-11\b", readme())


def test_the_readme_spells_no_clock_time_or_position() -> None:
    text = readme()
    assert not contains_clock_time(text)
    assert text_hits(text) == []


# --- the loader refuses a swapped file ----------------------------------------------------


def real_pem(name: str) -> bytes:
    return (trust.CERTS_DIR / name).read_bytes()


def test_load_pinned_accepts_the_pem_whose_fingerprint_it_expects() -> None:
    loaded = trust.load_pinned(real_pem(trust.ROOT_FILE), ROOT_SHA256, label="root")
    assert sha256(loaded) == ROOT_SHA256


def test_load_pinned_refuses_a_pem_with_another_fingerprint() -> None:
    swapped = tsa_factory.authority().root_certificate.public_bytes(Encoding.PEM)
    with pytest.raises(TrustError, match="fingerprint") as caught:
        trust.load_pinned(swapped, ROOT_SHA256, label="root")
    assert ROOT_SHA256 in str(caught.value), "the message gives the fingerprint it expected"
    assert "root" in str(caught.value)


def test_load_pinned_refuses_the_other_pinned_certificate_in_the_wrong_slot() -> None:
    with pytest.raises(TrustError, match="fingerprint"):
        trust.load_pinned(real_pem(trust.TSA_FILE), ROOT_SHA256, label="root")


@pytest.mark.parametrize(
    "pem",
    [
        b"",
        b"not a certificate",
        b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n",
    ],
)
def test_load_pinned_refuses_a_file_that_is_not_a_certificate(pem: bytes) -> None:
    with pytest.raises(TrustError):
        trust.load_pinned(pem, ROOT_SHA256, label="root")


def test_load_pinned_refuses_a_bundle_even_when_the_expected_certificate_is_in_it() -> None:
    bundle = real_pem(trust.ROOT_FILE) + real_pem(trust.TSA_FILE)
    with pytest.raises(TrustError, match="exactly one"):
        trust.load_pinned(bundle, ROOT_SHA256, label="root")


def copy_certs(destination: Path) -> Path:
    destination.mkdir()
    for name in (trust.ROOT_FILE, trust.TSA_FILE):
        (destination / name).write_bytes(real_pem(name))
    return destination


def test_load_freetsa_reads_an_untouched_copy_of_the_pins(tmp_path: Path) -> None:
    loaded = trust.load_freetsa(copy_certs(tmp_path / "certs"))
    assert (der(loaded.root), der(loaded.tsa)) == (
        der(trust.DEFAULT_TRUST.root),
        der(trust.DEFAULT_TRUST.tsa),
    )


@pytest.mark.parametrize("swapped", [trust.ROOT_FILE, trust.TSA_FILE])
def test_load_freetsa_refuses_a_directory_where_a_pem_was_swapped(
    tmp_path: Path, swapped: str
) -> None:
    certs = copy_certs(tmp_path / "certs")
    authority = tsa_factory.authority()
    replacement = (
        authority.root_certificate if swapped == trust.ROOT_FILE else authority.tsa_certificate
    )
    (certs / swapped).write_bytes(replacement.public_bytes(Encoding.PEM))
    with pytest.raises(TrustError, match="fingerprint") as caught:
        trust.load_freetsa(certs)
    assert swapped in str(caught.value), "the message names the file that was swapped"


def test_load_freetsa_refuses_a_directory_missing_a_pem(tmp_path: Path) -> None:
    certs = copy_certs(tmp_path / "certs")
    (certs / trust.TSA_FILE).unlink()
    with pytest.raises(TrustError, match=re.escape(trust.TSA_FILE)):
        trust.load_freetsa(certs)


def test_a_pair_must_be_issued_by_its_root() -> None:
    authority, other = tsa_factory.authority(), tsa_factory.other_authority()
    Trust(root=authority.root_certificate, tsa=authority.tsa_certificate)  # the good pair loads
    with pytest.raises(TrustError, match="issued by"):
        Trust(root=authority.root_certificate, tsa=other.tsa_certificate)
    with pytest.raises(TrustError, match="issued by"):
        Trust(root=trust.DEFAULT_TRUST.root, tsa=authority.tsa_certificate)
