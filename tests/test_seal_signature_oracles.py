"""The timestamp's signature against the real pins, with no test fixture (issue #25).

Everything here runs with the product's own trust: FreeTSA's root and TSA certificates, pinned in
the repository. No test in this file may request `trust_test_tsa` (the guard below fails it if
it does), because a verifier that accepted anything would pass every test that swaps the trust
for a fixture authority. The oracles are three:

1. the genuine response recorded from FreeTSA verifies against the pinned pair, offline;
2. the same response with its signature, its imprint or its time altered does not, and the
   reason names the signature;
3. a response from the throwaway test authority does not, and the reason names the chain.

The rest of this file is the same refusals one layer up (`verify` and `seal`), and the checks that
no trust can be added from outside.
"""

from __future__ import annotations

import ast
import datetime as dt
import inspect
from pathlib import Path

import pytest
import seal_helpers as helpers
import tsa_factory
from cryptography.hazmat.primitives.serialization import Encoding

from lastlight.cli import discover
from lastlight.seal import ledger as ledger_module
from lastlight.seal import ledger_paths, seal, tsa, verify
from lastlight.seal.tsa import TsaError

DATE = helpers.STUB_DATE
RECORDED_IMPRINT_HEX = "9cd91e89d63ac7a4633400eaba9dc69ffefb1b898a67f2c230700d2fb2378de6"
"""The SHA-256 the recorded FreeTSA response covers (`test_seal_tsa` derives it from the bytes)."""
SRC = Path(__file__).resolve().parents[1] / "src" / "lastlight"


@pytest.fixture(autouse=True)
def only_the_real_pins(request: pytest.FixtureRequest) -> None:
    assert "trust_test_tsa" not in request.fixturenames, (
        "a test of the real pins must not swap the trust for the test authority"
    )


def reject(der: bytes) -> tsa.TokenRejected:
    with pytest.raises(tsa.TokenRejected) as caught:
        tsa.authenticate(der)
    return caught.value


def write_sealed(ledger: Path, token: bytes) -> None:
    """Put the stub forecast and `token` where `seal` would, without going through `seal`."""
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    paths.forecast.write_bytes(helpers.STUB_CANONICAL)
    paths.timestamp.write_bytes(token)


# --- oracle 1: the genuine response, offline ----------------------------------------------


def test_the_recorded_freetsa_response_verifies_against_the_pinned_pair_offline() -> None:
    stamp = tsa.authenticate(helpers.recorded_response())  # the conftest guard blocks the network
    assert stamp.algorithm == "sha256"
    assert stamp.gen_time == helpers.recorded_gen_time()
    assert stamp.gen_time.utcoffset() == dt.timedelta(0)
    assert stamp.imprint.hex() == RECORDED_IMPRINT_HEX
    assert stamp.gen_time.date() == dt.date(2026, 10, 11)


# --- oracle 2: the same response, altered -------------------------------------------------


@pytest.mark.parametrize("from_the_end", [1, 10, 60])
def test_the_recorded_response_with_its_signature_altered_is_refused_naming_the_signature(
    from_the_end: int,
) -> None:
    # The last 103 bytes of the response are the ECDSA signature (an OCTET STRING at the end of
    # the SignerInfo); a bit flipped in its second integer (1, 10) or its first (60) leaves
    # every length and structure intact, so only the signature check can see it.
    raw = bytearray(helpers.recorded_response())
    raw[-from_the_end] ^= 0x01
    assert len(raw) == len(helpers.recorded_response())
    error = reject(bytes(raw))
    assert error.check == "signature"
    assert "signature" in str(error)


def test_the_recorded_response_with_its_imprint_edited_is_refused_naming_the_signature() -> None:
    forged = helpers.repointed_recorded_response()
    assert tsa.parse_response(forged).imprint.hex() == helpers.STUB_DIGEST  # parses, and lies
    error = reject(forged)
    assert error.check == "signature"
    assert "signature" in str(error)


def test_the_recorded_response_with_its_time_edited_is_refused_naming_the_signature() -> None:
    raw = helpers.recorded_response()
    old = helpers.recorded_gen_time().strftime("%Y%m%d%H%M%SZ").encode()
    new = (helpers.recorded_gen_time() - dt.timedelta(hours=1)).strftime("%Y%m%d%H%M%SZ").encode()
    assert raw.count(old) == 1
    error = reject(raw.replace(old, new))
    assert error.check == "signature"


def test_a_genuine_response_for_other_bytes_is_still_genuine() -> None:
    # The recorded response covers bytes that are not the stub's: authenticity says nothing
    # about what is covered, and the imprint is compared separately (digest mismatch).
    stamp = tsa.authenticate(helpers.recorded_response())
    assert stamp.imprint.hex() != helpers.STUB_DIGEST


# --- oracle 3: a token from a test authority, under the default anchors -------------------


def test_a_token_from_the_test_authority_is_refused_naming_the_certificate_chain() -> None:
    token = helpers.make_tsr(helpers.STUB_DIGEST, helpers.recorded_gen_time())
    error = reject(token)
    assert error.check == "chain"
    reason = str(error)
    assert "chain" in reason and "pinned" in reason
    assert "signature" not in reason, "the signature is fine; it is the authority that is unknown"


def test_a_token_from_a_second_unknown_authority_is_refused_naming_the_certificate_chain() -> None:
    token = tsa_factory.sign_response(
        bytes.fromhex(helpers.STUB_DIGEST),
        helpers.recorded_gen_time(),
        signer=tsa_factory.other_authority(),
    )
    assert reject(token).check == "chain"


def test_verify_refuses_a_ledger_whose_timestamp_is_from_the_test_authority(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger"
    write_sealed(ledger, helpers.tsr_before_sunset(DATE, helpers.STUB_DIGEST))
    outcome = verify(DATE, ledger)
    assert not outcome.ok
    reason = outcome.reason or ""
    assert "chain" in reason, reason
    assert outcome.strip_verified is False


def test_verify_refuses_a_ledger_whose_recorded_response_was_re_pointed_at_the_forecast(
    tmp_path: Path,
) -> None:
    # The forgery the issue describes: take a genuine response, patch its imprint to the sealed
    # forecast's digest. Every check but the signature passes.
    ledger = tmp_path / "ledger"
    write_sealed(ledger, helpers.repointed_recorded_response())
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "signature" in (outcome.reason or "")
    assert "digest mismatch" not in (outcome.reason or "")


def test_seal_refuses_a_reply_from_an_authority_that_is_not_pinned_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger"
    reply = helpers.tsr_before_sunset(DATE, helpers.STUB_DIGEST)
    fake = helpers.FakeTsa(reply)
    monkeypatch.setattr(tsa, "post_request", fake)
    with pytest.raises(TsaError, match="chain"):
        seal(helpers.stub_forecast(), ledger)
    assert len(fake.calls) == 1
    assert helpers.snapshot(ledger) == {}, "a reply that does not verify never enters the ledger"
    assert not ledger.exists()


def test_seal_refuses_the_re_pointed_recorded_response_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger"
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(helpers.repointed_recorded_response()))
    with pytest.raises(TsaError, match="signature"):
        seal(helpers.stub_forecast(), ledger)
    assert not ledger.exists()


# --- no trust from outside ----------------------------------------------------------------


def test_the_environment_cannot_add_trust(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    authority = tsa_factory.authority()
    root_pem = tmp_path / "root.pem"
    tsa_pem = tmp_path / "tsa.pem"
    root_pem.write_bytes(authority.root_certificate.public_bytes(Encoding.PEM))
    tsa_pem.write_bytes(authority.tsa_certificate.public_bytes(Encoding.PEM))
    for name in (
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "LASTLIGHT_TSA_CA",
        "LASTLIGHT_TSA_CERT",
        "LASTLIGHT_TSA_ROOT",
        "LASTLIGHT_TRUST",
        "LASTLIGHT_CA_FILE",
    ):
        monkeypatch.setenv(name, str(root_pem))
    token = helpers.make_tsr(helpers.STUB_DIGEST, helpers.recorded_gen_time())
    assert reject(token).check == "chain"


def test_no_module_of_the_seal_package_reads_the_environment_or_the_command_line() -> None:
    sources = sorted((SRC / "seal").glob("*.py")) + [
        SRC / "commands" / "seal.py",
        SRC / "commands" / "verify.py",
    ]
    assert len(sources) >= 7
    reads = {"environ", "environb", "getenv", "getenvb", "argv"}
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
            assert name not in reads, f"{path.name} line {node.lineno} reads {name}"


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("verify", {"-h", "--help", "--ledger-dir"}),
        ("seal", {"-h", "--help", "--strip", "--ledger-dir", "--tsa-url"}),
    ],
)
def test_the_commands_take_no_option_that_names_trust_anchors(name: str, options: set[str]) -> None:
    import argparse

    command = {command.name: command for command in discover()}[name]
    parser = argparse.ArgumentParser()
    command.configure(parser)
    assert {flag for action in parser._actions for flag in action.option_strings} == options


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (tsa, "authenticate"),
        (tsa, "request_timestamp"),
        (ledger_module, "seal"),
        (ledger_module, "verify"),
    ],
)
def test_trust_is_an_explicit_keyword_whose_default_is_the_module_default(
    module: object, name: str
) -> None:
    parameter = inspect.signature(getattr(module, name)).parameters["trust"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None, "None means the module default, resolved at call time"
