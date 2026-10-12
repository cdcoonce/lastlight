"""`verify` and `seal` check the timestamp's signature chain, not only its digest (issue #25).

These tests use the throwaway test authority (`tsa_factory`), trusted through the
`trust_test_tsa` fixture, so they can make a genuinely signed response for any bytes and any
time and then break it in exactly one way. Each failure must be a `FAIL` whose reason names the
check that failed: the signature, the certificate chain, the TSA certificate's extended key
usage, its validity at the token's time, the imprint, or the status. The oracles that need no
fixture (the real FreeTSA response against the real pins) are in `test_seal_signature_oracles`.
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import pytest
import seal_helpers as helpers
import tsa_factory
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509.oid import ExtendedKeyUsageOID

from lastlight.cli import main
from lastlight.forecast import Forecast
from lastlight.seal import ledger_paths, seal, trust, tsa, verify
from lastlight.seal.trust import Trust
from lastlight.seal.tsa import TokenRejected, TsaError

pytestmark = pytest.mark.usefixtures("trust_test_tsa")

DATE = helpers.STUB_DATE
DIGEST = helpers.STUB_DIGEST
UNKNOWN_KEY = tsa_factory.other_authority().tsa_key


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


def trust_of(authority: tsa_factory.Authority) -> Trust:
    return Trust(root=authority.root_certificate, tsa=authority.tsa_certificate)


def write_sealed(ledger: Path, token: bytes) -> None:
    """Put the stub forecast and `token` where `seal` would, without going through `seal`."""
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    paths.forecast.write_bytes(helpers.STUB_CANONICAL)
    paths.timestamp.write_bytes(token)


def verdict(ledger: Path, token: bytes) -> str | None:
    """Write `token` as the stub's timestamp; return the reason `verify` fails with, or None."""
    write_sealed(ledger, token)
    outcome = verify(DATE, ledger)
    return outcome.reason if not outcome.ok else None


def before_sunset(digest: str = DIGEST, **options: object) -> bytes:
    return helpers.make_tsr(digest, helpers.sunset(DATE) - dt.timedelta(minutes=30), **options)


# --- the pass ------------------------------------------------------------------------------


def test_a_genuinely_signed_token_for_the_sealed_bytes_passes(ledger: Path) -> None:
    write_sealed(ledger, before_sunset())
    outcome = verify(DATE, ledger)
    assert outcome.ok and outcome.reason is None


def test_authenticate_returns_what_the_token_says(trust_test_tsa: Trust) -> None:
    when = helpers.sunset(DATE) - dt.timedelta(hours=1, seconds=7)
    stamp = tsa.authenticate(helpers.make_tsr("ab" * 32, when))
    assert stamp == tsa.Timestamp(
        gen_time=when.replace(microsecond=0), imprint=bytes.fromhex("ab" * 32), algorithm="sha256"
    )
    assert tsa.authenticate(helpers.make_tsr("ab" * 32, when), trust=trust_test_tsa) == stamp


def test_a_token_without_embedded_certificates_is_checked_against_the_pinned_one(
    ledger: Path,
) -> None:
    assert verdict(ledger, before_sunset(include_certificates=False)) is None


def test_a_token_without_certificates_and_with_a_foreign_signature_fails_naming_the_signature(
    ledger: Path,
) -> None:
    reason = verdict(ledger, before_sunset(include_certificates=False, signing_key=UNKNOWN_KEY))
    assert reason is not None and "signature" in reason


def test_the_fixture_swaps_the_default_only_for_its_own_test(trust_test_tsa: Trust) -> None:
    assert trust.DEFAULT_TRUST is trust_test_tsa
    assert trust_test_tsa.tsa == tsa_factory.authority().tsa_certificate


# --- the signature -------------------------------------------------------------------------


def test_a_valid_structure_signed_by_a_different_key_fails_naming_the_signature(
    ledger: Path,
) -> None:
    reason = verdict(ledger, before_sunset(signing_key=UNKNOWN_KEY))
    assert reason is not None and "signature" in reason
    assert "digest mismatch" not in reason and "late seal" not in reason


def test_a_token_whose_imprint_was_edited_after_signing_fails_naming_the_signature(
    ledger: Path,
) -> None:
    other = "ee" * 32
    signed_for_other_bytes = before_sunset(other)
    assert signed_for_other_bytes.count(bytes.fromhex(other)) == 1
    forged = signed_for_other_bytes.replace(bytes.fromhex(other), bytes.fromhex(DIGEST))
    assert tsa.parse_response(forged).imprint.hex() == DIGEST  # the imprint check alone passes
    reason = verdict(ledger, forged)
    assert reason is not None and "signature" in reason
    assert "digest mismatch" not in reason


def test_a_late_token_with_its_time_edited_to_look_early_fails_naming_the_signature(
    ledger: Path,
) -> None:
    sunset = helpers.sunset(DATE).replace(microsecond=0)
    late = helpers.make_tsr(DIGEST, sunset + dt.timedelta(minutes=30))
    wanted = sunset - dt.timedelta(minutes=30)
    old, new = (
        t.strftime("%Y%m%d%H%M%SZ").encode() for t in (sunset + dt.timedelta(minutes=30), wanted)
    )
    assert late.count(old) == 1
    laundered = late.replace(old, new)
    assert tsa.parse_response(laundered).gen_time == wanted  # it parses as an early seal
    reason = verdict(ledger, laundered)
    assert reason is not None and "signature" in reason


@pytest.mark.parametrize("from_the_end", [1, 20])
def test_a_flipped_bit_in_the_signature_fails_naming_the_signature(
    ledger: Path, from_the_end: int
) -> None:
    token = bytearray(before_sunset())
    token[-from_the_end] ^= 0x01
    reason = verdict(ledger, bytes(token))
    assert reason is not None and "signature" in reason


def test_the_signature_failure_is_a_token_rejected_with_its_check_named() -> None:
    with pytest.raises(TokenRejected) as caught:
        tsa.authenticate(before_sunset(signing_key=UNKNOWN_KEY))
    assert caught.value.check == "signature"
    assert isinstance(caught.value, TsaError)
    assert str(caught.value).startswith("signature")


# --- the certificate chain -----------------------------------------------------------------


def test_a_token_from_an_authority_that_is_not_pinned_fails_naming_the_chain(ledger: Path) -> None:
    reason = verdict(ledger, before_sunset(signer=tsa_factory.other_authority()))
    assert reason is not None and "chain" in reason and "pinned" in reason


def test_a_forged_and_late_token_is_named_for_its_chain_not_its_lateness(ledger: Path) -> None:
    late = helpers.tsr_after_sunset(DATE, DIGEST, 30)
    forged = helpers.make_tsr(
        DIGEST,
        helpers.sunset(DATE) + dt.timedelta(minutes=30),
        signer=tsa_factory.other_authority(),
    )
    assert "late seal" in (verdict(ledger, late) or "")
    reason = verdict(ledger.parent / "second", forged)
    assert reason is not None and "chain" in reason and "late seal" not in reason


def test_a_forged_token_for_other_bytes_is_named_for_its_chain_not_its_digest(ledger: Path) -> None:
    reason = verdict(ledger, before_sunset("ee" * 32, signer=tsa_factory.other_authority()))
    assert reason is not None and "chain" in reason and "digest mismatch" not in reason


def test_a_certificate_that_copies_the_pinned_names_and_serial_is_refused_naming_the_chain(
    ledger: Path,
) -> None:
    # Same issuer name, subject name and serial number as the pinned TSA certificate, different
    # keys: the signer fields of the token point at "the pinned certificate", and only comparing
    # the certificate itself (not its name and serial) tells them apart.
    pinned = tsa_factory.authority().tsa_certificate
    impostor = tsa_factory.build_authority(tsa_serial=pinned.serial_number)
    assert impostor.tsa_certificate.issuer == pinned.issuer
    assert impostor.tsa_certificate.serial_number == pinned.serial_number
    assert impostor.tsa_certificate != pinned
    reason = verdict(ledger, before_sunset(signer=impostor))
    assert reason is not None and "chain" in reason and "pinned" in reason
    assert "signature" not in reason


def test_a_token_carrying_both_the_pinned_certificate_and_an_impostor_is_still_refused(
    ledger: Path,
) -> None:
    # The impostor has the pinned certificate's names and serial, so the signer fields fit both
    # certificates; the token is signed by the impostor's key and carries both. Whichever the
    # parser lists first, the signature cannot verify against the pinned key.
    pinned = tsa_factory.authority()
    impostor = tsa_factory.build_authority(tsa_serial=pinned.tsa_certificate.serial_number)
    token = before_sunset(
        signer=impostor, carried=(pinned.tsa_certificate, impostor.tsa_certificate)
    )
    reason = verdict(ledger, token)
    assert reason is not None and ("chain" in reason or "signature" in reason)


# --- the extended key usage ----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"key_purposes_critical": False},
        {"key_purposes": (ExtendedKeyUsageOID.CLIENT_AUTH,)},
        {"key_purposes": (ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.EMAIL_PROTECTION)},
    ],
    ids=["not-critical", "wrong-purpose", "no-timestamping-among-two"],
)
def test_a_tsa_certificate_without_a_critical_timestamping_purpose_fails_naming_it(
    ledger: Path, bad: dict[str, object]
) -> None:
    authority = tsa_factory.build_authority(**bad)  # type: ignore[arg-type]
    token = before_sunset(signer=authority)
    write_sealed(ledger, token)
    outcome = verify(DATE, ledger, trust=trust_of(authority))
    reason = outcome.reason or ""
    assert not outcome.ok
    assert "extended key usage" in reason and "EKU" in reason
    assert "signature" not in reason and "chain" not in reason


def test_the_same_authority_with_a_critical_timestamping_purpose_passes(ledger: Path) -> None:
    authority = tsa_factory.build_authority(name="good")
    write_sealed(ledger, before_sunset(signer=authority))
    assert verify(DATE, ledger, trust=trust_of(authority)).ok


def test_a_tsa_certificate_with_no_extended_key_usage_fails_naming_it(ledger: Path) -> None:
    authority = tsa_factory.build_authority(name="no eku", key_purposes=None)
    with pytest.raises(x509.ExtensionNotFound):  # the factory really left the extension out
        authority.tsa_certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    write_sealed(ledger, before_sunset(signer=authority))
    outcome = verify(DATE, ledger, trust=trust_of(authority))
    reason = outcome.reason or ""
    assert not outcome.ok
    assert "extended key usage" in reason and "EKU" in reason
    assert "signature" not in reason and "chain" not in reason


# --- validity is judged at the token's time, not today ------------------------------------

EXPIRED = tsa_factory.build_authority(name="expired", tsa_valid=tsa_factory.EXPIRED_TSA_VALID)


def issued(when: dt.datetime) -> bytes:
    return tsa_factory.sign_response(bytes.fromhex(DIGEST), when, signer=EXPIRED)


def test_the_certificate_the_tests_use_expired_before_today() -> None:
    assert EXPIRED.tsa_certificate.not_valid_after_utc < dt.datetime.now(dt.UTC)


def test_a_token_issued_while_the_certificate_was_valid_passes_after_it_expired() -> None:
    inside = dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    assert inside < EXPIRED.tsa_certificate.not_valid_after_utc < dt.datetime.now(dt.UTC)
    stamp = tsa.authenticate(issued(inside), trust=trust_of(EXPIRED))
    assert stamp.gen_time == inside


def test_a_token_issued_after_the_certificate_expired_fails_naming_the_validity() -> None:
    after = EXPIRED.tsa_certificate.not_valid_after_utc + dt.timedelta(days=30)
    with pytest.raises(TokenRejected) as caught:
        tsa.authenticate(issued(after), trust=trust_of(EXPIRED))
    assert caught.value.check == "validity"
    assert "validity" in str(caught.value) and "expired" in str(caught.value)


def test_a_token_issued_before_the_certificate_began_fails_naming_the_validity() -> None:
    before = EXPIRED.tsa_certificate.not_valid_before_utc - dt.timedelta(days=30)
    with pytest.raises(TokenRejected) as caught:
        tsa.authenticate(issued(before), trust=trust_of(EXPIRED))
    assert caught.value.check == "validity"
    assert "validity" in str(caught.value) and "not yet" in str(caught.value)


@pytest.mark.parametrize(("offset_seconds", "passes"), [(-1, True), (0, False), (1, False)])
def test_a_token_stamped_at_the_instant_the_certificate_ends_is_refused(
    offset_seconds: int, passes: bool
) -> None:
    # The last second before the end passes. The end instant itself is refused, as the library's
    # own check does, and the reason still names the validity.
    edge = EXPIRED.tsa_certificate.not_valid_after_utc.replace(microsecond=0)
    token = issued(edge + dt.timedelta(seconds=offset_seconds))
    if passes:
        tsa.authenticate(token, trust=trust_of(EXPIRED))
    else:
        with pytest.raises(TokenRejected, match="validity"):
            tsa.authenticate(token, trust=trust_of(EXPIRED))


@pytest.mark.parametrize(("offset_seconds", "passes"), [(-1, False), (0, True), (1, True)])
def test_the_certificate_is_valid_from_its_first_second(offset_seconds: int, passes: bool) -> None:
    edge = EXPIRED.tsa_certificate.not_valid_before_utc.replace(microsecond=0)
    token = issued(edge + dt.timedelta(seconds=offset_seconds))
    if passes:
        tsa.authenticate(token, trust=trust_of(EXPIRED))
    else:
        with pytest.raises(TokenRejected, match="validity"):
            tsa.authenticate(token, trust=trust_of(EXPIRED))


def test_verify_fails_a_seal_made_after_the_certificate_expired(ledger: Path) -> None:
    authority = tsa_factory.build_authority(
        name="ends before the evening",
        tsa_valid=(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 10, 1, tzinfo=dt.UTC)),
    )
    write_sealed(ledger, before_sunset(signer=authority))
    outcome = verify(DATE, ledger, trust=trust_of(authority))
    assert not outcome.ok and "validity" in (outcome.reason or "")


def test_a_token_issued_after_the_root_expired_fails_naming_the_validity() -> None:
    # The TSA certificate (valid through 2029) is fine at the token's time; the root it chains to
    # ended before. The reason must name the root's validity, not the library's generic refusal.
    root_ended = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    authority = tsa_factory.build_authority(
        name="root ends first",
        root_valid=(dt.datetime(2020, 1, 1, tzinfo=dt.UTC), root_ended),
    )
    after = root_ended + dt.timedelta(days=30)
    assert authority.tsa_certificate.not_valid_before_utc < after
    assert after < authority.tsa_certificate.not_valid_after_utc
    token = tsa_factory.sign_response(bytes.fromhex(DIGEST), after, signer=authority)
    with pytest.raises(TokenRejected) as caught:
        tsa.authenticate(token, trust=trust_of(authority))
    assert caught.value.check == "validity"
    reason = str(caught.value)
    assert "validity" in reason and "root" in reason and "expired" in reason
    assert "TSA" not in reason


def test_a_token_issued_before_the_root_began_fails_naming_the_validity() -> None:
    root_began = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    authority = tsa_factory.build_authority(
        name="root begins last",
        root_valid=(root_began, dt.datetime(2040, 1, 1, tzinfo=dt.UTC)),
    )
    before = root_began - dt.timedelta(days=30)
    assert authority.tsa_certificate.not_valid_before_utc < before
    token = tsa_factory.sign_response(bytes.fromhex(DIGEST), before, signer=authority)
    with pytest.raises(TokenRejected) as caught:
        tsa.authenticate(token, trust=trust_of(authority))
    assert caught.value.check == "validity"
    reason = str(caught.value)
    assert "validity" in reason and "root" in reason and "not yet" in reason
    assert "TSA" not in reason


# --- the imprint ---------------------------------------------------------------------------


def test_a_genuinely_signed_token_for_different_bytes_fails_naming_the_digest(ledger: Path) -> None:
    other = "ee" * 32
    reason = verdict(ledger, before_sunset(other))
    assert reason is not None and "digest mismatch" in reason
    assert DIGEST in reason and other in reason
    assert "signature" not in reason and "chain" not in reason


def test_request_timestamp_refuses_a_genuinely_signed_token_for_different_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(before_sunset("ee" * 32)))
    with pytest.raises(TsaError, match="imprint"):
        tsa.request_timestamp(helpers.STUB_CANONICAL, "https://tsa.example/tsr")


# --- the status ----------------------------------------------------------------------------


def test_a_response_granted_with_modifications_is_refused_by_the_parser() -> None:
    token = helpers.make_tsr(DIGEST, helpers.recorded_gen_time(), status=1)
    with pytest.raises(TsaError, match=r"status 1"):
        tsa.parse_response(token)


def test_verify_fails_a_response_granted_with_modifications_naming_the_status(
    ledger: Path,
) -> None:
    reason = verdict(ledger, before_sunset(status=1))
    assert reason is not None and "status" in reason


def test_seal_refuses_a_response_granted_with_modifications_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(before_sunset(status=1)))
    with pytest.raises(TsaError, match="status"):
        seal(helpers.stub_forecast(), ledger)
    assert not ledger.exists()


def test_a_granted_status_is_still_accepted() -> None:
    assert (
        tsa.parse_response(helpers.make_tsr(DIGEST, helpers.recorded_gen_time())).imprint.hex()
        == DIGEST
    )


# --- seal checks the reply before it writes anything ---------------------------------------

BAD_REPLIES = [
    pytest.param(lambda: before_sunset(signing_key=UNKNOWN_KEY), "signature", id="other-key"),
    pytest.param(lambda: before_sunset(signer=tsa_factory.other_authority()), "chain", id="chain"),
    pytest.param(lambda: before_sunset(status=1), "status", id="status-1"),
    pytest.param(lambda: before_sunset("ee" * 32), "imprint", id="other-bytes"),
    pytest.param(lambda: helpers.repointed_recorded_response(), "chain|signature", id="re-pointed"),
]


@pytest.mark.parametrize(("make_reply", "named"), BAD_REPLIES)
def test_seal_refuses_a_reply_that_does_not_verify_and_leaves_the_ledger_untouched(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, make_reply: object, named: str
) -> None:
    # An earlier day is already sealed; the refusal must not touch it, the index, or its folder.
    other = helpers.stub_dict()
    other["date"] = "2026-10-12"
    earlier = Forecast.model_validate(other)
    monkeypatch.setattr(
        tsa,
        "post_request",
        helpers.FakeTsa(helpers.tsr_before_sunset(earlier.date, helpers.digest_of(earlier))),
    )
    seal(earlier, ledger)
    before = helpers.snapshot(ledger)

    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(make_reply()))  # type: ignore[operator]
    with pytest.raises(TsaError, match=named):
        seal(helpers.stub_forecast(), ledger)
    assert helpers.snapshot(ledger) == before
    assert not ledger_paths(ledger, DATE).forecast.exists()
    assert not ledger_paths(ledger, DATE).timestamp.exists()


def test_seal_verifies_with_the_trust_it_is_given(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    authority = tsa_factory.build_authority(name="explicit")
    reply = before_sunset(signer=authority)
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(reply))
    with pytest.raises(TsaError, match="chain"):
        seal(helpers.stub_forecast(), ledger)  # the default trusts another authority
    result = seal(helpers.stub_forecast(), ledger, trust=trust_of(authority))
    assert result.paths.timestamp.read_bytes() == reply
    assert verify(DATE, ledger, trust=trust_of(authority)).ok


def test_a_reply_that_verifies_is_written_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    reply = before_sunset()
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(reply))
    result = seal(helpers.stub_forecast(), ledger)
    assert result.paths.timestamp.read_bytes() == reply
    assert verify(DATE, ledger).ok


# --- the command line ----------------------------------------------------------------------


def test_seal_exits_one_naming_the_signature_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    ledger: Path,
) -> None:
    forecast = tmp_path / "forecast.json"
    forecast.write_text(helpers.stub_text())
    monkeypatch.setattr(
        tsa, "post_request", helpers.FakeTsa(before_sunset(signing_key=UNKNOWN_KEY))
    )
    code = main(["seal", str(forecast), "--ledger-dir", str(ledger)])
    captured = capsys.readouterr()
    assert code == 1
    assert "signature" in captured.err and captured.out == ""
    assert not ledger.exists()


def test_verify_prints_one_fail_line_naming_the_signature(
    capsys: pytest.CaptureFixture[str], ledger: Path
) -> None:
    write_sealed(ledger, before_sunset(signing_key=UNKNOWN_KEY))
    code = main(["verify", DATE.isoformat(), "--ledger-dir", str(ledger)])
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("FAIL: ") and "signature" in out and out.count("\n") == 1


def test_verify_still_prints_pass_for_a_signed_token(
    capsys: pytest.CaptureFixture[str], ledger: Path
) -> None:
    write_sealed(ledger, before_sunset())
    assert main(["verify", DATE.isoformat(), "--ledger-dir", str(ledger)]) == 0
    assert capsys.readouterr().out == "PASS\n"


# --- the test authority is the real thing ---------------------------------------------------

OPENSSL = shutil.which("openssl")


def openssl_verifies(tmp_path: Path, token: bytes, digest: str) -> bool:
    authority = tsa_factory.authority()
    (tmp_path / "root.pem").write_bytes(authority.root_certificate.public_bytes(Encoding.PEM))
    (tmp_path / "tsa.pem").write_bytes(authority.tsa_certificate.public_bytes(Encoding.PEM))
    (tmp_path / "token.tsr").write_bytes(token)
    done = subprocess.run(
        [
            OPENSSL or "openssl",
            "ts",
            "-verify",
            "-in",
            str(tmp_path / "token.tsr"),
            "-digest",
            digest,
            "-CAfile",
            str(tmp_path / "root.pem"),
            "-untrusted",
            str(tmp_path / "tsa.pem"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return done.returncode == 0 and "Verification: OK" in done.stdout + done.stderr


@pytest.mark.skipif(OPENSSL is None, reason="needs the openssl command")
def test_an_independent_verifier_accepts_the_fixtures_and_refuses_a_forgery(tmp_path: Path) -> None:
    # The fixtures only mean something if they are real tokens: `openssl ts -verify` checks the
    # signature, the chain and the imprint of what `tsa_factory` writes, and refuses it once the
    # imprint is edited. Skipped where the command is missing or has no `ts` support.
    probe = subprocess.run(
        [OPENSSL or "openssl", "ts", "-help"], capture_output=True, text=True, check=False
    )
    if probe.returncode not in (0, 1) or "-verify" not in probe.stdout + probe.stderr:
        pytest.skip("this openssl has no `ts -verify`")
    token = before_sunset()
    assert openssl_verifies(tmp_path, token, DIGEST)
    forged = before_sunset("ee" * 32).replace(bytes.fromhex("ee" * 32), bytes.fromhex(DIGEST))
    assert not openssl_verifies(tmp_path, forged, DIGEST)
