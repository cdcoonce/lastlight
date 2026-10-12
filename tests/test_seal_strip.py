"""The painting is sealed with the forecast: `strip_sha256` in the forecast, the PNG in the ledger.

The headline claim is "painted before it happens". If only the numbers were sealed, the picture
anyone looks at could be swapped afterwards, so the forecast carries the SHA-256 of the strip and
`verify` checks the PNG beside it against that digest (issue #8).
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import image_factory
import pytest
import seal_helpers as helpers

from lastlight.forecast import Forecast
from lastlight.privacy.images import image_issues
from lastlight.seal import tsa
from lastlight.seal.ledger import (
    AlreadySealedError,
    StripError,
    VerifyResult,
    is_sealed,
    ledger_paths,
    seal,
    verify,
)
from lastlight.seal.tsa import TsaError

pytestmark = pytest.mark.usefixtures("trust_test_tsa")

DATE = helpers.STUB_DATE
TSA_URL = "https://tsa.example/tsr"


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def forecast_with(strip: bytes | None) -> Forecast:
    data = helpers.stub_dict()
    data["strip_sha256"] = None if strip is None else sha256(strip)
    return Forecast.model_validate(data)


def install_for(monkeypatch: pytest.MonkeyPatch, forecast: Forecast) -> helpers.FakeTsa:
    """A TSA that stamps exactly this forecast's digest half an hour before sunset."""
    reply = helpers.tsr_before_sunset(forecast.date, helpers.digest_of(forecast))
    fake = helpers.FakeTsa(reply)
    monkeypatch.setattr(tsa, "post_request", fake)
    return fake


def sealed(monkeypatch: pytest.MonkeyPatch, ledger: Path, strip: bytes) -> Forecast:
    forecast = forecast_with(strip)
    install_for(monkeypatch, forecast)
    seal(forecast, ledger, strip=strip, tsa_url=TSA_URL)
    return forecast


# --- layout -------------------------------------------------------------------------------


def test_the_strip_sits_beside_its_forecast_in_the_same_folder(ledger: Path) -> None:
    paths = ledger_paths(ledger, dt.date(2026, 3, 7))
    assert paths.strip == ledger / "2026" / "03" / "07.strip.png"
    assert paths.strip.parent == paths.forecast.parent


# --- seal ---------------------------------------------------------------------------------


def test_seal_writes_the_strip_byte_for_byte_beside_the_forecast(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    png = image_factory.plain_png()
    forecast = forecast_with(png)
    install_for(monkeypatch, forecast)

    result = seal(forecast, ledger, strip=png, tsa_url=TSA_URL)

    assert result.paths.strip.read_bytes() == png
    assert result.paths.forecast.is_file() and result.paths.timestamp.is_file()


def test_the_timestamp_covers_a_forecast_that_names_the_strip(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    png = image_factory.plain_png()
    forecast = forecast_with(png)
    fake = install_for(monkeypatch, forecast)
    seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    assert fake.calls == [
        (TSA_URL, tsa.build_request(ledger_paths(ledger, DATE).forecast.read_bytes()))
    ]
    assert sha256(png).encode() in ledger_paths(ledger, DATE).forecast.read_bytes()


def test_a_forecast_without_a_strip_seals_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    forecast = forecast_with(None)
    install_for(monkeypatch, forecast)
    result = seal(forecast, ledger, tsa_url=TSA_URL)
    assert not result.paths.strip.exists()
    assert result.paths.forecast.read_bytes() == helpers.STUB_CANONICAL


def refusal_cases() -> list[tuple[str, bytes | None, bytes | None]]:
    """(name, strip declared by the forecast, strip handed to seal)."""
    good = image_factory.plain_png()
    other = image_factory.png_with_text("Comment", "x")  # a different, metadata-carrying PNG
    return [
        ("declared-but-not-given", good, None),
        ("given-but-not-declared", None, good),
        ("different-bytes", good, image_factory.plain_png()[:-1] + b"\x00"),
        ("digest-of-another-png", good, other),
    ]


@pytest.mark.parametrize(("name", "declared", "given"), refusal_cases())
def test_a_strip_that_is_not_the_one_the_forecast_names_is_refused_before_any_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    ledger: Path,
    name: str,
    declared: bytes | None,
    given: bytes | None,
) -> None:
    forecast = forecast_with(declared)
    fake = install_for(monkeypatch, forecast)
    with pytest.raises(StripError):
        seal(forecast, ledger, strip=given, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == [], "a refused seal must not spend a timestamp"


@pytest.mark.parametrize(
    "png",
    [
        image_factory.png_with_text("Comment", "locating text"),
        image_factory.png_with_text("Comment", "locating text", kind="iTXt"),
        image_factory.png_with_chunk(b"eXIf"),
        image_factory.png_with_chunk(b"iCCP"),
        image_factory.png_with_chunk(b"tIME"),
        b"not a png at all",
        b"",
    ],
)
def test_a_strip_the_privacy_scanner_would_flag_is_never_sealed(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, png: bytes
) -> None:
    # The ledger is append-only and privacy-scan runs over it: a sealed leak could not be taken
    # back. The digest matches the file here, so only the metadata check can refuse it.
    forecast = forecast_with(png)
    fake = install_for(monkeypatch, forecast)
    with pytest.raises(StripError, match="strip"):
        seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == []


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(image_factory.plain_jpeg(), id="jpeg"),
        pytest.param(image_factory.plain_webp(), id="webp"),
        pytest.param(image_factory.plain_tiff(), id="tiff"),
    ],
)
def test_a_metadata_free_image_that_is_not_a_png_is_never_sealed_as_the_strip(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, image: bytes
) -> None:
    # The ledger names the painting `DD.strip.png` and the site serves it as a PNG. The scanner
    # finds nothing in these images, and the digest matches the bytes, so only the format check
    # can refuse them.
    assert image_issues(image, ".png") == [], "the fixture must be clean to the scanner"
    forecast = forecast_with(image)
    fake = install_for(monkeypatch, forecast)
    with pytest.raises(StripError, match="not a PNG image"):
        seal(forecast, ledger, strip=image, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == []


@pytest.mark.parametrize("junk", [b"not a png at all", b""])
def test_bytes_that_are_no_image_at_all_are_refused_as_not_a_png(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, junk: bytes
) -> None:
    forecast = forecast_with(junk)
    install_for(monkeypatch, forecast)
    with pytest.raises(StripError, match="not a PNG image"):
        seal(forecast, ledger, strip=junk, tsa_url=TSA_URL)


@pytest.mark.parametrize("existing", ["forecast", "timestamp", "strip", "index"])
def test_a_date_with_any_trace_of_a_seal_is_refused_with_a_strip_too(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, existing: str
) -> None:
    png = image_factory.plain_png()
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    if existing == "index":
        paths.index.write_text('{"date":"2026-10-11","digest":"00","model_version":"x"}\n')
    else:
        getattr(paths, existing).write_bytes(b"theirs")
    before = helpers.snapshot(ledger)
    forecast = forecast_with(png)
    fake = install_for(monkeypatch, forecast)
    with pytest.raises(AlreadySealedError, match="2026-10-11"):
        seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == before
    assert fake.calls == []


def test_a_tsa_failure_writes_no_strip_either(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    def down(url: str, request: bytes, **_options: object) -> bytes:
        raise TsaError("the timestamp authority is down")

    monkeypatch.setattr(tsa, "post_request", down)
    png = image_factory.plain_png()
    with pytest.raises(TsaError):
        seal(forecast_with(png), ledger, strip=png, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}


def test_a_failed_write_removes_the_strip_it_had_written(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    ledger.mkdir()
    (ledger / "index.jsonl").mkdir()  # appending to a directory fails after the files exist
    before = helpers.snapshot(ledger)
    png = image_factory.plain_png()
    forecast = forecast_with(png)
    install_for(monkeypatch, forecast)
    with pytest.raises(OSError, match="index"):
        seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    files = {key for key, value in helpers.snapshot(ledger).items() if value is not None}
    assert files == {key for key, value in before.items() if value is not None}
    assert not ledger_paths(ledger, DATE).strip.exists()


class RacingTsa(helpers.FakeTsa):
    """A reply that arrives after another process has started sealing the date's strip."""

    def __init__(self, reply: bytes, *, create: Path) -> None:
        super().__init__(reply)
        self.create = create

    def __call__(self, url: str, request: bytes, **options: object) -> bytes:
        self.create.parent.mkdir(parents=True, exist_ok=True)
        self.create.write_bytes(b"theirs")
        return super().__call__(url, request, **options)


def test_a_seal_that_loses_the_race_for_the_strip_removes_only_its_own_files(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    png = image_factory.plain_png()
    forecast = forecast_with(png)
    paths = ledger_paths(ledger, DATE)
    reply = helpers.tsr_before_sunset(DATE, helpers.digest_of(forecast))
    monkeypatch.setattr(tsa, "post_request", RacingTsa(reply, create=paths.strip))
    with pytest.raises(AlreadySealedError, match="2026-10-11"):
        seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    assert paths.strip.read_bytes() == b"theirs"
    assert not paths.forecast.exists() and not paths.timestamp.exists()
    assert not paths.index.exists()


# --- is_sealed ----------------------------------------------------------------------------


def test_is_sealed_is_false_for_an_empty_ledger_and_true_after_a_seal(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    assert not is_sealed(DATE, ledger)
    sealed(monkeypatch, ledger, image_factory.plain_png())
    assert is_sealed(DATE, ledger)
    assert not is_sealed(DATE + dt.timedelta(days=1), ledger)


@pytest.mark.parametrize("trace", ["forecast", "timestamp", "strip", "index"])
def test_any_trace_of_a_seal_counts_as_sealed(ledger: Path, trace: str) -> None:
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    if trace == "index":
        paths.index.write_text('{"date":"2026-10-11","digest":"00","model_version":"x"}\n')
    else:
        getattr(paths, trace).write_bytes(b"x")
    assert is_sealed(DATE, ledger)


# --- verify -------------------------------------------------------------------------------


def test_verify_passes_and_says_the_strip_was_verified(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed(monkeypatch, ledger, image_factory.plain_png())
    assert verify(DATE, ledger) == VerifyResult(ok=True, reason=None, strip_verified=True)


def test_verify_without_a_strip_does_not_claim_one_was_verified(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    forecast = forecast_with(None)
    install_for(monkeypatch, forecast)
    seal(forecast, ledger, tsa_url=TSA_URL)
    assert verify(DATE, ledger) == VerifyResult(ok=True, reason=None, strip_verified=False)


def test_a_changed_pixel_fails_verify_naming_the_strip(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed(monkeypatch, ledger, image_factory.plain_png())
    path = ledger_paths(ledger, DATE).strip
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and not outcome.strip_verified
    reason = outcome.reason or ""
    assert "strip" in reason and "mismatch" in reason


def test_a_whole_other_valid_png_in_place_of_the_strip_fails_verify(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed(monkeypatch, ledger, image_factory.plain_png())
    ledger_paths(ledger, DATE).strip.write_bytes(image_factory.plain_png()[:-2] + b"\x00\x00")
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "strip" in (outcome.reason or "")


def test_a_missing_strip_fails_verify_when_the_forecast_names_one(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed(monkeypatch, ledger, image_factory.plain_png())
    ledger_paths(ledger, DATE).strip.unlink()
    outcome = verify(DATE, ledger)
    assert not outcome.ok
    assert "no sealed strip" in (outcome.reason or "") and "2026-10-11" in (outcome.reason or "")


def test_an_unsealed_strip_beside_a_forecast_that_names_none_fails_verify(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    forecast = forecast_with(None)
    install_for(monkeypatch, forecast)
    seal(forecast, ledger, tsa_url=TSA_URL)
    ledger_paths(ledger, DATE).strip.write_bytes(image_factory.plain_png())
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "strip" in (outcome.reason or "")


def test_the_forecast_digest_check_still_comes_first(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    # Only an authenticated forecast may say what the strip's digest is, so a tampered forecast
    # is reported as a digest mismatch even when the strip was swapped too.
    sealed(monkeypatch, ledger, image_factory.plain_png())
    paths = ledger_paths(ledger, DATE)
    paths.strip.write_bytes(image_factory.plain_png()[:-2] + b"\x00\x00")
    paths.forecast.write_text(helpers.flip_digit(paths.forecast.read_text(), "78.1235"))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "digest mismatch" in (outcome.reason or "")


def test_a_late_seal_is_still_reported_when_the_strip_is_intact(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    png = image_factory.plain_png()
    forecast = forecast_with(png)
    digest = helpers.digest_of(forecast)
    monkeypatch.setattr(
        tsa, "post_request", helpers.FakeTsa(helpers.tsr_after_sunset(DATE, digest, 5))
    )
    seal(forecast, ledger, strip=png, tsa_url=TSA_URL)
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "late seal" in (outcome.reason or "")


def test_the_strip_failure_reason_has_no_clock_time(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    from lastlight.privacy import contains_clock_time

    sealed(monkeypatch, ledger, image_factory.plain_png())
    ledger_paths(ledger, DATE).strip.write_bytes(b"x")
    assert not contains_clock_time(verify(DATE, ledger).reason or "")
