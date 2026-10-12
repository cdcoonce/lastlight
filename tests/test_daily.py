"""The scheduled seal, end to end on recorded fixtures (issue #8).

`seal_daily` is what the workflow runs: paint the evening, seal it, check the seal. These tests
run it against the recorded HRRR crop and a TSA that stamps whatever digest it is sent, with the
clock set from the geometric sunset, so the three promises of the issue are tested where they
live (in Python, not in YAML): a sealed day verifies, a second run changes nothing, and a day is
never sealed late.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import daily_support as support
import hrrr_support
import pytest
import seal_helpers as helpers

from lastlight.cameras.archive import evening_date
from lastlight.daily import (
    DailyResult,
    SealVerificationError,
    TooLateError,
    default_evening,
    seal_daily,
)
from lastlight.forecast import Forecast
from lastlight.inputs.http import FetchError
from lastlight.privacy import scan
from lastlight.privacy.rules import text_hits
from lastlight.seal import ledger_paths, tsa, verify
from lastlight.seal.ledger import VerifyResult
from lastlight.site import REFERENCE_POINT

pytestmark = pytest.mark.usefixtures("trust_test_tsa")


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "hrrr-cache"


def run(
    ledger: Path,
    cache: Path,
    *,
    now: dt.datetime | None = None,
    transport: hrrr_support.FakeTransport | None = None,
    evening: dt.date = support.EVENING,
) -> DailyResult:
    return seal_daily(
        evening,
        ledger,
        transport=transport or support.transport(),
        now_utc=now or support.before_sunset(),
        cache_dir=cache,
        cycle_hour=support.CYCLE_HOUR,
        forecast_hour=support.FORECAST_HOUR,
        tsa_url=support.TSA_URL,
        code_sha=support.CODE_SHA,
    )


# --- acceptance: forecast, seal, verify PASS ----------------------------------------------


def test_a_day_is_painted_sealed_and_verifies(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())

    result = run(ledger, cache)

    paths = ledger_paths(ledger, support.EVENING)
    assert result.status == "sealed" and result.date == support.EVENING
    assert result.digest == hashlib.sha256(paths.forecast.read_bytes()).hexdigest()
    assert fake.calls == [result.digest]
    assert verify(support.EVENING, ledger) == VerifyResult(
        ok=True, reason=None, strip_verified=True
    )


def test_the_ledger_holds_the_forecast_its_timestamp_the_strip_and_one_index_line(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    files = sorted(str(path.relative_to(ledger)) for path in ledger.rglob("*") if path.is_file())
    assert files == [
        "2026/10/10.forecast.json",
        "2026/10/10.forecast.tsr",
        "2026/10/10.strip.png",
        "index.jsonl",
    ]
    (line,) = (ledger / "index.jsonl").read_text().splitlines()
    assert '"date":"2026-10-10"' in line


def test_the_sealed_forecast_names_the_sealed_strip(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    paths = ledger_paths(ledger, support.EVENING)
    forecast = Forecast.model_validate_json(paths.forecast.read_bytes())
    assert forecast.strip_sha256 == hashlib.sha256(paths.strip.read_bytes()).hexdigest()
    assert forecast.model.code_sha == support.CODE_SHA


# --- acceptance: nothing in the ledger can locate anyone ----------------------------------


def test_the_whole_sealed_ledger_passes_the_privacy_scanner(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    result = scan([ledger], root=ledger)
    assert [finding.format() for finding in result.findings] == []
    assert result.files_scanned == 4  # forecast, timestamp, strip, index


def test_the_sealed_forecast_has_no_time_of_day(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    text = ledger_paths(ledger, support.EVENING).forecast.read_text()
    assert text_hits(text) == []
    lat, lon, _ = REFERENCE_POINT
    assert f'"lat":{lat},"lon":{lon}' in text  # the one position in it is the public one


# --- acceptance: tampering ----------------------------------------------------------------


def test_changing_one_byte_of_the_painting_fails_verification(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    strip = ledger_paths(ledger, support.EVENING).strip
    data = bytearray(strip.read_bytes())
    data[-20] ^= 0x01
    strip.write_bytes(bytes(data))
    outcome = verify(support.EVENING, ledger)
    assert not outcome.ok and "strip" in (outcome.reason or "")


# --- acceptance: idempotent ---------------------------------------------------------------


def test_a_second_run_for_the_same_date_changes_nothing_and_asks_for_nothing(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    first = run(ledger, cache)
    before = helpers.snapshot(ledger)
    spent = len(fake.calls)
    transport = support.transport()

    second = run(ledger, cache, transport=transport)

    assert second.status == "already-sealed" and second.digest == first.digest
    assert helpers.snapshot(ledger) == before
    assert len(fake.calls) == spent, "an idempotent run must not spend a timestamp"
    assert transport.calls == [], "an idempotent run must not touch the weather service"


def test_a_rerun_after_the_sun_has_set_is_still_a_clean_no_op(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    # The seal check comes before the lateness check: a day that was sealed in time stays
    # sealed, and asking again later is not an error.
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    before = helpers.snapshot(ledger)
    again = run(ledger, cache, now=support.after_sunset(minutes=90))
    assert again.status == "already-sealed"
    assert helpers.snapshot(ledger) == before


def test_an_existing_seal_that_no_longer_verifies_is_reported_not_waved_through(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    run(ledger, cache)
    forecast_file = ledger_paths(ledger, support.EVENING).forecast
    forecast_file.write_bytes(forecast_file.read_bytes().replace(b'"date"', b'"date" '))
    with pytest.raises(SealVerificationError, match="digest mismatch"):
        run(ledger, cache)


def test_a_half_written_seal_counts_as_sealed_and_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    paths = ledger_paths(ledger, support.EVENING)
    paths.forecast.parent.mkdir(parents=True)
    paths.strip.write_bytes(b"left behind")
    before = helpers.snapshot(ledger)
    with pytest.raises(SealVerificationError):
        run(ledger, cache)
    assert helpers.snapshot(ledger) == before and fake.calls == []


# --- never late ---------------------------------------------------------------------------


def test_after_sunset_nothing_is_painted_or_sealed(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    transport = support.transport()
    with pytest.raises(TooLateError, match="2026-10-10"):
        run(ledger, cache, now=support.after_sunset(minutes=1), transport=transport)
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == [] and transport.calls == []


def test_the_instant_the_sun_reaches_the_horizon_is_already_too_late(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    with pytest.raises(TooLateError):
        run(ledger, cache, now=support.sunset())
    assert helpers.snapshot(ledger) == {}


def test_a_second_before_sunset_is_still_in_time(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    now = support.sunset() - dt.timedelta(seconds=1)
    support.install_tsa(monkeypatch, now)
    assert run(ledger, cache, now=now).status == "sealed"
    assert verify(support.EVENING, ledger).ok


def test_the_refusal_says_the_day_is_never_back_filled_and_prints_no_clock_time(
    ledger: Path, cache: Path
) -> None:
    with pytest.raises(TooLateError) as caught:
        run(ledger, cache, now=support.after_sunset(minutes=1))
    message = str(caught.value)
    assert "never" in message and "back-fill" in message
    assert text_hits(message) == []


def test_a_timestamp_that_lands_after_sunset_is_caught_and_the_run_fails(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    # The clock said "in time" when the run started, but the authority stamped it late.
    support.install_tsa(monkeypatch, support.after_sunset(minutes=2))
    with pytest.raises(SealVerificationError, match="late seal"):
        run(ledger, cache)


# --- failure leaves the ledger alone ------------------------------------------------------


def test_missing_weather_data_fails_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    with pytest.raises(FetchError):
        run(ledger, cache, transport=hrrr_support.FakeTransport({}))
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == []


def test_a_timestamp_authority_failure_leaves_the_ledger_empty(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, cache: Path
) -> None:
    def down(url: str, request: bytes, **_options: object) -> bytes:
        raise tsa.TsaError("the timestamp authority is down")

    monkeypatch.setattr(tsa, "post_request", down)
    with pytest.raises(tsa.TsaError):
        run(ledger, cache)
    assert helpers.snapshot(ledger) == {}


# --- which evening ------------------------------------------------------------------------

LON = REFERENCE_POINT[1]


@pytest.mark.parametrize(
    "when",
    [
        dt.datetime(2026, 10, 10, 20, tzinfo=dt.UTC),
        dt.datetime(2026, 10, 10, 8, tzinfo=dt.UTC),
        dt.datetime(2026, 10, 10, 5, tzinfo=dt.UTC),
        dt.datetime(2026, 12, 31, 23, 59, tzinfo=dt.UTC),
    ],
)
def test_the_default_evening_is_the_local_mean_solar_date_at_the_reference_point(
    when: dt.datetime,
) -> None:
    assert default_evening(when) == evening_date(when, LON)


def test_the_default_evening_for_the_scheduled_hour_is_the_same_utc_date() -> None:
    # Local mean solar time at the reference point runs about seven and a half hours behind UTC, so
    # the scheduled run, in the UTC afternoon, still belongs to that UTC date's evening.
    when = dt.datetime(2026, 6, 21, 21, tzinfo=dt.UTC)
    assert default_evening(when) == dt.date(2026, 6, 21)


def test_a_naive_clock_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        default_evening(dt.datetime(2026, 10, 10, 12))
