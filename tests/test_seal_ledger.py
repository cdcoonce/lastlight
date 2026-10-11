"""Sealing a forecast into a ledger directory and verifying it again (issue #5)."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest
import seal_helpers as helpers

from lastlight.forecast import Forecast
from lastlight.seal import tsa
from lastlight.seal.canonical import canonical_bytes
from lastlight.seal.ledger import (
    AlreadySealedError,
    SealResult,
    VerifyResult,
    ledger_paths,
    seal,
    verify,
)
from lastlight.seal.tsa import TsaError

DATE = helpers.STUB_DATE
TSA_URL = "https://tsa.example/tsr"


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


def install(monkeypatch: pytest.MonkeyPatch, reply: bytes) -> helpers.FakeTsa:
    fake = helpers.FakeTsa(reply)
    monkeypatch.setattr(tsa, "post_request", fake)
    return fake


def sealed_with(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, reply: bytes | None = None
) -> SealResult:
    install(monkeypatch, helpers.recorded_tsr() if reply is None else reply)
    return seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)


# --- layout -------------------------------------------------------------------------------


def test_ledger_paths_follow_the_design_layout(ledger: Path) -> None:
    paths = ledger_paths(ledger, dt.date(2026, 3, 7))
    assert paths.forecast == ledger / "2026" / "03" / "07.forecast.json"
    assert paths.timestamp == ledger / "2026" / "03" / "07.forecast.tsr"
    assert paths.index == ledger / "index.jsonl"


# --- seal ---------------------------------------------------------------------------------


def test_seal_writes_the_canonical_forecast_the_timestamp_and_one_index_line(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    paths = ledger_paths(ledger, DATE)
    assert result == SealResult(date=DATE, digest=helpers.STUB_DIGEST, paths=paths)
    assert paths.forecast.read_bytes() == helpers.STUB_CANONICAL
    assert paths.timestamp.read_bytes() == helpers.recorded_tsr()
    lines = paths.index.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"date": "2026-10-11", "digest": helpers.STUB_DIGEST, "model_version": "0.0.1"}
    ]
    assert lines[0] == (
        '{"date":"2026-10-11","digest":"' + helpers.STUB_DIGEST + '","model_version":"0.0.1"}'
    )
    assert paths.index.read_bytes().endswith(b"\n")


def test_seal_asks_the_tsa_for_exactly_the_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    fake = install(monkeypatch, helpers.recorded_tsr())
    seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    assert fake.calls == [(TSA_URL, tsa.build_request(helpers.STUB_CANONICAL))]


def test_seal_uses_freetsa_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    fake = install(monkeypatch, helpers.recorded_tsr())
    seal(helpers.stub_forecast(), ledger)
    assert [url for url, _ in fake.calls] == ["https://freetsa.org/tsr"]


def test_sealing_a_second_date_appends_to_the_index(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger)
    data = helpers.stub_dict()
    data["date"] = "2026-10-12"
    data["model"]["version"] = "0.0.2"
    forecast = Forecast.model_validate(data)
    digest = helpers.digest_of(forecast)
    install(monkeypatch, helpers.tsr_before_sunset(dt.date(2026, 10, 12), digest))
    seal(forecast, ledger, tsa_url=TSA_URL)
    rows = [json.loads(line) for line in (ledger / "index.jsonl").read_text().splitlines()]
    assert [(row["date"], row["model_version"]) for row in rows] == [
        ("2026-10-11", "0.0.1"),
        ("2026-10-12", "0.0.2"),
    ]
    assert rows[1]["digest"] == digest
    assert (ledger / "2026" / "10" / "12.forecast.json").is_file()


@pytest.mark.parametrize("existing", ["forecast", "timestamp", "index"])
def test_an_already_sealed_date_is_refused_and_the_ledger_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, existing: str
) -> None:
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    if existing == "forecast":
        paths.forecast.write_bytes(b"{}")
    elif existing == "timestamp":
        paths.timestamp.write_bytes(b"\x30\x00")
    else:
        paths.index.write_text('{"date":"2026-10-11","digest":"00","model_version":"x"}\n')
    before = helpers.snapshot(ledger)
    fake = install(monkeypatch, helpers.recorded_tsr())
    with pytest.raises(AlreadySealedError, match="2026-10-11"):
        seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == before
    assert fake.calls == [], "a refused seal must not spend a timestamp"


class RacingTsa(helpers.FakeTsa):
    """A TSA reply that arrives after another process has already started sealing the date."""

    def __init__(self, reply: bytes, *, create: Path) -> None:
        super().__init__(reply)
        self.create = create

    def __call__(self, url: str, request: bytes, **options: object) -> bytes:
        self.create.parent.mkdir(parents=True, exist_ok=True)
        self.create.write_bytes(b"theirs")
        return super().__call__(url, request, **options)


@pytest.mark.parametrize("raced", ["forecast", "timestamp"])
def test_a_seal_that_loses_a_race_replaces_nothing_and_removes_only_its_own_files(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, raced: str
) -> None:
    paths = ledger_paths(ledger, DATE)
    theirs = paths.forecast if raced == "forecast" else paths.timestamp
    ours = paths.timestamp if raced == "forecast" else paths.forecast
    monkeypatch.setattr(tsa, "post_request", RacingTsa(helpers.recorded_tsr(), create=theirs))
    with pytest.raises(AlreadySealedError, match="2026-10-11"):
        seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    assert theirs.read_bytes() == b"theirs"
    assert not ours.exists()
    assert not paths.index.exists()


def test_an_index_line_for_another_date_does_not_block_this_one(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    ledger.mkdir()
    (ledger / "index.jsonl").write_text('{"date":"2026-10-10","digest":"00","model_version":"x"}\n')
    sealed_with(monkeypatch, ledger)
    assert len((ledger / "index.jsonl").read_text().splitlines()) == 2


def test_an_index_whose_last_line_was_left_open_gets_the_new_row_on_its_own_line(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    ledger.mkdir()
    (ledger / "index.jsonl").write_text('{"date":"2026-10-10","digest":"00","model_version":"x"}')
    sealed_with(monkeypatch, ledger)
    rows = [json.loads(line) for line in (ledger / "index.jsonl").read_text().splitlines()]
    assert [row["date"] for row in rows] == ["2026-10-10", "2026-10-11"]


def test_a_tsa_failure_writes_nothing(monkeypatch: pytest.MonkeyPatch, ledger: Path) -> None:
    def down(url: str, request: bytes, **_options: object) -> bytes:
        raise TsaError("the timestamp authority is down")

    monkeypatch.setattr(tsa, "post_request", down)
    with pytest.raises(TsaError, match="down"):
        seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}


def test_a_response_for_a_different_digest_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    install(monkeypatch, helpers.make_tsr("cd" * 32, dt.datetime(2026, 10, 10, tzinfo=dt.UTC)))
    with pytest.raises(TsaError, match="imprint"):
        seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    assert helpers.snapshot(ledger) == {}


def test_a_failed_write_removes_what_it_had_written(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    ledger.mkdir()
    (ledger / "index.jsonl").mkdir()  # appending to a directory fails after the files exist
    before = helpers.snapshot(ledger)
    install(monkeypatch, helpers.recorded_tsr())
    with pytest.raises(OSError, match="index"):
        seal(helpers.stub_forecast(), ledger, tsa_url=TSA_URL)
    files = {key for key, value in helpers.snapshot(ledger).items() if value is not None}
    assert files == {key for key, value in before.items() if value is not None}
    assert not ledger_paths(ledger, DATE).forecast.exists()
    assert not ledger_paths(ledger, DATE).timestamp.exists()


# --- verify: the pass ---------------------------------------------------------------------


def test_round_trip_with_the_recorded_timestamp_passes(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger)
    assert verify(DATE, ledger) == VerifyResult(ok=True, reason=None)


def test_a_timestamp_shortly_before_sunset_passes(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger, helpers.tsr_before_sunset(DATE, helpers.STUB_DIGEST, 1))
    assert verify(DATE, ledger).ok


# --- verify: the failures -----------------------------------------------------------------


def test_one_changed_float_is_a_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    path = result.paths.forecast
    path.write_text(helpers.flip_digit(path.read_text(), "78.1235"))
    outcome = verify(DATE, ledger)
    assert not outcome.ok
    assert outcome.reason is not None and "digest mismatch" in outcome.reason


def test_a_timestamp_after_sunset_is_a_late_seal(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger, helpers.tsr_after_sunset(DATE, helpers.STUB_DIGEST, 1))
    outcome = verify(DATE, ledger)
    assert not outcome.ok
    assert outcome.reason is not None and "late seal" in outcome.reason


def test_sunset_is_taken_on_the_forecast_date_not_the_day_before(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    # Thirty minutes before sunset on the forecast date is many hours after the previous
    # day's sunset, so this only passes when verify uses the right date.
    sealed_with(monkeypatch, ledger, helpers.tsr_before_sunset(DATE, helpers.STUB_DIGEST, 30))
    assert verify(DATE, ledger).ok
    previous = helpers.sunset(DATE - dt.timedelta(days=1))
    assert previous < helpers.sunset(DATE) - dt.timedelta(hours=23)


def test_a_timestamp_a_whole_day_late_is_still_late(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger, helpers.tsr_after_sunset(DATE, helpers.STUB_DIGEST, 24 * 60))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "late seal" in (outcome.reason or "")


@pytest.mark.parametrize(
    ("minutes", "pattern"),
    [(2 / 3, r"\b\d+ seconds\b"), (12, r"\b12 minutes\b"), (5 * 60, r"\b5\.0 hours\b")],
)
def test_the_late_seal_reason_gives_the_lateness_in_readable_units(
    monkeypatch: pytest.MonkeyPatch, ledger: Path, minutes: float, pattern: str
) -> None:
    sealed_with(monkeypatch, ledger, helpers.tsr_after_sunset(DATE, helpers.STUB_DIGEST, minutes))
    assert re.search(pattern, verify(DATE, ledger).reason or "")


def test_the_late_seal_reason_has_no_clock_time(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    sealed_with(monkeypatch, ledger, helpers.tsr_after_sunset(DATE, helpers.STUB_DIGEST, 12))
    reason = verify(DATE, ledger).reason or ""
    from lastlight.privacy import contains_clock_time

    assert not contains_clock_time(reason)
    assert "12" in reason  # the lateness is reported as a duration


def test_a_missing_forecast_file_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    result.paths.forecast.unlink()
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "2026-10-11" in (outcome.reason or "")


def test_a_missing_timestamp_file_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    result.paths.timestamp.unlink()
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "timestamp" in (outcome.reason or "")


def test_a_date_that_was_never_sealed_is_a_failure(ledger: Path) -> None:
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "2026-10-11" in (outcome.reason or "")


def test_an_unreadable_timestamp_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    result.paths.timestamp.write_bytes(b"garbage")
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "timestamp" in (outcome.reason or "")


def test_a_timestamp_for_another_digest_is_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    result.paths.timestamp.write_bytes(helpers.tsr_before_sunset(DATE, "ee" * 32))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "digest mismatch" in (outcome.reason or "")


def test_a_forecast_filed_under_the_wrong_date_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    result = sealed_with(monkeypatch, ledger)
    other = ledger_paths(ledger, dt.date(2026, 10, 13))
    other.forecast.parent.mkdir(parents=True, exist_ok=True)
    other.forecast.write_bytes(result.paths.forecast.read_bytes())
    other.timestamp.write_bytes(result.paths.timestamp.read_bytes())
    outcome = verify(dt.date(2026, 10, 13), ledger)
    assert not outcome.ok and "date" in (outcome.reason or "")


def test_a_sealed_file_that_is_not_in_canonical_form_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    tidy = json.dumps(json.loads(helpers.STUB_CANONICAL), indent=2).encode()
    digest = helpers.digest_hex(tidy)
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    paths.forecast.write_bytes(tidy)
    paths.timestamp.write_bytes(helpers.tsr_before_sunset(DATE, digest))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "canonical" in (outcome.reason or "")


def test_a_sealed_file_that_is_not_a_valid_forecast_is_a_failure(ledger: Path) -> None:
    junk = b'{"hello":"world"}'
    paths = ledger_paths(ledger, DATE)
    paths.forecast.parent.mkdir(parents=True)
    paths.forecast.write_bytes(junk)
    paths.timestamp.write_bytes(helpers.tsr_before_sunset(DATE, helpers.digest_hex(junk)))
    outcome = verify(DATE, ledger)
    assert not outcome.ok and "forecast" in (outcome.reason or "")


def test_the_sealed_bytes_are_what_canonical_bytes_returns() -> None:
    assert canonical_bytes(helpers.stub_forecast()) == helpers.STUB_CANONICAL
