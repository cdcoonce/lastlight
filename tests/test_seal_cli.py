"""`lastlight seal` and `lastlight verify`: the acceptance criteria of issue #5, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import seal_helpers as helpers

from lastlight.cli import discover, main
from lastlight.seal import tsa
from lastlight.seal.tsa import TsaError

DATE = "2026-10-11"


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger"


@pytest.fixture
def forecast_file(tmp_path: Path) -> Path:
    path = tmp_path / "forecast.json"
    path.write_text(helpers.stub_text())
    return path


def run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def seal_cmd(
    capsys: pytest.CaptureFixture[str], forecast: Path, ledger: Path, *extra: str
) -> tuple[int, str, str]:
    return run(capsys, "seal", forecast, "--ledger-dir", ledger, *extra)


def verify_cmd(
    capsys: pytest.CaptureFixture[str], ledger: Path, date: str = DATE
) -> tuple[int, str, str]:
    return run(capsys, "verify", date, "--ledger-dir", ledger)


def install(monkeypatch: pytest.MonkeyPatch, reply: bytes) -> helpers.FakeTsa:
    fake = helpers.FakeTsa(reply)
    monkeypatch.setattr(tsa, "post_request", fake)
    return fake


def test_both_commands_are_registered() -> None:
    assert {"seal", "verify"} <= {command.name for command in discover()}


# --- acceptance: round trip ---------------------------------------------------------------


def test_seal_then_verify_passes_with_the_recorded_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    install(monkeypatch, helpers.recorded_tsr())

    code, out, _err = seal_cmd(capsys, forecast_file, ledger)
    assert code == 0
    assert DATE in out and helpers.STUB_DIGEST in out

    assert verify_cmd(capsys, ledger) == (0, "PASS\n", "")


def test_seal_writes_the_ledger_files_from_the_untidy_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    install(monkeypatch, helpers.recorded_tsr())
    seal_cmd(capsys, forecast_file, ledger)
    assert (ledger / "2026" / "10" / "11.forecast.json").read_bytes() == helpers.STUB_CANONICAL
    assert (ledger / "2026" / "10" / "11.forecast.tsr").read_bytes() == helpers.recorded_tsr()
    (row,) = [json.loads(line) for line in (ledger / "index.jsonl").read_text().splitlines()]
    assert row == {"date": DATE, "digest": helpers.STUB_DIGEST, "model_version": "0.0.1"}


# --- acceptance: tampering ----------------------------------------------------------------


def test_tampering_with_one_float_fails_verify_with_a_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    install(monkeypatch, helpers.recorded_tsr())
    seal_cmd(capsys, forecast_file, ledger)
    sealed = ledger / "2026" / "10" / "11.forecast.json"
    sealed.write_text(helpers.flip_digit(sealed.read_text(), "45.6789"))

    code, out, _err = verify_cmd(capsys, ledger)
    assert code == 1
    assert out.startswith("FAIL: ")
    assert "digest mismatch" in out
    assert out.count("\n") == 1


# --- acceptance: late seal ----------------------------------------------------------------


def test_a_timestamp_after_sunset_fails_verify_as_a_late_seal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    late = helpers.tsr_after_sunset(helpers.STUB_DATE, helpers.STUB_DIGEST, minutes=5)
    install(monkeypatch, late)
    assert seal_cmd(capsys, forecast_file, ledger)[0] == 0

    code, out, _err = verify_cmd(capsys, ledger)
    assert code == 1
    assert out.startswith("FAIL: ") and "late seal" in out


# --- acceptance: no overwrite -------------------------------------------------------------


def test_sealing_an_already_sealed_date_exits_two_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    fake = install(monkeypatch, helpers.recorded_tsr())
    assert seal_cmd(capsys, forecast_file, ledger)[0] == 0
    before = helpers.snapshot(ledger)
    spent = len(fake.calls)

    code, out, err = seal_cmd(capsys, forecast_file, ledger)

    assert code == 2
    assert DATE in err
    assert out == ""
    assert helpers.snapshot(ledger) == before
    assert len(fake.calls) == spent


def test_a_different_forecast_for_a_sealed_date_is_also_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
    tmp_path: Path,
) -> None:
    install(monkeypatch, helpers.recorded_tsr())
    seal_cmd(capsys, forecast_file, ledger)
    before = helpers.snapshot(ledger)
    changed = tmp_path / "changed.json"
    changed.write_text(helpers.stub_text().replace("0.0.1", "0.0.9"))
    assert seal_cmd(capsys, changed, ledger)[0] == 2
    assert helpers.snapshot(ledger) == before


# --- acceptance: a time of day is rejected ------------------------------------------------


def test_a_forecast_with_a_time_of_day_in_notes_is_rejected_and_nothing_is_written(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    ledger: Path,
) -> None:
    fake = install(monkeypatch, helpers.recorded_tsr())
    leaky = tmp_path / "leaky.json"
    leaky.write_text(
        helpers.stub_text().replace("stub forecast", f"dust arrives at {helpers.clock()}")
    )

    code, out, err = seal_cmd(capsys, leaky, ledger)

    assert code == 2
    assert "notes" in err
    assert helpers.clock() not in err + out
    assert helpers.snapshot(ledger) == {}
    assert fake.calls == []


# --- other ways to fail -------------------------------------------------------------------


@pytest.mark.parametrize("content", ["not json", "[]", '{"schema_version": 1}', ""])
def test_a_forecast_that_does_not_validate_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, ledger: Path, content: str
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    code, out, err = seal_cmd(capsys, bad, ledger)
    assert (code, out) == (2, "")
    assert "seal:" in err
    assert helpers.snapshot(ledger) == {}


def test_a_missing_forecast_file_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, ledger: Path
) -> None:
    code, _out, err = seal_cmd(capsys, tmp_path / "nope.json", ledger)
    assert code == 2 and "nope.json" in err


def test_a_tsa_failure_exits_one_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
) -> None:
    def down(url: str, request: bytes, **_options: object) -> bytes:
        raise TsaError("the timestamp authority is down")

    monkeypatch.setattr(tsa, "post_request", down)
    code, out, err = seal_cmd(capsys, forecast_file, ledger)
    assert (code, out) == (1, "")
    assert "down" in err
    assert helpers.snapshot(ledger) == {}


def test_the_tsa_url_is_configurable_and_defaults_to_freetsa(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    tmp_path: Path,
) -> None:
    fake = install(monkeypatch, helpers.recorded_tsr())
    seal_cmd(capsys, forecast_file, tmp_path / "one")
    seal_cmd(capsys, forecast_file, tmp_path / "two", "--tsa-url", "https://tsa.example/tsr")
    assert [url for url, _ in fake.calls] == [
        "https://freetsa.org/tsr",
        "https://tsa.example/tsr",
    ]


def test_the_ledger_dir_is_required(
    capsys: pytest.CaptureFixture[str], forecast_file: Path
) -> None:
    code, _out, err = run(capsys, "seal", forecast_file)
    assert code == 2 and "--ledger-dir" in err
    code, _out, err = run(capsys, "verify", DATE)
    assert code == 2 and "--ledger-dir" in err


@pytest.mark.parametrize(
    "date",
    [
        "2026-13-45",
        "10/11/2026",
        f"2026-10-11T{helpers.MIDNIGHT}",
        "tonight",
        "20261011",
        "2026-W41-6",
    ],
)
def test_verify_rejects_a_date_that_is_not_an_iso_calendar_date(
    capsys: pytest.CaptureFixture[str], ledger: Path, date: str
) -> None:
    code, out, _err = verify_cmd(capsys, ledger, date)
    assert (code, out) == (2, "")


def test_verify_of_an_unsealed_date_fails_with_exit_one(
    capsys: pytest.CaptureFixture[str], ledger: Path
) -> None:
    code, out, _err = verify_cmd(capsys, ledger)
    assert code == 1 and out.startswith("FAIL: ") and DATE in out


# --- the ledger is publishable ------------------------------------------------------------


def test_a_sealed_ledger_passes_the_privacy_scan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forecast_file: Path,
    ledger: Path,
    tmp_path: Path,
) -> None:
    install(monkeypatch, helpers.recorded_tsr())
    seal_cmd(capsys, forecast_file, ledger)
    code, out, err = run(capsys, "privacy-scan", "--root", tmp_path, ledger)
    assert (code, out) == (0, ""), err
