"""`lastlight forecast`, `lastlight seal --strip`, `lastlight verify` and `lastlight seal-daily`.

The commands are thin: they parse arguments, call `lastlight.painting` and `lastlight.daily`, and
turn the outcome into an exit code. Each test injects the recorded HRRR transport and a fixed
clock through the command's `execute`, the same way the `hrrr-transect` tests do.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
from pathlib import Path

import daily_support as support
import hrrr_support
import pytest
import seal_helpers as helpers

from lastlight.cli import Command, discover, main
from lastlight.commands import forecast as forecast_command
from lastlight.commands import seal_daily as seal_daily_command
from lastlight.forecast import Forecast
from lastlight.privacy.rules import text_hits
from lastlight.seal import ledger_paths

pytestmark = pytest.mark.usefixtures("trust_test_tsa")

COMMON = ["--cycle", "22", "--fhour", "0"]


def command_with(
    module,
    transport: hrrr_support.FakeTransport,
    now: dt.datetime,  # type: ignore[no-untyped-def]
) -> Command:
    def run(args):  # type: ignore[no-untyped-def]
        return module.execute(args, transport=transport, now_utc=now)

    return dataclasses.replace(module.COMMAND, run=run)


def invoke(
    capsys: pytest.CaptureFixture[str],
    module,  # type: ignore[no-untyped-def]
    *argv: str | Path,
    transport: hrrr_support.FakeTransport | None = None,
    now: dt.datetime | None = None,
) -> tuple[int, str, str]:
    command = command_with(module, transport or support.transport(), now or support.before_sunset())
    code = main([str(arg) for arg in argv], commands=[command])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def real_cli(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- registration -------------------------------------------------------------------------


def test_the_new_commands_are_discovered_by_name() -> None:
    assert {"forecast", "seal-daily", "ledger-branch"} <= {c.name for c in discover()}


@pytest.mark.parametrize(
    ("module", "options"),
    [
        (forecast_command, ["--date", "--out", "--strip", "--cycle", "--fhour", "--cache-dir"]),
        (
            seal_daily_command,
            ["--ledger-dir", "--date", "--cycle", "--fhour", "--cache-dir", "--tsa-url"],
        ),
    ],
)
def test_help_lists_the_options(
    capsys: pytest.CaptureFixture[str],
    module,
    options: list[str],  # type: ignore[no-untyped-def]
) -> None:
    assert main([module.COMMAND.name, "--help"], commands=[module.COMMAND]) == 0
    out = capsys.readouterr().out
    for option in options:
        assert option in out


# --- forecast -----------------------------------------------------------------------------


def test_forecast_writes_the_forecast_and_the_strip(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, strip = tmp_path / "forecast.json", tmp_path / "strip.png"
    code, stdout, err = invoke(
        capsys,
        forecast_command,
        "forecast", "--date", "2026-10-10", "--out", out, "--strip", strip,
        "--cache-dir", tmp_path / "cache", *COMMON,
    )  # fmt: skip
    assert code == 0, err
    forecast = Forecast.model_validate_json(out.read_bytes())
    assert forecast.date == support.EVENING
    assert forecast.strip_sha256 == hashlib.sha256(strip.read_bytes()).hexdigest()
    assert str(out) in stdout and str(strip) in stdout
    assert text_hits(stdout) == []


def test_forecast_without_a_strip_declares_none(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "forecast.json"
    code, _stdout, err = invoke(
        capsys,
        forecast_command,
        "forecast", "--date", "2026-10-10", "--out", out, "--cache-dir", tmp_path / "c", *COMMON,
    )  # fmt: skip
    assert code == 0, err
    assert Forecast.model_validate_json(out.read_bytes()).strip_sha256 is None


def test_forecast_defaults_to_the_coming_evening_at_the_reference_point(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "forecast.json"
    code, _stdout, err = invoke(
        capsys,
        forecast_command,
        "forecast", "--out", out, "--cache-dir", tmp_path / "c", *COMMON,
    )  # fmt: skip
    assert code == 0, err
    assert Forecast.model_validate_json(out.read_bytes()).date == support.EVENING


def test_forecast_that_cannot_fetch_writes_nothing_and_exits_one(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, strip = tmp_path / "forecast.json", tmp_path / "strip.png"
    code, stdout, err = invoke(
        capsys,
        forecast_command,
        "forecast", "--date", "2026-10-10", "--out", out, "--strip", strip,
        "--cache-dir", tmp_path / "c", *COMMON,
        transport=hrrr_support.FakeTransport({}),
    )  # fmt: skip
    assert code == 1
    assert err.startswith("forecast: ") and stdout == ""
    assert not out.exists() and not strip.exists()


def test_forecast_requires_an_output_file(capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = invoke(capsys, forecast_command, "forecast", "--date", "2026-10-10")
    assert code == 2 and "--out" in err


@pytest.mark.parametrize("date", ["2026-13-01", "20261010", "tomorrow"])
def test_forecast_rejects_a_bad_date(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, date: str
) -> None:
    code, _out, err = invoke(
        capsys, forecast_command, "forecast", "--date", date, "--out", tmp_path / "f.json"
    )
    assert code == 2 and "--date" in err


# --- forecast -> seal -> verify, by hand --------------------------------------------------


def make_files(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> tuple[Path, Path]:
    out, strip = tmp_path / "forecast.json", tmp_path / "strip.png"
    code, _stdout, err = invoke(
        capsys,
        forecast_command,
        "forecast", "--date", "2026-10-10", "--out", out, "--strip", strip,
        "--cache-dir", tmp_path / "cache", *COMMON,
    )  # fmt: skip
    assert code == 0, err
    return out, strip


def test_seal_with_the_strip_then_verify_says_the_strip_was_verified(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, strip = make_files(capsys, tmp_path)
    support.install_tsa(monkeypatch, support.before_sunset())
    ledger = tmp_path / "ledger"

    code, stdout, err = real_cli(capsys, "seal", out, "--strip", strip, "--ledger-dir", ledger)
    assert code == 0, err
    assert "2026-10-10" in stdout
    assert ledger_paths(ledger, support.EVENING).strip.read_bytes() == strip.read_bytes()

    assert real_cli(capsys, "verify", "2026-10-10", "--ledger-dir", ledger) == (
        0,
        "PASS (strip verified)\n",
        "",
    )


def test_verify_still_prints_a_bare_pass_when_there_was_no_strip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    forecast_file = tmp_path / "forecast.json"
    forecast_file.write_text(helpers.stub_text())
    monkeypatch.setattr("lastlight.seal.tsa.post_request", helpers.FakeTsa(helpers.recorded_tsr()))
    ledger = tmp_path / "ledger"
    assert real_cli(capsys, "seal", forecast_file, "--ledger-dir", ledger)[0] == 0
    assert real_cli(capsys, "verify", "2026-10-11", "--ledger-dir", ledger) == (0, "PASS\n", "")


def test_seal_refuses_a_forecast_that_names_a_strip_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, _strip = make_files(capsys, tmp_path)
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    ledger = tmp_path / "ledger"

    code, stdout, err = real_cli(capsys, "seal", out, "--ledger-dir", ledger)

    assert code == 2 and stdout == ""
    assert "strip" in err
    assert helpers.snapshot(ledger) == {} and fake.calls == []


def test_seal_refuses_a_strip_that_is_not_the_one_the_forecast_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, strip = make_files(capsys, tmp_path)
    strip.write_bytes(strip.read_bytes()[:-1] + b"\x00")
    support.install_tsa(monkeypatch, support.before_sunset())
    ledger = tmp_path / "ledger"
    code, _stdout, err = real_cli(capsys, "seal", out, "--strip", strip, "--ledger-dir", ledger)
    assert code == 2 and "strip" in err
    assert helpers.snapshot(ledger) == {}


def test_seal_reports_an_unreadable_strip_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out, _strip = make_files(capsys, tmp_path)
    code, _stdout, err = real_cli(
        capsys, "seal", out, "--strip", tmp_path / "missing.png", "--ledger-dir", tmp_path / "l"
    )
    assert code == 2 and "missing.png" in err


# --- seal-daily ---------------------------------------------------------------------------


def daily(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *extra: str | Path,
    transport: hrrr_support.FakeTransport | None = None,
    now: dt.datetime | None = None,
) -> tuple[int, str, str]:
    return invoke(
        capsys,
        seal_daily_command,
        "seal-daily", "--ledger-dir", tmp_path / "ledger", "--cache-dir", tmp_path / "cache",
        *COMMON, *extra,
        transport=transport, now=now,
    )  # fmt: skip


def test_seal_daily_seals_the_coming_evening_and_prints_its_digest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())

    code, stdout, err = daily(capsys, tmp_path)

    assert code == 0, err
    forecast_file = ledger_paths(tmp_path / "ledger", support.EVENING).forecast
    digest = hashlib.sha256(forecast_file.read_bytes()).hexdigest()
    assert stdout.splitlines()[-1] == f"sealed 2026-10-10 {digest}"
    assert real_cli(capsys, "verify", "2026-10-10", "--ledger-dir", tmp_path / "ledger")[0] == 0


def test_seal_daily_twice_exits_zero_both_times_and_the_second_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    assert daily(capsys, tmp_path)[0] == 0
    before = helpers.snapshot(tmp_path / "ledger")

    code, stdout, err = daily(capsys, tmp_path)

    assert code == 0, err
    assert stdout.splitlines()[-1].startswith("already sealed 2026-10-10 ")
    assert helpers.snapshot(tmp_path / "ledger") == before


def test_seal_daily_after_sunset_refuses_with_exit_three_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    code, stdout, err = daily(capsys, tmp_path, now=support.after_sunset(minutes=3))
    assert code == 3 and stdout == ""
    assert "never" in err and "back-fill" in err
    assert helpers.snapshot(tmp_path / "ledger") == {}


def test_seal_daily_takes_an_explicit_date(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    code, _stdout, err = daily(capsys, tmp_path, "--date", "2026-10-10")
    assert code == 0, err


def test_seal_daily_for_a_date_whose_sun_has_set_is_refused_whatever_the_default_says(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    code, _stdout, _err = daily(
        capsys, tmp_path, "--date", "2026-10-09", now=support.before_sunset()
    )
    assert code == 3


def test_seal_daily_with_no_weather_data_exits_one_and_says_why(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    code, stdout, err = daily(capsys, tmp_path, transport=hrrr_support.FakeTransport({}))
    assert code == 1 and stdout == ""
    assert err.startswith("seal-daily: ")
    assert helpers.snapshot(tmp_path / "ledger") == {}


def test_seal_daily_whose_timestamp_lands_late_exits_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.after_sunset(minutes=2))
    code, _stdout, err = daily(capsys, tmp_path)
    assert code == 1 and "late seal" in err


def test_seal_daily_requires_a_ledger_directory(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["seal-daily"], commands=[seal_daily_command.COMMAND])
    assert code == 2 and "--ledger-dir" in capsys.readouterr().err
