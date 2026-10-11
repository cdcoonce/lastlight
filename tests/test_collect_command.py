"""`lastlight collect`: argument handling, exit codes and the stdout summary."""

from __future__ import annotations

import datetime as dt
import re
from functools import partial
from pathlib import Path

import pytest
from camera_fakes import FakeClock, FakeFetch, fresh, not_modified, status_only

from lastlight.cameras import load_registry
from lastlight.cli import Command, discover, main
from lastlight.commands.collect import COMMAND, run_collect
from lastlight.geometry import elevation_crossings

EVENING = dt.date(2026, 10, 10)
SUMMARY_LINE = re.compile(
    r"^(?P<camera>[A-Z0-9]+) 2026-10-10: saved=(?P<saved>\d+) not_modified=(?P<nm>\d+) "
    r"stale=(?P<stale>\d+) errors=(?P<errors>\d+)$"
)


def cli(argv: list[str], clock: FakeClock, fetch: FakeFetch) -> int:
    """Run `lastlight collect <argv>` through the real parser with the fakes injected."""
    command = Command(
        name="collect",
        help=COMMAND.help,
        run=partial(run_collect, clock=clock, fetch=fetch),
        configure=COMMAND.configure,
    )
    return main(["collect", *argv], commands=[command])


def evening_clock() -> FakeClock:
    """A clock one hour before the sun reaches +6 degrees at the first registry camera."""
    camera = next(iter(load_registry().values()))
    instants = elevation_crossings(EVENING, camera.position.lat, camera.position.lon)
    return FakeClock(instants[6.0].replace(microsecond=0) - dt.timedelta(hours=1))


@pytest.fixture
def idle() -> tuple[FakeClock, FakeFetch]:
    """A clock and fetch that fail the test if the command touches them."""
    clock = FakeClock(dt.datetime(2026, 10, 10, 20, tzinfo=dt.UTC))
    return clock, FakeFetch(clock)


# --- registration -----------------------------------------------------------------------------


def test_collect_is_a_discovered_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert "collect" in {command.name for command in discover()}
    assert main(["--help"]) == 0
    assert "collect" in capsys.readouterr().out


def test_collect_help_lists_the_documented_options(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["collect", "--help"]) == 0
    out = capsys.readouterr().out
    for option in ("--camera", "--archive", "--start-elev", "--stop-elev", "--interval"):
        assert option in out


# --- refusals: exit 2 and nothing happens -------------------------------------------------------


def test_interval_60_is_refused_with_a_clear_message(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch], capsys: pytest.CaptureFixture[str]
) -> None:
    clock, fetch = idle
    assert cli(["--archive", str(tmp_path / "frames"), "--interval", "60"], clock, fetch) == 2
    err = capsys.readouterr().err
    assert "300" in err
    assert "60" in err
    assert fetch.calls == []
    assert clock.sleeps == []
    assert not (tmp_path / "frames").exists()


def test_interval_300_is_accepted(tmp_path: Path) -> None:
    clock = evening_clock()
    fetch = FakeFetch(clock, default=lambda call: not_modified())
    argv = [
        "--archive",
        str(tmp_path),
        "--interval",
        "300",
        "--start-elev",
        "6",
        "--stop-elev",
        "-8",
    ]
    assert cli(argv, clock, fetch) == 0


def test_a_non_numeric_interval_is_a_usage_error(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch]
) -> None:
    clock, fetch = idle
    assert cli(["--archive", str(tmp_path), "--interval", "soon"], clock, fetch) == 2
    assert fetch.calls == []


def test_an_archive_inside_a_git_work_tree_exits_2_and_writes_nothing(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch], capsys: pytest.CaptureFixture[str]
) -> None:
    clock, fetch = idle
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert cli(["--archive", str(repo / "frames")], clock, fetch) == 2

    assert "git" in capsys.readouterr().err
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not (repo / "frames").exists()
    assert fetch.calls == []
    assert clock.sleeps == []


def test_the_git_check_applies_to_the_environment_default_too(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, fetch = idle
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    monkeypatch.setenv("LASTLIGHT_ARCHIVE", str(tmp_path / "repo" / "nested" / "frames"))

    assert cli([], clock, fetch) == 2

    assert not (tmp_path / "repo" / "nested").exists()
    assert fetch.calls == []


def test_an_archive_that_is_a_file_exits_2(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch]
) -> None:
    clock, fetch = idle
    target = tmp_path / "frames"
    target.write_text("not a directory")
    assert cli(["--archive", str(target)], clock, fetch) == 2
    assert fetch.calls == []


def test_an_unknown_camera_exits_2_and_names_the_known_ones(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch], capsys: pytest.CaptureFixture[str]
) -> None:
    clock, fetch = idle
    assert cli(["--archive", str(tmp_path), "--camera", "NOPE9"], clock, fetch) == 2
    err = capsys.readouterr().err
    assert "NOPE9" in err
    assert "WHTM1" in err
    assert "ESMO1" in err
    assert fetch.calls == []


def test_a_start_elevation_at_or_below_the_stop_elevation_exits_2(
    tmp_path: Path, idle: tuple[FakeClock, FakeFetch]
) -> None:
    clock, fetch = idle
    argv = ["--archive", str(tmp_path), "--start-elev", "-10", "--stop-elev", "-10"]
    assert cli(argv, clock, fetch) == 2
    assert fetch.calls == []


# --- a run ------------------------------------------------------------------------------------


def test_a_run_prints_one_summary_line_per_camera_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = evening_clock()
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    code = cli(["--archive", str(tmp_path), "--start-elev", "6", "--stop-elev", "-8"], clock, fetch)

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert [SUMMARY_LINE.fullmatch(line).group("camera") for line in lines] == ["WHTM1", "ESMO1"]
    assert all(SUMMARY_LINE.fullmatch(line).group("errors") == "0" for line in lines)
    urls = {call.url for call in fetch.calls}
    assert urls == {camera.image_url for camera in load_registry().values()}


@pytest.mark.parametrize("argv", [["--camera", "WHTM1"], ["--camera", "WHTM1", "WHTM1"]])
def test_camera_selects_which_cameras_are_polled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    clock = evening_clock()
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    code = cli(
        ["--archive", str(tmp_path), "--start-elev", "6", "--stop-elev", "-8", *argv], clock, fetch
    )

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("WHTM1 ")
    assert {call.url for call in fetch.calls} == {load_registry()["WHTM1"].image_url}


def test_camera_may_be_repeated_or_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (["--camera", "ESMO1", "--camera", "WHTM1"], ["--camera", "ESMO1", "WHTM1"]):
        clock = evening_clock()
        fetch = FakeFetch(clock, default=lambda call: not_modified())
        assert (
            cli(
                ["--archive", str(tmp_path), "--start-elev", "6", "--stop-elev", "-8", *argv],
                clock,
                fetch,
            )
            == 0
        )
        lines = capsys.readouterr().out.splitlines()
        assert sorted(line.split()[0] for line in lines) == ["ESMO1", "WHTM1"]


def test_frames_go_to_the_archive_named_on_the_command_line_before_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASTLIGHT_ARCHIVE", str(tmp_path / "from-env"))
    clock = evening_clock()
    fetch = FakeFetch(clock, default=fresh("x"))
    argv = ["--archive", str(tmp_path / "from-cli"), "--camera", "WHTM1"]

    assert cli([*argv, "--start-elev", "6", "--stop-elev", "-8"], clock, fetch) == 0

    assert list((tmp_path / "from-cli" / "WHTM1" / EVENING.isoformat()).glob("*.jpg"))
    assert not (tmp_path / "from-env").exists()


def test_frames_go_to_the_environment_archive_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASTLIGHT_ARCHIVE", str(tmp_path / "from-env"))
    clock = evening_clock()
    fetch = FakeFetch(clock, default=fresh("x"))

    argv = ["--camera", "WHTM1", "--start-elev", "6", "--stop-elev", "-8"]
    assert cli(argv, clock, fetch) == 0

    assert list((tmp_path / "from-env" / "WHTM1" / EVENING.isoformat()).glob("*.jpg"))


def test_problems_go_to_stderr_and_stdout_holds_only_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = evening_clock()
    fetch = FakeFetch(clock, default=lambda call: status_only(404))

    code = cli(
        ["--archive", str(tmp_path), "--camera", "WHTM1", "--start-elev", "6", "--stop-elev", "-8"],
        clock,
        fetch,
    )

    captured = capsys.readouterr()
    assert code == 0  # an evening with errors is still a finished evening
    assert len(captured.out.splitlines()) == 1
    match = SUMMARY_LINE.fullmatch(captured.out.strip())
    assert match is not None
    assert int(match.group("errors")) == len(fetch.calls) > 0
    assert "404" in captured.err
