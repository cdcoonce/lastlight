"""`lastlight observe`: arguments, exit codes (0 ok, 1 no usable frame, 2 bad input) and output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from observe_archive import EVENING, fill_archive

from lastlight.cameras.archive import ARCHIVE_ENV_VAR, EveningArchive
from lastlight.cli import discover, main

SWEEP = [5.0, 0.0, -2.0, -5.0, -7.0]


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    fill_archive(tmp_path / "frames", SWEEP)
    return tmp_path / "frames"


def run(
    archive: Path | None, out: Path, *extra: str, camera: str = "WHTM1", date: str = "2026-10-10"
):
    argv = ["observe", "--camera", camera, "--date", date, "--out", str(out)]
    if archive is not None:
        argv += ["--archive", str(archive)]
    return main([*argv, *extra])


# --- registration --------------------------------------------------------------------------------


def test_observe_is_a_discovered_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert "observe" in {command.name for command in discover()}
    assert main(["--help"]) == 0
    assert "observe" in capsys.readouterr().out


def test_help_lists_the_documented_options(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["observe", "--help"]) == 0
    out = capsys.readouterr().out
    for option in ("--camera", "--date", "--archive", "--out", "--min-sun-elev", "--max-sun-elev"):
        assert option in out


# --- exit 0 --------------------------------------------------------------------------------------


def test_a_good_evening_writes_observed_json_and_exits_0(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "observed" / "observed.json"
    assert run(archive, out) == 0
    document = json.loads(out.read_text())
    assert document["camera_id"] == "WHTM1" and document["frames_used"] == 4  # +5, 0, -2, -5
    stdout = capsys.readouterr().out
    assert "WHTM1" in stdout and "2026-10-10" in stdout


def test_the_default_window_is_plus_6_to_minus_6_and_min_sun_elev_overrides_it(
    archive: Path, tmp_path: Path
) -> None:
    out = tmp_path / "default.json"
    assert run(archive, out) == 0
    assert json.loads(out.read_text())["frames_used"] == 4  # +5, 0, -2, -5; -7 is dropped
    wide = tmp_path / "wide.json"
    assert run(archive, wide, "--min-sun-elev", "-8") == 0
    assert json.loads(wide.read_text())["frames_used"] == 5
    assert json.loads(wide.read_text())["sun_window_deg"] == {"max": 6.0, "min": -8.0}


def test_max_sun_elev_overrides_the_upper_edge(archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    assert run(archive, out, "--max-sun-elev=1") == 0
    assert json.loads(out.read_text())["frames_used"] == 3  # 0, -2, -5


def test_the_archive_defaults_to_the_environment_variable(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ARCHIVE_ENV_VAR, str(archive))
    out = tmp_path / "o.json"
    assert run(None, out) == 0
    assert json.loads(out.read_text())["frames_used"] == 4


def test_the_summary_names_no_frame_and_gives_no_time(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(archive, tmp_path / "o.json") == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert ".jpg" not in text
    assert "T0" not in text and "Z " not in text
    for jpg in (archive / "WHTM1" / EVENING.isoformat()).glob("*.jpg"):
        assert jpg.stem not in text


# --- exit 1: nothing usable ----------------------------------------------------------------------


def test_no_frame_in_the_window_exits_1_and_still_records_what_was_dropped(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o.json"
    assert run(archive, out, "--min-sun-elev", "20", "--max-sun-elev", "30") == 1
    document = json.loads(out.read_text())
    assert document["frames"] == [] and document["frames_used"] == 0
    assert document["frames_dropped"]["sun_below_window"] == 5  # honesty rule 5: no silent caps
    assert "no usable frame" in capsys.readouterr().err


def test_an_empty_evening_folder_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    EveningArchive(tmp_path, "WHTM1", EVENING).directory.mkdir(parents=True)
    assert run(tmp_path, tmp_path / "o.json") == 1
    assert "no usable frame" in capsys.readouterr().err


# --- exit 2: bad input or configuration ----------------------------------------------------------


def test_esmo1_exits_2_and_says_it_has_no_contrast_table(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o.json"
    assert run(archive, out, camera="ESMO1") == 2
    err = capsys.readouterr().err
    assert "ESMO1" in err and "contrast table" in err and "frozen" in err
    assert not out.exists()


def test_an_unknown_camera_exits_2_and_lists_the_known_ones(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(archive, tmp_path / "o.json", camera="NOPE1") == 2
    err = capsys.readouterr().err
    assert "NOPE1" in err and "WHTM1" in err


def test_a_missing_evening_folder_exits_2_and_names_the_camera_and_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path, tmp_path / "o.json", date="2026-10-12") == 2
    err = capsys.readouterr().err
    assert "WHTM1" in err and "2026-10-12" in err
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("missing", ["--camera", "--date", "--out"])
def test_a_missing_required_argument_exits_2(missing: str, archive: Path, tmp_path: Path) -> None:
    argv = [
        "observe",
        "--camera",
        "WHTM1",
        "--date",
        "2026-10-10",
        "--out",
        str(tmp_path / "o.json"),
    ]
    index = argv.index(missing)
    del argv[index : index + 2]
    assert main([*argv, "--archive", str(archive)]) == 2


@pytest.mark.parametrize("date", ["20261010", "2026-13-01", "tonight", "2026-10", "10/10/2026"])
def test_a_bad_date_exits_2(date: str, archive: Path, tmp_path: Path) -> None:
    assert run(archive, tmp_path / "o.json", date=date) == 2


def test_an_inverted_window_exits_2(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(archive, tmp_path / "o.json", "--min-sun-elev", "3", "--max-sun-elev", "-3") == 2
    assert "window" in capsys.readouterr().err


def test_a_missing_frozen_fit_exits_2(
    archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(archive, tmp_path / "o.json", "--geometry-dir", str(tmp_path / "none")) == 2
    assert "geometry" in capsys.readouterr().err
