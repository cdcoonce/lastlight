"""`lastlight hrrr-transect`: the table of HRRR samples along the sightline (issue #6)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import hrrr_support as support
import pytest

from lastlight.cli import Command, discover, main
from lastlight.commands import hrrr_transect
from lastlight.inputs import hrrr
from lastlight.inputs.hrrr import DEFAULT_FIELDS, TRANSECT_DISTANCES_KM, TransectSample

NOW = dt.datetime(2026, 12, 1, tzinfo=dt.UTC)
ARGS = ["hrrr-transect", "--date", "2026-10-10", "--cycle", "22", "--fhour", "0"]


def _command(transport: support.FakeTransport) -> Command:
    def run(args):  # type: ignore[no-untyped-def]
        return hrrr_transect.execute(args, transport=transport, now_utc=NOW)

    return dataclasses.replace(hrrr_transect.COMMAND, run=run)


def _run(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, *extra: str, transport=None
) -> tuple[int, str, str]:
    transport = transport or support.crop_transport()
    code = main([*ARGS, "--cache-dir", str(tmp_path), *extra], commands=[_command(transport)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _rows(out: str) -> list[list[str]]:
    """The data rows of the table: lines that begin with a distance in km."""
    return [line.split() for line in out.splitlines() if re.match(r"\s*\d+\s", line)]


# --- registration --------------------------------------------------------------------------------------------


def test_the_command_is_discovered_by_name() -> None:
    assert "hrrr-transect" in {command.name for command in discover()}


def test_help_lists_the_date_cycle_and_forecast_hour_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["hrrr-transect", "--help"], commands=[hrrr_transect.COMMAND]) == 0

    out = capsys.readouterr().out
    for option in ("--date", "--cycle", "--fhour", "--cache-dir"):
        assert option in out


def test_starting_the_cli_does_not_import_the_grib_library() -> None:
    # Commands are discovered on every start; the decoder loads a native library, so it must wait
    # until a command that needs it actually runs.
    code = (
        "import sys; from lastlight.cli import main; main(['--version']); "
        "sys.exit(1 if 'eccodes' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


# --- the table -------------------------------------------------------------------------------------------------


def test_prints_one_row_per_25_km_out_to_400_km(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, err = _run(capsys, tmp_path)

    assert code == 0, err
    rows = _rows(out)
    assert [int(row[0]) for row in rows] == list(TRANSECT_DISTANCES_KM)
    assert len(rows) == 17


def test_each_row_has_a_value_for_each_default_field(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, _ = _run(capsys, tmp_path)

    assert code == 0
    for row in _rows(out):
        assert len(row) == 1 + len(DEFAULT_FIELDS)


def test_the_header_names_each_field_and_its_units(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, out, _ = _run(capsys, tmp_path)

    for spec in DEFAULT_FIELDS:
        assert spec.name in out
        assert spec.units in out


def test_the_first_row_is_the_reference_point_and_carries_the_real_values(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, out, _ = _run(capsys, tmp_path)

    first = _rows(out)[0]
    by_field = dict(zip([spec.name for spec in DEFAULT_FIELDS], first[1:], strict=True))
    for name, expected in support.EXPECTED_AT_REFERENCE_CELL.items():
        assert float(by_field[name]) == pytest.approx(expected, abs=0.6), name


def test_the_output_records_the_sampling_method_and_the_run(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _, out, _ = _run(capsys, tmp_path)

    assert hrrr.SAMPLING_METHOD in out
    assert "2026-10-10" in out


def test_the_output_holds_no_clock_time(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # Public artifacts are referenced to sun elevation, never to clock time; the table is a
    # command-line view, but it should still be safe to paste into one.
    _, out, err = _run(capsys, tmp_path)

    assert not re.search(r"\b\d{1,2}:\d{2}\b", out + err)


def test_the_run_downloads_only_the_default_fields_by_range(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    transport = support.crop_transport()

    _run(capsys, tmp_path, transport=transport)

    assert len(transport.ranged_gets()) == len(DEFAULT_FIELDS)


def test_a_second_run_reads_the_cache(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    _run(capsys, tmp_path)
    again = support.crop_transport()

    code, _, _ = _run(capsys, tmp_path, transport=again)

    assert code == 0
    assert again.ranged_gets() == []


# --- formatting values -------------------------------------------------------------------------------------------


def _sample(field: str, value: float | None, index: int = 0) -> TransectSample:
    spec = hrrr.field_spec(field)
    return TransectSample(
        point_index=index,
        field=field,
        lat=0.0,
        lon=0.0,
        value=value,
        units=spec.units,
        method=hrrr.SAMPLING_METHOD,
        cell=(0, 0),
        cell_distance_km=0.0,
    )


def test_a_missing_value_prints_as_a_dash_never_nan() -> None:
    samples = [_sample("LCDC", 12.0), _sample("HGT_CLOUD_TOP", None)]
    specs = [hrrr.field_spec("LCDC"), hrrr.field_spec("HGT_CLOUD_TOP")]

    table = hrrr.format_transect_table(samples, specs, distances_km=[0.0])

    row = _rows(table)[0]
    assert row[1:] == ["12", "-"]
    assert "nan" not in table.lower()


def test_the_table_has_one_line_per_distance() -> None:
    specs = [hrrr.field_spec("PWAT")]
    samples = [_sample("PWAT", 10.0 + index, index) for index in range(3)]

    table = hrrr.format_transect_table(samples, specs, distances_km=[0.0, 25.0, 50.0])

    assert [row[0] for row in _rows(table)] == ["0", "25", "50"]


def test_format_refuses_samples_that_do_not_match_the_distances() -> None:
    with pytest.raises(ValueError, match="distance"):
        hrrr.format_transect_table(
            [_sample("PWAT", 1.0)], [hrrr.field_spec("PWAT")], distances_km=[0.0, 25.0]
        )


# --- failures ----------------------------------------------------------------------------------------------------


def test_a_missing_run_is_exit_1_with_the_reason_on_stderr_and_nothing_on_stdout(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, out, err = _run(capsys, tmp_path, transport=support.FakeTransport({}))

    assert code == 1
    assert out == ""
    assert err.startswith("hrrr-transect: ")
    assert "404" in err


def test_a_run_that_does_not_cover_the_transect_is_exit_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_far(evening, lat, lon, distances):  # type: ignore[no-untyped-def]
        return [(lat, lon + 40.0 * index) for index, _ in enumerate(distances)]

    monkeypatch.setattr(hrrr_transect.geometry, "transect", too_far)

    code, out, err = _run(capsys, tmp_path)

    assert code == 1
    assert out == ""
    assert "outside" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["hrrr-transect"],
        ["hrrr-transect", "--date", "not-a-date"],
        ["hrrr-transect", "--date", "2026-10-10", "--cycle", "24"],
        ["hrrr-transect", "--date", "2026-10-10", "--cycle", "-1"],
        ["hrrr-transect", "--date", "2026-10-10", "--fhour", "-1"],
        ["hrrr-transect", "--date", "2026-10-10", "--fhour", "49"],
    ],
)
def test_bad_arguments_are_a_usage_error(argv: list[str]) -> None:
    assert main(argv, commands=[hrrr_transect.COMMAND]) == 2
