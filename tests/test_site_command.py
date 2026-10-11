"""`lastlight site build --ledger-dir DIR --out DIR`: arguments, output and exit codes."""

from __future__ import annotations

from pathlib import Path

import ledger_factory as factory
import pytest
import seal_helpers as helpers

from lastlight.cli import discover, main


def run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def build(
    capsys: pytest.CaptureFixture[str], ledger: Path, out: Path, *extra: str
) -> tuple[int, str, str]:
    return run(capsys, "site", "build", "--ledger-dir", ledger, "--out", out, *extra)


def test_the_site_command_is_registered() -> None:
    assert "site" in {command.name for command in discover()}


def test_building_a_site_exits_zero_and_says_what_it_made(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    code, stdout, stderr = build(capsys, ledger_dir, out)
    assert code == 0
    assert stderr == ""
    assert "built 6 pages for 3 evenings (3 verified)" in stdout
    assert (out / "index.html").is_file()
    assert "2026-10-13" in (out / "index.html").read_text(encoding="utf-8")


def test_a_privacy_finding_exits_one_prints_it_and_writes_no_site(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    factory.overwrite_notes(ledger_dir, factory.DATES[1], f"checked at {helpers.clock()}")
    out = tmp_path / "site"
    code, stdout, stderr = build(capsys, ledger_dir, out)
    assert code == 1
    assert "2026-10-12.html" in stdout and "P001" in stdout
    assert "privacy scan failed" in stderr and "nothing was written" in stderr
    assert not out.exists()


def test_a_missing_ledger_still_builds_the_empty_state_and_warns(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "site"
    code, stdout, stderr = build(capsys, tmp_path / "absent", out)
    assert code == 0
    assert "built 3 pages for 0 evenings" in stdout
    assert "no ledger" in stderr
    assert (out / "index.html").is_file()


def test_a_ledger_that_cannot_be_read_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    file = tmp_path / "ledger"
    file.write_text("not a directory")
    code, stdout, stderr = build(capsys, file, tmp_path / "site")
    assert code == 2
    assert stdout == ""
    assert "not a directory" in stderr


def test_an_output_directory_that_is_not_a_site_is_refused_with_exit_two(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    code, _stdout, stderr = build(capsys, ledger_dir, out)
    assert code == 2
    assert "not a lastlight site" in stderr
    assert (out / "notes.txt").read_text() == "keep me"


def test_an_output_directory_with_a_foreign_index_html_is_refused_with_exit_two(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "webroot"
    out.mkdir()
    (out / "index.html").write_bytes(b"<!doctype html><title>mine</title>")
    (out / "about.html").write_bytes(b"<p>about me</p>")
    code, stdout, stderr = build(capsys, ledger_dir, out)
    assert code == 2
    assert stdout == ""
    assert "not a lastlight site" in stderr
    assert (out / "index.html").read_bytes() == b"<!doctype html><title>mine</title>"
    assert (out / "about.html").read_bytes() == b"<p>about me</p>"


@pytest.mark.parametrize("missing", ["--ledger-dir", "--out"])
def test_both_directories_are_required(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path, missing: str
) -> None:
    arguments = {"--ledger-dir": str(ledger_dir), "--out": str(tmp_path / "site")}
    del arguments[missing]
    flat = [part for item in arguments.items() for part in item]
    code, _stdout, stderr = run(capsys, "site", "build", *flat)
    assert code == 2
    assert missing in stderr


def test_site_without_an_action_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _stdout, stderr = run(capsys, "site")
    assert code == 2
    assert "build" in stderr
