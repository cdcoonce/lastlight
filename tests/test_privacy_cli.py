"""`lastlight privacy-scan`: output format, exit codes and which files are scanned."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import image_factory as images
import pytest

from lastlight.cli import main


def run(
    capsys: pytest.CaptureFixture[str], root: Path, *args: str | Path
) -> tuple[int, list[str], str]:
    """Run `privacy-scan --root ROOT ARGS` in-process; return exit code, stdout lines, stderr."""
    code = main(["privacy-scan", "--root", str(root), *map(str, args)])
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, timeout=60)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    return tmp_path


# --- output format and exit codes ------------------------------------------------------------


def test_a_leak_exits_non_zero_with_one_path_line_rule_excerpt_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "leak.txt").write_text("clean\nseal at 17:42 today\n")
    code, out, _err = run(capsys, tmp_path, tmp_path / "leak.txt")
    assert code == 1
    assert out == ["leak.txt:2: P001: 17:42"]


def test_a_clean_file_exits_zero_with_no_findings_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean.txt").write_text("Sun elevation -6 degrees on 2026-10-10.\n")
    code, out, err = run(capsys, tmp_path, tmp_path / "clean.txt")
    assert (code, out) == (0, [])
    assert "clean" in err


def test_findings_are_one_per_line_and_sorted_by_path_then_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "b.txt").write_text("x\ncamera 33.4484, -112.0740\n")
    (tmp_path / "a.txt").write_text("at 17:42\nat 18:15\n")
    code, out, err = run(capsys, tmp_path, tmp_path / "b.txt", tmp_path / "a.txt")
    assert code == 1
    assert out == [
        "a.txt:1: P001: 17:42",
        "a.txt:2: P001: 18:15",
        "b.txt:2: P003: 33.4484, -112.0740",
    ]
    assert "3 finding" in err


def test_a_binary_finding_prints_line_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "leaky.png").write_bytes(images.png_with_text())
    code, out, _err = run(capsys, tmp_path, tmp_path / "leaky.png")
    assert (code, out) == (1, ["leaky.png:1: P004: tEXt chunk"])


def test_a_directory_argument_is_scanned_recursively_and_skips_dot_git(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "out" / "deep").mkdir(parents=True)
    (tmp_path / "out" / "deep" / "leak.txt").write_text("at 17:42\n")
    (tmp_path / "out" / ".git").mkdir()
    (tmp_path / "out" / ".git" / "config").write_text("at 18:15\n")
    code, out, _err = run(capsys, tmp_path, tmp_path / "out")
    assert (code, out) == (1, ["out/deep/leak.txt:1: P001: 17:42"])


def test_a_missing_path_is_a_usage_error_not_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(capsys, tmp_path, tmp_path / "nope.txt")
    assert (code, out) == (2, [])
    assert "nope.txt" in err


def test_a_symlink_is_not_followed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "real.txt").write_text("at 17:42\n")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    code, out, _err = run(capsys, tmp_path, tmp_path / "link.txt")
    assert (code, out) == (0, [])


# --- the allowlist option --------------------------------------------------------------------


def test_the_default_allowlist_is_privacy_allowlist_toml_in_the_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "seal.yml").write_text("cron at 17:42\n")
    (tmp_path / "privacy-allowlist.toml").write_text(
        '[[allow]]\npath = "seal.yml"\nrule = "P001"\nmatch = "17:42"\nreason = "fixed seal"\n'
    )
    assert run(capsys, tmp_path, tmp_path / "seal.yml")[:2] == (0, [])


def test_an_explicit_allowlist_replaces_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "seal.yml").write_text("cron at 17:42\n")
    other = tmp_path / "elsewhere.toml"
    other.write_text(
        '[[allow]]\npath = "seal.yml"\nrule = "P001"\nmatch = "17:42"\nreason = "fixed seal"\n'
    )
    code, out, _err = run(capsys, tmp_path, "--allowlist", other, tmp_path / "seal.yml")
    assert (code, out) == (0, [])


def test_a_missing_explicit_allowlist_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean.txt").write_text("fine\n")
    code, _out, err = run(capsys, tmp_path, "--allowlist", tmp_path / "gone.toml", tmp_path)
    assert code == 2
    assert "gone.toml" in err


def test_an_allowlist_entry_without_a_reason_fails_the_scan_with_an_a001_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean.txt").write_text("fine\n")
    (tmp_path / "privacy-allowlist.toml").write_text(
        '[[allow]]\npath = "x"\nrule = "P001"\nmatch = "17:42"\nreason = ""\n'
    )
    code, out, _err = run(capsys, tmp_path, tmp_path / "clean.txt")
    assert code == 1
    assert len(out) == 1
    assert out[0].startswith("privacy-allowlist.toml:1: A001: ")


# --- default targets: tracked files plus site/ -----------------------------------------------


def test_with_no_paths_every_tracked_file_is_scanned(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("seal at 17:42\n")
    (repo / "ok.txt").write_text("fine\n")
    git(repo, "add", "docs/note.md", "ok.txt")
    code, out, _err = run(capsys, repo)
    assert (code, out) == (1, ["docs/note.md:1: P001: 17:42"])


def test_with_no_paths_untracked_files_are_not_scanned(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ok.txt").write_text("fine\n")
    (repo / "scratch.txt").write_text("seal at 17:42\n")
    git(repo, "add", "ok.txt")
    assert run(capsys, repo)[:2] == (0, [])


def test_with_no_paths_an_untracked_site_directory_is_scanned_too(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ok.txt").write_text("fine\n")
    git(repo, "add", "ok.txt")
    (repo / "site" / "2026").mkdir(parents=True)
    (repo / "site" / "2026" / "index.html").write_text("<p>seal 17:42</p>\n")
    (repo / "site" / "strip.png").write_bytes(images.png_with_text())
    code, out, _err = run(capsys, repo)
    assert code == 1
    assert out == [
        "site/2026/index.html:1: P001: 17:42",
        "site/strip.png:1: P004: tEXt chunk",
    ]


def test_a_file_that_is_both_tracked_and_under_site_is_reported_once(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "site").mkdir()
    (repo / "site" / "index.html").write_text("<p>seal 17:42</p>\n")
    git(repo, "add", "site/index.html")
    assert run(capsys, repo)[1] == ["site/index.html:1: P001: 17:42"]


def test_a_tracked_file_deleted_from_the_working_tree_is_skipped(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "gone.txt").write_text("seal at 17:42\n")
    git(repo, "add", "gone.txt")
    (repo / "gone.txt").unlink()
    assert run(capsys, repo)[:2] == (0, [])


def test_with_no_paths_outside_a_git_repository_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(capsys, tmp_path)
    assert (code, out) == (2, [])
    assert "git" in err


def test_explicit_paths_do_not_need_a_git_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "leak.txt").write_text("at 17:42\n")
    assert run(capsys, tmp_path, tmp_path / "leak.txt")[0] == 1


# --- wiring ----------------------------------------------------------------------------------


def test_the_command_is_registered_and_appears_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    assert "privacy-scan" in capsys.readouterr().out


def test_module_entry_point_exits_non_zero_on_a_leak_and_zero_when_clean(tmp_path: Path) -> None:
    leak = tmp_path / "leak.txt"
    leak.write_text("at 17:42\n")
    clean = tmp_path / "clean.txt"
    clean.write_text("fine\n")
    base = [sys.executable, "-m", "lastlight", "privacy-scan", "--root", str(tmp_path)]
    failed = subprocess.run([*base, str(leak)], capture_output=True, text=True, timeout=60)
    passed = subprocess.run([*base, str(clean)], capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1
    assert failed.stdout == "leak.txt:1: P001: 17:42\n"
    assert passed.returncode == 0
    assert passed.stdout == ""
