"""Where the private archive lives, and the refusal to put raw frames inside a git work tree."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lastlight.cameras.archive import DEFAULT_ARCHIVE_SUBPATH, inside_git_worktree, resolve_archive


def test_the_default_archive_is_under_the_home_directory(tmp_path: Path) -> None:
    assert resolve_archive(None, environ={}, home=tmp_path) == (
        tmp_path / ".local" / "share" / "lastlight" / "frames"
    )
    assert DEFAULT_ARCHIVE_SUBPATH == ".local/share/lastlight/frames"


def test_the_environment_variable_replaces_the_default(tmp_path: Path) -> None:
    chosen = tmp_path / "from-env"
    assert resolve_archive(None, environ={"LASTLIGHT_ARCHIVE": str(chosen)}, home=tmp_path) == chosen


def test_the_command_line_beats_the_environment(tmp_path: Path) -> None:
    on_cli = tmp_path / "from-cli"
    environ = {"LASTLIGHT_ARCHIVE": str(tmp_path / "from-env")}
    assert resolve_archive(str(on_cli), environ=environ, home=tmp_path) == on_cli


def test_an_empty_environment_variable_counts_as_unset(tmp_path: Path) -> None:
    assert resolve_archive(None, environ={"LASTLIGHT_ARCHIVE": ""}, home=tmp_path) == (
        tmp_path / ".local" / "share" / "lastlight" / "frames"
    )


def test_a_leading_tilde_expands_to_the_home_directory(tmp_path: Path) -> None:
    assert resolve_archive("~/frames", environ={}, home=tmp_path) == tmp_path / "frames"


def test_a_relative_path_becomes_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_archive("frames", environ={}, home=tmp_path)
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "frames").resolve()


def test_a_symlink_is_resolved_so_it_cannot_hide_a_work_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(repo)
    resolved = resolve_archive(str(link), environ={}, home=tmp_path)
    assert resolved == repo.resolve()
    assert inside_git_worktree(resolved)


# --- inside_git_worktree ---------------------------------------------------------------------


def test_a_directory_with_a_dot_git_directory_is_a_work_tree(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert inside_git_worktree(tmp_path)


def test_a_linked_worktree_marks_itself_with_a_dot_git_file(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    assert inside_git_worktree(tmp_path)


def test_a_path_that_does_not_exist_yet_is_judged_by_its_ancestors(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert inside_git_worktree(tmp_path / "does" / "not" / "exist" / "yet")


def test_a_path_outside_any_work_tree_is_not_one(tmp_path: Path) -> None:
    assert not inside_git_worktree(tmp_path / "frames")


def test_a_repository_inside_the_archive_does_not_make_the_archive_a_work_tree(
    tmp_path: Path,
) -> None:
    (tmp_path / "frames" / "stray-clone" / ".git").mkdir(parents=True)
    assert not inside_git_worktree(tmp_path / "frames")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_check_agrees_with_git_on_a_real_repository_and_linked_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=60)

    git("init", "--quiet")
    git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "x",
    )
    linked = tmp_path / "linked"
    git("worktree", "add", "--quiet", str(linked))
    for tree in (repo, linked):
        probe = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert probe.stdout.strip() == "true"
        assert inside_git_worktree(tree / "frames")
