"""The orphan `ledger` branch: open it as a worktree, commit a seal to it, push (issue #8).

`main` never carries daily data. The seal workflow keeps it that way by committing only to an
orphan branch whose root holds nothing but `ledger/`. The git plumbing lives in
`lastlight.ledger_branch` (and the `ledger-branch` command) so it can be tested here against a
real local remote, not trusted to YAML that only runs after a merge.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from lastlight.cli import main
from lastlight.ledger_branch import (
    LedgerBranchError,
    OpenResult,
    PublishResult,
    open_worktree,
    publish_seal,
)

DIGEST = "0123456789abcdef" * 4
ROW = {"date": "2026-10-10", "digest": DIGEST, "model_version": "0.1.0"}
REAL_GIT = shutil.which("git")


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No user or system git config: no signing key, no hooks path, no identity to lean on."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout.strip()


def commit_all(cwd: Path, message: str) -> None:
    git(cwd, "add", "-A")
    git(
        cwd, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "--quiet", "-m", message,
    )  # fmt: skip


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    path = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(path)], check=True, capture_output=True
    )
    return path


@pytest.fixture
def checkout(tmp_path: Path, remote: Path) -> Path:
    """A working copy of the code on `main`, with `origin` pointing at the bare remote."""
    path = tmp_path / "code"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "remote", "add", "origin", str(remote))
    (path / "code.txt").write_text("the code\n")
    commit_all(path, "code")
    git(path, "push", "--quiet", "origin", "main")
    return path


@pytest.fixture
def second_checkout(tmp_path: Path, remote: Path, checkout: Path) -> Path:
    """Another runner's clone of the same repository (a branch can be checked out only once)."""
    path = tmp_path / "code-2"
    subprocess.run(
        ["git", "clone", "--quiet", str(remote), str(path)], check=True, capture_output=True
    )
    return path


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    return tmp_path / "ledger-wt"


def write_seal(worktree: Path, date: str = "2026-10-10", digest: str = DIGEST) -> None:
    year, month, day = date.split("-")
    folder = worktree / "ledger" / year / month
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{day}.forecast.json").write_text("{}")
    (folder / f"{day}.strip.png").write_bytes(b"png")
    row = {"date": date, "digest": digest, "model_version": "0.1.0"}
    with (worktree / "ledger" / "index.jsonl").open("a") as index:
        index.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def remote_files(remote: Path, branch: str = "ledger") -> list[str]:
    return git(remote, "ls-tree", "-r", "--name-only", branch).splitlines()


# --- open ---------------------------------------------------------------------------------


def test_with_no_ledger_branch_yet_an_orphan_is_started_holding_an_empty_ledger_directory(
    checkout: Path, worktree: Path
) -> None:
    result = open_worktree(checkout, worktree)

    assert result == OpenResult(created=True, worktree=worktree)
    assert git(worktree, "branch", "--show-current") == "ledger"
    assert sorted(path.name for path in worktree.iterdir()) == [".git", "ledger"]
    assert list((worktree / "ledger").iterdir()) == []
    unborn = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--verify", "--quiet", "HEAD"],
        capture_output=True,
    )
    assert unborn.returncode != 0, "an orphan branch has no commits and shares no history"


def test_an_existing_ledger_branch_is_checked_out_at_the_remote_tip(
    tmp_path: Path, checkout: Path, second_checkout: Path, remote: Path, worktree: Path
) -> None:
    other = tmp_path / "other"
    open_worktree(second_checkout, other)
    write_seal(other)
    publish_seal(other)

    result = open_worktree(checkout, worktree)

    assert result.created is False
    assert git(worktree, "branch", "--show-current") == "ledger"
    assert git(worktree, "rev-parse", "HEAD") == git(remote, "rev-parse", "ledger")
    assert (worktree / "ledger" / "index.jsonl").is_file()
    assert not (worktree / "code.txt").exists(), "the ledger branch never carries the code"


def test_a_remote_that_cannot_be_asked_is_an_error_not_a_reason_to_start_a_new_ledger(
    checkout: Path, worktree: Path
) -> None:
    git(checkout, "remote", "set-url", "origin", str(checkout / "no-such-remote.git"))
    with pytest.raises(LedgerBranchError, match="ledger"):
        open_worktree(checkout, worktree)
    assert not worktree.exists()


def test_a_worktree_directory_that_is_in_use_is_refused(checkout: Path, worktree: Path) -> None:
    worktree.mkdir()
    (worktree / "something").write_text("x")
    with pytest.raises(LedgerBranchError, match="not empty"):
        open_worktree(checkout, worktree)


def test_opening_leaves_the_code_checkout_on_main_with_nothing_changed(
    checkout: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    assert git(checkout, "branch", "--show-current") == "main"
    assert git(checkout, "status", "--porcelain") == ""


# --- publish ------------------------------------------------------------------------------


def test_publishing_with_nothing_new_makes_no_commit_and_no_push(
    checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    assert publish_seal(worktree) == PublishResult(pushed=False, message=None, commit=None)
    assert "ledger" not in git(remote, "branch", "--list").split()


def test_a_seal_is_committed_with_the_date_and_twelve_digits_and_pushed(
    checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)

    result = publish_seal(worktree)

    assert result.pushed is True
    assert result.message == f"seal: 2026-10-10 {DIGEST[:12]}"
    assert git(remote, "log", "-1", "--format=%s", "ledger") == result.message
    assert git(remote, "rev-parse", "ledger") == result.commit
    assert git(remote, "log", "-1", "--format=%an <%ae>", "ledger") == (
        "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"
    )


def test_the_branch_root_holds_only_the_ledger_directory_and_it_has_no_parent(
    checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)
    (worktree / "stray.txt").write_text("not part of the ledger")
    publish_seal(worktree)

    assert git(remote, "ls-tree", "--name-only", "ledger").splitlines() == ["ledger"]
    assert remote_files(remote) == [
        "ledger/2026/10/10.forecast.json",
        "ledger/2026/10/10.strip.png",
        "ledger/index.jsonl",
    ]
    assert git(remote, "rev-list", "--parents", "-n1", "ledger").split() == [
        git(remote, "rev-parse", "ledger")
    ], "the first commit is an orphan root"


def test_main_never_receives_the_daily_data(checkout: Path, remote: Path, worktree: Path) -> None:
    before = git(remote, "rev-parse", "main")
    open_worktree(checkout, worktree)
    write_seal(worktree)
    publish_seal(worktree)
    assert git(remote, "rev-parse", "main") == before
    assert "ledger/index.jsonl" not in git(remote, "ls-tree", "-r", "--name-only", "main")


def test_the_next_day_is_a_second_commit_on_top_of_the_first(
    tmp_path: Path, checkout: Path, second_checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)
    first = publish_seal(worktree)

    second_tree = tmp_path / "next-day"
    assert open_worktree(second_checkout, second_tree).created is False
    write_seal(second_tree, "2026-10-11", "ab" * 32)
    second = publish_seal(second_tree)

    assert second.message == f"seal: 2026-10-11 {'ab' * 6}"
    assert git(remote, "rev-parse", "ledger~1") == first.commit
    rows = git(remote, "show", "ledger:ledger/index.jsonl").splitlines()
    assert [json.loads(row)["date"] for row in rows] == ["2026-10-10", "2026-10-11"]


def test_publishing_twice_adds_one_commit_not_two(
    checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)
    publish_seal(worktree)
    again = publish_seal(worktree)
    assert again.pushed is False
    assert git(remote, "rev-list", "--count", "ledger") == "1"


def test_content_without_a_readable_index_row_is_refused_and_nothing_is_pushed(
    checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    (worktree / "ledger" / "index.jsonl").write_text("not json\n")
    with pytest.raises(LedgerBranchError, match="index"):
        publish_seal(worktree)
    assert "ledger" not in git(remote, "branch", "--list").split()


def test_a_remote_that_moved_on_rejects_the_push_and_is_never_overwritten(
    tmp_path: Path, checkout: Path, second_checkout: Path, remote: Path, worktree: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)
    other = tmp_path / "racer"
    open_worktree(second_checkout, other)
    write_seal(other, "2026-10-09", "cd" * 32)
    winner = publish_seal(other)

    with pytest.raises(LedgerBranchError, match="push"):
        publish_seal(worktree)

    assert git(remote, "rev-parse", "ledger") == winner.commit


# --- the token ----------------------------------------------------------------------------


@pytest.fixture
def git_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a `git` on PATH that records its arguments, one command per line, then runs real git."""
    assert REAL_GIT is not None
    log = tmp_path / "git-calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$GIT_CALL_LOG"\nexec "$REAL_GIT" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("GIT_CALL_LOG", str(log))
    monkeypatch.setenv("REAL_GIT", REAL_GIT)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_the_token_is_sent_on_the_push_and_on_no_other_git_command(
    checkout: Path, worktree: Path, git_log: Path
) -> None:
    secret = "ghs_exampletokenvalue"
    open_worktree(checkout, worktree)
    write_seal(worktree)
    publish_seal(worktree, token=secret)

    calls = git_log.read_text().splitlines()
    encoded = base64.b64encode(f"x-access-token:{secret}".encode()).decode()
    carrying = [call for call in calls if encoded in call]
    assert len(carrying) == 1 and " push " in f" {carrying[0]} "
    assert all(secret not in call for call in calls)


@pytest.fixture
def git_that_echoes_a_failed_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a `git` on PATH that runs real git, except that `push` prints its argv and fails.

    Real git never prints the `-c` header in a rejected-push message, so only a git that does
    (which a verbose or misbehaving one could) can show whether the error text is scrubbed.
    """
    assert REAL_GIT is not None
    bin_dir = tmp_path / "echo-bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'for argument in "$@"; do\n'
        '  if [ "$argument" = push ]; then echo "fatal: push failed: $*" >&2; exit 1; fi\n'
        "done\n"
        'exec "$REAL_GIT" "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("REAL_GIT", REAL_GIT)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return wrapper


SECRET = "ghs_exampletokenvalue"
ENCODED_SECRET = base64.b64encode(f"x-access-token:{SECRET}".encode()).decode()


def test_a_failed_push_does_not_echo_the_token(
    checkout: Path, worktree: Path, git_that_echoes_a_failed_push: Path
) -> None:
    open_worktree(checkout, worktree)
    write_seal(worktree)
    with pytest.raises(LedgerBranchError) as caught:
        publish_seal(worktree, token=SECRET)
    message = str(caught.value)
    # The fake git put the whole command line, header included, in its error: the scrub is
    # what keeps it out (the placeholder shows the header really was there to scrub).
    assert "push failed" in message and "***" in message
    assert SECRET not in message and ENCODED_SECRET not in message
    assert "AUTHORIZATION" in message, "only the secret is hidden, not the rest of the error"


def test_the_command_reports_a_failed_push_without_the_token(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    worktree: Path,
    git_that_echoes_a_failed_push: Path,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", SECRET)
    open_worktree(checkout, worktree)
    write_seal(worktree)

    assert main(["ledger-branch", "publish", "--worktree", str(worktree)]) == 1

    shown = capsys.readouterr()
    assert shown.err.startswith("ledger-branch: ") and "push failed" in shown.err
    assert SECRET not in shown.out + shown.err
    assert ENCODED_SECRET not in shown.out + shown.err


# --- the command --------------------------------------------------------------------------


def test_the_command_opens_and_publishes(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    remote: Path,
    worktree: Path,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    code = main(["ledger-branch", "open", "--worktree", str(worktree), "--repo", str(checkout)])
    assert code == 0 and "new orphan" in capsys.readouterr().out

    assert main(["ledger-branch", "publish", "--worktree", str(worktree)]) == 0
    assert "nothing to publish" in capsys.readouterr().out

    write_seal(worktree)
    assert main(["ledger-branch", "publish", "--worktree", str(worktree)]) == 0
    out = capsys.readouterr().out
    assert f"seal: 2026-10-10 {DIGEST[:12]}" in out
    assert git(remote, "log", "-1", "--format=%s", "ledger") == f"seal: 2026-10-10 {DIGEST[:12]}"


def test_the_command_hands_github_token_to_the_push_and_to_no_other_git_call(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    checkout: Path,
    worktree: Path,
    git_log: Path,
) -> None:
    secret = "ghs_commandleveltokenvalue"
    decoy = "ghs_adifferenttokenvariable"
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    monkeypatch.setenv("GH_TOKEN", decoy)
    assert (
        main(["ledger-branch", "open", "--worktree", str(worktree), "--repo", str(checkout)]) == 0
    )
    write_seal(worktree)
    git_log.write_text("")  # only what `publish` runs is under test

    assert main(["ledger-branch", "publish", "--worktree", str(worktree)]) == 0

    calls = git_log.read_text().splitlines()
    encoded = base64.b64encode(f"x-access-token:{secret}".encode()).decode()
    decoy_encoded = base64.b64encode(f"x-access-token:{decoy}".encode()).decode()
    carrying = [call for call in calls if encoded in call]
    assert len(carrying) == 1, "the token reaches git on exactly one call"
    assert " push " in f" {carrying[0]} ", "and that call is the push"
    assert f"AUTHORIZATION: basic {encoded}" in carrying[0]
    assert all(decoy_encoded not in call and decoy not in call for call in calls)
    assert all(secret not in call for call in calls), (
        "only the encoded header is on the command line"
    )
    shown = capsys.readouterr()
    assert secret not in shown.out + shown.err and encoded not in shown.out + shown.err


def test_the_command_fails_loudly_when_the_worktree_is_not_there(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = main(["ledger-branch", "publish", "--worktree", str(tmp_path / "missing")])
    assert code == 1 and capsys.readouterr().err.startswith("ledger-branch: ")


def test_the_command_needs_an_action_and_a_worktree(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ledger-branch"]) == 2
    assert main(["ledger-branch", "open"]) == 2
