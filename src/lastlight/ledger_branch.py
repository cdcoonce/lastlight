"""The orphan `ledger` branch: where sealed days live, so `main` never carries daily data.

The seal workflow writes the ledger into a git worktree of the `ledger` branch, whose root holds
nothing but `ledger/` (`docs/ledger.md`). Two functions do the git work, and the
`ledger-branch` command wraps them for the workflow:

* `open_worktree` attaches that branch at a path, starting it as an orphan (no history shared
  with `main`, no code in it) the first time. It asks the remote before deciding the branch is
  new: a remote that cannot be reached is an error, never a reason to start a second ledger.
  Starting an orphan uses `git worktree add --orphan`, which needs git 2.42 or newer.
* `publish_seal` commits what is new under `ledger/` and pushes it to the branch. It stages
  `ledger/` and nothing else, builds the message `seal: YYYY-MM-DD <first 12 hex of the digest>`
  from the newest `index.jsonl` row, and never forces: a remote that moved on rejects the push
  and the job fails visibly.

The token, when given, is passed to the push alone, as an HTTP header in `-c` configuration, and
is scrubbed from any error text. It is never written to the repository's configuration.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_BRANCH: Final = "ledger"
DEFAULT_REMOTE: Final = "origin"
LEDGER_DIRECTORY: Final = "ledger"
BOT_NAME: Final = "github-actions[bot]"
BOT_EMAIL: Final = "41898282+github-actions[bot]@users.noreply.github.com"
GIT_TIMEOUT_S: Final = 300

_LS_REMOTE_NO_MATCH = 2  # `git ls-remote --exit-code` when the remote answers but has no such ref
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


class LedgerBranchError(RuntimeError):
    """A git step failed or the worktree is not in a state that may be published."""


@dataclass(frozen=True)
class OpenResult:
    """`created` is true when the branch did not exist on the remote and was started here."""

    created: bool
    worktree: Path


@dataclass(frozen=True)
class PublishResult:
    """`pushed` with the commit and its message, or nothing new to publish."""

    pushed: bool
    message: str | None
    commit: str | None


def _scrub(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text.strip()


def _git(
    cwd: Path,
    *args: str,
    config: Sequence[str] = (),
    secrets: Sequence[str] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    for setting in config:
        command += ["-c", setting]
    command += ["-C", str(cwd), *args]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LedgerBranchError(
            f"git {args[0]} could not run: {_scrub(str(error), secrets)}"
        ) from None
    if check and done.returncode != 0:
        raise LedgerBranchError(
            f"git {args[0]} failed: {_scrub(done.stderr or done.stdout, secrets)}"
        )
    return done


def open_worktree(
    repo: Path,
    worktree: Path,
    *,
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
) -> OpenResult:
    """Attach the ledger branch of `repo` as a worktree at `worktree`; start it if it is new.

    The worktree directory must be absent or empty. Afterwards it holds a checkout of the
    branch (or an empty orphan) and an existing `ledger/` directory ready to be written into.
    """
    if worktree.exists() and (not worktree.is_dir() or any(worktree.iterdir())):
        raise LedgerBranchError(f"{worktree} is not empty; it cannot become the ledger worktree")
    probe = _git(repo, "ls-remote", "--exit-code", "--heads", remote, branch, check=False)
    if probe.returncode == 0:
        created = False
    elif probe.returncode == _LS_REMOTE_NO_MATCH:
        created = True
    else:
        raise LedgerBranchError(
            f"cannot ask {remote} whether the {branch} branch exists: {probe.stderr.strip()}"
        )
    if created:
        _git(repo, "worktree", "add", "--quiet", "--orphan", "-b", branch, str(worktree))
    else:
        _git(
            repo, "fetch", "--quiet", "--no-tags", remote,
            f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}",
        )  # fmt: skip
        _git(repo, "worktree", "add", "--quiet", "-B", branch, str(worktree), f"{remote}/{branch}")
    (worktree / LEDGER_DIRECTORY).mkdir(exist_ok=True)
    return OpenResult(created=created, worktree=worktree)


def _seal_message(index: Path) -> str:
    """`seal: YYYY-MM-DD <12 hex>` from the newest row of `index.jsonl`."""
    try:
        lines = [line for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = json.loads(lines[-1])
        date, digest = row["date"], row["digest"]
    except (OSError, UnicodeDecodeError, IndexError, ValueError, KeyError, TypeError):
        raise LedgerBranchError(
            f"cannot name the commit: {index} has no readable last row with a date and a digest"
        ) from None
    if not (
        isinstance(date, str)
        and isinstance(digest, str)
        and _ISO_DATE.fullmatch(date)
        and _HEX_DIGEST.fullmatch(digest)
    ):
        raise LedgerBranchError(f"cannot name the commit: the last row of {index} is malformed")
    return f"seal: {date} {digest[:12]}"


def _auth_config(token: str | None) -> tuple[list[str], list[str]]:
    """(git `-c` settings, secrets to scrub) for pushing to GitHub with an Actions token."""
    if not token:
        return [], []
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return [f"http.https://github.com/.extraheader=AUTHORIZATION: basic {encoded}"], [
        token,
        encoded,
    ]


def publish_seal(
    worktree: Path,
    *,
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    token: str | None = None,
) -> PublishResult:
    """Commit what is new under `ledger/` in `worktree` and push it to `branch` on `remote`.

    Nothing new means no commit and no push. Only `ledger/` is ever staged. The push is never
    forced; if the remote has moved on it is rejected and `LedgerBranchError` is raised.
    """
    status = _git(
        worktree, "status", "--porcelain", "--untracked-files=all", "--", LEDGER_DIRECTORY
    )
    if not status.stdout.strip():
        return PublishResult(pushed=False, message=None, commit=None)
    message = _seal_message(worktree / LEDGER_DIRECTORY / "index.jsonl")
    _git(worktree, "add", "--", LEDGER_DIRECTORY)
    _git(
        worktree, "commit", "--quiet", "-m", message,
        config=[f"user.name={BOT_NAME}", f"user.email={BOT_EMAIL}", "commit.gpgsign=false"],
    )  # fmt: skip
    commit = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    config, secrets = _auth_config(token)
    pushed = _git(
        worktree, "push", "--quiet", remote, f"HEAD:refs/heads/{branch}",
        config=config, secrets=secrets, check=False,
    )  # fmt: skip
    if pushed.returncode != 0:
        raise LedgerBranchError(
            f"the push to {remote}/{branch} failed (nothing was forced): "
            f"{_scrub(pushed.stderr or pushed.stdout, secrets)}"
        )
    return PublishResult(pushed=True, message=message, commit=commit)
