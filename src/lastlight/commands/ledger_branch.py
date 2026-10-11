"""`lastlight ledger-branch open|publish --worktree DIR`: the seal job's orphan `ledger` branch.

`open` attaches the branch as a git worktree (starting it as an orphan the first time) and
`publish` commits what is new under `ledger/` and pushes it, with the message
`seal: YYYY-MM-DD <digest[:12]>`. The token comes from `GITHUB_TOKEN` and is used for the push
alone. Thin by design: the git work is in `lastlight.ledger_branch`.

Exit codes: 0 done (including "nothing to publish"), 1 a git step failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.ledger_branch import (
    DEFAULT_BRANCH,
    DEFAULT_REMOTE,
    LedgerBranchError,
    open_worktree,
    publish_seal,
)


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        choices=["open", "publish"],
        help="open: attach the ledger branch as a worktree; publish: commit and push a seal",
    )
    parser.add_argument(
        "--worktree",
        type=Path,
        required=True,
        metavar="DIR",
        help="the ledger worktree directory (open: absent or empty)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(),
        metavar="DIR",
        help="the code checkout the worktree hangs off (default: the current directory)",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="default: %(default)s")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="default: %(default)s")


def _run(args: argparse.Namespace) -> int:
    try:
        if args.action == "open":
            opened = open_worktree(args.repo, args.worktree, remote=args.remote, branch=args.branch)
            verb = "started a new orphan" if opened.created else "checked out the existing"
            print(f"{verb} {args.branch} branch in {args.worktree}")
            return 0
        published = publish_seal(
            args.worktree,
            remote=args.remote,
            branch=args.branch,
            token=os.environ.get("GITHUB_TOKEN"),
        )
    except (LedgerBranchError, OSError) as error:
        print(f"ledger-branch: {error}", file=sys.stderr)
        return 1
    if published.pushed:
        print(f"pushed {(published.commit or '')[:12]} {published.message}")
    else:
        print("nothing to publish")
    return 0


COMMAND = Command(
    name="ledger-branch",
    help="open the orphan ledger branch as a worktree, or commit and push a seal to it",
    run=_run,
    configure=_configure,
)
