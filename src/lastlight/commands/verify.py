"""`lastlight verify DATE --ledger-dir DIR`: check that a sealed forecast really was sealed in time.

Thin by design: it parses the date and calls `lastlight.seal.ledger.verify`. It prints `PASS`
(or `PASS (strip verified)` when the forecast names a strip and the PNG in the ledger hashes to
it) or `FAIL: <reason>` on stdout. Exit codes: 0 pass, 1 fail, 2 the arguments were unusable.
This checks the digest, the timestamp's imprint and its time against sunset, and the strip; it
does not yet validate the timestamp's signature chain.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from lastlight.cli import Command
from lastlight.seal import verify

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_date(text: str) -> dt.date:
    if not _ISO_DATE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a date written YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a real calendar date") from error


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "date", type=_iso_date, metavar="DATE", help="the forecast date, YYYY-MM-DD"
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="the ledger directory to read",
    )


def _run(args: argparse.Namespace) -> int:
    outcome = verify(args.date, args.ledger_dir)
    if outcome.ok:
        print("PASS (strip verified)" if outcome.strip_verified else "PASS")
        return 0
    print(f"FAIL: {outcome.reason}")
    return 1


COMMAND = Command(
    name="verify",
    help="check a sealed forecast: digest, timestamp imprint, seal before sunset, and strip",
    run=_run,
    configure=_configure,
)
