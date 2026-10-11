"""`lastlight seal-daily --ledger-dir DIR [--date D]`: the scheduled seal, in one command.

Paints the evening, seals it into the ledger and checks the seal (`lastlight.daily.seal_daily`).
It is what `.github/workflows/seal.yml` runs. If the evening is already sealed it changes
nothing and exits 0; if the sun has already reached the horizon at the reference point it
refuses (a late day is never back-filled). Thin by design: parse, call, print, exit.

Exit codes: 0 sealed, or already sealed and still verifying; 1 failed (data, timestamp
authority, disk, or a seal that does not verify; the reason is on stderr); 2 usage error;
3 refused as too late.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands._arguments import iso_date
from lastlight.commands.forecast import configure_run_options
from lastlight.inputs.http import FetchError, Transport, UrllibTransport
from lastlight.seal import DEFAULT_TSA_URL

EXIT_TOO_LATE = 3


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="the ledger directory the seal is written to",
    )
    parser.add_argument(
        "--date",
        type=iso_date,
        help="the evening, as YYYY-MM-DD (default: the evening at the reference point now)",
    )
    configure_run_options(parser)
    parser.add_argument(
        "--tsa-url",
        default=DEFAULT_TSA_URL,
        metavar="URL",
        help="RFC 3161 timestamp authority (default: %(default)s)",
    )


def execute(args: argparse.Namespace, *, transport: Transport, now_utc: dt.datetime) -> int:
    """Run the command with the transport and clock given (the tests give their own)."""
    from lastlight.atmosphere import ReductionError
    from lastlight.daily import (
        SealVerificationError,
        TooLateError,
        default_evening,
        seal_daily,
    )
    from lastlight.inputs.hrrr import HrrrError
    from lastlight.seal import AlreadySealedError, StripError, TsaError

    evening = args.date if args.date is not None else default_evening(now_utc)
    try:
        result = seal_daily(
            evening,
            args.ledger_dir,
            transport=transport,
            now_utc=now_utc,
            cache_dir=args.cache_dir,
            cycle_hour=args.cycle,
            forecast_hour=args.fhour,
            tsa_url=args.tsa_url,
        )
    except TooLateError as error:
        print(f"seal-daily: {error}", file=sys.stderr)
        return EXIT_TOO_LATE
    except (
        HrrrError,
        FetchError,
        ReductionError,
        TsaError,
        StripError,
        AlreadySealedError,
        SealVerificationError,
        ValueError,
        OSError,
    ) as error:
        print(f"seal-daily: {error}", file=sys.stderr)
        return 1
    label = "sealed" if result.status == "sealed" else "already sealed"
    print(f"{label} {result.date.isoformat()} {result.digest}")
    return 0


def _run(args: argparse.Namespace) -> int:
    return execute(args, transport=UrllibTransport(), now_utc=dt.datetime.now(dt.UTC))


COMMAND = Command(
    name="seal-daily",
    help="paint and seal the coming evening into the ledger (idempotent; never seals late)",
    run=_run,
    configure=_configure,
)
