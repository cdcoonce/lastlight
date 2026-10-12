"""`lastlight seal-daily --ledger-dir DIR [--date D] [--attempts N] [--retry-pause SECONDS]`.

The scheduled seal, in one command. Paints the evening, seals it into the ledger and checks the
seal (`lastlight.daily.seal_daily`). It is what `.github/workflows/seal.yml` runs. If the
evening is already sealed it changes nothing and exits 0; if the sun has already reached the
horizon at the reference point it refuses (a late day is never back-filled).

`--attempts N` (default 1) tries again after `--retry-pause` seconds (default 300) when an
attempt fails for a reason a pause can cure (the weather service, the timestamp authority, the
disk), and only while the pause still ends before the sun is at the horizon
(`lastlight.daily.seal_with_retry`). Every failed attempt is named on stderr. The workflow asks
for two attempts; a run by hand makes one.

Exit codes are the last attempt's: 0 sealed, or already sealed and still verifying; 1 failed
(data, timestamp authority, disk, or a seal that does not verify; the reason is on stderr); 2
usage error; 3 refused as too late.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands._arguments import iso_date
from lastlight.commands.forecast import configure_run_options
from lastlight.inputs.http import Transport, UrllibTransport
from lastlight.seal import DEFAULT_TSA_URL

EXIT_TOO_LATE = 3
DEFAULT_ATTEMPTS = 1
DEFAULT_RETRY_PAUSE_S = 300.0


def _attempts(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number of seconds, got {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number of seconds, not below 0: {text!r}"
        )
    return value


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


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
    parser.add_argument(
        "--attempts",
        type=_attempts,
        default=DEFAULT_ATTEMPTS,
        metavar="N",
        help="how many times to try before giving up; a failed attempt is retried only if a "
        "pause can cure it and the evening is still ahead (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-pause",
        type=_seconds,
        default=DEFAULT_RETRY_PAUSE_S,
        metavar="SECONDS",
        help="how long to wait between attempts; no retry is made if the wait would reach the "
        "sun at the horizon (default: %(default)g)",
    )


def execute(
    args: argparse.Namespace,
    *,
    transport: Transport,
    now_utc: dt.datetime,
    clock: Callable[[], dt.datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """Run the command with the transport and clock given (the tests give their own).

    `now_utc` is the first attempt's clock reading; `clock` (read after a failure, and again
    after the pause) and `sleep` serve the retries and default to the real ones.
    """
    from lastlight.atmosphere import ReductionError
    from lastlight.daily import (
        DailyResult,
        SealVerificationError,
        TooLateError,
        default_evening,
        seal_daily,
        seal_with_retry,
    )
    from lastlight.inputs.hrrr import HrrrError
    from lastlight.inputs.http import FetchError
    from lastlight.seal import AlreadySealedError, StripError, TsaError

    evening = args.date if args.date is not None else default_evening(now_utc)

    def attempt(now: dt.datetime) -> DailyResult:
        return seal_daily(
            evening,
            args.ledger_dir,
            transport=transport,
            now_utc=now,
            cache_dir=args.cache_dir,
            cycle_hour=args.cycle,
            forecast_hour=args.fhour,
            tsa_url=args.tsa_url,
        )

    def say(message: str) -> None:
        print(f"seal-daily: {message}", file=sys.stderr)

    try:
        result = seal_with_retry(
            attempt,
            evening,
            now_utc=now_utc,
            attempts=args.attempts,
            pause_seconds=args.retry_pause,
            clock=clock if clock is not None else _utc_now,
            sleep=sleep if sleep is not None else time.sleep,
            note=say,
        )
    except TooLateError:
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
    ):
        return 1
    label = "sealed" if result.status == "sealed" else "already sealed"
    print(f"{label} {result.date.isoformat()} {result.digest}")
    return 0


def _run(args: argparse.Namespace) -> int:
    return execute(
        args,
        transport=UrllibTransport(),
        now_utc=_utc_now(),
        clock=_utc_now,
        sleep=time.sleep,
    )


COMMAND = Command(
    name="seal-daily",
    help="paint and seal the coming evening into the ledger (idempotent; never seals late)",
    run=_run,
    configure=_configure,
)
