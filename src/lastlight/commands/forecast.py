"""`lastlight forecast [--date D] --out FILE [--strip PNG]`: paint an evening, seal nothing.

Fetches the HRRR fields for the evening, reduces them to the renderer's v0 parameters, renders
the curve and the strip, and writes the forecast JSON (and the metadata-free strip PNG, whose
SHA-256 the forecast carries). It does not seal: `lastlight seal FORECAST --strip PNG` does that,
and `lastlight seal-daily` does both for the scheduled run. Thin by design: the work is in
`lastlight.painting`, imported only when the command runs because it loads the GRIB library.

Exit codes: 0 written, 1 the data could not be fetched or reduced (the reason is on stderr and
nothing is written), 2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands._arguments import bounded_int, iso_date
from lastlight.inputs.http import FetchError, Transport, UrllibTransport

DEFAULT_CACHE_DIR = Path("data/hrrr")  # under /data/, which git ignores


def configure_run_options(parser: argparse.ArgumentParser) -> None:
    """The options that choose the HRRR run, shared with `seal-daily`."""
    parser.add_argument(
        "--cycle",
        type=bounded_int(0, 23, "cycle hour"),
        metavar="HH",
        help="HRRR cycle hour (UTC), as for `hrrr-transect` (default: chosen from the evening)",
    )
    parser.add_argument(
        "--fhour",
        type=bounded_int(0, 48, "forecast hour"),
        metavar="N",
        help="HRRR forecast hour (default: the hours from the cycle to the sun reaching the "
        "horizon)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"where downloaded byte ranges are cached (default: {DEFAULT_CACHE_DIR})",
    )


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--date",
        type=iso_date,
        help="the evening, as YYYY-MM-DD (default: the evening at the reference point now)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="FILE", help="where to write the forecast JSON"
    )
    parser.add_argument(
        "--strip",
        type=Path,
        metavar="PNG",
        help="also write the strip PNG here; the forecast then carries its SHA-256",
    )
    configure_run_options(parser)


def execute(args: argparse.Namespace, *, transport: Transport, now_utc: dt.datetime) -> int:
    """Run the command with the transport and clock given (the tests give their own)."""
    from lastlight.atmosphere import ReductionError
    from lastlight.daily import default_evening
    from lastlight.inputs.hrrr import HrrrError
    from lastlight.painting import paint

    evening = args.date if args.date is not None else default_evening(now_utc)
    try:
        painting = paint(
            evening,
            transport=transport,
            now_utc=now_utc,
            cache_dir=args.cache_dir,
            cycle_hour=args.cycle,
            forecast_hour=args.fhour,
            with_strip=args.strip is not None,
        )
    except (HrrrError, FetchError, ReductionError, ValueError) as error:
        print(f"forecast: {error}", file=sys.stderr)
        return 1
    try:
        if args.strip is not None and painting.strip_png is not None:
            args.strip.write_bytes(painting.strip_png)
        args.out.write_text(painting.forecast.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"forecast: could not write the output: {error}", file=sys.stderr)
        return 1
    print(f"wrote the forecast for {evening.isoformat()} to {args.out}")
    if args.strip is not None:
        print(f"wrote the strip to {args.strip} (sha256 {painting.forecast.strip_sha256})")
    return 0


def _run(args: argparse.Namespace) -> int:
    return execute(args, transport=UrllibTransport(), now_utc=dt.datetime.now(dt.UTC))


COMMAND = Command(
    name="forecast",
    help="paint an evening's twilight from HRRR and write the forecast JSON and strip PNG",
    run=_run,
    configure=_configure,
)
