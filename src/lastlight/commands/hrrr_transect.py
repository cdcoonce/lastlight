"""`lastlight hrrr-transect --date D [--cycle HH --fhour N]`: HRRR along the sunlight's path.

Prints a table of the default HRRR fields at distances 0, 25, ..., 400 km from the public
reference point along the sunset azimuth of the evening D. Thin by design: it parses arguments
and prints; the work is in `lastlight.inputs.hrrr`, which is imported only when the command
runs, because it loads the GRIB library.

Exit codes: 0 printed, 1 the data could not be fetched or does not cover the transect
(the reason is on stderr), 2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from lastlight import geometry
from lastlight.cli import Command
from lastlight.inputs.http import FetchError, Transport, UrllibTransport

# The public reference point (docs/design.md section 3), downtown Phoenix.
REFERENCE_LAT = 33.45
REFERENCE_LON = -112.07

DEFAULT_CACHE_DIR = Path("data/hrrr")  # under /data/, which git ignores
_MAX_FORECAST_HOUR = 48


def _bounded_int(low: int, high: int, what: str):  # type: ignore[no-untyped-def]
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{what} must be a whole number, got {text!r}"
            ) from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{what} must be {low}..{high}, got {value}")
        return value

    return parse


def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {text!r}") from None


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", type=_date, required=True, help="the evening, as YYYY-MM-DD")
    parser.add_argument(
        "--cycle",
        type=_bounded_int(0, 23, "cycle hour"),
        metavar="HH",
        help="HRRR cycle hour (UTC): the latest such cycle at or before the sun reaching the "
        "horizon (default: the analysis of that hour, or the newest complete run if it is "
        "still ahead)",
    )
    parser.add_argument(
        "--fhour",
        type=_bounded_int(0, _MAX_FORECAST_HOUR, "forecast hour"),
        metavar="N",
        help="forecast hour (default: the hours from the cycle to the sun reaching the horizon)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"where downloaded byte ranges are cached (default: {DEFAULT_CACHE_DIR})",
    )


def execute(args: argparse.Namespace, *, transport: Transport, now_utc: dt.datetime) -> int:
    """Run the command with the transport and clock given (the tests give their own)."""
    from lastlight.inputs import hrrr  # deferred: importing it loads the GRIB library

    distances = [float(distance) for distance in hrrr.TRANSECT_DISTANCES_KM]
    try:
        plan = hrrr.plan_run(
            args.date,
            REFERENCE_LAT,
            REFERENCE_LON,
            cycle_hour=args.cycle,
            forecast_hour=args.fhour,
            now_utc=now_utc,
            transport=transport,
        )
        fields = hrrr.fetch_fields(
            plan.cycle_date,
            plan.cycle_hour,
            plan.forecast_hour,
            hrrr.DEFAULT_FIELDS,
            args.cache_dir,
            transport=transport,
        )
        points = geometry.transect(args.date, REFERENCE_LAT, REFERENCE_LON, distances)
        samples = hrrr.sample_transect(fields, points)
    except (hrrr.HrrrError, FetchError, ValueError) as error:
        print(f"hrrr-transect: {error}", file=sys.stderr)
        return 1
    print(
        f"HRRR 3 km surface fields, run of {plan.cycle_date} (cycle hour {plan.cycle_hour}), "
        f"forecast hour {plan.forecast_hour}"
    )
    print(
        f"evening of {args.date}: great circle from the public reference point toward the "
        f"sunset azimuth, every {int(distances[1])} km to {int(distances[-1])} km"
    )
    print(f"sampling: {hrrr.SAMPLING_METHOD}; '-' means the model has no value there")
    print()
    print(hrrr.format_transect_table(samples, hrrr.DEFAULT_FIELDS, distances))
    return 0


def _run(args: argparse.Namespace) -> int:
    return execute(args, transport=UrllibTransport(), now_utc=dt.datetime.now(dt.UTC))


COMMAND = Command(
    name="hrrr-transect",
    help="print HRRR cloud, aerosol and water-vapour samples along the sunset sightline",
    run=_run,
    configure=_configure,
)
