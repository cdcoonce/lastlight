"""`lastlight g0`: the G0 spread study over a period of archived inputs, resumable in chunks.

    lastlight g0 --start 2025-10-01 --end 2026-09-30 --out docs/gates/G0/
                 [--cache-dir DIR] [--stage fetch|report|all] [--budget-seconds N]
                 [--engine v0|v1]

`fetch` downloads (politely: one request at a time, at most two a second) and renders each date
that has no record in the cache yet, saving each record as it is made, so the run can be split
into chunks (one `--start`/`--end` per month, say) or stopped and resumed; a date that is not in
an archive is recorded as dropped with its reason, a network fault is retried on the next run.
`report` reads only the cache, with no network at all, and writes `report.md`, `results.csv` and
the figures into `--out`. `all` (the default) does both. The cache lives outside git, by default
under ~/.cache/lastlight; see `lastlight.g0` for what the study computes.

With `--engine v1` (default `v0`, which is everything above, unchanged) the study is re-run with
renderer v1 from the v0 cache: `fetch` makes no request, it re-renders each date that has a v0
record in `<cache-dir>/g0/records` from that record's atmosphere into the sibling store
`<cache-dir>/g0-v1/records` (resumably, a dropped date staying dropped), and `report` reads only
that sibling store. The v0 store, and whatever is in a v0 `--out`, are not touched.

Exit codes: 0 done, 1 not every date has a record (a fault, a time budget that stopped the
fetch, or dates the report stage found uncollected; the report is still written and lists them),
2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands._arguments import iso_date
from lastlight.inputs.http import FetchError, Transport, UrllibTransport

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lastlight"
MAX_PERIOD_DAYS = 800
"""A period longer than this is refused: it is a typo or a different study."""

STAGES = ("fetch", "report", "all")
ENGINES = ("v0", "v1")

_PROG = "lastlight g0"


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start", required=True, type=iso_date, metavar="YYYY-MM-DD", help="first evening"
    )
    parser.add_argument(
        "--end", required=True, type=iso_date, metavar="YYYY-MM-DD", help="last evening"
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="where the report stage writes report.md, results.csv and the figures",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="DIR",
        help="where archive byte ranges and per-date records are cached, outside git "
        "(default: ~/.cache/lastlight)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="fetch: download and render into the cache; report: read only the cache and write "
        "the outputs; all: both (default: %(default)s)",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        metavar="N",
        help="stop the fetch stage between dates after this many seconds, so a chunk can be "
        "kept under a time limit; rerun to continue",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="v0",
        help="the renderer: v0 is the study as registered; v1 re-renders the v0 cache with "
        "renderer v1 into a sibling store, with no network (default: %(default)s)",
    )


def _refuse(message: str) -> int:
    print(f"{_PROG}: error: {message}", file=sys.stderr)
    return 2


def execute(args: argparse.Namespace, *, transport: Transport, now_utc: dt.datetime) -> int:
    """Run the command with the transport and clock given (the tests give their own)."""
    if args.end < args.start:
        return _refuse(f"the end {args.end} is before the start {args.start}")
    days = [args.start + dt.timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    if len(days) > MAX_PERIOD_DAYS:
        return _refuse(f"{len(days)} dates is more than the {MAX_PERIOD_DAYS} this study allows")
    if args.budget_seconds is not None and args.budget_seconds < 0:
        return _refuse("the budget must not be negative")
    if getattr(args, "engine", "v0") == "v1":
        return _execute_v1(args, days)

    # Deferred: these import the GRIB library, which loads a native one.
    from lastlight.g0.analysis import build_study
    from lastlight.g0.collect import Collector, build_views, collect_range
    from lastlight.g0.record import RecordStore
    from lastlight.g0.report import write_outputs

    store = RecordStore(args.cache_dir / "g0" / "records")
    code = 0
    if args.stage in ("fetch", "all"):
        collector = Collector(transport, args.cache_dir, now_utc)
        summary = collect_range(
            days, store, collector, progress=print, budget_seconds=args.budget_seconds
        )
        print(
            f"{summary.collected} collected, {summary.dropped} dropped, "
            f"{summary.skipped} skipped, {summary.faults} faults"
        )
        if summary.faults:
            print(
                f"{_PROG}: {summary.faults} dates had a network fault and have no record; "
                "run again to retry them",
                file=sys.stderr,
            )
            code = 1
        if summary.stopped_early:
            print(
                f"stopped at the time budget with {len(summary.remaining)} dates left; "
                "run again to continue"
            )
            code = 1

    if args.stage in ("report", "all"):
        records = [record for day in days if (record := store.load(day)) is not None]
        try:
            study = build_study(records, args.start, args.end)
        except ValueError as error:
            print(f"{_PROG}: {error}", file=sys.stderr)
            return 1
        write_outputs(study, args.out, views=build_views())
        print(
            f"{len(study.attempted)} attempted, {len(study.used)} used, "
            f"{len(study.dropped)} dropped"
        )
        verdict = study.verdict
        print(
            f"G0: {'PASS' if verdict.passed else 'FAIL'} "
            f"({verdict.meeting} of {verdict.dates} dates used meet the rule, "
            f"{100 * verdict.fraction:.1f} percent; 25 percent needed)"
        )
        print(f"wrote the report, results and figures to {args.out}")
        missing = sum(1 for record in study.dropped if record.drop_reason == "not_collected")
        if missing:
            print(f"{_PROG}: {missing} not collected: no record for those dates", file=sys.stderr)
            code = 1
    return code


def _execute_v1(args: argparse.Namespace, days: list[dt.date]) -> int:
    """`--engine v1`: re-render the v0 cache into the sibling store, and report from that store."""
    # Deferred, as in `execute`: these import the GRIB library.
    from lastlight.g0.analysis import build_study
    from lastlight.g0.collect import build_views, rerender_range
    from lastlight.g0.record import RecordStore
    from lastlight.g0.report import write_outputs

    source = RecordStore(args.cache_dir / "g0" / "records")
    store = RecordStore(args.cache_dir / "g0-v1" / "records")
    code = 0
    if args.stage in ("fetch", "all"):
        summary = rerender_range(
            days, source, store, build_views(), progress=print, budget_seconds=args.budget_seconds
        )
        print(
            f"{summary.collected} re-rendered with v1, {summary.dropped} dropped, "
            f"{summary.skipped} skipped, {len(summary.unrecorded)} without a v0 record"
        )
        if summary.unrecorded:
            print(
                f"{_PROG}: {len(summary.unrecorded)} dates have no v0 record to re-render; "
                "collect them with the v0 run first",
                file=sys.stderr,
            )
            code = 1
        if summary.stopped_early:
            print(
                f"stopped at the time budget with {len(summary.remaining)} dates left; "
                "run again to continue"
            )
            code = 1

    if args.stage in ("report", "all"):
        records = [record for day in days if (record := store.load(day)) is not None]
        try:
            study = build_study(records, args.start, args.end, engine="v1")
        except ValueError as error:
            print(f"{_PROG}: {error}", file=sys.stderr)
            return 1
        write_outputs(study, args.out, views=build_views())
        print(
            f"{len(study.attempted)} attempted, {len(study.used)} used, "
            f"{len(study.dropped)} dropped"
        )
        verdict = study.verdict
        print(
            f"G0 (renderer v1): {'PASS' if verdict.passed else 'FAIL'} "
            f"({verdict.meeting} of {verdict.dates} dates used meet the rule, "
            f"{100 * verdict.fraction:.1f} percent; 25 percent needed)"
        )
        print(f"wrote the report, results and figures to {args.out}")
        missing = sum(1 for record in study.dropped if record.drop_reason == "not_collected")
        if missing:
            print(f"{_PROG}: {missing} not collected: no record for those dates", file=sys.stderr)
            code = 1
    return code


def _run(args: argparse.Namespace) -> int:
    from lastlight.g0.transport import PoliteTransport

    transport = PoliteTransport(UrllibTransport())
    return execute(args, transport=transport, now_utc=dt.datetime.now(dt.UTC))


COMMAND = Command(
    name="g0",
    help="the G0 spread study: render a year of archived inputs and write the gate report",
    run=_run,
    configure=_configure,
)

__all__ = ["COMMAND", "DEFAULT_CACHE_DIR", "FetchError", "execute"]
