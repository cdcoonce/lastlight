"""`lastlight collect`: archive the public cameras' frames through each evening's twilight.

Thin by design: this module parses arguments, applies the two refusals that must happen before
anything is created (a too-short interval, an archive inside a git work tree), and calls
`lastlight.cameras.collector.collect`. Exit codes: 0 after the evening's window closes (even if
some polls failed; the summary line counts them), 2 for any refusal or usage error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import TextIO

from lastlight.cameras import RegistryError, load_registry
from lastlight.cameras.archive import ARCHIVE_ENV_VAR, inside_git_worktree, resolve_archive
from lastlight.cameras.clock import Clock, SystemClock
from lastlight.cameras.collector import MIN_INTERVAL_S, CollectorConfig, collect, format_summary
from lastlight.cameras.transport import Fetch, urllib_fetch
from lastlight.cli import Command

_PROG = "lastlight collect"


def _interval_seconds(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of seconds") from None
    if value < MIN_INTERVAL_S:
        raise argparse.ArgumentTypeError(
            f"the interval must be at least {MIN_INTERVAL_S} seconds "
            f"(one poll per camera per 5 minutes); got {value}"
        )
    return value


def configure(parser: argparse.ArgumentParser) -> None:
    defaults = CollectorConfig()
    parser.add_argument(
        "--camera",
        metavar="ID",
        nargs="+",
        action="extend",
        help="camera id to collect (repeatable); default: every camera in the registry",
    )
    parser.add_argument(
        "--archive",
        metavar="DIR",
        help=f"private frame archive; default: ${ARCHIVE_ENV_VAR}, else ~/.local/share/lastlight/"
        "frames. Refused (exit 2) inside a git work tree.",
    )
    parser.add_argument(
        "--start-elev",
        type=float,
        default=defaults.start_elev,
        metavar="DEG",
        help="start polling when the sun falls below this elevation (default: %(default)s)",
    )
    parser.add_argument(
        "--stop-elev",
        type=float,
        default=defaults.stop_elev,
        metavar="DEG",
        help="stop when the sun is below this elevation (default: %(default)s)",
    )
    parser.add_argument(
        "--interval",
        type=_interval_seconds,
        default=defaults.interval_s,
        metavar="SECONDS",
        help=f"seconds between polls of a camera, at least {MIN_INTERVAL_S} (default: %(default)s)",
    )


def _refuse(message: str) -> int:
    print(f"{_PROG}: error: {message}", file=sys.stderr)
    return 2


def run_collect(
    args: argparse.Namespace,
    *,
    clock: Clock | None = None,
    fetch: Fetch | None = None,
    out: TextIO | None = None,
    report: Callable[[str], None] | None = None,
) -> int:
    """Run the command. `clock`, `fetch`, `out` and `report` are injected by tests."""
    try:
        registry = load_registry()
    except RegistryError as error:
        return _refuse(f"the camera registry is unusable: {error}")

    wanted = list(dict.fromkeys(args.camera)) if args.camera else list(registry)
    unknown = [camera_id for camera_id in wanted if camera_id not in registry]
    if unknown:
        return _refuse(f"unknown camera(s) {unknown}; known cameras: {', '.join(registry)}")

    try:
        config = CollectorConfig(
            start_elev=args.start_elev, stop_elev=args.stop_elev, interval_s=args.interval
        )
    except ValueError as error:
        return _refuse(str(error))

    archive = resolve_archive(args.archive)
    if inside_git_worktree(archive):
        return _refuse(
            f"the archive {archive} is inside a git work tree; raw frames must never be "
            f"committed. Choose a directory outside every repository (--archive DIR or "
            f"${ARCHIVE_ENV_VAR})."
        )
    if archive.exists() and not archive.is_dir():
        return _refuse(f"the archive {archive} exists and is not a directory")

    summaries = collect(
        [registry[camera_id] for camera_id in wanted],
        archive,
        clock=SystemClock() if clock is None else clock,
        fetch=urllib_fetch if fetch is None else fetch,
        config=config,
        report=report,
    )
    stream = sys.stdout if out is None else out
    for summary in summaries:
        print(format_summary(summary), file=stream)
    return 0


COMMAND = Command(
    name="collect",
    help="archive public-camera frames through tonight's twilight window (tier A truth)",
    run=run_collect,
    configure=configure,
)
