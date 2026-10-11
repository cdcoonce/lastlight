"""`lastlight observe`: an evening's saved frames to `observed.json` (tier A truth).

    lastlight observe --camera WHTM1 --date 2026-10-10 [--archive DIR] --out observed.json
                      [--min-sun-elev -6] [--max-sun-elev 6]

Thin by design: it parses arguments and calls `lastlight.observe.evening`. Exit codes: 0 after the
file is written with at least one frame, 1 when no frame was usable (the file is still written,
with every drop counted, so the evening is on the record), 2 for bad input or configuration
(an unknown camera, a camera without a contrast table or frozen fit, a missing evening folder,
a window that is empty). The output names no frame and carries no clock time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lastlight.cameras import RegistryError, load_registry
from lastlight.cameras.archive import ARCHIVE_ENV_VAR, resolve_archive
from lastlight.camgeom.fitfile import GEOMETRY_DIR
from lastlight.cli import Command
from lastlight.commands._arguments import iso_date
from lastlight.observe.evening import (
    DEFAULT_MAX_SUN_ELEV,
    DEFAULT_MIN_SUN_ELEV,
    DROP_REASONS,
    ObserveError,
    observe_evening,
    write_observed,
)

_PROG = "lastlight observe"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera", required=True, metavar="ID", help="camera id, for example WHTM1"
    )
    parser.add_argument(
        "--date", required=True, type=iso_date, metavar="YYYY-MM-DD", help="the evening's date"
    )
    parser.add_argument(
        "--archive",
        metavar="DIR",
        help=f"private frame archive; default: ${ARCHIVE_ENV_VAR}, else "
        "~/.local/share/lastlight/frames",
    )
    parser.add_argument(
        "--out", required=True, type=Path, metavar="FILE", help="observed.json to write"
    )
    parser.add_argument(
        "--min-sun-elev",
        type=float,
        default=DEFAULT_MIN_SUN_ELEV,
        metavar="DEG",
        help="keep frames with the sun at or above this elevation (default: %(default)s)",
    )
    parser.add_argument(
        "--max-sun-elev",
        type=float,
        default=DEFAULT_MAX_SUN_ELEV,
        metavar="DEG",
        help="keep frames with the sun at or below this elevation (default: %(default)s)",
    )
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        metavar="DIR",
        help="where the frozen fits live (default: the package's cameras/geometry)",
    )


def _refuse(message: str) -> int:
    print(f"{_PROG}: error: {' '.join(message.split())}", file=sys.stderr)
    return 2


def run(args: argparse.Namespace) -> int:
    try:
        registry = load_registry()
    except RegistryError as error:
        return _refuse(f"the camera registry is unusable: {error}")
    camera = registry.get(args.camera)
    if camera is None:
        return _refuse(f"unknown camera {args.camera!r}; known cameras: {', '.join(registry)}")

    try:
        observation = observe_evening(
            camera,
            archive=resolve_archive(args.archive),
            date=args.date,
            geometry_dir=args.geometry_dir or GEOMETRY_DIR,
            max_sun_elev=args.max_sun_elev,
            min_sun_elev=args.min_sun_elev,
        )
        write_observed(observation, args.out)
    except (ObserveError, OSError) as error:
        return _refuse(str(error))

    dropped = ", ".join(
        f"{reason}={observation.dropped[reason]}"
        for reason in DROP_REASONS
        if observation.dropped[reason]
    )
    print(
        f"{camera.id} {args.date.isoformat()}: {len(observation.frames)} of "
        f"{observation.frames_found} frames used, sun {args.max_sun_elev:+g} to "
        f"{args.min_sun_elev:+g} degrees"
        + (f"; dropped: {dropped}" if dropped else "")
        + ("; geometry PROVISIONAL (pixel scale unconfirmed)" if observation.provisional else "")
    )
    print(f"{camera.id}: wrote {args.out}")
    if not observation.frames:
        print(
            f"{_PROG}: no usable frame for {camera.id} on {args.date.isoformat()}; the file "
            "records what was dropped and why",
            file=sys.stderr,
        )
        return 1
    return 0


COMMAND = Command(
    name="observe",
    help="turn an evening's saved camera frames into observed contrast profiles (observed.json)",
    run=run,
    configure=configure,
)
