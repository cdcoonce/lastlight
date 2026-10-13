"""`lastlight renderer-check`: the pre-registered camera magnitude check for renderer v1.

    lastlight renderer-check --observed-dir DIR --out docs/gates/renderer-v1/
                             [--cache-dir DIR] [--stage fetch|report|all]

Implements `docs/gates/renderer-v1/criterion.md` and nothing else (see `lastlight.renderer_check`).
`--observed-dir` holds the private `YYYY-MM-DD.json` files `lastlight observe` writes for WHTM1.
`fetch` collects each considered evening's inputs (G0's atmosphere and HRRR's total cloud cover)
politely into `<cache-dir>/renderer-v1/`, `report` reads only that and the observed files and
writes `report.md`, `evenings.csv` and `frames.csv` into `--out`, and `all` does both. The verdict
is `ADOPT`, `KEEP V0` or `NOT YET DECIDABLE (k of 5 counted evenings)`; the command refuses to
give one for a v1 whose table or physics is not the registered one.

Exit codes: 0 done, 1 done but a considered evening has no input record yet, 2 usage or input
error, 3 refusal (nothing is written to `--out`).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands.g0 import DEFAULT_CACHE_DIR
from lastlight.inputs.http import Transport, UrllibTransport

STAGES = ("fetch", "report", "all")


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--observed-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="the private directory of WHTM1's YYYY-MM-DD.json files from `lastlight observe`",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="where the report stage writes report.md, evenings.csv and frames.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        metavar="DIR",
        help="where archive byte ranges and the check's per-evening records (under "
        "renderer-v1/) are cached, outside git (default: ~/.cache/lastlight)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="fetch: collect the evenings' inputs into the cache; report: read only the cache "
        "and the observed files and write the outputs; all: both (default: %(default)s)",
    )


def execute(args: argparse.Namespace, *, transport: Transport, now_utc: dt.datetime) -> int:
    """Run the command with the transport and clock given (the tests give their own)."""
    from lastlight.renderer_check.run import run_check

    return run_check(
        observed_dir=args.observed_dir,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        stage=args.stage,
        transport=transport,
        now_utc=now_utc,
    )


def _run(args: argparse.Namespace) -> int:
    from lastlight.g0.transport import PoliteTransport

    transport = PoliteTransport(UrllibTransport())
    return execute(args, transport=transport, now_utc=dt.datetime.now(dt.UTC))


COMMAND = Command(
    name="renderer-check",
    help="the camera magnitude check that decides whether renderer v1 may paint the forecast",
    run=_run,
    configure=_configure,
)

__all__ = ["COMMAND", "DEFAULT_CACHE_DIR", "execute"]
