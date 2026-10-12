"""`lastlight site build --ledger-dir DIR --out DIR [--through YYYY-MM-DD]`: build the almanac.

Thin by design: it parses arguments and prints; the work is `lastlight.publish.build_site`.
The build runs the privacy scanner over its own output before it finishes. `--through` is the
newest evening the site reports on: an evening between the first sealed one and it that has no
seal is listed as missed. The default, and the latest allowed, is the newest evening whose
deadline has passed: a later `--through` is refused because that evening has not ended. Exit
codes: 0 built, 1 the scan found something (findings are printed and nothing is written), 2 the
site could not be built (an unreadable ledger, an output directory that must not be replaced, or
a `--through` that has not ended) or a usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.commands._arguments import iso_date
from lastlight.publish import BuildError, build_site


def _configure(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, title="actions")
    build = actions.add_parser(
        "build",
        help="build the static almanac and privacy-scan it",
        description="Build the static almanac from a sealed ledger, then privacy-scan it.",
    )
    build.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="the ledger directory itself (the one holding index.jsonl); may be absent",
    )
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="where to write the site; an earlier build there is replaced",
    )
    build.add_argument(
        "--through",
        type=iso_date,
        metavar="YYYY-MM-DD",
        help="the newest evening the site reports on; an evening up to it with no seal is "
        "listed as missed (default, and latest allowed: the newest evening whose sunset has "
        "passed)",
    )
    build.set_defaults(run_action=_build)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _build(args: argparse.Namespace) -> int:
    try:
        report = build_site(args.ledger_dir, args.out, through=args.through)
    except (BuildError, OSError) as error:
        print(f"site build: {error}", file=sys.stderr)
        return 2
    for warning in report.warnings:
        print(f"site build: warning: {warning}", file=sys.stderr)
    for finding in report.findings:
        print(finding.format())
    if report.findings:
        print(
            f"site build: privacy scan failed ({_count(len(report.findings), 'finding')}); "
            f"nothing was written to {args.out}",
            file=sys.stderr,
        )
        return 1
    summary = (
        f"built {_count(report.pages, 'page')} for {_count(report.evenings, 'evening')} "
        f"({report.verified} verified) in {args.out}"
    )
    if report.missed:
        summary += f"; {_count(len(report.missed), 'missed evening')} listed"
    print(summary)
    return 0


def _run(args: argparse.Namespace) -> int:
    return int(args.run_action(args))


COMMAND = Command(
    name="site",
    help="build the static almanac site from the sealed ledger",
    run=_run,
    configure=_configure,
)
