"""`lastlight site build --ledger-dir DIR --out DIR`: build the almanac from a sealed ledger.

Thin by design: it parses arguments and prints; the work is `lastlight.publish.build_site`.
The build runs the privacy scanner over its own output before it finishes. Exit codes: 0 built,
1 the scan found something (findings are printed and nothing is written), 2 the site could not
be built (an unreadable ledger, or an output directory that must not be replaced).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lastlight.cli import Command
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
    build.set_defaults(run_action=_build)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _build(args: argparse.Namespace) -> int:
    try:
        report = build_site(args.ledger_dir, args.out)
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
    print(
        f"built {_count(report.pages, 'page')} for {_count(report.evenings, 'evening')} "
        f"({report.verified} verified) in {args.out}"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    return int(args.run_action(args))


COMMAND = Command(
    name="site",
    help="build the static almanac site from the sealed ledger",
    run=_run,
    configure=_configure,
)
