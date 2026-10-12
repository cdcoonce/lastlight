"""`lastlight seal FORECAST.json --ledger-dir DIR`: fix a forecast in the ledger before sunset.

Thin by design: it reads and validates the forecast, then calls `lastlight.seal.ledger.seal`,
which writes the canonical forecast, gets it timestamped and indexes it. A forecast that names a
strip (`strip_sha256`) is sealed together with that PNG: pass it with `--strip PNG`, and it is
written beside the forecast in the ledger.
Exit codes: 0 sealed, 1 sealing failed (the timestamp authority, a reply from it that does not
verify against the pinned certificates, or the disk; nothing is left behind), 2 refused (the
forecast is unreadable or invalid, the strip is missing, unreadable or not the one the forecast
names, or the date is already sealed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from lastlight.cli import Command
from lastlight.forecast import Forecast
from lastlight.seal import DEFAULT_TSA_URL, AlreadySealedError, StripError, TsaError, seal


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("forecast", type=Path, metavar="FORECAST.json", help="the forecast to seal")
    parser.add_argument(
        "--strip",
        type=Path,
        metavar="PNG",
        help="the strip PNG the forecast names (required exactly when it names one)",
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="the ledger directory; the sealed files and index.jsonl are written here",
    )
    parser.add_argument(
        "--tsa-url",
        default=DEFAULT_TSA_URL,
        metavar="URL",
        help="RFC 3161 timestamp authority (default: %(default)s)",
    )


def _problems(error: ValidationError) -> list[str]:
    """One line per problem, naming the field and never repeating the offending value."""
    lines = []
    for issue in error.errors(include_input=False, include_url=False, include_context=False):
        field = ".".join(str(step) for step in issue["loc"]) or "(whole document)"
        lines.append(f"  {field}: {issue['msg']}")
    return lines


def _run(args: argparse.Namespace) -> int:
    try:
        raw = args.forecast.read_bytes()
    except OSError as error:
        print(f"seal: cannot read {args.forecast}: {error.strerror}", file=sys.stderr)
        return 2
    try:
        forecast = Forecast.model_validate_json(raw)
    except ValidationError as error:
        print(f"seal: {args.forecast} is not a valid forecast:", file=sys.stderr)
        print("\n".join(_problems(error)), file=sys.stderr)
        return 2
    strip = None
    if args.strip is not None:
        try:
            strip = args.strip.read_bytes()
        except OSError as error:
            print(f"seal: cannot read {args.strip}: {error.strerror}", file=sys.stderr)
            return 2
    try:
        result = seal(forecast, args.ledger_dir, strip=strip, tsa_url=args.tsa_url)
    except (AlreadySealedError, StripError) as error:
        print(f"seal: {error}", file=sys.stderr)
        return 2
    except TsaError as error:
        print(f"seal: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"seal: could not write the ledger: {error}", file=sys.stderr)
        return 1
    print(f"sealed {result.date.isoformat()} {result.digest}")
    return 0


COMMAND = Command(
    name="seal",
    help="canonicalise, hash and RFC 3161 timestamp a forecast into the ledger",
    run=_run,
    configure=_configure,
)
