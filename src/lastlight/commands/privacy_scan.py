"""`lastlight privacy-scan [PATH ...]`: fail when anything public could locate the camera.

Thin by design: it parses arguments and prints findings; the rules live in `lastlight.privacy`.
Exit codes: 0 clean, 1 findings (or an invalid allowlist), 2 the scan could not run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lastlight.cli import Command
from lastlight.privacy import Finding, ScanError, find_root, load_allowlist, scan

ALLOWLIST_NAME = "privacy-allowlist.toml"


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        metavar="PATH",
        help="files or directories to scan; default: every git-tracked file plus site/",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="repository root, which paths are reported against (default: the git work tree)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        help=f"allowlist file (default: {ALLOWLIST_NAME} in the root, if it exists)",
    )


def _run(args: argparse.Namespace) -> int:
    root: Path = args.root if args.root is not None else find_root(Path.cwd())
    allowlist_path: Path = args.allowlist if args.allowlist is not None else root / ALLOWLIST_NAME
    try:
        if args.allowlist is not None and not allowlist_path.is_file():
            raise ScanError(f"no such allowlist file: {allowlist_path}")
        allowlist = load_allowlist(allowlist_path, root=root) if allowlist_path.is_file() else None
        result = scan(args.paths or None, root=root, allowlist=allowlist)
    except (ScanError, OSError) as error:
        print(f"privacy-scan: {error}", file=sys.stderr)
        return 2
    for finding in result.findings:
        print(finding.format())
    print(_summary(result.findings, result.files_scanned), file=sys.stderr)
    return 1 if result.findings else 0


def _summary(findings: list[Finding], files_scanned: int) -> str:
    if not findings:
        return f"privacy-scan: clean ({files_scanned} files scanned)"
    files = len({finding.path for finding in findings})
    noun = "finding" if len(findings) == 1 else "findings"
    return f"privacy-scan: {len(findings)} {noun} in {files} file(s) ({files_scanned} scanned)"


COMMAND = Command(
    name="privacy-scan",
    help="fail if any file could reveal the camera site (clock times, coordinates, metadata)",
    run=_run,
    configure=_configure,
)
