"""Walk the files to scan and apply every rule to each one.

The default target is the repository's git-tracked files plus `site/` (the generated
almanac), so the scan sees everything that is published. Explicit paths replace that default.
Symlinks are never followed. A file the scanner cannot classify as an image is scanned as text
unless it holds NUL bytes, in which case only the byte-level rules (P004, P005) apply.
"""

from __future__ import annotations

import os
import re
import subprocess
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lastlight.privacy.allowlist import Allowlist, is_self_exempt
from lastlight.privacy.findings import Finding, make_excerpt, relative_path
from lastlight.privacy.images import image_issues
from lastlight.privacy.rules import text_hits
from lastlight.site import PRIVATE_SITE_SENTINEL

_BINARY_PROBE = 8192
_CONTEXT_LIMIT = 2000
SITE_DIRECTORY = "site"


class ScanError(Exception):
    """The scan could not run (bad path, no git repository); distinct from finding a leak."""


@dataclass(frozen=True)
class ScanResult:
    findings: list[Finding]
    files_scanned: int


def find_root(start: Path) -> Path:
    """The git work tree containing `start`, or `start` itself outside any repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return start
    return Path(result.stdout.strip()) if result.returncode == 0 else start


def scan(
    paths: Sequence[Path] | None = None,
    *,
    root: Path,
    allowlist: Allowlist | None = None,
) -> ScanResult:
    """Scan `paths` (files or directories), or the tracked tree plus `site/` when None.

    Findings come back sorted, with allowlisted and self-exempt ones removed and the
    allowlist's own problems added. Only files inside `root` can be exempt or allowlisted.
    """
    files = default_targets(root) if paths is None else expand(paths)
    findings: list[Finding] = list(allowlist.problems) if allowlist is not None else []
    for file in files:
        relative = relative_path(file, root)
        display = relative if relative is not None else str(file)
        for finding in scan_file(file, display):
            if relative is not None and _is_excused(relative, finding, allowlist):
                continue
            findings.append(finding)
    return ScanResult(sorted(findings), len(files))


def _is_excused(relative: str, finding: Finding, allowlist: Allowlist | None) -> bool:
    if is_self_exempt(relative, finding.rule):
        return True
    return allowlist is not None and allowlist.suppresses(relative, finding)


# --- which files ---------------------------------------------------------------------------


def default_targets(root: Path) -> list[Path]:
    """Every git-tracked file under `root`, plus everything under `root/site` if it exists."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise ScanError(
            "git is not installed; pass PATH arguments to scan files directly"
        ) from error
    if listing.returncode != 0:
        raise ScanError(
            f"{root} is not a git repository; pass PATH arguments to scan files directly"
        )
    tracked = [root / os.fsdecode(name) for name in listing.stdout.split(b"\0") if name]
    site = root / SITE_DIRECTORY
    generated = list(_walk(site)) if site.is_dir() and not site.is_symlink() else []
    return _scannable([*tracked, *generated])


def expand(paths: Iterable[Path]) -> list[Path]:
    """Explicit PATH arguments: files as given, directories walked."""
    found: list[Path] = []
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_dir():
            found.extend(_walk(path))
        elif path.is_file():
            found.append(path)
        else:
            raise ScanError(f"no such file or directory: {path}")
    return _scannable(found)


def _walk(directory: Path) -> Iterable[Path]:
    for current, subdirectories, names in os.walk(directory):
        subdirectories[:] = sorted(name for name in subdirectories if name != ".git")
        for name in sorted(names):
            yield Path(current) / name


def _scannable(paths: Iterable[Path]) -> list[Path]:
    """Regular files only (no symlinks, no deleted-but-tracked entries), each once."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = Path(os.path.abspath(path))
        if key in seen or path.is_symlink() or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    return unique


# --- one file ------------------------------------------------------------------------------


def scan_file(path: Path, display: str) -> list[Finding]:
    """Every finding in one file, before any allowlist is applied."""
    data = path.read_bytes()
    findings = _sentinel_findings(data, display)
    issues = image_issues(data, path.suffix)
    if issues is not None:
        findings += [Finding(display, 1, "P004", issue, issue) for issue in issues]
    elif b"\x00" not in data[:_BINARY_PROBE]:
        findings += _text_findings(data.decode("utf-8", errors="replace"), display)
    return findings


def _sentinel_findings(data: bytes, display: str) -> list[Finding]:
    """P005 works on raw bytes so it also sees the sentinel inside binary files."""
    needle = PRIVATE_SITE_SENTINEL.encode()
    findings: list[Finding] = []
    position = data.find(needle)
    while position != -1:
        line_start = data.rfind(b"\n", 0, position) + 1
        line_end = data.find(b"\n", position)
        line_end = len(data) if line_end == -1 else line_end
        context = data[line_start:line_end][:_CONTEXT_LIMIT].decode("utf-8", errors="replace")
        line = data.count(b"\n", 0, position) + 1
        findings.append(Finding(display, line, "P005", PRIVATE_SITE_SENTINEL, context))
        position = data.find(needle, position + len(needle))
    return findings


def _text_findings(text: str, display: str) -> list[Finding]:
    line_starts = [0] + [newline.end() for newline in re.finditer("\n", text)]
    findings: list[Finding] = []
    for hit in text_hits(text):
        first_line = bisect_right(line_starts, hit.start)
        last_line = bisect_right(line_starts, max(hit.end - 1, hit.start))
        context_end = line_starts[last_line] - 1 if last_line < len(line_starts) else len(text)
        context = text[line_starts[first_line - 1] : context_end][:_CONTEXT_LIMIT]
        excerpt = make_excerpt(text[hit.start : hit.end])
        findings.append(Finding(display, first_line, hit.rule, excerpt, context))
    return findings
