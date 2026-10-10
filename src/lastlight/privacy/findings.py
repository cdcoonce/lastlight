"""What the scanner reports: one `Finding` per leak, printed as `path:line: RULE_ID: excerpt`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ALLOWLIST_PROBLEM = "A001"
"""Rule ID for a malformed allowlist entry. It is a finding like any other, so it fails the
scan, but it names a policy error rather than a leak and cannot itself be allowlisted."""

_EXCERPT_LIMIT = 80


@dataclass(frozen=True, order=True)
class Finding:
    """One leak (or one policy error) at a place in a file.

    `path` is repo-relative with forward slashes when the file is inside the scan root, and
    otherwise the path as given. `excerpt` is the matched text, shown to the reader.
    `context` is the whole source line(s) the match sits on; allowlist `match` strings are
    searched there, so an entry can name a stable fragment of the line.
    """

    path: str
    line: int
    rule: str
    excerpt: str
    context: str = field(default="", compare=False)

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.excerpt}"


def make_excerpt(matched: str) -> str:
    """The matched text on one line, whitespace collapsed, capped so a log line stays short."""
    text = " ".join(matched.split())
    if len(text) > _EXCERPT_LIMIT:
        return text[: _EXCERPT_LIMIT - 3] + "..."
    return text


def relative_path(path: Path, root: Path) -> str | None:
    """`path` relative to `root` as a forward-slash string, or None when it is outside."""
    absolute = Path(os.path.abspath(path))
    for base in (Path(os.path.abspath(root)), root.resolve()):
        try:
            return absolute.relative_to(base).as_posix()
        except ValueError:
            continue
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
