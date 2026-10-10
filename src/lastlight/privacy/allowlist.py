"""Exceptions to the privacy rules: `privacy-allowlist.toml`, plus the scanner's own files.

An entry names one place a rule may be broken and says why:

    [[allow]]
    path = "uv.lock"          # repo-relative, forward slashes; * is a glob (it spans /)
    rule = "P001"             # one of P001 ... P005
    match = "upload-time"     # literal text that must appear on the finding's source line
    reason = "PyPI release stamps of third-party wheels"

An entry suppresses a finding only when path, rule and match all agree, so it covers its own
lines and nothing else. An entry with a missing or blank `reason`, or any other malformed
field, is itself reported (rule A001), suppresses nothing, and fails the scan.

The scanner's own source and tests have to spell out the patterns they detect, so they are
exempt by path (`SELF_EXEMPT`). That list is code, not configuration, and is deliberately
narrow: it never covers image metadata (P004), and only the site module and the scanner's
tests may name the private-site sentinel (P005).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from lastlight.privacy.findings import ALLOWLIST_PROBLEM, Finding, make_excerpt, relative_path
from lastlight.privacy.rules import RULES

_ENTRY_KEYS = ("path", "rule", "match", "reason")
_TEXT_RULES = frozenset({"P001", "P002", "P003"})

SELF_EXEMPT: tuple[tuple[str, frozenset[str]], ...] = (
    ("src/lastlight/privacy/*", _TEXT_RULES),
    ("tests/test_privacy*.py", _TEXT_RULES | {"P005"}),
    ("src/lastlight/site.py", frozenset({"P005"})),
    ("privacy-allowlist.toml", _TEXT_RULES),
)
"""(path glob, rules) pairs a file may break without an allowlist entry."""


def is_self_exempt(relative: str, rule: str) -> bool:
    return any(fnmatchcase(relative, glob) and rule in rules for glob, rules in SELF_EXEMPT)


@dataclass(frozen=True)
class AllowEntry:
    path: str
    rule: str
    match: str
    reason: str

    def covers(self, relative: str, finding: Finding) -> bool:
        return (
            finding.rule == self.rule
            and fnmatchcase(relative, self.path)
            and self.match in finding.context
        )


@dataclass(frozen=True)
class Allowlist:
    """The valid entries, and one A001 finding per invalid entry or unreadable file."""

    entries: tuple[AllowEntry, ...] = ()
    problems: tuple[Finding, ...] = ()

    def suppresses(self, relative: str, finding: Finding) -> bool:
        return any(entry.covers(relative, finding) for entry in self.entries)


def load_allowlist(path: Path, *, root: Path | None = None) -> Allowlist:
    """Read `path`; problems are named relative to `root` when the file sits inside it."""
    display = (relative_path(path, root) if root is not None else None) or str(path)
    text = path.read_text(encoding="utf-8")

    def problem(line: int, message: str) -> Finding:
        return Finding(display, line, ALLOWLIST_PROBLEM, make_excerpt(message), message)

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        return Allowlist(problems=(problem(1, f"not valid TOML: {error}"),))

    problems = [
        problem(1, f"unknown top-level key {key!r}; entries go in [[allow]] tables")
        for key in data
        if key != "allow"
    ]
    raw_entries = data.get("allow", [])
    if not isinstance(raw_entries, list) or not all(isinstance(e, dict) for e in raw_entries):
        problems.append(problem(1, "allow must be an array of tables ([[allow]])"))
        return Allowlist(problems=tuple(problems))

    header_lines = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if re.match(r"\s*\[\[\s*allow\s*\]\]", line)
    ]
    entries: list[AllowEntry] = []
    for index, raw in enumerate(raw_entries, start=1):
        errors = _entry_errors(raw)
        if errors:
            line = header_lines[index - 1] if index <= len(header_lines) else 1
            where = f"entry {index} (path={raw.get('path')!r}, rule={raw.get('rule')!r})"
            problems.append(problem(line, f"{where}: {'; '.join(errors)}"))
        else:
            entries.append(AllowEntry(*(raw[key] for key in _ENTRY_KEYS)))
    return Allowlist(entries=tuple(entries), problems=tuple(problems))


def _entry_errors(raw: dict[str, object]) -> list[str]:
    errors = [f"unknown key {key!r}" for key in raw if key not in _ENTRY_KEYS]
    for key in _ENTRY_KEYS:
        value = raw.get(key)
        if key not in raw:
            errors.append(f"missing key {key!r}")
        elif not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")
    rule = raw.get("rule")
    if isinstance(rule, str) and rule.strip() and rule not in RULES:
        errors.append(f"rule must be one of {', '.join(RULES)}, not {rule!r}")
    return errors
