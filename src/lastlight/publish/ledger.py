"""Read a ledger for the almanac: which evenings were sealed, what they say, whether they verify.

The ledger is the directory `lastlight seal` writes (`lastlight.seal.ledger`), plus one file the
daily forecast adds beside each sealed forecast, its painting: `YYYY/MM/DD.strip.png`. The list
of dates comes from `index.jsonl`, newest first.

Whether an evening was sealed in time is never taken from a file or a clock. It is the result of
`lastlight.seal.verify`, run again here, so the site shows the same PASS or reason that anyone
running `lastlight verify` gets. An evening that does not verify is still listed, marked, with the
reason; it is never featured as a sealed painting.

What an evening says about itself is read from the forecast file with no schema check, because
the schema is `verify`'s job and the site must be able to show (and the privacy scan to catch)
whatever a ledger really holds. Fields of the wrong type are ignored, never trusted.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lastlight.seal import digest_hex, ledger_paths, verify
from lastlight.seal.ledger import INDEX_NAME

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Evening:
    """One sealed date as the site shows it."""

    date: dt.date
    digest: str
    """SHA-256 of the sealed forecast file as it sits on disk (the index's, if the file is gone)."""
    model_version: str
    model_name: str
    vividness: tuple[float, float] | None
    """The forecast's vividness (mean, spread), if the file holds both."""
    sun_elevations: tuple[float, ...]
    """The distinct sun elevations of the forecast curve in degrees, highest first."""
    view_elevations: tuple[float, ...]
    """The distinct view elevations of the forecast curve in degrees, lowest first."""
    notes: str
    verified: bool
    reason: str
    """Why `lastlight verify` failed; empty when it passed."""
    strip: Path | None
    """The painting on disk, or None when the date has none."""


@dataclass(frozen=True)
class Ledger:
    evenings: tuple[Evening, ...]
    """Newest first."""
    found: bool
    """False when the ledger directory does not exist."""
    skipped_index_lines: int
    """Index lines that were not usable (malformed, or a date already listed) and were left out."""


@dataclass(frozen=True)
class _Row:
    date: dt.date
    digest: str
    model_version: str


def strip_path(ledger_dir: Path, date: dt.date) -> Path:
    """Where the painting of `date` lives: beside its forecast, as `DD.strip.png`."""
    return ledger_paths(ledger_dir, date).forecast.with_name(f"{date.day:02d}.strip.png")


def read_ledger(ledger_dir: Path) -> Ledger:
    """Every indexed evening, newest first. A missing ledger directory is an empty ledger."""
    if not ledger_dir.is_dir():
        return Ledger(evenings=(), found=False, skipped_index_lines=0)
    rows, skipped = _read_index(ledger_dir / INDEX_NAME)
    newest_first = sorted(rows, key=lambda row: row.date, reverse=True)
    evenings = tuple(_read_evening(row, ledger_dir) for row in newest_first)
    return Ledger(evenings=evenings, found=True, skipped_index_lines=skipped)


# --- the index ------------------------------------------------------------------------------


def _read_index(index: Path) -> tuple[list[_Row], int]:
    if not index.is_file():
        return [], 0
    rows: dict[dt.date, _Row] = {}
    skipped = 0
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = _parse_row(line)
        if row is None or row.date in rows:
            skipped += 1
        else:
            rows[row.date] = row
    return list(rows.values()), skipped


def _parse_row(line: str) -> _Row | None:
    try:
        data = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    date, digest, version = data.get("date"), data.get("digest"), data.get("model_version")
    if not (
        isinstance(date, str)
        and _DATE.fullmatch(date)
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest)
        and isinstance(version, str)
    ):
        return None
    try:
        return _Row(dt.date.fromisoformat(date), digest, version)
    except ValueError:
        return None


# --- one evening ----------------------------------------------------------------------------


def _read_evening(row: _Row, ledger_dir: Path) -> Evening:
    outcome = verify(row.date, ledger_dir)
    forecast = ledger_paths(ledger_dir, row.date).forecast
    sealed = forecast.read_bytes() if forecast.is_file() else None
    digest = digest_hex(sealed) if sealed is not None else row.digest
    verified, reason = outcome.ok, outcome.reason or ""
    if not verified and not reason:
        reason = "verification failed"
    if verified and digest != row.digest:
        verified, reason = False, "the index lists a different digest than the sealed forecast"

    document = _document(sealed)
    model = _mapping(document.get("model"))
    vividness = _mapping(document.get("vividness"))
    mean, spread = _number(vividness.get("mean")), _number(vividness.get("sd"))
    points = document.get("curve")
    curve = [_mapping(point) for point in points] if isinstance(points, list) else []
    painting = strip_path(ledger_dir, row.date)
    return Evening(
        date=row.date,
        digest=digest,
        model_version=_text(model.get("version")) or row.model_version,
        model_name=_text(model.get("name")),
        vividness=(mean, spread) if mean is not None and spread is not None else None,
        sun_elevations=_elevations(curve, "sun_elevation", highest_first=True),
        view_elevations=_elevations(curve, "view_elevation", highest_first=False),
        notes=_text(document.get("notes")),
        verified=verified,
        reason=reason,
        strip=painting if painting.is_file() and not painting.is_symlink() else None,
    )


def _document(sealed: bytes | None) -> dict[str, Any]:
    if sealed is None:
        return {}
    try:
        document = json.loads(sealed)
    except (ValueError, RecursionError):
        return {}
    return document if isinstance(document, dict) else {}


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _elevations(curve: list[dict[str, Any]], key: str, *, highest_first: bool) -> tuple[float, ...]:
    found = {number for point in curve if (number := _number(point.get(key))) is not None}
    return tuple(sorted(found, reverse=highest_first))
