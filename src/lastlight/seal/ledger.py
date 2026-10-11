"""The ledger: seal a forecast, and verify a sealed date later.

Layout (`docs/design.md` section 4), under a ledger directory:

    YYYY/MM/DD.forecast.json   the canonical forecast bytes, exactly as hashed
    YYYY/MM/DD.forecast.tsr    the RFC 3161 response for the SHA-256 of those bytes
    index.jsonl                one line per sealed forecast: date, digest, model_version

`seal` never overwrites: a date with any trace of a seal is refused before a timestamp is
requested, and a seal that fails part-way removes what it had written, so the ledger is
either unchanged or holds a complete seal. `verify` re-derives everything from the files:
the digest of the sealed bytes must equal the timestamp's imprint, the file must be a valid
forecast for the requested date in canonical form, and the timestamp must have been issued
before geometric sunset at the reference point on that date (`lastlight.geometry`).

`verify` does not validate the timestamp's signature chain against the TSA certificate; that
is a follow-up. It reports durations, never clock times, in its reasons.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from lastlight.forecast import Forecast
from lastlight.geometry import elevation_crossings
from lastlight.seal import tsa
from lastlight.seal.canonical import canonical_bytes, digest_hex
from lastlight.seal.tsa import DEFAULT_TSA_URL, TsaError

INDEX_NAME = "index.jsonl"


class AlreadySealedError(Exception):
    """The date already has a sealed forecast (or the remains of one); nothing was changed."""


@dataclass(frozen=True)
class LedgerPaths:
    forecast: Path
    timestamp: Path
    index: Path


@dataclass(frozen=True)
class SealResult:
    date: dt.date
    digest: str
    paths: LedgerPaths


@dataclass(frozen=True)
class VerifyResult:
    """`ok` with no reason, or not ok with the one reason the first failed check gives."""

    ok: bool
    reason: str | None = None


def ledger_paths(ledger_dir: Path, date: dt.date) -> LedgerPaths:
    folder = ledger_dir / f"{date.year:04d}" / f"{date.month:02d}"
    stem = f"{date.day:02d}.forecast"
    return LedgerPaths(
        forecast=folder / f"{stem}.json",
        timestamp=folder / f"{stem}.tsr",
        index=ledger_dir / INDEX_NAME,
    )


# --- seal ---------------------------------------------------------------------------------


def _indexed(index: Path, date: dt.date) -> bool:
    if not index.is_file():
        return False
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("date") == date.isoformat():
            return True
    return False


def _refuse_if_sealed(date: dt.date, paths: LedgerPaths, ledger_dir: Path) -> None:
    if paths.forecast.exists() or paths.timestamp.exists() or _indexed(paths.index, date):
        raise AlreadySealedError(
            f"{date.isoformat()} is already sealed in {ledger_dir}; the ledger was not changed"
        )


def _index_line(date: dt.date, digest: str, forecast: Forecast) -> bytes:
    row = {"date": date.isoformat(), "digest": digest, "model_version": forecast.model.version}
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _append(index: Path, line: bytes) -> None:
    """Append one line, first ending the previous line if a hand edit left it open."""
    needs_break = False
    if index.is_file() and index.stat().st_size > 0:
        with index.open("rb") as handle:
            handle.seek(-1, 2)
            needs_break = handle.read(1) != b"\n"
    with index.open("ab") as handle:
        handle.write((b"\n" if needs_break else b"") + line)


def seal(forecast: Forecast, ledger_dir: Path, *, tsa_url: str = DEFAULT_TSA_URL) -> SealResult:
    """Write the canonical forecast, its timestamp and an index line; refuse a sealed date.

    Raises `AlreadySealedError` before any network call if the date has been sealed, and
    `TsaError` (writing nothing) if the timestamp authority fails or answers for other data.
    """
    date = forecast.date
    paths = ledger_paths(ledger_dir, date)
    _refuse_if_sealed(date, paths, ledger_dir)

    data = canonical_bytes(forecast)
    digest = digest_hex(data)
    reply = tsa.request_timestamp(data, tsa_url)
    line = _index_line(date, digest, forecast)

    written: list[Path] = []
    try:
        paths.forecast.parent.mkdir(parents=True, exist_ok=True)
        for path, payload in ((paths.forecast, data), (paths.timestamp, reply)):
            with path.open("xb") as handle:  # exclusive: never replaces an existing file
                written.append(path)
                handle.write(payload)
        _append(paths.index, line)
    except FileExistsError as error:
        _undo(written)
        raise AlreadySealedError(
            f"{date.isoformat()} is already sealed in {ledger_dir}; the ledger was not changed"
        ) from error
    except BaseException:
        _undo(written)
        raise
    return SealResult(date=date, digest=digest, paths=paths)


def _undo(written: list[Path]) -> None:
    for path in written:
        path.unlink(missing_ok=True)


# --- verify -------------------------------------------------------------------------------


def _fail(reason: str) -> VerifyResult:
    return VerifyResult(ok=False, reason=reason)


def _duration(span: dt.timedelta) -> str:
    seconds = span.total_seconds()
    if seconds < 120:
        return f"{round(seconds)} seconds"
    if seconds < 3 * 3600:
        return f"{round(seconds / 60)} minutes"
    return f"{seconds / 3600:.1f} hours"


def verify(date: dt.date, ledger_dir: Path) -> VerifyResult:
    """Check the sealed forecast of `date`: digest, form, date, and seal before sunset."""
    day = date.isoformat()
    paths = ledger_paths(ledger_dir, date)
    if not paths.forecast.is_file():
        return _fail(f"no sealed forecast for {day}")
    if not paths.timestamp.is_file():
        return _fail(f"no timestamp for the sealed forecast of {day}")
    data = paths.forecast.read_bytes()

    try:
        stamp = tsa.parse_response(paths.timestamp.read_bytes())
    except TsaError as error:
        return _fail(f"unusable timestamp for {day}: {error}")
    digest = digest_hex(data)
    if stamp.imprint.hex() != digest:
        return _fail(
            f"digest mismatch: the sealed forecast hashes to {digest} "
            f"but the timestamp covers {stamp.imprint.hex()}"
        )

    try:
        forecast = Forecast.model_validate_json(data)
    except ValidationError:
        return _fail(f"the sealed file for {day} is not a valid forecast")
    if forecast.date != date:
        return _fail(
            f"date mismatch: the file holds {forecast.date.isoformat()}, filed under {day}"
        )
    if canonical_bytes(forecast) != data:
        return _fail(f"the sealed file for {day} is not in canonical form")

    point = forecast.reference_point
    try:
        sunset = elevation_crossings(date, point.lat, point.lon)[0.0]
    except ValueError as error:
        return _fail(f"cannot find sunset on {day}: {error}")
    if stamp.gen_time >= sunset:
        return _fail(
            f"late seal: the timestamp is {_duration(stamp.gen_time - sunset)} "
            f"after sunset at the reference point on {day}"
        )
    return VerifyResult(ok=True)
