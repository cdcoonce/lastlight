"""The ledger: seal a forecast, and verify a sealed date later.

Layout (`docs/design.md` section 4), under a ledger directory:

    YYYY/MM/DD.forecast.json   the canonical forecast bytes, exactly as hashed
    YYYY/MM/DD.forecast.tsr    the RFC 3161 response for the SHA-256 of those bytes
    YYYY/MM/DD.strip.png       the painting, when the forecast names one (`strip_sha256`)
    index.jsonl                one line per sealed forecast: date, digest, model_version

`seal` never overwrites: a date with any trace of a seal is refused before a timestamp is
requested, and a seal that fails part-way removes what it had written, so the ledger is
either unchanged or holds a complete seal. The strip is sealed through the forecast: the
forecast carries the SHA-256 of the PNG, so the one timestamp covers the numbers and the
picture, and `seal` refuses a strip that is not the one the forecast names, is not a PNG, or
carries metadata (the ledger is append-only and a sealed leak could not be taken back). `verify`
re-derives everything from the files: the digest of the sealed bytes must equal the timestamp's
imprint, the file must be a valid forecast for the requested date in canonical form, the
timestamp must have been issued before geometric sunset at the reference point on that date
(`lastlight.geometry`), and the strip beside the forecast must hash to `strip_sha256`.

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
from lastlight.privacy.images import PNG_SIGNATURE, image_issues
from lastlight.seal import tsa
from lastlight.seal.canonical import canonical_bytes, digest_hex
from lastlight.seal.tsa import DEFAULT_TSA_URL, TsaError

INDEX_NAME = "index.jsonl"


class AlreadySealedError(Exception):
    """The date already has a sealed forecast (or the remains of one); nothing was changed."""


class StripError(Exception):
    """The strip handed to `seal` is not one that may be sealed with this forecast."""


@dataclass(frozen=True)
class LedgerPaths:
    forecast: Path
    timestamp: Path
    index: Path
    strip: Path


@dataclass(frozen=True)
class SealResult:
    date: dt.date
    digest: str
    paths: LedgerPaths


@dataclass(frozen=True)
class VerifyResult:
    """`ok` with no reason, or not ok with the one reason the first failed check gives.

    `strip_verified` is true only when the forecast named a strip and the PNG in the ledger
    hashed to it; a forecast with no strip passes with it false.
    """

    ok: bool
    reason: str | None = None
    strip_verified: bool = False


def ledger_paths(ledger_dir: Path, date: dt.date) -> LedgerPaths:
    folder = ledger_dir / f"{date.year:04d}" / f"{date.month:02d}"
    stem = f"{date.day:02d}.forecast"
    return LedgerPaths(
        forecast=folder / f"{stem}.json",
        timestamp=folder / f"{stem}.tsr",
        index=ledger_dir / INDEX_NAME,
        strip=folder / f"{date.day:02d}.strip.png",
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


def is_sealed(date: dt.date, ledger_dir: Path) -> bool:
    """True if the ledger holds any trace of a seal for `date`: a file of it, or an index row."""
    paths = ledger_paths(ledger_dir, date)
    return (
        paths.forecast.exists()
        or paths.timestamp.exists()
        or paths.strip.exists()
        or _indexed(paths.index, date)
    )


def _refuse_if_sealed(date: dt.date, paths: LedgerPaths, ledger_dir: Path) -> None:
    if is_sealed(date, ledger_dir):
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


def _check_strip(forecast: Forecast, strip: bytes | None) -> None:
    """Refuse a strip that is not the one the forecast names, or that could leak."""
    declared = forecast.strip_sha256
    if strip is None:
        if declared is not None:
            raise StripError("the forecast names a strip (strip_sha256) but none was given")
        return
    if declared is None:
        raise StripError("a strip was given but the forecast names none (strip_sha256 is null)")
    if digest_hex(strip) != declared:
        raise StripError("the strip is not the one the forecast names: its SHA-256 differs")
    if not strip.startswith(PNG_SIGNATURE):
        # `image_issues` goes by the bytes, not the name: a clean JPEG or WebP passes it, and
        # would be sealed as `DD.strip.png` and served as a PNG.
        raise StripError("the strip is not a PNG image")
    issues = image_issues(strip, ".png")
    if issues:
        raise StripError(f"the strip carries image metadata the privacy rules forbid: {issues[0]}")


def seal(
    forecast: Forecast,
    ledger_dir: Path,
    *,
    strip: bytes | None = None,
    tsa_url: str = DEFAULT_TSA_URL,
) -> SealResult:
    """Write the canonical forecast, its timestamp, its strip and an index line.

    A date with any trace of a seal is refused. `strip` must be given exactly when the forecast
    names one (`strip_sha256`) and must be that PNG, with no metadata. Raises
    `AlreadySealedError` or `StripError` before any network call, and `TsaError` (writing
    nothing) if the timestamp authority fails or answers for other data.
    """
    date = forecast.date
    paths = ledger_paths(ledger_dir, date)
    _refuse_if_sealed(date, paths, ledger_dir)
    _check_strip(forecast, strip)

    data = canonical_bytes(forecast)
    digest = digest_hex(data)
    reply = tsa.request_timestamp(data, tsa_url)
    line = _index_line(date, digest, forecast)

    written: list[Path] = []
    try:
        paths.forecast.parent.mkdir(parents=True, exist_ok=True)
        files = [(paths.forecast, data), (paths.timestamp, reply)]
        if strip is not None:
            files.append((paths.strip, strip))
        for path, payload in files:
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


def _strip_failure(forecast: Forecast, paths: LedgerPaths, day: str) -> str | None:
    """Why the strip beside the forecast does not match what the forecast names, or None."""
    declared = forecast.strip_sha256
    if declared is None:
        if paths.strip.exists():
            return f"an unsealed strip sits beside the forecast of {day} (the forecast names none)"
        return None
    if not paths.strip.is_file():
        return f"no sealed strip for {day} (the forecast names one)"
    actual = digest_hex(paths.strip.read_bytes())
    if actual != declared:
        return (
            f"strip mismatch: the strip file hashes to {actual} but the forecast names {declared}"
        )
    return None


def verify(date: dt.date, ledger_dir: Path) -> VerifyResult:
    """Check the sealed forecast of `date`: digest, form, date, seal before sunset, and strip."""
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
    problem = _strip_failure(forecast, paths, day)
    if problem is not None:
        return _fail(problem)
    return VerifyResult(ok=True, strip_verified=forecast.strip_sha256 is not None)
