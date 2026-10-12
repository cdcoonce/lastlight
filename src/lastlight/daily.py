"""The scheduled seal: paint the coming evening, seal it, check the seal (issue #8).

`seal_daily` is what the seal workflow runs, and it is where the two promises that cannot be
left to YAML live:

* **Idempotent.** If the evening already has any trace of a seal, nothing is painted, no
  timestamp is requested and nothing is written. The existing seal must still verify; if it does
  not, that is reported as an error rather than waved through as "already done".
* **Never late.** If the sun has already reached the horizon at the reference point on that
  evening, the day is refused before anything is fetched. A day that missed its seal is recorded
  as missing and never back-filled (`docs/design.md` section 2, rule 1). After sealing, the new
  seal is verified too, so a timestamp authority that stamped the request after sunset makes the
  run fail instead of leaving a late seal looking like a good one.

The evening defaults to the local mean solar date at the reference point (the same convention
as `lastlight.cameras.archive.evening_date`), so a run in the afternoon UTC seals the evening
that is coming.

`seal_with_retry` is the one retry the workflow asks for (issue #26). A failure of the outside
world (the HRRR bucket, the timestamp authority, the disk) must not cost the evening, so a failed
attempt is tried again after a pause, but only while the evening is still ahead: if the pause
would reach the deadline (the sun at the horizon, this module's own instant) there is no retry,
and the second attempt goes through `seal_daily` with a fresh clock reading, so the late refusal
above applies to it as it does to any run. A failure that would only repeat (a refusal, a reduction
that cannot work, a seal that does not verify) is never retried: a pause would only cost billed
minutes. The loop takes its clock and its sleep as arguments, so no test ever waits.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lastlight.cameras.archive import evening_date
from lastlight.geometry import elevation_crossings
from lastlight.inputs.hrrr import HrrrError
from lastlight.inputs.http import FetchError, Transport
from lastlight.painting import paint
from lastlight.seal import DEFAULT_TSA_URL, TsaError, is_sealed, ledger_paths, seal, verify
from lastlight.site import REFERENCE_POINT

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (HrrrError, FetchError, TsaError, OSError)
"""The failures a retry can cure: the weather service, the timestamp authority, the disk.

Everything else (`TooLateError`, `ReductionError`, `StripError`, `AlreadySealedError`,
`SealVerificationError`, `ValueError`) is deterministic or a refusal and is never retried.
"""


class TooLateError(Exception):
    """The sun is already down at the reference point for the evening; it is not sealed late."""


class SealVerificationError(Exception):
    """A seal that was just written, or one already there, does not verify."""


@dataclass(frozen=True)
class DailyResult:
    """`sealed` by this run, or `already-sealed` (and still verifying); `digest` is the SHA-256
    of the sealed forecast file."""

    status: Literal["sealed", "already-sealed"]
    date: dt.date
    digest: str


def _utc(when: dt.datetime) -> dt.datetime:
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("the time must be timezone-aware; a naive datetime is ambiguous")
    return when.astimezone(dt.UTC)


def default_evening(now_utc: dt.datetime) -> dt.date:
    """The evening a run at `now_utc` seals: the local mean solar date at the reference point."""
    return evening_date(_utc(now_utc), REFERENCE_POINT[1])


def evening_deadline(evening: dt.date) -> dt.datetime:
    """The instant `evening` closes: the sun at the horizon at the reference point (UTC).

    A seal made at or after it is late. INTERNAL USE ONLY, like `elevation_crossings`: the
    instant is for deciding, never for printing.
    """
    lat, lon, _elevation = REFERENCE_POINT
    return elevation_crossings(evening, lat, lon)[0.0]


def seal_daily(
    evening: dt.date,
    ledger_dir: Path,
    *,
    transport: Transport,
    now_utc: dt.datetime,
    cache_dir: Path | str,
    cycle_hour: int | None = None,
    forecast_hour: int | None = None,
    tsa_url: str = DEFAULT_TSA_URL,
    code_sha: str | None = None,
) -> DailyResult:
    """Seal the evening of `evening` into `ledger_dir` unless it is sealed or too late.

    Raises `TooLateError` (nothing fetched or written), `SealVerificationError` (an existing
    or a new seal does not verify), and whatever the weather ingest, the renderer or the
    timestamp authority raise; in every failure before the files are written the ledger is
    unchanged.
    """
    now = _utc(now_utc)
    if is_sealed(evening, ledger_dir):
        existing = verify(evening, ledger_dir)
        if not existing.ok:
            raise SealVerificationError(
                f"{evening.isoformat()} already has a seal in {ledger_dir} that does not "
                f"verify: {existing.reason}"
            )
        data = ledger_paths(ledger_dir, evening).forecast.read_bytes()
        return DailyResult("already-sealed", evening, hashlib.sha256(data).hexdigest())

    if now >= evening_deadline(evening):
        raise TooLateError(
            f"the sun has already reached the horizon at the reference point for the evening of "
            f"{evening.isoformat()}; sealing now would be late, and a late day is never "
            "back-filled (it is recorded as missing)"
        )

    painting = paint(
        evening,
        transport=transport,
        now_utc=now,
        cache_dir=cache_dir,
        cycle_hour=cycle_hour,
        forecast_hour=forecast_hour,
        code_sha=code_sha,
    )
    assert painting.strip_png is not None  # paint() makes a strip unless told not to
    result = seal(painting.forecast, ledger_dir, strip=painting.strip_png, tsa_url=tsa_url)
    outcome = verify(evening, ledger_dir)
    if not outcome.ok:
        raise SealVerificationError(
            f"the seal just written for {evening.isoformat()} does not verify: {outcome.reason}"
        )
    return DailyResult("sealed", evening, result.digest)


def _failure(number: int, attempts: int, error: Exception) -> str:
    reason = str(error) or type(error).__name__
    return reason if attempts == 1 else f"attempt {number} of {attempts} failed: {reason}"


def seal_with_retry(
    attempt: Callable[[dt.datetime], DailyResult],
    evening: dt.date,
    *,
    now_utc: dt.datetime,
    attempts: int,
    pause_seconds: float,
    clock: Callable[[], dt.datetime],
    sleep: Callable[[float], None],
    note: Callable[[str], None],
) -> DailyResult:
    """Run `attempt(now)` up to `attempts` times, pausing between tries, never past the deadline.

    The first attempt gets `now_utc`; each retry gets a fresh `clock()` reading taken after the
    pause. Only `TRANSIENT_ERRORS` are retried, and only if `clock()` plus the pause is still
    before `evening_deadline(evening)`; otherwise the failure is raised at once. Whatever the
    last attempt raised is what this raises, so the caller's exit code is the last attempt's.
    Every failed attempt, and every decision not to retry, is told to `note` (dates and
    durations only, never a clock time).
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    now = _utc(now_utc)
    for number in range(1, attempts + 1):
        try:
            return attempt(now)
        except Exception as error:
            note(_failure(number, attempts, error))
            if number == attempts:
                raise
            if not isinstance(error, TRANSIENT_ERRORS):
                note("not retrying: that failure would only repeat, so a pause cannot help")
                raise
            if _utc(clock()) + dt.timedelta(seconds=pause_seconds) >= evening_deadline(evening):
                note(
                    f"not retrying: a pause of {pause_seconds:g} seconds would reach the sun at "
                    "the horizon at the reference point, and a seal made then would be late"
                )
                raise
            note(f"pausing {pause_seconds:g} seconds before attempt {number + 1} of {attempts}")
            sleep(pause_seconds)
            now = _utc(clock())
    raise AssertionError("unreachable: the last attempt returns or raises")  # pragma: no cover
