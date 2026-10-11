"""The polite collector: wait for the evening twilight window, then archive each camera's frames.

`collect` is the whole evening. For each camera it waits until the sun, at that camera's own
position, falls below the start elevation; polls every `interval_s` seconds while the sun is
above the stop elevation; and stops. The sun comes from `lastlight.geometry`, so the window
follows the seasons with no table of sunset times. Only a *descending* sun opens a window, so
a launch in the morning twilight waits for the evening instead of polling through the morning.

Politeness (`docs/design.md` section 3): at most one poll per camera per five minutes
(`MIN_INTERVAL_S`, enforced here and not only in the CLI), an identifying User-Agent,
conditional requests from the previous response's validators, a 60 s timeout, and up to three
attempts 20 s apart for transient failures. A 304 is a normal outcome, not a problem.

Everything with side effects is injected: `clock` (time and sleep), `fetch` (HTTP) and the
archive directory. Tests drive a whole evening with a fake clock and a scripted fetch.

Which frames are kept:

- 200 with a JPEG body and a Last-Modified within `stale_after` of the fetch: saved.
- 200 whose Last-Modified is older than that: the camera is frozen or serving an old frame; it
  is logged as `stale` and not saved, but its validators are kept so the next poll can be a 304.
- 200 with the Last-Modified of a frame already saved: logged `unchanged`, counted with the
  304s (the server ignored the conditional request).
- 200 without a usable Last-Modified, or with an empty or non-JPEG body: an error; nothing
  is saved, because the frame's time (its name and its sun elevation) cannot be trusted.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import lastlight
from lastlight.cameras.archive import EveningArchive, evening_date, frame_stem
from lastlight.cameras.clock import Clock
from lastlight.cameras.registry import Camera
from lastlight.cameras.transport import Fetch, FetchError, Response
from lastlight.geometry import solar_position

MIN_INTERVAL_S: Final = 300
USER_AGENT: Final = f"lastlight/{lastlight.__version__} (+https://github.com/cdcoonce/lastlight)"

_JPEG_MAGIC: Final = b"\xff\xd8"
_KEPT_HEADERS: Final = ("Last-Modified", "ETag", "Content-Length")
_RETRYABLE_STATUSES: Final = frozenset({408, 429})

# The sun's elevation never falls faster than 15 degrees an hour (the Earth's rotation), so a
# gap of `degrees * 240` seconds cannot be crossed sooner: sleeping that long never overshoots.
_SECONDS_PER_DEGREE_AT_FASTEST: Final = 240.0
_MIN_WAIT_S: Final = 5.0
_MAX_WAIT_S: Final = 600.0
_DESCENT_PROBE: Final = dt.timedelta(seconds=60)

_WAITING, _POLLING, _DONE = "waiting", "polling", "done"
_NEVER: Final = dt.datetime.min.replace(tzinfo=dt.UTC)  # `next_due` of a camera not yet polling


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Tunables of `collect`. The defaults are the contract of `lastlight collect`."""

    start_elev: float = 8.0
    stop_elev: float = -10.0
    interval_s: int = MIN_INTERVAL_S
    timeout_s: float = 60.0
    attempts: int = 3
    backoff_s: float = 20.0
    stale_after: dt.timedelta = dt.timedelta(minutes=20)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError for a setting that would be impolite or meaningless."""
        if self.interval_s < MIN_INTERVAL_S:
            raise ValueError(
                f"the polling interval must be at least {MIN_INTERVAL_S} seconds "
                f"(one poll per camera per 5 minutes, docs/design.md section 3); "
                f"got {self.interval_s}"
            )
        if not (math.isfinite(self.start_elev) and math.isfinite(self.stop_elev)):
            raise ValueError("start and stop elevations must be finite numbers of degrees")
        if self.start_elev <= self.stop_elev:
            raise ValueError(
                f"the start elevation ({self.start_elev}) must be above the stop elevation "
                f"({self.stop_elev}): the window runs from the one down to the other"
            )
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if not self.timeout_s > 0:
            raise ValueError(f"the timeout must be positive, got {self.timeout_s}")
        if not self.backoff_s >= 0:
            raise ValueError(f"the backoff must not be negative, got {self.backoff_s}")
        if self.stale_after <= dt.timedelta(0):
            raise ValueError("stale_after must be a positive duration")


@dataclass(frozen=True, slots=True)
class CameraSummary:
    """One camera's evening: how many polls ended each way."""

    camera_id: str
    evening: dt.date
    saved: int = 0
    not_modified: int = 0
    stale: int = 0
    errors: int = 0


def format_summary(summary: CameraSummary) -> str:
    """The one stdout line per camera per evening. It carries a date and counts, no clock time."""
    return (
        f"{summary.camera_id} {summary.evening.isoformat()}: saved={summary.saved} "
        f"not_modified={summary.not_modified} stale={summary.stale} errors={summary.errors}"
    )


@dataclass(slots=True)
class _Run:
    """The mutable state of one camera through the evening."""

    camera: Camera
    phase: str = _WAITING
    evening: dt.date | None = None
    archive: EveningArchive | None = None
    next_due: dt.datetime = _NEVER
    etag: str | None = None
    last_modified: str | None = None
    saved: int = 0
    not_modified: int = 0
    stale: int = 0
    errors: int = 0


class _Fetched(NamedTuple):
    """What the retry loop got: a 200 or 304 response, or why it gave up."""

    response: Response | None
    status: int | None
    attempts: int
    problem: str | None


def _elevation(camera: Camera, when: dt.datetime) -> float:
    return solar_position(when, camera.position.lat, camera.position.lon).elevation_deg


def _is_descending(camera: Camera, when: dt.datetime) -> bool:
    return _elevation(camera, when + _DESCENT_PROBE) < _elevation(camera, when)


def _iso(when: dt.datetime) -> str:
    return when.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_http_date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed.tzinfo is None:  # RFC 2822 "-0000": a UTC time of unknown local zone
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


# --- the window ---------------------------------------------------------------------------------


def _advance_phase(run: _Run, now: dt.datetime, config: CollectorConfig, root: Path) -> None:
    """Move a camera between waiting, polling and done according to the sun at `now`."""
    if run.phase == _DONE:
        return
    camera = run.camera
    elevation = _elevation(camera, now)
    descending = _is_descending(camera, now)
    if run.phase == _WAITING:
        if not descending or elevation >= config.start_elev:
            return  # morning twilight or daytime: the window opens in the evening
        if elevation < config.stop_elev:
            _finish(run, now)  # launched after the window had already closed
            return
        run.phase = _POLLING
        run.evening = evening_date(now, camera.position.lon)
        run.archive = EveningArchive(root, camera.id, run.evening)
        run.next_due = now
    elif elevation < config.stop_elev or not descending:
        _finish(run, now)


def _finish(run: _Run, now: dt.datetime) -> None:
    run.phase = _DONE
    if run.evening is None:
        run.evening = evening_date(now, run.camera.position.lon)


def _seconds_until_next_event(run: _Run, now: dt.datetime, config: CollectorConfig) -> float:
    if run.phase == _POLLING:
        return max((run.next_due - now).total_seconds(), 0.0)
    elevation = _elevation(run.camera, now)
    if _is_descending(run.camera, now) and elevation > config.start_elev:
        gap = (elevation - config.start_elev) * _SECONDS_PER_DEGREE_AT_FASTEST
        return min(max(gap, _MIN_WAIT_S), _MAX_WAIT_S)
    return _MAX_WAIT_S  # the sun is rising: look again later


# --- one poll -----------------------------------------------------------------------------------


def _is_retryable(status: int) -> bool:
    return status >= 500 or status in _RETRYABLE_STATUSES


def _fetch_with_retries(
    fetch: Fetch, url: str, headers: dict[str, str], clock: Clock, config: CollectorConfig
) -> _Fetched:
    problem = "no attempt was made"
    status: int | None = None
    for attempt in range(1, config.attempts + 1):
        try:
            response = fetch(url, headers, config.timeout_s)
        except FetchError as error:
            problem, status = str(error), None
        else:
            if response.status in (200, 304):
                return _Fetched(response, response.status, attempt, None)
            problem, status = f"HTTP {response.status}", response.status
            if not _is_retryable(response.status):
                return _Fetched(None, status, attempt, problem)
        if attempt < config.attempts:
            clock.sleep(config.backoff_s)
    return _Fetched(None, status, config.attempts, f"{problem} after {config.attempts} attempts")


class _Verdict(NamedTuple):
    outcome: str  # saved | not_modified | unchanged | stale | error
    frame: str | None = None
    last_modified: dt.datetime | None = None
    detail: str | None = None


def _judge_200(
    run: _Run, response: Response, fetched_at: dt.datetime, config: CollectorConfig
) -> _Verdict:
    """Decide what a 200 means and, for a good new frame, save it. Never raises on OSError."""
    camera, archive, body = run.camera, run.archive, response.body
    assert archive is not None
    header = response.header("Last-Modified")
    modified = _parse_http_date(header)
    if header is None:
        return _Verdict("error", detail="response has no Last-Modified header")
    if modified is None:
        return _Verdict("error", detail=f"unreadable Last-Modified header {header!r}")
    if not body:
        return _Verdict("error", last_modified=modified, detail="empty body")
    if not body.startswith(_JPEG_MAGIC):
        return _Verdict("error", last_modified=modified, detail="body is not a JPEG")

    # A new representation: its validators replace the old ones, stale or not.
    run.etag, run.last_modified = response.header("ETag"), header
    if fetched_at - modified > config.stale_after:
        return _Verdict("stale", last_modified=modified)
    stem = frame_stem(modified)
    if archive.has_frame(stem):
        return _Verdict("unchanged", frame=stem, last_modified=modified)
    sidecar: dict[str, Any] = {
        "camera": camera.id,
        "url": camera.image_url,
        "status": response.status,
        "headers": {
            name: value for name in _KEPT_HEADERS if (value := response.header(name)) is not None
        },
        "fetched_at": _iso(fetched_at),
        "sun_elevation_deg": round(_elevation(camera, modified), 3),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "collector": USER_AGENT,
    }
    try:
        archive.save_frame(stem, body, sidecar)
    except OSError as error:
        return _Verdict("error", last_modified=modified, detail=f"could not save {stem}: {error}")
    return _Verdict("saved", frame=stem, last_modified=modified)


def _poll(
    run: _Run,
    clock: Clock,
    fetch: Fetch,
    config: CollectorConfig,
    say: Callable[[str], None],
) -> None:
    camera, archive = run.camera, run.archive
    assert archive is not None
    started = clock.now()
    run.next_due = started + dt.timedelta(seconds=config.interval_s)
    headers = {"User-Agent": USER_AGENT}
    if run.etag:
        headers["If-None-Match"] = run.etag
    if run.last_modified:
        headers["If-Modified-Since"] = run.last_modified

    fetched = _fetch_with_retries(fetch, camera.image_url, headers, clock, config)
    fetched_at = clock.now()

    if fetched.response is None:
        verdict = _Verdict("error", detail=fetched.problem)
    elif fetched.response.status == 304:
        run.etag = fetched.response.header("ETag") or run.etag
        run.last_modified = fetched.response.header("Last-Modified") or run.last_modified
        verdict = _Verdict("not_modified")
    else:
        verdict = _judge_200(run, fetched.response, fetched_at, config)

    if verdict.outcome == "saved":
        run.saved += 1
    elif verdict.outcome in ("not_modified", "unchanged"):
        run.not_modified += 1
    elif verdict.outcome == "stale":
        run.stale += 1
    else:
        run.errors += 1
        say(f"{camera.id}: {verdict.detail}")

    entry = {
        "fetched_at": _iso(fetched_at),
        "outcome": verdict.outcome,
        "status": fetched.status,
        "attempts": fetched.attempts,
        "last_modified": None if verdict.last_modified is None else _iso(verdict.last_modified),
        "frame": verdict.frame,
        "detail": verdict.detail,
    }
    try:
        archive.append_log(entry)
    except OSError as error:
        say(f"{camera.id}: could not write the poll log: {error}")


# --- the evening --------------------------------------------------------------------------------


def _say_on_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def collect(
    cameras: Sequence[Camera],
    archive: Path,
    *,
    clock: Clock,
    fetch: Fetch,
    config: CollectorConfig | None = None,
    report: Callable[[str], None] | None = None,
) -> list[CameraSummary]:
    """Run one evening for `cameras` and return a summary per camera, in the given order.

    Returns when every camera's window has closed: immediately for a camera whose window had
    already closed at launch, after the stop elevation otherwise. Problems (failed polls,
    unwritable files) are passed to `report`, by default printed to stderr; they never raise.
    """
    settings = CollectorConfig() if config is None else config
    settings.validate()  # a CollectorConfig edited after construction is still checked
    ids = [camera.id for camera in cameras]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate camera ids: {ids}")
    say = _say_on_stderr if report is None else report
    runs = [_Run(camera) for camera in cameras]

    while True:
        now = clock.now()
        for run in runs:
            _advance_phase(run, now, settings, archive)
        active = [run for run in runs if run.phase != _DONE]
        if not active:
            break
        due = [run for run in active if run.phase == _POLLING and run.next_due <= now]
        if due:
            _poll(min(due, key=lambda run: run.next_due), clock, fetch, settings, say)
            continue
        clock.sleep(min(_seconds_until_next_event(run, now, settings) for run in active))

    summaries = []
    for run in runs:
        assert run.evening is not None  # every finished run has one
        summaries.append(
            CameraSummary(
                camera_id=run.camera.id,
                evening=run.evening,
                saved=run.saved,
                not_modified=run.not_modified,
                stale=run.stale,
                errors=run.errors,
            )
        )
    return summaries
