"""Test doubles for the collector: a fake clock and a scripted fetch.

Nothing here touches a socket or really sleeps. `FakeClock.sleep` advances the clock, so a
whole evening of polling runs in milliseconds and every request carries an exact time.
"""

from __future__ import annotations

import datetime as dt
import email.utils
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from lastlight.cameras.registry import Camera, Position
from lastlight.cameras.transport import FetchError, Response

# The public reference point (docs/design.md section 3). Test cameras are placed relative to it
# so no other coordinates are written down.
REF_LAT = 33.45
REF_LON = -112.07
REF_ELEV_M = 340.0

FRAME_NAME_FORMAT = "%Y%m%dT%H%M%SZ"


def make_camera(camera_id: str = "TEST1", *, lon_shift_deg: float = 0.0) -> Camera:
    return Camera(
        id=camera_id,
        operator="Test Operator",
        page_url=f"https://example.test/{camera_id}",
        image_url=f"https://example.test/{camera_id}.jpg",
        view_azimuth_deg=270.0,
        attribution="Image source: Test Operator",
        position=Position(lat=REF_LAT, lon=REF_LON + lon_shift_deg, elev_m=REF_ELEV_M),
    )


class FakeClock:
    """A clock that only moves when `sleep` is called (or a fake fetch spends time)."""

    def __init__(self, start: dt.datetime) -> None:
        assert start.tzinfo is not None
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> dt.datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, f"negative sleep {seconds}"
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += dt.timedelta(seconds=seconds)


def jpeg(label: str = "") -> bytes:
    """A few bytes that start and end like a JPEG; enough for the collector's magic check."""
    return b"\xff\xd8\xff\xe0" + label.encode() + b"\xff\xd9"


def http_date(when: dt.datetime) -> str:
    return email.utils.format_datetime(when.astimezone(dt.UTC), usegmt=True)


def frame_name(when: dt.datetime) -> str:
    return when.astimezone(dt.UTC).strftime(FRAME_NAME_FORMAT)


def ok(
    body: bytes,
    last_modified: dt.datetime,
    *,
    etag: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> Response:
    headers = {"Last-Modified": http_date(last_modified), "Content-Length": str(len(body))}
    if etag is not None:
        headers["ETag"] = etag
    headers.update(extra or {})
    return Response(status=200, headers=headers, body=body)


def not_modified(**headers: str) -> Response:
    return Response(status=304, headers=dict(headers), body=b"")


def status_only(status: int) -> Response:
    return Response(status=status, headers={}, body=b"")


@dataclass(frozen=True)
class Call:
    at: dt.datetime
    url: str
    headers: dict[str, str]
    timeout_s: float


Item = Response | FetchError | Callable[[Call], Response | FetchError]


def fresh(label: str, *, age_s: float = 60.0, etag: str | None = None) -> Item:
    """A 200 whose Last-Modified is `age_s` seconds before the moment of the request."""

    def respond(call: Call) -> Response:
        return ok(jpeg(label), call.at - dt.timedelta(seconds=age_s), etag=etag)

    return respond


class FakeFetch:
    """The injectable HTTP layer: replays a script, then a default, and records every call.

    A request beyond the script with no default fails the test, so a collector that polls more
    often than expected is caught. `latency_s` is the time each attempt takes on the clock.
    """

    def __init__(
        self,
        clock: FakeClock,
        script: Sequence[Item] = (),
        *,
        default: Callable[[Call], Response | FetchError] | None = None,
        latency_s: float = 0.0,
    ) -> None:
        self._clock = clock
        self._script = list(script)
        self._default = default
        self._latency_s = latency_s
        self.calls: list[Call] = []

    @property
    def exhausted(self) -> bool:
        return not self._script

    def __call__(self, url: str, headers: Mapping[str, str], timeout_s: float) -> Response:
        call = Call(at=self._clock.now(), url=url, headers=dict(headers), timeout_s=timeout_s)
        self.calls.append(call)
        if self._script:
            item = self._script.pop(0)
        elif self._default is not None:
            item = self._default
        else:
            raise AssertionError(f"unexpected request #{len(self.calls)} to {url}")
        outcome = item(call) if callable(item) else item
        self._clock.advance(self._latency_s)
        if isinstance(outcome, FetchError):
            raise outcome
        return outcome

    def calls_to(self, url: str) -> list[Call]:
        return [call for call in self.calls if call.url == url]
