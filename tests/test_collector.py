"""The polling loop: archive contents, conditional requests, retries and the sun-driven window.

Every test drives `collect` with a fake clock and a scripted fetch, so an evening of polling
takes milliseconds and no socket is opened. Expected times come from `lastlight.geometry`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
from camera_fakes import (
    FakeClock,
    FakeFetch,
    frame_name,
    fresh,
    http_date,
    jpeg,
    make_camera,
    not_modified,
    ok,
    status_only,
)

import lastlight
from lastlight.cameras.collector import (
    MIN_INTERVAL_S,
    USER_AGENT,
    CameraSummary,
    CollectorConfig,
    collect,
    format_summary,
)
from lastlight.cameras.registry import Camera
from lastlight.cameras.transport import FetchError, Response
from lastlight.geometry import elevation_crossings, solar_position

EVENING = dt.date(2026, 10, 10)
DAY_DIR = EVENING.isoformat()
# Crossing instants are found by bisection to within a second and cut to whole seconds.
TOLERANCE = dt.timedelta(seconds=2)


def seconds(count: float) -> dt.timedelta:
    return dt.timedelta(seconds=count)


def crossing(camera: Camera, elevation_deg: float) -> dt.datetime:
    """When the setting sun reaches an elevation of the sweep, at the camera, on EVENING."""
    instants = elevation_crossings(EVENING, camera.position.lat, camera.position.lon)
    return instants[elevation_deg].replace(microsecond=0)


def sun_elevation(camera: Camera, when: dt.datetime) -> float:
    return solar_position(when, camera.position.lat, camera.position.lon).elevation_deg


def short_window(camera: Camera) -> tuple[dt.datetime, CollectorConfig]:
    """A launch with the sun at the horizon and a stop elevation 31 minutes later.

    With the default 300 s interval that is exactly seven polls, at +0, +300, ... +1800 s.
    """
    start = crossing(camera, 0.0)
    stop = sun_elevation(camera, start + dt.timedelta(minutes=31))
    return start, CollectorConfig(start_elev=6.0, stop_elev=stop)


def run(
    camera_list: list[Camera],
    archive: Path,
    clock: FakeClock,
    fetch: FakeFetch,
    config: CollectorConfig,
) -> tuple[list[CameraSummary], list[str]]:
    messages: list[str] = []
    summaries = collect(
        camera_list, archive, clock=clock, fetch=fetch, config=config, report=messages.append
    )
    return summaries, messages


def read_log(directory: Path) -> list[dict]:
    lines = (directory / "log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


# --- the archive after a scripted evening -----------------------------------------------------


def test_a_scripted_evening_leaves_exactly_the_expected_files_and_records(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(
        clock,
        [
            fresh("a"),  # poll 1: a new frame, saved
            not_modified(),  # poll 2: 304
            fresh("old", age_s=25 * 60),  # poll 3: Last-Modified 25 minutes old, stale
            status_only(503),  # poll 4: three failed attempts, one error
            FetchError("connection reset"),
            status_only(503),
            fresh("b"),  # poll 5: saved
            not_modified(),  # poll 6: 304
            status_only(404),  # poll 7: a permanent error, one attempt
        ],
    )

    (summary,), messages = run([camera], tmp_path, clock, fetch, config)

    assert fetch.exhausted
    assert summary == CameraSummary(
        camera_id="TEST1", evening=EVENING, saved=2, not_modified=2, stale=1, errors=2
    )
    day = tmp_path / "TEST1" / DAY_DIR
    first = frame_name(start - seconds(60))
    second = frame_name(start + seconds(1200 - 60))
    assert sorted(path.name for path in day.iterdir()) == sorted(
        [f"{first}.jpg", f"{first}.json", f"{second}.jpg", f"{second}.json", "log.jsonl"]
    )
    assert (day / f"{first}.jpg").read_bytes() == jpeg("a")
    assert (day / f"{second}.jpg").read_bytes() == jpeg("b")
    assert [entry["outcome"] for entry in read_log(day)] == [
        "saved",
        "not_modified",
        "stale",
        "error",
        "saved",
        "not_modified",
        "error",
    ]
    assert [entry["attempts"] for entry in read_log(day)] == [1, 1, 1, 3, 1, 1, 1]
    assert any("TEST1" in message and "404" in message for message in messages)


def test_the_stale_frame_is_logged_with_its_age_and_not_saved(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, [fresh("old", age_s=25 * 60)], default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    day = tmp_path / "TEST1" / DAY_DIR
    stale_entry = read_log(day)[0]
    assert stale_entry["outcome"] == "stale"
    assert stale_entry["status"] == 200
    assert stale_entry["frame"] is None
    assert stale_entry["last_modified"] == (start - seconds(25 * 60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert not list(day.glob("*.jpg"))


@pytest.mark.parametrize(
    ("age_s", "outcome"), [(20 * 60, "saved"), (20 * 60 + 1, "stale"), (0, "saved"), (-30, "saved")]
)
def test_a_frame_is_stale_only_when_more_than_twenty_minutes_old(
    tmp_path: Path, age_s: int, outcome: str
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, [fresh("x", age_s=age_s)], default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    assert read_log(tmp_path / "TEST1" / DAY_DIR)[0]["outcome"] == outcome


def test_a_repeated_200_with_the_same_last_modified_is_not_saved_twice(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    same_moment = start - seconds(60)
    fetch = FakeFetch(
        clock,
        [lambda call: ok(jpeg("first"), same_moment), lambda call: ok(jpeg("second"), same_moment)],
        default=lambda call: not_modified(),
    )

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    day = tmp_path / "TEST1" / DAY_DIR
    assert (summary.saved, summary.not_modified) == (1, 6)
    assert (day / f"{frame_name(same_moment)}.jpg").read_bytes() == jpeg("first")
    assert [entry["outcome"] for entry in read_log(day)][:2] == ["saved", "unchanged"]
    assert len(list(day.glob("*.jpg"))) == 1


@pytest.mark.parametrize(
    ("reason", "bad_response"),
    [
        ("Last-Modified", lambda call: Response(200, {}, jpeg("x"))),
        (
            "Last-Modified",
            lambda call: Response(200, {"Last-Modified": "not a date"}, jpeg("x")),
        ),
        ("JPEG", lambda call: ok(b"<html>busy</html>", call.at)),
        ("empty", lambda call: ok(b"", call.at)),
    ],
)
def test_a_200_that_cannot_be_trusted_is_an_error_and_saves_nothing(
    tmp_path: Path, reason: str, bad_response
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, [bad_response], default=lambda call: not_modified())

    (summary,), messages = run([camera], tmp_path, clock, fetch, config)

    day = tmp_path / "TEST1" / DAY_DIR
    assert summary.errors == 1
    assert summary.saved == 0
    assert not list(day.glob("*.jpg"))
    assert not list(day.glob("*.json"))
    assert reason in read_log(day)[0]["detail"]
    assert any(reason in message for message in messages)


def test_a_failing_archive_is_counted_as_errors_and_does_not_stop_the_evening(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    (tmp_path / "TEST1").write_text("a file where the camera directory should be")
    fetch = FakeFetch(clock, default=fresh("x"))

    (summary,), messages = run([camera], tmp_path, clock, fetch, config)

    assert len(fetch.calls) == 7
    assert (summary.saved, summary.errors) == (0, 7)
    assert messages


def test_the_sidecar_records_the_request_and_the_sun_at_the_frames_own_time(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    assert start.date() != EVENING  # the evening straddles UTC midnight; the folder is local
    clock = FakeClock(start)
    ten_minutes = seconds(600)
    fetch = FakeFetch(
        clock,
        [
            lambda call: ok(
                jpeg("a"),
                call.at - ten_minutes,
                etag='"e1"',
                extra={"Server": "x", "Content-Type": "image/jpeg"},
            )
        ],
        default=lambda call: not_modified(),
    )

    run([camera], tmp_path, clock, fetch, config)

    frame_time = start - ten_minutes
    sidecar = json.loads(
        (tmp_path / "TEST1" / DAY_DIR / f"{frame_name(frame_time)}.json").read_text()
    )
    assert sidecar["camera"] == "TEST1"
    assert sidecar["url"] == camera.image_url
    assert sidecar["status"] == 200
    assert sidecar["headers"] == {
        "Last-Modified": http_date(frame_time),
        "ETag": '"e1"',
        "Content-Length": str(len(jpeg("a"))),
    }
    assert dt.datetime.fromisoformat(sidecar["fetched_at"]) == start
    assert sidecar["sha256"] == hashlib.sha256(jpeg("a")).hexdigest()
    # The elevation is the sun's at the frame's Last-Modified time, not at the fetch time.
    assert sidecar["sun_elevation_deg"] == pytest.approx(
        sun_elevation(camera, frame_time), abs=1e-3
    )
    assert abs(sidecar["sun_elevation_deg"] - sun_elevation(camera, start)) > 1.0


def test_sidecar_headers_omit_what_the_server_did_not_send(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    body = jpeg("a")
    fetch = FakeFetch(
        clock,
        [lambda call: Response(200, {"Last-Modified": http_date(call.at - seconds(30))}, body)],
        default=lambda call: not_modified(),
    )

    run([camera], tmp_path, clock, fetch, config)

    (sidecar_path,) = (tmp_path / "TEST1" / DAY_DIR).glob("*.json")
    assert list(json.loads(sidecar_path.read_text())["headers"]) == ["Last-Modified"]


# --- the requests ---------------------------------------------------------------------------


def test_every_request_identifies_lastlight_and_uses_a_sixty_second_timeout(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    expected = f"lastlight/{lastlight.__version__} (+https://github.com/cdcoonce/lastlight)"
    assert expected == USER_AGENT
    assert len(fetch.calls) == 7
    assert {call.headers["User-Agent"] for call in fetch.calls} == {expected}
    assert {call.timeout_s for call in fetch.calls} == {60.0}
    assert {call.url for call in fetch.calls} == {camera.image_url}


def test_conditional_headers_follow_the_previous_response(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(
        clock,
        [fresh("a", etag='"v1"'), not_modified(), fresh("b", etag='"v2"')],
        default=lambda call: not_modified(),
    )

    run([camera], tmp_path, clock, fetch, config)

    first, second, third, fourth = fetch.calls[:4]
    assert "If-None-Match" not in first.headers
    assert "If-Modified-Since" not in first.headers
    assert second.headers["If-None-Match"] == '"v1"'
    assert second.headers["If-Modified-Since"] == http_date(start - seconds(60))
    # A 304 without validators leaves the previous ones in force.
    assert third.headers["If-None-Match"] == '"v1"'
    assert third.headers["If-Modified-Since"] == http_date(start - seconds(60))
    # A new 200 replaces them.
    assert fourth.headers["If-None-Match"] == '"v2"'
    assert fourth.headers["If-Modified-Since"] == http_date(start + seconds(600 - 60))


def test_a_stale_200_still_updates_the_validators_so_the_next_poll_can_be_a_304(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    stale_moment = start - seconds(25 * 60)
    fetch = FakeFetch(
        clock,
        [lambda call: ok(jpeg("old"), stale_moment, etag='"s1"')],
        default=lambda call: not_modified(),
    )

    run([camera], tmp_path, clock, fetch, config)

    assert fetch.calls[1].headers["If-None-Match"] == '"s1"'
    assert fetch.calls[1].headers["If-Modified-Since"] == http_date(stale_moment)


def test_a_304_that_carries_a_new_etag_replaces_it(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(
        clock,
        [fresh("a", etag='"v1"'), not_modified(ETag='"v1b"')],
        default=lambda call: not_modified(),
    )

    run([camera], tmp_path, clock, fetch, config)

    assert fetch.calls[2].headers["If-None-Match"] == '"v1b"'


def test_cameras_do_not_share_validators(tmp_path: Path) -> None:
    first, second = make_camera("AAA1"), make_camera("BBB1")
    start, config = short_window(first)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, default=lambda call: ok(jpeg("x"), call.at, etag=f'"{call.url}"'))

    run([first, second], tmp_path, clock, fetch, config)

    assert "If-None-Match" not in fetch.calls_to(second.image_url)[0].headers
    for camera in (first, second):
        later = fetch.calls_to(camera.image_url)[1]
        assert later.headers["If-None-Match"] == f'"{camera.image_url}"'


# --- retries and spacing --------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_a_transient_failure_is_retried_after_twenty_seconds(tmp_path: Path, status: int) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(
        clock,
        [status_only(status), FetchError("timed out"), fresh("a")],
        default=lambda call: not_modified(),
    )

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    assert [call.at for call in fetch.calls[:4]] == [
        start,
        start + seconds(20),
        start + seconds(40),
        start + seconds(300),
    ]
    assert (summary.saved, summary.errors) == (1, 0)
    assert read_log(tmp_path / "TEST1" / DAY_DIR)[0]["attempts"] == 3


def test_three_failed_attempts_are_one_error_and_there_is_no_fourth(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, [status_only(503)] * 3, default=lambda call: not_modified())

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    assert summary.errors == 1
    assert [call.at for call in fetch.calls[:4]] == [
        start,
        start + seconds(20),
        start + seconds(40),
        start + seconds(300),
    ]
    assert clock.sleeps[:3] == [20.0, 20.0, 260.0]  # no backoff after the last attempt


@pytest.mark.parametrize("status", [400, 403, 404, 410])
def test_a_client_error_is_not_retried(tmp_path: Path, status: int) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, [status_only(status)], default=lambda call: not_modified())

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    assert summary.errors == 1
    assert fetch.calls[1].at == start + seconds(300)


def test_slow_attempts_never_bring_the_next_poll_closer_than_the_interval(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(
        clock, [status_only(503)] * 3, default=lambda call: not_modified(), latency_s=120.0
    )

    run([camera], tmp_path, clock, fetch, config)

    # Attempts take 120 s each plus two 20 s backoffs: the poll ends at +400 s, after the
    # +300 s mark, so the next poll starts at once and is still 300 s or more after the last.
    assert [call.at for call in fetch.calls[:4]] == [
        start,
        start + seconds(140),
        start + seconds(280),
        start + seconds(400),
    ]


def test_an_interval_longer_than_the_minimum_is_honoured(tmp_path: Path) -> None:
    camera = make_camera()
    start, _ = short_window(camera)
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0, interval_s=600)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    gaps = {b.at - a.at for a, b in zip(fetch.calls, fetch.calls[1:], strict=False)}
    assert gaps == {seconds(600)}


# --- the sun-driven window ------------------------------------------------------------------


def test_polling_starts_at_the_start_elevation_and_stops_at_the_stop_elevation(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0)
    opens, closes = crossing(camera, 6.0), crossing(camera, -8.0)
    clock = FakeClock(opens - dt.timedelta(hours=3))
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    times = [call.at for call in fetch.calls]
    assert opens - TOLERANCE <= times[0] <= opens + seconds(10)  # not before the sun reaches +6
    assert (
        times[-1] <= closes + TOLERANCE < times[-1] + seconds(300)
    )  # the last poll was in the window
    assert {b - a for a, b in zip(times, times[1:], strict=False)} == {seconds(300)}
    assert len(times) == int((closes - times[0]).total_seconds() // 300) + 1
    for when in times:
        assert -8.0 - 0.01 <= sun_elevation(camera, when) <= 6.0 + 0.01
    assert max(clock.sleeps) <= 600.0  # the wait is chunked, never one blind sleep
    assert clock.now() <= closes + seconds(300 + 10)  # it exits within one interval of the stop
    assert summary.evening == EVENING
    assert summary.not_modified == len(times)


def test_a_launch_inside_the_window_polls_at_once(tmp_path: Path) -> None:
    camera = make_camera()
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0)
    launch = crossing(camera, 2.0)
    clock = FakeClock(launch)
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    assert fetch.calls[0].at == launch


def test_a_launch_after_the_window_polls_nothing_and_still_reports_the_evening(
    tmp_path: Path,
) -> None:
    camera = make_camera()
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0)
    clock = FakeClock(crossing(camera, -8.0) + dt.timedelta(minutes=10))
    fetch = FakeFetch(clock)  # no script and no default: any request fails the test

    (summary,), _ = run([camera], tmp_path, clock, fetch, config)

    assert fetch.calls == []
    assert clock.sleeps == []
    assert summary == CameraSummary("TEST1", EVENING, 0, 0, 0, 0)


def test_a_launch_in_the_morning_twilight_waits_for_the_evening(tmp_path: Path) -> None:
    camera = make_camera()
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0)
    opens = crossing(camera, 6.0)
    launch = opens - dt.timedelta(hours=10, minutes=30)
    # Precondition: the sun is inside the elevation window but rising, so only "descending"
    # tells the morning from the evening.
    assert -8.0 < sun_elevation(camera, launch) < 6.0
    assert sun_elevation(camera, launch + seconds(60)) > sun_elevation(camera, launch)
    clock = FakeClock(launch)
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    assert opens - TOLERANCE <= fetch.calls[0].at <= opens + seconds(10)


def test_each_camera_follows_the_sun_at_its_own_position(tmp_path: Path) -> None:
    east, west = make_camera("EAST1"), make_camera("WEST1", lon_shift_deg=-2.0)
    config = CollectorConfig(start_elev=6.0, stop_elev=-8.0)
    opens = {camera.id: crossing(camera, 6.0) for camera in (east, west)}
    closes = {camera.id: crossing(camera, -8.0) for camera in (east, west)}
    assert opens["WEST1"] - opens["EAST1"] > dt.timedelta(minutes=5)  # precondition
    clock = FakeClock(opens["EAST1"] - dt.timedelta(hours=1))
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    summaries, _ = run([east, west], tmp_path, clock, fetch, config)

    assert [summary.camera_id for summary in summaries] == ["EAST1", "WEST1"]
    for camera in (east, west):
        times = [call.at for call in fetch.calls_to(camera.image_url)]
        assert opens[camera.id] - TOLERANCE <= times[0] <= opens[camera.id] + seconds(10)
        assert times[-1] <= closes[camera.id] + TOLERANCE < times[-1] + seconds(300)
        assert {b - a for a, b in zip(times, times[1:], strict=False)} == {seconds(300)}


def test_the_start_and_stop_elevations_are_the_ones_given(tmp_path: Path) -> None:
    camera = make_camera()
    config = CollectorConfig(start_elev=2.0, stop_elev=-4.0)
    opens, closes = crossing(camera, 2.0), crossing(camera, -4.0)
    clock = FakeClock(opens - dt.timedelta(hours=1))
    fetch = FakeFetch(clock, default=lambda call: not_modified())

    run([camera], tmp_path, clock, fetch, config)

    times = [call.at for call in fetch.calls]
    assert opens - TOLERANCE <= times[0] <= opens + seconds(10)
    assert times[-1] <= closes + TOLERANCE < times[-1] + seconds(300)


def test_the_evening_folder_is_the_local_solar_date_not_the_utc_date(tmp_path: Path) -> None:
    camera = make_camera()
    start, config = short_window(camera)
    clock = FakeClock(start)
    fetch = FakeFetch(clock, default=fresh("x"))

    run([camera], tmp_path, clock, fetch, config)

    assert [path.name for path in (tmp_path / "TEST1").iterdir()] == [DAY_DIR]
    assert start.date() == EVENING + dt.timedelta(days=1)  # the UTC date has already rolled over


# --- configuration and the summary ------------------------------------------------------------


def test_the_defaults_are_the_ones_the_issue_specifies() -> None:
    config = CollectorConfig()
    assert (config.start_elev, config.stop_elev) == (8.0, -10.0)
    assert config.interval_s == MIN_INTERVAL_S == 300
    assert config.timeout_s == 60.0
    assert config.attempts == 3
    assert config.backoff_s == 20.0
    assert config.stale_after == dt.timedelta(minutes=20)


@pytest.mark.parametrize("interval", [0, 1, 60, 299])
def test_an_interval_below_five_minutes_is_refused_with_a_clear_message(interval: int) -> None:
    with pytest.raises(ValueError, match=rf"at least {MIN_INTERVAL_S} seconds.*{interval}"):
        CollectorConfig(interval_s=interval)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_elev": -10.0, "stop_elev": -10.0},
        {"start_elev": -12.0, "stop_elev": 0.0},
        {"start_elev": float("nan")},
        {"stop_elev": float("-inf")},
        {"attempts": 0},
        {"timeout_s": 0.0},
        {"backoff_s": -1.0},
    ],
)
def test_nonsense_settings_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CollectorConfig(**kwargs)


def test_collect_itself_refuses_a_too_short_interval_even_if_the_config_was_forged(
    tmp_path: Path,
) -> None:
    config = CollectorConfig()
    object.__setattr__(config, "interval_s", 60)  # bypass the frozen dataclass on purpose
    camera = make_camera()
    clock = FakeClock(crossing(camera, 0.0))
    fetch = FakeFetch(clock)
    with pytest.raises(ValueError, match="at least"):
        collect([camera], tmp_path, clock=clock, fetch=fetch, config=config, report=lambda m: None)
    assert fetch.calls == []


def test_the_summary_line_has_the_camera_the_evening_and_the_four_counts() -> None:
    summary = CameraSummary("WHTM1", EVENING, saved=9, not_modified=14, stale=1, errors=2)
    assert format_summary(summary) == "WHTM1 2026-10-10: saved=9 not_modified=14 stale=1 errors=2"
