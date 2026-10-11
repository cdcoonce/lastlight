"""The real HTTP layer against a throwaway server on loopback.

Marked `network` because the suite's socket guard refuses every connect, loopback included.
Run on purpose with `uv run pytest -m network`; nothing here leaves the machine.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from camera_fakes import FakeClock, http_date, jpeg, make_camera

from lastlight.cameras.collector import USER_AGENT, CollectorConfig, collect
from lastlight.cameras.transport import FetchError, urllib_fetch
from lastlight.geometry import elevation_crossings, solar_position

pytestmark = pytest.mark.network

ETAG = '"frame-1"'
EVENING = dt.date(2026, 10, 10)


class _Server:
    def __init__(self, last_modified: dt.datetime) -> None:
        self.requests: list[dict[str, str]] = []
        self.last_modified = last_modified
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802  (the stdlib's required name)
                outer.requests.append(dict(self.headers.items()))
                if self.path == "/missing.jpg":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.headers.get("If-None-Match") == ETAG:
                    self.send_response(304)
                    self.send_header("ETag", ETAG)
                    self.end_headers()
                    return
                body = jpeg("served")
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Last-Modified", http_date(outer.last_modified))
                self.send_header("ETag", ETAG)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return None

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _Server:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=10)


@pytest.fixture
def server() -> Iterator[_Server]:
    with _Server(last_modified=dt.datetime(2026, 10, 10, tzinfo=dt.UTC)) as running:
        yield running


def test_urllib_fetch_returns_200_then_304_and_sends_the_headers_it_was_given(
    server: _Server,
) -> None:
    first = urllib_fetch(f"{server.base}/frame.jpg", {"User-Agent": USER_AGENT}, 10.0)
    second = urllib_fetch(
        f"{server.base}/frame.jpg", {"User-Agent": USER_AGENT, "If-None-Match": ETAG}, 10.0
    )

    assert first.status == 200
    assert first.body == jpeg("served")
    assert first.header("ETag") == ETAG
    assert second.status == 304
    assert second.body == b""
    assert [request["User-Agent"] for request in server.requests] == [USER_AGENT, USER_AGENT]
    assert server.requests[1]["If-None-Match"] == ETAG


def test_urllib_fetch_returns_a_404_as_a_response(server: _Server) -> None:
    assert urllib_fetch(f"{server.base}/missing.jpg", {}, 10.0).status == 404


def test_urllib_fetch_turns_a_refused_connection_into_a_fetch_error() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(FetchError):
        urllib_fetch(f"http://127.0.0.1:{port}/frame.jpg", {}, 10.0)


def test_collect_end_to_end_over_real_http_with_a_fake_clock(
    tmp_path: Path, server: _Server
) -> None:
    camera = dataclasses.replace(make_camera(), image_url=f"{server.base}/frame.jpg")
    instants = elevation_crossings(EVENING, camera.position.lat, camera.position.lon)
    start = instants[0.0].replace(microsecond=0)
    stop = solar_position(
        start + dt.timedelta(minutes=31), camera.position.lat, camera.position.lon
    ).elevation_deg
    server.last_modified = start - dt.timedelta(seconds=60)
    clock = FakeClock(start)

    (summary,) = collect(
        [camera],
        tmp_path,
        clock=clock,
        fetch=urllib_fetch,
        config=CollectorConfig(start_elev=6.0, stop_elev=stop),
        report=lambda message: None,
    )

    assert (summary.saved, summary.not_modified, summary.stale, summary.errors) == (1, 6, 0, 0)
    assert "If-None-Match" not in server.requests[0]
    assert {request["If-None-Match"] for request in server.requests[1:]} == {ETAG}
    assert {request["User-Agent"] for request in server.requests} == {USER_AGENT}
    day = tmp_path / "TEST1" / EVENING.isoformat()
    assert len(list(day.glob("*.jpg"))) == 1
    assert len(list(day.glob("*.json"))) == 1
