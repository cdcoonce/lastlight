"""Shared support for the daily-painting tests (issue #8): a fixed evening, a fake clock and TSA.

The recorded HRRR crop is the run of 2026-10-10 hour 22, forecast hour 0, so every scenario
reads the evening of 2026-10-10 with `cycle_hour=22, forecast_hour=0` through the injectable
transport (`hrrr_support.crop_transport`). Nothing here spells a clock time: "now" is always
computed from the geometric sunset at the reference point, offset by a duration.
"""

from __future__ import annotations

import datetime as dt

import hrrr_support
import pytest
import seal_helpers

from lastlight.seal import tsa

EVENING = dt.date(2026, 10, 10)
CYCLE_HOUR = 22
FORECAST_HOUR = 0
CODE_SHA = "0123456789abcdef0123456789abcdef01234567"
TSA_URL = "https://tsa.example/tsr"


def sunset() -> dt.datetime:
    """Geometric sunset at the reference point on the evening under test (a UTC instant)."""
    return seal_helpers.sunset(EVENING)


def before_sunset(minutes: float = 20) -> dt.datetime:
    """A "now" that is `minutes` before sunset: after the recorded run, before the sun is down."""
    return sunset() - dt.timedelta(minutes=minutes)


def after_sunset(minutes: float = 5) -> dt.datetime:
    return sunset() + dt.timedelta(minutes=minutes)


class EchoTsa:
    """Stands in for `lastlight.seal.tsa.post_request`: stamps whatever digest it is sent.

    The request is DER with a fixed prefix, then the 32-byte SHA-256 imprint (no nonce), so the
    digest can be read straight out of it. The reply is a response for that digest issued at
    `gen_time` and signed by the test authority (`tsa_factory`), which is what a real authority
    would return; a test that seals through it requests the `trust_test_tsa` fixture.
    """

    def __init__(self, gen_time: dt.datetime) -> None:
        self.gen_time = gen_time
        self.calls: list[str] = []

    def __call__(self, url: str, request: bytes, **_options: object) -> bytes:
        prefix = bytes.fromhex(seal_helpers.REQUEST_PREFIX_HEX)
        assert request.startswith(prefix), "not the request this client builds"
        digest = request[len(prefix) : len(prefix) + 32].hex()
        self.calls.append(digest)
        return seal_helpers.make_tsr(digest, self.gen_time)


def install_tsa(monkeypatch: pytest.MonkeyPatch, gen_time: dt.datetime) -> EchoTsa:
    fake = EchoTsa(gen_time)
    monkeypatch.setattr(tsa, "post_request", fake)
    return fake


def transport() -> hrrr_support.FakeTransport:
    """Serves the recorded crop at the real URLs of the run the scenarios read."""
    return hrrr_support.crop_transport(hrrr_support.CYCLE_DATE, CYCLE_HOUR, FORECAST_HOUR)
