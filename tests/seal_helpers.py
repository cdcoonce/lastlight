"""Shared pieces for the seal-ledger tests: the stub forecast, a recorded timestamp, fakes.

`stub-forecast.json` is a hand-written, deliberately untidy forecast (shuffled keys, float
noise). `STUB_CANONICAL` is its canonical form written out by hand, so the golden bytes do not
depend on the code under test. `stub-forecast.tsr` is a real RFC 3161 response from the public
TSA, recorded once for exactly those canonical bytes and sealed a day ahead of the stub's date.

Tests never write a clock time. Where a scenario needs "before" or "after" sunset, it computes
the geometric sunset at the reference point and offsets from it; `make_tsr` then re-dates a copy
of the recorded response. The copy's signature no longer matches, which is fine: this slice
checks the message imprint and the time, and the signature chain is a follow-up.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rfc3161_client import decode_timestamp_response

from lastlight.forecast import Forecast
from lastlight.geometry import elevation_crossings
from lastlight.seal.canonical import canonical_bytes
from lastlight.site import REFERENCE_POINT

FIXTURES = Path(__file__).parent / "fixtures" / "seal"
STUB_DATE = dt.date(2026, 10, 11)

STUB_CANONICAL = (
    b'{"curve":['
    b'{"L":78.1235,"a":12.3457,"b":45.6789,"sun_elevation":6.0,"view_elevation":2.0},'
    b'{"L":65.4321,"a":20.0,"b":38.1,"sun_elevation":0.0,"view_elevation":2.0},'
    b'{"L":40.5556,"a":8.88889,"b":10.0,"sun_elevation":-4.0,"view_elevation":5.0},'
    b'{"L":10.1,"a":-1.23457,"b":-2.5e-07,"sun_elevation":-8.0,"view_elevation":10.0}],'
    b'"date":"2026-10-11",'
    b'"inputs":[{"cycle_date":"2026-10-10","cycle_hour_utc":18,'
    b'"digest":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",'
    b'"forecast_hour":6,"source":"hrrr"}],'
    b'"model":{"code_sha":"0123456789abcdef0123456789abcdef01234567",'
    b'"name":"lastlight-stub","version":"0.0.1"},'
    b'"notes":"stub forecast for the seal tests",'
    b'"reference_point":{"elev_m":340.0,"lat":33.45,"lon":-112.07},'
    b'"schema_version":1,'
    b'"vividness":{"mean":0.42,"sd":0.07}}'
)
STUB_DIGEST = hashlib.sha256(STUB_CANONICAL).hexdigest()  # sha256 of the golden bytes

# The request the client sends for STUB_CANONICAL: DER, SHA-256 imprint, no nonce, certReq set.
STUB_REQUEST_HEX = "30390201013031300d060960864801650304020105000420" + STUB_DIGEST + "0101ff"


def digest_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(forecast: Forecast) -> str:
    return digest_hex(canonical_bytes(forecast))


def stub_text() -> str:
    return (FIXTURES / "stub-forecast.json").read_text()


def stub_dict() -> dict[str, Any]:
    return json.loads(stub_text())


def stub_forecast() -> Forecast:
    return Forecast.model_validate(stub_dict())


def recorded_tsr() -> bytes:
    return (FIXTURES / "stub-forecast.tsr").read_bytes()


def sunset(date: dt.date) -> dt.datetime:
    """Geometric sunset (a UTC instant) at the public reference point on `date`."""
    lat, lon, _elevation = REFERENCE_POINT
    return elevation_crossings(date, lat, lon)[0.0]


def make_tsr(digest_hex: str, gen_time: dt.datetime) -> bytes:
    """A copy of the recorded response that covers `digest_hex` and was issued at `gen_time`.

    Both fields have fixed width in DER, so they are patched in place; whole seconds only.
    """
    template = recorded_tsr()
    old_time = decode_timestamp_response(template).tst_info.gen_time
    old_stamp = old_time.strftime("%Y%m%d%H%M%SZ").encode()
    new_stamp = gen_time.astimezone(dt.UTC).strftime("%Y%m%d%H%M%SZ").encode()
    old_digest = bytes.fromhex(STUB_DIGEST)
    new_digest = bytes.fromhex(digest_hex)
    assert template.count(old_stamp) == 1 and template.count(old_digest) == 1
    return template.replace(old_stamp, new_stamp).replace(old_digest, new_digest)


def tsr_before_sunset(date: dt.date, digest_hex: str, minutes: float = 30) -> bytes:
    return make_tsr(digest_hex, sunset(date) - dt.timedelta(minutes=minutes))


def tsr_after_sunset(date: dt.date, digest_hex: str, minutes: float = 30) -> bytes:
    return make_tsr(digest_hex, sunset(date) + dt.timedelta(minutes=minutes))


class FakeTsa:
    """Stands in for `lastlight.seal.tsa.post_request`: records calls, returns a canned reply."""

    def __init__(self, reply: bytes) -> None:
        self.reply = reply
        self.calls: list[tuple[str, bytes]] = []

    def __call__(self, url: str, request: bytes, **_options: object) -> bytes:
        self.calls.append((url, request))
        return self.reply


def snapshot(root: Path) -> dict[str, bytes | None]:
    """Every file (bytes) and directory (None) under `root`, keyed by relative path."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


def flip_digit(text: str, needle: str) -> str:
    """`text` with the last digit of the one occurrence of `needle` changed by one."""
    assert text.count(needle) == 1
    changed = re.sub(r"\d(?=\D*$)", lambda m: str((int(m.group()) + 1) % 10), needle)
    return text.replace(needle, changed)
