"""Small sealed ledgers for the almanac tests, built offline with the real seal module.

A date is sealed by `lastlight.seal.seal` with the TSA call replaced by a canned response
(`seal_helpers.FakeTsa`), so the ledger on disk is exactly what `lastlight seal` writes. The
timestamp is the recorded response re-dated relative to the geometric sunset, which keeps every
clock time out of the tests.

A strip is sealed with its forecast, as in the real ledger: the forecast names it by SHA-256
(`strip_sha256`) and `seal` writes the tiny PNG beside it, at `YYYY/MM/DD.strip.png`. Pass `strip`
to `forecast_for` and `seal_date` for a sealed painting; without it the forecast names none, and
a file written beside it afterwards (`write_strip`) is tampering that `verify` reports.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import pytest
import seal_helpers as helpers
from PIL import Image

from lastlight.forecast import Forecast
from lastlight.seal import ledger_paths, seal, tsa
from lastlight.seal.ledger import SealResult

DATES = (dt.date(2026, 10, 11), dt.date(2026, 10, 12), dt.date(2026, 10, 13))
"""Three consecutive evenings; the newest is the last."""

STRIP_SIZE = (29, 7)


def forecast_for(
    date: dt.date,
    *,
    version: str = "0.0.1",
    notes: str | None = None,
    model_name: str | None = None,
    strip: bytes | None = None,
) -> Forecast:
    """The stub forecast moved to `date`, with its own model version (notes and name, if given).

    `strip_sha256` is the SHA-256 of `strip` when one is given, and null otherwise.
    """
    data = helpers.stub_dict()
    data["date"] = date.isoformat()
    data["model"]["version"] = version
    data["strip_sha256"] = None if strip is None else helpers.digest_hex(strip)
    if model_name is not None:
        data["model"]["name"] = model_name
    if notes is not None:
        data["notes"] = notes
    return Forecast.model_validate(data)


def seal_date(
    ledger: Path,
    date: dt.date,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "0.0.1",
    notes: str | None = None,
    model_name: str | None = None,
    strip: bytes | None = None,
    late: bool = False,
) -> SealResult:
    """Seal one date into `ledger`, with its painting when `strip` is given.

    `late` makes the timestamp arrive after sunset.
    """
    forecast = forecast_for(date, version=version, notes=notes, model_name=model_name, strip=strip)
    digest = helpers.digest_of(forecast)
    make_reply = helpers.tsr_after_sunset if late else helpers.tsr_before_sunset
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(make_reply(date, digest)))
    return seal(forecast, ledger, strip=strip)


def seal_dates(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seal `DATES` in order, each with its painting, with model versions 0.0.1, 0.0.2 and 0.0.3."""
    for number, date in enumerate(DATES, start=1):
        seal_date(ledger, date, monkeypatch, version=f"0.0.{number}", strip=strip_png())


def seal_dates_out_of_date_order(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seal `DATES` newest, oldest, middle, so the index lists them in no date order at all.

    The ledger lets any unsealed date be sealed at any time; model versions still follow the
    date (0.0.3 for the newest, 0.0.1 for the oldest) so a page can be told from another.
    Each evening is sealed with its painting, as in `seal_dates`.
    """
    newest, oldest, middle = DATES[2], DATES[0], DATES[1]
    for date, version in ((newest, "0.0.3"), (oldest, "0.0.1"), (middle, "0.0.2")):
        seal_date(ledger, date, monkeypatch, version=version, strip=strip_png())


def strip_png(color: tuple[int, int, int] = (200, 120, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", STRIP_SIZE, color).save(buffer, format="PNG")
    return buffer.getvalue()


def strip_path(ledger: Path, date: dt.date) -> Path:
    return ledger_paths(ledger, date).strip


def write_strip(ledger: Path, date: dt.date, png: bytes | None = None) -> Path:
    """Put a PNG at the painting's path after the date was sealed: tampering, not sealing.

    Whatever it replaces (or adds beside a forecast that names no strip) no longer matches the
    forecast's `strip_sha256`, so `verify` fails that evening. For a painting that is meant to be
    there, pass `strip=` to `seal_date` instead.
    """
    path = strip_path(ledger, date)
    path.write_bytes(strip_png() if png is None else png)
    return path


def overwrite_notes(ledger: Path, date: dt.date, notes: str) -> None:
    """Change the notes inside a sealed forecast file, the way a hand edit would."""
    path = ledger_paths(ledger, date).forecast
    document = json.loads(path.read_text(encoding="utf-8"))
    document["notes"] = notes
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
