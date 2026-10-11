"""Small sealed ledgers for the almanac tests, built offline with the real seal module.

A date is sealed by `lastlight.seal.seal` with the TSA call replaced by a canned response
(`seal_helpers.FakeTsa`), so the ledger on disk is exactly what `lastlight seal` writes. The
timestamp is the recorded response re-dated relative to the geometric sunset, which keeps every
clock time out of the tests. Strips are tiny PNGs written beside their forecast, at the path
the almanac reads: `YYYY/MM/DD.strip.png`.
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


def forecast_for(date: dt.date, *, version: str = "0.0.1", notes: str | None = None) -> Forecast:
    """The stub forecast moved to `date`, with its own model version (and notes, if given)."""
    data = helpers.stub_dict()
    data["date"] = date.isoformat()
    data["model"]["version"] = version
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
    late: bool = False,
) -> SealResult:
    """Seal one date into `ledger`; `late` makes the timestamp arrive after sunset."""
    forecast = forecast_for(date, version=version, notes=notes)
    digest = helpers.digest_of(forecast)
    make_reply = helpers.tsr_after_sunset if late else helpers.tsr_before_sunset
    monkeypatch.setattr(tsa, "post_request", helpers.FakeTsa(make_reply(date, digest)))
    return seal(forecast, ledger)


def seal_dates(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seal `DATES` in order, with model versions 0.0.1, 0.0.2 and 0.0.3."""
    for number, date in enumerate(DATES, start=1):
        seal_date(ledger, date, monkeypatch, version=f"0.0.{number}")


def seal_dates_out_of_date_order(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seal `DATES` newest, oldest, middle, so the index lists them in no date order at all.

    The ledger lets any unsealed date be sealed at any time; model versions still follow the
    date (0.0.3 for the newest, 0.0.1 for the oldest) so a page can be told from another.
    """
    newest, oldest, middle = DATES[2], DATES[0], DATES[1]
    for date, version in ((newest, "0.0.3"), (oldest, "0.0.1"), (middle, "0.0.2")):
        seal_date(ledger, date, monkeypatch, version=version)


def strip_png(color: tuple[int, int, int] = (200, 120, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", STRIP_SIZE, color).save(buffer, format="PNG")
    return buffer.getvalue()


def strip_path(ledger: Path, date: dt.date) -> Path:
    return ledger_paths(ledger, date).forecast.with_name(f"{date.day:02d}.strip.png")


def write_strip(ledger: Path, date: dt.date, png: bytes | None = None) -> Path:
    path = strip_path(ledger, date)
    path.write_bytes(strip_png() if png is None else png)
    return path


def overwrite_notes(ledger: Path, date: dt.date, notes: str) -> None:
    """Change the notes inside a sealed forecast file, the way a hand edit would."""
    path = ledger_paths(ledger, date).forecast
    document = json.loads(path.read_text(encoding="utf-8"))
    document["notes"] = notes
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8")
