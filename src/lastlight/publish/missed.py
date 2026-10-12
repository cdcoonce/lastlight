"""Evenings that were never sealed: which they are, and the newest one the site may report on.

An absent seal must never be mistaken for a forecast, and a visitor must be able to tell a
missed evening from one the site simply has not got to yet. Two small rules, both about dates:

* The site reports on every evening up to `through`. By default `through` is the newest evening
  whose deadline has passed (`default_through`): an evening is closed at the instant sealing it
  would be refused, the sun at the horizon at the reference point, so the site never calls an
  evening missed while a seal for it could still be made.
* An evening is *missed* when it lies between the first sealed evening and `through`, inclusive,
  and has no row in the index (`missed_evenings`). Evenings before the first seal are not
  missed: the project had not started. An evening sealed too late has a row, so it is listed as
  sealed and marked not verified, not as missed.

Dates only: the deadline instant decides, and is never printed (`docs/design.md` section 3).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

from lastlight.cameras.archive import evening_date
from lastlight.geometry import elevation_crossings
from lastlight.site import REFERENCE_POINT

ONE_DAY = dt.timedelta(days=1)


def _deadline(evening: dt.date) -> dt.datetime:
    """The instant `evening` closes: the same one as `lastlight.daily.evening_deadline`.

    Computed here rather than imported because `lastlight.daily` loads the GRIB library and the
    site build has no use for it; a test pins that the two agree.
    """
    lat, lon, _elevation = REFERENCE_POINT
    return elevation_crossings(evening, lat, lon)[0.0]


def default_through(now_utc: dt.datetime) -> dt.date:
    """The newest evening whose deadline has passed at `now_utc`.

    That is the local mean solar date at the reference point (the convention of
    `lastlight.cameras.archive.evening_date`) once the sun has reached the horizon on it, and the
    day before until then.
    """
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("the time must be timezone-aware; a naive datetime is ambiguous")
    now = now_utc.astimezone(dt.UTC)
    today = evening_date(now, REFERENCE_POINT[1])
    return today if now >= _deadline(today) else today - ONE_DAY


def missed_evenings(sealed: Iterable[dt.date], through: dt.date) -> tuple[dt.date, ...]:
    """Dates from the first of `sealed` to `through`, inclusive, that are not in `sealed`.

    Newest first. No sealed evening at all, or a `through` before the first one, means none.
    """
    sealed_dates = set(sealed)
    if not sealed_dates:
        return ()
    days = (through - min(sealed_dates)).days
    candidates = (through - ONE_DAY * offset for offset in range(days + 1))
    return tuple(date for date in candidates if date not in sealed_dates)
