"""The forecast schema: what lastlight seals before sunset (`docs/design.md` sections 2 and 4).

A `Forecast` is one evening's prediction at the public reference point: the colour of the
western sky (CIELAB, D65) at each pair of sun elevation and view elevation, plus the
pre-registered vividness scalar with its spread. `lastlight.seal` turns it into canonical
JSON, hashes it and gets the hash timestamped.

The schema is strict on purpose, because a sealed forecast can never be edited or removed:

- Unknown fields are errors, numbers must be finite and are never read from strings.
- The date is a plain calendar date. No field holds a time of day.
- Every string is checked against the privacy scanner's text rules (clock times, sun-event
  times, coordinates), because `lastlight privacy-scan` runs over the ledger and a sealed leak
  could not be taken back. Error messages never repeat the offending text.
- The reference point must be the public one (`lastlight.site.REFERENCE_POINT`), so a forecast
  made with the private camera site's coordinates cannot be sealed by mistake.

`cycle_hour_utc` is the hour of the weather-model cycle an input came from (for example the
18 of an 18Z run). It is a public model-run label, not the time of anything lastlight did.

`strip_sha256` seals the painting. The strip is the picture of the curve that the almanac shows;
if only the numbers were sealed, the picture could be swapped afterwards. The field is required
but may be null: a forecast always says whether it has a strip, and a code path that forgets the
painting fails validation instead of sealing a forecast without one. `lastlight verify` checks
the PNG beside the forecast against it.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from lastlight.privacy.rules import RULES, text_hits
from lastlight.site import REFERENCE_POINT

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REFERENCE_TOLERANCE = 1e-6


def _plain_date(value: Any) -> dt.date:
    """A `date` or a `YYYY-MM-DD` string; a datetime or any other spelling is refused."""
    if isinstance(value, dt.datetime):
        raise ValueError("must be a calendar date, not a date with a time")
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        return dt.date.fromisoformat(value)  # a ValueError here becomes a validation error
    raise ValueError("must be a calendar date written YYYY-MM-DD")


def _clean_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("must be text that can be encoded as UTF-8") from None
    hits = text_hits(value)
    if hits:
        rule = hits[0].rule
        raise ValueError(f"contains text that privacy rule {rule} ({RULES[rule]}) forbids")
    return value


IsoDate = Annotated[dt.date, BeforeValidator(_plain_date)]
Sha256Hex = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, AfterValidator(_clean_text)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Spread = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Model(_Record):
    """The code that made the forecast."""

    name: Text
    version: Text
    code_sha: Text


class ReferencePoint(_Record):
    """Latitude and longitude in degrees (east positive) and elevation in metres."""

    lat: Number
    lon: Number
    elev_m: Number

    @model_validator(mode="after")
    def _is_the_public_reference_point(self) -> Self:
        lat, lon, elevation = REFERENCE_POINT
        same = (
            math.isclose(self.lat, lat, rel_tol=0, abs_tol=_REFERENCE_TOLERANCE)
            and math.isclose(self.lon, lon, rel_tol=0, abs_tol=_REFERENCE_TOLERANCE)
            and math.isclose(self.elev_m, elevation, rel_tol=0, abs_tol=_REFERENCE_TOLERANCE)
        )
        if not same:
            raise ValueError("must be the public reference point")
        return self


class Input(_Record):
    """One weather-model field the forecast read."""

    source: Text
    cycle_date: IsoDate
    cycle_hour_utc: Annotated[int, Field(strict=True, ge=0, le=23)]
    forecast_hour: Annotated[int, Field(strict=True, ge=0)]
    digest: Text


class CurvePoint(_Record):
    """Sky colour (CIELAB, D65) at one sun elevation and view elevation, both in degrees."""

    sun_elevation: Number
    view_elevation: Number
    L: Number
    a: Number
    b: Number


class Vividness(_Record):
    """The evening's vividness scalar: forecast mean and standard deviation."""

    mean: Number
    sd: Spread


class Forecast(_Record):
    """One sealed evening, referenced to sun elevation and never to clock time."""

    schema_version: Literal[1]
    date: IsoDate
    model: Model
    reference_point: ReferencePoint
    inputs: tuple[Input, ...]
    curve: tuple[CurvePoint, ...]
    vividness: Vividness
    notes: Text
    strip_sha256: Sha256Hex | None
