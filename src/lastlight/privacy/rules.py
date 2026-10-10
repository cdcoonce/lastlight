"""The text rules: P001 clock-time, P002 sun-event-time and P003 coordinates.

Each rule finds character spans in a string and knows nothing about files, lines or the
allowlist; `lastlight.privacy.scanner` turns spans into findings. The patterns are the
contract of docs/design.md section 3: public artifacts are referenced to sun elevation and
the public reference point, never to a clock time or to the camera's real position.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from dataclasses import dataclass

from lastlight.site import REFERENCE_POINT

RULES: dict[str, str] = {
    "P001": "clock-time",
    "P002": "sun-event-time",
    "P003": "coordinates",
    "P004": "image-metadata",
    "P005": "private-site",
}


@dataclass(frozen=True)
class Hit:
    """A rule matched `text[start:end]`."""

    rule: str
    start: int
    end: int


# --- P001 clock-time ------------------------------------------------------------------------

_HOUR = r"(?:[01]?\d|2[0-3])"
_HOUR2 = r"(?:[01]\d|2[0-3])"
_MINUTE = r"[0-5]\d"
_SECONDS = r"(?::[0-5]\d(?:\.\d+)?)?"
_ZONE = r"(?:Z|[+-]\d{2}:?\d{2})?"
_MERIDIEM = r"[AaPp]\.?[Mm]"

# Order matters: at one position the first alternative that matches wins, so a full ISO
# datetime is one match rather than a date followed by a time.
_CLOCK_TIME = re.compile(
    "|".join(
        [
            # 2026-10-10T00:24:00Z, 2026-10-10 17:42
            rf"(?<!\d)\d{{4}}-\d{{2}}-\d{{2}}[T ]{_HOUR2}:{_MINUTE}{_SECONDS}{_ZONE}(?!\d)",
            # 20261010T002400Z
            r"(?<!\d)\d{8}T\d{4}(?:\d{2})?Z?(?!\d)",
            # T17:42Z, the time half of an ISO datetime
            rf"(?<![A-Za-z0-9])T{_HOUR2}:{_MINUTE}{_SECONDS}{_ZONE}(?!\d)",
            # 17:42, 5:42 PM, 17:42:09.125. Letters and punctuation may touch it (a leak
            # named `sunset-17:42` or `frame_17:42` is still a leak), but not slices
            # (a[10:20]), UTC offsets (UTC-07:00), or the tail of a longer number.
            rf"(?<![\d.:\[])(?<![\dZ][+-])(?<![Uu][Tt][Cc][+-])(?<![Gg][Mm][Tt][+-])"
            rf"{_HOUR}:{_MINUTE}{_SECONDS}(?:\s?{_MERIDIEM}\b\.?)?(?![\d:\]]|\.\d)",
            # 17h42, but not a duration such as 17h42m
            rf"(?<![\d.]){_HOUR}[hH]{_MINUTE}(?!\w)",
        ]
    )
)


def contains_clock_time(text: str) -> bool:
    """True when `text` holds anything rule P001 would flag.

    Other modules (the forecast schema) call this to refuse a string a scan would reject.
    """
    return _CLOCK_TIME.search(text) is not None


def clock_time_hits(text: str) -> list[Hit]:
    return [Hit("P001", m.start(), m.end()) for m in _CLOCK_TIME.finditer(text)]


# --- P002 sun-event-time --------------------------------------------------------------------

_SUN_EVENT = re.compile(r"(?<![A-Za-z])(?:sunsets?|sunrises?|dusk|dawn)(?![A-Za-z])", re.IGNORECASE)

# Times P001 does not flag on their own but that still pin an event down: "5 PM", "1742Z".
_LOOSE_TIME = re.compile(
    "|".join(
        [
            rf"(?<![\d.:])(?:1[0-2]|0?[1-9])\s?{_MERIDIEM}(?![A-Za-z])\.?",
            r"(?<![\d.:])(?:[01]\d|2[0-3])[0-5]\d\s?(?:hrs?|hours|h|z|utc|local)(?![A-Za-z])",
        ]
    ),
    re.IGNORECASE,
)

SUN_EVENT_WINDOW = 40
"""A sun-event word and a time are linked when at most this many characters separate them."""


def _merged_time_spans(text: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in _CLOCK_TIME.finditer(text)]
    spans += [(m.start(), m.end()) for m in _LOOSE_TIME.finditer(text)]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def sun_event_hits(text: str) -> list[Hit]:
    words = list(_SUN_EVENT.finditer(text))
    if not words:
        return []
    spans = _merged_time_spans(text)
    starts = [start for start, _end in spans]
    hits: list[Hit] = []
    for word in words:
        nearest: tuple[int, int, int] | None = None  # (gap, start, end) of the closest time
        index = bisect_left(starts, word.end())
        if index < len(spans):
            start, end = spans[index]
            nearest = (start - word.end(), start, end)
        if index > 0:
            start, end = spans[index - 1]
            gap = max(word.start() - end, 0)
            if nearest is None or gap < nearest[0]:
                nearest = (gap, start, end)
        if nearest is not None and nearest[0] <= SUN_EVENT_WINDOW:
            hits.append(Hit("P002", min(word.start(), nearest[1]), max(word.end(), nearest[2])))
    return hits


# --- P003 coordinates -----------------------------------------------------------------------

LAT_RANGE = (31.0, 37.5)
LON_RANGE = (109.0, 115.0)
"""The Arizona bounding box. A longitude counts in either sign convention: -112.07 or 112.07."""

_COORDINATE_GAP = 40
_DECIMAL = re.compile(r"(?<![\d.])[-+\u2212]?\d{1,3}\.\d+(?!\d|\.\d)")


def _is_latitude(value: float) -> bool:
    return LAT_RANGE[0] <= value <= LAT_RANGE[1]


def _is_longitude(value: float) -> bool:
    return LON_RANGE[0] <= abs(value) <= LON_RANGE[1]


def _is_coordinate_pair(first: float, second: float) -> bool:
    return (_is_latitude(first) and _is_longitude(second)) or (
        _is_longitude(first) and _is_latitude(second)
    )


def _is_reference_point(first: float, second: float) -> bool:
    lat, lon, _elevation = REFERENCE_POINT
    for latitude, longitude in ((first, second), (second, first)):
        if math.isclose(latitude, lat, abs_tol=1e-6) and math.isclose(
            abs(longitude), abs(lon), abs_tol=1e-6
        ):
            return True
    return False


def _is_pair_separator(gap: str) -> bool:
    """Short, digit-free text such as ', ' or ' N, ' or '", "lon": ' between two numbers."""
    return len(gap) <= _COORDINATE_GAP and gap.count("\n") <= 1 and not re.search(r"\d", gap)


def coordinate_hits(text: str) -> list[Hit]:
    """Latitude/longitude decimal pairs in the Arizona box, other than the reference point.

    The pair is two consecutive decimal numbers with only a short, digit-free separator
    between them, so `33.4, -112.0`, `33.4 N 112.0 W`, `{"lat": 33.4, "lon": -112.0}` and
    a TOML `lat = ...` line followed by a `lon = ...` line are all pairs. Either order.
    """
    tokens = [
        (m.start(), m.end(), float(m.group().replace("\u2212", "-")))
        for m in _DECIMAL.finditer(text)
    ]
    hits: list[Hit] = []
    i = 0
    while i + 1 < len(tokens):
        (start, end, first), (next_start, next_end, second) = tokens[i], tokens[i + 1]
        if _is_coordinate_pair(first, second) and _is_pair_separator(text[end:next_start]):
            if not _is_reference_point(first, second):
                hits.append(Hit("P003", start, next_end))
            i += 2
        else:
            i += 1
    return hits


# --- all text rules -------------------------------------------------------------------------


def text_hits(text: str) -> list[Hit]:
    """Every P001, P002 and P003 hit in `text`, in order of position."""
    hits = [*clock_time_hits(text), *sun_event_hits(text), *coordinate_hits(text)]
    return sorted(hits, key=lambda hit: (hit.start, hit.rule))
