"""The private `observed.json` files from `lastlight observe`, read as untrusted input.

One file per evening, named `YYYY-MM-DD.json`, schema version 1. The check reads very little of
each: the camera, the date, and for every frame the sun elevation and the lowest band's
`log_bg` and `log_rg` (`bands[0]`). A frame is used when `bands[0].log_bg` is not null (criterion
definition 2); nothing else excludes a frame, so the light-dome flag, the other bands and the
pixel counts are not looked at.

Everything is checked before use: types (a bool is not a number), finiteness, the camera, the
schema version and that the file is named for its own date. A file that fails is refused with
one line naming the file and the field, and the error text never repeats more of the file than a
short `repr`. Bytes are decoded as UTF-8 with replacement, so a damaged file is a refusal and not
a crash.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lastlight.renderer_check.rules import CAMERA_ID

SCHEMA_VERSION: Final = 1
_FILE_NAME: Final = re.compile(r"(\d{4}-\d{2}-\d{2})\.json")
_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHOWN: Final = 40


class ObservedError(ValueError):
    """An observed file or directory the check refuses. The message is one line."""


class _Bad(Exception):
    """Internal: a field that fails its check."""

    def __init__(self, field: str, problem: str) -> None:
        super().__init__(f"{field}: {problem}")
        self.field = field
        self.problem = problem


@dataclass(frozen=True)
class UsedFrame:
    """A used frame: the sun's elevation and the lowest band's two observed contrasts."""

    sun_elevation_deg: float
    log_bg: float
    log_rg: float | None


@dataclass(frozen=True)
class ObservedEvening:
    """An evening's used frames, in the order the file lists them."""

    date: dt.date
    frames: tuple[UsedFrame, ...]


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _SHOWN else text[:_SHOWN] + "..."


def _number(value: Any, field: str) -> float:
    """`value` as a finite float; a bool, a string or anything non-finite is refused."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Bad(field, f"expected a number, got {_show(value)}")
    try:
        number = float(value)
    except OverflowError:
        raise _Bad(field, "the number is too large") from None
    if not math.isfinite(number):
        raise _Bad(field, f"expected a finite number, got {_show(value)}")
    return number


def _optional_number(value: Any, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _frame(index: int, raw: Any) -> UsedFrame | None:
    """The used frame at `index`, or None when `bands[0].log_bg` is null."""
    where = f"frames[{index}]"
    if not isinstance(raw, dict):
        raise _Bad(where, f"expected an object, got {_show(raw)}")
    bands = raw.get("bands")
    if not isinstance(bands, list) or not bands:
        raise _Bad(f"{where}.bands", "expected a non-empty list")
    band = bands[0]
    if not isinstance(band, dict):
        raise _Bad(f"{where}.bands[0]", f"expected an object, got {_show(band)}")
    if "log_bg" not in band:
        raise _Bad(f"{where}.bands[0].log_bg", "the field is missing")
    log_bg = _optional_number(band["log_bg"], f"{where}.bands[0].log_bg")
    if log_bg is None:
        return None
    return UsedFrame(
        sun_elevation_deg=_number(raw.get("sun_elevation_deg"), f"{where}.sun_elevation_deg"),
        log_bg=log_bg,
        log_rg=_optional_number(band.get("log_rg"), f"{where}.bands[0].log_rg"),
    )


def _evening(document: Any, name: str) -> ObservedEvening:
    if not isinstance(document, dict):
        raise _Bad("(file)", "expected a JSON object")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise _Bad("schema_version", f"expected {SCHEMA_VERSION}, got {_show(version)}")
    camera = document.get("camera_id")
    if camera != CAMERA_ID:
        raise _Bad("camera_id", f"expected {CAMERA_ID!r}, got {_show(camera)}")
    stem = name.removesuffix(".json")
    written = document.get("date")
    if not isinstance(written, str) or not _ISO_DATE.fullmatch(written):
        raise _Bad("date", f"expected a date written YYYY-MM-DD, got {_show(written)}")
    if written != stem:
        raise _Bad("date", f"{_show(written)} disagrees with the file name, which is for {stem}")
    try:
        day = dt.date.fromisoformat(written)
    except ValueError:
        raise _Bad("date", f"{_show(written)} is not a real calendar date") from None
    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise _Bad("frames", f"expected a list, got {_show(raw_frames)}")
    used = [frame for index, raw in enumerate(raw_frames) if (frame := _frame(index, raw))]
    return ObservedEvening(date=day, frames=tuple(used))


def read_observed_file(path: Path) -> ObservedEvening:
    """The evening in one observed file, or `ObservedError` naming the file and the field."""
    name = path.name
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
        document = json.loads(text)
    except OSError as error:
        raise ObservedError(f"{name}: (file): cannot be read ({error.strerror})") from None
    except (ValueError, RecursionError):
        raise ObservedError(f"{name}: (file): not valid JSON") from None
    try:
        return _evening(document, name)
    except _Bad as bad:
        raise ObservedError(f"{name}: {bad.field}: {bad.problem}") from None


def read_observed_dir(directory: Path) -> tuple[ObservedEvening, ...]:
    """Every `YYYY-MM-DD.json` in `directory`, in date order; any refused file refuses all."""
    if not directory.is_dir():
        raise ObservedError(f"{directory}: not a directory")
    evenings = []
    for path in sorted(directory.iterdir()):
        match = _FILE_NAME.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        evenings.append(read_observed_file(path))
    if not evenings:
        raise ObservedError(f"{directory}: no YYYY-MM-DD.json observed files in it")
    return tuple(sorted(evenings, key=lambda evening: evening.date))
