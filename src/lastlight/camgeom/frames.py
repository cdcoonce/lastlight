"""Frames in the private archive: finding them, reading them and naming them.

The collector writes `<archive>/<camera>/<date>/<YYYYmmddTHHMMSSZ>.jpg`, where the stamp is
the frame's Last-Modified time in UTC. Frames never enter the repository (`docs/design.md`
section 3); a fit records each frame it used by the date of its evening, the sun's elevation
at the frame and the SHA-256 of its bytes, never by name or time.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_STAMP = re.compile(r"^(\d{8})T(\d{6})Z$")
_SUFFIXES = {".jpg", ".jpeg"}
_NEVER = dt.datetime.max.replace(tzinfo=dt.UTC)


@dataclass(frozen=True, slots=True)
class FrameFile:
    """A frame on disk and the UTC time in its name, if it has one."""

    path: Path
    when_utc: dt.datetime | None


def parse_frame_time(name: str | Path) -> dt.datetime | None:
    """The UTC time a frame's file name carries, or None if it is not a collector stamp."""
    match = _STAMP.match(Path(name).stem)
    if match is None:
        return None
    try:
        parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.UTC)


def discover_frames(root: Path) -> list[FrameFile]:
    """Every JPEG under `root`, oldest first; frames with no time in their name come last."""
    if not root.is_dir():
        raise FileNotFoundError(f"frames directory not found: {root}")
    found = [
        FrameFile(path, parse_frame_time(path))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUFFIXES
    ]
    return sorted(found, key=lambda frame: (frame.when_utc or _NEVER, frame.path.name))


def load_rgb(path: Path) -> np.ndarray:
    """A frame as rows by columns by three uint8 values."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def frame_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evening_of(when_utc: dt.datetime, lon: float) -> dt.date:
    """The local mean solar date of an instant: the evening a late-afternoon frame belongs to.

    Local mean solar time is UTC plus four minutes per degree of east longitude. A frame taken
    after 00:00 UTC is usually still the previous evening in Arizona, and this keeps it there.
    """
    return (when_utc + dt.timedelta(hours=lon / 15.0)).date()
