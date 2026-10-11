"""A synthetic private archive for the observe tests, its frames named as the collector names them.

The frames are generated here (a horizon-to-zenith gradient laid over the shipped WHTM1 skyline,
with a gain per channel), written as JPEGs under `<archive>/WHTM1/<date>/<stem>.jpg`, and named by
`lastlight.cameras.archive.frame_stem`, the collector's own naming code, at the instant the sun is
at the elevation asked for. No real frame, name or time is used.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from observe_support import (
    gradient_linear,
    offsets_above_skyline,
    srgb_encode,
    to_uint8,
)
from PIL import Image

from lastlight.cameras import Camera, load_registry
from lastlight.cameras.archive import EveningArchive, frame_stem
from lastlight.camgeom.fitfile import load_fit
from lastlight.camgeom.mask import skyline_top_rows
from lastlight.geometry import elevation_crossings, solar_position

EVENING = dt.date(2026, 10, 10)


def whtm1() -> Camera:
    return load_registry()["WHTM1"]


def instant_at_sun(elevation_deg: float, camera: Camera | None = None) -> dt.datetime:
    """The UTC instant, to the second, at which the sun is setting through `elevation_deg`.

    Found by bisection on `solar_position` between a few hours before the sun reaches +6 degrees
    and the instant it reaches -8 (inside that span the elevation falls monotonically).
    """
    camera = camera or whtm1()
    lat, lon = camera.position.lat, camera.position.lon
    crossings = elevation_crossings(EVENING, lat, lon)
    early = crossings[6.0] - dt.timedelta(hours=4)
    late = crossings[-8.0]
    for _ in range(40):
        middle = early + (late - early) / 2
        if solar_position(middle, lat, lon).elevation_deg > elevation_deg:
            early = middle
        else:
            late = middle
    return (early + (late - early) / 2).replace(microsecond=0)


def gradient_frame_for_whtm1(gains=(1.0, 1.0, 1.0)) -> np.ndarray:
    """The synthetic gradient over the frozen WHTM1 skyline, as 8-bit codes."""
    top = skyline_top_rows(load_fit("WHTM1"))
    linear = gradient_linear(offsets_above_skyline(top)) * np.asarray(gains, dtype=float)
    return to_uint8(srgb_encode(linear))


def write_jpeg(path: Path, codes: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(codes).save(path, format="JPEG", quality=95, subsampling=0)
    return path


def fill_archive(
    root: Path,
    sun_elevations: list[float],
    *,
    gains: list[tuple[float, float, float]] | None = None,
) -> dict[float, Path]:
    """Write one frame per sun elevation into `root`; return each frame's path by elevation."""
    folder = EveningArchive(root, "WHTM1", EVENING).directory
    paths = {}
    for index, elevation in enumerate(sun_elevations):
        stem = frame_stem(instant_at_sun(elevation))
        gain = (1.0, 1.0, 1.0) if gains is None else gains[index % len(gains)]
        paths[elevation] = write_jpeg(folder / f"{stem}.jpg", gradient_frame_for_whtm1(gain))
    return paths
