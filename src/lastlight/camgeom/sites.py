"""The private camera site file: where each public camera stands and which way it looks.

A public webcam's position is published by its operator, but this project keeps every
coordinate out of the repository, its CI and its artifacts on purpose (`docs/design.md`
section 3): the privacy scanner flags any latitude and longitude pair in Arizona other than
the public reference point, and a rule with exceptions for "harmless" coordinates is a rule
that will leak the one that was not. So a fit reads the position at run time from a file the
operator keeps on the machine that runs it, and the frozen fit stores only the orientation.

The file is TOML with one table per camera:

    [WHTM1]
    lat = ...                 # degrees north
    lon = ...                 # degrees east, negative in the west
    view_azimuth_deg = ...    # the operator's stated view direction
    mount_height_m = 10.0     # optional, metres above the ground at the camera

Its path is the `--site-file` argument or the `LASTLIGHT_CAMERA_SITES` environment variable.
Error messages name keys and files, never the values in them.
"""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

SITES_ENV = "LASTLIGHT_CAMERA_SITES"
DEFAULT_MOUNT_HEIGHT_M = 10.0


class SiteFileError(ValueError):
    """The site file is missing, unreadable or does not describe the camera."""


@dataclass(frozen=True, slots=True)
class CameraSite:
    camera_id: str
    lat: float
    lon: float
    view_azimuth_deg: float
    mount_height_m: float = DEFAULT_MOUNT_HEIGHT_M


def resolve_site_file(argument: Path | None) -> Path:
    """The site file named by the command line, else by the environment."""
    if argument is not None:
        return argument
    from_environment = os.environ.get(SITES_ENV)
    if from_environment:
        return Path(from_environment)
    raise SiteFileError(f"no camera site file: pass --site-file or set {SITES_ENV}")


def _number(table: dict, key: str, camera_id: str, low: float, high: float) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SiteFileError(f"camera {camera_id} in the site file needs a number for `{key}`")
    if not (math.isfinite(value) and low <= value <= high):
        raise SiteFileError(f"`{key}` of camera {camera_id} is outside {low:g} to {high:g}")
    return float(value)


def load_camera_site(camera_id: str, path: Path) -> CameraSite:
    """Read one camera's entry from the site file."""
    try:
        text = path.read_text()
    except OSError as error:
        raise SiteFileError(f"camera site file not found or unreadable: {path}") from error
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise SiteFileError(f"camera site file {path.name} is not valid TOML") from error
    table = document.get(camera_id)
    if not isinstance(table, dict):
        raise SiteFileError(f"no entry for camera {camera_id} in the site file {path.name}")
    mount = table.get("mount_height_m", DEFAULT_MOUNT_HEIGHT_M)
    return CameraSite(
        camera_id=camera_id,
        lat=_number(table, "lat", camera_id, -90.0, 90.0),
        lon=_number(table, "lon", camera_id, -180.0, 180.0),
        view_azimuth_deg=_number(table, "view_azimuth_deg", camera_id, 0.0, 360.0),
        mount_height_m=_number({"mount_height_m": mount}, "mount_height_m", camera_id, 0.0, 500.0),
    )
