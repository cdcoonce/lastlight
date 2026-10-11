"""The registry of tier A cameras: public webcams whose frames are the primary truth.

The registry lives in `registry.toml` beside this module. `load_registry` reads and validates
it, and refuses anything that could make the collector or the site misbehave quietly: a
missing field, a typo'd key, an id that is not a safe directory name, or a camera that is
published without a recorded reason (`publish_thumbnails = true` needs a `basis`;
`docs/design.md` section 3).
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final

_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_REQUIRED_TEXT: Final = ("id", "operator", "page_url", "image_url", "attribution")
_KNOWN_KEYS: Final = frozenset(
    {
        *_REQUIRED_TEXT,
        "view_azimuth_deg",
        "position",
        "publish_thumbnails",
        "basis",
        "position_note",
    }
)
_POSITION_KEYS: Final = ("lat", "lon", "elev_m")


class RegistryError(ValueError):
    """The registry is malformed. The message names the camera and the offending field."""


@dataclass(frozen=True, slots=True)
class Position:
    """Where a camera stands: latitude and longitude in degrees (east positive), elevation in m.

    These are approximate and public: the operator's own site description, not an exact survey.
    """

    lat: float
    lon: float
    elev_m: float


@dataclass(frozen=True, slots=True)
class Camera:
    """One tier A camera. `view_azimuth_deg` is degrees clockwise from true north."""

    id: str
    operator: str
    page_url: str
    image_url: str
    view_azimuth_deg: float
    attribution: str
    position: Position
    publish_thumbnails: bool = False
    basis: str = ""
    position_note: str = ""


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _text(entry: dict[str, Any], key: str, label: str) -> str:
    if key not in entry:
        raise RegistryError(f"{label}: missing required field '{key}'")
    value = entry[key]
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{label}: '{key}' must be a non-empty string")
    return value


def _url(entry: dict[str, Any], key: str, label: str) -> str:
    value = _text(entry, key, label)
    if not value.startswith(("http://", "https://")):
        raise RegistryError(f"{label}: '{key}' must be an http or https URL, got {value!r}")
    return value


def _position(entry: dict[str, Any], label: str) -> Position:
    if "position" not in entry:
        raise RegistryError(f"{label}: missing required field 'position'")
    raw = entry["position"]
    if not isinstance(raw, dict):
        raise RegistryError(f"{label}: 'position' must be a table with lat, lon and elev_m")
    for key in _POSITION_KEYS:
        if key not in raw:
            raise RegistryError(f"{label}: position is missing required field '{key}'")
    unknown = sorted(set(raw) - set(_POSITION_KEYS))
    if unknown:
        raise RegistryError(f"{label}: position has unknown key(s) {unknown}")
    if not all(_number(raw[key]) for key in _POSITION_KEYS):
        raise RegistryError(f"{label}: position lat, lon and elev_m must be numbers")
    lat, lon, elev_m = float(raw["lat"]), float(raw["lon"]), float(raw["elev_m"])
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise RegistryError(f"{label}: position {lat}, {lon} is outside the globe")
    return Position(lat=lat, lon=lon, elev_m=elev_m)


def _camera(entry: object, index: int) -> Camera:
    if not isinstance(entry, dict):
        raise RegistryError(f"camera #{index}: expected a table")
    label = f"camera #{index}"
    camera_id = _text(entry, "id", label)
    label = f"camera {camera_id!r}"
    if not _ID_PATTERN.fullmatch(camera_id):
        raise RegistryError(
            f"{label}: 'id' must be letters, digits, '-' or '_' (it names a directory)"
        )
    unknown = sorted(set(entry) - _KNOWN_KEYS)
    if unknown:
        raise RegistryError(f"{label}: unknown key(s) {unknown}")

    operator = _text(entry, "operator", label)
    page_url = _url(entry, "page_url", label)
    image_url = _url(entry, "image_url", label)
    attribution = _text(entry, "attribution", label)

    if "view_azimuth_deg" not in entry:
        raise RegistryError(f"{label}: missing required field 'view_azimuth_deg'")
    azimuth = entry["view_azimuth_deg"]
    if not _number(azimuth) or not 0.0 <= azimuth < 360.0:
        raise RegistryError(f"{label}: 'view_azimuth_deg' must be a number in [0, 360)")

    publish = entry.get("publish_thumbnails", False)
    if not isinstance(publish, bool):
        raise RegistryError(f"{label}: 'publish_thumbnails' must be true or false")
    basis = entry.get("basis", "")
    if not isinstance(basis, str):
        raise RegistryError(f"{label}: 'basis' must be a string")
    if publish and not basis.strip():
        raise RegistryError(
            f"{label}: publish_thumbnails = true needs a non-empty 'basis' recording why "
            "publishing is acceptable"
        )
    note = entry.get("position_note", "")
    if not isinstance(note, str):
        raise RegistryError(f"{label}: 'position_note' must be a string")

    return Camera(
        id=camera_id,
        operator=operator,
        page_url=page_url,
        image_url=image_url,
        view_azimuth_deg=float(azimuth),
        attribution=attribution,
        position=_position(entry, label),
        publish_thumbnails=publish,
        basis=basis,
        position_note=note,
    )


def parse_registry(text: str) -> dict[str, Camera]:
    """Parse registry TOML into cameras keyed by id, in file order."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RegistryError(f"the registry is not valid TOML: {error}") from error
    entries = document.get("camera", [])
    if not isinstance(entries, list) or not entries:
        raise RegistryError("the registry lists no cameras (expected [[camera]] tables)")
    cameras: dict[str, Camera] = {}
    for index, entry in enumerate(entries, start=1):
        camera = _camera(entry, index)
        if camera.id in cameras:
            raise RegistryError(f"duplicate camera id {camera.id!r}")
        cameras[camera.id] = camera
    return cameras


def load_registry(path: Path | None = None) -> dict[str, Camera]:
    """Load the registry shipped with the package, or the TOML file at `path`."""
    if path is None:
        text = resources.files(__package__).joinpath("registry.toml").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse_registry(text)
