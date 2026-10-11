"""The registry of tier A cameras: public webcams whose frames are the primary truth.

The registry lives in `registry.toml` beside this module. `load_registry` reads and validates
it, and refuses anything that could make the collector or the site misbehave quietly: a
missing field, a typo'd key, an id that is not a safe directory name, or a camera that is
published without a recorded reason (`publish_thumbnails = true` needs a `basis`;
`docs/design.md` section 3).

A camera may also carry a `[camera.contrast]` table: the tier A observation's bands, defined in
pixel rows above the camera's frozen skyline (`ContrastBands`). Whether those rows lie inside the
camera's sky is a question for its frozen fit, so it is answered by `lastlight.observe.bands`,
not here; the loader checks only that the table is well formed.
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
        "contrast",
    }
)
_POSITION_KEYS: Final = ("lat", "lon", "elev_m")
_CONTRAST_KEYS: Final = frozenset(
    {
        "skyline_margin_rows",
        "bands_rows",
        "reference_rows",
        "light_dome_columns",
        "light_dome_provisional",
    }
)


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
class ContrastBands:
    """Where a camera's tier A contrast bands are, in pixel space.

    Rows are counted upward from the frozen skyline, column by column: a pair `(lo, hi)` is the
    half-open range of offsets `lo <= d < hi`, where `d` is how many rows a pixel sits above the
    skyline in its own column. The bands are in pixels, not degrees, because the pixel scale of a
    public camera is not identified until the sun enters its frame (`docs/design.md` section 1).

    - `skyline_margin_rows`: rows kept clear above the skyline, where terrain and sky blend. No
      band or reference may start inside it.
    - `bands_rows`: the scored bands, lowest first, ascending and not overlapping.
    - `reference_rows`: the band every other band is differenced against.
    - `light_dome_columns`: half-open column ranges `(lo, hi)` of artificial city light, excluded
      from every band while the sun is below -4 degrees. Columns, not azimuths: the frame is only
      about 19 degrees wide.
    - `light_dome_provisional`: true while the dome's columns rest on too little data to be
      called a measurement (one overcast frame with the sun below -6 degrees). Echoed into
      `observed.json`, so every observation says whether it was read with a seed or a refit.
    """

    skyline_margin_rows: int
    bands_rows: tuple[tuple[int, int], ...]
    reference_rows: tuple[int, int]
    light_dome_columns: tuple[tuple[int, int], ...] = ()
    light_dome_provisional: bool = False


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
    contrast: ContrastBands | None = None


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


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _range_list(
    value: object, label: str, field: str, *, many: bool
) -> tuple[tuple[int, int], ...]:
    """`[[lo, hi], ...]` (or one `[lo, hi]` when not `many`) of whole numbers, 0 <= lo < hi."""
    pairs = value if many else [value]
    if not isinstance(pairs, list) or (many and not pairs):
        shape = "a non-empty list of [lo, hi] pairs" if many else "a [lo, hi] pair"
        raise RegistryError(f"{label}: contrast '{field}' must be {shape}")
    out: list[tuple[int, int]] = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or not all(_whole(v) for v in pair):
            raise RegistryError(
                f"{label}: contrast '{field}' entries must be [lo, hi] whole numbers"
            )
        low, high = pair
        if not 0 <= low < high:
            raise RegistryError(
                f"{label}: contrast '{field}' entry {pair} must have 0 <= lo < hi (half-open)"
            )
        out.append((low, high))
    return tuple(out)


def _check_ascending(ranges: tuple[tuple[int, int], ...], label: str, field: str) -> None:
    """Refuse ranges that are out of order or overlap (touching ends are fine: half-open)."""
    for (first_low, first_high), (low, _high) in zip(ranges, ranges[1:], strict=False):
        if low < first_low:
            raise RegistryError(f"{label}: contrast '{field}' must be in ascending order")
        if low < first_high:
            raise RegistryError(f"{label}: contrast '{field}' must not overlap, got {list(ranges)}")


def _contrast(entry: dict[str, Any], label: str) -> ContrastBands | None:
    if "contrast" not in entry:
        return None
    raw = entry["contrast"]
    if not isinstance(raw, dict):
        raise RegistryError(
            f"{label}: 'contrast' must be a table with skyline_margin_rows, bands_rows and "
            "reference_rows"
        )
    for key in ("skyline_margin_rows", "bands_rows", "reference_rows"):
        if key not in raw:
            raise RegistryError(f"{label}: contrast is missing required field '{key}'")
    unknown = sorted(set(raw) - _CONTRAST_KEYS)
    if unknown:
        raise RegistryError(f"{label}: contrast has unknown key(s) {unknown}")

    margin = raw["skyline_margin_rows"]
    if not _whole(margin) or margin < 0:
        raise RegistryError(f"{label}: contrast 'skyline_margin_rows' must be a whole number >= 0")
    bands = _range_list(raw["bands_rows"], label, "bands_rows", many=True)
    (reference,) = _range_list(raw["reference_rows"], label, "reference_rows", many=False)
    domes = (
        _range_list(raw["light_dome_columns"], label, "light_dome_columns", many=True)
        if "light_dome_columns" in raw
        else ()
    )

    provisional = raw.get("light_dome_provisional", False)
    if not isinstance(provisional, bool):
        raise RegistryError(f"{label}: contrast 'light_dome_provisional' must be true or false")

    _check_ascending(bands, label, "bands_rows")
    _check_ascending(domes, label, "light_dome_columns")
    if bands[0][0] < margin:
        raise RegistryError(
            f"{label}: contrast bands start at row {bands[0][0]}, inside the "
            f"skyline_margin_rows of {margin}"
        )
    if reference[0] < margin:
        raise RegistryError(
            f"{label}: contrast reference_rows start at row {reference[0]}, inside the "
            f"skyline_margin_rows of {margin}"
        )
    for low, high in bands:
        if reference[0] < high and low < reference[1]:
            raise RegistryError(
                f"{label}: contrast reference_rows {list(reference)} overlap the band "
                f"[{low}, {high}]"
            )
    return ContrastBands(
        skyline_margin_rows=margin,
        bands_rows=bands,
        reference_rows=reference,
        light_dome_columns=domes,
        light_dome_provisional=provisional,
    )


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
        contrast=_contrast(entry, label),
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
