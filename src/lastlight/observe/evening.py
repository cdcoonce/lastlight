"""An evening's saved frames to the `observed.json` document (tier A, `docs/design.md` section 1).

`observe_evening` reads every frame the collector saved for one camera's evening, works out the
sun's elevation at each frame itself (from the UTC time in the frame's name, at the camera's
registry position, with `lastlight.geometry`; a sidecar is never consulted), keeps the frames
whose sun elevation is inside the window (default +6 to -6 degrees, civil twilight), and takes
each kept frame's contrast profile (`lastlight.observe.contrast`). Every frame it does not keep is
counted by reason, never silently skipped (`docs/design.md` section 2, rule 5).

The document is derived numbers only (`docs/design.md` section 3): a frame is identified by a
short prefix of its SHA-256 and the sun's elevation, never by its name, its time or its pixels.
It names the geometry it used (the SHA-256 of the frozen fit file and the fit's `provisional`
flag), so every observation says which fit it was read with.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lastlight.cameras.archive import EveningArchive
from lastlight.cameras.registry import Camera, ContrastBands
from lastlight.camgeom.fitfile import GEOMETRY_DIR, FitFileError, load_fit
from lastlight.camgeom.frames import FrameError, discover_frames, frame_sha256, load_rgb
from lastlight.geometry import solar_position
from lastlight.observe.bands import ContrastConfigError, validate_contrast
from lastlight.observe.contrast import (
    DARK_LEVEL,
    LIGHT_DOME_SUN_ELEVATION_DEG,
    MIN_VALID_PIXELS,
    SATURATION_LEVEL,
    ContrastProfile,
    ExcludedPixels,
    contrast_profile,
)

SCHEMA_VERSION: Final = 1
DEFAULT_MAX_SUN_ELEV: Final = 6.0
DEFAULT_MIN_SUN_ELEV: Final = -6.0
"""Civil twilight: tier A keeps frames with the sun between +6 and -6 degrees by default."""

SHA_PREFIX_CHARS: Final = 12

DROP_REASONS: Final = (
    "no_time_in_name",
    "unreadable",
    "wrong_size",
    "sun_above_window",
    "sun_below_window",
)
"""Why a frame may be left out. Every one is always in the document, zero or not."""

VIEW_ELEVATION_NOTE: Final = (
    "view_elevation_deg is nominal: it follows from the camera fit's provisional pixel scale, "
    "with a range from that fit's px_per_deg uncertainty. The observation itself is in pixel "
    "rows above the frozen skyline and does not depend on it."
)

_UNIT: Final = (
    "pixel rows above the frozen skyline, column by column; the band [lo, hi) is the rows lo to "
    "hi - 1 above it"
)


class ObserveError(ValueError):
    """The evening cannot be observed: bad input or configuration. The message says why."""


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    """One kept frame: a SHA-256 prefix, the sun's elevation, and its contrast profile."""

    sha256_prefix: str
    sun_elevation_deg: float
    profile: ContrastProfile

    def to_json(self) -> dict[str, Any]:
        return {
            "sha256_prefix": self.sha256_prefix,
            "sun_elevation_deg": round(self.sun_elevation_deg, 2),
            **self.profile.to_json(),
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """An evening's observation of one camera: everything `observed.json` holds."""

    camera_id: str
    date: dt.date
    fit_sha256: str
    provisional: bool
    contrast: ContrastBands
    min_sun_elev: float
    max_sun_elev: float
    frames_found: int
    frames: tuple[ObservedFrame, ...]
    dropped: dict[str, int]

    def _excluded_totals(self) -> dict[str, Any]:
        bands = [ExcludedPixels() for _ in self.contrast.bands_rows]
        reference = ExcludedPixels()
        for frame in self.frames:
            reference += frame.profile.reference.excluded
            for index, band in enumerate(frame.profile.bands):
                bands[index] += band.excluded
        return {
            "reference": {
                "rows": list(self.contrast.reference_rows),
                "n_excluded": reference.total,
                "excluded": reference.to_json(),
            },
            "bands": [
                {"rows": list(rows), "n_excluded": total.total, "excluded": total.to_json()}
                for rows, total in zip(self.contrast.bands_rows, bands, strict=True)
            ],
        }

    def to_json(self) -> dict[str, Any]:
        contrast = self.contrast
        return {
            "schema_version": SCHEMA_VERSION,
            "camera_id": self.camera_id,
            "date": self.date.isoformat(),
            "geometry": {"fit_sha256": self.fit_sha256, "provisional": self.provisional},
            "contrast_definition": {
                "unit": _UNIT,
                "skyline_margin_rows": contrast.skyline_margin_rows,
                "bands_rows": [list(rows) for rows in contrast.bands_rows],
                "reference_rows": list(contrast.reference_rows),
                "light_dome_columns": [list(columns) for columns in contrast.light_dome_columns],
                "light_dome_provisional": contrast.light_dome_provisional,
                "light_dome_below_sun_elevation_deg": LIGHT_DOME_SUN_ELEVATION_DEG,
                "saturated_at_or_above_code": SATURATION_LEVEL,
                "too_dark_at_or_below_code": DARK_LEVEL,
                "min_valid_pixels": MIN_VALID_PIXELS,
            },
            "view_elevation_note": VIEW_ELEVATION_NOTE,
            "sun_window_deg": {"max": self.max_sun_elev, "min": self.min_sun_elev},
            "frames_found": self.frames_found,
            "frames_used": len(self.frames),
            "frames_dropped": {reason: self.dropped[reason] for reason in DROP_REASONS},
            "frames": [frame.to_json() for frame in self.frames],
            "excluded_pixels": self._excluded_totals(),
        }


def observe_evening(
    camera: Camera,
    *,
    archive: Path,
    date: dt.date,
    geometry_dir: Path = GEOMETRY_DIR,
    max_sun_elev: float = DEFAULT_MAX_SUN_ELEV,
    min_sun_elev: float = DEFAULT_MIN_SUN_ELEV,
) -> Observation:
    """Observe every saved frame of `camera`'s evening on `date` from the private `archive`.

    Raises `ObserveError` for anything wrong with the input or the configuration: a camera with
    no contrast table or no frozen fit, a table that does not fit the fit, an empty window, or an
    evening folder that does not exist. A frame that cannot be used is not an error: it is counted.
    """
    if not (math.isfinite(min_sun_elev) and math.isfinite(max_sun_elev)):
        raise ObserveError("the sun-elevation window must be finite numbers")
    if min_sun_elev > max_sun_elev:
        raise ObserveError(
            f"the sun-elevation window is empty: the minimum {min_sun_elev:g} is above the "
            f"maximum {max_sun_elev:g}"
        )
    if camera.contrast is None:
        raise ObserveError(
            f"{camera.id} has no contrast table in the camera registry (it has no frozen "
            "geometry fit yet), so there is nothing to observe it with"
        )
    try:
        fit = load_fit(camera.id, geometry_dir)
        fit_sha256 = hashlib.sha256((geometry_dir / f"{camera.id}.json").read_bytes()).hexdigest()
    except (FitFileError, OSError) as error:
        raise ObserveError(f"cannot observe {camera.id}: {error}") from error
    try:
        validate_contrast(fit, camera.contrast)
    except ContrastConfigError as error:
        raise ObserveError(str(error)) from error

    directory = EveningArchive(archive, camera.id, date).directory
    if not directory.is_dir():
        raise ObserveError(f"no archive folder for {camera.id} on {date.isoformat()} in {archive}")

    dropped = dict.fromkeys(DROP_REASONS, 0)
    kept: list[ObservedFrame] = []
    found = discover_frames(directory)
    for frame in found:
        if frame.when_utc is None:
            dropped["no_time_in_name"] += 1
            continue
        sun = solar_position(frame.when_utc, camera.position.lat, camera.position.lon)
        if sun.elevation_deg > max_sun_elev:
            dropped["sun_above_window"] += 1
            continue
        if sun.elevation_deg < min_sun_elev:
            dropped["sun_below_window"] += 1
            continue
        try:
            rgb = load_rgb(frame.path)
            digest = frame_sha256(frame.path)
        except FrameError:
            dropped["unreadable"] += 1
            continue
        if rgb.shape[:2] != (fit.height, fit.width):
            dropped["wrong_size"] += 1
            continue
        profile = contrast_profile(rgb, fit, camera.contrast, sun_elevation_deg=sun.elevation_deg)
        kept.append(ObservedFrame(digest[:SHA_PREFIX_CHARS], sun.elevation_deg, profile))

    return Observation(
        camera_id=camera.id,
        date=date,
        fit_sha256=fit_sha256,
        provisional=fit.provisional,
        contrast=camera.contrast,
        min_sun_elev=min_sun_elev,
        max_sun_elev=max_sun_elev,
        frames_found=len(found),
        frames=tuple(kept),
        dropped=dropped,
    )


def write_observed(observation: Observation, path: Path) -> Path:
    """Write the document as indented JSON with a trailing newline, creating parent directories."""
    text = json.dumps(observation.to_json(), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
