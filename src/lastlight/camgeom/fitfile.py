"""The frozen fit of one camera: a small JSON file of orientation, residuals and provenance.

`src/lastlight/cameras/geometry/<ID>.json` is committed once a fit is frozen (`docs/design.md`
section 2, rule 2) and is what every later slice reads. It holds:

- the orientation: azimuth, elevation, roll and horizontal field of view, in degrees;
- the skyline residual of the fit, in pixels and in degrees, and how many columns it spans;
- the predicted skyline in the frame (the fitted curve the sky mask is cut from);
- the frames the fit was frozen on, each by the date of its evening, the sun's elevation at the
  frame and the SHA-256 of its bytes: never a file name, a time of day or a pixel;
- the sun check, the radial-distortion check and (when it was run) the position search, and
  the date the fit was frozen;
- whether the fit is `provisional`, and how far the pointing and pixel scale could be off.

A fit is provisional unless the sun check passed. The skyline alone fixes azimuth, field of view
and the camera's position only together, so until the sun (an independent judge) has confirmed
the pixel scale, every number in the file is the best of a family of nearly equal fits.
`uncertainty` gives that family's range, and `azimuth_minus_stated_deg` how far the fit is from
the direction the camera's operator states. A consumer that needs the pixel scale to be right
(any conversion of a pixel offset to an angle) should refuse a provisional fit unless the caller
asks for it explicitly.

It does not hold the camera's coordinates. The pixel-to-sky mapping needs only the
orientation, and coordinates stay out of the repository (see `lastlight.camgeom.sites`);
`CameraFit.camera_model` takes a position from the caller when a `CameraModel` is wanted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lastlight.camgeom.model import CameraModel, pixels_per_degree
from lastlight.camgeom.orientation import Orientation
from lastlight.camgeom.skyline import Box

SCHEMA_VERSION = 2
GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "cameras" / "geometry"


class FitFileError(ValueError):
    """A fit file is missing or does not follow the schema."""


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One frame a fit was frozen on, identified without naming it or timing it."""

    evening: str
    sun_elevation_deg: float
    sha256: str


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """The range of azimuth, field of view and pixel scale across nearly equal fits.

    Each range is `(low, high)` over the positions whose skyline residual was within 10 percent
    of the best one, so it is where the fit could just as well have landed, not a confidence
    interval. When `lower_bounds` is true the best position sat at the edge of the searched box
    or the residual kept falling beyond it, and the true ranges are wider.
    """

    azimuth_deg: tuple[float, float]
    hfov_deg: tuple[float, float]
    px_per_deg: tuple[float, float]
    basis: str
    lower_bounds: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "azimuth_deg": [round(v, 2) for v in self.azimuth_deg],
            "hfov_deg": [round(v, 2) for v in self.hfov_deg],
            "px_per_deg": [round(v, 2) for v in self.px_per_deg],
            "basis": self.basis,
            "lower_bounds": self.lower_bounds,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Uncertainty:
        ranges = {}
        for key, positive in (("azimuth_deg", False), ("hfov_deg", True), ("px_per_deg", True)):
            pair = data[key]
            if not isinstance(pair, list) or len(pair) != 2:
                raise FitFileError(f"uncertainty.{key} must be a [low, high] pair")
            low, high = (_number(v, f"uncertainty.{key}") for v in pair)
            if low > high or (positive and low <= 0.0):
                raise FitFileError(f"uncertainty.{key} must run from a low to a higher number")
            ranges[key] = (low, high)
        if not isinstance(data["lower_bounds"], bool):
            raise FitFileError("uncertainty.lower_bounds must be true or false")
        return cls(
            azimuth_deg=ranges["azimuth_deg"],
            hfov_deg=ranges["hfov_deg"],
            px_per_deg=ranges["px_per_deg"],
            basis=str(data["basis"]),
            lower_bounds=data["lower_bounds"],
        )


def uncertainty_from_ranges(
    azimuth_deg: tuple[float, float],
    hfov_deg: tuple[float, float],
    width: int,
    basis: str,
    lower_bounds: bool,
) -> Uncertainty:
    """An `Uncertainty` whose pixel scale follows from the field-of-view range."""
    scale = sorted(pixels_per_degree(h, width) for h in hfov_deg)
    return Uncertainty(azimuth_deg, hfov_deg, (scale[0], scale[1]), basis, lower_bounds)


def azimuth_difference_deg(fitted: float, stated: float) -> float:
    """Fitted minus stated azimuth, wrapped to the range -180 to 180 degrees."""
    return (fitted - stated + 180.0) % 360.0 - 180.0


def _number(value: Any, name: str) -> float:
    """A finite float, or a `FitFileError` naming the field (booleans and text are not numbers)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise FitFileError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _count(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FitFileError(f"{name} must be a whole number of at least {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class CameraFit:
    """A frozen fit. `provisional` and `uncertainty` say how far to trust its pixel scale.

    The orientation is the best fit of a family of nearly equal ones. While `provisional` is true
    (the sun check has not passed) nothing independent has confirmed the field of view, so
    `camera_model()` and `sky_mask()` give a mapping that may be off by the ranges in
    `uncertainty`: the sky mask is pinned in pixel space, but a margin or an elevation band
    converted to rows or pixels carries the pixel-scale uncertainty.
    """

    camera_id: str
    width: int
    height: int
    orientation: Orientation
    skyline_rms_px: float
    skyline_rms_deg: float
    skyline_median_abs_px: float
    n_columns: int
    skyline_x_px: tuple[float, ...]
    skyline_y_px: tuple[float | None, ...]
    overlay_boxes: tuple[Box, ...]
    frames_used: tuple[FrameRecord, ...]
    sun_check: dict[str, Any]
    radial_term: dict[str, Any] | None
    position_search: dict[str, Any] | None
    frozen_on: str
    position_basis: str
    provisional: bool
    uncertainty: Uncertainty | None
    stated_azimuth_deg: float | None
    azimuth_minus_stated_deg: float | None

    def camera_model(self, lat: float, lon: float, elev_m: float) -> CameraModel:
        """A `CameraModel` with this orientation, standing where the caller says.

        The mapping from pixels to the sky is provisional when `self.provisional` is true: its
        pixel scale is the best of a family of fits, known only to the ranges in `uncertainty`.
        """
        o = self.orientation
        return CameraModel(
            camera_id=self.camera_id,
            lat=lat,
            lon=lon,
            elev_m=elev_m,
            azimuth_deg=o.azimuth_deg,
            elevation_deg=o.elevation_deg,
            roll_deg=o.roll_deg,
            hfov_deg=o.hfov_deg,
            width=self.width,
            height=self.height,
        )

    def skyline_y(self, columns: np.ndarray) -> np.ndarray:
        """Row of the fitted skyline at each column; NaN beside a knot where it was not seen."""
        knots_x = np.asarray(self.skyline_x_px, dtype=float)
        knots_y = np.array([np.nan if y is None else y for y in self.skyline_y_px], dtype=float)
        columns = np.asarray(columns, dtype=float)
        right = np.clip(np.searchsorted(knots_x, columns, side="left"), 1, knots_x.size - 1)
        left = right - 1
        span = knots_x[right] - knots_x[left]
        weight = np.clip((columns - knots_x[left]) / span, 0.0, 1.0)
        at_left = columns <= knots_x[left]
        with np.errstate(invalid="ignore"):
            value = knots_y[left] * (1.0 - weight) + knots_y[right] * weight
        value = np.where(at_left, knots_y[left], value)
        value = np.where(weight >= 1.0, knots_y[right], value)
        return value

    def to_json(self) -> dict[str, Any]:
        o = self.orientation
        return {
            "schema_version": SCHEMA_VERSION,
            "camera_id": self.camera_id,
            "image": {"width": self.width, "height": self.height},
            "orientation": {
                "azimuth_deg": round(o.azimuth_deg, 4),
                "elevation_deg": round(o.elevation_deg, 4),
                "roll_deg": round(o.roll_deg, 4),
                "hfov_deg": round(o.hfov_deg, 4),
            },
            "provisional": self.provisional,
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_json(),
            "stated_azimuth_deg": self.stated_azimuth_deg,
            "azimuth_minus_stated_deg": (
                None
                if self.azimuth_minus_stated_deg is None
                else round(self.azimuth_minus_stated_deg, 2)
            ),
            "skyline_fit": {
                "rms_px": round(self.skyline_rms_px, 3),
                "rms_deg": round(self.skyline_rms_deg, 4),
                "median_abs_px": round(self.skyline_median_abs_px, 3),
                "n_columns": self.n_columns,
            },
            "skyline": {
                "x_px": [round(x, 1) for x in self.skyline_x_px],
                "y_px": [None if y is None else round(y, 1) for y in self.skyline_y_px],
            },
            "overlay_boxes": [[b.x0, b.y0, b.x1, b.y1] for b in self.overlay_boxes],
            "frames_used": [
                {
                    "evening": f.evening,
                    "sun_elevation_deg": round(f.sun_elevation_deg, 2),
                    "sha256": f.sha256,
                }
                for f in self.frames_used
            ],
            "sun_check": self.sun_check,
            "radial_term": self.radial_term,
            "position_search": self.position_search,
            "fit_frozen_on": self.frozen_on,
            "position_basis": self.position_basis,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CameraFit:
        """Read a fit, refusing anything `CameraModel` or the mask could not use safely."""
        try:
            return cls._from_json(data)
        except FitFileError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise FitFileError(f"the fit file does not follow the schema: {error!r}") from error

    @classmethod
    def _from_json(cls, data: dict[str, Any]) -> CameraFit:
        if data["schema_version"] != SCHEMA_VERSION:
            found = data["schema_version"]
            raise FitFileError(f"unsupported schema_version {found!r}, expected {SCHEMA_VERSION}")
        width = _count(data["image"]["width"], "image.width", minimum=1)
        height = _count(data["image"]["height"], "image.height", minimum=1)
        o = data["orientation"]
        orientation = Orientation(
            _number(o["azimuth_deg"], "orientation.azimuth_deg"),
            _number(o["elevation_deg"], "orientation.elevation_deg"),
            _number(o["roll_deg"], "orientation.roll_deg"),
            _number(o["hfov_deg"], "orientation.hfov_deg"),
        )
        if not 0.0 < orientation.hfov_deg < 180.0:
            raise FitFileError(
                "orientation.hfov_deg must be between 0 and 180 degrees, "
                f"got {orientation.hfov_deg!r}"
            )
        if not -90.0 <= orientation.elevation_deg <= 90.0:
            raise FitFileError("orientation.elevation_deg must be within +-90 degrees")

        x_px = tuple(_number(x, "skyline.x_px") for x in data["skyline"]["x_px"])
        y_px = tuple(
            None if y is None else _number(y, "skyline.y_px") for y in data["skyline"]["y_px"]
        )
        if len(x_px) != len(y_px) or len(x_px) < 2:
            raise FitFileError("skyline needs the same number (at least two) of x_px and y_px")
        if any(b <= a for a, b in zip(x_px, x_px[1:], strict=False)):
            raise FitFileError("skyline x_px must be strictly increasing (no repeated column)")

        fit = data["skyline_fit"]
        sun_check = dict(data["sun_check"])
        provisional = data["provisional"]
        if not isinstance(provisional, bool):
            raise FitFileError("provisional must be true or false")
        if provisional != (sun_check.get("status") != "passed"):
            raise FitFileError(
                "provisional must be true exactly when the sun check has not passed "
                f"(sun_check.status is {sun_check.get('status')!r}, provisional is {provisional!r})"
            )
        stated = data["stated_azimuth_deg"]
        difference = data["azimuth_minus_stated_deg"]
        if (stated is None) != (difference is None):
            raise FitFileError("stated_azimuth_deg and azimuth_minus_stated_deg go together")
        if stated is not None:
            stated = _number(stated, "stated_azimuth_deg")
            difference = _number(difference, "azimuth_minus_stated_deg")
            expected = azimuth_difference_deg(orientation.azimuth_deg, stated)
            if abs(difference - expected) > 0.02:
                raise FitFileError(
                    f"azimuth_minus_stated_deg is {difference!r} but the orientation and the "
                    f"stated azimuth differ by {expected:.2f}"
                )
        raw_uncertainty = data["uncertainty"]
        return cls(
            camera_id=str(data["camera_id"]),
            width=width,
            height=height,
            orientation=orientation,
            skyline_rms_px=_number(fit["rms_px"], "skyline_fit.rms_px"),
            skyline_rms_deg=_number(fit["rms_deg"], "skyline_fit.rms_deg"),
            skyline_median_abs_px=_number(fit["median_abs_px"], "skyline_fit.median_abs_px"),
            n_columns=_count(fit["n_columns"], "skyline_fit.n_columns", minimum=0),
            skyline_x_px=x_px,
            skyline_y_px=y_px,
            overlay_boxes=tuple(Box(*box) for box in data["overlay_boxes"]),
            frames_used=tuple(
                FrameRecord(
                    str(f["evening"]),
                    _number(f["sun_elevation_deg"], "frames_used.sun_elevation_deg"),
                    str(f["sha256"]),
                )
                for f in data["frames_used"]
            ),
            sun_check=sun_check,
            radial_term=None if data["radial_term"] is None else dict(data["radial_term"]),
            position_search=(
                None if data["position_search"] is None else dict(data["position_search"])
            ),
            frozen_on=str(data["fit_frozen_on"]),
            position_basis=str(data["position_basis"]),
            provisional=provisional,
            uncertainty=None if raw_uncertainty is None else Uncertainty.from_json(raw_uncertainty),
            stated_azimuth_deg=stated,
            azimuth_minus_stated_deg=difference,
        )


def write_fit(fit: CameraFit, path: Path) -> Path:
    """Write a fit as indented JSON with a trailing newline, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fit.to_json(), indent=2) + "\n")
    return path


def load_fit(camera_id: str, directory: Path = GEOMETRY_DIR) -> CameraFit:
    """The frozen fit of a camera from `<directory>/<camera_id>.json`."""
    fit = load_fit_file(
        directory / f"{camera_id}.json",
        missing=f"no geometry fit for camera {camera_id} in {directory}",
    )
    if fit.camera_id != camera_id:
        raise FitFileError(f"camera_id {fit.camera_id!r} in {camera_id}.json is not {camera_id!r}")
    return fit


def load_fit_file(path: Path, *, missing: str | None = None) -> CameraFit:
    """The fit stored at `path`; every way it can be unusable is a `FitFileError`."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FitFileError(missing or f"fit file not found: {path}") from error
    except OSError as error:
        raise FitFileError(f"fit file {path.name} cannot be read: {error}") from error
    except json.JSONDecodeError as error:
        raise FitFileError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise FitFileError(f"{path.name} does not hold a fit (expected a JSON object)")
    return CameraFit.from_json(data)
