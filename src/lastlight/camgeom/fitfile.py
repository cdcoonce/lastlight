"""The frozen fit of one camera: a small JSON file of orientation, residuals and provenance.

`src/lastlight/cameras/geometry/<ID>.json` is committed once a fit is frozen (`docs/design.md`
section 2, rule 2) and is what every later slice reads. It holds:

- the orientation: azimuth, elevation, roll and horizontal field of view, in degrees;
- the skyline residual of the fit, in pixels and in degrees, and how many columns it spans;
- the predicted skyline in the frame (the fitted curve the sky mask is cut from);
- the frames the fit was frozen on, each by the date of its evening, the sun's elevation at the
  frame and the SHA-256 of its bytes: never a file name, a time of day or a pixel;
- the sun check, the radial-distortion check and (when it was run) the position search, and
  the date the fit was frozen.

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

from lastlight.camgeom.model import CameraModel
from lastlight.camgeom.orientation import Orientation
from lastlight.camgeom.skyline import Box

SCHEMA_VERSION = 1
GEOMETRY_DIR = Path(__file__).resolve().parent.parent / "cameras" / "geometry"


class FitFileError(ValueError):
    """A fit file is missing or does not follow the schema."""


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One frame a fit was frozen on, identified without naming it or timing it."""

    evening: str
    sun_elevation_deg: float
    sha256: str


@dataclass(frozen=True)
class CameraFit:
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

    def camera_model(self, lat: float, lon: float, elev_m: float) -> CameraModel:
        """A `CameraModel` with this orientation, standing where the caller says."""
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
        try:
            if data["schema_version"] != SCHEMA_VERSION:
                found = data["schema_version"]
                raise FitFileError(
                    f"unsupported schema_version {found!r}, expected {SCHEMA_VERSION}"
                )
            x_px = tuple(float(x) for x in data["skyline"]["x_px"])
            y_px = tuple(None if y is None else float(y) for y in data["skyline"]["y_px"])
            if len(x_px) != len(y_px) or len(x_px) < 2:
                raise FitFileError("skyline needs the same number (at least two) of x_px and y_px")
            o, fit = data["orientation"], data["skyline_fit"]
            return cls(
                camera_id=str(data["camera_id"]),
                width=int(data["image"]["width"]),
                height=int(data["image"]["height"]),
                orientation=Orientation(
                    float(o["azimuth_deg"]),
                    float(o["elevation_deg"]),
                    float(o["roll_deg"]),
                    float(o["hfov_deg"]),
                ),
                skyline_rms_px=float(fit["rms_px"]),
                skyline_rms_deg=float(fit["rms_deg"]),
                skyline_median_abs_px=float(fit["median_abs_px"]),
                n_columns=int(fit["n_columns"]),
                skyline_x_px=x_px,
                skyline_y_px=y_px,
                overlay_boxes=tuple(Box(*box) for box in data["overlay_boxes"]),
                frames_used=tuple(
                    FrameRecord(f["evening"], float(f["sun_elevation_deg"]), f["sha256"])
                    for f in data["frames_used"]
                ),
                sun_check=dict(data["sun_check"]),
                radial_term=None if data["radial_term"] is None else dict(data["radial_term"]),
                position_search=(
                    None if data["position_search"] is None else dict(data["position_search"])
                ),
                frozen_on=str(data["fit_frozen_on"]),
                position_basis=str(data["position_basis"]),
            )
        except FitFileError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise FitFileError(f"the fit file does not follow the schema: {error!r}") from error


def write_fit(fit: CameraFit, path: Path) -> Path:
    """Write a fit as indented JSON with a trailing newline, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fit.to_json(), indent=2) + "\n")
    return path


def load_fit(camera_id: str, directory: Path = GEOMETRY_DIR) -> CameraFit:
    """The frozen fit of a camera from `<directory>/<camera_id>.json`."""
    path = directory / f"{camera_id}.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FitFileError(f"no geometry fit for camera {camera_id} in {directory}") from error
    except json.JSONDecodeError as error:
        raise FitFileError(f"{path.name} is not valid JSON: {error}") from error
    fit = CameraFit.from_json(data)
    if fit.camera_id != camera_id:
        raise FitFileError(f"camera_id {fit.camera_id!r} in {path.name} is not {camera_id!r}")
    for number in (fit.skyline_rms_px, fit.orientation.hfov_deg):
        if not math.isfinite(number):
            raise FitFileError(f"{path.name} holds a number that is not finite")
    return fit
