"""The pinhole camera model: pixel to sky direction, and sky direction to pixel.

Conventions, all in degrees:

- Azimuth runs clockwise from true north (90 is east), as in `lastlight.geometry`.
  Elevation is above the local horizontal.
- `azimuth_deg` and `elevation_deg` give the direction of the optical axis. `roll_deg` is a
  clockwise tilt of the camera about that axis: a positive roll makes the horizon run up
  toward the right of the image. `hfov_deg` is the horizontal field of view between the left
  and right edges of the image.
- Pixel coordinates have x to the right and y down, with the centre of the top-left pixel at
  (0, 0), so the image spans -0.5 to width - 0.5 and the principal point is the centre,
  ((width - 1) / 2, (height - 1) / 2).

The projection is rectilinear (a straight pinhole). The fit may measure a radial distortion
term (see `lastlight.camgeom.orientation.fit_radial_term`), but the model does not carry one:
the term is added only if the residuals justify it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _focal_px(hfov_deg: float, width: int) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _axes(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, right and up unit vectors of the camera, as (east, north, up) components."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    forward = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    right = np.array([math.cos(az), -math.sin(az), 0.0])
    return forward, right, np.cross(right, forward)


def _is_scalar(*values: object) -> bool:
    return all(np.ndim(value) == 0 for value in values)


def project_sky_to_pixel(
    azimuth_deg,
    elevation_deg,
    *,
    camera_azimuth_deg: float,
    camera_elevation_deg: float,
    roll_deg: float,
    hfov_deg: float,
    width: int,
    height: int,
    radial_k1: float = 0.0,
):
    """Pixel (x, y) of sky directions; NaN for a direction at or behind the image plane.

    `radial_k1` bends the image the way a lens would: a point at normalised radius r from the
    centre lands at r * (1 + k1 * r^2), with r in units of the focal length. The model proper
    uses 0; the fit uses a non-zero value only to measure whether a distortion term pays off.
    """
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    el = np.radians(np.asarray(elevation_deg, dtype=float))
    forward, right, up = _axes(camera_azimuth_deg, camera_elevation_deg)
    east, north, vertical = np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)
    ahead = east * forward[0] + north * forward[1] + vertical * forward[2]
    side = east * right[0] + north * right[1] + vertical * right[2]
    down = -(east * up[0] + north * up[1] + vertical * up[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        in_front = ahead > 1e-12
        u_rolled = np.where(in_front, side / ahead, np.nan)
        v_rolled = np.where(in_front, down / ahead, np.nan)
    roll = math.radians(roll_deg)
    u = u_rolled * math.cos(roll) + v_rolled * math.sin(roll)
    v = -u_rolled * math.sin(roll) + v_rolled * math.cos(roll)
    if radial_k1:
        scale = 1.0 + radial_k1 * (u * u + v * v)
        u, v = u * scale, v * scale
    focal = _focal_px(hfov_deg, width)
    x = (width - 1) / 2.0 + focal * u
    y = (height - 1) / 2.0 + focal * v
    return x, y


def unproject_pixel_to_sky(
    x,
    y,
    *,
    camera_azimuth_deg: float,
    camera_elevation_deg: float,
    roll_deg: float,
    hfov_deg: float,
    width: int,
    height: int,
):
    """(azimuth, elevation) in degrees of the direction each pixel looks at."""
    focal = _focal_px(hfov_deg, width)
    u = (np.asarray(x, dtype=float) - (width - 1) / 2.0) / focal
    v = (np.asarray(y, dtype=float) - (height - 1) / 2.0) / focal
    roll = math.radians(roll_deg)
    u_rolled = u * math.cos(roll) - v * math.sin(roll)
    v_rolled = u * math.sin(roll) + v * math.cos(roll)
    forward, right, up = _axes(camera_azimuth_deg, camera_elevation_deg)
    ray = (
        forward[:, None] + right[:, None] * np.ravel(u_rolled) - up[:, None] * np.ravel(v_rolled)
    ).reshape((3, *np.shape(u_rolled)))
    east, north, vertical = ray
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    azimuth = np.where(azimuth >= 360.0, 0.0, azimuth)
    elevation = np.degrees(np.arctan2(vertical, np.hypot(east, north)))
    return azimuth, elevation


@dataclass(frozen=True, slots=True)
class CameraModel:
    """A fixed camera: where it stands, where it points, and how wide it sees."""

    camera_id: str
    lat: float
    lon: float
    elev_m: float
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    hfov_deg: float
    width: int
    height: int

    def __post_init__(self) -> None:
        numbers = {
            "lat": self.lat,
            "lon": self.lon,
            "elev_m": self.elev_m,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "roll_deg": self.roll_deg,
            "hfov_deg": self.hfov_deg,
        }
        for name, value in numbers.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError(f"hfov_deg must be between 0 and 180 degrees, got {self.hfov_deg!r}")
        if not -90.0 <= self.elevation_deg <= 90.0:
            raise ValueError(
                f"elevation_deg must be within +-90 degrees, got {self.elevation_deg!r}"
            )
        for name, size in (("width", self.width), ("height", self.height)):
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"{name} must be a positive number of pixels, got {size!r}")

    @property
    def focal_px(self) -> float:
        """Focal length in pixels: the distance from the pinhole to the image plane."""
        return _focal_px(self.hfov_deg, self.width)

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view between the top and bottom edges of the image."""
        half = math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) * self.height / self.width)
        return 2.0 * math.degrees(half)

    def pixel_to_sky(self, x, y):
        """(azimuth, elevation) in degrees looked at by pixel (x, y); arrays are accepted."""
        az, el = unproject_pixel_to_sky(
            x,
            y,
            camera_azimuth_deg=self.azimuth_deg,
            camera_elevation_deg=self.elevation_deg,
            roll_deg=self.roll_deg,
            hfov_deg=self.hfov_deg,
            width=self.width,
            height=self.height,
        )
        return (float(az), float(el)) if _is_scalar(x, y) else (az, el)

    def sky_to_pixel(self, azimuth_deg, elevation_deg):
        """Pixel (x, y) of a sky direction, NaN if it lies behind the camera."""
        x, y = project_sky_to_pixel(
            azimuth_deg,
            elevation_deg,
            camera_azimuth_deg=self.azimuth_deg,
            camera_elevation_deg=self.elevation_deg,
            roll_deg=self.roll_deg,
            hfov_deg=self.hfov_deg,
            width=self.width,
            height=self.height,
        )
        return (float(x), float(y)) if _is_scalar(azimuth_deg, elevation_deg) else (x, y)
