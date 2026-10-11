"""The sun check: an independent test of a fit against the sun's own disk.

The skyline fit never sees the sun, so the sun can judge it. For a frame taken at a known
time, `lastlight.geometry` gives where the sun is in the sky; the camera model turns that into
a pixel; and if the sun's disk is visible there, the centroid of the saturated blob must land
within a stated angular tolerance of the prediction.

Most frames cannot say anything: the sun is often below the horizon, out of the frame, or
behind cloud, and an auto-exposed webcam shows a cloud-veiled sun as a soft glow with no disk.
Those frames are counted and named, never scored, and a check in which no frame shows a disk
is reported as skipped with the reason, not as passed.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from lastlight.camgeom.model import CameraModel
from lastlight.camgeom.skyline import Box
from lastlight.geometry import solar_position

SATURATION = 245  # a pixel is part of the disk when every channel is at least this bright
MIN_DISK_PIXELS = 4
MAX_DISK_PIXELS = 600  # larger than this is overexposed sky or glare, not a centroidable disk
SEARCH_RADIUS_DEG = 5.0


@dataclass(frozen=True, slots=True)
class SunPrediction:
    """Where the sun should appear in a frame, and where it is in the sky."""

    x: float
    y: float
    sun_azimuth_deg: float
    sun_elevation_deg: float


@dataclass(frozen=True, slots=True)
class SunDetection:
    """The centroid of a saturated blob and how many pixels it holds."""

    x: float
    y: float
    n_pixels: int


@dataclass(frozen=True, slots=True)
class SunSample:
    """What one frame said about the fit: a status, and the error when a disk was found.

    The status is one of `below_horizon`, `out_of_frame`, `no_disk`, `match` and `mismatch`.
    """

    sun_elevation_deg: float
    status: str
    error_deg: float | None = None
    error_px: float | None = None


@dataclass(frozen=True)
class SunCheck:
    """The outcome over all frames: passed, failed, or skipped with the reason why."""

    status: str
    reason: str
    tolerance_deg: float
    samples: tuple[SunSample, ...]

    @property
    def errors_deg(self) -> list[float]:
        return [s.error_deg for s in self.samples if s.error_deg is not None]

    @property
    def max_error_deg(self) -> float | None:
        return max(self.errors_deg, default=None)

    @property
    def median_error_deg(self) -> float | None:
        errors = self.errors_deg
        return float(np.median(errors)) if errors else None

    def to_json(self) -> dict[str, Any]:
        """Numbers and statuses only: no times, no pixels."""
        in_frame = {"no_disk", "match", "mismatch"}
        return {
            "status": self.status,
            "reason": self.reason,
            "tolerance_deg": self.tolerance_deg,
            "frames_examined": len(self.samples),
            "sun_in_frame": sum(s.status in in_frame for s in self.samples),
            "detected": len(self.errors_deg),
            "max_error_deg": _rounded(self.max_error_deg, 3),
            "median_error_deg": _rounded(self.median_error_deg, 3),
            "samples": [
                {
                    "sun_elevation_deg": round(s.sun_elevation_deg, 2),
                    "status": s.status,
                    "error_deg": _rounded(s.error_deg, 3),
                }
                for s in self.samples
            ],
        }


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def predict_sun_pixel(model: CameraModel, when_utc: dt.datetime) -> SunPrediction | None:
    """Pixel of the sun at `when_utc` for a camera at the model's position, or None if behind it."""
    sun = solar_position(when_utc, model.lat, model.lon)
    x, y = model.sky_to_pixel(sun.azimuth_deg, sun.elevation_deg)
    if math.isnan(x) or math.isnan(y):
        return None
    return SunPrediction(x, y, sun.azimuth_deg, sun.elevation_deg)


def locate_sun_disk(
    image: np.ndarray,
    near: tuple[float, float],
    radius_px: float,
    *,
    exclude: Sequence[Box] = (),
) -> SunDetection | None:
    """The saturated blob nearest `near` within `radius_px`, or None if there is no disk.

    A disk is a connected run of pixels with every channel at `SATURATION` or more, between
    `MIN_DISK_PIXELS` and `MAX_DISK_PIXELS` large. Smaller is noise; larger is overexposed sky.
    Pixels in `exclude` boxes (the timestamp's white text) are ignored.
    """
    pixels = np.asarray(image)
    brightness = pixels.min(axis=2) if pixels.ndim == 3 else pixels
    bright = brightness >= SATURATION
    for box in exclude:
        bright[box.y0 : box.y1, box.x0 : box.x1] = False
    rows, cols = np.indices(bright.shape)
    bright &= np.hypot(cols - near[0], rows - near[1]) <= radius_px
    labels, count = ndimage.label(bright, structure=np.ones((3, 3)))
    best: SunDetection | None = None
    best_distance = math.inf
    for label in range(1, count + 1):
        member_rows, member_cols = np.nonzero(labels == label)
        if not MIN_DISK_PIXELS <= member_rows.size <= MAX_DISK_PIXELS:
            continue
        x, y = float(member_cols.mean()), float(member_rows.mean())
        distance = math.hypot(x - near[0], y - near[1])
        if distance < best_distance:
            best, best_distance = SunDetection(x, y, int(member_rows.size)), distance
    return best


def _separation_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    """Angle between two sky directions."""
    a1, e1, a2, e2 = map(math.radians, (az1, el1, az2, el2))
    cosine = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _sample_one(
    model: CameraModel,
    image: np.ndarray,
    when: dt.datetime,
    tolerance_deg: float,
    search_radius_deg: float,
    exclude: Sequence[Box],
) -> SunSample:
    sun = solar_position(when, model.lat, model.lon)
    if sun.elevation_deg < 0.0:
        return SunSample(sun.elevation_deg, "below_horizon")
    prediction = predict_sun_pixel(model, when)
    if (
        prediction is None
        or not 0.0 <= prediction.x < model.width
        or not 0.0 <= prediction.y < model.height
    ):
        return SunSample(sun.elevation_deg, "out_of_frame")
    pixels_per_degree = model.focal_px * math.pi / 180.0
    found = locate_sun_disk(
        image,
        (prediction.x, prediction.y),
        search_radius_deg * pixels_per_degree,
        exclude=exclude,
    )
    if found is None:
        return SunSample(sun.elevation_deg, "no_disk")
    az, el = model.pixel_to_sky(found.x, found.y)
    error_deg = _separation_deg(az, el, sun.azimuth_deg, sun.elevation_deg)
    error_px = math.hypot(found.x - prediction.x, found.y - prediction.y)
    status = "match" if error_deg <= tolerance_deg else "mismatch"
    return SunSample(sun.elevation_deg, status, error_deg, error_px)


def run_sun_check(
    model: CameraModel,
    frames: Sequence[tuple[np.ndarray, dt.datetime]],
    *,
    tolerance_deg: float = 1.0,
    search_radius_deg: float = SEARCH_RADIUS_DEG,
    exclude: Sequence[Box] = (),
) -> SunCheck:
    """Test the model's prediction of the sun against every frame that shows the sun's disk."""
    samples = tuple(
        _sample_one(model, image, when, tolerance_deg, search_radius_deg, exclude)
        for image, when in frames
    )
    counts = {
        status: sum(s.status == status for s in samples)
        for status in ("below_horizon", "out_of_frame", "no_disk", "match", "mismatch")
    }
    detected = counts["match"] + counts["mismatch"]
    tolerance = f"{tolerance_deg:g} degrees"
    if counts["mismatch"]:
        worst = max(s.error_deg for s in samples if s.error_deg is not None)
        reason = (
            f"{counts['mismatch']} of {detected} observed sun disks lay more than {tolerance} from "
            f"the predicted pixel (largest error {worst:.2f} degrees)"
        )
        return SunCheck("failed", reason, tolerance_deg, samples)
    if counts["match"]:
        worst = max(s.error_deg for s in samples if s.error_deg is not None)
        reason = (
            f"{counts['match']} of {len(samples)} frames showed the sun's disk within {tolerance} "
            f"of the predicted pixel (largest error {worst:.2f} degrees)"
        )
        return SunCheck("passed", reason, tolerance_deg, samples)
    if not samples:
        return SunCheck("skipped", "no frames were given", tolerance_deg, samples)
    parts = []
    if counts["below_horizon"]:
        parts.append(f"the sun was below the horizon in {counts['below_horizon']}")
    if counts["out_of_frame"]:
        parts.append(f"the sun was out of frame in {counts['out_of_frame']}")
    if counts["no_disk"]:
        parts.append(
            f"no saturated disk was found near the predicted pixel in {counts['no_disk']} "
            "(hidden by cloud, or too dim to centroid)"
        )
    reason = f"in {len(samples)} frames, " + "; ".join(parts)
    return SunCheck("skipped", reason, tolerance_deg, samples)
