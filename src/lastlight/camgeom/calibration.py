"""The calibration pipeline behind `lastlight camgeom fit`: frames and terrain in, a fit out.

Steps: choose the frames (timed, with the sun high enough that the sky is brighter than the
terrain); compute the horizon the DEM predicts from the camera's position; fit the orientation
by matching skylines; measure whether a radial distortion term would explain the residual;
check the fit against the sun's disk; and assemble the record that gets frozen.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lastlight.camgeom.dem import Dem, horizon_profile
from lastlight.camgeom.fitfile import CameraFit, FrameRecord
from lastlight.camgeom.frames import discover_frames, evening_of, frame_sha256, load_rgb
from lastlight.camgeom.model import CameraModel
from lastlight.camgeom.orientation import (
    RADIAL_K1_BOUND,
    HorizonProfile,
    Orientation,
    Prior,
    RadialTerm,
    SkylineFit,
    fit_orientation,
    fit_radial_term,
)
from lastlight.camgeom.sites import CameraSite
from lastlight.camgeom.skyline import default_overlay_boxes
from lastlight.camgeom.suncheck import SunCheck, run_sun_check
from lastlight.geometry import solar_position

VIEW_HALF_SPAN_DEG = 80.0  # horizon computed this far either side of the stated view direction
AZIMUTH_STEP_DEG = 0.05
SKYLINE_KNOT_STEP_PX = 10
KM_PER_DEGREE_LAT = 111.2  # a flat-Earth offset is exact enough for a scan of a few kilometres
POSITION_BASIS = (
    "read from the camera site file at fit time; coordinates are kept out of the repository "
    "(docs/design.md section 3)"
)


class CalibrationError(ValueError):
    """The inputs cannot support a fit; the message says what is missing."""


@dataclass(frozen=True)
class Calibration:
    """A finished fit and the bookkeeping the command reports."""

    fit: CameraFit
    skyline: SkylineFit
    radial: RadialTerm
    sun_check: SunCheck
    frames_used: int
    frames_too_dark: int
    frames_without_time: int


@dataclass(frozen=True, slots=True)
class PositionTrial:
    """The best fit found with the camera moved `east_km` and `north_km` from the site."""

    east_km: float
    north_km: float
    skyline_rms_px: float
    orientation: Orientation


def view_azimuths(view_azimuth_deg: float) -> np.ndarray:
    """The azimuths the horizon is computed at: the stated view direction and a wide margin."""
    return np.arange(
        view_azimuth_deg - VIEW_HALF_SPAN_DEG,
        view_azimuth_deg + VIEW_HALF_SPAN_DEG + AZIMUTH_STEP_DEG / 2,
        AZIMUTH_STEP_DEG,
    )


def _knot_columns(width: int) -> list[int]:
    columns = list(range(0, width, SKYLINE_KNOT_STEP_PX))
    if columns[-1] != width - 1:
        columns.append(width - 1)
    return columns


def scan_offsets(box_km: float, step_km: float) -> list[tuple[float, float]]:
    """(east, north) offsets in km: every multiple of the step inside the box, centre included."""
    if box_km < 0 or step_km <= 0:
        raise ValueError("the scan box must not be negative and the step must be positive")
    count = math.floor(box_km / step_km + 1e-9)
    steps = range(-count, count + 1)
    return [(east * step_km, north * step_km) for north in steps for east in steps]


def summarize_scan(
    trials: Sequence[PositionTrial], box_km: float, step_km: float, site_rms_px: float
) -> dict:
    """The spread of residual and orientation over the scanned positions, for the fit file.

    Only statistics are kept, never the positions: the scan says whether the skyline can place
    the camera inside the described area, and a flat residual with an orientation that trades
    off against position says it cannot.
    """
    if not trials:
        raise ValueError("no position in the scan gave a usable fit")
    rms = [trial.skyline_rms_px for trial in trials]
    azimuth = [trial.orientation.azimuth_deg for trial in trials]
    hfov = [trial.orientation.hfov_deg for trial in trials]
    return {
        "box_km": box_km,
        "step_km": step_km,
        "positions": len(trials),
        "rms_px_at_site": round(site_rms_px, 3),
        "rms_px_min": round(min(rms), 3),
        "rms_px_max": round(max(rms), 3),
        "azimuth_deg_range": [round(min(azimuth), 2), round(max(azimuth), 2)],
        "hfov_deg_range": [round(min(hfov), 2), round(max(hfov), 2)],
        "adopted": False,
        "reason": (
            "position held at the described site: the scan is a diagnostic, and a position "
            "would be adopted only if one candidate beat the rest clearly while the "
            "orientation stayed put"
        ),
    }


def _fit_at(
    images: list[np.ndarray],
    dem: Dem,
    site: CameraSite,
    east_km: float,
    north_km: float,
    max_range_km: float,
) -> tuple[SkylineFit, HorizonProfile, float] | None:
    """Fit with the camera moved from the site; None where the DEM cannot support it."""
    lat = site.lat + north_km / KM_PER_DEGREE_LAT
    lon = site.lon + east_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(site.lat)))
    ground = dem.sample(lat, lon)
    if np.isnan(ground):
        return None
    elev_m = float(ground) + site.mount_height_m
    azimuths = view_azimuths(site.view_azimuth_deg)
    elevations = horizon_profile(dem, lat, lon, elev_m, azimuths, max_range_km=max_range_km)
    if np.isfinite(elevations).mean() < 0.5:
        return None
    horizon = HorizonProfile.from_samples(azimuths, elevations)
    height, width = images[0].shape[:2]
    skyline = fit_orientation(
        images,
        horizon,
        Prior(azimuth_deg=site.view_azimuth_deg),
        exclude=default_overlay_boxes(width, height),
    )
    return skyline, horizon, elev_m


def calibrate_camera(
    camera_id: str,
    site: CameraSite,
    frames_dir: Path,
    dem: Dem,
    *,
    frozen_on: dt.date,
    min_sun_elevation_deg: float = 0.0,
    max_range_km: float = 50.0,
    sun_tolerance_deg: float = 1.0,
    scan_box_km: float = 0.0,
    scan_step_km: float = 2.0,
) -> Calibration:
    """Fit one camera from the frames under `frames_dir` and the terrain in `dem`.

    With `scan_box_km` above zero the fit is repeated with the camera moved over a grid inside
    that box, and the spread is recorded in the fit (see `summarize_scan`); the reported fit is
    always the one at the site.
    """
    timed, without_time = [], 0
    for frame in discover_frames(frames_dir):
        if frame.when_utc is None:
            without_time += 1
        else:
            timed.append(frame)
    chosen, too_dark = [], 0
    for frame in timed:
        sun = solar_position(frame.when_utc, site.lat, site.lon)
        if sun.elevation_deg >= min_sun_elevation_deg:
            chosen.append((frame, sun.elevation_deg))
        else:
            too_dark += 1
    if not chosen:
        raise CalibrationError(
            f"no frame has the sun at or above {min_sun_elevation_deg:g} degrees "
            f"({len(timed)} with a time, {too_dark} too dark, {without_time} without a time)"
        )

    images = [load_rgb(frame.path) for frame, _ in chosen]
    if len({image.shape for image in images}) != 1:
        raise CalibrationError("the frames are not all the same size")
    height, width = images[0].shape[:2]

    at_site = _fit_at(images, dem, site, 0.0, 0.0, max_range_km)
    if at_site is None:
        raise CalibrationError("the DEM does not cover the camera's position and view")
    skyline, horizon, elev_m = at_site
    radial = fit_radial_term(skyline.observed, horizon, skyline.orientation, width, height)

    position_scan = None
    if scan_box_km > 0:
        trials = [PositionTrial(0.0, 0.0, skyline.rms_px, skyline.orientation)]
        for east, north in scan_offsets(scan_box_km, scan_step_km):
            if (east, north) == (0.0, 0.0):
                continue
            found = _fit_at(images, dem, site, east, north, max_range_km)
            if found is not None:
                trials.append(PositionTrial(east, north, found[0].rms_px, found[0].orientation))
        position_scan = summarize_scan(trials, scan_box_km, scan_step_km, skyline.rms_px)

    overlay = default_overlay_boxes(width, height)
    o = skyline.orientation
    model = CameraModel(
        camera_id,
        site.lat,
        site.lon,
        elev_m,
        o.azimuth_deg,
        o.elevation_deg,
        o.roll_deg,
        o.hfov_deg,
        width,
        height,
    )
    sun_check = run_sun_check(
        model,
        [(image, frame.when_utc) for image, (frame, _) in zip(images, chosen, strict=True)],
        tolerance_deg=sun_tolerance_deg,
        exclude=overlay,
    )

    columns = _knot_columns(width)
    knots = skyline.predicted[columns]
    fit = CameraFit(
        camera_id=camera_id,
        width=width,
        height=height,
        orientation=skyline.orientation,
        skyline_rms_px=skyline.rms_px,
        skyline_rms_deg=skyline.rms_deg,
        skyline_median_abs_px=skyline.median_abs_px,
        n_columns=skyline.n_columns,
        skyline_x_px=tuple(float(c) for c in columns),
        skyline_y_px=tuple(None if np.isnan(y) else float(y) for y in knots),
        overlay_boxes=overlay,
        frames_used=tuple(
            FrameRecord(
                evening_of(frame.when_utc, site.lon).isoformat(),
                sun_elevation,
                frame_sha256(frame.path),
            )
            for frame, sun_elevation in chosen
        ),
        sun_check=sun_check.to_json(),
        radial_term=_radial_summary(skyline, radial),
        position_scan=position_scan,
        frozen_on=frozen_on.isoformat(),
        position_basis=POSITION_BASIS,
    )
    return Calibration(fit, skyline, radial, sun_check, len(chosen), too_dark, without_time)


def _radial_summary(plain: SkylineFit, radial: RadialTerm) -> dict:
    gain = 100.0 * (1.0 - radial.rms_px / plain.rms_px) if plain.rms_px > 0 else 0.0
    hfov_change = 100.0 * (radial.orientation.hfov_deg / plain.orientation.hfov_deg - 1.0)
    return {
        "k1": round(radial.k1, 3),
        "at_bound": abs(radial.k1) >= RADIAL_K1_BOUND - 1e-3,
        "skyline_rms_px": round(radial.rms_px, 3),
        "skyline_rms_px_without": round(plain.rms_px, 3),
        "rms_gain_percent": round(gain, 1),
        "hfov_change_percent": round(hfov_change, 1),
        "adopted": False,
        "reason": (
            "diagnostic only: the camera model is rectilinear; a term would be adopted only if "
            "it cut the residual substantially without shifting the field of view"
        ),
    }
