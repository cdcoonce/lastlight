"""The calibration pipeline behind `lastlight camgeom fit`: frames and terrain in, a fit out.

Steps: choose the frames (timed, with the sun high enough that the sky is brighter than the
terrain); compute the horizon the DEM predicts from the camera's position; fit the orientation
by matching skylines; optionally search the camera's position inside a stated box; measure
whether a radial distortion term would explain the residual; check the fit against the sun's
disk; and assemble the record that gets frozen.

The position search exists because an operator's description ("looks west from Avondale") is
vague, and the horizon a terrain model predicts depends on where the camera stands. It fits
from scratch at every point of a fine grid (1 km by default) over the whole stated box, and then
at a ring of probes just outside the box. A skyline alone constrains position and field of view
only jointly (move the camera toward the ridge and a narrower lens fits as well), so the record
keeps how flat the valley of good fits is, not just its floor; and the ring says whether that
valley's floor lies beyond the box that was allowed, which the box itself cannot show.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lastlight.camgeom.dem import Dem, horizon_profile
from lastlight.camgeom.fitfile import (
    CameraFit,
    FrameRecord,
    Uncertainty,
    azimuth_difference_deg,
    uncertainty_from_ranges,
)
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
    hfov_is_pinned,
)
from lastlight.camgeom.sites import CameraSite
from lastlight.camgeom.skyline import default_overlay_boxes
from lastlight.camgeom.suncheck import SunCheck, run_sun_check
from lastlight.geometry import solar_position

VIEW_HALF_SPAN_DEG = 80.0  # horizon computed this far either side of the stated view direction
AZIMUTH_STEP_DEG = 0.05
SKYLINE_KNOT_STEP_PX = 10
KM_PER_DEGREE_LAT = 111.2  # a flat-Earth offset is exact enough for a search of some kilometres
NEAR_BEST_FACTOR = 1.1  # a position within 10 percent of the best residual is "as good"
DEFAULT_STEP_KM = 1.0  # grid spacing of the position search
POSITION_BASIS = (
    "read from the camera site file at fit time and moved by the offset in position_search; "
    "coordinates are kept out of the repository (docs/design.md section 3)"
)


class CalibrationError(ValueError):
    """The inputs cannot support a fit; the message says what is missing."""


@dataclass(frozen=True, slots=True)
class PositionTrial:
    """The best fit found with the camera moved `east_km` and `north_km` from the site."""

    east_km: float
    north_km: float
    skyline_rms_px: float
    orientation: Orientation


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


# --- the position search: pure logic over a function that fits at an offset -----------------------


def search_offsets(box_km: float, step_km: float) -> list[tuple[float, float]]:
    """(east, north) offsets in km: every multiple of the step inside the box about the site."""
    if box_km < 0 or step_km <= 0:
        raise ValueError("the search box must not be negative and the step must be positive")
    count = math.floor(box_km / step_km + 1e-9)
    steps = range(-count, count + 1)
    return [(east * step_km, north * step_km) for north in steps for east in steps]


def ring_offsets(box_km: float, gap_km: float, spacing_km: float) -> list[tuple[float, float]]:
    """Offsets on the square `gap_km` outside the box, `spacing_km` apart, corners included."""
    if gap_km <= 0 or spacing_km <= 0:
        raise ValueError("the ring's gap and spacing must be positive")
    half = box_km + gap_km
    segments = max(1, round(2.0 * half / spacing_km))
    coords = [-half + 2.0 * half * i / segments for i in range(segments + 1)]
    return [
        (east, north)
        for north in coords
        for east in coords
        if max(abs(east), abs(north)) >= half - 1e-9
    ]


def choose_position(trials: Sequence[PositionTrial]) -> PositionTrial:
    """The trial with the lowest residual, never one whose field of view sits on a limit.

    A fit pinned to the edge of the allowed field-of-view range is the model asking for more
    than the range permits, so its low residual says nothing about where the camera could be.
    """
    usable = [t for t in trials if not hfov_is_pinned(t.orientation.hfov_deg)]
    if not usable:
        raise ValueError("no position gave a fit away from the field-of-view limits")
    return min(usable, key=lambda t: t.skyline_rms_px)


@dataclass(frozen=True)
class PositionSearch:
    """What the search found: the chosen trial, every trial in the box, and the ring's trials."""

    best: PositionTrial
    trials: list[PositionTrial]
    ring: list[PositionTrial]


def search_position(
    fit_at: Callable[[float, float], PositionTrial | None],
    box_km: float,
    step_km: float,
    *,
    ring_gap_km: float | None = None,
    ring_spacing_km: float | None = None,
) -> PositionSearch:
    """Fit at every point of a grid of `step_km` over the box, then probe a ring outside it.

    `fit_at(east_km, north_km)` returns the trial there, or None where no fit is possible. No
    position is fitted twice. The chosen position is the best inside the box: the ring is only
    evidence about what lies beyond it, and is never chosen. The ring is a square `ring_gap_km`
    outside the box with a probe every `ring_spacing_km`; both default to the grid step, so the
    ring is sampled as finely as the box and its best is comparable with the box's best (the
    residual is bumpy at the scale of a kilometre, and a coarser ring would miss the dips). A box
    smaller than the step asks for the site and nothing else, with no ring.
    """
    offsets = search_offsets(box_km, step_km)
    trials = [found for offset in offsets if (found := fit_at(*offset)) is not None]
    if not trials:
        raise ValueError("no position in the box gave a usable fit")
    ring: list[PositionTrial] = []
    if len(offsets) > 1:
        ring = [
            found
            for offset in ring_offsets(box_km, ring_gap_km or step_km, ring_spacing_km or step_km)
            if (found := fit_at(*offset)) is not None
        ]
    return PositionSearch(choose_position(trials), trials, ring)


def summarize_search(
    trials: Sequence[PositionTrial],
    best: PositionTrial,
    box_km: float,
    step_km: float,
    site_rms_px: float | None,
    *,
    ring: Sequence[PositionTrial] = (),
) -> dict:
    """What the fit file keeps of the search: the offset adopted and the valley it sits in.

    Offsets are from the site file's position, which stays out of the repository, so the record
    holds distances and statistics and no coordinates.

    `best_on_box_edge` is true when the best position lies within one grid step of the box's
    edge, where the box (a prior the operator's description set) and not the data may have
    stopped the search. `residual_falls_beyond_box` is true when a probe in the ring just outside
    the box fits better than anything inside it, which says the valley's floor is out there.
    """
    near = [
        t
        for t in trials
        if not hfov_is_pinned(t.orientation.hfov_deg)
        and t.skyline_rms_px <= NEAR_BEST_FACTOR * best.skyline_rms_px
    ]
    azimuth = [t.orientation.azimuth_deg for t in near]
    hfov = [t.orientation.hfov_deg for t in near]
    on_edge = box_km > 0 and max(abs(best.east_km), abs(best.north_km)) >= box_km - step_km - 1e-9
    ring_usable = [t.skyline_rms_px for t in ring if not hfov_is_pinned(t.orientation.hfov_deg)]
    ring_best = min(ring_usable, default=None)
    falls_beyond = ring_best is not None and ring_best < best.skyline_rms_px
    notes = []
    if on_edge:
        notes.append("the floor lies on the edge of the box, so the valley may continue beyond it")
    if falls_beyond:
        notes.append(
            f"the ring just outside the box fits better still ({ring_best:.2f} px against "
            f"{best.skyline_rms_px:.2f}), so the valley continues beyond the box"
        )
    return {
        "box_km": box_km,
        "step_km": step_km,
        "positions": len(trials),
        "offset_east_km": round(best.east_km, 1),
        "offset_north_km": round(best.north_km, 1),
        "rms_px_at_site": None if site_rms_px is None else round(site_rms_px, 3),
        "rms_px_best": round(best.skyline_rms_px, 3),
        "near_best_positions": len(near),
        "near_best_azimuth_deg_range": [round(min(azimuth), 2), round(max(azimuth), 2)],
        "near_best_hfov_deg_range": [round(min(hfov), 2), round(max(hfov), 2)],
        "positions_at_hfov_bound": int(sum(hfov_is_pinned(t.orientation.hfov_deg) for t in trials)),
        "best_on_box_edge": bool(on_edge),
        "ring_positions": len(ring),
        "ring_rms_px_best": None if ring_best is None else round(ring_best, 3),
        "residual_falls_beyond_box": bool(falls_beyond),
        "adopted": True,
        "reason": (
            f"lowest residual in a flat valley: {len(near)} positions lie within 10 percent of "
            "it, and azimuth and field of view move with the position along it, so the offset "
            "is the floor of the valley and not a measured location"
            + "".join(f"; {note}" for note in notes)
        ),
    }


def search_uncertainty(summary: dict, width: int) -> Uncertainty:
    """The ranges of the nearly equal fits, from a search summary, as the fit file's block."""
    count = summary["near_best_positions"]
    lower_bounds = bool(summary["best_on_box_edge"] or summary["residual_falls_beyond_box"])
    basis = (
        f"{count} positions inside the {summary['box_km']:g} km position box fit within 10 "
        f"percent of the best skyline residual ({summary['rms_px_best']:g} px)"
    )
    if lower_bounds:
        basis += (
            "; the best sits at the edge of the box or the residual keeps falling beyond it, "
            "so these ranges are lower bounds"
        )
    if summary["residual_falls_beyond_box"]:
        basis += " (the ring of probes just outside the box fits better than anything inside)"
    return uncertainty_from_ranges(
        tuple(summary["near_best_azimuth_deg_range"]),
        tuple(summary["near_best_hfov_deg_range"]),
        width,
        basis,
        lower_bounds,
    )


@dataclass(frozen=True, slots=True)
class Observer:
    """Where a camera stands for one trial: the position and the height above sea level."""

    lat: float
    lon: float
    elev_m: float


def observer_at(dem: Dem, site: CameraSite, east_km: float, north_km: float) -> Observer | None:
    """The camera moved `east_km` and `north_km` from the site, on the terrain model's ground.

    East adds to the longitude (a degree of longitude is shorter by the cosine of the latitude)
    and north to the latitude. The camera stands on the model's ground there plus the mount
    height. None where the model has no ground.
    """
    lat = site.lat + north_km / KM_PER_DEGREE_LAT
    lon = site.lon + east_km / (KM_PER_DEGREE_LAT * math.cos(math.radians(site.lat)))
    ground = dem.sample(lat, lon)
    if np.isnan(ground):
        return None
    return Observer(lat, lon, float(ground) + site.mount_height_m)


# --- the pipeline ---------------------------------------------------------------------------------


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


@dataclass(frozen=True)
class _Placement:
    """A fit with the camera at one offset from the site, and what it was fitted against."""

    trial: PositionTrial
    skyline: SkylineFit
    horizon: HorizonProfile
    lat: float
    lon: float
    elev_m: float


def _fit_placed(
    images: list[np.ndarray],
    dem: Dem,
    site: CameraSite,
    east_km: float,
    north_km: float,
    max_range_km: float,
) -> _Placement | None:
    """Fit with the camera moved from the site; None where the DEM cannot support it."""
    observer = observer_at(dem, site, east_km, north_km)
    if observer is None:
        return None
    lat, lon, elev_m = observer.lat, observer.lon, observer.elev_m
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
    trial = PositionTrial(east_km, north_km, skyline.rms_px, skyline.orientation)
    return _Placement(trial, skyline, horizon, lat, lon, elev_m)


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
    position_box_km: float = 0.0,
    position_step_km: float = DEFAULT_STEP_KM,
    stated_azimuth_deg: float | None = None,
) -> Calibration:
    """Fit one camera from the frames under `frames_dir` and the terrain in `dem`.

    With `position_box_km` above zero the camera's position is searched inside that box about
    the site (see `search_position`) and the fit is the one at the best position; otherwise the
    camera is taken to stand at the site. `stated_azimuth_deg` is the direction the operator
    states (the camera registry's `view_azimuth_deg`); the fit records how far it landed from it.
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

    placements: dict[tuple[float, float], _Placement] = {}

    def fit_at(east_km: float, north_km: float) -> PositionTrial | None:
        placed = _fit_placed(images, dem, site, east_km, north_km, max_range_km)
        if placed is None:
            return None
        placements[(east_km, north_km)] = placed
        return placed.trial

    position_search = None
    if position_box_km > 0:
        try:
            found = search_position(fit_at, position_box_km, position_step_km)
        except ValueError as error:
            raise CalibrationError(str(error)) from error
        site_placement = placements.get((0.0, 0.0))
        position_search = summarize_search(
            found.trials,
            found.best,
            position_box_km,
            position_step_km,
            None if site_placement is None else site_placement.trial.skyline_rms_px,
            ring=found.ring,
        )
        placed = placements[(found.best.east_km, found.best.north_km)]
    else:
        if fit_at(0.0, 0.0) is None:
            raise CalibrationError("the DEM does not cover the camera's position and view")
        placed = placements[(0.0, 0.0)]

    skyline = placed.skyline
    radial = fit_radial_term(skyline.observed, placed.horizon, skyline.orientation, width, height)

    overlay = default_overlay_boxes(width, height)
    o = skyline.orientation
    model = CameraModel(
        camera_id,
        placed.lat,
        placed.lon,
        placed.elev_m,
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
        position_search=position_search,
        frozen_on=frozen_on.isoformat(),
        position_basis=POSITION_BASIS,
        provisional=sun_check.status != "passed",
        uncertainty=(
            None if position_search is None else search_uncertainty(position_search, width)
        ),
        stated_azimuth_deg=stated_azimuth_deg,
        azimuth_minus_stated_deg=(
            None
            if stated_azimuth_deg is None
            else azimuth_difference_deg(o.azimuth_deg, stated_azimuth_deg)
        ),
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
