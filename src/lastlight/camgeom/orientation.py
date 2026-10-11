"""Fitting a camera's azimuth, elevation, roll and field of view by matching two skylines.

The skyline the camera sees (found in its frames, `lastlight.camgeom.skyline`) and the skyline
the terrain predicts (`lastlight.camgeom.dem.horizon_profile`) are the same curve seen through
unknown optics, so the optics are the parameters that make them agree.

The match runs in three stages, because the objective has many local minima (a ridge looks
like itself shifted by a peak spacing) and real frames hold edges that are not the horizon:

1. A coarse search over azimuth, field of view and a vertical shift scores each candidate by
   the edge strength the predicted curve passes through. No skyline has been chosen yet, so a
   cloud edge cannot be mistaken for one.
2. The best few candidates are climbed on the same score; the best of those is kept.
3. From there, the skyline is traced in each frame in a band around the prediction, the
   frames' skylines are combined by their median, and a robust least-squares fit moves the four
   parameters onto it. Trace and fit repeat with a narrower band.

The residual is vertical, in pixels, one value per image column. It is reported as an RMS in
pixels and in degrees, and the fit also measures whether a radial distortion term would
explain the residual (`fit_radial_term`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize

from lastlight.camgeom.model import project_sky_to_pixel
from lastlight.camgeom.skyline import (
    Box,
    combine_skylines,
    edge_strength,
    trace_skyline,
)

_COARSE_BLUR = 4.0  # extra smoothing of the edge map for the coarse search, in pixels
_COARSE_AZIMUTH_STEP = 1.0
_COARSE_HFOV_RATIO = 1.15  # neighbouring candidate fields of view differ by this factor
_COARSE_SHIFT_STEP = 5.0
_CANDIDATES_KEPT = 6
_BANDS = (30.0, 18.0, 12.0)  # half-widths of the trace band, wide to narrow
_ROBUST_SCALE_PX = 2.0
_MISSING_PENALTY_PX = 30.0  # residual charged to a column the predicted curve does not cover
RADIAL_K1_BOUND = 0.6  # the radial term is searched within plus or minus this
DEFAULT_HFOV_RANGE_DEG = (10.0, 120.0)  # from a long telephoto to a wide angle: a physical lens
HFOV_BOUND_TOLERANCE_DEG = 0.05  # a fit this close to a limit counts as pinned to it


def hfov_is_pinned(hfov_deg: float, bounds: tuple[float, float] = DEFAULT_HFOV_RANGE_DEG) -> bool:
    """True when a fitted field of view sits on a limit of the range it was searched in."""
    low, high = bounds
    return hfov_deg <= low + HFOV_BOUND_TOLERANCE_DEG or hfov_deg >= high - HFOV_BOUND_TOLERANCE_DEG


@dataclass(frozen=True, slots=True)
class Orientation:
    """Where the optical axis points and how wide the camera sees, in degrees."""

    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    hfov_deg: float

    def as_array(self) -> np.ndarray:
        return np.array([self.azimuth_deg, self.elevation_deg, self.roll_deg, self.hfov_deg])


@dataclass(frozen=True, slots=True)
class Prior:
    """What is known before fitting: roughly where the camera looks, and how far it may be off.

    The azimuth is the operator's stated view direction. The other bounds are wide on purpose:
    they keep the search physical (a webcam does not have a 140 degree rectilinear lens), not
    tight.
    """

    azimuth_deg: float
    azimuth_halfwidth_deg: float = 20.0
    elevation_halfwidth_deg: float = 10.0
    roll_halfwidth_deg: float = 5.0
    hfov_range_deg: tuple[float, float] = DEFAULT_HFOV_RANGE_DEG


@dataclass(frozen=True)
class HorizonProfile:
    """The terrain's skyline: elevation angle in degrees at each of a run of azimuths.

    Azimuths are increasing labels in degrees and may run past 360 or below 0 so that a view
    across north is one unbroken run.
    """

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray

    def __post_init__(self) -> None:
        az, el = np.asarray(self.azimuth_deg), np.asarray(self.elevation_deg)
        if az.ndim != 1 or az.shape != el.shape:
            raise ValueError(
                "a horizon profile needs matching one-dimensional azimuths and elevations"
            )
        if not (np.all(np.isfinite(az)) and np.all(np.isfinite(el))):
            raise ValueError("a horizon profile must have a value at every azimuth (no NaN)")
        if az.size and np.any(np.diff(az) <= 0):
            raise ValueError("the azimuths of a horizon profile must be strictly increasing")

    @classmethod
    def from_samples(cls, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> HorizonProfile:
        """Build a profile from samples, dropping the azimuths where there was no terrain."""
        keep = np.isfinite(elevation_deg)
        return cls(np.asarray(azimuth_deg)[keep], np.asarray(elevation_deg)[keep])


@dataclass(frozen=True)
class SkylineFit:
    """The fitted orientation and how well the predicted skyline matches the observed one."""

    orientation: Orientation
    rms_px: float
    rms_deg: float
    median_abs_px: float
    n_columns: int
    observed: np.ndarray
    predicted: np.ndarray


@dataclass(frozen=True)
class RadialTerm:
    """What a one-parameter lens distortion would do to the fit; see `fit_radial_term`."""

    k1: float
    rms_px: float
    orientation: Orientation


def predict_skyline(
    orientation: Orientation,
    horizon: HorizonProfile,
    width: int,
    height: int,
    *,
    radial_k1: float = 0.0,
) -> np.ndarray:
    """Row of the predicted skyline in each image column; NaN where the horizon is not seen.

    The horizon samples are projected into the image and the resulting curve is read off at
    every column by linear interpolation, with no extrapolation past its ends.
    """
    x, y = project_sky_to_pixel(
        horizon.azimuth_deg,
        horizon.elevation_deg,
        camera_azimuth_deg=orientation.azimuth_deg,
        camera_elevation_deg=orientation.elevation_deg,
        roll_deg=orientation.roll_deg,
        hfov_deg=orientation.hfov_deg,
        width=width,
        height=height,
        radial_k1=radial_k1,
    )
    seen = np.isfinite(x) & np.isfinite(y)
    if seen.sum() < 2:
        return np.full(width, np.nan)
    order = np.argsort(x[seen])
    return np.interp(
        np.arange(width, dtype=float), x[seen][order], y[seen][order], left=np.nan, right=np.nan
    )


def _focal_px(hfov_deg: float, width: int) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


# --- stage 1 and 2: the edge-energy search -----------------------------------------------------


def _energy(strength: np.ndarray, predicted: np.ndarray) -> float:
    """Mean edge strength along a predicted skyline, counting columns it misses as zero."""
    height, width = strength.shape
    inside = np.isfinite(predicted) & (predicted >= 0.0) & (predicted <= height - 1.0)
    if inside.sum() < width / 2:
        return -np.inf
    rows = np.where(inside, predicted, 0.0)
    values = ndimage.map_coordinates(strength, [rows, np.arange(width)], order=1)
    return float(values[inside].sum() / width)


def _coarse_candidates(
    strength: np.ndarray, horizon: HorizonProfile, prior: Prior
) -> list[tuple[float, Orientation]]:
    height, width = strength.shape
    broad = ndimage.gaussian_filter(strength, _COARSE_BLUR)
    azimuths = np.arange(
        prior.azimuth_deg - prior.azimuth_halfwidth_deg,
        prior.azimuth_deg + prior.azimuth_halfwidth_deg + 1e-9,
        _COARSE_AZIMUTH_STEP,
    )
    low, high = prior.hfov_range_deg
    count = max(2, math.ceil(math.log(high / low) / math.log(_COARSE_HFOV_RATIO)) + 1)
    hfovs = np.geomspace(low, high, count)
    shifts = np.arange(-height, height + 1.0, _COARSE_SHIFT_STEP)
    columns = np.arange(width)[None, :]
    found: list[tuple[float, Orientation]] = []
    for azimuth in azimuths:
        for hfov in hfovs:
            base = predict_skyline(Orientation(azimuth, 0.0, 0.0, hfov), horizon, width, height)
            rows = base[None, :] + shifts[:, None]
            inside = np.isfinite(rows) & (rows >= 0.0) & (rows <= height - 1.0)
            # The map is already blurred over several pixels, so the nearest row will do.
            nearest = np.rint(np.where(inside, rows, 0.0)).astype(np.intp)
            sampled = broad[nearest, columns]
            energy = (sampled * inside).sum(axis=1) / width
            energy[inside.sum(axis=1) < width / 2] = -np.inf
            best = int(np.argmax(energy))
            if np.isfinite(energy[best]):
                # A skyline `shift` rows lower is a camera pitched up by atan(shift / f).
                pitch = math.degrees(math.atan(shifts[best] / _focal_px(hfov, width)))
                found.append((float(energy[best]), Orientation(azimuth, pitch, 0.0, hfov)))
    return _distinct_best(found)


def _distinct_best(found: list[tuple[float, Orientation]]) -> list[tuple[float, Orientation]]:
    """The best candidates, skipping any that sits next to a better one."""
    kept: list[tuple[float, Orientation]] = []
    for energy, orientation in sorted(found, key=lambda item: item[0], reverse=True):
        near = any(
            abs(orientation.azimuth_deg - other.azimuth_deg) < 2.5
            and abs(orientation.hfov_deg - other.hfov_deg) < 0.12 * other.hfov_deg
            for _, other in kept
        )
        if not near:
            kept.append((energy, orientation))
        if len(kept) == _CANDIDATES_KEPT:
            break
    return kept


def _climb(
    start: Orientation, strength: np.ndarray, horizon: HorizonProfile, prior: Prior
) -> tuple[float, Orientation]:
    """Hill-climb the edge score from a coarse candidate."""
    height, width = strength.shape
    bounds = _bounds(prior)

    def negative_energy(params: np.ndarray) -> float:
        if np.any(params < bounds[0]) or np.any(params > bounds[1]):
            return 1e9
        energy = _energy(strength, predict_skyline(Orientation(*params), horizon, width, height))
        return 1e9 if not np.isfinite(energy) else -energy

    result = optimize.minimize(
        negative_energy,
        start.as_array(),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 600, "initial_simplex": _simplex(start)},
    )
    return -float(result.fun), Orientation(*result.x)


def _simplex(start: Orientation) -> np.ndarray:
    """A starting simplex sized to the coarse grid: about a grid step in each parameter."""
    origin = start.as_array()
    steps = np.array([_COARSE_AZIMUTH_STEP, 1.0, 0.5, 0.05 * start.hfov_deg])
    return np.vstack([origin, *(origin + np.eye(4)[i] * steps[i] for i in range(4))])


def _bounds(prior: Prior) -> tuple[np.ndarray, np.ndarray]:
    margin = 5.0
    low = [
        prior.azimuth_deg - prior.azimuth_halfwidth_deg - margin,
        -prior.elevation_halfwidth_deg - margin,
        -prior.roll_halfwidth_deg,
        prior.hfov_range_deg[0],
    ]
    high = [
        prior.azimuth_deg + prior.azimuth_halfwidth_deg + margin,
        prior.elevation_halfwidth_deg + margin,
        prior.roll_halfwidth_deg,
        prior.hfov_range_deg[1],
    ]
    return np.array(low), np.array(high)


# --- stage 3: trace and fit -------------------------------------------------------------------


def _fit_to_skyline(
    start: Orientation,
    observed: np.ndarray,
    horizon: HorizonProfile,
    width: int,
    height: int,
    prior: Prior,
) -> Orientation:
    """Robust least squares of the four parameters against an observed skyline."""
    low, high = _bounds(prior)

    def residual(params: np.ndarray) -> np.ndarray:
        predicted = predict_skyline(Orientation(*params), horizon, width, height)
        difference = predicted - observed
        return np.where(np.isfinite(difference), difference, _MISSING_PENALTY_PX)

    result = optimize.least_squares(
        residual,
        np.clip(start.as_array(), low, high),
        bounds=(low, high),
        loss="soft_l1",
        f_scale=_ROBUST_SCALE_PX,
        x_scale=np.array([1.0, 1.0, 1.0, 0.07 * start.hfov_deg]),
        diff_step=1e-6,
    )
    return Orientation(*result.x)


def fit_orientation(
    frames: Sequence[np.ndarray],
    horizon: HorizonProfile,
    prior: Prior,
    *,
    exclude: Sequence[Box] = (),
) -> SkylineFit:
    """Fit the orientation to one or more frames of the same, fixed camera.

    `frames` are RGB (or grayscale) arrays of one size; `exclude` lists the boxes whose edges
    must not count, such as the burned-in timestamp.
    """
    if len(frames) == 0:
        raise ValueError("fit_orientation needs at least one frame")
    if len({np.shape(frame)[:2] for frame in frames}) != 1:
        raise ValueError("all frames must be the same size")
    height, width = np.shape(frames[0])[:2]
    strengths = [edge_strength(frame, exclude) for frame in frames]
    mean_strength = np.mean(strengths, axis=0)

    climbed = [
        _climb(orientation, mean_strength, horizon, prior)
        for _, orientation in _coarse_candidates(mean_strength, horizon, prior)
    ]
    if not climbed:
        raise ValueError("no placement of the horizon puts half of it inside the frame")
    orientation = max(climbed, key=lambda item: item[0])[1]

    observed = np.full(width, np.nan)
    for half_width in _BANDS:
        centre = predict_skyline(orientation, horizon, width, height)
        traces = [trace_skyline(s, around=centre, half_width=half_width) for s in strengths]
        observed = combine_skylines(traces)
        orientation = _fit_to_skyline(orientation, observed, horizon, width, height, prior)

    predicted = predict_skyline(orientation, horizon, width, height)
    difference = (predicted - observed)[np.isfinite(predicted - observed)]
    rms = float(np.sqrt(np.mean(difference**2)))
    focal = _focal_px(orientation.hfov_deg, width)
    return SkylineFit(
        orientation=orientation,
        rms_px=rms,
        rms_deg=rms / (focal * math.pi / 180.0),
        median_abs_px=float(np.median(np.abs(difference))),
        n_columns=int(difference.size),
        observed=observed,
        predicted=predicted,
    )


def fit_radial_term(
    observed: np.ndarray,
    horizon: HorizonProfile,
    start: Orientation,
    width: int,
    height: int,
) -> RadialTerm:
    """Refit with a radial distortion term k1 to see whether the lens explains the residual.

    The projection is bent as in `project_sky_to_pixel` (`r' = r (1 + k1 r^2)`) and k1 is fitted
    together with the four orientation parameters, starting from the plain fit's solution. The
    result is a diagnostic: the camera model does not carry k1. A term is worth adding only if it
    lowers the residual by far more than its cost and does not simply trade off against the field
    of view (when the field of view moves by several percent to make room for it, the data cannot
    tell the two apart).
    """

    def residual(params: np.ndarray) -> np.ndarray:
        predicted = predict_skyline(
            Orientation(*params[:4]), horizon, width, height, radial_k1=params[4]
        )
        difference = predicted - observed
        return np.where(np.isfinite(difference), difference, _MISSING_PENALTY_PX)

    origin = np.append(start.as_array(), 0.0)
    low = np.array(
        [
            start.azimuth_deg - 10.0,
            start.elevation_deg - 10.0,
            -5.0,
            0.6 * start.hfov_deg,
            -RADIAL_K1_BOUND,
        ]
    )
    high = np.array(
        [
            start.azimuth_deg + 10.0,
            start.elevation_deg + 10.0,
            5.0,
            min(1.6 * start.hfov_deg, 170.0),
            RADIAL_K1_BOUND,
        ]
    )
    result = optimize.least_squares(
        residual,
        np.clip(origin, low, high),
        bounds=(low, high),
        loss="soft_l1",
        f_scale=_ROBUST_SCALE_PX,
        x_scale=np.array([1.0, 1.0, 1.0, 0.07 * start.hfov_deg, 0.1]),
        diff_step=1e-6,
    )
    final = residual(result.x)
    missing = final == _MISSING_PENALTY_PX
    rms = float(np.sqrt(np.mean(final[~missing] ** 2)))
    return RadialTerm(k1=float(result.x[4]), rms_px=rms, orientation=Orientation(*result.x[:4]))
