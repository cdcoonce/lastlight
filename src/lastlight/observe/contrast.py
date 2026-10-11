"""The within-frame contrast profile of one frame (`docs/design.md` section 1, tier A).

A public webcam's automatic exposure and white balance act, to first order, as one gain per
colour channel on the linear signal. In `log(R/G)` and `log(B/G)` such a gain is an additive
constant, so the difference between two regions of the same frame cancels it. `contrast_profile`
returns those differences: for each band, the median `log(R/G)` and `log(B/G)` of its pixels
minus the reference band's. Absolute colour is never reported.

**Bands are pixel rows above the camera's frozen skyline**, column by column
(`lastlight.cameras.ContrastBands`): the band `(lo, hi)` is the pixels `d` rows above the skyline
with `lo <= d < hi`, where the skyline of a column is its fitted row rounded half up to a whole
row. They are in pixels, not degrees, so an observation is exact and repeatable across refits;
only the label `view_elevation_deg` (nominal, with a range) depends on the fit's pixel scale.

What is counted, and what is left out (never filled):

1. A column in which the whole band is not inside the frame and the camera's sky mask (above
   the margin, outside the overlay boxes, skyline seen) is left out of that band: `geometry`.
2. While the sun is below -4 degrees, the camera's light-dome columns are left out of every
   band and the reference: `light_dome`.
3. A pixel with any channel at or above 250 (of 255) is `saturated`.
4. A pixel with every channel at or below 5, or any channel at 0 (its log is undefined), is
   `too_dark`.

The four counts are disjoint (a pixel is charged to the first that applies) and, with the valid
pixels, add up to the band's nominal size: its rows times the frame's width. A band with fewer
than 50 valid pixels is missing, and so is every contrast when the reference is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from lastlight.cameras.registry import ContrastBands
from lastlight.camgeom.fitfile import CameraFit
from lastlight.camgeom.mask import skyline_top_rows
from lastlight.camgeom.model import pixels_per_degree, unproject_pixel_to_sky
from lastlight.observe.bands import columns_inside

SATURATION_LEVEL: Final = 250
"""A pixel with any channel at or above this code (of 255) is saturated and left out."""

DARK_LEVEL: Final = 5
"""A pixel with every channel at or below this code is too dark to carry colour and left out."""

MIN_VALID_PIXELS: Final = 50
"""A band with fewer valid pixels than this is reported missing, not filled."""

LIGHT_DOME_SUN_ELEVATION_DEG: Final = -4.0
"""The light-dome columns are excluded while the sun is below this elevation (strictly)."""

MISSING_TOO_FEW: Final = "too_few_pixels"
MISSING_REFERENCE: Final = "reference_missing"

_LOG_DECIMALS: Final = 6
_DEGREE_DECIMALS: Final = 2


@dataclass(frozen=True, slots=True)
class ExcludedPixels:
    """Pixels of a band's nominal footprint that did not count, by the first reason that applies."""

    geometry: int = 0
    light_dome: int = 0
    saturated: int = 0
    too_dark: int = 0

    @property
    def total(self) -> int:
        return self.geometry + self.light_dome + self.saturated + self.too_dark

    def __add__(self, other: ExcludedPixels) -> ExcludedPixels:
        return ExcludedPixels(
            self.geometry + other.geometry,
            self.light_dome + other.light_dome,
            self.saturated + other.saturated,
            self.too_dark + other.too_dark,
        )

    def to_json(self) -> dict[str, int]:
        return {
            "geometry": self.geometry,
            "light_dome": self.light_dome,
            "saturated": self.saturated,
            "too_dark": self.too_dark,
        }


@dataclass(frozen=True, slots=True)
class ViewElevation:
    """The elevation a band looks at, NOMINAL: from the fit's provisional pixel scale.

    `range_deg` is the same band re-scaled with the ends of the fit's `uncertainty.px_per_deg`
    (and the nominal scale), keeping the skyline where the fit put it. Nothing independent has
    confirmed the scale until the sun enters the frame, so this is a label for the forecast's
    mapping from rows to elevation, not part of the observation.
    """

    nominal_deg: float
    range_deg: tuple[float, float]

    def to_json(self) -> dict[str, Any]:
        low, high = self.range_deg
        return {
            "nominal": round(self.nominal_deg, _DEGREE_DECIMALS),
            "range": [round(low, _DEGREE_DECIMALS), round(high, _DEGREE_DECIMALS)],
        }


@dataclass(frozen=True, slots=True)
class BandProfile:
    """One band's contrast. `log_rg` and `log_bg` are the band's medians minus the reference's.

    `missing` is None for a band with a contrast, `"too_few_pixels"` when the band itself has
    fewer than 50 valid pixels, and `"reference_missing"` when only the reference is short.
    """

    rows: tuple[int, int]
    log_rg: float | None
    log_bg: float | None
    n_pixels: int
    excluded: ExcludedPixels
    view_elevation: ViewElevation | None
    missing: str | None

    @property
    def n_excluded(self) -> int:
        return self.excluded.total

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": list(self.rows),
            "log_rg": None if self.log_rg is None else round(self.log_rg, _LOG_DECIMALS),
            "log_bg": None if self.log_bg is None else round(self.log_bg, _LOG_DECIMALS),
            "n_pixels": self.n_pixels,
            "n_excluded": self.n_excluded,
            "excluded": self.excluded.to_json(),
            "view_elevation_deg": (
                None if self.view_elevation is None else self.view_elevation.to_json()
            ),
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """The reference band's own counts. Its colour is never reported (it is not a contrast)."""

    rows: tuple[int, int]
    n_pixels: int
    excluded: ExcludedPixels
    missing: str | None

    @property
    def n_excluded(self) -> int:
        return self.excluded.total

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": list(self.rows),
            "n_pixels": self.n_pixels,
            "n_excluded": self.n_excluded,
            "excluded": self.excluded.to_json(),
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class ContrastProfile:
    """The contrast of every band of one frame, and what was left out to get it."""

    sun_elevation_deg: float
    light_dome_excluded: bool
    bands: tuple[BandProfile, ...]
    reference: ReferenceProfile

    def to_json(self) -> dict[str, Any]:
        return {
            "light_dome_excluded": self.light_dome_excluded,
            "reference": self.reference.to_json(),
            "bands": [band.to_json() for band in self.bands],
        }


def srgb_to_linear(codes: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, inverted: codes on the 0 to 255 scale to linear light, 0 to 1."""
    encoded = np.asarray(codes, dtype=float) / 255.0
    return np.where(encoded <= 0.04045, encoded / 12.92, np.power((encoded + 0.055) / 1.055, 2.4))


def _as_frame(rgb: np.ndarray, fit: CameraFit) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"a frame must be rows by columns by three RGB channels, got shape {frame.shape}"
        )
    if frame.shape[:2] != (fit.height, fit.width):
        raise ValueError(
            f"the frame is {frame.shape[0]}x{frame.shape[1]} (rows by columns) but the fit of "
            f"{fit.camera_id} is {fit.height}x{fit.width}; a frame is never resized"
        )
    return frame.astype(float, copy=False)


def _dome_columns(contrast: ContrastBands, width: int) -> np.ndarray:
    columns = np.zeros(width, dtype=bool)
    for low, high in contrast.light_dome_columns:
        columns[low : min(high, width)] = True
    return columns


class _Measured:
    """What one band's pixels came to: counts, and the medians when there are enough."""

    __slots__ = ("excluded", "log_bg", "log_rg", "n_pixels", "view_elevation")

    def __init__(self) -> None:
        self.excluded = ExcludedPixels()
        self.log_rg: float | None = None
        self.log_bg: float | None = None
        self.n_pixels = 0
        self.view_elevation: ViewElevation | None = None

    @property
    def enough(self) -> bool:
        return self.n_pixels >= MIN_VALID_PIXELS


def _measure(
    frame: np.ndarray,
    fit: CameraFit,
    rows: tuple[int, int],
    *,
    margin_rows: int,
    dome: np.ndarray | None,
) -> _Measured:
    low, high = rows
    height = high - low
    inside = columns_inside(fit, rows, margin_rows=margin_rows)
    in_dome = np.zeros_like(inside) if dome is None else dome & inside
    used = np.flatnonzero(inside & ~in_dome)

    out = _Measured()
    top = skyline_top_rows(fit)
    out.view_elevation = _view_elevation(fit, rows, np.flatnonzero(inside), top)

    top_int = np.where(np.isnan(top), 0.0, top).astype(int)
    row_index = top_int[used][None, :] - np.arange(low, high)[:, None]
    pixels = frame[row_index, used[None, :]]  # rows of the band by used columns by 3
    saturated = (pixels >= SATURATION_LEVEL).any(axis=-1)
    dark = ((pixels <= DARK_LEVEL).all(axis=-1) | (pixels <= 0).any(axis=-1)) & ~saturated
    valid = ~saturated & ~dark

    out.excluded = ExcludedPixels(
        geometry=int((~inside).sum()) * height,
        light_dome=int(in_dome.sum()) * height,
        saturated=int(saturated.sum()),
        too_dark=int(dark.sum()),
    )
    out.n_pixels = int(valid.sum())
    if out.enough:
        linear = srgb_to_linear(pixels[valid])
        out.log_rg = float(np.median(np.log(linear[:, 0] / linear[:, 1])))
        out.log_bg = float(np.median(np.log(linear[:, 2] / linear[:, 1])))
    return out


def _view_elevation(
    fit: CameraFit, rows: tuple[int, int], columns: np.ndarray, top: np.ndarray
) -> ViewElevation | None:
    """The band's nominal elevation: the skyline's, plus the band's height at the fit's scale.

    The skyline's elevation and the band's angular height above it come from the fit's own
    camera model. The height is rescaled by the nominal scale over each end of the fit's
    `uncertainty.px_per_deg`, the skyline staying where the fit put it, because the skyline is
    pinned in pixel space and the scale is what is not identified.
    """
    if columns.size == 0:
        return None
    o = fit.orientation
    low, high = rows
    x = columns.astype(float)
    y_skyline = top[columns]
    y_centre = y_skyline - (low + high - 1) / 2.0
    kwargs = {
        "camera_azimuth_deg": o.azimuth_deg,
        "camera_elevation_deg": o.elevation_deg,
        "roll_deg": o.roll_deg,
        "hfov_deg": o.hfov_deg,
        "width": fit.width,
        "height": fit.height,
    }
    _, skyline_elevation = unproject_pixel_to_sky(x, y_skyline, **kwargs)
    _, centre_elevation = unproject_pixel_to_sky(x, y_centre, **kwargs)
    skyline = float(np.mean(skyline_elevation))
    height_deg = float(np.mean(centre_elevation - skyline_elevation))
    nominal_scale = pixels_per_degree(o.hfov_deg, fit.width)
    scales = [nominal_scale]
    if fit.uncertainty is not None:
        scales += list(fit.uncertainty.px_per_deg)
    elevations = [skyline + height_deg * nominal_scale / scale for scale in scales]
    return ViewElevation(
        nominal_deg=skyline + height_deg,
        range_deg=(min(elevations), max(elevations)),
    )


def contrast_profile(
    rgb: np.ndarray,
    fit: CameraFit,
    contrast: ContrastBands,
    *,
    sun_elevation_deg: float,
) -> ContrastProfile:
    """The contrast of each of `contrast.bands_rows` in one frame.

    `rgb` is the frame as rows by columns by three channels of sRGB codes on the 0 to 255
    scale: the `uint8` array a JPEG decodes to, or floats (the tests use unrounded floats so
    that a gain can be undone exactly). It must be the size of the fit's image.
    `sun_elevation_deg` decides whether the light-dome columns are excluded: strictly below
    `LIGHT_DOME_SUN_ELEVATION_DEG`. Run `lastlight.observe.bands.validate_contrast` once per
    camera and fit first; this function trusts that the bands fit the sky.
    """
    if not math.isfinite(sun_elevation_deg):
        raise ValueError(f"the sun elevation must be a finite number, got {sun_elevation_deg!r}")
    frame = _as_frame(rgb, fit)
    dome_active = bool(contrast.light_dome_columns) and (
        sun_elevation_deg < LIGHT_DOME_SUN_ELEVATION_DEG
    )
    dome = _dome_columns(contrast, fit.width) if dome_active else None
    margin = contrast.skyline_margin_rows

    reference = _measure(frame, fit, contrast.reference_rows, margin_rows=margin, dome=dome)
    reference_missing = not reference.enough
    bands = []
    for rows in contrast.bands_rows:
        band = _measure(frame, fit, rows, margin_rows=margin, dome=dome)
        if not band.enough:
            missing: str | None = MISSING_TOO_FEW
        elif reference_missing:
            missing = MISSING_REFERENCE
        else:
            missing = None
        has_contrast = missing is None
        bands.append(
            BandProfile(
                rows=rows,
                log_rg=band.log_rg - reference.log_rg if has_contrast else None,
                log_bg=band.log_bg - reference.log_bg if has_contrast else None,
                n_pixels=band.n_pixels,
                excluded=band.excluded,
                view_elevation=band.view_elevation,
                missing=missing,
            )
        )
    return ContrastProfile(
        sun_elevation_deg=sun_elevation_deg,
        light_dome_excluded=dome_active,
        bands=tuple(bands),
        reference=ReferenceProfile(
            rows=contrast.reference_rows,
            n_pixels=reference.n_pixels,
            excluded=reference.excluded,
            missing=MISSING_TOO_FEW if reference_missing else None,
        ),
    )
