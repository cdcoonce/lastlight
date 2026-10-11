"""WHTM1's view for the G0 gate: pixel rows above the frozen skyline as elevations in the sky.

The camera's contrast bands are rows above its skyline, column by column
(`lastlight.cameras.ContrastBands`). The renderer works in elevations, so each band becomes one
view elevation, by the pre-registered mapping (`docs/gates/G0/criterion.md`, definition 5):

    elevation(band) = skyline elevation + centre row of the band / pixel scale

where the skyline elevation is the median, over the fit's sky columns, of the elevation of the
fitted skyline row (the fit's own camera model); the centre row of the half-open band `[lo, hi)`
is `(lo + hi - 1) / 2` rows above the skyline; and the pixel scale is the fit's nominal
`pixels_per_degree`, or either end of `uncertainty.px_per_deg` for the sensitivity runs, with the
skyline kept where it is. Because the fit is provisional, the scale is the best of a family of
nearly equal fits, not a measurement; that is why the gate reports the statistic at both ends.

The view direction is the fit's centre azimuth. The sun's azimuth, needed for the relative
azimuth the renderer takes, is computed here from `lastlight.geometry` at each sweep elevation of
the evening. It is used internally only: no clock time and no sun azimuth is ever published.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from lastlight import geometry
from lastlight.cameras.registry import ContrastBands
from lastlight.camgeom.fitfile import CameraFit
from lastlight.camgeom.mask import skyline_top_rows
from lastlight.camgeom.model import pixels_per_degree, unproject_pixel_to_sky
from lastlight.observe.bands import columns_inside


class ViewError(ValueError):
    """The camera's bands cannot be turned into a view the renderer can look at."""


@dataclass(frozen=True)
class SkyView:
    """Where WHTM1 looks, for the renderer: one elevation per band, and one for the reference.

    `azimuth_deg` is the fit's centre azimuth, `skyline_elevation_deg` the median elevation of
    the skyline over the sky columns, and `scale_px_per_deg` the pixel scale the rows were
    converted with.
    """

    camera_id: str
    azimuth_deg: float
    scale_px_per_deg: float
    skyline_elevation_deg: float
    band_elevations_deg: tuple[float, ...]
    reference_elevation_deg: float

    @property
    def elevations_deg(self) -> tuple[float, ...]:
        """The bands lowest first, then the reference: the order the renderer is asked in."""
        return (*self.band_elevations_deg, self.reference_elevation_deg)


def pixel_scales(fit: CameraFit) -> dict[str, float]:
    """The nominal pixel scale and both ends of the fit's `uncertainty.px_per_deg` range."""
    if fit.uncertainty is None:
        raise ViewError(
            f"the fit of {fit.camera_id} has no uncertainty block (px_per_deg), so there is no "
            "pixel-scale range to run the sensitivity at"
        )
    low, high = fit.uncertainty.px_per_deg
    return {
        "nominal": pixels_per_degree(fit.orientation.hfov_deg, fit.width),
        "low": float(low),
        "high": float(high),
    }


def row_elevation(skyline_deg: float, rows: tuple[int, int], scale_px_per_deg: float) -> float:
    """Elevation of the band of rows `[lo, hi)` above a skyline at `skyline_deg`."""
    low, high = rows
    return skyline_deg + (low + high - 1) / 2.0 / scale_px_per_deg


def sky_columns(fit: CameraFit, contrast: ContrastBands) -> np.ndarray:
    """Boolean per column: every contrast band and the reference lie inside the fit's sky mask."""
    margin = contrast.skyline_margin_rows
    inside = [
        columns_inside(fit, rows, margin_rows=margin)
        for rows in (*contrast.bands_rows, contrast.reference_rows)
    ]
    return np.logical_and.reduce(inside)


def skyline_median_elevation(fit: CameraFit, columns: np.ndarray) -> float:
    """Median, over the chosen columns, of the elevation of the fitted skyline row."""
    chosen = np.flatnonzero(columns)
    if chosen.size == 0:
        raise ViewError("there is no sky column to take the skyline's elevation over")
    o = fit.orientation
    _, elevation = unproject_pixel_to_sky(
        chosen.astype(float),
        skyline_top_rows(fit)[chosen],
        camera_azimuth_deg=o.azimuth_deg,
        camera_elevation_deg=o.elevation_deg,
        roll_deg=o.roll_deg,
        hfov_deg=o.hfov_deg,
        width=fit.width,
        height=fit.height,
    )
    return float(np.median(elevation))


def sky_view(
    fit: CameraFit, contrast: ContrastBands, *, px_per_deg: float | None = None
) -> SkyView:
    """The view of a camera at a pixel scale (default: the fit's nominal one)."""
    columns = sky_columns(fit, contrast)
    if not columns.any():
        raise ViewError(
            f"no sky column of {fit.camera_id} holds every contrast band and the reference"
        )
    scale = (
        pixels_per_degree(fit.orientation.hfov_deg, fit.width) if px_per_deg is None else px_per_deg
    )
    if not scale > 0.0:
        raise ViewError(f"the pixel scale must be positive, got {scale!r}")
    skyline = skyline_median_elevation(fit, columns)
    bands = tuple(row_elevation(skyline, rows, scale) for rows in contrast.bands_rows)
    reference = row_elevation(skyline, contrast.reference_rows, scale)
    if min(bands) < 0.0:
        raise ViewError(
            f"the lowest band of {fit.camera_id} is at {min(bands):.2f} degrees, below the "
            "horizon: the renderer looks at the sky only"
        )
    return SkyView(
        camera_id=fit.camera_id,
        azimuth_deg=fit.orientation.azimuth_deg,
        scale_px_per_deg=scale,
        skyline_elevation_deg=skyline,
        band_elevations_deg=bands,
        reference_elevation_deg=reference,
    )


def relative_azimuth(view_azimuth_deg: float, sun_azimuth_deg: float) -> float:
    """The view's azimuth minus the sun's, wrapped to [-180, 180)."""
    return (view_azimuth_deg - sun_azimuth_deg + 180.0) % 360.0 - 180.0


def sun_azimuths(evening: dt.date, lat: float, lon: float) -> tuple[float, ...]:
    """The sun's azimuth at each sweep elevation on the evening of `evening` (internal use only).

    One value per `geometry.ELEVATION_SWEEP` entry, at the instant the setting sun crosses it.
    """
    crossings = geometry.elevation_crossings(evening, lat, lon)
    return tuple(
        geometry.solar_position(crossings[elevation], lat, lon).azimuth_deg
        for elevation in geometry.ELEVATION_SWEEP
    )
