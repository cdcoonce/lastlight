"""WHTM1's view for the G0 gate: rows above the skyline to elevations, and the sun's azimuth.

Pins conductor decisions 4 and 5 of issue #10: bands are mapped from pixel rows to view
elevation with the frozen fit's nominal pixel scale and the skyline's median elevation over the
sky columns; the relative azimuth is the fit's centre azimuth minus the sun's.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, WIDTH, example_fit
from observe_support import BANDS, flat_fit

from lastlight import geometry
from lastlight.cameras import load_registry
from lastlight.camgeom.fitfile import load_fit
from lastlight.camgeom.model import pixels_per_degree
from lastlight.camgeom.orientation import Orientation
from lastlight.g0.view import (
    SkyView,
    ViewError,
    pixel_scales,
    relative_azimuth,
    row_elevation,
    sky_columns,
    sky_view,
    skyline_median_elevation,
    sun_azimuths,
)
from lastlight.site import REFERENCE_POINT

LAT, LON, _ = REFERENCE_POINT


# --- rows to elevation --------------------------------------------------------------------------


def test_a_band_sits_at_its_centre_row_over_the_pixel_scale_above_the_skyline() -> None:
    # Rows 8 to 27 above the skyline have their centre at 17.5 rows; at 40 px per degree that is
    # 0.4375 degrees above a skyline at 0.7.
    assert row_elevation(0.7, (8, 28), 40.0) == pytest.approx(0.7 + 17.5 / 40.0, abs=1e-12)
    assert row_elevation(0.7, (118, 138), 40.0) == pytest.approx(0.7 + 127.5 / 40.0, abs=1e-12)


def test_a_smaller_pixel_scale_puts_the_same_rows_higher() -> None:
    assert row_elevation(0.7, (88, 108), 27.5) > row_elevation(0.7, (88, 108), 44.9)


def test_the_nominal_scale_is_the_fits_own_and_the_ends_are_its_uncertainty() -> None:
    fit = example_fit()
    scales = pixel_scales(fit)
    assert scales["nominal"] == pytest.approx(pixels_per_degree(66.0, WIDTH), abs=1e-12)
    assert (scales["low"], scales["high"]) == (10.3, 12.4)


def test_a_fit_without_an_uncertainty_has_no_sensitivity_ends() -> None:
    with pytest.raises(ViewError, match="uncertainty"):
        pixel_scales(example_fit(uncertainty=None))


# --- the sky columns and the skyline's median elevation ------------------------------------------


def test_every_column_is_a_sky_column_on_a_flat_skyline_with_room_above() -> None:
    columns = sky_columns(flat_fit(top=150.0), BANDS)
    assert columns.shape == (WIDTH,)
    assert columns.all()


def test_a_column_whose_reference_band_leaves_the_frame_is_not_a_sky_column() -> None:
    # Rows 118 to 137 above a skyline at row 140 are above the top of the frame at the right.
    xs = tuple(float(x) for x in range(WIDTH))
    ys = tuple(140.0 if x < 400 else 100.0 for x in xs)
    columns = sky_columns(flat_fit(skyline_y_px=ys), BANDS)
    assert columns[:400].all()
    assert not columns[400:].any()


def test_no_sky_column_at_all_is_an_error() -> None:
    with pytest.raises(ViewError, match="sky column"):
        sky_view(flat_fit(top=100.0), BANDS)


def test_the_skyline_elevation_is_the_median_over_the_sky_columns() -> None:
    """Zero roll and zero tilt: the closed form el = -atan(v / sqrt(1 + u^2)) per column."""
    orientation = Orientation(280.0, 0.0, 0.0, 40.0)
    fit = flat_fit(top=170.0, orientation=orientation)
    focal = (WIDTH / 2.0) / math.tan(math.radians(20.0))
    u = (np.arange(WIDTH) - (WIDTH - 1) / 2.0) / focal
    v = (170.0 - (HEIGHT - 1) / 2.0) / focal
    expected = np.median(np.degrees(-np.arctan(v / np.sqrt(1.0 + u**2))))
    got = skyline_median_elevation(fit, np.ones(WIDTH, dtype=bool))
    assert got == pytest.approx(float(expected), abs=1e-9)


def test_only_the_sky_columns_count_toward_the_median() -> None:
    fit = flat_fit(top=170.0, orientation=Orientation(280.0, 0.0, 0.0, 40.0))
    everywhere = skyline_median_elevation(fit, np.ones(WIDTH, dtype=bool))
    left_edge = np.zeros(WIDTH, dtype=bool)
    left_edge[:50] = True
    assert skyline_median_elevation(fit, left_edge) != everywhere


# --- the assembled view --------------------------------------------------------------------------


def test_a_view_lists_the_band_elevations_lowest_first_and_the_reference_above_them() -> None:
    fit = flat_fit(top=150.0)
    view = sky_view(fit, BANDS)
    assert isinstance(view, SkyView)
    assert len(view.band_elevations_deg) == 5
    assert list(view.band_elevations_deg) == sorted(view.band_elevations_deg)
    assert view.reference_elevation_deg > view.band_elevations_deg[-1]
    scale = pixels_per_degree(66.0, WIDTH)
    assert view.scale_px_per_deg == pytest.approx(scale)
    for (lo, hi), got in zip(BANDS.bands_rows, view.band_elevations_deg, strict=True):
        assert got == pytest.approx(view.skyline_elevation_deg + (lo + hi - 1) / 2.0 / scale)
    assert view.azimuth_deg == pytest.approx(274.0)


def test_the_sensitivity_view_keeps_the_skyline_and_changes_only_the_scale() -> None:
    fit = flat_fit(top=150.0)
    nominal = sky_view(fit, BANDS)
    low = sky_view(fit, BANDS, px_per_deg=pixel_scales(fit)["low"])
    assert low.skyline_elevation_deg == nominal.skyline_elevation_deg
    assert low.azimuth_deg == nominal.azimuth_deg
    assert low.scale_px_per_deg == 10.3
    assert low.reference_elevation_deg > nominal.reference_elevation_deg


def test_a_view_that_would_look_below_the_horizon_is_refused() -> None:
    fit = flat_fit(top=250.0, orientation=Orientation(274.0, -3.0, 0.0, 66.0))
    with pytest.raises(ViewError, match="below the horizon"):
        sky_view(fit, BANDS)


def test_the_shipped_whtm1_fit_and_registry_give_a_low_view_just_above_the_ridge() -> None:
    fit = load_fit("WHTM1")
    contrast = load_registry()["WHTM1"].contrast
    assert contrast is not None
    view = sky_view(fit, contrast)
    assert view.camera_id == "WHTM1"
    assert view.azimuth_deg == pytest.approx(fit.orientation.azimuth_deg)
    # Frozen fit (provisional): skyline a little under one degree up, nominal scale about 44 px/deg.
    assert 0.2 < view.skyline_elevation_deg < 1.3
    assert 40.0 < view.scale_px_per_deg < 48.0
    assert view.band_elevations_deg[0] > view.skyline_elevation_deg
    assert 0.0 < view.band_elevations_deg[0] < view.band_elevations_deg[-1] < 4.0
    assert view.band_elevations_deg[-1] < view.reference_elevation_deg < 6.0
    low, high = pixel_scales(fit)["low"], pixel_scales(fit)["high"]
    assert (low, high) == tuple(fit.uncertainty.px_per_deg)  # type: ignore[union-attr]
    assert sky_view(fit, contrast, px_per_deg=low).reference_elevation_deg > 4.5


# --- relative azimuth ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("view", "sun", "expected"),
    [
        (289.29, 255.0, 34.29),
        (255.0, 289.29, -34.29),
        (10.0, 350.0, 20.0),
        (350.0, 10.0, -20.0),
        (180.0, 0.0, -180.0),
        (270.0, 270.0, 0.0),
    ],
)
def test_the_relative_azimuth_is_the_view_minus_the_sun_wrapped_to_half_a_turn(
    view: float, sun: float, expected: float
) -> None:
    assert relative_azimuth(view, sun) == pytest.approx(expected, abs=1e-9)


# --- the sun's azimuth at each sweep elevation --------------------------------------------------


def test_the_sun_azimuth_is_computed_at_each_sweep_elevation_of_the_evening() -> None:
    evening = dt.date(2026, 10, 10)
    got = sun_azimuths(evening, LAT, LON)
    assert len(got) == len(geometry.ELEVATION_SWEEP) == 29
    crossings = geometry.elevation_crossings(evening, LAT, LON)
    for elevation, azimuth in zip(geometry.ELEVATION_SWEEP, got, strict=True):
        sun = geometry.solar_position(crossings[elevation], LAT, LON)
        assert azimuth == pytest.approx(sun.azimuth_deg, abs=1e-9)
        assert sun.elevation_deg == pytest.approx(elevation, abs=0.01)


def test_the_sun_moves_north_along_the_horizon_through_the_evening() -> None:
    got = sun_azimuths(dt.date(2026, 10, 10), LAT, LON)
    assert list(got) == sorted(got)
    assert 245.0 < got[0] < 265.0  # an October evening, the sun sets southwest of west
    assert got[-1] - got[0] < 15.0


def test_the_june_sun_sets_next_to_the_cameras_field_of_view() -> None:
    """Why the relative azimuth matters: from autumn to midsummer it moves from 34 to 10 degrees."""
    june = sun_azimuths(dt.date(2026, 6, 21), LAT, LON)
    october = sun_azimuths(dt.date(2026, 10, 10), LAT, LON)
    assert 295.0 < june[14] < 303.0
    assert abs(relative_azimuth(289.29, june[14])) < 15.0
    assert abs(relative_azimuth(289.29, october[14])) > 25.0
