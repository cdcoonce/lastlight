"""The pinhole camera model: pixel to sky direction and back."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from camgeom_synthetic import TRUE_MODEL
from lastlight.camgeom.model import CameraModel


def variant(**changes: float) -> CameraModel:
    return dataclasses.replace(TRUE_MODEL, **changes)


LEVEL = variant(azimuth_deg=90.0, elevation_deg=0.0, roll_deg=0.0, hfov_deg=60.0)
CENTRE = ((LEVEL.width - 1) / 2, (LEVEL.height - 1) / 2)

MODELS = [
    pytest.param(TRUE_MODEL, id="typical"),
    pytest.param(variant(roll_deg=-3.0, elevation_deg=-8.0, hfov_deg=45.0), id="rolled-down"),
    pytest.param(variant(azimuth_deg=2.0, elevation_deg=35.0, hfov_deg=90.0), id="north-high"),
    pytest.param(variant(azimuth_deg=359.0, roll_deg=7.0, hfov_deg=20.0), id="wraps-north"),
]


# --- the acceptance criterion: a round trip within 0.01 pixel ---------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_pixel_to_sky_and_back_round_trips_within_a_hundredth_of_a_pixel(
    model: CameraModel,
) -> None:
    xs, ys = np.meshgrid(np.linspace(-60.0, 860.0, 47), np.linspace(-40.0, 340.0, 31))
    az, el = model.pixel_to_sky(xs, ys)
    x_back, y_back = model.sky_to_pixel(az, el)
    assert np.max(np.abs(x_back - xs)) < 0.01
    assert np.max(np.abs(y_back - ys)) < 0.01


@pytest.mark.parametrize("model", MODELS)
def test_sky_to_pixel_and_back_round_trips_for_directions_in_front(model: CameraModel) -> None:
    az = model.azimuth_deg + np.linspace(-25.0, 25.0, 21)[:, None]
    el = model.elevation_deg + np.linspace(-10.0, 10.0, 15)[None, :]
    az, el = np.broadcast_arrays(az, el)
    x, y = model.sky_to_pixel(az, el)
    az_back, el_back = model.pixel_to_sky(x, y)
    # Azimuth is compared as an angle so the wrap at north does not matter.
    assert np.max(np.abs((az_back - az + 180.0) % 360.0 - 180.0)) < 1e-6
    assert np.max(np.abs(el_back - el)) < 1e-6


# --- what the parameters mean ---------------------------------------------------------------------


@pytest.mark.parametrize("roll", [0.0, 5.0, -12.0])
def test_the_image_centre_looks_along_the_optical_axis(roll: float) -> None:
    model = variant(roll_deg=roll)
    az, el = model.pixel_to_sky((model.width - 1) / 2, (model.height - 1) / 2)
    assert az == pytest.approx(model.azimuth_deg, abs=1e-9)
    assert el == pytest.approx(model.elevation_deg, abs=1e-9)


def test_azimuth_grows_to_the_right_and_elevation_grows_up_the_image() -> None:
    left, _ = LEVEL.pixel_to_sky(CENTRE[0] - 100, CENTRE[1])
    right, _ = LEVEL.pixel_to_sky(CENTRE[0] + 100, CENTRE[1])
    _, above = LEVEL.pixel_to_sky(CENTRE[0], CENTRE[1] - 100)
    _, below = LEVEL.pixel_to_sky(CENTRE[0], CENTRE[1] + 100)
    assert left < LEVEL.azimuth_deg < right
    assert above > LEVEL.elevation_deg > below


def test_hfov_is_the_azimuth_between_the_left_and_right_edges_of_the_image() -> None:
    mid_row = CENTRE[1]
    left, _ = LEVEL.pixel_to_sky(-0.5, mid_row)
    right, _ = LEVEL.pixel_to_sky(LEVEL.width - 0.5, mid_row)
    assert right - left == pytest.approx(LEVEL.hfov_deg, abs=1e-9)


def test_vfov_follows_from_hfov_and_the_aspect_ratio() -> None:
    expected = 2.0 * math.degrees(
        math.atan(math.tan(math.radians(LEVEL.hfov_deg) / 2.0) * LEVEL.height / LEVEL.width)
    )
    assert LEVEL.vfov_deg == pytest.approx(expected, abs=1e-9)
    _, top = LEVEL.pixel_to_sky(CENTRE[0], -0.5)
    _, bottom = LEVEL.pixel_to_sky(CENTRE[0], LEVEL.height - 0.5)
    assert top - bottom == pytest.approx(LEVEL.vfov_deg, abs=1e-9)


def test_focal_length_in_pixels_matches_the_half_width_over_tan_half_hfov() -> None:
    expected = (LEVEL.width / 2) / math.tan(math.radians(LEVEL.hfov_deg) / 2)
    assert LEVEL.focal_px == pytest.approx(expected, rel=1e-12)


def test_positive_roll_raises_the_horizon_toward_the_right_of_the_image() -> None:
    rolled = dataclasses.replace(LEVEL, roll_deg=3.0)
    az = LEVEL.azimuth_deg + np.array([-20.0, 20.0])
    _, y_level = LEVEL.sky_to_pixel(az, np.zeros(2))
    _, y_rolled = rolled.sky_to_pixel(az, np.zeros(2))
    assert y_level[0] == pytest.approx(y_level[1], abs=0.5)  # level camera: flat horizon
    assert y_rolled[1] < y_rolled[0] - 5.0, "the right end must sit higher (smaller row)"


def test_the_horizon_of_a_rolled_camera_tilts_by_the_roll_angle() -> None:
    rolled = dataclasses.replace(LEVEL, roll_deg=4.0, hfov_deg=30.0)
    az = LEVEL.azimuth_deg + np.array([-3.0, 3.0])
    x, y = rolled.sky_to_pixel(az, np.zeros(2))
    tilt = math.degrees(math.atan2(y[0] - y[1], x[1] - x[0]))
    assert tilt == pytest.approx(4.0, abs=0.05)


def test_azimuth_is_returned_in_zero_to_360_and_wraps_at_north() -> None:
    model = variant(azimuth_deg=359.5, roll_deg=0.0, elevation_deg=0.0)
    az, _ = model.pixel_to_sky(
        np.array([(model.width - 1) / 2, model.width - 1.0]), np.full(2, 149.5)
    )
    assert np.all((az >= 0.0) & (az < 360.0))
    assert az[1] < 90.0, "to the right of a camera looking just west of north is east of north"


def test_directions_behind_the_camera_have_no_pixel() -> None:
    x, y = TRUE_MODEL.sky_to_pixel(TRUE_MODEL.azimuth_deg + 180.0, 0.0)
    assert math.isnan(x) and math.isnan(y)
    xs, ys = TRUE_MODEL.sky_to_pixel(
        np.array([TRUE_MODEL.azimuth_deg, 10.0]), np.array([2.0, -80.0])
    )
    assert np.isfinite(xs[0]) and np.isnan(xs[1])


def test_scalar_inputs_give_plain_floats_and_array_inputs_give_arrays() -> None:
    az, el = TRUE_MODEL.pixel_to_sky(400.0, 150.0)
    assert type(az) is float and type(el) is float
    az_a, el_a = TRUE_MODEL.pixel_to_sky(np.array([400.0, 10.0]), np.array([150.0, 20.0]))
    assert az_a.shape == (2,) and el_a.shape == (2,)


@pytest.mark.parametrize(
    "changes",
    [
        {"hfov_deg": 0.0},
        {"hfov_deg": 180.0},
        {"hfov_deg": math.nan},
        {"width": 0},
        {"height": -1},
        {"elevation_deg": 91.0},
        {"azimuth_deg": math.inf},
    ],
)
def test_nonsense_parameters_are_refused(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        variant(**changes)


def test_the_model_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TRUE_MODEL.azimuth_deg = 1.0  # type: ignore[misc]
