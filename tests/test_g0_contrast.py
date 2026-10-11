"""The rendered within-frame contrast, the quantity `lastlight.observe` measures (issue #10).

The renderer's XYZ becomes linear sRGB (the webcam proxy), `log(R/G)` and `log(B/G)` are taken at
each band's elevation and at the reference's, and the contrast is band minus reference. The
guards: a gain on the light cancels (as a camera's exposure does), a channel outside the sRGB
gamut is floored rather than logged, and, end to end, a frame painted with the rendered sky and
read by `contrast_profile` gives the contrasts this module computes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, WIDTH
from observe_support import BANDS, flat_fit, srgb_encode

from lastlight.g0.contrast import (
    CHANNEL_FLOOR,
    day_contrasts,
    log_chromaticity,
    rendered_contrasts,
)
from lastlight.g0.view import SkyView, sky_view
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.observe.contrast import (
    DARK_LEVEL,
    SATURATION_LEVEL,
    contrast_profile,
    srgb_to_linear,
)
from lastlight.render import Atmosphere
from lastlight.render.colorimetry import xyz_to_linear_srgb
from lastlight.render.curve import sky_xyz

ATMOSPHERE = Atmosphere(aod_550=0.11, ozone_du=300.0)

# sRGB (D65) matrix from linear RGB to XYZ, IEC 61966-2-1, written out here as the oracle.
RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def xyz_of(rgb) -> np.ndarray:
    return RGB_TO_XYZ @ np.asarray(rgb, dtype=float)


# --- XYZ to linear sRGB -----------------------------------------------------------------------


def test_linear_srgb_inverts_the_iec_matrix() -> None:
    for rgb in ([1.0, 1.0, 1.0], [0.8, 0.3, 0.1], [0.05, 0.4, 0.9]):
        got = xyz_to_linear_srgb(xyz_of(rgb))
        np.testing.assert_allclose(got, rgb, atol=2e-4)


def test_linear_srgb_is_not_clipped_and_not_gamma_encoded() -> None:
    out_of_gamut = xyz_to_linear_srgb(xyz_of([1.0, 0.1, -0.05]))
    assert out_of_gamut[2] < 0.0  # a negative channel is kept; the caller decides what to do
    half = xyz_to_linear_srgb(xyz_of([0.5, 0.5, 0.5]))
    np.testing.assert_allclose(half, 0.5, atol=2e-4)  # linear light, not an encoded 0.735


def test_linear_srgb_takes_a_stack_of_xyz() -> None:
    stack = np.array([xyz_of([0.2, 0.4, 0.6]), xyz_of([0.9, 0.5, 0.1])])
    got = xyz_to_linear_srgb(stack)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got[1], [0.9, 0.5, 0.1], atol=2e-4)


# --- log chromaticity ---------------------------------------------------------------------------


def test_log_chromaticity_is_log_r_over_g_and_log_b_over_g() -> None:
    xyz = np.array([xyz_of([0.6, 0.3, 0.15]), xyz_of([0.2, 0.4, 0.8])])
    logs, floored = log_chromaticity(xyz)
    np.testing.assert_allclose(logs[:, 0], [math.log(2.0), math.log(0.5)], atol=2e-3)
    np.testing.assert_allclose(logs[:, 1], [math.log(0.5), math.log(2.0)], atol=2e-3)
    assert floored == 0


def test_log_chromaticity_does_not_move_when_the_light_is_scaled() -> None:
    xyz = np.array([xyz_of([0.6, 0.3, 0.15])])
    np.testing.assert_allclose(
        log_chromaticity(xyz * 7.3)[0], log_chromaticity(xyz)[0], rtol=0, atol=1e-12
    )


def test_a_grey_has_zero_log_chromaticity() -> None:
    logs, _ = log_chromaticity(np.array([xyz_of([0.4, 0.4, 0.4])]))
    np.testing.assert_allclose(logs, 0.0, atol=2e-3)


def test_the_floor_is_the_widest_channel_ratio_an_8_bit_frame_can_show() -> None:
    """Code 6 against code 249: the darkest and brightest channels `observe` still counts."""
    assert DARK_LEVEL == 5 and SATURATION_LEVEL == 250
    low, high = srgb_to_linear(np.array([DARK_LEVEL + 1, SATURATION_LEVEL - 1]))
    assert pytest.approx(low / high, rel=1e-12) == CHANNEL_FLOOR
    assert 0.0018 < CHANNEL_FLOOR < 0.0020


def test_a_channel_outside_the_gamut_is_floored_not_logged() -> None:
    xyz = np.array([xyz_of([1.0, 0.2, -0.02])])  # blue is negative: a deep orange
    logs, floored = log_chromaticity(xyz)
    assert floored == 1
    assert logs[0, 0] == pytest.approx(math.log(1.0 / 0.2), abs=3e-3)
    assert logs[0, 1] == pytest.approx(math.log(CHANNEL_FLOOR / 0.2), abs=3e-3)
    assert np.isfinite(logs).all()


def test_a_channel_below_the_floor_is_raised_to_it_and_counted_per_channel() -> None:
    xyz = np.array([xyz_of([1.0, 0.0005, -0.3]), xyz_of([0.5, 0.5, 0.5])])
    logs, floored = log_chromaticity(xyz)
    assert floored == 2  # green and blue of the first pixel; the second pixel is clean
    assert np.isfinite(logs).all()
    # R/G is now R over the floored green: 1 / (CHANNEL_FLOOR * 1).
    assert logs[0, 0] == pytest.approx(-math.log(CHANNEL_FLOOR), abs=2e-3)


def test_a_pixel_with_no_light_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no light"):
        log_chromaticity(np.array([[0.0, 0.0, 0.0]]))


# --- the rendered contrast of a day -------------------------------------------------------------


def _view(elevations=(1.1, 1.6, 2.1), reference=3.6) -> SkyView:
    return SkyView(
        camera_id="SYN",
        azimuth_deg=289.0,
        scale_px_per_deg=44.0,
        skyline_elevation_deg=0.7,
        band_elevations_deg=tuple(elevations),
        reference_elevation_deg=reference,
    )


def test_the_contrast_is_each_band_minus_the_reference() -> None:
    view = _view()
    contrast, floored = rendered_contrasts(ATMOSPHERE, 3.0, view, 34.0)
    assert contrast.shape == (3, 2)
    xyz = sky_xyz(ATMOSPHERE, 3.0, (*view.band_elevations_deg, view.reference_elevation_deg), 34.0)
    logs, expected_floor = log_chromaticity(xyz)
    np.testing.assert_allclose(contrast, logs[:3] - logs[3], rtol=0, atol=1e-12)
    assert floored == expected_floor


def test_the_contrast_is_zero_when_the_band_is_the_reference() -> None:
    view = _view(elevations=(3.6, 3.6, 3.6), reference=3.6)
    contrast, _ = rendered_contrasts(ATMOSPHERE, -2.0, view, 34.0)
    np.testing.assert_allclose(contrast, 0.0, atol=1e-12)


def test_the_relative_azimuth_changes_the_contrast() -> None:
    view = _view()
    near, _ = rendered_contrasts(ATMOSPHERE, 3.0, view, 8.0)
    far, _ = rendered_contrasts(ATMOSPHERE, 3.0, view, 70.0)
    assert not np.allclose(near, far)


def test_a_day_is_one_contrast_per_sweep_elevation() -> None:
    view = _view()
    azimuths = tuple(34.0 - 0.1 * i for i in range(len(ELEVATION_SWEEP)))
    day, floored = day_contrasts(ATMOSPHERE, view, azimuths)
    assert day.shape == (29, 3, 2)
    one, _ = rendered_contrasts(ATMOSPHERE, ELEVATION_SWEEP[5], view, azimuths[5])
    np.testing.assert_array_equal(day[5], one)
    assert floored >= 0


def test_a_day_needs_one_azimuth_per_sweep_elevation() -> None:
    with pytest.raises(ValueError, match="29"):
        day_contrasts(ATMOSPHERE, _view(), (34.0,) * 28)


# --- the same quantity `observe` measures -------------------------------------------------------


def test_a_frame_painted_with_the_rendered_sky_reads_back_as_the_rendered_contrast() -> None:
    """End to end: paint rows above a flat skyline with the rendered colour at each row's
    elevation, run the real `contrast_profile`, and compare with this module's contrasts."""
    fit = flat_fit(top=150.0)
    view = sky_view(fit, BANDS)
    sun, rel_azimuth = 12.0, 70.0
    atmosphere = Atmosphere(aod_550=0.08, ozone_du=280.0)

    rows = np.arange(HEIGHT)
    offset_above = 150 - rows  # the skyline row is 150 in every column
    elevation = view.skyline_elevation_deg + np.maximum(offset_above, 0) / view.scale_px_per_deg
    xyz = sky_xyz(atmosphere, sun, tuple(float(e) for e in elevation), rel_azimuth)
    linear = np.clip(xyz_to_linear_srgb(xyz), 1e-6, None)
    linear = 0.4 * linear / linear.max()  # a mid exposure: nothing saturated, nothing too dark
    terrain = np.array([0.05, 0.03, 0.02])
    column = np.where((offset_above >= 0)[:, None], linear, terrain)
    frame = np.repeat(srgb_encode(column)[:, None, :], WIDTH, axis=1)
    assert frame.max() < SATURATION_LEVEL and frame[offset_above >= 8].min() > DARK_LEVEL + 1

    observed = contrast_profile(frame, fit, BANDS, sun_elevation_deg=0.0)
    rendered, floored = rendered_contrasts(atmosphere, sun, view, rel_azimuth)
    assert floored == 0
    for band, expected in zip(observed.bands, rendered, strict=True):
        assert band.missing is None
        assert band.log_rg == pytest.approx(expected[0], abs=1e-3)
        assert band.log_bg == pytest.approx(expected[1], abs=1e-3)
