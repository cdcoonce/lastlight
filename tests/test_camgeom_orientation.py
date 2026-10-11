"""Fitting azimuth, elevation, roll and field of view by matching skylines."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from camgeom_synthetic import (
    HEIGHT,
    TRUE_MODEL,
    WIDTH,
    render_frame,
    ridge_elevation_deg,
    true_skyline_y,
)
from lastlight.camgeom.orientation import (
    HorizonProfile,
    Orientation,
    Prior,
    fit_orientation,
    fit_radial_term,
    predict_skyline,
)
from lastlight.camgeom.skyline import Box, default_overlay_boxes

AZIMUTHS = np.arange(190.0, 360.0, 0.05)
HORIZON = HorizonProfile(AZIMUTHS, ridge_elevation_deg(AZIMUTHS))
OVERLAY = default_overlay_boxes(WIDTH, HEIGHT)
TRUTH = Orientation(
    TRUE_MODEL.azimuth_deg, TRUE_MODEL.elevation_deg, TRUE_MODEL.roll_deg, TRUE_MODEL.hfov_deg
)


def frames_of(model, *, seeds=(11, 12, 13), noise=5.0, edit=None):
    out = []
    for seed in seeds:
        image = render_frame(model, seed=seed, noise=noise, stamp=OVERLAY[0])
        out.append(edit(image) if edit else image)
    return out


def assert_recovers(found: Orientation, truth: Orientation) -> None:
    assert found.azimuth_deg == pytest.approx(truth.azimuth_deg, abs=0.2)
    assert found.elevation_deg == pytest.approx(truth.elevation_deg, abs=0.2)
    assert found.hfov_deg == pytest.approx(truth.hfov_deg, rel=0.01)
    assert found.roll_deg == pytest.approx(truth.roll_deg, abs=0.5)


# --- the acceptance criterion: a known camera, noisy frames, the parameters back  -----------------


def test_the_fit_recovers_pointing_and_field_of_view_from_noisy_frames() -> None:
    result = fit_orientation(
        frames_of(TRUE_MODEL), HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY
    )
    assert_recovers(result.orientation, TRUTH)
    assert result.rms_px < 1.0


def test_the_fit_recovers_a_camera_with_a_different_pointing_and_a_narrower_view() -> None:
    model = dataclasses.replace(
        TRUE_MODEL, azimuth_deg=262.5, elevation_deg=-1.0, roll_deg=-1.2, hfov_deg=52.0
    )
    truth = Orientation(262.5, -1.0, -1.2, 52.0)
    result = fit_orientation(frames_of(model), HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY)
    assert_recovers(result.orientation, truth)


def test_a_dark_cloud_edge_in_the_sky_does_not_capture_the_fit() -> None:
    def add_cloud(image: np.ndarray) -> np.ndarray:
        image = image.copy()
        image[70:85, 100:700] = 55
        return image

    result = fit_orientation(
        frames_of(TRUE_MODEL, edit=add_cloud), HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY
    )
    assert_recovers(result.orientation, TRUTH)


def test_excluded_regions_do_not_take_part_in_the_fit() -> None:
    # A bright ghost of the skyline, 25 to 45 rows lower, has a bottom edge stronger than the
    # real horizon. A fit that let it count would lock onto it and read the camera 4 degrees
    # too high; excluding the box that holds it must give the right answer.
    rows = true_skyline_y(TRUE_MODEL)

    def add_ghost(image: np.ndarray) -> np.ndarray:
        image = image.copy()
        for column in range(WIDTH):
            top = int(round(rows[column])) + 25
            image[top : top + 20, column] = 255
        return image

    ghost_box = Box(0, 165, WIDTH, 230)
    frames = frames_of(TRUE_MODEL, edit=add_ghost)
    prior = Prior(azimuth_deg=270.0)

    fooled = fit_orientation(frames, HORIZON, prior, exclude=OVERLAY)
    guarded = fit_orientation(frames, HORIZON, prior, exclude=[*OVERLAY, ghost_box])

    assert abs(fooled.orientation.elevation_deg - TRUTH.elevation_deg) > 1.5, "the ghost must bite"
    assert_recovers(guarded.orientation, TRUTH)


# --- how the residuals are reported ---------------------------------------------------------------


def test_the_predicted_skyline_matches_the_camera_model_and_the_horizon() -> None:
    predicted = predict_skyline(TRUTH, HORIZON, WIDTH, HEIGHT)
    assert predicted.shape == (WIDTH,)
    assert np.max(np.abs(predicted - true_skyline_y(TRUE_MODEL))) < 0.05


def test_columns_the_horizon_does_not_cover_are_nan_not_extrapolated() -> None:
    narrow = HorizonProfile(np.arange(268.0, 276.0, 0.05), np.full(160, 2.0))
    predicted = predict_skyline(TRUTH, narrow, WIDTH, HEIGHT)
    assert np.isnan(predicted[:50]).all()
    assert np.isfinite(predicted[400])


def test_residuals_come_in_pixels_and_in_degrees_at_the_fitted_scale() -> None:
    clean = [render_frame(TRUE_MODEL, seed=1, noise=0.0, stamp=OVERLAY[0])]
    result = fit_orientation(clean, HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY)
    focal = (WIDTH / 2) / math.tan(math.radians(result.orientation.hfov_deg) / 2)
    assert result.rms_px < 0.6
    assert result.rms_deg == pytest.approx(result.rms_px / (focal * math.pi / 180), rel=1e-9)
    assert 0.0 <= result.median_abs_px <= result.rms_px * 1.5
    assert 700 <= result.n_columns <= WIDTH
    assert result.observed.shape == result.predicted.shape == (WIDTH,)


def test_a_fit_refuses_no_frames_and_frames_of_different_sizes() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        fit_orientation([], HORIZON, Prior(azimuth_deg=270.0))
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same size"):
        fit_orientation([frames_of(TRUE_MODEL)[0], small], HORIZON, Prior(azimuth_deg=270.0))


# --- the optional radial term --------------------------------------------------------------------


def test_a_distortion_free_skyline_gets_no_radial_term() -> None:
    observed = predict_skyline(TRUTH, HORIZON, WIDTH, HEIGHT)
    term = fit_radial_term(observed, HORIZON, TRUTH, WIDTH, HEIGHT)
    assert abs(term.k1) < 0.01
    assert term.rms_px < 0.2


def test_a_skyline_seen_through_a_distorting_lens_gets_its_radial_term_back() -> None:
    observed = predict_skyline(TRUTH, HORIZON, WIDTH, HEIGHT, radial_k1=0.2)
    term = fit_radial_term(observed, HORIZON, TRUTH, WIDTH, HEIGHT)
    assert term.k1 == pytest.approx(0.2, abs=0.02)
    assert term.rms_px < 0.3


def test_radial_distortion_moves_the_corners_of_the_frame_and_not_the_centre() -> None:
    plain = predict_skyline(TRUTH, HORIZON, WIDTH, HEIGHT)
    bent = predict_skyline(TRUTH, HORIZON, WIDTH, HEIGHT, radial_k1=0.2)
    assert abs(bent[400] - plain[400]) < abs(bent[20] - plain[20])


# --- the horizon profile  -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("azimuths", "elevations"),
    [
        ([1.0, 2.0, 2.0], [0.0, 0.0, 0.0]),  # not strictly increasing
        ([1.0, 2.0], [0.0]),  # lengths differ
        ([1.0, 2.0, 3.0], [0.0, math.nan, 0.0]),  # a hole
    ],
)
def test_a_horizon_profile_must_be_increasing_complete_and_matched(azimuths, elevations) -> None:
    with pytest.raises(ValueError):
        HorizonProfile(np.array(azimuths), np.array(elevations))


def test_a_horizon_profile_can_be_built_from_samples_with_holes_by_dropping_them() -> None:
    azimuths = np.array([1.0, 2.0, 3.0, 4.0])
    elevations = np.array([0.5, math.nan, 0.7, 0.8])
    profile = HorizonProfile.from_samples(azimuths, elevations)
    assert profile.azimuth_deg.tolist() == [1.0, 3.0, 4.0]
    assert profile.elevation_deg.tolist() == [0.5, 0.7, 0.8]
