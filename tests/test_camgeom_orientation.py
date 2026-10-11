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
    comb_elevation_deg,
    haze,
    mismatched_ridge_elevation_deg,
    paint_cloud_base,
    render_frame,
    ridge_elevation_deg,
    true_skyline_y,
)

from lastlight.camgeom import orientation as orientation_module
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


def frames_of(model, *, horizon=ridge_elevation_deg, seeds=(11, 12, 13), noise=5.0, edit=None):
    out = []
    for seed in seeds:
        image = render_frame(model, horizon, seed=seed, noise=noise, stamp=OVERLAY[0])
        out.append(edit(image) if edit else image)
    return out


# The fit recovers a clean synthetic camera to about 0.002 degrees and 0.01 percent. The bounds
# are ten times that, not two hundred: a bound far looser than the fit's accuracy lets a stage of
# the fit be skipped without any test noticing (the reviewer of this slice did exactly that).
def assert_recovers(found: Orientation, truth: Orientation) -> None:
    assert found.azimuth_deg == pytest.approx(truth.azimuth_deg, abs=0.02)
    assert found.elevation_deg == pytest.approx(truth.elevation_deg, abs=0.02)
    assert found.hfov_deg == pytest.approx(truth.hfov_deg, rel=0.001)
    assert found.roll_deg == pytest.approx(truth.roll_deg, abs=0.05)


# --- the acceptance criterion: a known camera, noisy frames, the parameters back  -----------------


def test_the_fit_recovers_pointing_and_field_of_view_from_noisy_frames() -> None:
    result = fit_orientation(
        frames_of(TRUE_MODEL), HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY
    )
    assert_recovers(result.orientation, TRUTH)
    assert result.rms_px < 0.15


def test_the_fit_recovers_a_camera_with_a_different_pointing_and_a_narrower_view() -> None:
    model = dataclasses.replace(
        TRUE_MODEL, azimuth_deg=262.5, elevation_deg=-1.0, roll_deg=-1.2, hfov_deg=52.0
    )
    truth = Orientation(262.5, -1.0, -1.2, 52.0)
    result = fit_orientation(frames_of(model), HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY)
    assert_recovers(result.orientation, truth)


def test_the_fit_recovers_a_telephoto_camera_with_a_narrow_view() -> None:
    model = dataclasses.replace(
        TRUE_MODEL, azimuth_deg=281.0, elevation_deg=2.5, roll_deg=0.3, hfov_deg=20.0
    )
    truth = Orientation(281.0, 2.5, 0.3, 20.0)
    result = fit_orientation(frames_of(model), HORIZON, Prior(azimuth_deg=274.0), exclude=OVERLAY)
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


# --- what each stage of the fit is for ------------------------------------------------------------
#
# On a clean scene rendered from the very horizon the fit is given, every stage recovers the truth
# on its own, so no test of that scene can tell whether a stage is there. Each scene below is built
# so that one stage has a job the others cannot do, and each first shows the job is real by
# running the fit with that stage removed.

MIDDLE_MODEL = dataclasses.replace(
    TRUE_MODEL, azimuth_deg=279.3, elevation_deg=1.2, roll_deg=-0.5, hfov_deg=30.0
)
MIDDLE_TRUTH = Orientation(279.3, 1.2, -0.5, 30.0)
MIDDLE_SKYLINE = true_skyline_y(MIDDLE_MODEL)


def test_the_climb_finds_the_true_alignment_of_a_ridge_that_repeats(monkeypatch) -> None:
    # The ridge has a peak every 5.5 degrees, and the true 21.5 degree lens falls between two of
    # the coarse search's field-of-view steps. The coarse search ranks first an alignment two
    # periods to the right with a 6 percent wider lens: the peaks line up as well as the grid
    # allows. Climbing from several of the best candidates is what finds the true alignment.
    model = dataclasses.replace(
        TRUE_MODEL, azimuth_deg=278.0, elevation_deg=1.4, roll_deg=0.3, hfov_deg=21.5
    )
    truth = Orientation(278.0, 1.4, 0.3, 21.5)
    frames = frames_of(model, horizon=comb_elevation_deg)
    horizon = HorizonProfile(AZIMUTHS, comb_elevation_deg(AZIMUTHS))
    prior = Prior(azimuth_deg=274.0)

    with monkeypatch.context() as stage:
        stage.setattr(orientation_module, "_climb", lambda start, *rest: (0.0, start))
        stuck = fit_orientation(frames, horizon, prior, exclude=OVERLAY)
    assert abs(stuck.orientation.azimuth_deg - truth.azimuth_deg) > 5.0, "the decoy must bite"
    assert stuck.rms_px > 3.0

    found = fit_orientation(frames, horizon, prior, exclude=OVERLAY)
    assert_recovers(found.orientation, truth)
    assert found.rms_px < 0.15


def test_a_strong_cloud_edge_in_one_frame_does_not_pull_the_fit_off_a_faint_ridge(
    monkeypatch,
) -> None:
    # In haze the ridge is faint (30 percent of its contrast) and a cloud's dark underside in the
    # first of three frames is nine times stronger. Averaged into the edge map the cloud outweighs
    # the ridge, so the climb on that map settles a third of a degree too low. The skyline traced
    # in each frame and combined by median outvotes the cloud, and the least-squares fit onto it
    # puts the camera back: that second half is what this scene needs.
    rows = MIDDLE_SKYLINE

    def hazy(image: np.ndarray) -> np.ndarray:
        return haze(image, 0.7)

    frames = frames_of(MIDDLE_MODEL, edit=hazy)
    frames[0] = paint_cloud_base(frames[0], rows, offset_start=10, offset_end=4, x0=100, x1=700)
    prior = Prior(azimuth_deg=274.0)

    with monkeypatch.context() as stage:
        stage.setattr(orientation_module, "_fit_to_skyline", lambda start, *rest: start)
        climbed = fit_orientation(frames, HORIZON, prior, exclude=OVERLAY)
    assert abs(climbed.orientation.elevation_deg - MIDDLE_TRUTH.elevation_deg) > 0.1
    assert abs(climbed.orientation.hfov_deg / MIDDLE_TRUTH.hfov_deg - 1.0) > 0.002
    assert climbed.rms_px > 3.0

    found = fit_orientation(frames, HORIZON, prior, exclude=OVERLAY)
    assert_recovers(found.orientation, MIDDLE_TRUTH)
    assert found.rms_px < 0.15


def test_a_cloud_in_every_frame_is_dropped_by_the_narrower_trace_bands(monkeypatch) -> None:
    # A cloud base 22 rows above the ridge, in all three frames, sits inside the first band (30
    # rows) and outside the last two (18 and 12), once the fit has moved off it. With one pass of
    # the wide band the traced skyline follows the cloud over 160 columns and the reported
    # residual is the cloud's, not the ridge's.
    frames = frames_of(MIDDLE_MODEL)
    frames = [
        paint_cloud_base(
            frame,
            MIDDLE_SKYLINE,
            offset_start=22,
            offset_end=22,
            x0=300,
            x1=460,
            shade_below=60.0,
        )
        for frame in frames
    ]
    prior = Prior(azimuth_deg=274.0)

    with monkeypatch.context() as stage:
        stage.setattr(orientation_module, "_BANDS", orientation_module._BANDS[:1])
        one_pass = fit_orientation(frames, HORIZON, prior, exclude=OVERLAY)
    assert one_pass.rms_px > 5.0, "the cloud must be traced by a single wide pass"
    assert abs(one_pass.orientation.azimuth_deg - MIDDLE_TRUTH.azimuth_deg) > 0.05

    found = fit_orientation(frames, HORIZON, prior, exclude=OVERLAY)
    assert_recovers(found.orientation, MIDDLE_TRUTH)
    assert found.rms_px < 0.15


def test_a_slightly_wrong_terrain_model_shows_as_residual_and_barely_moves_the_pointing() -> None:
    # The frames come from a ridge that differs from the horizon the fit is given by a smooth
    # 0.1 degree. No orientation can remove that, so the residual is what it is (about 0.8 px at
    # this lens), and the fitted camera stays close to the true one.
    frames = frames_of(TRUE_MODEL, horizon=mismatched_ridge_elevation_deg)
    result = fit_orientation(frames, HORIZON, Prior(azimuth_deg=270.0), exclude=OVERLAY)

    mismatch = true_skyline_y(TRUE_MODEL, mismatched_ridge_elevation_deg) - predict_skyline(
        TRUTH, HORIZON, WIDTH, HEIGHT
    )
    injected_rms = float(np.sqrt(np.nanmean(mismatch**2)))
    assert 0.6 < injected_rms < 1.0
    assert 0.8 * injected_rms < result.rms_px < 1.05 * injected_rms
    assert result.orientation.azimuth_deg == pytest.approx(TRUTH.azimuth_deg, abs=0.08)
    assert result.orientation.elevation_deg == pytest.approx(TRUTH.elevation_deg, abs=0.03)
    assert result.orientation.hfov_deg == pytest.approx(TRUTH.hfov_deg, rel=0.01)


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
