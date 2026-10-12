"""A cloud edge held in every frame must not capture the least-squares stage of the fit.

The fit climbs the edge strength, then moves the four parameters onto the skyline traced in each
frame (stage 3). A cloud base that sits 15 to 25 rows above the ridge over a stretch near the
edge of the frame, in every frame, is a stronger edge than the ridge and is a continuous one, so
the trace follows it there and least squares drags a correct climb off the ridge. The guard
(`LeastSquaresGuard`) notices that the fitted skyline scores worse on the edge maps than the
climbed one and refits without the columns that lie far from the climb.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import (
    HEIGHT,
    REF_LAT,
    REF_LON,
    TRUE_MODEL,
    WIDTH,
    haze,
    mismatched_ridge_elevation_deg,
    paint_cloud_base,
    render_frame,
    ridge_dem,
    ridge_elevation_deg,
    true_skyline_y,
)
from PIL import Image

from lastlight.camgeom import orientation as orientation_module
from lastlight.camgeom.calibration import calibrate_camera
from lastlight.camgeom.dem import horizon_profile
from lastlight.camgeom.fitfile import load_fit, write_fit
from lastlight.camgeom.orientation import (
    HorizonProfile,
    Orientation,
    Prior,
    SkylineFit,
    fit_orientation,
)
from lastlight.camgeom.sites import CameraSite
from lastlight.camgeom.skyline import default_overlay_boxes, edge_strength

AZIMUTHS = np.arange(190.0, 360.0, 0.05)
HORIZON = HorizonProfile(AZIMUTHS, ridge_elevation_deg(AZIMUTHS))
OVERLAY = default_overlay_boxes(WIDTH, HEIGHT)
PRIOR = Prior(azimuth_deg=274.0)

# The camera of the neighbouring stage scenes (a 30 degree lens), with its ridge in rows 97 to 144.
MODEL = dataclasses.replace(
    TRUE_MODEL, azimuth_deg=279.3, elevation_deg=1.2, roll_deg=-0.5, hfov_deg=30.0
)
TRUTH = Orientation(279.3, 1.2, -0.5, 30.0)
SKYLINE = true_skyline_y(MODEL)

# The persistent edge: a cloud base 20 rows above the ridge over the last 200 columns, in the
# frames named. On the code before the guard it moved the fit by 0.39 degrees of azimuth, 1.6
# degrees of roll and 6.3 percent of the field of view, with the climb exact to 0.001 degrees.
CLOUD = {"offset_start": 20, "offset_end": 20, "x0": 600, "x1": 800, "shade_below": 60.0}


def scene(
    held_in: tuple[int, ...] = (0, 1, 2), cloud: dict[str, float] = CLOUD
) -> list[np.ndarray]:
    """Three frames of the camera; the cloud edge is painted into those listed in `held_in`."""
    frames = []
    for index, seed in enumerate((11, 12, 13)):
        frame = render_frame(MODEL, ridge_elevation_deg, seed=seed, stamp=OVERLAY[0])
        frames.append(paint_cloud_base(frame, SKYLINE, **cloud) if index in held_in else frame)
    return frames


def mismatched_scene(cloud: dict[str, float] | None = None) -> list[np.ndarray]:
    """Three frames from a ridge that differs from the horizon the fit is given by 0.1 degree.

    With `cloud`, its edge is painted into every frame, along the mismatched ridge.
    """
    rows = true_skyline_y(MODEL, mismatched_ridge_elevation_deg)
    frames = []
    for seed in (11, 12, 13):
        frame = render_frame(MODEL, mismatched_ridge_elevation_deg, seed=seed, stamp=OVERLAY[0])
        frames.append(paint_cloud_base(frame, rows, **cloud) if cloud else frame)
    return frames


def hazy_scene_with_a_strong_cloud_in_one_frame() -> list[np.ndarray]:
    """Haze (30 percent of the contrast) and, in the first frame only, a dark cloud underside."""
    frames = [haze(frame, 0.7) for frame in scene(held_in=())]
    frames[0] = paint_cloud_base(frames[0], SKYLINE, offset_start=10, offset_end=4, x0=100, x1=700)
    return frames


def fit(frames: list[np.ndarray]) -> SkylineFit:
    return fit_orientation(frames, HORIZON, PRIOR, exclude=OVERLAY)


# The tolerances of the recovery tests in test_camgeom_orientation.py, ten times what the fit
# achieves on a clean scene: the acceptance criterion names them.
def assert_recovers(found: Orientation, truth: Orientation) -> None:
    assert found.azimuth_deg == pytest.approx(truth.azimuth_deg, abs=0.02)
    assert found.elevation_deg == pytest.approx(truth.elevation_deg, abs=0.02)
    assert found.hfov_deg == pytest.approx(truth.hfov_deg, rel=0.001)
    assert found.roll_deg == pytest.approx(truth.roll_deg, abs=0.05)


@pytest.fixture(scope="module")
def clean() -> SkylineFit:
    return fit(scene(held_in=()))


@pytest.fixture(scope="module")
def persistent() -> SkylineFit:
    return fit(scene())


# --- the acceptance criterion ---------------------------------------------------------------------


def test_a_cloud_edge_held_in_every_frame_near_the_frames_edge_does_not_capture_the_fit(
    persistent: SkylineFit,
) -> None:
    assert_recovers(persistent.orientation, TRUTH)
    assert persistent.rms_px < 0.15, "the residual is the ridge's, not the cloud's"


def test_the_persistent_edge_does_capture_least_squares_when_the_guard_cannot_refuse_it(
    monkeypatch,
) -> None:
    # The scene must bite: with a margin that no loss of edge strength can exceed, the guard never
    # fires and the stage-3 result stands, 20 times the azimuth tolerance and 60 times that of the
    # field of view away from the truth.
    monkeypatch.setattr(orientation_module, "_CAPTURE_MARGIN", 1.0)
    captured = fit(scene())
    assert abs(captured.orientation.azimuth_deg - TRUTH.azimuth_deg) > 0.3
    assert abs(captured.orientation.roll_deg - TRUTH.roll_deg) > 1.0
    assert abs(captured.orientation.hfov_deg / TRUTH.hfov_deg - 1.0) > 0.05
    assert captured.guard is not None and captured.guard.outcome == "accepted"


# --- what the guard decides, and the record it leaves ---------------------------------------------


def test_the_guard_accepts_least_squares_where_nothing_has_captured_it(clean: SkylineFit) -> None:
    assert_recovers(clean.orientation, TRUTH)
    guard = clean.guard
    assert guard is not None
    assert guard.outcome == "accepted" and guard.refused is False
    assert guard.columns_dropped == 0 and guard.refit_energy is None
    assert guard.frames == 3 and guard.frames_lowered == 0
    assert guard.fitted_energy == pytest.approx(guard.climbed_energy, rel=0.001)
    assert guard.climbed_energy > 10.0, "a clean ridge scores its own edge strength"


def test_the_guard_refits_without_the_captured_columns_and_says_so(persistent: SkylineFit) -> None:
    guard = persistent.guard
    assert guard is not None
    assert guard.outcome == "refit" and guard.refused is True
    assert guard.frames == 3 and guard.frames_lowered == 3
    # least squares lost about 40 percent of the climb's edge strength; the refit lost none
    assert guard.fitted_energy < 0.7 * guard.climbed_energy
    assert guard.refit_energy is not None
    assert guard.refit_energy == pytest.approx(guard.climbed_energy, rel=0.01)
    # the cloud covers the last 200 columns, and the path leaves the ridge a few columns before it
    assert 180 <= guard.columns_dropped <= 230
    # the last, narrow pass no longer reaches the cloud, so the reported residual spans the frame
    assert persistent.n_columns >= WIDTH - 20


def test_the_climb_is_kept_when_the_refit_is_refused_too(monkeypatch) -> None:
    # With the drop distance beyond the frame no column is dropped, the refit is the captured
    # result again and is refused; the climb, which was right, is what the fit returns.
    monkeypatch.setattr(orientation_module, "_CAPTURE_DROP_PX", 1e6)
    kept = fit(scene())
    guard = kept.guard
    assert guard is not None
    assert guard.outcome == "kept_climb" and guard.refused is True and guard.columns_dropped == 0
    assert guard.refit_energy is not None and guard.refit_energy < 0.7 * guard.climbed_energy
    assert_recovers(kept.orientation, TRUTH)


def test_the_recorded_energies_are_scores_on_the_mean_of_the_edge_maps(
    clean: SkylineFit, persistent: SkylineFit
) -> None:
    # The same score the climb maximises (`_energy` on the mean of the frames' edge maps), so the
    # numbers in the fit file can be compared with the coarse search's and the climb's.
    for result, held_in, key in (
        (clean, (), "fitted_energy"),
        (persistent, (0, 1, 2), "refit_energy"),
    ):
        mean = np.mean([edge_strength(frame, OVERLAY) for frame in scene(held_in)], axis=0)
        predicted = orientation_module.predict_skyline(result.orientation, HORIZON, WIDTH, HEIGHT)
        recorded = getattr(result.guard, key)
        assert recorded == pytest.approx(orientation_module._energy(mean, predicted), abs=1e-4)


@pytest.mark.parametrize(
    ("held_in", "outcome"),
    [((0,), "accepted"), ((0, 1), "refit")],
    ids=["one of three frames", "two of three frames"],
)
def test_the_guard_votes_by_frame_as_the_median_of_the_traces_does(
    held_in: tuple[int, ...], outcome: str
) -> None:
    # One frame's cloud is outvoted by the median trace, so least squares is right and nothing is
    # refused. Two frames' cloud outvote the ridge, so the fit is captured, and the ridge, which
    # is in all three frames, scores worse for the fit in every one of them.
    found = fit(scene(held_in))
    assert_recovers(found.orientation, TRUTH)
    assert found.guard is not None and found.guard.outcome == outcome


def test_a_frame_that_shows_no_ridge_does_not_stop_the_majority_refusing() -> None:
    # The vote is a majority, not unanimity: two frames hold the cloud and the third is a flat
    # grey (fog, or a frame the exposure lost). The fitted skyline scores lower in the two frames
    # that hold an edge and cannot score lower in the one that holds none.
    frames = scene(held_in=(0, 1))
    frames[2] = np.full_like(frames[2], 140)
    found = fit(frames)
    assert_recovers(found.orientation, TRUTH)
    guard = found.guard
    assert guard is not None and guard.frames == 3 and guard.frames_lowered == 2
    assert guard.outcome == "refit"


START = Orientation(280.0, 1.0, 0.0, 30.0)  # any three distinct orientations will do
FITTED = Orientation(281.0, 2.0, 1.0, 32.0)
REFIT = Orientation(280.5, 1.5, 0.5, 30.5)


@pytest.mark.parametrize(
    ("fitted_votes", "refit_votes", "outcome"),
    [
        (3, 0, "refit"),
        (3, 1, "refit"),
        (2, 1, "refit"),
        (3, 2, "kept_climb"),
        (2, 2, "kept_climb"),
        (3, 3, "kept_climb"),
        (1, 0, "accepted"),
    ],
)
def test_the_refit_is_judged_by_the_same_majority_as_stage_3(
    monkeypatch: pytest.MonkeyPatch, fitted_votes: int, refit_votes: int, outcome: str
) -> None:
    # No scene of the suite has the refit lose in a minority of the frames or in a bare majority
    # (it loses in none, or in all three when the drop distance is made huge), so the vote on the
    # refit is stubbed here: three frames, and for the fitted skyline and then for the refit the
    # number of frames that score it lower than the climb. Two of three refuse; one does not.
    votes = iter([fitted_votes, refit_votes])
    monkeypatch.setattr(orientation_module, "_frames_lowered", lambda *a, **k: next(votes))
    runs = iter([(FITTED, np.zeros(8), 0), (REFIT, np.zeros(8), 5)])
    monkeypatch.setattr(orientation_module, "_trace_and_fit", lambda *a, **k: next(runs))
    monkeypatch.setattr(orientation_module, "_recorded_energy", lambda *a, **k: 1.0)
    monkeypatch.setattr(orientation_module, "_CAPTURE_MARGIN", 0.37)
    result, _, guard = orientation_module._stage_3(
        START, [np.zeros((4, 4))] * 3, np.zeros((4, 4)), None, None
    )
    assert guard.outcome == outcome
    assert guard.margin == 0.37, "the record carries the margin the vote used"
    assert result is {"accepted": FITTED, "refit": REFIT, "kept_climb": START}[outcome]


def test_a_cloud_edge_in_one_frame_is_outvoted_even_when_the_mean_map_says_the_fit_lost() -> None:
    # In haze (30 percent of the contrast) a cloud's dark underside in one frame is nine times
    # stronger than the ridge. The climb on the mean of the edge maps is drawn to it, and the
    # median of the traces puts the camera back on the ridge. On the mean map the fitted skyline
    # then scores less than half of the climbed one (4.0 against 11.6), but that is the one
    # frame with the cloud: in the other two the fitted skyline scores twenty times higher, and
    # the guard, which votes by frame, accepts the least-squares result.
    found = fit(hazy_scene_with_a_strong_cloud_in_one_frame())
    assert_recovers(found.orientation, TRUTH)
    guard = found.guard
    assert guard is not None and guard.outcome == "accepted"
    assert guard.frames_lowered == 1 and guard.frames == 3
    assert guard.fitted_energy < 0.5 * guard.climbed_energy


def test_the_refit_keeps_the_columns_a_slightly_wrong_terrain_model_only_misplaces() -> None:
    # The frames come from a ridge that differs from the horizon the fit is given by about
    # 0.1 degree, so honest ridge columns lie a pixel or two from the climbed skyline. The refit
    # drops the cloud's columns (about 200) and none of those, and ends where the fit ends
    # without the cloud.
    without_cloud, with_cloud = fit(mismatched_scene()), fit(mismatched_scene(CLOUD))
    assert without_cloud.guard is not None and without_cloud.guard.outcome == "accepted"
    guard = with_cloud.guard
    assert guard is not None and guard.outcome == "refit"
    assert 180 <= guard.columns_dropped <= 260
    assert_recovers(with_cloud.orientation, without_cloud.orientation)


# --- how far the margin sits from the scenes it separates -----------------------------------------
#
# A skyline "loses" in a frame by the share of the climbed score it falls short by, and the guard
# refuses stage 3 when more than half the frames lose more than the margin. The losses of the
# scenes below are measured numbers that a few percent of noise or a slightly different cloud can
# move, so a margin that only just separates the scenes the suite has would give way to the first
# scene that differs from them. These tests hold it a quarter of itself away from both sides.
HEADROOM = 1.25

# A capture by 150 columns of cloud at the right edge, 20 rows above a ridge the model has exactly.
SHORT_CLOUD = {**CLOUD, "x0": 650}


def fit_with_vote(
    monkeypatch: pytest.MonkeyPatch, frames: list[np.ndarray]
) -> tuple[SkylineFit, float]:
    """Fit `frames`; also return the loss that decides stage 3's vote on them.

    The loss of the fitted skyline in a frame is `1 - fitted / climbed` on that frame's edge map,
    the quantity the vote compares with the margin. Stage 3 is refused when more than half the
    frames lose more than the margin, which is when the middle one of the losses, sorted (the
    lower middle one when the frames are even in number), exceeds it. That one is returned.
    """
    votes = []
    vote = orientation_module._frames_lowered

    def spy(*args):
        votes.append(args)
        return vote(*args)

    monkeypatch.setattr(orientation_module, "_frames_lowered", spy)
    found = fit(frames)
    strengths, climbed, fitted, horizon, _margin = votes[0]  # the first vote is stage 3's
    height, width = strengths[0].shape
    kept = orientation_module.predict_skyline(climbed, horizon, width, height)
    moved = orientation_module.predict_skyline(fitted, horizon, width, height)
    losses = sorted(
        1.0 - orientation_module._energy(s, moved) / orientation_module._energy(s, kept)
        for s in strengths
    )
    return found, losses[(len(losses) - 1) // 2]


@pytest.mark.parametrize(
    "frames_of",
    [
        pytest.param(lambda: mismatched_scene(CLOUD), id="200 columns, terrain 0.1 degree off"),
        pytest.param(lambda: scene(cloud=SHORT_CLOUD), id="150 columns, exact terrain"),
    ],
)
def test_the_margin_has_headroom_below_the_loss_of_a_realistic_capture(
    monkeypatch: pytest.MonkeyPatch, frames_of
) -> None:
    # Real terrain is never exact (the real camera's residual is 1.6 px), so a capture on real
    # frames loses what these two lose, about 21 percent, and not the 33 to 44 percent of a
    # capture on exact terrain (the persistent-edge scene). A margin near that loss is one
    # evening's noise from letting a capture through.
    found, decisive = fit_with_vote(monkeypatch, frames_of())
    margin = orientation_module._CAPTURE_MARGIN
    guard = found.guard
    assert guard is not None and guard.outcome == "refit"
    assert decisive >= HEADROOM * margin, f"loses {decisive:.3f} against a margin of {margin}"
    mean_map_loss = 1.0 - guard.fitted_energy / guard.climbed_energy
    assert mean_map_loss >= HEADROOM * margin, f"loses {mean_map_loss:.3f} on the mean map"


# A cloud base 22 rows above the ridge over 160 columns of every frame: the scene of
# test_a_cloud_in_every_frame_is_dropped_by_the_narrower_trace_bands (test_camgeom_orientation.py).
BANDS_CLOUD = {"offset_start": 22, "offset_end": 22, "x0": 300, "x1": 460, "shade_below": 60.0}


@pytest.mark.parametrize(
    ("frames_of", "one_wide_pass"),
    [
        pytest.param(lambda: scene(held_in=()), False, id="clean frames"),
        pytest.param(lambda: scene(held_in=(0,)), False, id="a cloud in one of three frames"),
        pytest.param(hazy_scene_with_a_strong_cloud_in_one_frame, False, id="haze and one cloud"),
        pytest.param(mismatched_scene, False, id="terrain 0.1 degree off"),
        pytest.param(lambda: scene(cloud=BANDS_CLOUD), True, id="one wide pass and a cloud"),
    ],
)
def test_the_margin_has_headroom_above_the_loss_of_the_scenes_the_guard_must_accept(
    monkeypatch: pytest.MonkeyPatch, frames_of, one_wide_pass: bool
) -> None:
    # The scenes the suite requires the guard to leave alone, scored as the vote scores them. The
    # last is the one that sets the floor: one pass of the wide band followed by a cloud held in
    # every frame loses 11.5 percent per frame, and the narrowing bands are what repair it, so a
    # guard that refused it would leave no scene for the bands to be tested on. The others lose
    # at most 0.1 percent, or gain (the haze scene's two frames without the cloud). The rest of
    # the recovery scenes of test_camgeom_orientation.py lose 0.12 percent at most (the build log
    # of the guard, docs/log/2026-10-11-camgeom-capture.md).
    if one_wide_pass:
        monkeypatch.setattr(orientation_module, "_BANDS", orientation_module._BANDS[:1])
    found, decisive = fit_with_vote(monkeypatch, frames_of())
    margin = orientation_module._CAPTURE_MARGIN
    assert found.guard is not None and found.guard.outcome == "accepted"
    assert HEADROOM * decisive <= margin, f"loses {decisive:.3f} against a margin of {margin}"


# --- the record reaches the fit file --------------------------------------------------------------


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory):
    """`calibrate_camera` on three clean synthetic frames and the synthetic terrain."""
    root = tmp_path_factory.mktemp("capture-archive") / "SYN1" / "2026-10-10"
    root.mkdir(parents=True)
    azimuths = np.arange(180.0, 360.0, 0.05)
    elevation = horizon_profile(ridge_dem(), REF_LAT, REF_LON, 310.0, azimuths)
    when = dt.datetime(2026, 10, 11, 0, 30, tzinfo=dt.UTC)
    for index in range(3):
        image = render_frame(
            TRUE_MODEL,
            lambda az: np.interp(az, azimuths, elevation),
            seed=40 + index,
            stamp=OVERLAY[0],
        )
        stamp = (when + dt.timedelta(minutes=5 * index)).strftime("%Y%m%dT%H%M%SZ")
        Image.fromarray(image).save(root / f"{stamp}.jpg", quality=92)
    site = CameraSite("SYN1", REF_LAT, REF_LON, view_azimuth_deg=272.0, mount_height_m=10.0)
    return calibrate_camera("SYN1", site, root.parent, ridge_dem(), frozen_on=dt.date(2026, 10, 10))


def test_a_calibration_carries_the_guard_record_and_write_fit_writes_it(
    calibrated, tmp_path: Path
) -> None:
    guard = calibrated.fit.least_squares_guard
    assert guard is not None and guard == calibrated.skyline.guard
    assert guard.outcome == "accepted"
    path = write_fit(calibrated.fit, tmp_path / "SYN1.json")
    assert '"least_squares_guard"' in path.read_text()
    assert load_fit("SYN1", directory=tmp_path).least_squares_guard == guard
