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


def scene(held_in: tuple[int, ...] = (0, 1, 2)) -> list[np.ndarray]:
    """Three frames of the camera; the cloud edge is painted into those listed in `held_in`."""
    frames = []
    for index, seed in enumerate((11, 12, 13)):
        frame = render_frame(MODEL, ridge_elevation_deg, seed=seed, stamp=OVERLAY[0])
        frames.append(paint_cloud_base(frame, SKYLINE, **CLOUD) if index in held_in else frame)
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
