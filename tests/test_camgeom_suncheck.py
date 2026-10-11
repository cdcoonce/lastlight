"""The sun check: does the sun's predicted pixel land on the sun's observed disk?"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math

import numpy as np
import pytest
from camgeom_synthetic import REF_LAT, REF_LON, TRUE_MODEL, paint_disk, paint_glow, render_frame
from lastlight.camgeom.suncheck import locate_sun_disk, predict_sun_pixel, run_sun_check

from lastlight.geometry import solar_position

WHEN = dt.datetime(2026, 10, 11, 0, 30, tzinfo=dt.UTC)
LATER = WHEN + dt.timedelta(minutes=10)
NIGHT = dt.datetime(2026, 10, 11, 4, 0, tzinfo=dt.UTC)
SUN = solar_position(WHEN, REF_LAT, REF_LON)
# A camera aimed so that the sun sits right of centre and a little above it.
AIMED = dataclasses.replace(
    TRUE_MODEL,
    azimuth_deg=SUN.azimuth_deg - 8.0,
    elevation_deg=SUN.elevation_deg - 2.0,
    roll_deg=0.0,
)
PIXELS_PER_DEGREE = AIMED.focal_px * math.pi / 180.0


def sky_frame(seed: int = 0) -> np.ndarray:
    return render_frame(AIMED, seed=seed, noise=2.0)


def where_the_sun_should_be(when: dt.datetime = WHEN) -> tuple[float, float]:
    prediction = predict_sun_pixel(AIMED, when)
    assert prediction is not None
    return prediction.x, prediction.y


# --- predicting the sun's pixel  ------------------------------------------------------------------


def test_the_predicted_pixel_looks_at_the_sun() -> None:
    prediction = predict_sun_pixel(AIMED, WHEN)
    assert prediction is not None
    az, el = AIMED.pixel_to_sky(prediction.x, prediction.y)
    assert az == pytest.approx(SUN.azimuth_deg, abs=1e-6)
    assert el == pytest.approx(SUN.elevation_deg, abs=1e-6)
    assert prediction.sun_elevation_deg == pytest.approx(SUN.elevation_deg, abs=1e-9)
    assert 0 <= prediction.x < AIMED.width and 0 <= prediction.y < AIMED.height


def test_a_sun_behind_the_camera_has_no_predicted_pixel() -> None:
    away = dataclasses.replace(AIMED, azimuth_deg=(SUN.azimuth_deg + 180.0) % 360.0)
    assert predict_sun_pixel(away, WHEN) is None


def test_the_prediction_follows_the_camera_position_through_the_solar_position() -> None:
    moved = dataclasses.replace(AIMED, lon=AIMED.lon + 2.0)  # 8 minutes of solar time
    base = predict_sun_pixel(AIMED, WHEN)
    other = predict_sun_pixel(moved, WHEN)
    assert base is not None and other is not None
    assert math.hypot(base.x - other.x, base.y - other.y) > 10.0


# --- finding the disk  ----------------------------------------------------------------------------


def test_the_disk_centroid_is_found_to_a_third_of_a_pixel() -> None:
    image = paint_disk(sky_frame(), 412.3, 118.6)
    found = locate_sun_disk(image, near=(410.0, 120.0), radius_px=60.0)
    assert found is not None
    assert math.hypot(found.x - 412.3, found.y - 118.6) < 0.35
    assert 20 <= found.n_pixels <= 60


def test_a_glow_with_no_saturated_disk_is_not_a_sun() -> None:
    image = paint_glow(sky_frame(), 412.0, 118.0)
    assert locate_sun_disk(image, near=(410.0, 120.0), radius_px=60.0) is None


def test_an_overexposed_patch_is_not_a_sun() -> None:
    image = sky_frame().copy()
    image[80:150, 380:450] = 255
    assert locate_sun_disk(image, near=(410.0, 120.0), radius_px=60.0) is None


def test_a_disk_outside_the_search_radius_is_not_found() -> None:
    image = paint_disk(sky_frame(), 600.0, 60.0)
    assert locate_sun_disk(image, near=(410.0, 120.0), radius_px=60.0) is None


def test_the_disk_nearest_the_prediction_wins() -> None:
    image = paint_disk(paint_disk(sky_frame(), 300.0, 100.0), 420.0, 130.0)
    found = locate_sun_disk(image, near=(415.0, 128.0), radius_px=200.0)
    assert found is not None
    assert math.hypot(found.x - 420.0, found.y - 130.0) < 0.5


# --- the check itself  ----------------------------------------------------------------------------


def frame_with_disk_at(when: dt.datetime, *, east_deg: float = 0.0, seed: int = 0) -> np.ndarray:
    x, y = where_the_sun_should_be(when)
    return paint_disk(sky_frame(seed), x + east_deg * PIXELS_PER_DEGREE, y)


def test_the_check_passes_when_the_disk_is_where_the_sun_should_be() -> None:
    frames = [(frame_with_disk_at(WHEN, east_deg=0.2), WHEN), (frame_with_disk_at(LATER), LATER)]
    result = run_sun_check(AIMED, frames, tolerance_deg=1.0)
    assert result.status == "passed"
    assert [sample.status for sample in result.samples] == ["match", "match"]
    assert result.max_error_deg < 0.5
    assert "2" in result.reason


def test_the_check_fails_when_the_disk_is_too_far_from_the_prediction() -> None:
    frames = [(frame_with_disk_at(WHEN, east_deg=3.0), WHEN)]
    result = run_sun_check(AIMED, frames, tolerance_deg=1.0)
    assert result.status == "failed"
    assert result.samples[0].status == "mismatch"
    assert result.samples[0].error_deg == pytest.approx(3.0, abs=0.2)


def test_the_tolerance_is_an_angle_not_a_pixel_count() -> None:
    frames = [(frame_with_disk_at(WHEN, east_deg=0.9), WHEN)]
    assert run_sun_check(AIMED, frames, tolerance_deg=1.0).status == "passed"
    assert run_sun_check(AIMED, frames, tolerance_deg=0.5).status == "failed"


def test_a_sun_hidden_by_cloud_skips_the_check_and_says_why() -> None:
    x, y = where_the_sun_should_be()
    frames = [(paint_glow(sky_frame(), x, y), WHEN)]
    result = run_sun_check(AIMED, frames, tolerance_deg=1.0)
    assert result.status == "skipped"
    assert "no saturated disk" in result.reason
    assert result.samples[0].status == "no_disk"
    assert result.samples[0].error_deg is None


def test_a_sun_outside_the_frame_skips_the_check_and_says_why() -> None:
    wide_of_the_sun = dataclasses.replace(AIMED, azimuth_deg=SUN.azimuth_deg - 60.0)
    frames = [(render_frame(wide_of_the_sun, seed=1), WHEN)]
    result = run_sun_check(wide_of_the_sun, frames, tolerance_deg=1.0)
    assert result.status == "skipped"
    assert "out of frame" in result.reason
    assert result.samples[0].status == "out_of_frame"


def test_a_frame_taken_after_the_sun_has_set_is_not_asked_for_a_disk() -> None:
    result = run_sun_check(AIMED, [(sky_frame(), NIGHT)], tolerance_deg=1.0)
    assert result.status == "skipped"
    assert result.samples[0].status == "below_horizon"
    assert "below the horizon" in result.reason


def test_frames_without_a_disk_do_not_veto_a_frame_that_matches() -> None:
    x, y = where_the_sun_should_be(WHEN)
    frames = [(paint_glow(sky_frame(), x, y), WHEN), (frame_with_disk_at(LATER), LATER)]
    result = run_sun_check(AIMED, frames, tolerance_deg=1.0)
    assert result.status == "passed"
    assert [sample.status for sample in result.samples] == ["no_disk", "match"]


def test_a_check_over_no_frames_is_skipped() -> None:
    result = run_sun_check(AIMED, [], tolerance_deg=1.0)
    assert result.status == "skipped"
    assert result.reason


def test_the_json_summary_has_numbers_and_statuses_but_no_times() -> None:
    frames = [(frame_with_disk_at(WHEN, east_deg=0.2), WHEN), (sky_frame(), NIGHT)]
    summary = run_sun_check(AIMED, frames, tolerance_deg=1.0).to_json()
    text = json.dumps(summary)
    assert summary["status"] == "passed"
    assert summary["tolerance_deg"] == 1.0
    assert summary["frames_examined"] == 2
    assert summary["detected"] == 1
    assert [s["status"] for s in summary["samples"]] == ["match", "below_horizon"]
    assert all("sun_elevation_deg" in s for s in summary["samples"])
    assert "2026" not in text and "T0" not in text
