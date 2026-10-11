"""Finding the skyline in a frame: the strongest sky-to-terrain edge in every column."""

from __future__ import annotations

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, TRUE_MODEL, WIDTH, render_frame, true_skyline_y

from lastlight.camgeom.skyline import (
    Box,
    combine_skylines,
    default_overlay_boxes,
    edge_strength,
    extract_skyline,
)

TRUTH = true_skyline_y(TRUE_MODEL)


def test_the_skyline_of_a_frame_is_found_to_within_a_pixel() -> None:
    found = extract_skyline(render_frame(TRUE_MODEL, seed=1, noise=3.0))
    error = np.abs(found - TRUTH)
    assert found.shape == (WIDTH,)
    assert np.median(error) < 0.4
    assert np.percentile(error, 99) < 1.5


def test_a_straight_edge_between_two_rows_is_located_to_a_tenth_of_a_pixel() -> None:
    edge = 120.3  # rows above are sky, rows below terrain, the row it crosses is blended
    rows = np.arange(HEIGHT, dtype=float)[:, None]
    terrain = np.clip(rows + 0.5 - edge, 0.0, 1.0) * np.ones((1, WIDTH))
    image = 200.0 * (1.0 - terrain) + 60.0 * terrain
    found = extract_skyline(image)
    assert np.max(np.abs(found - edge)) < 0.15


def test_a_grayscale_frame_is_accepted_as_well_as_an_rgb_one() -> None:
    rgb = render_frame(TRUE_MODEL, seed=1, noise=3.0)
    gray = rgb.mean(axis=2)
    assert np.median(np.abs(extract_skyline(gray) - TRUTH)) < 0.5


def test_the_timestamp_overlay_does_not_pull_the_skyline_off_the_horizon() -> None:
    # A bright box under the skyline has a bright-above, dark-below bottom edge that is
    # stronger than the real horizon. Without the exclusion the path rides that edge.
    stamp = Box(300, 175, 560, 205)
    image = render_frame(TRUE_MODEL, seed=2, noise=3.0, stamp=stamp)
    inside = slice(stamp.x0 + 20, stamp.x1 - 20)

    fooled = extract_skyline(image)
    excluded = extract_skyline(image, exclude=[stamp])

    assert np.median(np.abs(fooled - TRUTH)[inside]) > 10.0, "the decoy must really be a decoy"
    assert np.max(np.abs(excluded - TRUTH)[inside]) < 2.0
    assert np.median(np.abs(excluded - TRUTH)) < 0.5


def test_a_stronger_edge_high_in_the_sky_is_ignored_when_the_search_is_banded() -> None:
    image = render_frame(TRUE_MODEL, seed=3, noise=3.0).astype(float)
    image[70:85, 100:700] = 20.0  # a dark cloud: its upper edge outshines the horizon
    image = image.astype(np.uint8)
    middle = slice(150, 650)

    unbanded = extract_skyline(image)
    banded = extract_skyline(image, around=TRUTH + 3.0, half_width=25)

    assert np.median(np.abs(unbanded - TRUTH)[middle]) > 40.0, "the cloud must win without a band"
    assert np.median(np.abs(banded - TRUTH)) < 0.5
    assert np.max(np.abs(banded - TRUTH)) < 2.5


def test_a_band_with_gaps_in_the_prediction_leaves_those_columns_unconstrained() -> None:
    image = render_frame(TRUE_MODEL, seed=4, noise=3.0)
    around = TRUTH.copy()
    around[:50] = np.nan
    found = extract_skyline(image, around=around, half_width=15)
    assert np.all(np.isfinite(found))
    assert np.median(np.abs(found - TRUTH)) < 0.5


def test_a_band_wholly_outside_the_frame_gives_no_skyline_there_and_does_not_crash() -> None:
    # A candidate placement can predict the skyline below the bottom row or above the top row
    # for part of the frame. No row of the frame lies in such a band, so there is nothing to
    # trace: the column is NaN, and the neighbouring columns are traced as before.
    image = render_frame(TRUE_MODEL, seed=4, noise=3.0)
    around = TRUTH.copy()
    around[:60] = HEIGHT + 200.0
    around[60:120] = -80.0
    found = extract_skyline(image, around=around, half_width=30)
    assert found.shape == (WIDTH,)
    assert np.isnan(found[:120]).all()
    assert np.all(np.isfinite(found[120:]))
    assert np.median(np.abs(found - TRUTH)[120:]) < 0.5


def test_edge_strength_is_positive_only_where_brightness_falls_going_down() -> None:
    bright_over_dark = np.zeros((40, 30))
    bright_over_dark[:20] = 200.0
    bright_over_dark[20:] = 50.0
    dark_over_bright = bright_over_dark[::-1]
    assert edge_strength(bright_over_dark)[18:22].max() > 20.0
    assert edge_strength(dark_over_bright).max() == pytest.approx(0.0, abs=1e-9)
    assert edge_strength(bright_over_dark)[:12].max() == pytest.approx(0.0, abs=1e-6)


def test_edge_strength_is_zero_inside_an_excluded_box() -> None:
    image = np.zeros((40, 30))
    image[:20] = 200.0
    image[20:] = 50.0
    strength = edge_strength(image, exclude=[Box(10, 10, 20, 30)])
    assert strength[:, 10:20].max() == 0.0
    assert strength[:, :10].max() > 20.0


def test_combining_skylines_takes_the_median_so_one_bad_frame_is_outvoted() -> None:
    good_a, good_b = np.full(10, 100.0), np.full(10, 101.0)
    bad = np.full(10, 40.0)
    assert combine_skylines([good_a, good_b, bad]) == pytest.approx(np.full(10, 100.0))


def test_combining_skylines_ignores_missing_columns() -> None:
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([3.0, 5.0, np.nan])
    assert combine_skylines([a, b]) == pytest.approx([2.0, 5.0, 3.0])


def test_the_default_overlay_boxes_cover_the_timestamp_corner_at_any_size() -> None:
    (box,) = default_overlay_boxes(800, 300)
    assert box.x0 <= 610 and box.y0 <= 275, "the stamp text starts near column 610, row 277"
    assert (box.x1, box.y1) == (800, 300)
    (big,) = default_overlay_boxes(1600, 600)
    assert (big.x0, big.y0) == (2 * box.x0, 2 * box.y0)


@pytest.mark.parametrize("bad", [(5, 5, 5, 9), (5, 9, 8, 9), (-1, 0, 3, 3)])
def test_a_box_must_have_positive_area_and_lie_in_the_positive_quadrant(
    bad: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        Box(*bad)
