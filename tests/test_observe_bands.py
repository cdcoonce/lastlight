"""The bands against a frozen fit: does every band lie inside the camera's sky?

`validate_contrast` is the guard the conductor asked for: every band and the reference must lie
inside the sky mask of the frozen fit in at least 95 percent of columns, or the table is wrong
for that fit and `observe` refuses to read a frame with it. A column counts as inside only when
the whole band (every one of its rows) is inside the frame and the mask there.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, WIDTH, example_fit

from lastlight.cameras import ContrastBands, load_registry
from lastlight.camgeom.fitfile import load_fit
from lastlight.camgeom.mask import sky_mask_rows
from lastlight.camgeom.skyline import Box
from lastlight.observe.bands import (
    MIN_COLUMN_COVERAGE,
    ContrastConfigError,
    band_coverage,
    validate_contrast,
)

COLUMNS = tuple(float(x) for x in range(WIDTH))


def flat_fit(top: float = 200.0, **changes):
    """A fit whose skyline sits at row `top` in every one of the 800 columns, with no overlay.

    `changes` replace any field of the fit, `skyline_y_px` and `overlay_boxes` included.
    """
    fields = {
        "skyline_x_px": COLUMNS,
        "skyline_y_px": tuple(top for _ in COLUMNS),
        "overlay_boxes": (),
    }
    return example_fit(**{**fields, **changes})


def bands(**changes) -> ContrastBands:
    fields = {
        "skyline_margin_rows": 8,
        "bands_rows": ((8, 28), (28, 48)),
        "reference_rows": (58, 78),
        "light_dome_columns": (),
    }
    return ContrastBands(**{**fields, **changes})


# --- the sky mask in pixel rows ------------------------------------------------------------------


def test_the_row_mask_is_true_from_the_margin_up_and_false_below_it() -> None:
    mask = sky_mask_rows(flat_fit(top=200.0), margin_rows=8)
    assert mask.shape == (HEIGHT, WIDTH) and mask.dtype == bool
    assert mask[192, 100]  # 8 rows above the skyline: the first row a band may use
    assert not mask[193, 100]  # 7 rows above
    assert not mask[200, 100]  # on the skyline
    assert mask[0, 100]  # the top row of the frame


def test_the_row_mask_follows_the_skyline_column_by_column() -> None:
    fit = flat_fit(top=200.0, skyline_y_px=tuple(100.0 + x / 10.0 for x in COLUMNS))
    mask = sky_mask_rows(fit, margin_rows=0)
    for column in (0, 300, 799):
        top = round(100.0 + column / 10.0)
        assert mask[top, column] and not mask[top + 1, column]  # d = top - row >= 0


def test_the_row_mask_excludes_the_overlay_boxes_and_unseen_skyline_columns() -> None:
    y = [None if x == 400.0 else 200.0 for x in COLUMNS]
    fit = flat_fit(skyline_y_px=tuple(y), overlay_boxes=(Box(600, 0, 700, 50),))
    mask = sky_mask_rows(fit, margin_rows=8)
    assert not mask[:, 400].any()  # a column where the skyline was not seen is never sky
    assert not mask[0:50, 600:700].any()  # the timestamp box
    assert mask[50, 600] and mask[0, 700]  # and only the box


def test_the_row_mask_rejects_a_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        sky_mask_rows(flat_fit(), margin_rows=-1)


# --- coverage ------------------------------------------------------------------------------------


def test_coverage_is_one_where_every_band_lies_inside_the_frame_and_the_mask() -> None:
    coverage = band_coverage(flat_fit(top=200.0), bands())
    assert [c.rows for c in coverage] == [(8, 28), (28, 48), (58, 78)]
    assert [c.is_reference for c in coverage] == [False, False, True]
    assert [c.fraction for c in coverage] == [1.0, 1.0, 1.0]


def test_a_column_where_the_band_leaves_the_top_of_the_frame_is_outside() -> None:
    # With the skyline at row 40 the band [28, 48) would start at row 40 - 47 = -7: off the frame.
    fit = flat_fit(top=200.0, skyline_y_px=tuple(40.0 if x < 100 else 200.0 for x in COLUMNS))
    by_rows = {c.rows: c.fraction for c in band_coverage(fit, bands())}
    assert by_rows[(8, 28)] == 1.0  # rows 40 - 27 = 13 and up: still inside
    assert by_rows[(28, 48)] == pytest.approx(700 / 800)
    assert by_rows[(58, 78)] == pytest.approx(700 / 800)


def test_a_band_is_outside_in_a_column_if_any_one_of_its_rows_is() -> None:
    # The skyline at row 47 puts the band [28, 48) on rows 0 to 19 (offsets 28 to 47 are rows
    # 47 - 47 = 0 and up): inside. At row 46 it needs row -1: outside.
    inside = flat_fit(top=47.0)
    outside = flat_fit(top=46.0)
    assert {c.rows: c.fraction for c in band_coverage(inside, bands())}[(28, 48)] == 1.0
    assert {c.rows: c.fraction for c in band_coverage(outside, bands())}[(28, 48)] == 0.0


def test_the_overlay_box_takes_its_columns_out_of_the_band() -> None:
    fit = flat_fit(top=200.0, overlay_boxes=(Box(0, 0, 100, HEIGHT),))
    fractions = [c.fraction for c in band_coverage(fit, bands())]
    assert fractions == [pytest.approx(700 / 800)] * 3


def test_a_band_that_is_only_partly_inside_the_overlay_box_is_outside_in_those_columns() -> None:
    # The box covers rows 190 to 199 at columns 0 to 99. Band [8, 28) is rows 173 to 192, so it
    # overlaps the box on its last three rows; band [28, 48) is rows 153 to 172 and does not.
    fit = flat_fit(top=200.0, overlay_boxes=(Box(0, 190, 100, 200),))
    by_rows = {c.rows: c.fraction for c in band_coverage(fit, bands())}
    assert by_rows[(8, 28)] == pytest.approx(700 / 800)
    assert by_rows[(28, 48)] == 1.0


# --- the 95 percent rule -------------------------------------------------------------------------


def _fit_with_inside_columns(inside: int):
    """The first `inside` columns hold the skyline low enough for every band; the rest do not."""
    y = tuple(200.0 if x < inside else 20.0 for x in range(WIDTH))
    return flat_fit(skyline_y_px=y)


def test_exactly_95_percent_of_columns_passes() -> None:
    coverage = validate_contrast(_fit_with_inside_columns(760), bands())
    assert min(c.fraction for c in coverage) == 0.95 == MIN_COLUMN_COVERAGE


def test_just_under_95_percent_fails_and_names_the_band_and_the_fraction() -> None:
    with pytest.raises(ContrastConfigError, match=r"\(8, 28\).*94\.9"):
        validate_contrast(_fit_with_inside_columns(759), bands())


def test_the_reference_is_held_to_the_same_rule_and_the_message_says_so() -> None:
    fit = flat_fit(top=60.0)  # reference (58, 78) needs row 60 - 77 < 0 everywhere
    with pytest.raises(ContrastConfigError, match=r"reference.*\(58, 78\).*0\.0"):
        validate_contrast(fit, bands())


def test_a_failure_lists_every_band_that_fails() -> None:
    fit = flat_fit(top=30.0)
    with pytest.raises(ContrastConfigError) as caught:
        validate_contrast(fit, bands())
    message = str(caught.value)
    assert "(28, 48)" in message and "reference" in message


def test_light_dome_columns_must_lie_inside_the_frame() -> None:
    with pytest.raises(ContrastConfigError, match=r"light_dome_columns.*800"):
        validate_contrast(flat_fit(), bands(light_dome_columns=((530, 801),)))


def test_a_light_dome_that_fits_the_frame_is_accepted() -> None:
    validate_contrast(flat_fit(), bands(light_dome_columns=((530, 800),)))


# --- the shipped tables against the shipped, frozen fits ----------------------------------------


def test_every_shipped_contrast_table_lies_inside_its_frozen_fit() -> None:
    checked = []
    for camera in load_registry().values():
        if camera.contrast is None:
            continue
        coverage = validate_contrast(load_fit(camera.id), camera.contrast)
        assert all(c.fraction >= MIN_COLUMN_COVERAGE for c in coverage), camera.id
        checked.append(camera.id)
    assert checked == ["WHTM1"], "the check must actually run on WHTM1"


def test_whtm1_reference_was_not_moved_from_the_conductors_rows() -> None:
    # The conductor's rule: move the reference down in 10-row steps only if the 95 percent rule
    # needs it. At the frozen fit it does not, so the seeded rows stand.
    contrast = load_registry()["WHTM1"].contrast
    assert contrast is not None and contrast.reference_rows == (118, 138)
    coverage = band_coverage(load_fit("WHTM1"), contrast)
    assert coverage[-1].is_reference and coverage[-1].fraction >= MIN_COLUMN_COVERAGE


def test_the_next_reference_step_up_would_fail_on_whtm1_so_the_check_has_teeth() -> None:
    # Positive control: the rule is not vacuous on the real fit. A reference 50 rows higher
    # than the seeded one runs off the top of the frame in many columns.
    contrast = dataclasses.replace(load_registry()["WHTM1"].contrast, reference_rows=(168, 188))
    with pytest.raises(ContrastConfigError, match="reference"):
        validate_contrast(load_fit("WHTM1"), contrast)


def test_coverage_arrays_are_per_column_booleans_of_the_frame_width() -> None:
    from lastlight.observe.bands import columns_inside

    inside = columns_inside(flat_fit(top=200.0), (8, 28), margin_rows=8)
    assert inside.shape == (WIDTH,) and inside.dtype == bool and inside.all()
    assert isinstance(inside, np.ndarray)
