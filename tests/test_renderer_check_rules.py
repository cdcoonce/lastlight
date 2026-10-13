"""The pure rules of `docs/gates/renderer-v1/criterion.md`: ratio, cloud filter and verdict.

Every definition the criterion fixes is pinned here on plain numbers, so the tests need no
rendering and no files.
"""

from __future__ import annotations

import math

import pytest

from lastlight.renderer_check import rules

# --- r_e: a ratio of two medians of absolute values --------------------------------------------


def test_the_ratio_is_the_median_absolute_rendered_over_the_median_absolute_observed() -> None:
    assert rules.evening_ratio([2.0, -4.0, 6.0], [-1.0, -2.0, 3.0]) == pytest.approx(2.0)


def test_the_ratio_is_a_ratio_of_medians_not_a_median_of_ratios() -> None:
    rendered, observed = [1.0, 2.0, 3.0], [1.0, 10.0, 1.0]
    per_frame = sorted(r / o for r, o in zip(rendered, observed, strict=True))
    assert per_frame[1] == pytest.approx(1.0)  # what a median of ratios would say
    assert rules.evening_ratio(rendered, observed) == pytest.approx(2.0)  # 2 over 1


def test_the_absolute_value_is_taken_before_the_median() -> None:
    # Medians of the signed values are 2 and -1 (ratio -2); of the absolute ones 4 and 2.
    assert rules.evening_ratio([2.0, -4.0, 6.0], [-1.0, -2.0, 3.0]) > 0.0


def test_an_even_number_of_frames_takes_the_mean_of_the_middle_two() -> None:
    # |rendered| medians: (2 + 4) / 2 = 3; |observed| medians: (1 + 1) / 2 = 1.
    assert rules.evening_ratio([1.0, 2.0, 4.0, 8.0], [1.0, 1.0, 1.0, 1.0]) == pytest.approx(3.0)


def test_a_zero_denominator_is_positive_infinity() -> None:
    assert rules.evening_ratio([0.5, 0.7, 0.9], [0.0, 0.0, 0.0]) == math.inf


def test_zero_over_zero_is_still_positive_infinity() -> None:
    assert rules.evening_ratio([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == math.inf


def test_a_denominator_median_of_zero_is_infinity_even_when_some_frames_are_not_zero() -> None:
    assert rules.evening_ratio([1.0, 1.0, 1.0], [0.0, 0.0, 0.4]) == math.inf


def test_a_zero_numerator_over_a_positive_denominator_is_zero() -> None:
    assert rules.evening_ratio([0.0, 0.0, 0.0], [0.2, 0.2, 0.2]) == 0.0


def test_frame_counts_must_match_and_not_be_empty() -> None:
    with pytest.raises(ValueError):
        rules.evening_ratio([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        rules.evening_ratio([], [])


# --- the cloud filter ----------------------------------------------------------------------------


def test_the_cloud_mean_is_the_arithmetic_mean_of_the_values() -> None:
    assert rules.cloud_mean([0.0, 10.0, 20.0, 30.0]) == pytest.approx(15.0)


def test_exactly_ten_percent_counts() -> None:
    values = [17.0, 3.0] + [10.0] * 15  # 17 points whose mean is exactly 10
    assert rules.cloud_mean(values) == 10.0
    assert rules.is_clear(rules.cloud_mean(values)) is True


def test_a_hair_over_ten_percent_does_not_count() -> None:
    assert rules.is_clear(10.0) is True
    assert rules.is_clear(math.nextafter(10.0, 11.0)) is False
    assert rules.is_clear(10.01) is False
    assert rules.is_clear(0.0) is True


def test_seventeen_tens_have_a_mean_of_exactly_ten() -> None:
    assert rules.cloud_mean([10.0] * 17) == 10.0


# --- the verdict ---------------------------------------------------------------------------------


def test_the_three_verdict_strings_are_exactly_these() -> None:
    assert rules.ADOPT == "ADOPT"
    assert rules.KEEP_V0 == "KEEP V0"
    assert rules.not_yet_decidable(3) == "NOT YET DECIDABLE (3 of 5 counted evenings)"
    assert rules.not_yet_decidable(0) == "NOT YET DECIDABLE (0 of 5 counted evenings)"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_fewer_than_five_counted_evenings_is_not_yet_decidable(count: int) -> None:
    assert rules.decide([1.0] * count) == f"NOT YET DECIDABLE ({count} of 5 counted evenings)"


def test_five_counted_evenings_inside_the_window_adopt() -> None:
    assert rules.decide([1.0, 1.1, 0.9, 1.2, 0.8]) == "ADOPT"


@pytest.mark.parametrize("median", [0.5, 2.0])
def test_the_bounds_are_inclusive(median: float) -> None:
    # The median of five is the middle value once sorted.
    assert rules.decide([median, median, median, 1.0, 1.0]) == "ADOPT"
    assert rules.decide([0.0, 0.1, median, 5.0, 9.0]) == "ADOPT"


@pytest.mark.parametrize("median", [math.nextafter(0.5, 0.0), math.nextafter(2.0, 3.0), 0.1, 7.0])
def test_a_median_outside_the_bounds_keeps_v0(median: float) -> None:
    assert rules.decide([median, median, median, 1.0, 1.0]) == "KEEP V0"


def test_the_verdict_is_the_median_not_the_mean() -> None:
    # The mean is 20.6 (outside), the median 1.0 (inside).
    assert rules.decide([1.0, 1.0, 1.0, 1.0, 99.0]) == "ADOPT"
    # The mean is 0.9 (inside), the median 0.1 (outside).
    assert rules.decide([0.1, 0.1, 0.1, 2.0, 2.2]) == "KEEP V0"


def test_the_median_does_not_depend_on_the_order_of_the_evenings() -> None:
    assert rules.decide([5.0, 0.2, 1.0, 1.0, 1.0]) == rules.decide([1.0, 1.0, 1.0, 0.2, 5.0])


def test_infinity_takes_part_in_the_median() -> None:
    assert rules.decide([math.inf, math.inf, 1.0, 1.0, 1.0]) == "ADOPT"
    assert rules.decide([math.inf, math.inf, math.inf, 1.0, 1.0]) == "KEEP V0"


def test_only_the_first_five_evenings_count_never_the_latest() -> None:
    first_five_adopt = [1.0, 1.0, 1.0, 1.0, 1.0]
    later = [9.0] * 10  # would flip any median that included them
    assert rules.decide(first_five_adopt + later) == "ADOPT"
    first_five_keep = [9.0] * 5
    assert rules.decide(first_five_keep + [1.0] * 10) == "KEEP V0"


def test_the_first_five_are_the_first_in_the_order_given() -> None:
    # `decide` takes the ratios already in date order; a sixth never displaces the first five.
    assert rules.decide([0.1, 0.1, 0.1, 0.1, 0.1, 1.0]) == "KEEP V0"


def test_the_constants_are_the_criterions() -> None:
    assert rules.CAMERA_ID == "WHTM1"
    assert rules.MIN_USED_FRAMES == 3
    assert rules.COUNTED_NEEDED == 5
    assert rules.CLOUD_LIMIT_PCT == 10.0
    assert (rules.RATIO_LOW, rules.RATIO_HIGH) == (0.5, 2.0)
    assert rules.FIRST_EVENING.isoformat() == "2026-10-10"
