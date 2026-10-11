"""The helpers behind the G0 figures: robust axis limits and a histogram that loses no date.

A single dust storm in a year of dates can be ten times larger than anything else, and an axis
that goes to its maximum flattens the other 364 dates into one bar. The figures cap the axis at a
high percentile and say how many dates went past it; these helpers are the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from lastlight.g0.figures import axis_limit, clipped_histogram


def test_the_axis_limit_is_a_high_percentile_with_a_little_room() -> None:
    values = np.arange(1.0, 101.0)
    assert axis_limit(values, percentile=95.0, floor=0.0) == pytest.approx(
        np.percentile(values, 95) * 1.05
    )


def test_one_extreme_value_does_not_set_the_axis() -> None:
    values = np.concatenate([np.linspace(0.0, 0.5, 400), [50.0]])
    assert axis_limit(values, percentile=99.0, floor=0.0) < 1.0


def test_the_axis_never_falls_below_the_floor() -> None:
    assert axis_limit(np.zeros(10), percentile=99.0, floor=0.1) == 0.1
    assert axis_limit(np.array([0.01, 0.02]), percentile=99.0, floor=0.1) == 0.1


def test_the_histogram_keeps_every_value_by_putting_the_overflow_in_the_last_bin() -> None:
    values = np.array([0.0, 0.1, 0.1, 0.45, 0.95, 3.0, 40.0])
    edges, counts, clipped = clipped_histogram(values, limit=1.0, bins=4)
    assert list(edges) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert counts.sum() == len(values)
    assert list(counts) == [3, 1, 0, 3]  # 0.95, 3.0 and 40.0 are in the last bin
    assert clipped == 2  # the two values past the limit


def test_a_value_exactly_at_the_limit_is_not_clipped() -> None:
    _, counts, clipped = clipped_histogram(np.array([1.0, 0.5]), limit=1.0, bins=2)
    assert counts.sum() == 2 and clipped == 0
