"""The G0 statistic and rule on synthetic contrast profiles with known answers (issue #10).

D(d, s) is the mean over the sweep elevations inside band s, WHTM1's 5 contrast bands and the 2
channels of |C_d - median over dates of C|. The gate passes when, on at least 25 percent of
dates, D reaches 0.05 in at least 2 of the 3 counted bands (golden, early, mid; late is reported
and never counted). The arrays here are built so that every answer can be worked out by hand:
most dates sit exactly on the median (all zero), and a few are shifted by a known amount.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from lastlight import geometry
from lastlight.g0 import stats

N_ELEVATIONS, N_BANDS, N_CHANNELS = 29, 5, 2
CRITERION_FILE = Path(__file__).parents[1] / "docs" / "gates" / "G0" / "criterion.md"
BAND_ORDER = ("golden", "early", "mid", "late")


def zeros(n: int) -> np.ndarray:
    return np.zeros((n, N_ELEVATIONS, N_BANDS, N_CHANNELS))


def shift(contrast: np.ndarray, date: int, band: str, amount: float, *, channel=None) -> None:
    """Add `amount` to every elevation of `band`, every contrast band and `channel` (or both)."""
    members = list(stats.band_members(band))
    index = slice(None) if channel is None else channel
    contrast[date, members, :, index] += amount


def series(n: int, shifted: int, bands: dict[str, float]) -> np.ndarray:
    """`n` dates, the first `shifted` of them moved by `bands[band]` in each named band."""
    contrast = zeros(n)
    for date in range(shifted):
        for band, amount in bands.items():
            shift(contrast, date, band, amount)
    return contrast


def distances(contrast: np.ndarray) -> np.ndarray:
    return stats.band_distances(contrast, stats.period_median(contrast))


# --- which sweep elevations are inside a band -------------------------------------------------


def test_the_bands_are_the_four_geometry_bands_and_three_are_counted() -> None:
    assert stats.BANDS == tuple(geometry.ELEVATION_BANDS) == BAND_ORDER
    assert stats.COUNTED_BANDS == ("golden", "early", "mid")
    assert (stats.THRESHOLD, stats.MIN_BANDS, stats.MIN_FRACTION) == (0.05, 2, 0.25)


def test_every_sweep_elevation_is_in_exactly_one_band() -> None:
    seen = [i for band in stats.BANDS for i in stats.band_members(band)]
    assert sorted(seen) == list(range(29))


def test_band_membership_follows_the_pre_registered_half_open_rule() -> None:
    sweep = geometry.ELEVATION_SWEEP

    def values(band: str) -> list[float]:
        return [sweep[i] for i in stats.band_members(band)]

    assert [len(values(b)) for b in stats.BANDS] == [12, 4, 6, 7]
    assert values("golden")[0] == 6.0 and values("golden")[-1] == 0.5
    assert values("early") == [0.0, -0.5, -1.0, -1.5]
    assert values("mid") == [-2.0, -2.5, -3.0, -3.5, -4.0, -4.5]
    assert values("late")[0] == -5.0 and values("late")[-1] == -8.0


def test_an_unknown_band_is_an_error() -> None:
    with pytest.raises(ValueError, match="noon"):
        stats.band_members("noon")


# --- D(d, s) ------------------------------------------------------------------------------------


def test_d_is_zero_when_every_date_is_the_same() -> None:
    contrast = np.full((6, N_ELEVATIONS, N_BANDS, N_CHANNELS), 0.3)
    np.testing.assert_array_equal(distances(contrast), np.zeros((6, 4)))


def test_d_is_the_absolute_difference_from_the_period_median_per_band() -> None:
    contrast = zeros(5)
    shift(contrast, 0, "golden", 0.2)
    shift(contrast, 1, "early", -0.3)
    shift(contrast, 2, "mid", 0.07)
    shift(contrast, 3, "late", -0.11)
    d = distances(contrast)
    assert d.shape == (5, 4)
    np.testing.assert_allclose(d[0], [0.2, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(d[1], [0, 0.3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(d[2], [0, 0, 0.07, 0], atol=1e-12)
    np.testing.assert_allclose(d[3], [0, 0, 0, 0.11], atol=1e-12)
    np.testing.assert_array_equal(d[4], np.zeros(4))


def test_d_is_a_mean_over_channels_not_a_maximum() -> None:
    contrast = zeros(5)
    shift(contrast, 0, "golden", 0.2, channel=0)  # log(R/G) moves 0.2, log(B/G) does not move
    assert distances(contrast)[0, 0] == pytest.approx(0.1, abs=1e-12)


def test_d_is_a_mean_over_the_five_contrast_bands() -> None:
    contrast = zeros(5)
    contrast[0, list(stats.band_members("early")), 2, :] += 0.5  # only the middle band moves
    assert distances(contrast)[0, 1] == pytest.approx(0.5 / 5, abs=1e-12)


def test_d_is_a_mean_over_the_elevations_inside_the_band() -> None:
    contrast = zeros(5)
    members = list(stats.band_members("mid"))
    contrast[0, members[:3], :, :] += 0.4  # half of the six mid elevations
    assert distances(contrast)[0, 2] == pytest.approx(0.2, abs=1e-12)


def test_d_uses_the_absolute_value_so_opposite_shifts_do_not_cancel() -> None:
    contrast = zeros(5)
    shift(contrast, 0, "golden", 0.1, channel=0)
    shift(contrast, 0, "golden", -0.1, channel=1)
    assert distances(contrast)[0, 0] == pytest.approx(0.1, abs=1e-12)


def test_the_median_is_taken_over_dates_at_each_elevation_band_and_channel() -> None:
    contrast = zeros(5)
    for date, level in enumerate([0.0, 0.0, 1.0, 1.0, 1.0]):
        contrast[date, :, :, :] = level
    median = stats.period_median(contrast)
    assert median.shape == (N_ELEVATIONS, N_BANDS, N_CHANNELS)
    np.testing.assert_array_equal(median, np.ones_like(median))
    d = stats.band_distances(contrast, median)
    np.testing.assert_allclose(d[:2], 1.0)
    np.testing.assert_allclose(d[2:], 0.0)


# --- the threshold: just below and just above --------------------------------------------------


def test_a_shift_just_above_the_threshold_counts_and_just_below_does_not() -> None:
    above = series(10, 1, {"golden": 0.0501})
    below = series(10, 1, {"golden": 0.0499})
    assert stats.bands_at_threshold(distances(above))[0] == 1
    assert stats.bands_at_threshold(distances(below))[0] == 0


def test_a_d_of_exactly_the_threshold_counts_in_every_counted_band() -> None:
    # The criterion says D(d, s) >= 0.05: a value that equals the threshold is a hit. The rows are
    # built, not computed, so the 0.05 is the same float as `stats.THRESHOLD`, with no rounding
    # from a mean in between.
    at_threshold = 0.05
    assert at_threshold == stats.THRESHOLD
    matrix = np.array(
        [
            [at_threshold, at_threshold, 0.0, 0.0],  # exactly 0.05 in golden and early: meets
            [at_threshold, 0.0, 0.0, at_threshold],  # one counted band and late: does not
            [at_threshold, at_threshold, at_threshold, 0.0],  # all three counted bands
        ]
    )
    assert list(stats.bands_at_threshold(matrix)) == [2, 1, 3]
    assert list(stats.meets_rule(matrix)) == [True, False, True]
    assert stats.verdict(matrix).meeting == 2


def test_a_date_needs_two_of_the_three_counted_bands() -> None:
    one = series(10, 1, {"golden": 0.2})
    two = series(10, 1, {"golden": 0.2, "mid": 0.2})
    three = series(10, 1, {"golden": 0.2, "early": 0.2, "mid": 0.2})
    assert list(stats.bands_at_threshold(distances(one))[:1]) == [1]
    assert list(stats.bands_at_threshold(distances(two))[:1]) == [2]
    assert list(stats.bands_at_threshold(distances(three))[:1]) == [3]
    assert not stats.meets_rule(distances(one))[0]
    assert stats.meets_rule(distances(two))[0]
    assert stats.meets_rule(distances(three))[0]


def test_the_late_band_is_reported_and_never_counted() -> None:
    contrast = series(10, 1, {"golden": 0.2, "late": 5.0})
    d = distances(contrast)
    assert d[0, 3] == pytest.approx(5.0)
    assert stats.bands_at_threshold(d)[0] == 1
    assert not stats.meets_rule(d)[0]
    only_late = series(10, 1, {"late": 5.0})
    assert stats.bands_at_threshold(distances(only_late))[0] == 0


# --- the 25 percent rule ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shifted", "passes"),
    [(24, False), (25, True), (26, True), (0, False), (100, False)],
)
def test_the_gate_needs_a_quarter_of_the_dates_not_one_less(shifted: int, passes: bool) -> None:
    # 100 dates; `shifted` of them are moved 0.0501 in two counted bands. With 100 dates shifted
    # the median moves with them and nothing differs from it, which is a fail by construction.
    contrast = series(100, shifted, {"golden": 0.0501, "early": 0.0501})
    verdict = stats.verdict(distances(contrast))
    assert verdict.dates == 100
    assert verdict.meeting == (shifted if shifted < 100 else 0)
    assert verdict.passed is passes


def test_just_below_the_threshold_in_every_band_fails_however_many_dates_move() -> None:
    contrast = series(100, 40, {"golden": 0.0499, "early": 0.0499, "mid": 0.0499})
    verdict = stats.verdict(distances(contrast))
    assert verdict.meeting == 0
    assert not verdict.passed
    assert verdict.fraction == 0.0


def test_the_fraction_is_over_the_dates_used() -> None:
    contrast = series(40, 12, {"golden": 0.1, "early": 0.1})
    verdict = stats.verdict(distances(contrast))
    assert verdict.fraction == pytest.approx(12 / 40)
    assert verdict.passed  # 30 percent
    assert verdict.dates == 40 and verdict.meeting == 12


def test_a_verdict_over_no_dates_is_an_error() -> None:
    with pytest.raises(ValueError, match="no dates"):
        stats.verdict(np.zeros((0, 4)))


# --- the seasonal control (reported, not the verdict) ------------------------------------------


def test_a_windowed_median_removes_a_smooth_seasonal_trend_the_year_median_keeps() -> None:
    n = 120
    start = dt.date(2026, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    contrast = np.zeros((n, N_ELEVATIONS, N_BANDS, N_CHANNELS))
    contrast += (0.01 * np.arange(n))[:, None, None, None]  # a ramp: 0.01 per day
    whole = stats.band_distances(contrast, stats.period_median(contrast))
    local = stats.band_distances(contrast, stats.windowed_median(dates, contrast, 15))
    assert whole[0, 0] == pytest.approx(0.01 * 59.5)  # the first date vs the middle of the year
    assert whole[:, 0].max() > 0.5
    interior = slice(15, n - 15)
    np.testing.assert_allclose(local[interior], 0.0, atol=1e-12)


def test_the_windowed_median_uses_only_dates_within_the_window() -> None:
    dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in (0, 1, 2, 40, 41, 42)]
    contrast = np.zeros((6, N_ELEVATIONS, N_BANDS, N_CHANNELS))
    contrast[3:] = 9.0
    baseline = stats.windowed_median(dates, contrast, 15)
    np.testing.assert_array_equal(baseline[0], contrast[0])
    np.testing.assert_array_equal(baseline[4], contrast[4])
    assert stats.band_distances(contrast, baseline).max() == 0.0


def test_the_windowed_median_is_taken_over_calendar_days_so_gaps_do_not_matter() -> None:
    dates = [dt.date(2026, 3, 1), dt.date(2026, 3, 2), dt.date(2026, 3, 29)]
    contrast = np.zeros((3, N_ELEVATIONS, N_BANDS, N_CHANNELS))
    contrast[2] = 1.0
    baseline = stats.windowed_median(dates, contrast, 15)
    np.testing.assert_array_equal(baseline[0], contrast[0])  # 03-29 is 28 days away
    np.testing.assert_array_equal(baseline[2], contrast[2])


# --- delta-E 2000 for tier B ------------------------------------------------------------------


SHARMA_1 = (50.0000, 2.6772, -79.7751)
SHARMA_2 = (50.0000, 0.0000, -82.7485)  # delta-E 2000 between the two is 2.0425 (Sharma et al.)


def lab_curves(n: int, labs: list[tuple[float, float, float]]) -> np.ndarray:
    curves = np.empty((n, N_ELEVATIONS, 7, 3))
    for i, lab in enumerate(labs):
        curves[i] = lab
    return curves


def test_delta_e_of_a_date_equal_to_the_median_is_zero() -> None:
    curves = lab_curves(3, [SHARMA_1, SHARMA_1, SHARMA_1])
    np.testing.assert_allclose(stats.delta_e_by_band(curves), 0.0, atol=1e-12)


def test_delta_e_2000_matches_the_published_pair() -> None:
    # Two dates at SHARMA_2 and one at SHARMA_1: the median is SHARMA_2, so the third date is
    # exactly one published pair away from it, at every elevation and view.
    curves = lab_curves(3, [SHARMA_2, SHARMA_2, SHARMA_1])
    de = stats.delta_e_by_band(curves)
    assert de.shape == (3, 3)  # early, mid, late
    assert stats.DELTA_E_BANDS == ("early", "mid", "late")
    np.testing.assert_allclose(de[:2], 0.0, atol=1e-9)
    np.testing.assert_allclose(de[2], 2.0425, atol=1e-4)


def test_delta_e_is_a_mean_over_the_band_elevations_and_the_views() -> None:
    curves = lab_curves(3, [SHARMA_2, SHARMA_2, SHARMA_2])
    early = list(stats.band_members("early"))
    curves[2, early[:2], :, :] = SHARMA_1  # half of the early elevations moved
    de = stats.delta_e_by_band(curves)
    assert de[2, 0] == pytest.approx(2.0425 / 2.0, abs=1e-4)
    assert de[2, 1] == 0.0 and de[2, 2] == 0.0


# --- the pre-registered text ------------------------------------------------------------------


def test_the_criterion_sentence_in_the_code_is_the_one_in_the_committed_criterion() -> None:
    text = CRITERION_FILE.read_text(encoding="utf-8")
    assert stats.CRITERION_SENTENCE in text
    assert (
        "D(d, s) >= 0.05 in at least 2 of the 3 counted bands on at least 25 percent of dates"
        in (stats.CRITERION_SENTENCE)
    )


def test_the_constants_match_the_committed_criterion_table() -> None:
    text = CRITERION_FILE.read_text(encoding="utf-8")
    assert f"| Threshold on D(d, s) | {stats.THRESHOLD} |" in text
    assert "| Counted bands a date must reach the threshold in | at least 2 of 3 |" in text
    share = round(stats.MIN_FRACTION * 100)
    assert f"| Share of dates that must be such dates | at least {share} percent |" in text
    assert math.isclose(stats.MIN_FRACTION, 0.25)
