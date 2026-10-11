"""From per-date records to the G0 study: which dates count, the verdict, and its sensitivities.

`build_study` takes the records the run wrote and the period, and decides nothing by hand: dates
without a rendered curve are listed with their reason (a date with no record at all is
`not_collected`), the verdict is the pre-registered rule over the dates used, and the same rule is
re-run at the ends of the pixel-scale range, against a seasonal baseline, and for tier B.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from g0_support import dates, dropped_record, shifted_contrast, synthetic_record

from lastlight.g0 import stats
from lastlight.g0.analysis import build_study

START = dt.date(2026, 1, 1)


def study_of(records, n=None):
    n = n or len(records)
    return build_study(records, START, START + dt.timedelta(days=n - 1))


def population(n: int, moved: int, amount: float = 0.1, **kwargs):
    """`n` dates; the first `moved` shifted in golden and early (D = `amount` there)."""
    days = dates(START, n)
    return [
        synthetic_record(day, {"golden": amount, "early": amount} if i < moved else {}, **kwargs)
        for i, day in enumerate(days)
    ]


# --- which dates count -------------------------------------------------------------------------


def test_the_period_is_every_date_between_the_ends_inclusive() -> None:
    study = build_study(
        [synthetic_record(dt.date(2026, 2, 27))], dt.date(2026, 2, 27), dt.date(2026, 3, 2)
    )
    assert study.attempted == tuple(dt.date(2026, 2, 27) + dt.timedelta(days=i) for i in range(4))
    assert study.period == (dt.date(2026, 2, 27), dt.date(2026, 3, 2))


def test_dates_are_used_dropped_or_not_collected_and_every_one_is_accounted_for() -> None:
    records = population(10, 3)
    records[2] = dropped_record(records[2].date, "hrrr_run_missing", "not in the archive")
    records[5] = dropped_record(records[5].date, "aerosol_run_missing")
    del records[8]  # no record at all
    study = build_study(records, START, START + dt.timedelta(days=9))
    assert len(study.attempted) == 10
    assert len(study.used) == 7
    reasons = {d.date: d.drop_reason for d in study.dropped}
    assert reasons == {
        START + dt.timedelta(days=2): "hrrr_run_missing",
        START + dt.timedelta(days=5): "aerosol_run_missing",
        START + dt.timedelta(days=8): "not_collected",
    }
    assert len(study.used) + len(study.dropped) == len(study.attempted)
    assert [r.date for r in study.used] == sorted(r.date for r in study.used)


def test_a_record_outside_the_period_is_not_used() -> None:
    inside = synthetic_record(START)
    outside = synthetic_record(START + dt.timedelta(days=30))
    study = build_study([inside, outside], START, START + dt.timedelta(days=2))
    assert [r.date for r in study.used] == [START]
    assert len(study.dropped) == 2  # the two dates with no record


def test_an_end_before_the_start_is_an_error() -> None:
    with pytest.raises(ValueError, match="end"):
        build_study([], START, START - dt.timedelta(days=1))


# --- the verdict --------------------------------------------------------------------------------


def test_the_verdict_is_the_rule_over_the_dates_used() -> None:
    study = study_of(population(20, 6))  # 30 percent move by 0.1 in two counted bands
    assert study.verdict.passed
    assert (study.verdict.dates, study.verdict.meeting) == (20, 6)
    assert study.outcomes["nominal"].meets.sum() == 6


def test_just_under_a_quarter_fails_and_a_quarter_passes() -> None:
    assert not study_of(population(20, 4)).verdict.passed  # 20 percent
    assert study_of(population(20, 5)).verdict.passed  # 25 percent exactly


def test_dropped_dates_do_not_enter_the_share_but_are_reported_beside_it() -> None:
    records = population(16, 4)
    records += [
        dropped_record(START + dt.timedelta(days=16 + i), "hrrr_run_missing") for i in range(4)
    ]
    study = build_study(records, START, START + dt.timedelta(days=19))
    assert study.verdict.passed  # 4 of the 16 dates used
    assert study.verdict.fraction == pytest.approx(4 / 16)
    assert study.fraction_of_attempted == pytest.approx(4 / 20)


def test_no_dates_used_is_an_error_not_a_pass() -> None:
    with pytest.raises(ValueError, match="no dates"):
        study_of([dropped_record(START, "hrrr_run_missing")])


def test_the_distances_are_kept_per_date_and_band() -> None:
    study = study_of(population(10, 3, amount=0.2))
    nominal = study.outcomes["nominal"]
    assert nominal.distances.shape == (10, 4)
    np.testing.assert_allclose(nominal.distances[:3, 0], 0.2)
    np.testing.assert_allclose(nominal.distances[3:], 0.0, atol=1e-12)
    assert list(nominal.at_threshold[:3]) == [2, 2, 2]


# --- the sensitivity runs: reported, never the verdict -----------------------------------------


def test_the_ends_of_the_pixel_scale_range_are_recomputed_with_their_own_median() -> None:
    days = dates(START, 20)
    records = [
        synthetic_record(
            day,
            {"golden": 0.1, "early": 0.1} if i < 6 else {},  # nominal: 30 percent
            low={"golden": 0.1, "early": 0.1} if i < 2 else {},  # low-scale end: 10 percent
            high={"golden": 0.1, "early": 0.1} if i < 9 else {},  # high-scale end: 45 percent
        )
        for i, day in enumerate(days)
    ]
    study = study_of(records)
    assert study.verdict.passed
    assert not study.outcomes["low"].verdict.passed
    assert study.outcomes["high"].verdict.passed
    assert study.outcomes["high"].verdict.meeting == 9  # the median there is still zero
    assert study.verdict is study.outcomes["nominal"].verdict  # the sensitivity never replaces it


def test_a_pixel_scale_end_takes_its_median_from_its_own_contrast_not_the_nominal_one() -> None:
    # At the low-scale end 12 of 20 dates (more than half) sit 0.1 above zero in golden and early,
    # so that scale's own median is 0.1 there and the 8 unmoved dates are the ones 0.1 away from
    # it. Against the nominal contrast (all zero, so a median of zero) it would be the 12 moved
    # dates instead: 12 of 20, not 8 of 20.
    days = dates(START, 20)
    moved = {"golden": 0.1, "early": 0.1}
    records = [synthetic_record(day, {}, low=moved if i < 12 else {}) for i, day in enumerate(days)]
    low = study_of(records).outcomes["low"]
    assert low.verdict.meeting == 8
    assert low.meets.tolist() == [False] * 12 + [True] * 8
    np.testing.assert_allclose(low.baseline[list(stats.band_members("golden"))], 0.1)


def test_a_sensitivity_that_flips_does_not_change_the_headline_verdict() -> None:
    records = [
        synthetic_record(day, {"golden": 0.1, "early": 0.1} if i < 6 else {}, low={})
        for i, day in enumerate(dates(START, 20))
    ]
    study = study_of(records)
    assert study.verdict.passed is True
    assert study.outcomes["low"].verdict.passed is False


# --- the seasonal control ---------------------------------------------------------------------


def test_the_seasonal_control_uses_a_windowed_median_and_can_disagree() -> None:
    n = 90
    days = dates(START, n)
    records = [
        synthetic_record(day, {"golden": 0.01 * i, "early": 0.01 * i, "mid": 0.01 * i})
        for i, day in enumerate(days)
    ]
    study = study_of(records)
    assert study.verdict.passed  # a ramp across the period is far from the period median
    assert not study.seasonal.verdict.passed  # but not from the median of the neighbours
    assert study.seasonal.verdict.fraction < study.verdict.fraction
    assert study.seasonal.name == "seasonal"


# --- floor use ---------------------------------------------------------------------------------


def test_the_floor_share_is_channels_raised_over_channels_rendered() -> None:
    study = study_of(population(10, 0))
    # synthetic_record: 3, 5 and 0 channels raised per date at nominal, low and high;
    # each date renders 29 elevations x 6 views x 3 channels.
    per_date = 29 * 6 * 3
    assert study.floor_share["nominal"] == pytest.approx(3 / per_date)
    assert study.floor_share["low"] == pytest.approx(5 / per_date)
    assert study.floor_share["high"] == 0.0


# --- tier B ------------------------------------------------------------------------------------


def test_delta_e_by_band_is_computed_from_the_absolute_curves() -> None:
    days = dates(START, 5)
    records = [synthetic_record(day, lab_shift=0.0) for day in days[:4]]
    records.append(synthetic_record(days[4], lab_shift=10.0))  # 10 L* units lighter everywhere
    study = study_of(records)
    assert study.delta_e.shape == (5, 3)
    np.testing.assert_allclose(study.delta_e[:4], 0.0, atol=1e-9)
    assert (study.delta_e[4] > 5.0).all() and (study.delta_e[4] < 12.0).all()


# --- the five most extreme dates --------------------------------------------------------------


def test_the_five_most_extreme_dates_are_ranked_by_the_mean_distance_over_counted_bands() -> None:
    days = dates(START, 20)
    amounts = {3: 0.5, 7: 0.9, 9: 0.2, 12: 0.7, 15: 0.3, 16: 0.1}
    records = [
        synthetic_record(day, {"golden": amounts.get(i, 0.0), "late": 9.0 if i == 1 else 0.0})
        for i, day in enumerate(days)
    ]
    study = study_of(records)
    ranked = [(e.date, round(e.mean_d, 6)) for e in study.extremes]
    assert [d for d, _ in ranked] == [days[7], days[12], days[3], days[15], days[9]]
    assert ranked[0][1] == pytest.approx(0.9 * 12 / 12 / 3)  # golden only, over three bands
    assert days[1] not in [e.date for e in study.extremes]  # `late` is never counted
    assert study.extremes[0].aod_total == 0.08 and study.extremes[0].aod_dust == 0.02
    assert len(study.extremes) == 5


def test_counted_bands_and_the_late_band_are_named_in_the_study() -> None:
    study = study_of(population(10, 3))
    assert study.counted_bands == stats.COUNTED_BANDS
    assert study.bands == stats.BANDS


def test_contrast_helper_matches_the_band_membership() -> None:
    assert shifted_contrast({"early": 1.0}).sum() == 4 * 5 * 2


# --- the channel decomposition (post hoc, reported only) ---------------------------------------


def blue_only_population(n: int, moved: int, amount: float = 0.2):
    """Dates whose first `moved` are shifted in log(B/G) only, in golden and early."""
    records = population(n, 0)
    for record in records[:moved]:
        for band in ("golden", "early"):
            record.contrast["nominal"][list(stats.band_members(band)), :, 1] += amount
    return records


def test_the_statistic_is_also_split_by_channel() -> None:
    study = study_of(blue_only_population(20, 6))
    assert list(study.channels) == ["log(R/G)", "log(B/G)"]
    red, blue = study.channels["log(R/G)"], study.channels["log(B/G)"]
    np.testing.assert_allclose(red.distances, 0.0, atol=1e-12)
    np.testing.assert_allclose(blue.distances[:6, 0], 0.2)  # golden, blue only
    np.testing.assert_allclose(blue.distances[:6, 1], 0.2)  # early
    # With both channels averaged (the pre-registered D) the same shift is half as large.
    np.testing.assert_allclose(study.outcomes["nominal"].distances[:6, 0], 0.1)


def test_a_shift_that_is_only_in_blue_is_carried_by_blue_alone() -> None:
    study = study_of(blue_only_population(20, 6, amount=0.12))
    assert study.channels["log(B/G)"].verdict.passed  # 6 of 20 dates, D = 0.12
    assert not study.channels["log(R/G)"].verdict.passed
    assert study.verdict.passed  # over both channels D is 0.06, still at the threshold


def test_the_channel_runs_never_replace_the_verdict() -> None:
    study = study_of(blue_only_population(20, 6, amount=0.12))
    assert study.verdict is study.outcomes["nominal"].verdict
