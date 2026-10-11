"""Choosing which HRRR run and forecast hour to read for an evening (issue #6).

The planner aims at the hour in which the sun reaches the horizon at the public reference point.
The oracle below finds that hour straight from `geometry`, so no clock time is written into this
file: every expectation is a relation (cycle plus forecast hour equals the target hour).
"""

from __future__ import annotations

import datetime as dt
from datetime import UTC, timedelta

import hrrr_support as support
import pytest

from lastlight import geometry
from lastlight.inputs import hrrr
from lastlight.inputs.hrrr import HrrrError, RunPlan, plan_run

REF_LAT = 33.45
REF_LON = -112.07
EVENING = dt.date(2026, 10, 10)


def _target_hour(evening: dt.date) -> dt.datetime:
    """The whole UTC hour nearest the sun reaching geometric elevation 0 on the evening."""
    instant = geometry.elevation_crossings(evening, REF_LAT, REF_LON)[0.0]
    return (instant + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)


def _cycle_instant(plan: RunPlan) -> dt.datetime:
    return dt.datetime.combine(plan.cycle_date, dt.time(plan.cycle_hour), tzinfo=UTC)


def _no_network() -> support.FakeTransport:
    return support.FakeTransport({})


LONG_AFTER = dt.datetime(2026, 12, 1, tzinfo=UTC)


# --- an evening that is already over --------------------------------------------------------------------


def test_a_past_evening_reads_the_analysis_of_the_target_hour_with_no_network() -> None:
    transport = _no_network()

    plan = plan_run(
        EVENING, cycle_hour=None, forecast_hour=None, now_utc=LONG_AFTER, transport=transport
    )

    assert _cycle_instant(plan) == _target_hour(EVENING)
    assert plan.forecast_hour == 0
    assert transport.calls == []


def test_the_plan_is_a_frozen_run_plan() -> None:
    plan = plan_run(
        EVENING, cycle_hour=None, forecast_hour=None, now_utc=LONG_AFTER, transport=_no_network()
    )

    assert isinstance(plan, RunPlan)
    with pytest.raises(AttributeError):
        plan.cycle_hour = 1  # type: ignore[misc]


# --- an explicit cycle hour or forecast hour ----------------------------------------------------------------


@pytest.mark.parametrize("cycle_hour", [0, 5, 12, 18, 23])
def test_an_explicit_cycle_hour_is_the_latest_such_cycle_at_or_before_the_target(
    cycle_hour: int,
) -> None:
    target = _target_hour(EVENING)

    plan = plan_run(
        EVENING,
        cycle_hour=cycle_hour,
        forecast_hour=None,
        now_utc=LONG_AFTER,
        transport=_no_network(),
    )

    cycle = _cycle_instant(plan)
    assert plan.cycle_hour == cycle_hour
    assert cycle <= target
    assert target - cycle < timedelta(hours=24)
    assert cycle + timedelta(hours=plan.forecast_hour) == target


def test_an_explicit_forecast_hour_picks_the_cycle_that_makes_it_valid_at_the_target() -> None:
    plan = plan_run(
        EVENING, cycle_hour=None, forecast_hour=5, now_utc=LONG_AFTER, transport=_no_network()
    )

    assert plan.forecast_hour == 5
    assert _cycle_instant(plan) + timedelta(hours=5) == _target_hour(EVENING)


def test_an_explicit_cycle_and_forecast_hour_are_taken_as_given() -> None:
    plan = plan_run(
        EVENING, cycle_hour=22, forecast_hour=3, now_utc=LONG_AFTER, transport=_no_network()
    )

    assert (plan.cycle_hour, plan.forecast_hour) == (22, 3)
    assert plan.cycle_date <= _target_hour(EVENING).date()


def test_an_explicit_cycle_and_forecast_hour_need_not_land_on_the_target() -> None:
    # Asking for run 22 at forecast hour 0 is a deliberate choice, whatever hour that is valid.
    plan = plan_run(
        EVENING, cycle_hour=22, forecast_hour=0, now_utc=LONG_AFTER, transport=_no_network()
    )

    assert (plan.cycle_hour, plan.forecast_hour) == (22, 0)
    assert plan.cycle_date == EVENING  # the 22 cycle just before the sun reaches the horizon


# --- an evening still to come ----------------------------------------------------------------------------


def test_an_evening_still_ahead_uses_the_newest_complete_run_and_counts_forward_to_it() -> None:
    target = _target_hour(EVENING)
    now = target - timedelta(hours=6, minutes=27)
    newest = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    transport = support.FakeTransport(
        {hrrr.idx_url(newest.date(), newest.hour, 18): b""}
    )

    plan = plan_run(
        EVENING, cycle_hour=None, forecast_hour=None, now_utc=now, transport=transport
    )

    assert _cycle_instant(plan) == newest
    assert _cycle_instant(plan) + timedelta(hours=plan.forecast_hour) == target


def test_a_target_hour_that_is_only_just_past_is_still_read_from_the_newest_run() -> None:
    # The run that starts at the target hour is not yet complete: two hours must pass first.
    target = _target_hour(EVENING)
    now = target + timedelta(minutes=45)
    newest = target - timedelta(hours=2)
    transport = support.FakeTransport({hrrr.idx_url(newest.date(), newest.hour, 18): b""})

    plan = plan_run(
        EVENING, cycle_hour=None, forecast_hour=None, now_utc=now, transport=transport
    )

    assert _cycle_instant(plan) == newest
    assert plan.forecast_hour == 2


def test_a_target_hour_beyond_the_forecast_range_is_a_clear_error() -> None:
    target = _target_hour(EVENING)
    now = target - timedelta(days=5)
    newest = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    transport = support.FakeTransport({hrrr.idx_url(newest.date(), newest.hour, 18): b""})

    with pytest.raises(HrrrError, match="forecast hour"):
        plan_run(EVENING, cycle_hour=None, forecast_hour=None, now_utc=now, transport=transport)


def test_an_explicit_forecast_hour_the_cycle_cannot_reach_is_a_clear_error() -> None:
    # Hour-30 forecasts exist only for the 00, 06, 12 and 18 cycles; the cycle that would make
    # a hour-30 forecast valid at the target is, for this evening, not one of them.
    for forecast_hour in range(19, 49):
        cycle = _target_hour(EVENING) - timedelta(hours=forecast_hour)
        if cycle.hour % 6 != 0:
            break
    with pytest.raises(HrrrError, match="forecast hour"):
        plan_run(
            EVENING,
            cycle_hour=None,
            forecast_hour=forecast_hour,
            now_utc=LONG_AFTER,
            transport=_no_network(),
        )


def test_naive_now_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        plan_run(
            EVENING,
            cycle_hour=None,
            forecast_hour=None,
            now_utc=dt.datetime(2026, 12, 1),
            transport=_no_network(),
        )


def test_max_forecast_hour_is_48_for_the_long_cycles_and_18_for_the_rest() -> None:
    assert [hrrr.max_forecast_hour(hour) for hour in (0, 6, 12, 18)] == [48, 48, 48, 48]
    assert [hrrr.max_forecast_hour(hour) for hour in (1, 5, 13, 22, 23)] == [18] * 5
