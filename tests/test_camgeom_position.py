"""The position search: where in the described area does the skyline put the camera?"""

from __future__ import annotations

import pytest

from lastlight.camgeom.calibration import (
    PositionTrial,
    choose_position,
    search_offsets,
    search_position,
    summarize_search,
)
from lastlight.camgeom.orientation import DEFAULT_HFOV_RANGE_DEG, Orientation


def trial(east: float, north: float, rms: float, az: float = 285.0, hfov: float = 20.0):
    return PositionTrial(east, north, rms, Orientation(az, 2.0, 0.3, hfov))


# --- the grid ---------------------------------------------------------------------------------


def test_the_grid_is_every_multiple_of_the_step_inside_the_box() -> None:
    offsets = search_offsets(6.0, 2.0)
    assert len(offsets) == 49
    assert (0.0, 0.0) in offsets
    assert {east for east, _ in offsets} == {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0}


def test_a_grid_can_be_centred_on_a_chosen_offset() -> None:
    offsets = search_offsets(2.0, 1.0, around=(4.0, -2.0))
    assert len(offsets) == 25
    assert {east for east, _ in offsets} == {2.0, 3.0, 4.0, 5.0, 6.0}
    assert {north for _, north in offsets} == {-4.0, -3.0, -2.0, -1.0, 0.0}


def test_a_box_smaller_than_the_step_is_only_its_centre() -> None:
    assert search_offsets(1.0, 2.0) == [(0.0, 0.0)]
    assert search_offsets(0.0, 2.0, around=(3.0, 1.0)) == [(3.0, 1.0)]


@pytest.mark.parametrize(("box", "step"), [(-1.0, 2.0), (4.0, 0.0), (4.0, -1.0)])
def test_a_nonsense_box_or_step_is_refused(box: float, step: float) -> None:
    with pytest.raises(ValueError):
        search_offsets(box, step)


# --- choosing a position ----------------------------------------------------------------------


def test_the_position_with_the_lowest_residual_is_chosen() -> None:
    trials = [trial(0.0, 0.0, 3.6), trial(2.0, 0.0, 1.8), trial(4.0, 0.0, 2.5)]
    assert choose_position(trials).east_km == 2.0


def test_a_fit_pinned_to_a_field_of_view_limit_is_never_chosen() -> None:
    low, high = DEFAULT_HFOV_RANGE_DEG
    trials = [
        trial(0.0, 0.0, 3.6, hfov=72.0),
        trial(2.0, 0.0, 1.0, hfov=low),  # the lowest residual, but it sits on the limit
        trial(4.0, 0.0, 1.2, hfov=high),
        trial(6.0, 0.0, 2.2, hfov=21.0),
    ]
    assert choose_position(trials).east_km == 6.0


def test_with_nothing_usable_there_is_no_position_to_choose() -> None:
    low, _ = DEFAULT_HFOV_RANGE_DEG
    with pytest.raises(ValueError, match="no position"):
        choose_position([])
    with pytest.raises(ValueError, match="no position"):
        choose_position([trial(0.0, 0.0, 1.0, hfov=low)])


# --- the search: coarse, then fine around the best coarse position -------------------------------


def surface(minimum: tuple[float, float], log: list[tuple[float, float]]):
    def fit_at(east: float, north: float) -> PositionTrial:
        log.append((east, north))
        distance_squared = (east - minimum[0]) ** 2 + (north - minimum[1]) ** 2
        return trial(east, north, 1.0 + 0.05 * distance_squared)

    return fit_at


def test_the_search_refines_around_the_best_coarse_position() -> None:
    calls: list[tuple[float, float]] = []
    best, trials = search_position(surface((2.0, -2.0), calls), box_km=8.0, step_km=4.0)
    assert (best.east_km, best.north_km) == (2.0, -2.0)
    assert best.skyline_rms_px == pytest.approx(1.0)
    assert len(trials) == len(calls) == len(set(calls)), "no position is fitted twice"
    assert len(trials) > 25, "the coarse grid is followed by a finer one"


def test_the_fine_grid_never_leaves_the_stated_box() -> None:
    calls: list[tuple[float, float]] = []
    best, trials = search_position(surface((20.0, 0.0), calls), box_km=8.0, step_km=4.0)
    assert all(abs(east) <= 8.0 and abs(north) <= 8.0 for east, north in calls)
    assert (best.east_km, best.north_km) == (8.0, 0.0), "the floor is beyond the box: take its edge"


def test_a_position_where_the_fit_is_impossible_is_skipped() -> None:
    def fit_at(east: float, north: float):
        return None if east > 0 else trial(east, north, 2.0 + abs(east))

    best, trials = search_position(fit_at, box_km=4.0, step_km=2.0)
    assert best.east_km == 0.0
    assert all(t.east_km <= 0.0 for t in trials)


def test_a_box_smaller_than_the_step_fits_only_the_site() -> None:
    calls: list[tuple[float, float]] = []
    best, trials = search_position(surface((0.0, 0.0), calls), box_km=1.0, step_km=2.0)
    assert calls == [(0.0, 0.0)] and len(trials) == 1 and best.east_km == 0.0


def test_a_search_with_no_usable_position_is_an_error() -> None:
    with pytest.raises(ValueError, match="no position"):
        search_position(lambda east, north: None, box_km=4.0, step_km=2.0)


# --- what the fit file records ----------------------------------------------------------------


def test_the_summary_records_the_chosen_offset_and_how_flat_the_valley_is() -> None:
    low, _ = DEFAULT_HFOV_RANGE_DEG
    trials = [
        trial(0.0, 0.0, 3.6, az=277.4, hfov=72.5),
        trial(2.0, -2.0, 1.7, az=285.0, hfov=20.0),
        trial(2.0, 0.0, 1.8, az=283.0, hfov=22.0),
        trial(4.0, 4.0, 2.9, az=279.0, hfov=19.0),
        trial(6.0, 0.0, 1.0, az=290.0, hfov=low),  # pinned: counted, never near-best
    ]
    summary = summarize_search(trials, trials[1], box_km=8.0, step_km=4.0, site_rms_px=3.6)
    assert summary["box_km"] == 8.0 and summary["step_km"] == 4.0
    assert summary["positions"] == 5
    assert (summary["offset_east_km"], summary["offset_north_km"]) == (2.0, -2.0)
    assert summary["rms_px_at_site"] == 3.6 and summary["rms_px_best"] == 1.7
    assert summary["near_best_positions"] == 2
    assert summary["near_best_azimuth_deg_range"] == [283.0, 285.0]
    assert summary["near_best_hfov_deg_range"] == [20.0, 22.0]
    assert summary["positions_at_hfov_bound"] == 1
    assert summary["best_on_box_edge"] is False
    assert summary["adopted"] is True
    assert "flat" in summary["reason"]


def test_the_summary_says_when_the_best_position_is_on_the_edge_of_the_box() -> None:
    edge = trial(8.0, 2.0, 1.6)
    summary = summarize_search([trial(0.0, 0.0, 3.0), edge], edge, 8.0, 4.0, 3.0)
    assert summary["best_on_box_edge"] is True
    assert "box" in summary["reason"]


def test_the_summary_holds_statistics_and_offsets_never_a_position() -> None:
    only = trial(0.0, 0.0, 2.0)
    summary = summarize_search([only], only, 0.0, 2.0, 2.0)
    assert set(summary) == {
        "box_km",
        "step_km",
        "positions",
        "offset_east_km",
        "offset_north_km",
        "rms_px_at_site",
        "rms_px_best",
        "near_best_positions",
        "near_best_azimuth_deg_range",
        "near_best_hfov_deg_range",
        "positions_at_hfov_bound",
        "best_on_box_edge",
        "adopted",
        "reason",
    }


def test_the_summary_tolerates_a_search_that_never_fitted_the_site_itself() -> None:
    trial_ = trial(2.0, 2.0, 2.0)
    summary = summarize_search([trial_], trial_, 4.0, 2.0, site_rms_px=None)
    assert summary["rms_px_at_site"] is None
