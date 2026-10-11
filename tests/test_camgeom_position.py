"""The position search: where in the described area does the skyline put the camera?"""

from __future__ import annotations

import numpy as np
import pytest
from camgeom_synthetic import REF_LAT, REF_LON, flat_dem, haversine_m

from lastlight.camgeom.calibration import (
    PositionTrial,
    choose_position,
    observer_at,
    ring_offsets,
    search_offsets,
    search_position,
    search_uncertainty,
    summarize_search,
)
from lastlight.camgeom.dem import Dem
from lastlight.camgeom.orientation import DEFAULT_HFOV_RANGE_DEG, Orientation
from lastlight.camgeom.sites import CameraSite


def trial(east: float, north: float, rms: float, az: float = 285.0, hfov: float = 20.0):
    return PositionTrial(east, north, rms, Orientation(az, 2.0, 0.3, hfov))


# --- the grid ---------------------------------------------------------------------------------


def test_the_grid_is_every_multiple_of_the_step_inside_the_box() -> None:
    offsets = search_offsets(6.0, 2.0)
    assert len(offsets) == 49
    assert (0.0, 0.0) in offsets
    assert {east for east, _ in offsets} == {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0}


def test_a_box_smaller_than_the_step_is_only_its_centre() -> None:
    assert search_offsets(1.0, 2.0) == [(0.0, 0.0)]
    assert search_offsets(0.0, 2.0) == [(0.0, 0.0)]


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


# --- the search: one fine grid over the whole box, and a ring of probes just outside ----------


def surface(minimum: tuple[float, float], log: list[tuple[float, float]]):
    def fit_at(east: float, north: float) -> PositionTrial:
        log.append((east, north))
        distance_squared = (east - minimum[0]) ** 2 + (north - minimum[1]) ** 2
        return trial(east, north, 1.0 + 0.05 * distance_squared)

    return fit_at


def test_the_search_fits_every_position_of_a_fine_grid_over_the_whole_box_once() -> None:
    calls: list[tuple[float, float]] = []
    result = search_position(surface((2.0, -2.0), calls), box_km=8.0, step_km=1.0)
    best, trials = result.best, result.trials
    assert (best.east_km, best.north_km) == (2.0, -2.0)
    assert best.skyline_rms_px == pytest.approx(1.0)
    assert len(trials) == 17 * 17, "a 1 km grid over a 16 km box"
    assert {(t.east_km, t.north_km) for t in trials} == set(search_offsets(8.0, 1.0))
    assert len(calls) == len(set(calls)), "no position is fitted twice"


def test_the_ring_is_a_square_of_probes_just_outside_the_box() -> None:
    ring = ring_offsets(8.0, gap_km=2.0, spacing_km=4.0)
    assert len(ring) == len(set(ring)) == 20
    assert all(max(abs(east), abs(north)) == pytest.approx(10.0) for east, north in ring)
    assert (10.0, 10.0) in ring and (-10.0, 10.0) in ring and (10.0, -2.0) in ring
    assert (0.0, 0.0) not in ring


def test_the_ring_is_one_step_outside_the_box_and_as_fine_as_the_grid_by_default() -> None:
    calls: list[tuple[float, float]] = []
    result = search_position(surface((0.0, 0.0), calls), box_km=4.0, step_km=2.0)
    assert {(t.east_km, t.north_km) for t in result.ring} == set(ring_offsets(4.0, 2.0, 2.0))
    assert len(result.ring) == 7 * 7 - 5 * 5, "a probe every 2 km on the square 6 km out"
    assert {max(abs(t.east_km), abs(t.north_km)) for t in result.ring} == {6.0}


@pytest.mark.parametrize(("gap", "spacing"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_a_ring_needs_a_positive_gap_and_spacing(gap: float, spacing: float) -> None:
    with pytest.raises(ValueError, match="gap and spacing"):
        ring_offsets(8.0, gap, spacing)


def test_the_floor_beyond_the_box_gives_the_edge_of_the_box_and_a_ring_that_fits_better() -> None:
    calls: list[tuple[float, float]] = []
    result = search_position(surface((20.0, 0.0), calls), box_km=8.0, step_km=2.0)
    best = result.best
    assert (best.east_km, best.north_km) == (8.0, 0.0), "the floor is beyond the box: take its edge"
    inside = [offset for offset in calls if max(abs(offset[0]), abs(offset[1])) <= 8.0]
    outside = [offset for offset in calls if offset not in inside]
    assert len(inside) == 81, "the grid itself never leaves the box"
    assert outside and all(max(abs(e), abs(n)) == pytest.approx(10.0) for e, n in outside)
    assert {(t.east_km, t.north_km) for t in result.ring} == set(outside)
    assert min(t.skyline_rms_px for t in result.ring) < best.skyline_rms_px


def test_a_position_on_the_ring_is_probed_but_never_chosen() -> None:
    result = search_position(surface((10.0, 2.0), []), box_km=8.0, step_km=2.0)
    assert (result.best.east_km, result.best.north_km) == (8.0, 2.0)
    assert (10.0, 2.0) in [(t.east_km, t.north_km) for t in result.ring]
    assert all(max(abs(t.east_km), abs(t.north_km)) <= 8.0 for t in result.trials)


def test_a_position_where_the_fit_is_impossible_is_skipped() -> None:
    def fit_at(east: float, north: float):
        return None if east > 0 else trial(east, north, 2.0 + abs(east))

    result = search_position(fit_at, box_km=4.0, step_km=2.0)
    assert result.best.east_km == 0.0
    assert all(t.east_km <= 0.0 for t in result.trials + result.ring)


def test_a_box_smaller_than_the_step_fits_only_the_site_and_probes_nothing() -> None:
    calls: list[tuple[float, float]] = []
    result = search_position(surface((0.0, 0.0), calls), box_km=1.0, step_km=2.0)
    assert calls == [(0.0, 0.0)] and len(result.trials) == 1 and result.best.east_km == 0.0
    assert result.ring == []


def test_a_search_with_no_usable_position_is_an_error() -> None:
    with pytest.raises(ValueError, match="no position"):
        search_position(lambda east, north: None, box_km=4.0, step_km=2.0)


# --- where a moved camera stands -----------------------------------------------------------------

SITE = CameraSite("SYN1", REF_LAT, REF_LON, 272.0, mount_height_m=12.0)
PLAIN = flat_dem(300.0)


def moved_by(east_km: float, north_km: float):
    found = observer_at(PLAIN, SITE, east_km, north_km)
    assert found is not None
    return found


def test_an_east_offset_moves_the_camera_east_by_that_many_kilometres() -> None:
    found = moved_by(10.0, 0.0)
    assert found.lon > REF_LON, "east is toward larger longitude"
    assert found.lat == pytest.approx(REF_LAT, abs=1e-6)
    assert haversine_m(REF_LAT, REF_LON, found.lat, found.lon) == pytest.approx(10_000, rel=0.005)


def test_a_north_offset_moves_the_camera_north_by_that_many_kilometres() -> None:
    found = moved_by(0.0, 10.0)
    assert found.lat > REF_LAT, "north is toward larger latitude"
    assert found.lon == pytest.approx(REF_LON, abs=1e-6)
    assert haversine_m(REF_LAT, REF_LON, found.lat, found.lon) == pytest.approx(10_000, rel=0.005)


def test_the_camera_stands_on_the_ground_plus_the_mount_height() -> None:
    assert moved_by(0.0, 0.0).elev_m == pytest.approx(312.0), "300 m of ground and a 12 m mast"


def test_a_moved_camera_stands_on_the_ground_where_it_was_moved_to() -> None:
    columns = PLAIN.elevation_m.shape[1]
    slope = np.linspace(0.0, 400.0, columns, dtype=np.float32)[None, :]
    ramp = Dem(
        PLAIN.elevation_m + slope,
        PLAIN.lat_top,
        PLAIN.lon_left,
        PLAIN.step_lat_deg,
        PLAIN.step_lon_deg,
    )
    here, there = observer_at(ramp, SITE, 0.0, 0.0), observer_at(ramp, SITE, 10.0, 0.0)
    assert here is not None and there is not None
    assert there.elev_m == pytest.approx(ramp.sample(there.lat, there.lon) + 12.0)
    assert there.elev_m - here.elev_m > 10.0, "ten kilometres east is uphill on this ramp"


def test_a_position_the_terrain_model_does_not_cover_has_no_observer() -> None:
    assert observer_at(PLAIN, SITE, 500.0, 0.0) is None


# --- what the fit file records ----------------------------------------------------------------


def valley():
    low, _ = DEFAULT_HFOV_RANGE_DEG
    return [
        trial(0.0, 0.0, 3.6, az=277.4, hfov=72.5),
        trial(2.0, -2.0, 1.7, az=285.0, hfov=20.0),
        trial(2.0, 0.0, 1.8, az=283.0, hfov=22.0),
        trial(1.0, 1.0, 1.95, az=281.0, hfov=21.0),  # 15 percent worse than the best: not near
        trial(4.0, 4.0, 2.9, az=279.0, hfov=19.0),
        trial(6.0, 0.0, 1.0, az=290.0, hfov=low),  # pinned: counted, never near-best
    ]


def test_the_summary_records_the_chosen_offset_and_how_flat_the_valley_is() -> None:
    trials = valley()
    summary = summarize_search(trials, trials[1], box_km=8.0, step_km=1.0, site_rms_px=3.6)
    assert summary["box_km"] == 8.0 and summary["step_km"] == 1.0
    assert summary["positions"] == 6
    assert (summary["offset_east_km"], summary["offset_north_km"]) == (2.0, -2.0)
    assert summary["rms_px_at_site"] == 3.6 and summary["rms_px_best"] == 1.7
    assert summary["near_best_positions"] == 2
    assert summary["near_best_azimuth_deg_range"] == [283.0, 285.0]
    assert summary["near_best_hfov_deg_range"] == [20.0, 22.0]
    assert summary["positions_at_hfov_bound"] == 1
    assert summary["best_on_box_edge"] is False
    assert summary["residual_falls_beyond_box"] is False
    assert summary["adopted"] is True
    assert "flat" in summary["reason"]


@pytest.mark.parametrize(
    ("east", "north", "on_edge"),
    [(8.0, 0.0, True), (7.0, 2.0, True), (0.0, -7.0, True), (6.0, 6.0, False), (0.0, 0.0, False)],
)
def test_the_best_is_on_the_edge_when_it_is_within_one_step_of_the_box(
    east: float, north: float, on_edge: bool
) -> None:
    best = trial(east, north, 1.6)
    summary = summarize_search([trial(0.0, 0.0, 3.0), best], best, 8.0, 1.0, 3.0)
    assert summary["best_on_box_edge"] is on_edge
    assert ("box" in summary["reason"]) is on_edge


def test_a_ring_that_fits_better_than_the_box_says_the_valley_continues_beyond_it() -> None:
    inside = [trial(0.0, 0.0, 3.0), trial(8.0, 0.0, 1.6)]
    ring = [trial(10.0, 0.0, 1.5), trial(10.0, 4.0, 1.9), trial(-10.0, 0.0, 3.5)]
    summary = summarize_search(inside, inside[1], 8.0, 1.0, 3.0, ring=ring)
    assert summary["residual_falls_beyond_box"] is True
    assert summary["ring_positions"] == 3 and summary["ring_rms_px_best"] == 1.5
    assert "beyond" in summary["reason"]


def test_a_ring_that_fits_worse_than_the_box_says_the_floor_is_inside_it() -> None:
    inside = [trial(0.0, 0.0, 1.2), trial(8.0, 0.0, 1.6)]
    ring = [trial(10.0, 0.0, 1.7), trial(-10.0, 0.0, 3.5)]
    summary = summarize_search(inside, inside[0], 8.0, 1.0, 1.2, ring=ring)
    assert summary["residual_falls_beyond_box"] is False
    assert summary["ring_rms_px_best"] == 1.7


def test_a_ring_fit_pinned_to_a_field_of_view_limit_does_not_count_as_a_better_fit() -> None:
    low, _ = DEFAULT_HFOV_RANGE_DEG
    inside = [trial(0.0, 0.0, 1.6)]
    ring = [trial(10.0, 0.0, 0.9, hfov=low), trial(10.0, 4.0, 2.0)]
    summary = summarize_search(inside, inside[0], 8.0, 1.0, 1.6, ring=ring)
    assert summary["residual_falls_beyond_box"] is False
    assert summary["ring_rms_px_best"] == 2.0


def test_a_search_without_a_ring_records_that_nothing_was_probed() -> None:
    only = trial(0.0, 0.0, 2.0)
    summary = summarize_search([only], only, 0.0, 2.0, 2.0)
    assert summary["residual_falls_beyond_box"] is False
    assert summary["ring_positions"] == 0 and summary["ring_rms_px_best"] is None


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
        "ring_positions",
        "ring_rms_px_best",
        "residual_falls_beyond_box",
        "adopted",
        "reason",
    }


def test_the_summary_tolerates_a_search_that_never_fitted_the_site_itself() -> None:
    trial_ = trial(2.0, 2.0, 2.0)
    summary = summarize_search([trial_], trial_, 4.0, 2.0, site_rms_px=None)
    assert summary["rms_px_at_site"] is None


# --- the uncertainty block the fit file carries ------------------------------------------------


def test_the_uncertainty_is_the_range_of_the_nearly_equal_fits() -> None:
    trials = valley()
    summary = summarize_search(trials, trials[1], 8.0, 1.0, 3.6)
    found = search_uncertainty(summary, width=800)
    assert found.azimuth_deg == (283.0, 285.0)
    assert found.hfov_deg == (20.0, 22.0)
    assert found.px_per_deg[0] == pytest.approx(400.0 / np.tan(np.radians(11.0)) * np.pi / 180.0)
    assert found.px_per_deg[1] == pytest.approx(400.0 / np.tan(np.radians(10.0)) * np.pi / 180.0)
    assert found.lower_bounds is False
    assert "2 positions" in found.basis


def test_the_uncertainty_is_marked_as_a_lower_bound_when_the_box_clips_the_valley() -> None:
    clipped = trial(8.0, 0.0, 1.6)
    on_edge = summarize_search([trial(0.0, 0.0, 3.0), clipped], clipped, 8.0, 1.0, 3.0)
    assert search_uncertainty(on_edge, 800).lower_bounds is True

    inside = trial(0.0, 0.0, 1.6)
    beyond = summarize_search([inside], inside, 8.0, 1.0, 1.6, ring=[trial(10.0, 0.0, 1.5)])
    assert search_uncertainty(beyond, 800).lower_bounds is True
    assert "beyond" in search_uncertainty(beyond, 800).basis
