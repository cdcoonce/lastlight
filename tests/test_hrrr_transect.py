"""Sampling the decoded fields along the sightline transect (issue #6)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from pathlib import Path

import hrrr_support as support
import numpy as np
import pytest

from lastlight import geometry
from lastlight.inputs import hrrr
from lastlight.inputs.hrrr import (
    DEFAULT_FIELDS,
    TRANSECT_DISTANCES_KM,
    HrrrError,
    HrrrFields,
    OutsideGridError,
    TransectSample,
    fetch_fields,
    sample_transect,
)

REF_LAT = 33.45
REF_LON = -112.07
EVENING = dt.date(2026, 10, 10)


@pytest.fixture(scope="module")
def fields(tmp_path_factory: pytest.TempPathFactory) -> HrrrFields:
    cache = tmp_path_factory.mktemp("hrrr-cache")
    return fetch_fields(
        support.CYCLE_DATE,
        support.CYCLE_HOUR,
        0,
        DEFAULT_FIELDS,
        cache,
        transport=support.crop_transport(),
    )


def _cell_point(fields: HrrrFields, cell: tuple[int, int]) -> tuple[float, float]:
    return float(fields.grid.lats[cell]), float(fields.grid.lons[cell])


def _nearest_cell_by_brute_force(
    fields: HrrrFields, lat: float, lon: float
) -> tuple[tuple[int, int], float]:
    p1 = math.radians(lat)
    p2 = np.radians(fields.grid.lats)
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(
        np.radians(fields.grid.lons - lon) / 2
    ) ** 2
    distance = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    cell = np.unravel_index(np.argmin(distance), distance.shape)
    return (int(cell[0]), int(cell[1])), float(distance[cell])


def _transect_points(distances: list[float]) -> list[tuple[float, float]]:
    return geometry.transect(EVENING, REF_LAT, REF_LON, distances)


# --- shape of the result ------------------------------------------------------------------------------


def test_one_row_per_point_and_field_in_point_then_field_order(fields: HrrrFields) -> None:
    points = _transect_points([0.0, 50.0, 100.0])

    samples = sample_transect(fields, points)

    assert len(samples) == 3 * len(DEFAULT_FIELDS)
    assert [(s.point_index, s.field) for s in samples] == [
        (index, spec.name) for index in range(3) for spec in DEFAULT_FIELDS
    ]


def test_every_row_carries_its_units_and_the_method(fields: HrrrFields) -> None:
    samples = sample_transect(fields, _transect_points([0.0]))

    assert {s.field: s.units for s in samples} == {spec.name: spec.units for spec in DEFAULT_FIELDS}
    assert {s.method for s in samples} == {"nearest-cell"}
    assert hrrr.SAMPLING_METHOD == "nearest-cell"


def test_a_row_is_a_frozen_transect_sample(fields: HrrrFields) -> None:
    (sample, *_) = sample_transect(fields, _transect_points([0.0]))

    assert isinstance(sample, TransectSample)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.value = 1.0  # type: ignore[misc]


def test_no_points_gives_no_rows(fields: HrrrFields) -> None:
    assert sample_transect(fields, []) == []


def test_the_default_transect_is_every_25_km_out_to_400_km() -> None:
    assert TRANSECT_DISTANCES_KM == tuple(range(0, 401, 25))
    assert len(TRANSECT_DISTANCES_KM) == 17


# --- values ---------------------------------------------------------------------------------------------


def test_the_reference_point_samples_the_cell_the_real_file_holds_there(fields: HrrrFields) -> None:
    samples = sample_transect(fields, [(REF_LAT, REF_LON)])

    assert {s.cell for s in samples} == {support.REFERENCE_CELL}
    for sample in samples:
        assert sample.value == pytest.approx(
            support.EXPECTED_AT_REFERENCE_CELL[sample.field], abs=1e-3
        )


def test_a_point_on_the_second_cell_samples_the_second_cell(fields: HrrrFields) -> None:
    point = _cell_point(fields, support.WEST_CELL)

    samples = sample_transect(fields, [point])

    assert {s.cell for s in samples} == {support.WEST_CELL}
    for sample in samples:
        assert sample.value == pytest.approx(
            support.EXPECTED_AT_WEST_CELL[sample.field], abs=1e-3
        )


def test_each_transect_point_takes_the_value_of_its_brute_force_nearest_cell(
    fields: HrrrFields,
) -> None:
    points = _transect_points([float(d) for d in TRANSECT_DISTANCES_KM])

    samples = sample_transect(fields, points)

    for sample in samples:
        lat, lon = points[sample.point_index]
        cell, distance = _nearest_cell_by_brute_force(fields, lat, lon)
        assert sample.cell == cell
        assert sample.cell_distance_km == pytest.approx(distance, abs=1e-6)
        assert sample.cell_distance_km < 2.2
        expected = fields.fields[sample.field].values[cell]
        assert sample.value == pytest.approx(float(expected))


def test_the_transect_moves_through_the_grid_so_the_samples_change(fields: HrrrFields) -> None:
    samples = sample_transect(fields, _transect_points([0.0, 100.0, 200.0, 300.0, 400.0]))

    cells = {s.cell for s in samples}
    assert len(cells) == 5
    lcdc = [s.value for s in samples if s.field == "LCDC"]
    assert len(set(lcdc)) > 1


def test_the_full_400_km_transect_lies_inside_the_crop_on_every_season(fields: HrrrFields) -> None:
    # The crop is wide enough to test the CLI's whole default transect, at any sunset azimuth.
    for month in range(1, 13):
        day = dt.date(2026, month, 15)
        points = geometry.transect(day, REF_LAT, REF_LON, [float(d) for d in TRANSECT_DISTANCES_KM])
        assert len(sample_transect(fields, points)) == 17 * len(DEFAULT_FIELDS)


# --- cells the model leaves empty -----------------------------------------------------------------------


def test_a_cell_the_model_masks_out_gives_none_with_its_units(fields: HrrrFields) -> None:
    point = _cell_point(fields, support.CORNER_CELL)  # cloud top has no value in this corner of the crop

    samples = {s.field: s for s in sample_transect(fields, [point])}

    assert samples["HGT_CLOUD_TOP"].value is None
    assert samples["HGT_CLOUD_TOP"].units == "gpm"
    assert samples["LCDC"].value is not None  # other fields are unaffected


def test_a_sample_is_never_nan(fields: HrrrFields) -> None:
    points = _transect_points([float(d) for d in TRANSECT_DISTANCES_KM]) + [
        _cell_point(fields, support.CORNER_CELL)
    ]

    for sample in sample_transect(fields, points):
        assert sample.value is None or math.isfinite(sample.value)


def test_a_nan_in_the_decoded_data_is_an_error_not_a_nan_sample(fields: HrrrFields) -> None:
    poisoned = np.ma.array(fields.fields["LCDC"].values.data.copy(), mask=False)
    poisoned[support.REFERENCE_CELL] = np.nan
    broken = dataclasses.replace(
        fields,
        fields={
            **fields.fields,
            "LCDC": dataclasses.replace(fields.fields["LCDC"], values=poisoned),
        },
    )

    with pytest.raises(HrrrError, match="LCDC"):
        sample_transect(broken, [(REF_LAT, REF_LON)])


# --- outside the grid ----------------------------------------------------------------------------------


def test_a_point_far_outside_the_grid_fails_loudly(fields: HrrrFields) -> None:
    with pytest.raises(OutsideGridError, match="outside"):
        sample_transect(fields, [(0.0, 0.0)])


def test_a_transect_that_runs_off_the_grid_fails_naming_the_first_point_that_does(
    fields: HrrrFields,
) -> None:
    points = _transect_points([0.0, 100.0, 1500.0, 2000.0])

    with pytest.raises(OutsideGridError, match="point 2"):
        sample_transect(fields, points)


def test_one_cell_beyond_the_western_edge_is_outside_the_grid(fields: HrrrFields) -> None:
    lat, lon = _cell_point(fields, support.WEST_EDGE_CELL)
    just_off = (lat, lon - 5.0 / (111.2 * math.cos(math.radians(lat))))  # 5 km further west

    with pytest.raises(OutsideGridError):
        sample_transect(fields, [just_off])


def test_a_point_within_half_a_cell_of_the_edge_cell_still_belongs_to_it(
    fields: HrrrFields,
) -> None:
    lat, lon = _cell_point(fields, support.WEST_EDGE_CELL)
    nudged = (lat, lon - 1.0 / (111.2 * math.cos(math.radians(lat))))  # 1 km further west

    samples = sample_transect(fields, [nudged])

    assert {s.cell for s in samples} == {support.WEST_EDGE_CELL}


def test_the_error_for_an_outside_point_never_returns_partial_rows(fields: HrrrFields) -> None:
    # All-or-nothing: no half-filled table for the caller to mistake for a result.
    with pytest.raises(OutsideGridError):
        sample_transect(fields, [(REF_LAT, REF_LON), (0.0, 0.0)])


@pytest.mark.parametrize(
    "point",
    [(math.nan, REF_LON), (REF_LAT, math.inf), (91.0, REF_LON), (REF_LAT, 181.0)],
)
def test_a_point_that_is_not_a_place_on_earth_is_refused(
    fields: HrrrFields, point: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="point 0"):
        sample_transect(fields, [point])


def test_longitudes_given_east_of_greenwich_as_0_to_360_are_not_accepted_silently(
    fields: HrrrFields,
) -> None:
    # 247.93 east is -112.07: a caller passing GRIB-style longitudes gets a clear refusal rather
    # than a sample from the wrong side of the world.
    with pytest.raises(ValueError, match="longitude"):
        sample_transect(fields, [(REF_LAT, 247.93)])


# --- end to end ----------------------------------------------------------------------------------------


def test_a_fresh_fetch_samples_the_same_as_the_module_fixture(tmp_path: Path) -> None:
    fresh = fetch_fields(
        support.CYCLE_DATE,
        support.CYCLE_HOUR,
        0,
        [hrrr.field_spec("PWAT")],
        tmp_path,
        transport=support.crop_transport(),
    )

    (sample,) = sample_transect(fresh, [(REF_LAT, REF_LON)])

    assert sample.field == "PWAT"
    assert sample.value == pytest.approx(support.EXPECTED_AT_REFERENCE_CELL["PWAT"], abs=1e-3)
