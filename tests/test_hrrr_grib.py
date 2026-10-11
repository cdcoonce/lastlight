"""Decoding one GRIB2 message from the recorded crop (issue #6).

The expected values are what the real HRRR file holds at two grid cells, decoded from the full
grid before it was cropped; `hrrr_support.py` records where they come from.
"""

from __future__ import annotations

import math

import hrrr_support as support
import numpy as np
import pytest

from lastlight.inputs.grib import GribError, decode_message, grid_coordinates

REF_LAT = 33.45
REF_LON = -112.07

# (discipline, parameter category, parameter number, type of first fixed surface)
IDENTITIES = {
    "LCDC": (0, 6, 3, 214),
    "MCDC": (0, 6, 4, 224),
    "HCDC": (0, 6, 5, 234),
    "TCDC": (0, 6, 1, 10),
    "AOTK": (0, 20, 102, 200),
    "PWAT": (0, 1, 3, 200),
    "VIS": (0, 19, 0, 1),
    "HGT_CLOUD_BASE": (0, 3, 5, 2),
    "HGT_CLOUD_TOP": (0, 3, 5, 3),
}


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = math.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(
        np.radians(lon2 - lon1) / 2
    ) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# --- values at known cells ------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(support.EXPECTED_AT_REFERENCE_CELL))
def test_every_default_field_decodes_to_the_real_value_at_the_reference_cell(name: str) -> None:
    decoded = decode_message(support.crop_message(name))

    value = decoded.values[support.REFERENCE_CELL]

    assert not np.ma.is_masked(value)
    assert float(value) == pytest.approx(support.EXPECTED_AT_REFERENCE_CELL[name], abs=1e-3)


@pytest.mark.parametrize("name", list(support.EXPECTED_AT_WEST_CELL))
def test_every_default_field_decodes_to_the_real_value_at_a_second_cell_to_the_west(
    name: str,
) -> None:
    decoded = decode_message(support.crop_message(name))

    value = decoded.values[support.WEST_CELL]

    assert not np.ma.is_masked(value)
    assert float(value) == pytest.approx(support.EXPECTED_AT_WEST_CELL[name], abs=1e-3)


def test_the_two_cells_differ_so_a_wrong_cell_cannot_pass_by_luck() -> None:
    differing = [
        name
        for name in support.EXPECTED_AT_REFERENCE_CELL
        if support.EXPECTED_AT_REFERENCE_CELL[name] != support.EXPECTED_AT_WEST_CELL[name]
    ]
    assert len(differing) >= 6


def test_the_crop_is_a_small_sub_grid_well_inside_the_two_megabyte_budget() -> None:
    assert len(support.CROP_GRIB) <= 2 * 1024 * 1024
    decoded = decode_message(support.crop_message("LCDC"))
    assert decoded.values.shape == support.CROP_SHAPE


# --- identity and grid ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(IDENTITIES))
def test_a_message_reports_which_parameter_it_is(name: str) -> None:
    decoded = decode_message(support.crop_message(name))

    assert (
        decoded.discipline,
        decoded.category,
        decoded.number,
        decoded.surface_type,
    ) == IDENTITIES[name]


def test_a_message_reports_its_reference_time_and_forecast_step() -> None:
    decoded = decode_message(support.crop_message("PWAT"))

    assert (decoded.reference_date, decoded.reference_hour) == (support.CYCLE_DATE, 22)
    assert decoded.step_hours == 0


def test_the_grid_spec_is_the_3_km_lambert_grid_and_is_shared_by_every_field() -> None:
    specs = {decode_message(support.crop_message(name)).grid for name in IDENTITIES}

    assert len(specs) == 1
    (spec,) = specs
    assert (spec.ni, spec.nj) == (151, 141)
    assert spec.dx_m == pytest.approx(3000.0)
    assert spec.dy_m == pytest.approx(3000.0)


def test_grid_coordinates_put_the_reference_cell_on_the_public_reference_point() -> None:
    lats, lons = grid_coordinates(support.crop_message("LCDC"))

    assert lats.shape == lons.shape == support.CROP_SHAPE
    distance = _haversine_km(REF_LAT, REF_LON, lats, lons)
    assert np.unravel_index(np.argmin(distance), distance.shape) == support.REFERENCE_CELL
    assert distance[support.REFERENCE_CELL] < 2.2  # inside half a 3 km cell diagonal


def test_grid_longitudes_are_signed_degrees_east_in_minus_180_to_180() -> None:
    lats, lons = grid_coordinates(support.crop_message("LCDC"))

    assert lons.max() < 0.0  # Arizona is west of Greenwich: -180..180, not 0..360
    assert lons.min() > -180.0
    assert lats.min() > 0.0


def test_grid_cells_are_3_km_apart() -> None:
    lats, lons = grid_coordinates(support.crop_message("LCDC"))
    j, i = support.REFERENCE_CELL

    east = _haversine_km(lats[j, i], lons[j, i], lats[j, i + 1 : i + 2], lons[j, i + 1 : i + 2])
    north = _haversine_km(lats[j, i], lons[j, i], lats[j + 1 : j + 2, i], lons[j + 1 : j + 2, i])

    assert east[0] == pytest.approx(3.0, abs=0.05)
    assert north[0] == pytest.approx(3.0, abs=0.05)


def test_the_grid_runs_west_to_east_and_south_to_north() -> None:
    lats, lons = grid_coordinates(support.crop_message("LCDC"))

    row, column = support.REFERENCE_CELL
    assert (np.diff(lons[row]) > 0).all()
    assert (np.diff(lats[:, column]) > 0).all()


# --- masked cells -------------------------------------------------------------------------------------


def test_cells_the_file_masks_out_are_masked_not_a_sentinel_number() -> None:
    # Cloud top has no value where there is no cloud; the file marks them with a bitmap and a
    # missing-value code (9999). That code must never escape as if it were a height.
    decoded = decode_message(support.crop_message("HGT_CLOUD_TOP"))

    assert int(np.ma.getmaskarray(decoded.values).sum()) == support.CLOUD_TOP_MASKED_CELLS
    assert np.ma.is_masked(decoded.values[support.CORNER_CELL])
    assert 9999.0 not in decoded.values.compressed()
    assert decoded.values.compressed().min() > 0.0


def test_a_field_without_a_bitmap_has_no_masked_cells() -> None:
    decoded = decode_message(support.crop_message("LCDC"))

    assert not np.ma.getmaskarray(decoded.values).any()


def test_values_are_plain_finite_floats_where_not_masked() -> None:
    for name in IDENTITIES:
        decoded = decode_message(support.crop_message(name))
        assert np.isfinite(decoded.values.compressed()).all()


# --- bad input ----------------------------------------------------------------------------------------


def test_bytes_that_are_not_grib_are_refused() -> None:
    with pytest.raises(GribError):
        decode_message(b"this is not a grib message at all")


def test_empty_bytes_are_refused() -> None:
    with pytest.raises(GribError):
        decode_message(b"")


def test_a_truncated_message_is_refused() -> None:
    whole = support.crop_message("PWAT")

    with pytest.raises(GribError):
        decode_message(whole[: len(whole) // 2])


def test_two_messages_in_one_buffer_are_refused() -> None:
    both = support.crop_message("LCDC") + support.crop_message("MCDC")

    with pytest.raises(GribError, match="one"):
        decode_message(both)


def test_grid_coordinates_refuse_bytes_that_are_not_grib() -> None:
    with pytest.raises(GribError):
        grid_coordinates(b"nope")
