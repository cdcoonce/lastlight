"""From HRRR transect samples to the renderer's v0 parameters (issue #8, `lastlight.atmosphere`).

The reduction is the part of the daily forecast that turns model fields into physics inputs, and
the G0 spread study (#10) will call it per archived date, so it is tested on its own: the AOD
sum, the ozone table (and where its numbers came from), the cloud summary and the notes text.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import hrrr_support as support
import pytest

from lastlight import geometry
from lastlight.atmosphere import (
    BACKGROUND_AOD,
    OZONE_DU_BY_MONTH,
    OZONE_TABLE_PERIOD,
    ReductionError,
    ozone_du,
    reduce_transect,
)
from lastlight.inputs import hrrr
from lastlight.inputs.hrrr import TransectSample
from lastlight.privacy.rules import text_hits
from lastlight.render import Atmosphere
from lastlight.site import REFERENCE_POINT

EVENING = dt.date(2026, 10, 10)
OZONE_SOURCE = Path(__file__).parent / "fixtures" / "ozone" / "sbuv_v87_mod_30-35N_2005-2024.txt"

UNITS = {
    "LCDC": "%",
    "MCDC": "%",
    "HCDC": "%",
    "TCDC": "%",
    "AOTK": "1",
    "PWAT": "kg m-2",
    "VIS": "m",
    "HGT_CLOUD_BASE": "gpm",
    "HGT_CLOUD_TOP": "gpm",
}
BASELINE: dict[str, float | None] = {
    "LCDC": 40.0,
    "MCDC": 20.0,
    "HCDC": 10.0,
    "TCDC": 55.0,
    "AOTK": 0.0,
    "PWAT": 12.0,
    "VIS": 20000.0,
    "HGT_CLOUD_BASE": 1500.0,
    "HGT_CLOUD_TOP": 4000.0,
}


def make_samples(
    points: int = 17, /, **per_field: float | None | list[float | None]
) -> list[TransectSample]:
    """`points` transect points with every default field; a field may be one value or a list."""
    samples = []
    for index in range(points):
        for name, default in BASELINE.items():
            given = per_field.get(name, default)
            value = given[index] if isinstance(given, list) else given
            samples.append(
                TransectSample(
                    point_index=index,
                    field=name,
                    lat=33.45,
                    lon=-112.07,
                    value=value,
                    units=UNITS[name],
                    method="nearest-cell",
                    cell=(0, 0),
                    cell_distance_km=1.0,
                )
            )
    return samples


# --- ozone: a table with a provenance -----------------------------------------------------


def source_climatology() -> list[float]:
    """Monthly means of the committed source rows, ignoring 0.0 (no data), to 0.1 DU."""
    columns: list[list[float]] = [[] for _ in range(12)]
    for line in OZONE_SOURCE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        year, south, north, *months = line.split()
        assert (south, north) == ("30", "35") and len(months) == 12
        for column, value in zip(columns, months, strict=True):
            if float(value) > 0.0:
                column.append(float(value))
    return [round(sum(column) / len(column), 1) for column in columns]


def test_the_ozone_table_is_the_monthly_mean_of_the_committed_nasa_rows() -> None:
    assert list(OZONE_DU_BY_MONTH) == source_climatology()


def test_the_table_names_the_period_its_rows_cover() -> None:
    assert OZONE_TABLE_PERIOD == (2005, 2024)
    years = [
        int(line.split()[0])
        for line in OZONE_SOURCE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert (min(years), max(years)) == OZONE_TABLE_PERIOD


def test_the_table_has_twelve_plausible_values_with_the_spring_maximum_and_autumn_minimum() -> None:
    assert len(OZONE_DU_BY_MONTH) == 12
    assert all(250.0 < value < 350.0 for value in OZONE_DU_BY_MONTH)
    assert OZONE_DU_BY_MONTH.index(max(OZONE_DU_BY_MONTH)) == 3  # April
    assert OZONE_DU_BY_MONTH.index(min(OZONE_DU_BY_MONTH)) == 10  # November


@pytest.mark.parametrize(
    ("month", "expected"),
    [(1, 286.6), (4, 317.5), (6, 303.2), (10, 272.4), (11, 270.5), (12, 276.9)],
)
def test_ozone_du_looks_up_the_month(month: int, expected: float) -> None:
    assert ozone_du(month) == expected


@pytest.mark.parametrize("month", [0, 13, -1, 1.5, True, "10", None])
def test_ozone_du_refuses_anything_but_a_month_number(month: object) -> None:
    with pytest.raises(ValueError, match="month"):
        ozone_du(month)  # type: ignore[arg-type]


# --- the aerosol optical depth ------------------------------------------------------------


def test_the_background_aod_is_the_fixed_006() -> None:
    assert BACKGROUND_AOD == 0.06


def test_with_no_smoke_the_aod_is_the_background_alone() -> None:
    result = reduce_transect(make_samples(AOTK=0.0), EVENING)
    assert result.aod_smoke == 0.0
    assert result.atmosphere.aod_550 == pytest.approx(0.06)


def test_aod_is_the_transect_mean_of_aotk_plus_the_background() -> None:
    aotk: list[float | None] = [0.1, 0.3] + [0.2] * 15  # mean 0.2
    result = reduce_transect(make_samples(AOTK=aotk), EVENING)
    assert result.aod_smoke == pytest.approx(0.2)
    assert result.atmosphere.aod_550 == pytest.approx(0.26)


def test_the_mean_does_not_depend_on_the_order_of_the_samples() -> None:
    aotk: list[float | None] = [0.0, 0.4] + [0.1] * 15
    samples = make_samples(AOTK=aotk)
    forward = reduce_transect(samples, EVENING)
    backward = reduce_transect(list(reversed(samples)), EVENING)
    assert forward == backward


def test_points_without_an_aotk_value_are_left_out_of_the_mean() -> None:
    aotk: list[float | None] = [None, None, 0.3] + [0.3] * 14
    result = reduce_transect(make_samples(AOTK=aotk), EVENING)
    assert result.aod_smoke == pytest.approx(0.3)


def test_no_aotk_anywhere_is_an_error_not_clean_air() -> None:
    with pytest.raises(ReductionError, match="AOTK"):
        reduce_transect(make_samples(AOTK=None), EVENING)


def test_a_transect_without_the_aotk_field_is_an_error() -> None:
    samples = [sample for sample in make_samples() if sample.field != "AOTK"]
    with pytest.raises(ReductionError, match="AOTK"):
        reduce_transect(samples, EVENING)


def test_an_empty_transect_is_an_error() -> None:
    with pytest.raises(ReductionError):
        reduce_transect([], EVENING)


def test_low_cloud_cover_is_required_because_the_vividness_rule_reads_it() -> None:
    with pytest.raises(ReductionError, match="LCDC"):
        reduce_transect(make_samples(LCDC=None), EVENING)


# --- the renderer's parameters ------------------------------------------------------------


def test_the_atmosphere_has_the_computed_aod_and_the_months_ozone_and_nothing_else_changed() -> (
    None
):
    result = reduce_transect(make_samples(AOTK=0.05), dt.date(2026, 4, 3))
    atmosphere = result.atmosphere
    assert atmosphere.aod_550 == pytest.approx(0.11)
    assert atmosphere.ozone_du == 317.5 == result.ozone_du
    # Every other parameter is the renderer's own default, so the v0 forecast makes no claim
    # about pressure, the aerosol's slope or its shape that nothing in the inputs supports.
    assert dataclasses.replace(atmosphere, aod_550=0.0, ozone_du=280) == Atmosphere(aod_550=0.0)


def test_ozone_follows_the_month_of_the_evening() -> None:
    october = reduce_transect(make_samples(), dt.date(2026, 10, 31))
    november = reduce_transect(make_samples(), dt.date(2026, 11, 1))
    assert (october.ozone_du, november.ozone_du) == (272.4, 270.5)


def test_precipitable_water_is_recorded_but_the_renderer_has_no_place_for_it() -> None:
    result = reduce_transect(make_samples(PWAT=20.0), EVENING)
    assert result.pwat_kg_m2 == pytest.approx(20.0)
    assert not any("pwat" in name or "water" in name for name in vars(result.atmosphere))


# --- the cloud summary --------------------------------------------------------------------


def test_cloud_cover_is_summarised_as_transect_means() -> None:
    low: list[float | None] = [0.0, 100.0] + [50.0] * 15
    result = reduce_transect(make_samples(LCDC=low, MCDC=30.0, HCDC=60.0, TCDC=90.0), EVENING)
    clouds = result.clouds
    assert clouds.points == 17
    assert clouds.low_pct == pytest.approx(sum(v for v in low if v is not None) / 17)
    assert (clouds.mid_pct, clouds.high_pct, clouds.total_pct) == (30.0, 60.0, 90.0)


def test_cloud_base_and_top_are_medians_over_the_points_that_have_cloud() -> None:
    base: list[float | None] = [None] * 10 + [
        1000.0,
        2000.0,
        3000.0,
        4000.0,
        5000.0,
        6000.0,
        7000.0,
    ]
    top: list[float | None] = [None] * 10 + [8000.0] * 7
    clouds = reduce_transect(make_samples(HGT_CLOUD_BASE=base, HGT_CLOUD_TOP=top), EVENING).clouds
    assert clouds.base_m == 4000.0
    assert clouds.top_m == 8000.0
    assert clouds.cloudy_points == 7


def test_a_clear_transect_has_no_cloud_base_or_top() -> None:
    clouds = reduce_transect(
        make_samples(HGT_CLOUD_BASE=None, HGT_CLOUD_TOP=None, LCDC=0.0, TCDC=0.0), EVENING
    ).clouds
    assert clouds.base_m is None and clouds.top_m is None and clouds.cloudy_points == 0


# --- the notes ----------------------------------------------------------------------------


def test_the_notes_say_what_was_used_and_what_was_not_rendered() -> None:
    notes = reduce_transect(make_samples(AOTK=0.05, PWAT=20.0), EVENING).notes
    assert "0.060" in notes and "0.050" in notes and "0.110" in notes  # background, smoke, total
    assert "272" in notes and "DU" in notes  # October ozone
    assert "SBUV" in notes and "2005" in notes
    assert "20.0 kg m-2" in notes
    assert "not rendered" in notes
    assert "low 40%" in notes and "total 55%" in notes
    assert "1500 m" in notes and "4000 m" in notes


def test_the_notes_for_clear_air_say_there_is_no_cloud_base() -> None:
    notes = reduce_transect(
        make_samples(HGT_CLOUD_BASE=None, HGT_CLOUD_TOP=None, LCDC=0.0, TCDC=0.0), EVENING
    ).notes
    assert "no cloud base" in notes


@pytest.mark.parametrize("pwat", [0.0, 12.5, 31.0, 33.5, 37.4, 44.4])
@pytest.mark.parametrize("aotk", [0.0, 0.034, 1.2])
def test_the_notes_never_trip_the_privacy_text_rules(pwat: float, aotk: float) -> None:
    # The notes are sealed text, and the forecast schema refuses anything the scanner would flag;
    # decimals that happen to fall in the Arizona coordinate box (water vapour of 31 to 37) must
    # not be read as a position.
    notes = reduce_transect(make_samples(PWAT=pwat, AOTK=aotk), EVENING).notes
    assert text_hits(notes) == []


def test_the_notes_are_deterministic_ascii_without_newlines() -> None:
    first = reduce_transect(make_samples(AOTK=0.05), EVENING).notes
    assert first == reduce_transect(make_samples(AOTK=0.05), EVENING).notes
    assert first.isascii() and "\n" not in first


# --- the real fixture ---------------------------------------------------------------------


def test_the_recorded_hrrr_crop_reduces_to_a_physical_atmosphere(tmp_path: Path) -> None:
    lat, lon, _elevation = REFERENCE_POINT
    fields = hrrr.fetch_fields(
        support.CYCLE_DATE, support.CYCLE_HOUR, 0, hrrr.DEFAULT_FIELDS, tmp_path,
        transport=support.crop_transport(),
    )  # fmt: skip
    distances = [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
    samples = hrrr.sample_transect(fields, geometry.transect(EVENING, lat, lon, distances))

    result = reduce_transect(samples, EVENING)

    assert result.clouds.points == 17
    assert 0.06 <= result.atmosphere.aod_550 < 0.5
    assert result.ozone_du == 272.4
    assert 0.0 <= result.clouds.total_pct <= 100.0
    assert result.pwat_kg_m2 is not None and result.pwat_kg_m2 > 0.0


# --- reuse --------------------------------------------------------------------------------


def test_importing_the_reduction_does_not_load_the_grib_library() -> None:
    # The spread study and the CLI import this module; neither should pay for ecCodes.
    code = (
        "import sys, json; import lastlight.atmosphere; print(json.dumps('eccodes' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is False
