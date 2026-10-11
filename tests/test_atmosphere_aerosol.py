"""The reduction's optional aerosol input (issue #10, conductor decision 3).

G0 reads total and dust aerosol optical depth from an archive that needs no account, because
HRRR carries smoke only. `reduce_transect` takes them as an optional `aerosol` argument. Two
things must hold: without it the reduction is exactly what the daily forecast has always
produced (pinned below against numbers and text copied from the code before the argument
existed), and with it the total replaces smoke plus the fixed background, with the pieces
recorded.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from test_atmosphere_reduction import make_samples

from lastlight.atmosphere import (
    BACKGROUND_AOD,
    AerosolColumn,
    CloudSummary,
    ReductionError,
    reduce_transect,
)
from lastlight.privacy.rules import text_hits
from lastlight.render import Atmosphere

JULY = dt.date(2026, 7, 15)
SAMPLES = {"AOTK": 0.05, "PWAT": 20.0, "LCDC": 35.0}

# reduce_transect(make_samples(AOTK=0.05, PWAT=20.0, LCDC=35.0), 2026-07-15) before the argument.
NOTES_BEFORE = (
    "v0 reduction of HRRR along the sightline, 17 points. Aerosol optical depth at 550 nm 0.110 "
    "= 0.060 fixed background + 0.050 HRRR smoke (AOTK transect mean). Ozone 296 DU, SBUV "
    "monthly climatology for 30-35N, July, 2005-2024. Precipitable water 20.0 kg m-2, recorded "
    "and not rendered. Cloud cover transect mean: low 35%, middle 20%, high 10%, total 55%; "
    "cloud base median 1500 m and top median 4000 m (HRRR geopotential metres; cloud at 17 of "
    "17 points). Cloud is recorded and not rendered in v0."
)


def column(total: float = 0.21, dust: float = 0.12) -> AerosolColumn:
    return AerosolColumn(total_aod_550=total, dust_aod_550=dust, source="GEFS-Aerosols analysis")


# --- without the argument nothing changes -----------------------------------------------------


def test_without_the_argument_the_reduction_is_what_the_daily_forecast_always_produced() -> None:
    reduction = reduce_transect(make_samples(**SAMPLES), JULY)
    assert reduction.atmosphere == Atmosphere(aod_550=0.11, ozone_du=295.6)
    assert reduction.atmosphere.aod_550 == 0.05 + BACKGROUND_AOD
    assert reduction.aod_smoke == 0.05
    assert reduction.ozone_du == 295.6
    assert reduction.pwat_kg_m2 == 20.0
    assert reduction.clouds == CloudSummary(
        points=17,
        low_pct=35.0,
        mid_pct=20.0,
        high_pct=10.0,
        total_pct=55.0,
        base_m=1500.0,
        top_m=4000.0,
        cloudy_points=17,
    )
    assert reduction.notes == NOTES_BEFORE


def test_an_explicit_none_is_the_same_as_leaving_the_argument_out() -> None:
    plain = reduce_transect(make_samples(**SAMPLES), JULY)
    explicit = reduce_transect(make_samples(**SAMPLES), JULY, aerosol=None)
    assert explicit == plain
    assert explicit.notes == NOTES_BEFORE  # byte for byte
    assert (explicit.aod_dust, explicit.aerosol_source) == (None, None)


# --- with it, the archive's total replaces smoke plus background -------------------------------


def test_the_archives_total_is_the_aerosol_optical_depth_painted() -> None:
    reduction = reduce_transect(make_samples(**SAMPLES), JULY, aerosol=column(0.21, 0.12))
    assert reduction.atmosphere == Atmosphere(aod_550=0.21, ozone_du=295.6)
    assert reduction.aod_dust == 0.12
    assert reduction.aerosol_source == "GEFS-Aerosols analysis"


def test_hrrr_smoke_is_still_recorded_but_not_added_to_the_total() -> None:
    low = reduce_transect(make_samples(AOTK=0.0, PWAT=20.0, LCDC=35.0), JULY, aerosol=column())
    smoky = reduce_transect(make_samples(AOTK=0.4, PWAT=20.0, LCDC=35.0), JULY, aerosol=column())
    assert low.atmosphere.aod_550 == smoky.atmosphere.aod_550 == 0.21
    assert (low.aod_smoke, smoky.aod_smoke) == (0.0, 0.4)


def test_everything_but_the_aerosol_is_unchanged_by_the_argument() -> None:
    plain = reduce_transect(make_samples(**SAMPLES), JULY)
    with_archive = reduce_transect(make_samples(**SAMPLES), JULY, aerosol=column())
    assert with_archive.ozone_du == plain.ozone_du
    assert with_archive.pwat_kg_m2 == plain.pwat_kg_m2
    assert with_archive.clouds == plain.clouds
    assert dataclasses.replace(with_archive.atmosphere, aod_550=0.11) == plain.atmosphere


def test_the_notes_say_where_the_aerosol_came_from_and_what_was_not_added() -> None:
    notes = reduce_transect(make_samples(**SAMPLES), JULY, aerosol=column(0.21, 0.12)).notes
    assert "0.210" in notes and "0.120" in notes and "GEFS-Aerosols analysis" in notes
    assert "0.050" in notes  # the HRRR smoke, recorded
    assert "fixed background" not in notes
    assert text_hits(notes) == []


def test_the_reduction_still_refuses_missing_hrrr_fields_when_given_an_archive() -> None:
    samples = make_samples(AOTK=None)
    with pytest.raises(ReductionError, match="AOTK"):
        reduce_transect(samples, JULY, aerosol=column())


# --- the column itself ------------------------------------------------------------------------


def test_a_column_holds_what_it_was_given() -> None:
    value = column(0.3, 0.1)
    assert (value.total_aod_550, value.dust_aod_550) == (0.3, 0.1)


@pytest.mark.parametrize(
    ("total", "dust"),
    [(-0.01, 0.0), (0.2, -0.01), (float("nan"), 0.0), (0.2, float("inf"))],
)
def test_a_column_refuses_negative_or_non_finite_optical_depths(total: float, dust: float) -> None:
    with pytest.raises(ValueError):
        AerosolColumn(total_aod_550=total, dust_aod_550=dust, source="x")


def test_dust_cannot_exceed_the_total() -> None:
    with pytest.raises(ValueError, match="dust"):
        AerosolColumn(total_aod_550=0.1, dust_aod_550=0.2, source="x")


def test_a_column_needs_a_named_source() -> None:
    with pytest.raises(ValueError, match="source"):
        AerosolColumn(total_aod_550=0.1, dust_aod_550=0.0, source=" ")
