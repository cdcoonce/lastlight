"""Solar geometry and the sun-elevation frame (issue #4).

The independent reference is pvlib's NREL SPA implementation, a dev-only dependency. Its
*geometric* `elevation` column is the comparison; `apparent_elevation` includes refraction and
would be the wrong target (twilight definitions use geometric elevation).

Test locations are the public reference point or points derived from it (mirrored, shifted) or
generic bounds; no other site coordinates are written down.
"""

from __future__ import annotations

import math
import subprocess
import sys
import tomllib
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pvlib
import pytest

from lastlight import geometry
from lastlight.geometry import (
    ELEVATION_BANDS,
    ELEVATION_SWEEP,
    SolarPosition,
    elevation_crossings,
    solar_position,
    sunset_azimuth,
    transect,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The public reference point (docs/design.md section 3): downtown Phoenix.
REF_LAT = 33.45
REF_LON = -112.07
REF_ALTITUDE_M = 340.0

ELEVATION_LIMIT_DEG = 0.02
AZIMUTH_LIMIT_DEG = 0.05
EARTH_RADIUS_KM = 6371.0

# The reference point keeps a fixed offset from UTC all year; used only to ask which local
# calendar date an instant falls on.
LOCAL_ZONE = timezone(timedelta(hours=-7))

EDGE_DATES = [
    date(2026, 1, 1),
    date(2026, 2, 28),
    date(2026, 3, 20),
    date(2026, 6, 21),
    date(2026, 9, 22),
    date(2026, 10, 10),
    date(2026, 12, 21),
    date(2026, 12, 31),
    date(2028, 2, 29),
]
EVERY_SEVENTH_DAY_OF_2026 = [date(2026, 1, 1) + timedelta(days=7 * k) for k in range(53)]


# --- independent helpers (spherical geometry written out in the test) -----------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def circular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pvlib_frame(times: list[datetime], lat: float, lon: float) -> pd.DataFrame:
    return pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex(times), lat, lon, altitude=REF_ALTITUDE_M
    )


@dataclass(frozen=True)
class OracleSample:
    when: datetime
    elevation: float
    azimuth: float


@dataclass(frozen=True)
class OracleErrors:
    max_elevation_error: float
    max_azimuth_error: float
    azimuth_samples: int

    @property
    def within_limits(self) -> bool:
        return (
            self.max_elevation_error < ELEVATION_LIMIT_DEG
            and self.max_azimuth_error < AZIMUTH_LIMIT_DEG
        )


def oracle_samples(times: list[datetime], lat: float, lon: float) -> list[OracleSample]:
    frame = pvlib_frame(times, lat, lon)
    return [
        OracleSample(when, float(elevation), float(azimuth))
        for when, elevation, azimuth in zip(
            times, frame["elevation"], frame["azimuth"], strict=True
        )
    ]


def compare_with_pvlib(
    samples: list[OracleSample], lat: float, lon: float, azimuth_floor_deg: float = -10.0
) -> OracleErrors:
    """Max elevation error over every sample; max azimuth error where the sun is above the floor."""
    max_elevation_error = 0.0
    max_azimuth_error = 0.0
    azimuth_samples = 0
    for sample in samples:
        mine = geometry.solar_position(sample.when, lat, lon)
        max_elevation_error = max(max_elevation_error, abs(mine.elevation_deg - sample.elevation))
        if sample.elevation > azimuth_floor_deg:
            azimuth_samples += 1
            max_azimuth_error = max(
                max_azimuth_error, circular_difference_deg(mine.azimuth_deg, sample.azimuth)
            )
    return OracleErrors(max_elevation_error, max_azimuth_error, azimuth_samples)


@pytest.fixture(scope="module")
def hourly_2026() -> list[OracleSample]:
    """pvlib SPA at the reference point: every hour of every 7th day of 2026 (1272 instants)."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    times = [start + timedelta(days=7 * day, hours=hour) for day in range(53) for hour in range(24)]
    return oracle_samples(times, REF_LAT, REF_LON)


# --- the pvlib oracle ---------------------------------------------------------------------------


def test_oracle_grid_is_every_hour_of_every_seventh_day_of_2026(
    hourly_2026: list[OracleSample],
) -> None:
    assert len(hourly_2026) == 53 * 24
    assert hourly_2026[0].when == datetime(2026, 1, 1, tzinfo=UTC)
    assert hourly_2026[24].when == datetime(2026, 1, 8, tzinfo=UTC)
    assert hourly_2026[-1].when == datetime(2026, 12, 31, 23, tzinfo=UTC)


def test_solar_position_matches_pvlib_spa_at_the_reference_point(
    hourly_2026: list[OracleSample],
) -> None:
    errors = compare_with_pvlib(hourly_2026, REF_LAT, REF_LON)
    assert errors.azimuth_samples > 600, "the azimuth comparison must not be vacuous"
    assert errors.max_elevation_error < ELEVATION_LIMIT_DEG
    assert errors.max_azimuth_error < AZIMUTH_LIMIT_DEG


def test_flipping_the_sign_of_the_equation_of_time_fails_the_pvlib_comparison(
    hourly_2026: list[OracleSample], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The teeth mutation: the oracle comparison must go red for a sign error in the equation
    # of time, and by far more than the limits (the error is degrees, not hundredths).
    real = geometry._equation_of_time_minutes
    assert compare_with_pvlib(hourly_2026, REF_LAT, REF_LON).within_limits
    monkeypatch.setattr(geometry, "_equation_of_time_minutes", lambda century: -real(century))

    errors = compare_with_pvlib(hourly_2026, REF_LAT, REF_LON)

    assert not errors.within_limits
    assert errors.max_elevation_error > 100 * ELEVATION_LIMIT_DEG
    assert errors.max_azimuth_error > 100 * AZIMUTH_LIMIT_DEG


@pytest.mark.parametrize(
    ("name", "lat", "lon"),
    [
        ("mirrored latitude", -REF_LAT, REF_LON),
        ("shifted longitude", REF_LAT, REF_LON + 180.0),
        ("across the antimeridian", REF_LAT, -179.95),
    ],
)
def test_solar_position_also_matches_pvlib_away_from_the_reference_point(
    name: str, lat: float, lon: float
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    times = [
        start + timedelta(days=13 * day, hours=3 * slot) for day in range(28) for slot in range(8)
    ]
    samples = oracle_samples(times, lat, lon)
    # Azimuth is ill-conditioned near the zenith; compare it where the sun is below 75 degrees.
    low_sun = [sample for sample in samples if sample.elevation < 75.0]

    errors = compare_with_pvlib(low_sun, lat, lon)

    assert errors.azimuth_samples > 100, name
    assert errors.max_elevation_error < ELEVATION_LIMIT_DEG, name
    assert errors.max_azimuth_error < AZIMUTH_LIMIT_DEG, name


def test_elevation_is_geometric_not_refracted() -> None:
    # At geometric zero pvlib's apparent elevation is about half a degree higher; ours must
    # follow the geometric column.
    crossing = elevation_crossings(date(2026, 3, 20), REF_LAT, REF_LON)[0.0]
    frame = pvlib_frame([crossing], REF_LAT, REF_LON)
    assert abs(float(frame["elevation"].iloc[0])) < ELEVATION_LIMIT_DEG
    assert float(frame["apparent_elevation"].iloc[0]) > 0.4
    assert abs(solar_position(crossing, REF_LAT, REF_LON).elevation_deg) < ELEVATION_LIMIT_DEG


# --- solar_position -----------------------------------------------------------------------------


def test_solar_position_returns_a_frozen_solar_position() -> None:
    position = solar_position(datetime(2026, 6, 21, 20, tzinfo=UTC), REF_LAT, REF_LON)
    assert isinstance(position, SolarPosition)
    with pytest.raises(FrozenInstanceError):
        position.elevation_deg = 0.0


def test_azimuth_is_degrees_clockwise_from_north_in_zero_to_360(
    hourly_2026: list[OracleSample],
) -> None:
    azimuths = [
        solar_position(sample.when, REF_LAT, REF_LON).azimuth_deg for sample in hourly_2026[:240]
    ]
    assert all(0.0 <= azimuth < 360.0 for azimuth in azimuths)


def test_sun_is_east_in_the_morning_south_at_noon_and_west_in_the_evening() -> None:
    # Local mean solar noon is 12 hours minus longitude/15 hours after the UTC date starts.
    noon = datetime(2026, 3, 20, 12, tzinfo=UTC) - timedelta(hours=REF_LON / 15.0)
    morning = solar_position(noon - timedelta(hours=4), REF_LAT, REF_LON)
    midday = solar_position(noon, REF_LAT, REF_LON)
    evening = solar_position(noon + timedelta(hours=4), REF_LAT, REF_LON)
    assert 90.0 < morning.azimuth_deg < 150.0
    assert abs(midday.azimuth_deg - 180.0) < 5.0
    assert 210.0 < evening.azimuth_deg < 270.0
    assert midday.elevation_deg > 50.0


def test_a_zone_aware_time_equals_its_utc_instant() -> None:
    utc = datetime(2026, 6, 22, 2, 17, 40, tzinfo=UTC)
    local = utc.astimezone(LOCAL_ZONE)
    assert solar_position(local, REF_LAT, REF_LON) == solar_position(utc, REF_LAT, REF_LON)


def test_a_naive_datetime_is_refused_rather_than_read_as_some_local_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        solar_position(datetime(2026, 6, 22, 2, 17), REF_LAT, REF_LON)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (90.5, REF_LON),
        (-90.5, REF_LON),
        (math.nan, REF_LON),
        (REF_LAT, 180.5),
        (REF_LAT, -180.5),
        (REF_LAT, math.nan),
        (REF_LAT, math.inf),
    ],
)
def test_out_of_range_or_non_finite_coordinates_are_refused(lat: float, lon: float) -> None:
    with pytest.raises(ValueError, match="lat|lon"):
        solar_position(datetime(2026, 6, 22, 2, tzinfo=UTC), lat, lon)


# --- the elevation frame: sweep and bands ------------------------------------------------------


def test_elevation_sweep_runs_from_plus_6_to_minus_8_in_half_degree_steps() -> None:
    assert isinstance(ELEVATION_SWEEP, tuple)
    assert len(ELEVATION_SWEEP) == 29
    assert ELEVATION_SWEEP[0] == 6.0
    assert ELEVATION_SWEEP[-1] == -8.0
    assert all(isinstance(value, float) for value in ELEVATION_SWEEP)
    assert [b - a for a, b in pairwise(ELEVATION_SWEEP)] == [-0.5] * 28
    assert 0.0 in ELEVATION_SWEEP


def test_elevation_bands_are_the_four_twilight_bands() -> None:
    assert ELEVATION_BANDS == {
        "golden": (6, 0),
        "early": (0, -2),
        "mid": (-2, -5),
        "late": (-5, -8),
    }
    assert list(ELEVATION_BANDS) == ["golden", "early", "mid", "late"]


def test_bands_tile_the_sweep_without_gaps_or_overlap() -> None:
    bands = list(ELEVATION_BANDS.values())
    assert bands[0][0] == ELEVATION_SWEEP[0]
    assert bands[-1][1] == ELEVATION_SWEEP[-1]
    for (_, lower), (upper, _) in pairwise(bands):
        assert lower == upper
    for upper, lower in bands:
        assert upper > lower
        assert upper in ELEVATION_SWEEP
        assert lower in ELEVATION_SWEEP


# --- elevation_crossings ------------------------------------------------------------------------


def test_crossings_are_keyed_by_the_sweep_in_order_and_are_utc_instants() -> None:
    crossings = elevation_crossings(date(2026, 10, 10), REF_LAT, REF_LON)
    assert list(crossings) == list(ELEVATION_SWEEP)
    for instant in crossings.values():
        assert isinstance(instant, datetime)
        assert instant.utcoffset() == timedelta(0)


@pytest.mark.parametrize("day", EVERY_SEVENTH_DAY_OF_2026, ids=str)
def test_crossings_strictly_increase_as_elevation_decreases(day: date) -> None:
    instants = list(elevation_crossings(day, REF_LAT, REF_LON).values())
    assert all(later > earlier for earlier, later in pairwise(instants))


@pytest.mark.parametrize("day", EVERY_SEVENTH_DAY_OF_2026, ids=str)
def test_solar_position_at_each_crossing_is_within_five_thousandths_of_its_target(
    day: date,
) -> None:
    for target, instant in elevation_crossings(day, REF_LAT, REF_LON).items():
        found = solar_position(instant, REF_LAT, REF_LON).elevation_deg
        assert abs(found - target) < 0.005, (day, target)


@pytest.mark.parametrize("day", EDGE_DATES, ids=str)
def test_the_true_crossing_lies_within_one_second_of_each_instant(day: date) -> None:
    one_second = timedelta(seconds=1)
    for target, instant in elevation_crossings(day, REF_LAT, REF_LON).items():
        before = solar_position(instant - one_second, REF_LAT, REF_LON).elevation_deg
        after = solar_position(instant + one_second, REF_LAT, REF_LON).elevation_deg
        assert before >= target >= after, (day, target)


@pytest.mark.parametrize("day", EDGE_DATES, ids=str)
def test_the_evening_belongs_to_the_local_date_even_when_utc_has_rolled_over(day: date) -> None:
    for target, instant in elevation_crossings(day, REF_LAT, REF_LON).items():
        assert instant.astimezone(LOCAL_ZONE).date() == day, (day, target)
        # The evening sun is in the west.
        assert 180.0 < solar_position(instant, REF_LAT, REF_LON).azimuth_deg < 360.0


@pytest.mark.parametrize("day", EDGE_DATES, ids=str)
def test_crossings_agree_with_pvlib_at_the_crossing_instants(day: date) -> None:
    crossings = elevation_crossings(day, REF_LAT, REF_LON)
    frame = pvlib_frame(list(crossings.values()), REF_LAT, REF_LON)
    for target, found in zip(crossings, frame["elevation"], strict=True):
        assert abs(float(found) - target) < ELEVATION_LIMIT_DEG, (day, target)


def test_late_crossings_roll_into_the_next_utc_month_and_year() -> None:
    # Date arithmetic must not assume the evening falls in the same UTC month or year.
    leap = elevation_crossings(date(2028, 2, 29), REF_LAT, REF_LON)
    year_end = elevation_crossings(date(2026, 12, 31), REF_LAT, REF_LON)
    assert max(leap.values()).date() == date(2028, 3, 1)
    assert max(year_end.values()).date() == date(2027, 1, 1)


def test_the_sun_that_does_not_reach_a_sweep_elevation_is_an_error() -> None:
    with pytest.raises(ValueError, match="does not"):
        elevation_crossings(date(2026, 6, 21), 80.0, REF_LON)
    with pytest.raises(ValueError, match="does not"):
        elevation_crossings(date(2026, 12, 21), 80.0, REF_LON)


def test_a_datetime_is_refused_where_a_date_is_expected() -> None:
    with pytest.raises(TypeError, match="date"):
        elevation_crossings(datetime(2026, 6, 21, 12, tzinfo=UTC), REF_LAT, REF_LON)


def test_the_docstring_says_this_is_the_one_clock_time_function() -> None:
    doc = (elevation_crossings.__doc__ or "").lower()
    assert "clock time" in doc
    assert "internal" in doc


# --- sunset_azimuth -----------------------------------------------------------------------------


def test_sunset_azimuth_is_the_azimuth_at_the_zero_elevation_crossing() -> None:
    day = date(2026, 10, 10)
    instant = elevation_crossings(day, REF_LAT, REF_LON)[0.0]
    expected = solar_position(instant, REF_LAT, REF_LON).azimuth_deg
    assert sunset_azimuth(day, REF_LAT, REF_LON) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("day", "declination_deg", "tolerance"),
    [
        (date(2026, 6, 21), 23.44, 0.1),
        (date(2026, 12, 21), -23.44, 0.1),
        (date(2026, 3, 20), 0.0, 0.5),
        (date(2026, 9, 22), 0.0, 0.5),
    ],
    ids=["june solstice", "december solstice", "march equinox", "september equinox"],
)
def test_sunset_azimuth_matches_the_closed_form_at_solstices_and_equinoxes(
    day: date, declination_deg: float, tolerance: float
) -> None:
    # Horizon-crossing azimuth from north: 360 - acos(sin(declination) / cos(latitude)).
    ratio = math.sin(math.radians(declination_deg)) / math.cos(math.radians(REF_LAT))
    expected = 360.0 - math.degrees(math.acos(ratio))
    assert sunset_azimuth(day, REF_LAT, REF_LON) == pytest.approx(expected, abs=tolerance)


def test_sunset_azimuth_swings_northwest_in_summer_and_southwest_in_winter() -> None:
    azimuths = {day: sunset_azimuth(day, REF_LAT, REF_LON) for day in EVERY_SEVENTH_DAY_OF_2026}
    assert max(azimuths.values()) < 299.0
    assert min(azimuths.values()) > 241.0
    assert azimuths[date(2026, 6, 25)] > 295.0
    assert azimuths[date(2026, 12, 24)] < 245.0


def test_sunset_azimuth_agrees_with_pvlib_at_the_crossing_instant() -> None:
    for day in EVERY_SEVENTH_DAY_OF_2026[::4]:
        instant = elevation_crossings(day, REF_LAT, REF_LON)[0.0]
        oracle = float(pvlib_frame([instant], REF_LAT, REF_LON)["azimuth"].iloc[0])
        assert circular_difference_deg(sunset_azimuth(day, REF_LAT, REF_LON), oracle) < (
            AZIMUTH_LIMIT_DEG
        ), day


# --- transect -----------------------------------------------------------------------------------

DISTANCES_KM = [0.0, 1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]


def test_transect_at_distance_zero_is_the_start_point() -> None:
    assert transect(date(2026, 10, 10), REF_LAT, REF_LON, [0.0]) == [(REF_LAT, REF_LON)]


def test_transect_returns_one_point_per_distance_in_order() -> None:
    points = transect(date(2026, 10, 10), REF_LAT, REF_LON, DISTANCES_KM)
    assert len(points) == len(DISTANCES_KM)
    assert all(isinstance(point, tuple) and len(point) == 2 for point in points)
    assert points[0] == (REF_LAT, REF_LON)
    assert transect(date(2026, 10, 10), REF_LAT, REF_LON, []) == []


@pytest.mark.parametrize("day", EDGE_DATES, ids=str)
def test_transect_distances_agree_with_haversine_within_a_tenth_of_a_percent(day: date) -> None:
    for requested, (lat, lon) in zip(
        DISTANCES_KM, transect(day, REF_LAT, REF_LON, DISTANCES_KM), strict=True
    ):
        measured = haversine_km(REF_LAT, REF_LON, lat, lon)
        assert measured == pytest.approx(requested, rel=0.001, abs=1e-9), (day, requested)


@pytest.mark.parametrize("day", EDGE_DATES, ids=str)
def test_transect_points_lie_on_the_great_circle_toward_the_sunset_azimuth(day: date) -> None:
    azimuth = sunset_azimuth(day, REF_LAT, REF_LON)
    for lat, lon in transect(day, REF_LAT, REF_LON, DISTANCES_KM[1:]):
        bearing = initial_bearing_deg(REF_LAT, REF_LON, lat, lon)
        assert circular_difference_deg(bearing, azimuth) < 0.001, (day, lat, lon)


def test_transect_runs_west_of_the_start_all_year() -> None:
    for day in EVERY_SEVENTH_DAY_OF_2026:
        points = transect(day, REF_LAT, REF_LON, [100.0, 500.0])
        assert all(lon < REF_LON for _, lon in points), day


def test_transect_leans_north_in_summer_and_south_in_winter() -> None:
    summer = transect(date(2026, 6, 21), REF_LAT, REF_LON, [300.0])[0]
    winter = transect(date(2026, 12, 21), REF_LAT, REF_LON, [300.0])[0]
    assert summer[0] > REF_LAT
    assert winter[0] < REF_LAT


def test_transect_consecutive_spacing_matches_the_requested_spacing() -> None:
    points = transect(date(2026, 10, 10), REF_LAT, REF_LON, [0.0, 100.0, 200.0, 300.0])
    gaps = [haversine_km(*a, *b) for a, b in pairwise(points)]
    assert gaps == pytest.approx([100.0, 100.0, 100.0], rel=0.001)


def test_transect_longitude_wraps_at_the_antimeridian() -> None:
    lat, lon = REF_LAT, -179.95
    points = transect(date(2026, 3, 20), lat, lon, [0.0, 100.0, 500.0])
    assert all(-180.0 <= point_lon < 180.0 for _, point_lon in points)
    assert points[-1][1] > 0.0, "heading west from just east of the antimeridian wraps to +lon"
    assert haversine_km(lat, lon, *points[-1]) == pytest.approx(500.0, rel=0.001)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_transect_refuses_negative_or_non_finite_distances(bad: float) -> None:
    with pytest.raises(ValueError, match="distance"):
        transect(date(2026, 10, 10), REF_LAT, REF_LON, [10.0, bad])


# --- packaging and import hygiene --------------------------------------------------------------


def test_pvlib_is_a_dev_dependency_only() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    runtime = [
        item.split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for item in pyproject["project"]["dependencies"]
    ]
    assert not {"pvlib", "astropy"} & set(runtime)
    assert any(item.startswith("pvlib") for item in pyproject["dependency-groups"]["dev"])


def test_importing_geometry_pulls_in_no_numeric_stack_or_oracle() -> None:
    code = (
        "import sys, lastlight.geometry\n"
        "heavy = {'pvlib', 'astropy', 'numpy', 'pandas', 'scipy'} & set(sys.modules)\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
