"""Solar geometry and the sun-elevation frame.

Every public lastlight quantity is indexed by sun elevation, never by clock time
(`docs/design.md` sections 1 and 3). This module is the one place that converts between the
two: it computes where the sun is, defines the elevation sweep and bands the forecast is
reported in, and lays out the sightline transect along the solar azimuth.

The sun's position comes from the NOAA solar-position equations (the low-order series behind
NOAA's solar calculator), written out here with the standard library only. pvlib's NREL SPA
implementation is the independent reference in the tests and is a dev dependency, never a
runtime one. Elevation is geometric: no atmospheric refraction, and no parallax (about 0.002
degrees), because twilight definitions use geometric elevation. Against SPA the NOAA series
stays within 0.02 degrees in elevation and 0.05 degrees in azimuth.

Conventions: latitude is positive north, longitude positive east, azimuth is degrees clockwise
from true north, and the Earth is a sphere of radius 6371 km for the transect.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

EARTH_RADIUS_KM: Final = 6371.0

# Sun elevations, in degrees, from +6.0 down to -8.0 inclusive in half-degree steps.
ELEVATION_SWEEP: Final[tuple[float, ...]] = tuple(6.0 - 0.5 * step for step in range(29))

# Named twilight bands as (upper, lower) sun elevation in degrees. They tile the sweep.
ELEVATION_BANDS: Final[dict[str, tuple[int, int]]] = {
    "golden": (6, 0),
    "early": (0, -2),
    "mid": (-2, -5),
    "late": (-5, -8),
}

_UNIX_EPOCH_JULIAN_DAY: Final = 2440587.5
_J2000_JULIAN_DAY: Final = 2451545.0
_SECONDS_PER_DAY: Final = 86400.0
_MINUTES_PER_DAY: Final = 1440.0

# The evening search window, in seconds either side of local mean solar noon. True solar noon
# differs from mean noon by the equation of time (under 17 minutes), and the sun's elevation
# falls monotonically from true noon to true midnight, so inside this window every target
# elevation is crossed at most once.
_WINDOW_START_S: Final = 30 * 60.0
_WINDOW_END_S: Final = 12 * 3600.0 - 30 * 60.0
_CROSSING_TOLERANCE_S: Final = 1.0


@dataclass(frozen=True, slots=True)
class SolarPosition:
    """Where the sun is: geometric elevation above the horizon and azimuth, both in degrees.

    Azimuth runs clockwise from true north: 90 is east, 180 south, 270 west.
    """

    elevation_deg: float
    azimuth_deg: float


# --- the NOAA series ----------------------------------------------------------------------------


def _mean_longitude_deg(century: float) -> float:
    return (280.46646 + century * (36000.76983 + century * 0.0003032)) % 360.0


def _mean_anomaly_deg(century: float) -> float:
    return 357.52911 + century * (35999.05029 - 0.0001537 * century)


def _orbit_eccentricity(century: float) -> float:
    return 0.016708634 - century * (0.000042037 + 0.0000001267 * century)


def _omega_rad(century: float) -> float:
    """Longitude of the Moon's ascending node, which drives nutation in the NOAA series."""
    return math.radians(125.04 - 1934.136 * century)


def _obliquity_deg(century: float) -> float:
    mean = (
        23.0
        + (26.0 + (21.448 - century * (46.815 + century * (0.00059 - century * 0.001813))) / 60.0)
        / 60.0
    )
    return mean + 0.00256 * math.cos(_omega_rad(century))


def _declination_rad(century: float) -> float:
    anomaly = math.radians(_mean_anomaly_deg(century))
    center = (
        math.sin(anomaly) * (1.914602 - century * (0.004817 + 0.000014 * century))
        + math.sin(2.0 * anomaly) * (0.019993 - 0.000101 * century)
        + math.sin(3.0 * anomaly) * 0.000289
    )
    true_longitude = _mean_longitude_deg(century) + center
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * math.sin(_omega_rad(century))
    return math.asin(
        math.sin(math.radians(_obliquity_deg(century))) * math.sin(math.radians(apparent_longitude))
    )


def _equation_of_time_minutes(century: float) -> float:
    """Apparent minus mean solar time, in minutes (positive when the sun runs ahead of the clock).

    The tests replace this function with its negation to prove the pvlib comparison fails on a
    sign error, so `solar_position` must keep reading it through this module-level name.
    """
    y = math.tan(math.radians(_obliquity_deg(century)) / 2.0) ** 2
    longitude = math.radians(_mean_longitude_deg(century))
    anomaly = math.radians(_mean_anomaly_deg(century))
    eccentricity = _orbit_eccentricity(century)
    return 4.0 * math.degrees(
        y * math.sin(2.0 * longitude)
        - 2.0 * eccentricity * math.sin(anomaly)
        + 4.0 * eccentricity * y * math.sin(anomaly) * math.cos(2.0 * longitude)
        - 0.5 * y * y * math.sin(4.0 * longitude)
        - 1.25 * eccentricity * eccentricity * math.sin(2.0 * anomaly)
    )


def _position_at(epoch_s: float, lat: float, lon: float) -> SolarPosition:
    """Sun position at a Unix timestamp. Inputs are already validated."""
    century = (_UNIX_EPOCH_JULIAN_DAY + epoch_s / _SECONDS_PER_DAY - _J2000_JULIAN_DAY) / 36525.0
    declination = _declination_rad(century)

    utc_minutes = (epoch_s % _SECONDS_PER_DAY) / 60.0
    solar_minutes = (
        utc_minutes + _equation_of_time_minutes(century) + 4.0 * lon
    ) % _MINUTES_PER_DAY
    hour_angle = math.radians(solar_minutes / 4.0 - 180.0)

    phi = math.radians(lat)
    # The sun's direction in the local horizon frame. Writing the azimuth as an atan2 of the
    # east and north components is the same quantity as NOAA's acos form, without its branch
    # cut and without a division by zero at the zenith.
    east = -math.cos(declination) * math.sin(hour_angle)
    north = math.sin(declination) * math.cos(phi) - math.cos(declination) * math.cos(
        hour_angle
    ) * math.sin(phi)
    up = math.sin(declination) * math.sin(phi) + math.cos(declination) * math.cos(
        hour_angle
    ) * math.cos(phi)

    elevation = math.degrees(math.atan2(up, math.hypot(east, north)))
    azimuth = math.degrees(math.atan2(east, north)) % 360.0
    if azimuth >= 360.0:  # a tiny negative angle rounds up to exactly 360 under `%`
        azimuth = 0.0
    return SolarPosition(elevation_deg=elevation, azimuth_deg=azimuth)


def _check_location(lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be a finite number in [-90, 90] degrees, got {lat!r}")
    if not (math.isfinite(lon) and -180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be a finite number in [-180, 180] degrees, got {lon!r}")


def solar_position(when_utc: dt.datetime, lat: float, lon: float) -> SolarPosition:
    """Geometric sun elevation and azimuth at an instant, for a latitude and east longitude.

    `when_utc` must carry a timezone (any zone; it is read as the instant it names). A naive
    datetime is refused: reading it as local time would silently shift the sun by hours.
    """
    _check_location(lat, lon)
    if when_utc.tzinfo is None or when_utc.utcoffset() is None:
        raise ValueError("when_utc must be timezone-aware; a naive datetime is ambiguous")
    return _position_at(when_utc.timestamp(), lat, lon)


# --- the evening: elevation crossings and the sunset azimuth -----------------------------------


def _evening_window(date: dt.date, lon: float) -> tuple[float, float]:
    """Unix-time bounds of the setting sun on the evening of the local mean solar date."""
    if isinstance(date, dt.datetime):
        raise TypeError("date must be a datetime.date, not a datetime: the evening is the unit")
    midday_utc = dt.datetime(date.year, date.month, date.day, 12, tzinfo=dt.UTC)
    mean_noon = midday_utc - dt.timedelta(hours=lon / 15.0)
    noon_s = mean_noon.timestamp()
    return noon_s + _WINDOW_START_S, noon_s + _WINDOW_END_S


def _crossing_epoch(target_deg: float, date: dt.date, lat: float, lon: float) -> float:
    """Unix time at which the setting sun reaches `target_deg`, by bisection to the tolerance."""
    _check_location(lat, lon)
    early, late = _evening_window(date, lon)
    if _position_at(early, lat, lon).elevation_deg < target_deg or (
        _position_at(late, lat, lon).elevation_deg > target_deg
    ):
        raise ValueError(
            f"the sun does not set through {target_deg} degrees on {date} at latitude {lat}"
        )
    while late - early > _CROSSING_TOLERANCE_S:
        middle = (early + late) / 2.0
        if _position_at(middle, lat, lon).elevation_deg > target_deg:
            early = middle
        else:
            late = middle
    return (early + late) / 2.0


def elevation_crossings(date: dt.date, lat: float, lon: float) -> dict[float, dt.datetime]:
    """UTC instant the setting sun crosses each elevation in `ELEVATION_SWEEP`.

    INTERNAL USE ONLY. This is the one function in lastlight that returns clock times, and
    clock times must never reach a public artifact (`docs/design.md` section 3): use the
    instants to sample inputs and schedule work, then publish by sun elevation.

    "The evening of `date`" is the evening after local mean solar noon on that date, so the
    instants usually fall on the next UTC calendar day. Each is found by bisection on
    `solar_position` to within 1 second (the returned midpoint is within half a second).
    Keys follow the sweep, from +6.0 down to -8.0, so instants strictly increase.

    Raises ValueError if the sun does not pass through every sweep elevation that evening,
    as at high latitudes near a solstice.
    """
    return {
        target: dt.datetime.fromtimestamp(_crossing_epoch(target, date, lat, lon), dt.UTC)
        for target in ELEVATION_SWEEP
    }


def sunset_azimuth(date: dt.date, lat: float, lon: float) -> float:
    """Azimuth in degrees clockwise from north where the sun is at geometric elevation 0.

    That is the direction of the setting sun on the evening of `date`, ignoring refraction.
    """
    epoch_s = _crossing_epoch(0.0, date, lat, lon)
    return _position_at(epoch_s, lat, lon).azimuth_deg


# --- the sightline transect ----------------------------------------------------------------------


def _destination(
    lat: float, lon: float, bearing_rad: float, distance_km: float
) -> tuple[float, float]:
    """Point reached from (lat, lon) after `distance_km` along a great circle."""
    if distance_km == 0.0:
        return lat, lon
    phi = math.radians(lat)
    angular = distance_km / EARTH_RADIUS_KM
    sin_phi2 = math.sin(phi) * math.cos(angular) + math.cos(phi) * math.sin(angular) * math.cos(
        bearing_rad
    )
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    dlon = math.atan2(
        math.sin(bearing_rad) * math.sin(angular) * math.cos(phi),
        math.cos(angular) - math.sin(phi) * sin_phi2,
    )
    lon2 = (math.degrees(math.radians(lon) + dlon) + 180.0) % 360.0 - 180.0
    return math.degrees(phi2), lon2


def transect(
    date: dt.date, lat: float, lon: float, distances_km: Iterable[float]
) -> list[tuple[float, float]]:
    """Points (lat, lon) along the great circle from the start toward the sunset azimuth.

    One point per requested distance, in order, on a spherical Earth of radius 6371 km. A
    distance of 0 returns the start point exactly. Distances must be finite and not negative.
    """
    distances = list(distances_km)
    for distance in distances:
        if not (math.isfinite(distance) and distance >= 0.0):
            raise ValueError(
                f"distance must be a finite number of km, not negative, got {distance!r}"
            )
    bearing = math.radians(sunset_azimuth(date, lat, lon))
    return [_destination(lat, lon, bearing, distance) for distance in distances]
