"""Single scattering of sunlight in a spherical-shell atmosphere.

The observer stands on the Earth's surface at the top of a sphere of radius `EARTH_RADIUS_KM`,
looking along the sun's azimuth (or a given angle from it) at a chosen elevation. Sunlight is
scattered once, toward the observer, by the air and aerosol along the view ray, and is weakened
twice on the way: from the top of the atmosphere to the scattering point, and from there to the
observer.

The atmosphere is a stack of concentric shells (`LEVELS_KM`). Each shell is a layer of uniform
air with the mean density of the real profile in it, so the optical depth of any straight path is
the sum over shells of path length in the shell times the shell's extinction per km. The
sunward path from a scattering point is cut by the Earth: if it crosses the ground the point is
in Earth's shadow and receives no direct sunlight. That one test makes twilight.

Units: the result is radiance in W m^-2 sr^-1 nm^-1 for the extraterrestrial sun at mean
distance, with no ground reflection and no multiple scattering.
"""

from __future__ import annotations

import math

import numpy as np

from lastlight.render.atmosphere import (
    EARTH_RADIUS_KM,
    LEVELS_KM,
    WAVELENGTHS_NM,
    Atmosphere,
    layer_optics,
    solar_irradiance,
)

_SHELL_RADII_KM = EARTH_RADIUS_KM + LEVELS_KM


# --- phase functions -----------------------------------------------------------------------------


def rayleigh_phase(cos_theta: np.ndarray) -> np.ndarray:
    """Rayleigh phase function, per steradian: 3 / (16 pi) * (1 + cos^2 theta)."""
    cos_theta = np.asarray(cos_theta, dtype=float)
    return 3.0 / (16.0 * math.pi) * (1.0 + cos_theta**2)


def henyey_greenstein_phase(cos_theta: np.ndarray, g: float) -> np.ndarray:
    """Henyey-Greenstein phase function, per steradian, with asymmetry parameter `g`."""
    cos_theta = np.asarray(cos_theta, dtype=float)
    return (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5)


# --- geometry ------------------------------------------------------------------------------------


def sunlit(radius_km: np.ndarray, sun_dot_km: np.ndarray) -> np.ndarray:
    """True where the straight path from a point toward the sun does not cross the Earth.

    `radius_km` is the point's distance from the Earth's centre and `sun_dot_km` the dot product
    of its position vector with the unit vector pointing at the sun. The path toward the sun
    intersects the Earth's sphere when it heads inward (negative dot product) and its closest
    approach to the centre is inside the sphere: dot^2 > radius^2 - EARTH_RADIUS^2 is the
    discriminant of the ray-sphere intersection.
    """
    discriminant = sun_dot_km**2 - (radius_km**2 - EARTH_RADIUS_KM**2)
    return ~((sun_dot_km < 0.0) & (discriminant > 0.0))


def shell_path_lengths(
    radius_km: np.ndarray, sun_dot_km: np.ndarray, shell_radii_km: np.ndarray
) -> np.ndarray:
    """Length of an outward straight path inside each shell, in km.

    Each path starts at a point `radius_km` from the Earth's centre and runs in a fixed
    direction, with `sun_dot_km` the dot product of the start position and that direction. It
    continues to infinity; shells beyond the atmosphere's top are not asked for. For a sphere of
    radius r the length of the path inside it is, with discriminant d = dot^2 - (start^2 - r^2):

    * starting inside the sphere: -dot + sqrt(d);
    * starting outside, heading inward (dot < 0) and hitting it: 2 sqrt(d);
    * otherwise 0.

    The length in the shell between two spheres is the difference of the two. Result shape is
    (points, shells).
    """
    start = radius_km[:, None]
    dot = sun_dot_km[:, None]
    discriminant = dot**2 - (start**2 - shell_radii_km[None, :] ** 2)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    inside_start = start < shell_radii_km[None, :]
    crosses = (dot < 0.0) & (discriminant > 0.0)
    inside_length = np.where(inside_start, -dot + root, np.where(crosses, 2.0 * root, 0.0))
    return np.diff(inside_length, axis=1)


def _distance_to_height(height_km: np.ndarray, sin_view: float) -> np.ndarray:
    """Distance along a ray leaving the observer at elevation `view` to reach a given height.

    Solves |origin + t * direction| = R + h for t, with the origin on the surface:
    t^2 + 2 R sin(view) t - (2 R h + h^2) = 0.
    """
    lift = EARTH_RADIUS_KM * sin_view
    return -lift + np.sqrt(lift**2 + 2.0 * EARTH_RADIUS_KM * height_km + height_km**2)


def scattering_cosine(
    sun_elevation_deg: float, view_elevation_deg: float, rel_azimuth_deg: float
) -> float:
    """Cosine of the angle between the direction of the sun and the line of sight.

    Spherical geometry on the sky: the sun is at elevation `sun_elevation_deg` and azimuth 0, the
    view at `view_elevation_deg` and `rel_azimuth_deg` round from it, so
    cos(theta) = sin(view) sin(sun) + cos(view) cos(sun) cos(azimuth). It is even in the azimuth,
    so a view to the left of the sun and the mirror view to the right are the same sky. Looking
    straight at the sun is 1; over the zenith, 30 degrees up on each side, it is cos(120) = -1/2.
    """
    sun = math.radians(sun_elevation_deg)
    view = math.radians(view_elevation_deg)
    azimuth = math.radians(rel_azimuth_deg)
    return math.cos(view) * math.cos(azimuth) * math.cos(sun) + math.sin(view) * math.sin(sun)


# --- the integral --------------------------------------------------------------------------------


def sky_spectrum(
    atm: Atmosphere,
    sun_elevation_deg: float,
    view_elevation_deg: float,
    rel_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """Spectral radiance of the sky in one direction, at `WAVELENGTHS_NM`.

    `sun_elevation_deg` is the sun's geometric elevation at the observer (negative when the sun
    is below the horizon), `view_elevation_deg` the elevation of the line of sight (0 to 90) and
    `rel_azimuth_deg` its azimuth relative to the sun's: 0 looks toward the sun.

    The line of sight is cut at the shell boundaries. For the middle of each piece the sunlight
    that reaches it (zero in Earth's shadow), the share scattered toward the observer, and the
    transmittance back to the observer are multiplied and summed.
    """
    if not 0.0 <= view_elevation_deg <= 90.0:
        raise ValueError(f"view_elevation_deg must be in [0, 90], got {view_elevation_deg!r}")
    optics = layer_optics(atm)
    extinction = optics.extinction

    sun = math.radians(sun_elevation_deg)
    view = math.radians(view_elevation_deg)
    cos_theta = scattering_cosine(sun_elevation_deg, view_elevation_deg, rel_azimuth_deg)

    # Along the line of sight: pieces between shell boundaries, and optical depth to each middle.
    mid_height = 0.5 * (LEVELS_KM[:-1] + LEVELS_KM[1:])
    sin_view = math.sin(view)
    chord = np.diff(_distance_to_height(LEVELS_KM, sin_view))
    distance_to_mid = _distance_to_height(mid_height, sin_view)
    piece_depth = extinction * chord[:, None]
    depth_to_mid = np.cumsum(piece_depth, axis=0) - 0.5 * piece_depth

    # From each middle toward the sun: shadowed or not, and the optical depth if not.
    mid_radius = EARTH_RADIUS_KM + mid_height
    sun_dot = EARTH_RADIUS_KM * math.sin(sun) + distance_to_mid * cos_theta
    sun_depth = shell_path_lengths(mid_radius, sun_dot, _SHELL_RADII_KM) @ extinction
    lit = sunlit(mid_radius, sun_dot)

    scattered = optics.rayleigh_scattering * rayleigh_phase(cos_theta) + (
        optics.aerosol_scattering * henyey_greenstein_phase(cos_theta, atm.asymmetry)
    )
    transmittance = np.exp(-(sun_depth + depth_to_mid)) * lit[:, None]
    radiance = (scattered * transmittance * chord[:, None]).sum(axis=0)
    return solar_irradiance(WAVELENGTHS_NM) * radiance
