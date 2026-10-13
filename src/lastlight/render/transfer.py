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

Two ways of integrating along the line of sight (`QUADRATURES`). The registered one samples each
piece of the sightline once, in its middle. That is exact for a piece the light barely dims, and
reads low for one it dims by a large factor: a view 1.5 degrees up is 1.9 km long inside each 50 m
shell, so under heavy aerosol (`aod_550` 0.3 and more) the first pieces carry an optical depth of
one or more and the midpoint of a falling exponential is below its mean (issue #40). The adaptive
one cuts every piece into parts of at most `ADAPTIVE_PIECE_DEPTH` of optical depth, but only when
some piece is thicker than `ADAPTIVE_TRIGGER_DEPTH`; otherwise it is the registered integration,
to the last bit. The registered one is renderer v1's inherited single scattering, which is
pre-registered, so it is not changed and stays the default everywhere.
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

QUADRATURES: tuple[str, ...] = ("registered", "adaptive")
"""How `sky_spectrum` integrates along the line of sight; the first is the default everywhere."""

ADAPTIVE_TRIGGER_DEPTH = 0.7
"""Adaptive refines only if some piece of the sightline has more optical depth than this, at any
wavelength. Below it the registered result is within about 1 percent of the refined one (measured
over atmospheres just under the trigger, `docs/log/2026-10-12-v0-heavy-aerosol.md`), and adaptive
returns the registered result exactly, so paintings at ordinary aerosol loads do not move."""

ADAPTIVE_PIECE_DEPTH = 0.1
"""Once refining, no piece is left with more optical depth than this (the midpoint rule is then
within 0.05 percent on that piece) unless it would need more than `ADAPTIVE_MAX_PARTS` parts."""

ADAPTIVE_MAX_PARTS = 128
"""Most parts one shell's piece of the sightline is cut into. It bounds the work for a level view
(25 km long in the lowest shell) or an absurd aerosol load; `aod_550` 1 at a 1.5 degree view
needs at most 44."""


def check_quadrature(quadrature: str) -> None:
    """Raise `ValueError` unless `quadrature` is one of `QUADRATURES`."""
    if quadrature not in QUADRATURES:
        raise ValueError(f"quadrature must be one of {QUADRATURES}, got {quadrature!r}")


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


def _sightline_cuts(
    extinction: np.ndarray, sin_view: float, quadrature: str
) -> tuple[np.ndarray, np.ndarray]:
    """Heights where the line of sight is cut, and the shell each resulting piece lies in.

    The registered quadrature cuts at the shell boundaries, one piece per shell. The adaptive one
    returns exactly that unless some piece has more than `ADAPTIVE_TRIGGER_DEPTH` of optical
    depth at some wavelength; then every shell's piece is cut into equal parts (equal in height,
    so equal in optical depth, the shell being uniform) of at most `ADAPTIVE_PIECE_DEPTH`.
    The parts keep their shell's densities: only the sampling of the integral is finer, not the
    model. Shell boundaries stay cut points, so the arithmetic of unrefined pieces is unchanged.
    """
    shells = np.arange(len(LEVELS_KM) - 1)
    if quadrature == "registered":
        return LEVELS_KM, shells
    chord = np.diff(_distance_to_height(LEVELS_KM, sin_view))
    depth = (extinction * chord[:, None]).max(axis=1)
    if depth.max() <= ADAPTIVE_TRIGGER_DEPTH:
        return LEVELS_KM, shells
    parts = np.clip(np.ceil(depth / ADAPTIVE_PIECE_DEPTH), 1, ADAPTIVE_MAX_PARTS).astype(int)
    shell = np.repeat(shells, parts)
    position = np.arange(shell.size) - (np.cumsum(parts) - parts)[shell]
    low = LEVELS_KM[shell]
    heights = low + (LEVELS_KM[shell + 1] - low) * position / parts[shell]
    return np.append(heights, LEVELS_KM[-1]), shell


# --- the integral --------------------------------------------------------------------------------


def sky_spectrum(
    atm: Atmosphere,
    sun_elevation_deg: float,
    view_elevation_deg: float,
    rel_azimuth_deg: float = 0.0,
    *,
    quadrature: str = "registered",
) -> np.ndarray:
    """Spectral radiance of the sky in one direction, at `WAVELENGTHS_NM`.

    `sun_elevation_deg` is the sun's geometric elevation at the observer (negative when the sun
    is below the horizon), `view_elevation_deg` the elevation of the line of sight (0 to 90) and
    `rel_azimuth_deg` its azimuth relative to the sun's: 0 looks toward the sun.

    The line of sight is cut into pieces. For the middle of each piece the sunlight that reaches
    it (zero in Earth's shadow), the share scattered toward the observer, and the transmittance
    back to the observer are multiplied and summed.

    `quadrature` chooses the pieces (`QUADRATURES`, else `ValueError`). `"registered"` (the
    default, and what renderer v1 and G0 are registered with) cuts at the shell boundaries.
    `"adaptive"` cuts a sightline whose pieces are optically thick into thinner ones, which
    removes the under-reading of a low view through heavy aerosol, and is otherwise bit-for-bit
    the registered result (see the module docstring).

    This is renderer v0 only. Renderer v1 has no spectra: its table stores the multiply
    scattered light as CIE XYZ, so v1 is read through `lastlight.render.curve.sky_xyz` with
    `engine="v1"`, which adds that XYZ to the XYZ of this spectrum.
    """
    check_quadrature(quadrature)
    if not 0.0 <= view_elevation_deg <= 90.0:
        raise ValueError(f"view_elevation_deg must be in [0, 90], got {view_elevation_deg!r}")
    optics = layer_optics(atm)
    extinction = optics.extinction

    sun = math.radians(sun_elevation_deg)
    view = math.radians(view_elevation_deg)
    cos_theta = scattering_cosine(sun_elevation_deg, view_elevation_deg, rel_azimuth_deg)

    # Along the line of sight: the pieces, and the optical depth to each middle.
    sin_view = math.sin(view)
    heights, shell = _sightline_cuts(extinction, sin_view, quadrature)
    mid_height = 0.5 * (heights[:-1] + heights[1:])
    chord = np.diff(_distance_to_height(heights, sin_view))
    distance_to_mid = _distance_to_height(mid_height, sin_view)
    piece_depth = extinction[shell] * chord[:, None]
    depth_to_mid = np.cumsum(piece_depth, axis=0) - 0.5 * piece_depth

    # From each middle toward the sun: shadowed or not, and the optical depth if not.
    mid_radius = EARTH_RADIUS_KM + mid_height
    sun_dot = EARTH_RADIUS_KM * math.sin(sun) + distance_to_mid * cos_theta
    sun_depth = shell_path_lengths(mid_radius, sun_dot, _SHELL_RADII_KM) @ extinction
    lit = sunlit(mid_radius, sun_dot)

    scattered = optics.rayleigh_scattering[shell] * rayleigh_phase(cos_theta) + (
        optics.aerosol_scattering[shell] * henyey_greenstein_phase(cos_theta, atm.asymmetry)
    )
    transmittance = np.exp(-(sun_depth + depth_to_mid)) * lit[:, None]
    radiance = (scattered * transmittance * chord[:, None]).sum(axis=0)
    return solar_irradiance(WAVELENGTHS_NM) * radiance
