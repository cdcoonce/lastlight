"""Multiple scattering for renderer v1: a numpy port of Bruneton's 2017 precomputation.

Ported from Eric Bruneton's "Precomputed Atmospheric Scattering" reference implementation
(https://github.com/ebruneton/precomputed_atmospheric_scattering, `atmosphere/functions.glsl` and
`atmosphere/model.cc`), which is distributed under the BSD 3-Clause licence below. The port keeps
its algorithm: a transmittance table, single scattering, then for each order n >= 2 the
scattering density J_n (the light scattered at a point, gathered over all incident directions from
order n-1, plus order n-2 light reflected by the ground), the indirect ground irradiance of order
n-1, and the radiance of order n by integrating J_n along rays. The physics is lastlight's v0
atmosphere, evaluated point by point from v0's own constants and functions
(`lastlight.render.atmosphere`): the Rayleigh cross-section and column, the aerosol's Angstrom
law, exponential profile, Henyey-Greenstein phase function and single-scattering albedo, the
Gaussian ozone layer and its absorption table, the solar spectrum, the Earth's radius and the
100 km top. The sun is a point, as in v0, so Earth's shadow is sharp. The ground is Lambertian
with albedo `GROUND_ALBEDO`.

What differs from the reference, and why (none of it changes the physics):

* Wavelengths: all 41 of v0's grid at once, not three per pass.
* Bruneton's exponential and linear density layers cannot hold a Gaussian ozone layer, so the
  densities are v0's functions evaluated directly, and the transmittance table stores the optical
  length of each of the three profiles (Rayleigh, aerosol, ozone) instead of one transmittance per
  wavelength. The transmittance of a path is exp(-sum of profile optical depth times length).
* The 4D tables are indexed by altitude, view zenith angle, local sun elevation and relative
  azimuth (`Resolution`), on fixed nodes. The view nodes are spaced in angle from the local
  horizon, densest there, on each side of it (Bruneton's split between rays that reach the ground
  and rays that do not). The relative azimuth replaces Bruneton's nu, so every node is a real
  direction and the scattering density can be gathered with a fixed quadrature
  (`_DirectionSet`) whose phase-function weights are computed once per altitude. The density is
  stored divided by the scattering coefficients (one array per species), so the steep vertical
  profile of the aerosol is applied exactly at every sample, and it is interpolated in log
  space, where twilight's exponential darkening with sun elevation is linear.
* Rays are sampled more densely where they lose their light and where the air is dense (a
  blend of the transmittance, the column density and the distance along the ray), not at even
  steps. Single scattering is integrated only over the lit parts of each ray, whose ends are
  the exact intersections with Earth's shadow cylinder. Along a ray, the higher orders are
  integrated with the transmittance between samples taken as exponential, not linear, so an
  optically thick hazy sightline is not overestimated.
* The ground observer's multiple scattering is gathered along the exact sightline from the
  stored scattering density of every order, not read back from an interpolated radiance table.
* Bruneton sums four orders by default. Here the number of orders is chosen by a convergence
  check (`Resolution.orders`): the next order adds under 1 percent at sun 0 for WHTM1's view.

Only the ground observer's multiple scattering leaves this module: `lastlight.render.v1` adds its
CIE XYZ, read from a table that `tools/make_bruneton_table.py` builds with `precompute`, to v0's
single scattering.

Copyright and licence of the reference implementation (kept as its licence requires):

    Copyright (c) 2017 Eric Bruneton
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holders nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.

    Precomputed Atmospheric Scattering
    Copyright (c) 2008 INRIA
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holders nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache, lru_cache

import numpy as np

from lastlight.render.atmosphere import (
    ATMOSPHERE_TOP_KM,
    EARTH_RADIUS_KM,
    OZONE_LAYER_CENTRE_KM,
    OZONE_LAYER_SIGMA_KM,
    RAYLEIGH_SCALE_HEIGHT_KM,
    REFERENCE_WAVELENGTH_NM,
    WAVELENGTHS_NM,
    Atmosphere,
    ozone_absorption_per_atm_cm,
    rayleigh_optical_depth,
    solar_irradiance,
)
from lastlight.render.colorimetry import spectrum_to_xyz
from lastlight.render.transfer import henyey_greenstein_phase, rayleigh_phase, scattering_cosine

GROUND_ALBEDO = 0.2
"""Albedo of the Lambertian ground under the whole atmosphere (recorded in the table)."""

_R = EARTH_RADIUS_KM
_R_TOP = EARTH_RADIUS_KM + ATMOSPHERE_TOP_KM
_H = math.sqrt(_R_TOP**2 - _R**2)
"""Distance to the top of the atmosphere along a horizontal ray from the ground (Bruneton's H)."""

_VIEW_POWER = 2.0
"""View nodes: the angle from the local horizon grows as the square of the node index."""
_AZIMUTH_POWER = 1.5
"""Azimuth nodes: the relative azimuth grows as the 1.5 power of the node index (dense at 0)."""
_SUN_SCALE_DEG = 10.0
_SUN_RANGE_DEG = (-60.0, 90.0)
"""Local sun elevation nodes: uniform in asinh(elevation / 10 degrees) from -60 to 90 degrees.

At the table's resolution about 1 degree apart around the horizon, 2.7 at 25 degrees and wider
beyond. Below -60 degrees a point is so deep in Earth's shadow that its light is negligible; a
lookup there takes the -60 value.
"""
_PLACEMENT_SHARES = (0.5, 0.3, 0.2)
"""Ray samples are spread evenly in a blend of three measures along the ray: the light the ray
can still bring back (1 - transmittance at 550 nm, which crowds samples where a hazy or grazing
sightline goes opaque), the air's column density, and plain distance (which keeps samples in
thin, high air, where a deep-twilight sightline is lit)."""
_FINE_SAMPLES = 2049
"""Points of the fine grid along each ray, for its optical lengths and sample placement."""
_TRANSMITTANCE_STEPS = 1000
"""Trapezoid steps of each optical length in the transmittance table (Bruneton uses 500)."""
_LOG_FLOOR = 1e-300
_PLACEMENT_WAVELENGTH = int(np.flatnonzero(WAVELENGTHS_NM == REFERENCE_WAVELENGTH_NM)[0])


@dataclass(frozen=True)
class Resolution:
    """Every resolution of the precomputation, and the highest scattering order N.

    The 4D tables have `altitudes` x (`views_sky` + `views_ground`) x `sun_elevations` x
    `azimuths` nodes. Single scattering takes `single_samples` samples on each lit stretch of a
    ray, and the higher orders `march_samples` samples along it.
    The scattering density and the ground irradiance are gathered over a quadrature of
    `sphere_sky` + `sphere_ground` zenith angles by `sphere_azimuths` azimuths. The
    transmittance table has `transmittance_views` x `transmittance_altitudes` nodes. Orders 2 to
    `orders` are summed.
    """

    altitudes: int
    views_sky: int
    views_ground: int
    sun_elevations: int
    azimuths: int
    single_samples: int
    march_samples: int
    sphere_sky: int
    sphere_ground: int
    sphere_azimuths: int
    transmittance_views: int
    transmittance_altitudes: int
    orders: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2, got {value!r}")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


RESOLUTION = Resolution(
    altitudes=16,
    views_sky=24,
    views_ground=12,
    sun_elevations=55,
    azimuths=16,
    single_samples=32,
    march_samples=24,
    sphere_sky=24,
    sphere_ground=12,
    sphere_azimuths=24,
    transmittance_views=256,
    transmittance_altitudes=256,
    orders=13,
)
"""The resolution the committed table was built with (see `docs/renderer/`)."""

TINY_RESOLUTION = Resolution(
    altitudes=4,
    views_sky=5,
    views_ground=3,
    sun_elevations=8,
    azimuths=4,
    single_samples=6,
    march_samples=6,
    sphere_sky=6,
    sphere_ground=3,
    sphere_azimuths=4,
    transmittance_views=32,
    transmittance_altitudes=8,
    orders=3,
)
"""A coarse resolution for tests: the same code, in about a second."""


# --- the atmosphere at a point: v0's density functions -----------------------------------------


@cache
def _ozone_normaliser() -> float:
    """Share of an unbounded Gaussian ozone layer between the ground and the top (as v0's)."""
    scale = OZONE_LAYER_SIGMA_KM * math.sqrt(2.0)
    return 0.5 * (
        math.erf((ATMOSPHERE_TOP_KM - OZONE_LAYER_CENTRE_KM) / scale)
        - math.erf((0.0 - OZONE_LAYER_CENTRE_KM) / scale)
    )


def _profiles(height_km: np.ndarray, aerosol_scale_height_km: float) -> np.ndarray:
    """Density of air, aerosol and ozone per km of height, each integrating to 1 vertically.

    Shape `(3,) + height.shape`. Multiplied by a species' vertical optical depth this is its
    coefficient per km: the point form of `lastlight.render.atmosphere.layer_optics`.
    """
    h = np.clip(height_km, 0.0, ATMOSPHERE_TOP_KM)
    rayleigh = np.exp(-h / RAYLEIGH_SCALE_HEIGHT_KM) / RAYLEIGH_SCALE_HEIGHT_KM
    aerosol = np.exp(-h / aerosol_scale_height_km) / aerosol_scale_height_km
    sigma = OZONE_LAYER_SIGMA_KM
    ozone = np.exp(-0.5 * ((h - OZONE_LAYER_CENTRE_KM) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi) * _ozone_normaliser()
    )
    return np.stack([rayleigh, aerosol, ozone])


@dataclass(frozen=True, eq=False)
class _Optics:
    """Vertical optical depths per wavelength: extinction (3, w) and scattering (2, w)."""

    extinction: np.ndarray
    scattering: np.ndarray
    aerosol_scale_height_km: float
    asymmetry: float
    sun: np.ndarray


def _optics(atm: Atmosphere) -> _Optics:
    wavelengths = WAVELENGTHS_NM
    rayleigh = rayleigh_optical_depth(wavelengths, atm.pressure_hpa)
    aerosol = atm.aod_550 * (wavelengths / REFERENCE_WAVELENGTH_NM) ** -atm.angstrom
    ozone = ozone_absorption_per_atm_cm(wavelengths) * atm.ozone_du / 1000.0
    return _Optics(
        extinction=np.stack([rayleigh, aerosol, ozone]),
        scattering=np.stack([rayleigh, atm.single_scatter_albedo * aerosol]),
        aerosol_scale_height_km=atm.aerosol_scale_height_km,
        asymmetry=atm.asymmetry,
        sun=solar_irradiance(wavelengths),
    )


# --- geometry (Bruneton's DistanceToTopAtmosphereBoundary and friends) -------------------------


def _distance_to_top(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    discriminant = r * r * (mu * mu - 1.0) + _R_TOP * _R_TOP
    return np.maximum(-r * mu + np.sqrt(np.maximum(discriminant, 0.0)), 0.0)


def _distance_to_ground(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    discriminant = r * r * (mu * mu - 1.0) + _R * _R
    return np.maximum(-r * mu - np.sqrt(np.maximum(discriminant, 0.0)), 0.0)


def _horizon_zenith(r: np.ndarray) -> np.ndarray:
    """Zenith angle of the local horizon (the ray grazing the ground) seen from radius r."""
    return 0.5 * math.pi + np.arccos(np.clip(_R / np.maximum(r, _R), -1.0, 1.0))


# --- transmittance: optical lengths to the top of the atmosphere --------------------------------


@lru_cache(maxsize=8)
def _optical_length_table(scale_height_km: float, views: int, altitudes: int) -> np.ndarray:
    """Log of the optical length (km of unit-depth column) of each profile from (r, mu) to the
    top of the atmosphere: Bruneton's transmittance texture, one layer per profile.

    Shape (3, altitudes, views), on Bruneton's transmittance mapping: altitude uniform in the
    distance to the horizon rho, view uniform in the distance to the top between its least
    (straight up) and greatest (along the horizon) values. Only rays that miss the ground. The
    log is stored because lookups interpolate it (`_lookup_optical_length`).
    """
    table = np.empty((3, altitudes, views))
    x_mu = np.linspace(0.0, 1.0, views)
    steps = np.linspace(0.0, 1.0, _TRANSMITTANCE_STEPS + 1)
    weights = np.full(steps.shape, 1.0 / _TRANSMITTANCE_STEPS)
    weights[[0, -1]] *= 0.5
    for i, x_r in enumerate(np.linspace(0.0, 1.0, altitudes)):
        rho = _H * x_r
        r = math.sqrt(rho * rho + _R * _R)
        d_min, d_max = _R_TOP - r, rho + _H
        d = d_min + x_mu * (d_max - d_min)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(d == 0.0, 1.0, (_H * _H - rho * rho - d * d) / (2.0 * r * d))
        mu = np.clip(mu, -1.0, 1.0)
        s = d[:, None] * steps[None, :]
        radius = np.sqrt(s * s + 2.0 * r * mu[:, None] * s + r * r)
        density = _profiles(radius - _R, scale_height_km)
        table[:, i, :] = (density * weights).sum(axis=-1) * d
    logs = np.log(np.maximum(table, _LOG_FLOOR))
    logs.flags.writeable = False
    return logs


def _lookup_optical_length(table: np.ndarray, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Optical lengths to the top, shape (3,) + r.shape: bilinear in the table's logs.

    In log space an exponential profile is linear in height, so a near-ground aerosol layer a
    few hundred metres thick is interpolated without bias between altitude nodes.
    """
    logs = table
    _, altitudes, views = logs.shape
    rho = np.sqrt(np.maximum(r * r - _R * _R, 0.0))
    d = _distance_to_top(r, mu)
    d_min, d_max = _R_TOP - r, rho + _H
    x_mu = np.clip((d - d_min) / np.maximum(d_max - d_min, 1e-12), 0.0, 1.0) * (views - 1)
    x_r = np.clip(rho / _H, 0.0, 1.0) * (altitudes - 1)
    i0 = np.minimum(x_r.astype(int), altitudes - 2)
    j0 = np.minimum(x_mu.astype(int), views - 2)
    fi, fj = x_r - i0, x_mu - j0
    value = (
        logs[:, i0, j0] * ((1 - fi) * (1 - fj))
        + logs[:, i0, j0 + 1] * ((1 - fi) * fj)
        + logs[:, i0 + 1, j0] * (fi * (1 - fj))
        + logs[:, i0 + 1, j0 + 1] * (fi * fj)
    )
    return np.exp(value)


# --- rays: optical lengths along them and where to sample them --------------------------------


@dataclass(frozen=True, eq=False)
class _Ray:
    """A ray from radius `r` with zenith cosine `mu`, to the ground or the top: length, fine
    grid, cumulative optical length of each profile (3, fine) and the placement measure."""

    r: float
    mu: float
    length: float
    reaches_ground: bool
    cumulative: np.ndarray
    placement: np.ndarray

    def at(self, distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Radius and zenith cosine of the ray's direction at `distance` along it."""
        radius = np.sqrt(np.maximum(distance**2 + 2.0 * self.r * self.mu * distance + self.r**2, 0))
        radius = np.clip(radius, _R, _R_TOP)
        mu = np.clip((self.r * self.mu + distance) / radius, -1.0, 1.0)
        return radius, mu

    def optical_length(self, distance: np.ndarray) -> np.ndarray:
        """Optical length of each profile from the start to `distance`, shape (3,) + shape."""
        if self.length <= 0.0:
            return np.zeros((3, *np.shape(distance)))
        position = np.clip(np.asarray(distance) / self.length, 0.0, 1.0) * (_FINE_SAMPLES - 1)
        lower = np.minimum(position.astype(int), _FINE_SAMPLES - 2)
        fraction = position - lower
        return self.cumulative[:, lower] * (1 - fraction) + self.cumulative[:, lower + 1] * fraction

    def samples(self, start: np.ndarray, stop: np.ndarray, count: int) -> np.ndarray:
        """`count` distances from `start` to `stop` (each broadcast), evenly in the placement."""
        if self.length <= 0.0:
            return np.zeros((*np.shape(start), count))
        grid = np.linspace(0.0, 1.0, _FINE_SAMPLES) * self.length
        low = np.interp(start, grid, self.placement)
        high = np.interp(stop, grid, self.placement)
        targets = low[..., None] + (high - low)[..., None] * np.linspace(0.0, 1.0, count)
        distance = np.interp(targets, self.placement, grid)
        distance[..., 0] = start
        distance[..., -1] = stop
        return distance


def _make_ray(r: float, mu: float, reaches_ground: bool, optics: _Optics) -> _Ray:
    length = float(
        _distance_to_ground(np.array(r), np.array(mu))
        if reaches_ground
        else _distance_to_top(np.array(r), np.array(mu))
    )
    grid = np.linspace(0.0, 1.0, _FINE_SAMPLES) * length
    radius = np.sqrt(np.maximum(grid**2 + 2.0 * r * mu * grid + r * r, 0.0))
    density = _profiles(np.clip(radius, _R, _R_TOP) - _R, optics.aerosol_scale_height_km)
    step = length / (_FINE_SAMPLES - 1)
    pieces = 0.5 * (density[:, 1:] + density[:, :-1]) * step
    cumulative = np.concatenate([np.zeros((3, 1)), np.cumsum(pieces, axis=1)], axis=1)
    if length > 0.0:
        depth = optics.extinction[:, _PLACEMENT_WAVELENGTH] @ cumulative
        column = cumulative[0] + cumulative[1]
        measures = (-np.expm1(-depth), column, grid)
        placement = np.zeros(_FINE_SAMPLES)
        shares = 0.0
        for share, measure in zip(_PLACEMENT_SHARES, measures, strict=True):
            if measure[-1] > 0.0:
                placement += share * measure / measure[-1]
                shares += share
        placement /= shares
    else:
        placement = np.linspace(0.0, 1.0, _FINE_SAMPLES)
    return _Ray(r, mu, length, reaches_ground, cumulative, placement)


def _trapezoid_weights(distance: np.ndarray) -> np.ndarray:
    """Trapezoid weights for samples at `distance` (last axis), zero for a zero-length stretch."""
    gaps = np.diff(distance, axis=-1)
    weights = np.zeros(distance.shape)
    weights[..., :-1] += 0.5 * gaps
    weights[..., 1:] += 0.5 * gaps
    return weights


def _transmittance_weights(distance: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Quadrature weights for the integral of f(s) exp(-depth(s)) ds along a ray: (samples, w).

    Between two samples the optical depth is taken as linear in distance and f as linear, and
    the product is integrated exactly, so an optically thick stretch (a hazy horizon, where
    the transmittance falls by e between samples) is not overestimated as the trapezoid rule
    would. With no extinction it is the trapezoid rule.
    """
    gaps = np.diff(distance)[:, None]
    delta = np.maximum(np.diff(depth, axis=0), 0.0)
    small = delta < 1e-4
    safe = np.where(small, 1.0, delta)
    decay = np.exp(-safe)
    whole = np.where(small, 1.0 - delta / 2.0 + delta**2 / 6.0, (1.0 - decay) / safe)
    later = np.where(
        small, 0.5 - delta / 3.0 + delta**2 / 8.0, (1.0 - decay * (1.0 + safe)) / safe**2
    )
    scale = gaps * np.exp(-depth[:-1])
    weights = np.zeros(depth.shape)
    weights[:-1] += scale * (whole - later)
    weights[1:] += scale * later
    return weights


# --- single scattering ------------------------------------------------------------------------


def _shadow_interval(ray: _Ray, mu_s: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The stretch [a, b] of the ray inside Earth's shadow, for each sun (mu_s, nu).

    A point y is in the shadow when its sunward ray hits the Earth: it lies behind the Earth
    (y . s < 0) and inside the shadow cylinder (|y|^2 - (y . s)^2 < R^2). Along the ray the first
    is a half-line and the second a quadratic in the distance, so the shadow is one interval
    (empty when a > b). The result is clipped to the ray.
    """
    r, mu, length = ray.r, ray.mu, ray.length
    big = 4.0 * (length + _R_TOP)
    # behind the Earth: r mu_s + s nu < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -r * mu_s / nu
    behind_low = np.where(nu > 0.0, -big, np.where(nu < 0.0, root, -big))
    behind_high = np.where(nu > 0.0, root, np.where(nu < 0.0, big, big))
    never_behind = (nu == 0.0) & (mu_s >= 0.0)
    behind_low = np.where(never_behind, big, behind_low)
    behind_high = np.where(never_behind, -big, behind_high)
    # inside the cylinder: a s^2 + b s + c < 0
    qa = 1.0 - nu * nu
    qb = 2.0 * (r * mu - r * mu_s * nu)
    qc = r * r * (1.0 - mu_s * mu_s) - _R * _R
    discriminant = qb * qb - 4.0 * qa * qc
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
        low = (-qb - sqrt_disc) / (2.0 * qa)
        high = (-qb + sqrt_disc) / (2.0 * qa)
    flat = qa < 1e-12  # sunward ray parallel to the view: inside iff c < 0 (to first order)
    inside = (discriminant > 0.0) & ~flat
    cyl_low = np.where(inside, low, np.where(flat & (qc < 0.0), -big, big))
    cyl_high = np.where(inside, high, np.where(flat & (qc < 0.0), big, -big))
    start = np.clip(np.maximum(behind_low, cyl_low), 0.0, length)
    stop = np.clip(np.minimum(behind_high, cyl_high), 0.0, length)
    empty = stop <= start
    return np.where(empty, length, start), np.where(empty, length, stop)


def _single_scattering(
    ray: _Ray,
    mu_s: np.ndarray,
    nu: np.ndarray,
    optics: _Optics,
    transmittance: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bruneton's ComputeSingleScattering for one ray and many suns: Rayleigh and aerosol.

    Returns two arrays of shape (suns, wavelengths): the light scattered once toward the ray's
    start, without the solar irradiance and the phase functions (added by the caller), as
    Bruneton stores them. Only the lit stretches of the ray are integrated.
    """
    mu_s = np.asarray(mu_s, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if ray.length <= 0.0:
        zero = np.zeros((mu_s.size, WAVELENGTHS_NM.size))
        return zero, zero
    shadow_start, shadow_stop = _shadow_interval(ray, mu_s, nu)
    stretches = [
        ray.samples(np.zeros_like(mu_s), shadow_start, samples),
        ray.samples(shadow_stop, np.full_like(mu_s, ray.length), samples),
    ]
    distance = np.concatenate(stretches, axis=-1)
    weights = np.concatenate([_trapezoid_weights(s) for s in stretches], axis=-1)
    radius, _ = ray.at(distance)
    mu_s_here = np.clip((ray.r * mu_s[:, None] + distance * nu[:, None]) / radius, -1.0, 1.0)
    to_sun = _lookup_optical_length(transmittance, radius, mu_s_here)
    path = ray.optical_length(distance) + to_sun  # (3, suns, samples)
    depth = np.einsum("kts,kw->tsw", path, optics.extinction).astype(np.float32)
    density = _profiles(radius - _R, optics.aerosol_scale_height_km)
    weighted = np.exp(np.negative(depth, out=depth), out=depth)
    weighted *= weights.astype(np.float32)[..., None]
    rayleigh = np.einsum("tsw,ts->tw", weighted, density[0].astype(np.float32))
    aerosol = np.einsum("tsw,ts->tw", weighted, density[1].astype(np.float32))
    rayleigh = rayleigh.astype(float) * optics.scattering[0]
    aerosol = aerosol.astype(float) * optics.scattering[1]
    return rayleigh, aerosol


def _transmittance_table(atm_or_optics: _Optics, resolution: Resolution) -> np.ndarray:
    return _optical_length_table(
        atm_or_optics.aerosol_scale_height_km,
        resolution.transmittance_views,
        resolution.transmittance_altitudes,
    )


def single_scattering_spectrum(
    atm: Atmosphere,
    sun_elevation_deg: float,
    view_elevation_deg: float,
    rel_azimuth_deg: float = 0.0,
    *,
    resolution: Resolution = RESOLUTION,
) -> np.ndarray:
    """The port's own single scattering at the ground observer, at `WAVELENGTHS_NM`.

    The same quantity as v0's `sky_spectrum` (W m^-2 sr^-1 nm^-1), computed the port's way:
    the transmittance table, the ray samples and the shadow test of the precomputation at
    `resolution`. It is the order-1 oracle of the port; v1 itself uses v0's single scattering.
    """
    if not 0.0 <= view_elevation_deg <= 90.0:
        raise ValueError(f"view_elevation_deg must be in [0, 90], got {view_elevation_deg!r}")
    optics = _optics(atm)
    transmittance = _transmittance_table(optics, resolution)
    mu = math.sin(math.radians(view_elevation_deg))
    ray = _make_ray(_R, mu, False, optics)
    nu = scattering_cosine(sun_elevation_deg, view_elevation_deg, rel_azimuth_deg)
    mu_s = math.sin(math.radians(sun_elevation_deg))
    rayleigh, aerosol = _single_scattering(
        ray, np.array([mu_s]), np.array([nu]), optics, transmittance, resolution.single_samples
    )
    phase_r = rayleigh_phase(np.array(nu))
    phase_a = henyey_greenstein_phase(np.array(nu), optics.asymmetry)
    return optics.sun * (rayleigh[0] * phase_r + aerosol[0] * phase_a)


# --- the 4D grid ------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Grid:
    """Nodes of the 4D tables: radius, view zenith cosine per radius, sun and azimuth."""

    radii: np.ndarray  # (altitudes,)
    views_sky: int
    views_ground: int
    mu: np.ndarray  # (altitudes, views): sky nodes from the horizon up, then ground nodes down
    sun_mu: np.ndarray  # (sun_elevations,)
    sun_elevation: np.ndarray  # degrees
    sun_t: np.ndarray  # asinh coordinate of each sun node
    azimuth: np.ndarray  # radians, (azimuths,)

    @property
    def views(self) -> int:
        return self.views_sky + self.views_ground


@cache
def _grid(resolution: Resolution) -> _Grid:
    x_r = np.linspace(0.0, 1.0, resolution.altitudes)
    radii = np.sqrt((_H * x_r) ** 2 + _R * _R)
    radii[0], radii[-1] = _R, _R_TOP
    horizon = _horizon_zenith(radii)
    u_sky = np.linspace(0.0, 1.0, resolution.views_sky) ** _VIEW_POWER
    u_ground = np.linspace(0.0, 1.0, resolution.views_ground) ** _VIEW_POWER
    zenith_sky = horizon[:, None] * (1.0 - u_sky[None, :])
    zenith_ground = horizon[:, None] + (math.pi - horizon[:, None]) * u_ground[None, :]
    mu = np.cos(np.concatenate([zenith_sky, zenith_ground], axis=1))
    low, high = (math.asinh(e / _SUN_SCALE_DEG) for e in _SUN_RANGE_DEG)
    sun_t = np.linspace(low, high, resolution.sun_elevations)
    sun_elevation = _SUN_SCALE_DEG * np.sinh(sun_t)
    sun_elevation[-1] = 90.0
    azimuth = math.pi * np.linspace(0.0, 1.0, resolution.azimuths) ** _AZIMUTH_POWER
    return _Grid(
        radii=radii,
        views_sky=resolution.views_sky,
        views_ground=resolution.views_ground,
        mu=mu,
        sun_mu=np.sin(np.radians(sun_elevation)),
        sun_elevation=sun_elevation,
        sun_t=sun_t,
        azimuth=azimuth,
    )


def _radius_index(grid: _Grid, radius: np.ndarray) -> np.ndarray:
    rho = np.sqrt(np.maximum(radius * radius - _R * _R, 0.0))
    return np.clip(rho / _H, 0.0, 1.0) * (grid.radii.size - 1)


def _view_index(
    grid: _Grid, radius: np.ndarray, mu: np.ndarray, reaches_ground: bool
) -> np.ndarray:
    """Continuous view-node index (within the sky or the ground nodes) of a direction."""
    horizon = _horizon_zenith(radius)
    zenith = np.arccos(np.clip(mu, -1.0, 1.0))
    if reaches_ground:
        span = np.maximum(math.pi - horizon, 1e-12)
        u = np.clip((zenith - horizon) / span, 0.0, 1.0)
        return u ** (1.0 / _VIEW_POWER) * (grid.views_ground - 1)
    u = np.clip((horizon - zenith) / horizon, 0.0, 1.0)
    return u ** (1.0 / _VIEW_POWER) * (grid.views_sky - 1)


def _sun_index(grid: _Grid, mu_s: np.ndarray) -> np.ndarray:
    elevation = np.degrees(np.arcsin(np.clip(mu_s, -1.0, 1.0)))
    t = np.arcsinh(elevation / _SUN_SCALE_DEG)
    position = (t - grid.sun_t[0]) / (grid.sun_t[-1] - grid.sun_t[0])
    return np.clip(position, 0.0, 1.0) * (grid.sun_t.size - 1)


def _azimuth_index(grid: _Grid, azimuth: np.ndarray) -> np.ndarray:
    u = np.clip(azimuth / math.pi, 0.0, 1.0)
    return u ** (1.0 / _AZIMUTH_POWER) * (grid.azimuth.size - 1)


def _split(position: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.minimum(np.floor(position).astype(int), size - 2)
    return lower, position - lower


def _relative_azimuth(mu: np.ndarray, mu_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Azimuth of a view direction from the sun's, from the three cosines (radians, 0 to pi)."""
    denominator = np.sqrt(np.maximum((1.0 - mu * mu) * (1.0 - mu_s * mu_s), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(denominator > 1e-12, (nu - mu * mu_s) / denominator, 1.0)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _nu(mu: np.ndarray, mu_s: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    return mu * mu_s + np.sqrt(np.maximum((1 - mu * mu) * (1 - mu_s * mu_s), 0.0)) * np.cos(azimuth)


# --- the quadrature over incident directions ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class _DirectionSet:
    """Incident directions at one radius: zenith cosines (sky, then ground), azimuths in (0, pi)
    and solid-angle weights (the mirror half of the sphere is added by the caller)."""

    mu: np.ndarray  # (zeniths,)
    reaches_ground: np.ndarray  # (zeniths,) bool
    azimuth: np.ndarray  # (azimuths,)
    weight: np.ndarray  # (zeniths, azimuths): solid angle of each direction


def _gauss(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * (nodes + 1.0), 0.5 * weights


@cache
def _directions(resolution: Resolution, radius: float) -> _DirectionSet:
    """Gauss-Legendre in the square root of the angle from the horizon (dense near it) on each
    side of the horizon, and in azimuth to the 1/1.5 power (dense toward the sun)."""
    horizon = float(_horizon_zenith(np.array(radius)))
    u, wu = _gauss(resolution.sphere_sky)
    zenith_sky = horizon * (1.0 - u**2)
    jacobian_sky = horizon * 2.0 * u * wu
    u, wu = _gauss(resolution.sphere_ground)
    zenith_ground = horizon + (math.pi - horizon) * u**2
    jacobian_ground = (math.pi - horizon) * 2.0 * u * wu
    zenith = np.concatenate([zenith_sky, zenith_ground])
    dzenith = np.concatenate([jacobian_sky, jacobian_ground])
    t, wt = _gauss(resolution.sphere_azimuths)
    azimuth = math.pi * t**_AZIMUTH_POWER
    dazimuth = math.pi * _AZIMUTH_POWER * t ** (_AZIMUTH_POWER - 1.0) * wt
    reaches_ground = np.concatenate(
        [np.zeros(resolution.sphere_sky, bool), np.ones(resolution.sphere_ground, bool)]
    )
    if radius <= _R:  # from the ground, a horizontal ray does not reach it
        reaches_ground = zenith > 0.5 * math.pi
    weight = (np.sin(zenith) * dzenith)[:, None] * dazimuth[None, :]
    return _DirectionSet(np.cos(zenith), reaches_ground, azimuth, weight)


def _phase_weights(
    view_mu: np.ndarray, view_azimuth: np.ndarray, directions: _DirectionSet, asymmetry: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weights (views, directions) that gather incident radiance into scattered radiance.

    For Rayleigh and aerosol: solid angle times the phase function at the angle between a view
    node and an incident direction, the azimuth's mirror image included (the radiance field is
    symmetric about the sun's vertical plane). Each row is scaled to sum to 1, the phase
    function's integral over the sphere, so the quadrature neither creates nor loses light.
    """
    vm = view_mu[:, None, None]
    vs = np.sqrt(np.maximum(1.0 - vm * vm, 0.0))
    va = view_azimuth[:, None, None]
    dm = directions.mu[None, :, None]
    ds = np.sqrt(np.maximum(1.0 - dm * dm, 0.0))
    da = directions.azimuth[None, None, :]
    plus = np.clip(vm * dm + vs * ds * np.cos(va - da), -1.0, 1.0)
    minus = np.clip(vm * dm + vs * ds * np.cos(va + da), -1.0, 1.0)
    weight = directions.weight[None, :, :]
    result = []
    for phase in (
        lambda c: rayleigh_phase(c),
        lambda c: henyey_greenstein_phase(c, asymmetry),
    ):
        w = (phase(plus) + phase(minus)) * weight
        w = w.reshape(view_mu.size, -1)
        result.append(w / w.sum(axis=1, keepdims=True))
    return result[0], result[1]


def _interpolation_weights(grid: _Grid, level: int, directions: _DirectionSet) -> np.ndarray:
    """Matrix (directions, view nodes) that interpolates a table at one radius bilinearly."""
    radius = grid.radii[level]
    views, azimuths = grid.views, grid.azimuth.size
    phis = directions.azimuth.size
    matrix = np.zeros((directions.mu.size, phis, views, azimuths))
    a_lower, a_frac = _split(_azimuth_index(grid, directions.azimuth), azimuths)
    for q, (mu, ground) in enumerate(zip(directions.mu, directions.reaches_ground, strict=True)):
        index = float(_view_index(grid, np.array(radius), np.array(mu), bool(ground)))
        offset = grid.views_sky if ground else 0
        count = grid.views_ground if ground else grid.views_sky
        v_lower, v_frac = _split(np.array(index), count)
        for dv, wv in ((0, 1.0 - v_frac), (1, v_frac)):
            for da, wa in ((0, 1.0 - a_frac), (1, a_frac)):
                np.add.at(
                    matrix, (q, np.arange(phis), offset + v_lower + dv, a_lower + da), wv * wa
                )
    return matrix.reshape(directions.mu.size * phis, views * azimuths)


# --- the precomputation -----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Level:
    """Everything fixed at one radius: the table's rays, the quadrature and its weights."""

    rays: tuple[_Ray, ...]  # one per view node
    directions: _DirectionSet
    direction_rays: tuple[_Ray, ...]  # one per quadrature zenith
    gather_rayleigh: np.ndarray  # (views*azimuths, directions)
    gather_aerosol: np.ndarray
    table_to_directions: np.ndarray  # (directions, views*azimuths)
    table_rayleigh: np.ndarray  # gather_rayleigh @ table_to_directions: table to table
    table_aerosol: np.ndarray


def _levels(grid: _Grid, resolution: Resolution, optics: _Optics) -> list[_Level]:
    levels = []
    view_azimuth = np.tile(grid.azimuth, grid.views)
    for i, radius in enumerate(grid.radii):
        mu_nodes = grid.mu[i]
        ground_nodes = np.arange(grid.views) >= grid.views_sky
        rays = tuple(
            _make_ray(float(radius), float(mu), bool(g), optics)
            for mu, g in zip(mu_nodes, ground_nodes, strict=True)
        )
        directions = _directions(resolution, float(radius))
        direction_rays = tuple(
            _make_ray(float(radius), float(mu), bool(g), optics)
            for mu, g in zip(directions.mu, directions.reaches_ground, strict=True)
        )
        view_mu = np.repeat(mu_nodes, grid.azimuth.size)
        rayleigh, aerosol = _phase_weights(view_mu, view_azimuth, directions, optics.asymmetry)
        to_directions = _interpolation_weights(grid, i, directions)
        levels.append(
            _Level(
                rays,
                directions,
                direction_rays,
                rayleigh,
                aerosol,
                to_directions,
                rayleigh @ to_directions,
                aerosol @ to_directions,
            )
        )
    return levels


def _log(values: np.ndarray) -> np.ndarray:
    """Log of a density table, as float32: the march reads it many times per order."""
    return np.log(np.maximum(values, _LOG_FLOOR)).astype(np.float32)


def _march(
    grid: _Grid,
    log_density: np.ndarray,
    ray: _Ray,
    mu_s: np.ndarray,
    nu: np.ndarray,
    optics: _Optics,
    samples: int,
) -> np.ndarray:
    """Bruneton's ComputeMultipleScattering: integrate the scattering density along one ray.

    `log_density` has shape (altitudes, views, azimuths, suns, 2 * wavelengths): the log of the
    density divided by the Rayleigh and aerosol scattering coefficients. The density at each
    sample is interpolated over (radius, view) first, which the samples share for every sun, then
    over (sun, azimuth); the coefficients are applied exactly at the sample's height. Returns
    (suns, wavelengths) of radiance arriving at the ray's start, for each (mu_s, nu) at the start.
    """
    count = mu_s.size
    width = WAVELENGTHS_NM.size
    if ray.length <= 0.0:
        return np.zeros((count, width))
    distance = ray.samples(np.array(0.0), np.array(ray.length), samples)
    radius, mu = ray.at(distance)
    r_lower, r_frac = _split(_radius_index(grid, radius), grid.radii.size)
    count_side = grid.views_ground if ray.reaches_ground else grid.views_sky
    offset = grid.views_sky if ray.reaches_ground else 0
    v_lower, v_frac = _split(_view_index(grid, radius, mu, ray.reaches_ground), count_side)
    v_lower = v_lower + offset
    weight_rv = (
        ((1 - r_frac) * (1 - v_frac), r_lower, v_lower),
        ((1 - r_frac) * v_frac, r_lower, v_lower + 1),
        (r_frac * (1 - v_frac), r_lower + 1, v_lower),
        (r_frac * v_frac, r_lower + 1, v_lower + 1),
    )
    corner = np.zeros((samples, *log_density.shape[2:]), np.float32)
    for w, ri, vi in weight_rv:
        part = log_density[ri, vi]
        part *= w.astype(np.float32)[:, None, None, None]
        corner += part
    azimuths, suns = grid.azimuth.size, grid.sun_mu.size
    flat = corner.reshape(samples * azimuths * suns, 2 * width)
    mu_s_here = np.clip(
        (ray.r * mu_s[:, None] + distance[None, :] * nu[:, None]) / radius[None, :], -1.0, 1.0
    )  # (count, samples)
    azimuth_here = _relative_azimuth(mu[None, :], mu_s_here, nu[:, None])
    s_lower, s_frac = _split(_sun_index(grid, mu_s_here), suns)
    a_lower, a_frac = _split(_azimuth_index(grid, azimuth_here), azimuths)
    base = np.arange(samples)[None, :] * (azimuths * suns)
    value = np.zeros((count, samples, 2 * width), np.float32)
    for da, wa in ((0, 1.0 - a_frac), (1, a_frac)):
        for ds, ws in ((0, 1.0 - s_frac), (1, s_frac)):
            part = flat[base + (a_lower + da) * suns + (s_lower + ds)]
            part *= (wa * ws).astype(np.float32)[..., None]
            value += part
    density = np.exp(value, out=value)
    path = ray.optical_length(distance)  # (3, samples)
    weighted = _transmittance_weights(distance, path.T @ optics.extinction)  # (samples, w)
    profile = _profiles(radius - _R, optics.aerosol_scale_height_km)
    rayleigh = profile[0][:, None] * weighted * optics.scattering[0]
    aerosol = profile[1][:, None] * weighted * optics.scattering[1]
    factors = np.concatenate([rayleigh, aerosol], axis=1).astype(np.float32)  # (samples, 2w)
    density *= factors[None, :, :]
    summed = density.sum(axis=1, dtype=np.float64)  # (count, 2w)
    return summed[:, :width] + summed[:, width:]


def _grid_suns(grid: _Grid, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """(mu_s, nu) of every (azimuth, sun) node pair for a view of zenith cosine `mu`."""
    azimuth = np.repeat(grid.azimuth, grid.sun_mu.size)
    mu_s = np.tile(grid.sun_mu, grid.azimuth.size)
    return mu_s, _nu(np.full_like(mu_s, mu), mu_s, azimuth)


def _ground_irradiance_direct(
    mu_s: np.ndarray, optics: _Optics, transmittance: np.ndarray
) -> np.ndarray:
    """Bruneton's ComputeDirectIrradiance at the ground for a point sun: (..., wavelengths)."""
    lengths = _lookup_optical_length(transmittance, np.full_like(mu_s, _R), np.maximum(mu_s, 0.0))
    depth = np.moveaxis(lengths, 0, -1) @ optics.extinction
    return optics.sun * np.exp(-depth) * np.maximum(mu_s, 0.0)[..., None]


def _interpolate_irradiance(grid: _Grid, irradiance: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """Ground irradiance (suns, w) on the sun nodes at any `mu_s`, linear in log."""
    s_lower, s_frac = _split(_sun_index(grid, mu_s), grid.sun_mu.size)
    logs = np.log(np.maximum(irradiance, _LOG_FLOOR))
    value = logs[s_lower] * (1 - s_frac)[..., None] + logs[s_lower + 1] * s_frac[..., None]
    result = np.exp(value)
    return np.where(result <= 10 * _LOG_FLOOR, 0.0, result)


def _gather_spectra(
    grid: _Grid,
    optics: _Optics,
    samples: int,
    log_density: np.ndarray,
    sun_elevations: Sequence[float],
    view_elevations: Sequence[float],
    azimuths: Sequence[float],
) -> np.ndarray:
    """A ground observer's radiance from a density table: (views, suns, azimuths, w)."""
    suns = np.asarray(sun_elevations, dtype=float)
    rel = np.radians(np.asarray(azimuths, dtype=float))
    mu_s = np.repeat(np.sin(np.radians(suns)), rel.size)
    azimuth = np.tile(rel, suns.size)
    result = np.empty((len(view_elevations), suns.size * rel.size, WAVELENGTHS_NM.size))
    for n, view in enumerate(view_elevations):
        if not 0.0 <= float(view) <= 90.0:
            raise ValueError(f"view elevation must be in [0, 90], got {view!r}")
        mu = math.sin(math.radians(float(view)))
        ray = _make_ray(_R, mu, False, optics)
        nu = _nu(np.full_like(mu_s, mu), mu_s, azimuth)
        result[n] = _march(grid, log_density, ray, mu_s, nu, optics, samples)
    return result.reshape(len(view_elevations), suns.size, rel.size, -1)


def _gather_xyz(
    grid: _Grid,
    optics: _Optics,
    resolution: Resolution,
    log_density: np.ndarray,
    sun_elevations: Sequence[float],
    view_elevations: Sequence[float],
    azimuths: Sequence[float],
) -> np.ndarray:
    """CIE XYZ of `_gather_spectra`: (suns, azimuths, views, 3)."""
    spectra = _gather_spectra(
        grid,
        optics,
        resolution.march_samples,
        log_density,
        sun_elevations,
        view_elevations,
        azimuths,
    )
    return np.transpose(spectrum_to_xyz(spectra), (1, 2, 0, 3))


@dataclass(frozen=True, eq=False)
class ScatteringField:
    """The result of `precompute`: the summed scattering density of orders 2 to N.

    `multiple_spectrum` and `multiple_xyz` gather it along a ground observer's sightline at
    any geometry. With `next_order=True`, `next_order_xyz` gives order N+1 alone, for the check
    that the orders have converged. `order_xyz` holds each order's XYZ at the probe, if any.
    """

    resolution: Resolution
    ground_albedo: float
    _grid: _Grid
    _optics: _Optics
    _log_density: np.ndarray
    _log_next: np.ndarray | None
    order_xyz: np.ndarray | None = None

    def multiple_spectrum(
        self, sun_elevation: float, view_elevations: Sequence[float], rel_azimuth_deg: float
    ) -> np.ndarray:
        """Radiance of orders 2 to N at the ground, (views, wavelengths), W m^-2 sr^-1 nm^-1."""
        spectra = _gather_spectra(
            self._grid,
            self._optics,
            self.resolution.march_samples,
            self._log_density,
            [sun_elevation],
            view_elevations,
            [rel_azimuth_deg],
        )
        return spectra[:, 0, 0, :]

    def multiple_xyz(
        self, sun_elevation: float, view_elevations: Sequence[float], rel_azimuth_deg: float
    ) -> np.ndarray:
        """CIE XYZ of orders 2 to N at the ground, (views, 3), on v0's white-card scale."""
        return spectrum_to_xyz(
            self.multiple_spectrum(sun_elevation, view_elevations, rel_azimuth_deg)
        )

    def xyz_block(
        self,
        sun_elevations: Sequence[float],
        view_elevations: Sequence[float],
        azimuths: Sequence[float],
    ) -> np.ndarray:
        """XYZ of orders 2 to N on a grid: shape (suns, views, azimuths, 3)."""
        xyz = _gather_xyz(
            self._grid,
            self._optics,
            self.resolution,
            self._log_density,
            sun_elevations,
            view_elevations,
            azimuths,
        )
        return np.transpose(xyz, (0, 2, 1, 3))

    def next_order_xyz(
        self, sun_elevation: float, view_elevations: Sequence[float], rel_azimuth_deg: float
    ) -> np.ndarray:
        """CIE XYZ of order N+1 alone, (views, 3): what one more order would add."""
        if self._log_next is None:
            raise ValueError("precompute(next_order=True) is needed for the next order")
        xyz = _gather_xyz(
            self._grid,
            self._optics,
            self.resolution,
            self._log_next,
            [sun_elevation],
            view_elevations,
            [rel_azimuth_deg],
        )
        return xyz[0, 0]


def precompute(
    atm: Atmosphere,
    *,
    resolution: Resolution = RESOLUTION,
    ground_albedo: float = GROUND_ALBEDO,
    next_order: bool = False,
    probe: tuple[float, Sequence[float], Sequence[float]] | None = None,
) -> ScatteringField:
    """Run the precomputation for one atmosphere: orders 2 to `resolution.orders`.

    The loop is Bruneton's `Model::Precompute`: single scattering, then for each order n the
    scattering density from order n-1 (and from order n-2 light on the ground), the ground
    irradiance of order n-1, and the radiance of order n along every table ray. Densities are
    kept per species and divided by the scattering coefficients; their sum over orders is what
    the ground observer gathers.

    With `next_order`, order N+1 is kept apart (`ScatteringField.next_order_xyz`). A `probe`
    (sun elevation, view elevations, relative azimuths) records each order's XYZ there
    (`ScatteringField.order_xyz`, shape (orders, azimuths, views, 3), from order 2).
    """
    if not 0.0 <= ground_albedo <= 1.0:
        raise ValueError(f"ground_albedo must be in [0, 1], got {ground_albedo!r}")
    optics = _optics(atm)
    grid = _grid(resolution)
    transmittance = _transmittance_table(optics, resolution)
    levels = _levels(grid, resolution, optics)
    width = WAVELENGTHS_NM.size
    shape = (grid.radii.size, grid.views, grid.azimuth.size, grid.sun_mu.size, 2 * width)
    last = resolution.orders + (1 if next_order else 0)
    albedo = ground_albedo / math.pi
    asymmetry = optics.asymmetry

    # Order 1 at every quadrature direction of every level, and order 2's density from it.
    density = np.empty(shape)
    ground_e1 = np.zeros((grid.sun_mu.size, width))
    for i, level in enumerate(levels):
        dirs = level.directions
        incident = np.empty((dirs.mu.size, dirs.azimuth.size, grid.sun_mu.size, width))
        azimuth = np.repeat(dirs.azimuth, grid.sun_mu.size)
        mu_s = np.tile(grid.sun_mu, dirs.azimuth.size)
        for q, ray in enumerate(level.direction_rays):
            nu = _nu(np.full_like(mu_s, ray.mu), mu_s, azimuth)
            rayleigh, aerosol = _single_scattering(
                ray, mu_s, nu, optics, transmittance, resolution.single_samples
            )
            light = optics.sun * (
                rayleigh * rayleigh_phase(nu)[:, None]
                + aerosol * henyey_greenstein_phase(nu, asymmetry)[:, None]
            )
            if ray.reaches_ground:
                light = light + _ground_light(ray, mu_s, nu, optics, transmittance, albedo, None)
            incident[q] = light.reshape(dirs.azimuth.size, grid.sun_mu.size, width)
        if i == 0:
            ground_e1 = _irradiance_from_directions(dirs, incident)
        flat = incident.reshape(dirs.mu.size * dirs.azimuth.size, -1)
        density[i, ..., :width] = (level.gather_rayleigh @ flat).reshape(shape[1:-1] + (width,))
        density[i, ..., width:] = (level.gather_aerosol @ flat).reshape(shape[1:-1] + (width,))

    def gathered(log_density: np.ndarray) -> np.ndarray:
        assert probe is not None
        sun, views, azimuths = probe
        return _gather_xyz(grid, optics, resolution, log_density, [sun], views, azimuths)[0]

    probes = []
    log_density = _log(density)
    if probe is not None:
        probes.append(gathered(log_density))
    total = density.copy()
    irradiance_before = ground_e1  # order n-2 ground irradiance for order n's density
    next_density: np.ndarray | None = None
    for order in range(3, last + 1):
        radiance = _radiance_table(grid, levels, log_density, optics, resolution)
        irradiance_now = _irradiance_from_table(levels[0], grid, radiance)
        density = np.empty(shape)
        for i, level in enumerate(levels):
            incident = radiance[i] + _ground_light_table(
                grid, level, i, optics, irradiance_before, albedo
            )
            flat = incident.reshape(grid.views * grid.azimuth.size, -1)
            density[i, ..., :width] = (level.table_rayleigh @ flat).reshape(shape[1:-1] + (width,))
            density[i, ..., width:] = (level.table_aerosol @ flat).reshape(shape[1:-1] + (width,))
        irradiance_before = irradiance_now
        log_density = _log(density)
        if probe is not None:
            probes.append(gathered(log_density))
        if order <= resolution.orders:
            total += density
        else:
            next_density = density
    return ScatteringField(
        resolution=resolution,
        ground_albedo=ground_albedo,
        _grid=grid,
        _optics=optics,
        _log_density=_log(total),
        _log_next=None if next_density is None else _log(next_density),
        order_xyz=np.array(probes) if probes else None,
    )


def _ground_light(
    ray: _Ray,
    mu_s: np.ndarray,
    nu: np.ndarray,
    optics: _Optics,
    transmittance: np.ndarray,
    albedo_over_pi: float,
    irradiance: tuple[_Grid, np.ndarray] | None,
) -> np.ndarray:
    """Light reflected toward a ray's start by the ground where the ray ends: (suns, w).

    The transmittance to the ground, times albedo / pi, times the ground irradiance at that
    point: the direct sun's (`irradiance` None) or an indirect one on the sun nodes.
    """
    mu_s_ground = np.clip((ray.r * mu_s + ray.length * nu) / _R, -1.0, 1.0)
    if irradiance is None:
        light = _ground_irradiance_direct(mu_s_ground, optics, transmittance)
    else:
        grid, table = irradiance
        light = _interpolate_irradiance(grid, table, mu_s_ground)
    path = ray.optical_length(np.array(ray.length))
    return light * np.exp(-(path @ optics.extinction)) * albedo_over_pi


def _ground_light_table(
    grid: _Grid,
    level: _Level,
    index: int,
    optics: _Optics,
    irradiance: np.ndarray,
    albedo_over_pi: float,
) -> np.ndarray:
    """`_ground_light` at every ground-reaching view node of a level: (views, az, suns, w)."""
    result = np.zeros((grid.views, grid.azimuth.size, grid.sun_mu.size, WAVELENGTHS_NM.size))
    for v, ray in enumerate(level.rays):
        if not ray.reaches_ground:
            continue
        mu_s, nu = _grid_suns(grid, ray.mu)
        light = _ground_light(ray, mu_s, nu, optics, None, albedo_over_pi, (grid, irradiance))
        result[v] = light.reshape(grid.azimuth.size, grid.sun_mu.size, -1)
    return result


def _irradiance_from_directions(directions: _DirectionSet, incident: np.ndarray) -> np.ndarray:
    """Bruneton's ComputeIndirectIrradiance at the ground from radiance on the quadrature."""
    sky = ~directions.reaches_ground
    weight = 2.0 * directions.weight[sky] * directions.mu[sky][:, None]
    return np.einsum("qa,qasw->sw", weight, incident[sky])


def _irradiance_from_table(level: _Level, grid: _Grid, radiance: np.ndarray) -> np.ndarray:
    """The ground irradiance of a radiance table (level 0), through the quadrature."""
    dirs = level.directions
    flat = radiance[0].reshape(grid.views * grid.azimuth.size, -1)
    incident = (level.table_to_directions @ flat).reshape(
        dirs.mu.size, dirs.azimuth.size, grid.sun_mu.size, WAVELENGTHS_NM.size
    )
    return _irradiance_from_directions(dirs, incident)


def _radiance_table(
    grid: _Grid,
    levels: list[_Level],
    log_density: np.ndarray,
    optics: _Optics,
    resolution: Resolution,
) -> np.ndarray:
    """Radiance of one order at every table node, (altitudes, views, azimuths, suns, w)."""
    result = np.empty(
        (grid.radii.size, grid.views, grid.azimuth.size, grid.sun_mu.size, WAVELENGTHS_NM.size)
    )
    for i, level in enumerate(levels):
        for v, ray in enumerate(level.rays):
            mu_s, nu = _grid_suns(grid, ray.mu)
            light = _march(grid, log_density, ray, mu_s, nu, optics, resolution.march_samples)
            result[i, v] = light.reshape(grid.azimuth.size, grid.sun_mu.size, -1)
    return result
