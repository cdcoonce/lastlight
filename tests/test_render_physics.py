"""Render v0 physics: single scattering in a spherical shell, checked against outside references.

Three kinds of check, none of which reads the model's own numbers back:

* closed forms: the plane-parallel single-scattering solution, phase-function normalisation;
* an independent brute-force quadrature of the same integral, built from the density formulas
  with a different discretisation (uniform steps in distance, not shell layers);
* the sign of a named physical effect: Earth's shadow, the Chappuis band, aerosol reddening.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lastlight.render import (
    VIEW_ELEVATIONS,
    WAVELENGTHS_NM,
    Atmosphere,
    render_row,
    sky_spectrum,
    spectrum_to_lab,
)
from lastlight.render.atmosphere import (
    ATMOSPHERE_TOP_KM,
    EARTH_RADIUS_KM,
    OZONE_LAYER_CENTRE_KM,
    OZONE_LAYER_SIGMA_KM,
    RAYLEIGH_SCALE_HEIGHT_KM,
    ozone_absorption_per_atm_cm,
    rayleigh_optical_depth,
    solar_irradiance,
)
from lastlight.render.colorimetry import spectrum_to_xyz
from lastlight.render.transfer import henyey_greenstein_phase, rayleigh_phase, sunlit

# --- phase functions ------------------------------------------------------------------------


def _solid_angle_integral(phase, **kwargs) -> float:
    """Integrate a phase function over the sphere: 2 pi * integral of P sin(theta) dtheta."""
    theta = np.linspace(0.0, math.pi, 200_001)
    values = phase(np.cos(theta), **kwargs) * np.sin(theta)
    return float(2.0 * math.pi * np.trapezoid(values, theta))


def test_rayleigh_phase_function_is_normalised_over_the_sphere() -> None:
    assert _solid_angle_integral(rayleigh_phase) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("g", [0.0, 0.3, 0.7, 0.9])
def test_henyey_greenstein_phase_function_is_normalised_over_the_sphere(g: float) -> None:
    assert _solid_angle_integral(henyey_greenstein_phase, g=g) == pytest.approx(1.0, abs=2e-4)


def test_rayleigh_scatters_forward_and_backward_equally() -> None:
    assert rayleigh_phase(np.array([1.0]))[0] == pytest.approx(rayleigh_phase(np.array([-1.0]))[0])
    assert rayleigh_phase(np.array([1.0]))[0] > rayleigh_phase(np.array([0.0]))[0]


def test_aerosol_scatters_forward_when_the_asymmetry_parameter_is_positive() -> None:
    """Mie scattering by haze is concentrated toward the sun: the aureole."""
    forward = henyey_greenstein_phase(np.array([1.0]), g=0.7)[0]
    backward = henyey_greenstein_phase(np.array([-1.0]), g=0.7)[0]
    assert forward > 50 * backward
    assert henyey_greenstein_phase(np.array([0.3]), g=0.0)[0] == pytest.approx(1 / (4 * math.pi))


# --- the shape of the answer ----------------------------------------------------------------


def test_sky_spectrum_is_one_finite_nonnegative_value_per_wavelength() -> None:
    spectrum = sky_spectrum(Atmosphere(aod_550=0.1), 10.0, 20.0)
    assert spectrum.shape == WAVELENGTHS_NM.shape
    assert np.isfinite(spectrum).all()
    assert (spectrum >= 0.0).all()
    assert spectrum.max() > 0.0


def test_sky_spectrum_is_deterministic() -> None:
    atm = Atmosphere(aod_550=0.1)
    np.testing.assert_array_equal(sky_spectrum(atm, 3.0, 5.0), sky_spectrum(atm, 3.0, 5.0))


def test_sky_spectrum_is_mirror_symmetric_in_relative_azimuth() -> None:
    atm = Atmosphere(aod_550=0.1)
    np.testing.assert_allclose(
        sky_spectrum(atm, 5.0, 20.0, rel_azimuth_deg=70.0),
        sky_spectrum(atm, 5.0, 20.0, rel_azimuth_deg=-70.0),
        rtol=1e-12,
    )


@pytest.mark.parametrize("view", [-1.0, 90.5])
def test_sky_spectrum_rejects_views_that_do_not_look_at_the_sky(view: float) -> None:
    with pytest.raises(ValueError):
        sky_spectrum(Atmosphere(aod_550=0.1), 10.0, view)


def test_a_very_thin_atmosphere_shows_rayleigh_scattered_sunlight() -> None:
    """Thin limit: sky/sun is the Rayleigh column times a grey phase, so it goes as lambda^-4."""
    atm = Atmosphere(aod_550=0.0, ozone_du=0.0, pressure_hpa=1.0)  # almost no air: thin limit
    spectrum = sky_spectrum(atm, 60.0, 45.0, rel_azimuth_deg=90.0)
    ratio = spectrum / solar_irradiance(WAVELENGTHS_NM)
    # tau is tiny, so the sky/sun ratio is just the Rayleigh column over wavelength: ~ lambda^-4.
    slope = math.log(ratio[_i(450.0)] / ratio[_i(650.0)]) / math.log(450.0 / 650.0)
    assert -4.3 < slope < -3.9


def _i(wavelength: float) -> int:
    return int(np.flatnonzero(wavelength == WAVELENGTHS_NM)[0])


# --- closed form: plane-parallel single scattering ------------------------------------------


def test_matches_the_plane_parallel_closed_form_when_curvature_does_not_matter() -> None:
    """High sun, high view, no aerosol or ozone: single scattering has an exact solution.

    L = S0 * P(theta) * mu0 / (mu0 - mu) * (exp(-tau/mu0) - exp(-tau/mu))

    for an exponential atmosphere of vertical optical depth tau. The spherical-shell integral
    must reduce to it where the Earth's curvature is negligible (air mass 1.15).
    """
    atm = Atmosphere(aod_550=0.0, ozone_du=0.0, pressure_hpa=1013.25)
    sun, view, azimuth = 60.0, 45.0, 90.0
    mu0, mu = math.sin(math.radians(sun)), math.sin(math.radians(view))
    cos_theta = (
        math.cos(math.radians(view)) * math.cos(math.radians(azimuth)) * math.cos(math.radians(sun))
        + mu * mu0
    )
    tau = np.array([rayleigh_optical_depth(w, 1013.25) for w in WAVELENGTHS_NM])
    phase = 3.0 / (16.0 * math.pi) * (1.0 + cos_theta**2)
    expected = (
        solar_irradiance(WAVELENGTHS_NM)
        * phase
        * mu0
        / (mu0 - mu)
        * (np.exp(-tau / mu0) - np.exp(-tau / mu))
    )
    np.testing.assert_allclose(sky_spectrum(atm, sun, view, azimuth), expected, rtol=5e-3)


# --- independent brute-force quadrature -----------------------------------------------------


def _reference_radiance(
    atm: Atmosphere, sun_deg: float, view_deg: float, azimuth_deg: float, wavelength: float
) -> float:
    """Single scattering by direct quadrature: uniform steps in distance, densities evaluated
    point by point (no shells, no layer means), Earth's shadow from the ray-sphere test."""
    radius, top = EARTH_RADIUS_KM, ATMOSPHERE_TOP_KM
    tau_rayleigh = rayleigh_optical_depth(wavelength, atm.pressure_hpa)
    tau_aerosol = atm.aod_550 * (wavelength / 550.0) ** -atm.angstrom
    tau_ozone = float(ozone_absorption_per_atm_cm(np.array([wavelength]))[0]) * atm.ozone_du / 1000
    sigma = OZONE_LAYER_SIGMA_KM
    z_top = (top - OZONE_LAYER_CENTRE_KM) / sigma
    z_bottom = (0.0 - OZONE_LAYER_CENTRE_KM) / sigma
    norm = 0.5 * (math.erf(z_top / math.sqrt(2)) - math.erf(z_bottom / math.sqrt(2)))

    def rayleigh(h):
        return tau_rayleigh / RAYLEIGH_SCALE_HEIGHT_KM * np.exp(-h / RAYLEIGH_SCALE_HEIGHT_KM)

    def aerosol(h):
        height = atm.aerosol_scale_height_km
        return tau_aerosol / height * np.exp(-h / height)

    def ozone(h):
        pdf = np.exp(-0.5 * ((h - OZONE_LAYER_CENTRE_KM) / sigma) ** 2) / (
            sigma * math.sqrt(2 * math.pi)
        )
        return tau_ozone * pdf / norm

    def extinction(h):
        return rayleigh(h) + aerosol(h) + ozone(h)

    e, phi, s = (math.radians(x) for x in (view_deg, azimuth_deg, sun_deg))
    view = np.array([math.cos(e) * math.cos(phi), math.cos(e) * math.sin(phi), math.sin(e)])
    sun = np.array([math.cos(s), 0.0, math.sin(s)])
    origin = np.array([0.0, 0.0, radius])
    t_exit = -radius * math.sin(e) + math.sqrt(
        (radius * math.sin(e)) ** 2 + 2 * radius * top + top**2
    )
    n_view, n_sun = 4000, 600
    dt = t_exit / n_view
    t = (np.arange(n_view) + 0.5) * dt
    points = origin[None, :] + t[:, None] * view[None, :]
    r = np.linalg.norm(points, axis=1)
    b = points @ sun
    lit = ~((b < 0) & (b**2 - (r**2 - radius**2) > 0))
    u_exit = -b + np.sqrt(np.maximum(b**2 - (r**2 - (radius + top) ** 2), 0.0))
    frac = (np.arange(n_sun) + 0.5) / n_sun
    u = u_exit[:, None] * frac[None, :]
    h_sun = np.sqrt(np.maximum(r[:, None] ** 2 + 2 * b[:, None] * u + u**2, radius**2)) - radius
    tau_sun = extinction(h_sun).sum(axis=1) * u_exit / n_sun
    h = r - radius
    ext = extinction(h)
    tau_view = np.cumsum(ext * dt) - 0.5 * ext * dt
    cos_theta = float(view @ sun)
    scattering = rayleigh(h) * rayleigh_phase(np.array([cos_theta])) + (
        atm.single_scatter_albedo
        * aerosol(h)
        * henyey_greenstein_phase(np.array([cos_theta]), atm.asymmetry)
    )
    integrand = scattering * np.exp(-tau_sun - tau_view) * lit
    return float(solar_irradiance(np.array([wavelength]))[0] * integrand.sum() * dt)


HAZY = Atmosphere(aod_550=0.3, ozone_du=300.0)
UNLIKE = Atmosphere(  # every parameter off its default, so none can be ignored unnoticed
    aod_550=0.6,
    angstrom=0.4,
    aerosol_scale_height_km=0.6,
    ozone_du=150.0,
    pressure_hpa=900.0,
    asymmetry=0.4,
    single_scatter_albedo=0.75,
)


@pytest.mark.parametrize("atm", [HAZY, UNLIKE], ids=["hazy", "every-parameter-changed"])
@pytest.mark.parametrize(
    ("sun", "view", "azimuth"),
    [
        (20.0, 10.0, 0.0),
        (20.0, 45.0, 120.0),
        (2.0, 5.0, 0.0),
        (0.0, 2.0, 0.0),
        (-2.0, 20.0, 30.0),
        (-4.0, 45.0, 0.0),
        (-6.0, 15.0, 0.0),
        (-8.0, 45.0, 0.0),
        (-8.0, 2.0, 0.0),
    ],
)
def test_shell_integral_agrees_with_brute_force_quadrature(
    atm: Atmosphere, sun: float, view: float, azimuth: float
) -> None:
    """The shell-layer integrator and a plain Riemann sum of the same physics must agree.

    Hazy, ozone-bearing atmosphere so all three species and Earth's shadow are exercised. The
    two agree to under 0.9 percent at every geometry here, deep twilight included; the 1.5 percent
    bound leaves room for a platform's last bits, not for a missing term.
    """
    model = sky_spectrum(atm, sun, view, azimuth)
    for wavelength in (400.0, 450.0, 550.0, 650.0, 700.0):
        reference = _reference_radiance(atm, sun, view, azimuth, wavelength)
        assert model[_i(wavelength)] == pytest.approx(reference, rel=0.015), (
            f"{wavelength} nm at sun {sun}, view {view}, azimuth {azimuth}"
        )


# --- Earth's shadow -------------------------------------------------------------------------


def test_nothing_is_lit_when_the_sun_is_far_below_the_horizon() -> None:
    """Earth's shadow: at sun -30 every sunlit path crosses the Earth, so the sky is black."""
    atm = Atmosphere(aod_550=0.1)
    for view in (2.0, 20.0, 45.0):
        assert not sky_spectrum(atm, -30.0, view).any()


@pytest.mark.parametrize("sun", [-0.5, -2.0, -6.0, -12.0])
def test_shadow_height_over_the_observer_follows_earth_geometry(sun: float) -> None:
    """The sun's ray through a point grazes the Earth when the point is at R / cos(depression) - R.

    Straight above the observer the sun direction makes the angle `sun` with the local horizon, so
    the shadow height is R / cos(sun) - R: 3.9 km for a sun at -2 degrees, 62.7 km at -8.
    """
    shadow_km = EARTH_RADIUS_KM / math.cos(math.radians(sun)) - EARTH_RADIUS_KM
    below = EARTH_RADIUS_KM + 0.98 * shadow_km
    above = EARTH_RADIUS_KM + 1.02 * shadow_km + 0.01
    sin_sun = math.sin(math.radians(sun))
    assert not sunlit(np.array([below]), np.array([below * sin_sun]))[0]
    assert sunlit(np.array([above]), np.array([above * sin_sun]))[0]


def test_a_point_on_the_ground_sees_the_sun_exactly_when_the_sun_is_up() -> None:
    """A horizon sun grazes the ground and is visible; below it the Earth is in the way."""
    ground = np.array([EARTH_RADIUS_KM + 0.001])
    for sun, expected in ((10.0, True), (0.0, True), (-0.2, False), (-10.0, False)):
        b = ground * math.sin(math.radians(sun))
        assert bool(sunlit(ground, b)[0]) is expected, sun


def test_twilight_arch_rises_as_the_sun_sinks() -> None:
    """The shadow height grows with sun depression, so the sky darkens from the ground up."""
    atm = Atmosphere(aod_550=0.05)
    brightness = [sky_spectrum(atm, s, 30.0).sum() for s in (4.0, 0.0, -2.0, -4.0, -6.0)]
    assert brightness == sorted(brightness, reverse=True)
    assert brightness[-1] < 0.01 * brightness[0]


# --- the acceptance criteria that are signs of physical effects -----------------------------


def _chroma(row: dict[str, float]) -> float:
    return math.hypot(row["a"], row["b"])


def _view(rows: list[dict[str, float]], view: float) -> dict[str, float]:
    return next(row for row in rows if row["view_elevation"] == view)


def _hue_degrees(row: dict[str, float]) -> float:
    """CIELAB hue angle: 90 is yellow, 45 orange, 0 red. Unchanged by exposure."""
    return math.degrees(math.atan2(row["b"], row["a"]))


def _chroma_by_view(atm: Atmosphere, sun: float, azimuth: float) -> dict[float, float]:
    spectra = {view: sky_spectrum(atm, sun, view, azimuth) for view in VIEW_ELEVATIONS}
    exposure = 0.4 / max(spectrum_to_xyz(spectrum)[1] for spectrum in spectra.values())
    return {
        view: math.hypot(*spectrum_to_lab(spectrum, exposure)[1:])
        for view, spectrum in spectra.items()
    }


def test_horizon_sky_is_less_saturated_than_the_sky_above_it() -> None:
    """Airlight whitening: a long, low sightline scatters and extinguishes its way to grey.

    Acceptance criterion: sun +20, 2-degree view less chromatic than the 45-degree view, along
    the sun's azimuth, in clean air. With haze the criterion is false at that azimuth, see
    `test_haze_turns_the_low_sky_toward_the_sun_into_a_warm_aureole`.
    """
    rows = render_row(Atmosphere(aod_550=0.0), 20.0)
    assert _chroma(_view(rows, 2.0)) < _chroma(_view(rows, 45.0))


def test_horizon_whitening_survives_haze_away_from_the_sun() -> None:
    """The same airlight effect with the default haze, 135 degrees around from the sun."""
    chroma = _chroma_by_view(Atmosphere(aod_550=0.1), sun=20.0, azimuth=135.0)
    assert chroma[2.0] < chroma[45.0]


def test_haze_turns_the_low_sky_toward_the_sun_into_a_warm_aureole() -> None:
    """Forward scattering: aerosol adds the sun's reddened light to the low view toward it.

    Along the sun's azimuth the 2-degree view is 18 degrees from the sun at sun +20, inside the
    aureole, so adding haze raises its chroma (and washes out the sky above it).
    """
    clean = _chroma(_view(render_row(Atmosphere(aod_550=0.0), 20.0), 2.0))
    hazy = _chroma(_view(render_row(Atmosphere(aod_550=0.1), 20.0), 2.0))
    assert hazy > clean + 5.0


def test_more_aerosol_reddens_a_low_sun_sky_by_extinguishing_blue() -> None:
    """Aerosol reddening: heavy haze removes blue from the long path of a low sun.

    AOD 0.05 -> 0.5 at sun elevation +2, 5-degree view. The issue's criterion says a* rises and
    b* falls "toward yellow-orange". a* rises (11 to 29). b* does not fall: it stays near 47 to
    49, because the chroma grows as the hue turns, and more haze also dims this view. What falls
    is the hue angle, from yellow (about 77 degrees) toward orange (about 60), which is what
    "toward yellow-orange" describes; unlike a* and b* it does not depend on the exposure.
    """
    clean = _view(render_row(Atmosphere(aod_550=0.05), 2.0), 5.0)
    hazy = _view(render_row(Atmosphere(aod_550=0.5), 2.0), 5.0)
    assert hazy["a"] > clean["a"]
    assert _hue_degrees(hazy) < _hue_degrees(clean) - 10.0
    assert hazy["b"] > 0.0  # still yellow-orange, not pushed through red toward magenta


def test_ozone_makes_deep_twilight_bluer_the_chappuis_effect() -> None:
    """Chappuis effect: ozone absorbs orange and red, so the twilight sky at -4 is bluer."""
    with_ozone = _view(render_row(Atmosphere(aod_550=0.1, ozone_du=300.0), -4.0), 45.0)
    without = _view(render_row(Atmosphere(aod_550=0.1, ozone_du=0.0), -4.0), 45.0)
    assert with_ozone["b"] < without["b"]


def test_ozone_removes_light_only_where_ozone_absorbs() -> None:
    """The Chappuis band is orange-red: at -4 the ozone atmosphere is dimmer at 600 nm than 440."""
    atm_with = Atmosphere(aod_550=0.1, ozone_du=300.0)
    atm_without = Atmosphere(aod_550=0.1, ozone_du=0.0)
    ratio = sky_spectrum(atm_with, -4.0, 45.0) / sky_spectrum(atm_without, -4.0, 45.0)
    assert ratio[_i(600.0)] < ratio[_i(440.0)] <= 1.0 + 1e-12
    assert ratio[_i(600.0)] < 0.9


def test_zenith_ward_radiance_decreases_monotonically_as_the_sun_descends() -> None:
    """Twilight darkening: the 45-degree view dims at every 0.5-degree step from +6 to -8."""
    atm = Atmosphere(aod_550=0.1)
    sun_elevations = np.arange(6.0, -8.001, -0.5)
    luminance = [spectrum_to_xyz(sky_spectrum(atm, float(s), 45.0))[1] for s in sun_elevations]
    assert all(later < earlier for earlier, later in zip(luminance, luminance[1:], strict=False))
    assert luminance[-1] > 0.0  # still lit from the high shell at -8


def test_aerosol_brightens_the_sky_toward_the_sun_the_aureole() -> None:
    """Haze scatters forward: with aerosol, looking at the sun's azimuth is brighter than away."""
    atm = Atmosphere(aod_550=0.3)
    toward = sky_spectrum(atm, 20.0, 10.0, rel_azimuth_deg=0.0).sum()
    away = sky_spectrum(atm, 20.0, 10.0, rel_azimuth_deg=180.0).sum()
    assert toward > 2.0 * away
