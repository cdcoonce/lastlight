"""Render v0 atmosphere: the optical properties that feed the single-scattering integral.

Each test names the physical effect it guards, so a failure reads as a physics bug and not a
number mismatch. Reference values come from outside this code base (Bucholtz 1995, the Angstrom
law, the definition of a Dobson unit), never from the model's own output.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from lastlight.render import WAVELENGTHS_NM, Atmosphere
from lastlight.render.atmosphere import (
    ATMOSPHERE_TOP_KM,
    EARTH_RADIUS_KM,
    LEVELS_KM,
    OZONE_LAYER_CENTRE_KM,
    RAYLEIGH_SCALE_HEIGHT_KM,
    layer_optics,
    ozone_absorption_per_atm_cm,
    rayleigh_optical_depth,
    solar_irradiance,
)

THICKNESS_KM = np.diff(LEVELS_KM)
MID_KM = 0.5 * (LEVELS_KM[:-1] + LEVELS_KM[1:])


def _vertical_column(per_km: np.ndarray) -> np.ndarray:
    """Vertical optical depth (per wavelength) of a layered per-km coefficient array."""
    return (per_km * THICKNESS_KM[:, None]).sum(axis=0)


def _index_of(wavelength_nm: float) -> int:
    return int(np.flatnonzero(wavelength_nm == WAVELENGTHS_NM)[0])


# --- the grid and the parameter object ------------------------------------------------------


def test_wavelength_grid_is_380_to_780_nm_in_10_nm_steps() -> None:
    assert WAVELENGTHS_NM[0] == 380.0
    assert WAVELENGTHS_NM[-1] == 780.0
    assert np.all(np.diff(WAVELENGTHS_NM) == 10.0)
    assert WAVELENGTHS_NM.shape == (41,)


def test_model_constants_are_the_ones_the_issue_specifies() -> None:
    """Earth radius 6371 km, atmosphere top 100 km, Rayleigh scale height 8 km, ozone at 22 km."""
    assert EARTH_RADIUS_KM == 6371.0
    assert ATMOSPHERE_TOP_KM == 100.0
    assert RAYLEIGH_SCALE_HEIGHT_KM == 8.0
    assert OZONE_LAYER_CENTRE_KM == 22.0


def test_atmosphere_defaults_are_the_documented_phoenix_background() -> None:
    atm = Atmosphere(aod_550=0.1)
    assert atm == Atmosphere(
        0.1,
        angstrom=1.3,
        aerosol_scale_height_km=1.5,
        ozone_du=280,
        pressure_hpa=970,
        asymmetry=0.7,
        single_scatter_albedo=0.93,
    )


def test_atmosphere_is_immutable_and_hashable() -> None:
    atm = Atmosphere(aod_550=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        atm.aod_550 = 0.2  # type: ignore[misc]
    assert hash(atm) == hash(Atmosphere(aod_550=0.1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"aod_550": -0.01},
        {"aerosol_scale_height_km": 0.0},
        {"ozone_du": -1.0},
        {"pressure_hpa": 0.0},
        {"asymmetry": 1.0},
        {"asymmetry": -1.0},
        {"single_scatter_albedo": 1.1},
        {"single_scatter_albedo": -0.1},
        {"angstrom": float("nan")},
    ],
)
def test_atmosphere_rejects_unphysical_parameters(overrides: dict[str, float]) -> None:
    params = {"aod_550": 0.1, **overrides}
    with pytest.raises(ValueError):
        Atmosphere(**params)


# --- Rayleigh scattering ----------------------------------------------------------------------


def test_rayleigh_optical_depth_at_550_nm_matches_bucholtz_1995() -> None:
    """Bucholtz (1995): 0.0973 for the standard atmosphere at 1013.25 hPa."""
    assert rayleigh_optical_depth(550.0, 1013.25) == pytest.approx(0.0973, rel=0.02)


def test_rayleigh_optical_depth_is_proportional_to_surface_pressure() -> None:
    """The scattering column is the air column, and the air column is pressure over gravity."""
    full = rayleigh_optical_depth(550.0, 1013.25)
    assert rayleigh_optical_depth(550.0, 506.625) == pytest.approx(full / 2, rel=1e-12)
    assert rayleigh_optical_depth(550.0, 970.0) == pytest.approx(full * 970.0 / 1013.25, rel=1e-12)


def test_rayleigh_scattering_falls_roughly_as_wavelength_to_the_minus_four() -> None:
    """Why the sky is blue: the log slope between 450 and 650 nm is near -4.08."""
    blue = rayleigh_optical_depth(450.0)
    red = rayleigh_optical_depth(650.0)
    slope = math.log(blue / red) / math.log(450.0 / 650.0)
    assert -4.25 < slope < -3.95


def test_layered_rayleigh_column_conserves_the_vertical_optical_depth() -> None:
    """Splitting the exponential atmosphere into shells must not create or lose air."""
    optics = layer_optics(Atmosphere(aod_550=0.0, ozone_du=0.0, pressure_hpa=1013.25))
    expected = np.array([rayleigh_optical_depth(w, 1013.25) for w in WAVELENGTHS_NM])
    kept = 1.0 - math.exp(-ATMOSPHERE_TOP_KM / RAYLEIGH_SCALE_HEIGHT_KM)
    np.testing.assert_allclose(
        _vertical_column(optics.rayleigh_scattering), expected * kept, rtol=1e-9
    )


# --- aerosol extinction ---------------------------------------------------------------------


@pytest.mark.parametrize("wavelength", [450.0, 550.0, 650.0])
def test_aerosol_extinction_follows_the_angstrom_law(wavelength: float) -> None:
    """tau(lambda) = tau(550) * (lambda / 550) ** -alpha, summed over the column."""
    atm = Atmosphere(aod_550=0.2, angstrom=1.3)
    column = _vertical_column(layer_optics(atm).aerosol_extinction)[_index_of(wavelength)]
    assert column == pytest.approx(0.2 * (wavelength / 550.0) ** -1.3, rel=1e-6)


def test_aerosol_extinction_is_grey_when_the_angstrom_exponent_is_zero() -> None:
    """Coarse dust has a flat spectrum: alpha = 0 gives the same optical depth at every colour."""
    column = _vertical_column(
        layer_optics(Atmosphere(aod_550=0.3, angstrom=0.0)).aerosol_extinction
    )
    np.testing.assert_allclose(column, 0.3, rtol=1e-6)


def test_aerosol_sits_in_an_exponential_layer_with_the_given_scale_height() -> None:
    """Half of an exponential column lies below H * ln 2."""
    for height in (0.5, 1.5, 3.0):
        extinction = layer_optics(
            Atmosphere(aod_550=0.2, aerosol_scale_height_km=height)
        ).aerosol_extinction[:, _index_of(550.0)]
        cumulative = np.cumsum(extinction * THICKNESS_KM) / extinction.dot(THICKNESS_KM)
        below_median = np.interp(height * math.log(2.0), LEVELS_KM[1:], cumulative)
        assert below_median == pytest.approx(0.5, abs=0.02)


def test_aerosol_scattering_is_the_single_scatter_albedo_share_of_extinction() -> None:
    """Absorption is what is left over: omega0 = scattering / extinction."""
    optics = layer_optics(Atmosphere(aod_550=0.3, single_scatter_albedo=0.8))
    np.testing.assert_allclose(optics.aerosol_scattering, 0.8 * optics.aerosol_extinction)


def test_no_aerosol_means_no_aerosol_optics() -> None:
    optics = layer_optics(Atmosphere(aod_550=0.0))
    assert not optics.aerosol_extinction.any()
    assert not optics.aerosol_scattering.any()


# --- ozone absorption -----------------------------------------------------------------------


def test_ozone_column_is_the_dobson_unit_times_the_absorption_coefficient() -> None:
    """1 DU is 0.001 atm-cm, so the vertical optical depth is k * DU / 1000."""
    atm = Atmosphere(aod_550=0.1, ozone_du=300.0)
    column = _vertical_column(layer_optics(atm).ozone_absorption)
    expected = ozone_absorption_per_atm_cm(WAVELENGTHS_NM) * 300.0 / 1000.0
    np.testing.assert_allclose(column, expected, rtol=1e-9)


def test_ozone_layer_is_centred_at_22_km() -> None:
    """The ozone layer is a layer, not a column of the whole atmosphere."""
    absorption = layer_optics(Atmosphere(aod_550=0.1, ozone_du=300.0)).ozone_absorption
    weights = absorption[:, _index_of(600.0)] * THICKNESS_KM
    centroid = float((weights * MID_KM).sum() / weights.sum())
    assert OZONE_LAYER_CENTRE_KM == 22.0
    assert centroid == pytest.approx(22.0, abs=0.3)
    assert weights[MID_KM < 5.0].sum() < 0.01 * weights.sum()


def test_zero_ozone_absorbs_nothing() -> None:
    assert not layer_optics(Atmosphere(aod_550=0.1, ozone_du=0.0)).ozone_absorption.any()


def test_chappuis_band_absorbs_orange_red_and_leaves_blue_alone() -> None:
    """The Chappuis band (about 500 to 700 nm) peaks in the orange; blue and deep red are clear."""
    k = {
        w: float(ozone_absorption_per_atm_cm(np.array([w]))[0])
        for w in (400.0, 450.0, 600.0, 780.0)
    }
    assert k[600.0] > 10 * max(k[400.0], k[450.0])
    assert k[600.0] > 10 * k[780.0]
    assert 0.08 < k[600.0] < 0.20  # (atm-cm)^-1; sigma = 5e-21 cm^2 at the peak


# --- the sun --------------------------------------------------------------------------------


def test_solar_irradiance_is_positive_and_peaks_in_the_visible_blue_green() -> None:
    sun = solar_irradiance(WAVELENGTHS_NM)
    assert sun.shape == WAVELENGTHS_NM.shape
    assert (sun > 0).all()
    assert 430.0 <= WAVELENGTHS_NM[int(np.argmax(sun))] <= 520.0
    # Extraterrestrial irradiance is about 1.9 W m^-2 nm^-1 near 500 nm.
    assert solar_irradiance(np.array([500.0]))[0] == pytest.approx(1.9, rel=0.05)


# --- the shell grid -------------------------------------------------------------------------


def test_shell_levels_span_the_atmosphere_and_resolve_the_boundary_layer() -> None:
    assert LEVELS_KM[0] == 0.0
    assert LEVELS_KM[-1] == ATMOSPHERE_TOP_KM == 100.0
    assert (np.diff(LEVELS_KM) > 0).all()
    assert np.diff(LEVELS_KM)[MID_KM < 1.0].max() <= 0.1
    assert np.diff(LEVELS_KM).max() <= 1.0
