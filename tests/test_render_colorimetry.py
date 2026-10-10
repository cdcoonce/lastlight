"""Spectrum to colour: CIE 1931 2-degree observer, D65 white, CIELAB and sRGB.

The oracle is `colour.sd_to_XYZ`, which integrates at 1 nm and knows nothing about our 10 nm
grid, our normalisation or our white card; and a neutral test: a D65 spectrum must come out grey.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lastlight.render import WAVELENGTHS_NM, spectrum_to_lab, spectrum_to_srgb
from lastlight.render.atmosphere import solar_irradiance
from lastlight.render.colorimetry import lab_to_srgb, spectrum_to_xyz

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import colour

OBSERVER = "CIE 1931 2 Degree Standard Observer"


def _d65_on_the_grid() -> np.ndarray:
    d65 = colour.SDS_ILLUMINANTS["D65"]
    return np.array([d65[w] for w in WAVELENGTHS_NM])


def _gaussian(centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS_NM - centre) / width) ** 2)


def test_a_d65_spectrum_is_neutral_in_lab_and_srgb() -> None:
    spectrum = _d65_on_the_grid()
    _, a, b = spectrum_to_lab(spectrum, exposure=0.5 / spectrum_to_xyz(spectrum)[1])
    assert abs(a) < 0.5
    assert abs(b) < 0.5
    r, g, blue = spectrum_to_srgb(spectrum, exposure=0.5 / spectrum_to_xyz(spectrum)[1])
    assert max(r, g, blue) - min(r, g, blue) < 0.01


@pytest.mark.parametrize(
    ("centre", "width"),
    [(450.0, 30.0), (520.0, 40.0), (600.0, 35.0), (650.0, 50.0), (550.0, 120.0)],
)
def test_chromaticity_matches_the_colour_science_integration(centre: float, width: float) -> None:
    """Our 10 nm weights give the same chromaticity as colour's 1 nm integration."""
    spectrum = _gaussian(centre, width)
    xyz = spectrum_to_xyz(spectrum)
    ours = xyz / xyz.sum()
    sd = colour.SpectralDistribution(dict(zip(WAVELENGTHS_NM, spectrum, strict=True)))
    cmfs = colour.MSDS_CMFS[OBSERVER]
    flat = colour.SpectralDistribution(
        dict(zip(WAVELENGTHS_NM, np.ones_like(WAVELENGTHS_NM), strict=True))
    )
    reference = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=flat, method="Integration")
    reference = reference / reference.sum()
    np.testing.assert_allclose(ours, reference, atol=2e-3)


def test_xyz_is_linear_in_exposure_and_in_the_spectrum() -> None:
    spectrum = _gaussian(560.0, 60.0)
    np.testing.assert_allclose(
        spectrum_to_xyz(spectrum, exposure=3.0), 3.0 * spectrum_to_xyz(spectrum)
    )
    np.testing.assert_allclose(
        spectrum_to_xyz(spectrum + _gaussian(450.0, 20.0)),
        spectrum_to_xyz(spectrum) + spectrum_to_xyz(_gaussian(450.0, 20.0)),
    )


def test_the_white_card_under_the_sun_has_unit_luminance() -> None:
    """Y = 1 is a perfect white diffuser lit by the extraterrestrial sun at normal incidence."""
    white_card = solar_irradiance(WAVELENGTHS_NM) / np.pi
    assert spectrum_to_xyz(white_card)[1] == pytest.approx(1.0, rel=1e-12)


def test_lightness_follows_the_cie_cube_root_of_luminance() -> None:
    spectrum = _gaussian(555.0, 80.0)
    y = spectrum_to_xyz(spectrum)[1]
    exposure = ((70.0 + 16.0) / 116.0) ** 3 / y
    lightness, _, _ = spectrum_to_lab(spectrum, exposure=exposure)
    assert lightness == pytest.approx(70.0, abs=1e-9)


def test_a_black_spectrum_is_black() -> None:
    zeros = np.zeros_like(WAVELENGTHS_NM)
    assert spectrum_to_lab(zeros) == (0.0, 0.0, 0.0)
    assert spectrum_to_srgb(zeros, exposure=1.0) == (0.0, 0.0, 0.0)


def test_warm_light_has_positive_a_and_b_and_blue_light_has_negative_b() -> None:
    orange = spectrum_to_lab(
        _gaussian(610.0, 25.0), exposure=0.3 / spectrum_to_xyz(_gaussian(610.0, 25.0))[1]
    )
    blue = spectrum_to_lab(
        _gaussian(460.0, 20.0), exposure=0.3 / spectrum_to_xyz(_gaussian(460.0, 20.0))[1]
    )
    assert orange[1] > 20.0
    assert orange[2] > 20.0
    assert blue[2] < -20.0


def test_srgb_values_are_encoded_and_clipped_to_the_unit_cube() -> None:
    saturated = spectrum_to_srgb(_gaussian(520.0, 8.0), exposure=50.0)  # far out of gamut
    assert all(0.0 <= channel <= 1.0 for channel in saturated)
    mid_grey_luminance = spectrum_to_srgb(
        _d65_on_the_grid(), exposure=0.2 / spectrum_to_xyz(_d65_on_the_grid())[1]
    )
    # Y = 0.2 encodes to about 0.484 in sRGB, not 0.2: the transfer function is applied.
    assert mid_grey_luminance[0] == pytest.approx(0.484, abs=0.02)


def test_lab_to_srgb_agrees_with_going_straight_from_the_spectrum() -> None:
    """The strip is painted from stored Lab values; it must match the direct conversion."""
    spectrum = _gaussian(580.0, 70.0) + 0.3 * _gaussian(450.0, 30.0)
    exposure = 0.25 / spectrum_to_xyz(spectrum)[1]
    direct = spectrum_to_srgb(spectrum, exposure)
    via_lab = lab_to_srgb(*spectrum_to_lab(spectrum, exposure))
    np.testing.assert_allclose(via_lab, direct, atol=1e-9)
