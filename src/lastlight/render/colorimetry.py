"""From a spectrum to a colour: CIE 1931 2-degree XYZ, CIELAB and sRGB, with a D65 white.

`colour-science` (BSD-3-Clause) supplies the colour-matching functions, the D65 white point and
the XYZ to CIELAB and sRGB maths. The spectrum-to-XYZ integral is a plain sum on the renderer's
10 nm grid, which is accurate for the smooth spectra of a clear sky.

Convention for relative luminance: Y = 1 is a perfect white diffuser lit by the extraterrestrial
sun at normal incidence, i.e. a spectrum equal to the solar irradiance over pi. A sky radiance
spectrum from `sky_spectrum` is then a fraction of that, and `exposure` is the factor a camera
(or the curve's normalisation) applies to bring it into range. There is no chromatic adaptation:
the colour is that of the light, read against a D65 white.
"""

from __future__ import annotations

import warnings
from functools import cache
from types import ModuleType

import numpy as np

from lastlight.render.atmosphere import WAVELENGTHS_NM, solar_irradiance

_OBSERVER = "CIE 1931 2 Degree Standard Observer"


@cache
def _colour() -> ModuleType:
    """Import colour-science without its notices about optional SciPy and Matplotlib."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour

    return colour


@cache
def _tables() -> tuple[np.ndarray, float, np.ndarray]:
    """Colour-matching functions on the grid, the white card's luminance, the D65 white point."""
    colour = _colour()
    cmfs = colour.MSDS_CMFS[_OBSERVER]
    positions = np.searchsorted(cmfs.wavelengths, WAVELENGTHS_NM)
    if not np.array_equal(cmfs.wavelengths[positions], WAVELENGTHS_NM):
        raise RuntimeError("the colour-matching functions do not cover the renderer's grid")
    matching = np.asarray(cmfs.values)[positions]
    white_card = solar_irradiance(WAVELENGTHS_NM) / np.pi
    white_point = np.asarray(colour.CCS_ILLUMINANTS[_OBSERVER]["D65"], dtype=float)
    return matching, float(white_card @ matching[:, 1]), white_point


def spectrum_to_xyz(spectrum: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """CIE XYZ of a radiance spectrum on `WAVELENGTHS_NM`, scaled by `exposure`."""
    matching, norm, _ = _tables()
    return exposure * (np.asarray(spectrum, dtype=float) @ matching) / norm


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """CIELAB of XYZ values (last axis of size 3) against the D65 white."""
    _, _, white_point = _tables()
    return np.asarray(_colour().XYZ_to_Lab(xyz, illuminant=white_point))


def _xyz_to_srgb(xyz: np.ndarray) -> tuple[float, float, float]:
    _, _, white_point = _tables()
    encoded = np.asarray(_colour().XYZ_to_sRGB(xyz, illuminant=white_point))
    r, g, b = (float(channel) for channel in np.clip(encoded, 0.0, 1.0))
    return r, g, b


def spectrum_to_lab(spectrum: np.ndarray, exposure: float = 1.0) -> tuple[float, float, float]:
    """CIELAB (L*, a*, b*) of a radiance spectrum, scaled by `exposure`."""
    lightness, a, b = (float(v) for v in xyz_to_lab(spectrum_to_xyz(spectrum, exposure)))
    return lightness, a, b


def spectrum_to_srgb(spectrum: np.ndarray, exposure: float) -> tuple[float, float, float]:
    """Encoded sRGB (r, g, b), each in [0, 1], of a radiance spectrum scaled by `exposure`.

    Out-of-gamut and over-exposed values are clipped.
    """
    return _xyz_to_srgb(spectrum_to_xyz(spectrum, exposure))


def lab_to_srgb(lightness: float, a: float, b: float) -> tuple[float, float, float]:
    """Encoded sRGB (r, g, b), each in [0, 1], of a CIELAB colour against the D65 white."""
    _, _, white_point = _tables()
    xyz = _colour().Lab_to_XYZ(np.array([lightness, a, b]), illuminant=white_point)
    return _xyz_to_srgb(xyz)
