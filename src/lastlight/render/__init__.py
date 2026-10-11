"""The renderer: a spectral model of the atmosphere that paints the western twilight sky.

v0 is single scattering in a spherical-shell atmosphere (Rayleigh air, aerosol, ozone) with
Earth's shadow; v1 (a Bruneton precomputed-scattering table) will replace it behind the same
interface. See `docs/design.md`, section 4.

    atm = Atmosphere(aod_550=0.1)
    curve = render_curve(atm)          # colour along the sky, per sun and view elevation
    save_strip(render_strip(curve), "strip.png")
"""

from lastlight.render.atmosphere import WAVELENGTHS_NM, Atmosphere
from lastlight.render.colorimetry import spectrum_to_lab, spectrum_to_srgb
from lastlight.render.curve import (
    SUN_ELEVATIONS,
    VIEW_ELEVATIONS,
    render_curve,
    render_row,
    render_strip,
    save_strip,
)
from lastlight.render.transfer import sky_spectrum

__all__ = [
    "SUN_ELEVATIONS",
    "VIEW_ELEVATIONS",
    "WAVELENGTHS_NM",
    "Atmosphere",
    "render_curve",
    "render_row",
    "render_strip",
    "save_strip",
    "sky_spectrum",
    "spectrum_to_lab",
    "spectrum_to_srgb",
]
