"""The renderer: a spectral model of the atmosphere that paints the western twilight sky.

v0 is single scattering in a spherical-shell atmosphere (Rayleigh air, aerosol, ozone) with
Earth's shadow. v1 is v0 plus the light scattered more than once (and reflected by the ground),
read as CIE XYZ from a table precomputed with a port of Bruneton's 2017 method
(`lastlight.render.v1`, `lastlight.render.bruneton`). Both sit behind the same interface: `sky_xyz`,
`render_row` and `render_curve` take `engine="v0"` (the default) or `engine="v1"`; `sky_spectrum`
is v0 only. They also take `quadrature="registered"` (the default, pinned by v1's registration) or
`"adaptive"` (finer along optically thick low sightlines; what the daily v0 painting uses); see
`QUADRATURES`. See `docs/design.md`, section 4.

    atm = Atmosphere(aod_550=0.1)
    curve = render_curve(atm)          # colour along the sky, per sun and view elevation
    save_strip(render_strip(curve), "strip.png")
    twilight = render_curve(atm, engine="v1")
"""

from lastlight.render.atmosphere import WAVELENGTHS_NM, Atmosphere
from lastlight.render.colorimetry import spectrum_to_lab, spectrum_to_srgb
from lastlight.render.curve import (
    ENGINES,
    SUN_ELEVATIONS,
    VIEW_ELEVATIONS,
    render_curve,
    render_row,
    render_strip,
    save_strip,
    sky_xyz,
    strip_png_bytes,
)
from lastlight.render.transfer import QUADRATURES, sky_spectrum

__all__ = [
    "ENGINES",
    "QUADRATURES",
    "SUN_ELEVATIONS",
    "VIEW_ELEVATIONS",
    "WAVELENGTHS_NM",
    "Atmosphere",
    "render_curve",
    "render_row",
    "render_strip",
    "save_strip",
    "sky_spectrum",
    "sky_xyz",
    "spectrum_to_lab",
    "spectrum_to_srgb",
    "strip_png_bytes",
]
