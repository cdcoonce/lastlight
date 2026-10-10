"""The curve and the strip: the sky's colour along the sun's azimuth, as the sun sets.

`render_curve` is the forecast's product: for every sun elevation from +6 to -8 degrees in half
degree steps and every view elevation in `VIEW_ELEVATIONS`, the CIELAB colour of the sky.

Exposure convention (recorded choice). Each sun elevation is exposed on its own, so that its
brightest view elevation reads L* = 70 and the others are darker by their true ratio. The curve
therefore describes how colour varies along the sky at one moment, not how fast the evening
darkens. The camera will be normalised the same way, per sun elevation, before the two are
compared.

`SUN_ELEVATIONS` is the sweep of `geometry.ELEVATION_SWEEP` (issue 4); a test compares the two
once the geometry module exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from lastlight.render.atmosphere import Atmosphere
from lastlight.render.colorimetry import lab_to_srgb, spectrum_to_xyz, xyz_to_lab
from lastlight.render.transfer import sky_spectrum

SUN_ELEVATIONS: tuple[float, ...] = tuple(6.0 - 0.5 * step for step in range(29))
"""Sun elevations in degrees, from +6.0 down to -8.0 inclusive in 0.5 degree steps."""

VIEW_ELEVATIONS: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0)
"""View elevations in degrees, along the sun's azimuth."""

PEAK_LIGHTNESS = 70.0
"""CIELAB L* given to the brightest view elevation at each sun elevation."""

_PEAK_LUMINANCE = ((PEAK_LIGHTNESS + 16.0) / 116.0) ** 3

Curve = list[dict[str, float]]


def render_row(atm: Atmosphere, sun_elevation: float) -> Curve:
    """The sky's colour at every view elevation for one sun elevation, exposed for L* = 70."""
    spectra = [sky_spectrum(atm, sun_elevation, view) for view in VIEW_ELEVATIONS]
    xyz = np.array([spectrum_to_xyz(spectrum) for spectrum in spectra])
    brightest = xyz[:, 1].max()
    exposure = _PEAK_LUMINANCE / brightest if brightest > 0.0 else 0.0
    lab = xyz_to_lab(xyz * exposure)
    return [
        {
            "sun_elevation": float(sun_elevation),
            "view_elevation": float(view),
            "L": float(lightness),
            "a": float(a),
            "b": float(b),
        }
        for view, (lightness, a, b) in zip(VIEW_ELEVATIONS, lab, strict=True)
    ]


def render_curve(atm: Atmosphere) -> Curve:
    """One row per (sun elevation, view elevation): sun elevation descending, view ascending."""
    return [row for sun in SUN_ELEVATIONS for row in render_row(atm, sun)]


def render_strip(
    curve: Sequence[dict[str, float]], *, cell_width: int = 16, cell_height: int = 24
) -> Image.Image:
    """Paint the curve as an image: sun elevation across (highest at left), view elevation up.

    Each curve entry becomes one cell of sRGB. Rows come from the entries themselves, in any
    order, and must form a complete grid. The image carries no metadata.
    """
    if not curve:
        raise ValueError("cannot paint an empty curve")
    cells = {(row["sun_elevation"], row["view_elevation"]): row for row in curve}
    suns = sorted({sun for sun, _ in cells}, reverse=True)
    views = sorted({view for _, view in cells}, reverse=True)  # the top row is the highest view
    if len(cells) != len(curve) or len(cells) != len(suns) * len(views):
        raise ValueError("curve must be a rectangular grid of sun and view elevations")

    pixels = np.empty((len(views), len(suns), 3), dtype=np.uint8)
    for y, view in enumerate(views):
        for x, sun in enumerate(suns):
            row = cells[(sun, view)]
            rgb = np.array(lab_to_srgb(row["L"], row["a"], row["b"]))
            pixels[y, x] = np.rint(255.0 * rgb)
    blocks = np.repeat(np.repeat(pixels, cell_height, axis=0), cell_width, axis=1)
    return Image.fromarray(blocks)


def save_strip(image: Image.Image, path: str | Path) -> None:
    """Write the strip as a PNG with no ancillary chunks.

    The pixels are copied into a fresh image first, so nothing the caller attached to
    `image.info` (EXIF, ICC profile, dpi, text) can reach the file.
    """
    clean = Image.frombytes(image.mode, image.size, image.tobytes())
    clean.save(path, format="PNG")
