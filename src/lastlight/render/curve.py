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

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from lastlight.render import v1
from lastlight.render.atmosphere import Atmosphere
from lastlight.render.colorimetry import lab_to_srgb, spectrum_to_xyz, xyz_to_lab
from lastlight.render.transfer import sky_spectrum

ENGINES: tuple[str, ...] = ("v0", "v1")
"""The renderers: v0 is single scattering; v1 adds multiple scattering from the Bruneton table."""

SUN_ELEVATIONS: tuple[float, ...] = tuple(6.0 - 0.5 * step for step in range(29))
"""Sun elevations in degrees, from +6.0 down to -8.0 inclusive in 0.5 degree steps."""

VIEW_ELEVATIONS: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0)
"""View elevations in degrees, along the sun's azimuth."""

PEAK_LIGHTNESS = 70.0
"""CIELAB L* given to the brightest view elevation at each sun elevation."""

_PEAK_LUMINANCE = ((PEAK_LIGHTNESS + 16.0) / 116.0) ** 3

Curve = list[dict[str, float]]


def check_engine(engine: str) -> None:
    """Raise `ValueError` unless `engine` is one of `ENGINES`."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")


def sky_xyz(
    atm: Atmosphere,
    sun_elevation: float,
    view_elevations: Sequence[float],
    rel_azimuth_deg: float = 0.0,
    *,
    engine: str = "v0",
) -> np.ndarray:
    """CIE XYZ of the sky at each of `view_elevations`, shape (views, 3), before any exposure.

    The view is `rel_azimuth_deg` round from the sun's azimuth (0 is the solar vertical, the
    default); the elevations may be any heights from 0 to 90 degrees. XYZ is relative to a white
    card lit by the extraterrestrial sun (`lastlight.render.colorimetry`), so contrasts taken from
    it (ratios between channels) need no exposure at all.

    `engine="v0"` (the default) is single scattering. `engine="v1"` adds the multiply scattered
    light of the Bruneton table (`lastlight.render.v1`) to exactly the v0 result; it raises
    `ValueError` outside the table's domain.
    """
    check_engine(engine)
    if engine == "v1":
        v1.check_domain(atm, sun_elevation, view_elevations)
    spectra = [sky_spectrum(atm, sun_elevation, view, rel_azimuth_deg) for view in view_elevations]
    xyz = np.array([spectrum_to_xyz(spectrum) for spectrum in spectra])
    if engine == "v1":
        xyz = xyz + v1.multiple_scattering_xyz(atm, sun_elevation, view_elevations, rel_azimuth_deg)
    return xyz


def render_row(
    atm: Atmosphere, sun_elevation: float, rel_azimuth_deg: float = 0.0, *, engine: str = "v0"
) -> Curve:
    """The sky's colour at every view elevation for one sun elevation, exposed for L* = 70.

    `rel_azimuth_deg` is the view's azimuth minus the sun's: 0 (the default, and what the sealed
    forecast uses) is the solar vertical, and plus and minus the same angle are the same sky.
    `engine` is "v0" (the default) or "v1", as for `sky_xyz`.
    """
    xyz = sky_xyz(atm, sun_elevation, VIEW_ELEVATIONS, rel_azimuth_deg, engine=engine)
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


def render_curve(atm: Atmosphere, *, engine: str = "v0") -> Curve:
    """One row per (sun elevation, view elevation): sun elevation descending, view ascending.

    `engine` is "v0" (the default, what the daily forecast paints) or "v1".
    """
    check_engine(engine)
    if engine == "v1":
        v1.check_domain(atm, max(SUN_ELEVATIONS), VIEW_ELEVATIONS)
    return [row for sun in SUN_ELEVATIONS for row in render_row(atm, sun, engine=engine)]


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


def strip_png_bytes(image: Image.Image) -> bytes:
    """The strip encoded as a PNG with no ancillary chunks.

    The pixels are copied into a fresh image first, so nothing the caller attached to
    `image.info` (EXIF, ICC profile, dpi, text) can reach the bytes.
    """
    clean = Image.frombytes(image.mode, image.size, image.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def save_strip(image: Image.Image, path: str | Path) -> None:
    """Write the strip as a PNG with no ancillary chunks (see `strip_png_bytes`)."""
    Path(path).write_bytes(strip_png_bytes(image))
