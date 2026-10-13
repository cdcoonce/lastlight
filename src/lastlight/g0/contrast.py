"""The rendered within-frame contrast: what `lastlight.observe` measures, from the sky model.

For each of WHTM1's bands and its reference the renderer gives CIE XYZ at the band's view elevation
(`lastlight.render.sky_xyz`, toward the camera's azimuth); XYZ becomes linear sRGB, the webcam
proxy (a camera's exposure and white balance act as one gain per channel on exactly that
signal); and the contrast of a band is its `log(R/G)` and `log(B/G)` minus the reference's. A gain
cancels, as it does for the camera.

Single scattering near the horizon is redder than the sRGB gamut, so a channel can come out
negative, and a logarithm needs a positive number. A channel is therefore raised to a floor of
`CHANNEL_FLOOR` times the largest channel of its pixel before the logarithm, and every raise is
counted. The floor is tied to the observed side: it is the widest ratio between two channels that
an 8-bit frame can show once `observe`'s own pixel rules are applied (code 6, the darkest it
keeps, against code 249, the brightest), so the rendered contrast is never asked to be more
extreme than a camera frame can report. The G0 report says how often the floor was used.
"""

from __future__ import annotations

import numpy as np

from lastlight.g0.view import SkyView
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.observe.contrast import DARK_LEVEL, SATURATION_LEVEL, srgb_to_linear
from lastlight.render import Atmosphere
from lastlight.render.colorimetry import xyz_to_linear_srgb
from lastlight.render.curve import sky_xyz

CHANNEL_FLOOR = float(
    srgb_to_linear(np.array(DARK_LEVEL + 1.0)) / srgb_to_linear(np.array(SATURATION_LEVEL - 1.0))
)
"""Smallest channel, as a fraction of the pixel's largest, that the logarithm is taken of."""


def log_chromaticity(xyz: np.ndarray) -> tuple[np.ndarray, int]:
    """`[log(R/G), log(B/G)]` of XYZ in linear sRGB, floored, and how many channels were raised.

    `xyz` has shape (pixels, 3); the result has shape (pixels, 2).
    """
    rgb = xyz_to_linear_srgb(np.atleast_2d(np.asarray(xyz, dtype=float)))
    peak = rgb.max(axis=1)
    if (peak <= 0.0).any():
        raise ValueError("a pixel with no light at all has no chromaticity")
    floor = (CHANNEL_FLOOR * peak)[:, None]
    raised = rgb < floor
    clipped = np.where(raised, floor, rgb)
    logs = np.log(clipped[:, [0, 2]] / clipped[:, [1]])
    return logs, int(raised.sum())


def rendered_contrasts(
    atm: Atmosphere,
    sun_elevation_deg: float,
    view: SkyView,
    rel_azimuth_deg: float,
    *,
    engine: str = "v0",
) -> tuple[np.ndarray, int]:
    """The contrast of each band at one sun elevation: shape (bands, 2), and channels floored.

    `rel_azimuth_deg` is the camera's azimuth minus the sun's at this sun elevation. `engine` is
    the renderer, "v0" (the default, what G0 was run with) or "v1".
    """
    xyz = sky_xyz(atm, sun_elevation_deg, view.elevations_deg, rel_azimuth_deg, engine=engine)
    logs, floored = log_chromaticity(xyz)
    return logs[:-1] - logs[-1], floored


def day_contrasts(
    atm: Atmosphere,
    view: SkyView,
    rel_azimuths_deg: tuple[float, ...],
    *,
    engine: str = "v0",
) -> tuple[np.ndarray, int]:
    """One evening's contrasts at every sweep elevation: shape (29, bands, 2), channels floored.

    `rel_azimuths_deg` holds the camera's azimuth minus the sun's at each sweep elevation.
    `engine` is as for `rendered_contrasts`.
    """
    if len(rel_azimuths_deg) != len(ELEVATION_SWEEP):
        raise ValueError(
            f"one relative azimuth per sweep elevation is needed: {len(ELEVATION_SWEEP)} (29) "
            f"expected, got {len(rel_azimuths_deg)}"
        )
    rows = []
    floored = 0
    for elevation, azimuth in zip(ELEVATION_SWEEP, rel_azimuths_deg, strict=True):
        contrast, count = rendered_contrasts(atm, elevation, view, azimuth, engine=engine)
        rows.append(contrast)
        floored += count
    return np.stack(rows), floored
