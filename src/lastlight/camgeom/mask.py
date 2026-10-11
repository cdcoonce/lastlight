"""The sky mask: which pixels of a frame are sky that may be scored.

A pixel is sky when it lies above the fitted skyline by more than a margin and outside the
burned-in timestamp. The skyline is the fitted curve stored in the fit file, not a skyline
traced in some frame, so clouds and haze cannot move it. The margin keeps the soft edge of the
ridge, where terrain and sky blend, out of every colour statistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lastlight.camgeom.fitfile import CameraFit
from lastlight.camgeom.model import pixels_per_degree


def sky_mask(fit: CameraFit, margin_deg: float = 1.0) -> np.ndarray:
    """Boolean rows-by-columns array, True where the pixel is sky.

    The mask is cut in pixel space from the stored skyline, so it is as well pinned as that
    skyline (a pixel or two). Only the margin is converted: it becomes rows with the pixel scale
    at the image centre, which is within a few percent of the true scale across a frame that is
    under 30 degrees tall. That scale is a property of the fit, about 41 pixels per degree (a row
    is 0.024 degrees) for a 19 degree lens and 11 pixels per degree (0.09 degrees a row) for a
    66 degree lens.

    The mapping is provisional while `fit.provisional` is true: the field of view, and with it
    the number of rows a margin in degrees comes to, is known only to the ranges in
    `fit.uncertainty` until the sun has confirmed it. A column where the skyline was not seen is
    never sky.
    """
    if margin_deg < 0.0:
        raise ValueError("the margin must not be negative")
    skyline = fit.skyline_y(np.arange(fit.width))
    limit = skyline - margin_deg * pixels_per_degree(fit.orientation.hfov_deg, fit.width)
    with np.errstate(invalid="ignore"):
        mask = np.arange(fit.height, dtype=float)[:, None] < limit[None, :]
    for box in fit.overlay_boxes:
        mask[box.y0 : box.y1, box.x0 : box.x1] = False
    return mask


def skyline_top_rows(fit: CameraFit) -> np.ndarray:
    """The fitted skyline's row in each column, rounded half up to a whole row (floats).

    NaN where the skyline was not seen. Pixel-row bands (`lastlight.observe`) count rows upward
    from this row, so the rounding is part of their definition: it makes a band's rows the same
    whole rows in every frame and under every refit that leaves the skyline in place.
    """
    return np.floor(fit.skyline_y(np.arange(fit.width)) + 0.5)


def sky_mask_rows(fit: CameraFit, margin_rows: int) -> np.ndarray:
    """Boolean rows-by-columns array, True where the pixel is sky, with the margin in pixel rows.

    The row-band counterpart of `sky_mask`: nothing here is converted from degrees, so it does
    not depend on the fit's pixel scale and is the same whether the fit is provisional or not.
    A pixel `d` rows above the skyline (`d = skyline_top_rows - row`) is sky when `d` is at least
    `margin_rows`; so a band `[lo, hi)` with `lo >= margin_rows` lies wholly in the mask. A column
    where the skyline was not seen is never sky, and neither is a pixel in an overlay box.
    """
    if margin_rows < 0:
        raise ValueError("the margin must not be negative")
    limit = skyline_top_rows(fit) - margin_rows
    with np.errstate(invalid="ignore"):
        mask = np.arange(fit.height, dtype=float)[:, None] <= limit[None, :]
    for box in fit.overlay_boxes:
        mask[box.y0 : box.y1, box.x0 : box.x1] = False
    return mask


def write_mask_png(mask: np.ndarray, path: Path) -> Path:
    """Write the mask as a PNG of 0 and 255 with no metadata chunks at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
    return path
