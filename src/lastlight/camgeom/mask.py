"""The sky mask: which pixels of a frame are sky that may be scored.

A pixel is sky when it lies above the fitted skyline by more than a margin and outside the
burned-in timestamp. The skyline is the fitted curve stored in the fit file, not a skyline
traced in some frame, so clouds and haze cannot move it. The margin keeps the soft edge of the
ridge, where terrain and sky blend, out of every colour statistic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from lastlight.camgeom.fitfile import CameraFit


def sky_mask(fit: CameraFit, margin_deg: float = 1.0) -> np.ndarray:
    """Boolean rows-by-columns array, True where the pixel is sky.

    The margin is converted to rows with the pixel scale at the image centre (about one row
    per 0.09 degrees for a 66 degree lens), which is within a few percent of the true scale
    across a frame that is under 30 degrees tall. A column where the skyline was not seen is
    never sky.
    """
    if margin_deg < 0.0:
        raise ValueError("the margin must not be negative")
    focal = (fit.width / 2.0) / math.tan(math.radians(fit.orientation.hfov_deg) / 2.0)
    pixels_per_degree = focal * math.pi / 180.0
    skyline = fit.skyline_y(np.arange(fit.width))
    limit = skyline - margin_deg * pixels_per_degree
    with np.errstate(invalid="ignore"):
        mask = np.arange(fit.height, dtype=float)[:, None] < limit[None, :]
    for box in fit.overlay_boxes:
        mask[box.y0 : box.y1, box.x0 : box.x1] = False
    return mask


def write_mask_png(mask: np.ndarray, path: Path) -> Path:
    """Write the mask as a PNG of 0 and 255 with no metadata chunks at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, format="PNG")
    return path
