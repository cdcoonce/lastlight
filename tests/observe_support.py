"""Synthetic frames and fits for the observe tests; none is, or comes from, a real frame.

A frame is built in LINEAR light and then encoded with the sRGB transfer function, so a test can
apply a white-balance or exposure change the way a camera does (a gain per channel on the linear
signal) and re-encode. `encode` keeps the result as floats on the 0 to 255 scale, unrounded, so the
invariance test is exact to floating point; `to_uint8` is the 8-bit path a decoded JPEG takes.
"""

from __future__ import annotations

import numpy as np
from camgeom_synthetic import HEIGHT, WIDTH, example_fit

from lastlight.cameras import ContrastBands

COLUMNS = tuple(float(x) for x in range(WIDTH))

# WHTM1's seeded table, without its light dome (tests that need a dome add their own).
BANDS = ContrastBands(
    skyline_margin_rows=8,
    bands_rows=((8, 28), (28, 48), (48, 68), (68, 88), (88, 108)),
    reference_rows=(118, 138),
    light_dome_columns=(),
)
BAND_CENTRES = tuple((lo + hi - 1) / 2.0 for lo, hi in BANDS.bands_rows)  # offsets above skyline
REFERENCE_CENTRE = (BANDS.reference_rows[0] + BANDS.reference_rows[1] - 1) / 2.0

# The colour of the sky at `d` rows above the skyline, in linear light: warm at the horizon and
# bluer overhead. log(R/G) falls by 0.002 per row and log(B/G) rises by 0.0015 per row.
RG_SLOPE = -0.002
BG_SLOPE = 0.0015
RG_AT_SKYLINE = 0.30
BG_AT_SKYLINE = -0.10
BRIGHTNESS = 0.25
TERRAIN = (0.05, 0.03, 0.02)  # linear light


def flat_fit(top: float = 200.0, **changes):
    """A fit whose skyline sits at row `top` in every one of the 800 columns, with no overlay.

    `changes` replace any field of the fit, `skyline_y_px` and `overlay_boxes` included.
    """
    fields = {
        "skyline_x_px": COLUMNS,
        "skyline_y_px": tuple(top for _ in COLUMNS),
        "overlay_boxes": (),
    }
    return example_fit(**{**fields, **changes})


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """The sRGB transfer function: linear light in 0 to 1 to the encoded 0 to 255 scale (float)."""
    linear = np.asarray(linear, dtype=float)
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0), 1 / 2.4) - 0.055,
    )
    return 255.0 * encoded


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """The inverse of `srgb_encode`: the encoded 0 to 255 scale to linear light in 0 to 1."""
    e = np.asarray(encoded, dtype=float) / 255.0
    return np.where(e <= 0.04045, e / 12.92, np.power((e + 0.055) / 1.055, 2.4))


def to_uint8(encoded: np.ndarray) -> np.ndarray:
    """What an 8-bit decoder would hand over: the encoded values rounded to whole codes."""
    return np.clip(np.rint(encoded), 0, 255).astype(np.uint8)


def offsets_above_skyline(top_rows: np.ndarray) -> np.ndarray:
    """Rows-by-columns array of `d = top - row` for whole skyline rows `top_rows` per column."""
    return top_rows[None, :] - np.arange(HEIGHT)[:, None]


def gradient_linear(d: np.ndarray) -> np.ndarray:
    """Linear RGB at offset `d`: the horizon-to-zenith gradient described at the top.

    Below the skyline (`d < 0`) it is dark brown terrain instead, so a band read from the wrong
    rows finds ground, not a shifted copy of the ramp (a linear ramp alone cannot tell).
    """
    green = np.full(d.shape, BRIGHTNESS)
    red = green * np.exp(RG_AT_SKYLINE + RG_SLOPE * d)
    blue = green * np.exp(BG_AT_SKYLINE + BG_SLOPE * d)
    sky = np.stack([red, green, blue], axis=-1)
    return np.where(d[..., None] >= 0, sky, np.array(TERRAIN))


def gradient_frame(top: float = 200.0, gains=(1.0, 1.0, 1.0)) -> np.ndarray:
    """The gradient over a flat skyline at row `top`, with a gain per channel in linear light,
    encoded to floats on the 0 to 255 scale."""
    d = offsets_above_skyline(np.full(WIDTH, float(top)))
    return srgb_encode(gradient_linear(d) * np.asarray(gains, dtype=float))


def grey_frame(level: float = 120.0) -> np.ndarray:
    """A frame of one neutral grey (encoded code `level`): every contrast is zero."""
    return np.full((HEIGHT, WIDTH, 3), level, dtype=float)


def paint_band(
    frame: np.ndarray,
    rows: tuple[int, int],
    colour,
    *,
    top: int = 200,
    columns: slice = slice(None),
) -> np.ndarray:
    """Set the pixels of the band `rows` (offsets above a flat skyline at `top`) to `colour`."""
    low, high = rows
    out = frame.copy()
    out[top - high + 1 : top - low + 1, columns] = colour
    return out
