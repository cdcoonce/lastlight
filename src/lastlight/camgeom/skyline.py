"""Finding the skyline in a frame: the strongest sky-to-terrain edge in every column.

Sky is brighter than terrain at the horizon (by day, and by the glow of twilight), so the
skyline is where brightness falls going down the image. `edge_strength` measures that drop and
nothing else, so the bottom of a cloud (dark above, bright below) and the top of a white
overlay never count. `trace_skyline` then picks one row per column with a dynamic program that
keeps the path continuous, which a per-column arg-max does not: a bright roof or a cloud edge
would otherwise pull single columns off the ridge.

Two inputs steer the trace. `exclude` boxes (the burned-in timestamp) contribute no edges at
all. `around` and `half_width` confine each column to a band around an expected skyline, which
is how the fit ignores cloud edges far from the horizon once it has a first estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_RGB_TO_LUMINANCE = np.array([0.299, 0.587, 0.114])
_MAX_STEP = 2  # rows the path may move between neighbouring columns
_SMOOTHNESS = 0.03  # cost of one row of movement, in units of the strong-edge level
_GUARD_ROWS = 3  # the edge filter reaches two rows either side


@dataclass(frozen=True, slots=True)
class Box:
    """A rectangle of pixels, half-open: columns x0 to x1 - 1 and rows y0 to y1 - 1."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"a box needs 0 <= x0 < x1 and 0 <= y0 < y1, got {self}")


def default_overlay_boxes(width: int, height: int) -> tuple[Box, ...]:
    """Where the burned-in timestamp sits in the PhoenixVis frames: the bottom-right corner.

    Given as fractions of the frame so it scales with the image. At 800 by 300 it is columns
    592 and up, rows 264 and down, which holds the stamp text (columns 610 to 795, rows 277
    to 297) with a margin.
    """
    return (Box(int(0.74 * width), int(0.88 * height), width, height),)


def _luminance(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image, dtype=float)
    if pixels.ndim == 3:
        return pixels[..., :3] @ _RGB_TO_LUMINANCE
    if pixels.ndim == 2:
        return pixels
    raise ValueError(f"an image is rows by columns, or rows by columns by 3; got {pixels.shape}")


def edge_strength(
    image: np.ndarray, exclude: Sequence[Box] = (), *, sigma: float = 1.2
) -> np.ndarray:
    """How much brighter the pixels just above are than the pixels just below, never negative.

    Brightness is smoothed (`sigma` pixels) and compared two rows up with two rows down. Rows
    inside an `exclude` box score zero, and so do the three rows at the top and bottom edge.
    Each box is widened by the reach of the smoothing and the comparison, because they spread
    the edge of a bright stamp a few pixels beyond its rectangle.
    """
    smooth = ndimage.gaussian_filter(_luminance(image), sigma)
    drop = np.zeros_like(smooth)
    drop[2:-2] = (smooth[:-4] - smooth[4:]) / 4.0
    drop[:_GUARD_ROWS] = 0.0
    drop[-_GUARD_ROWS:] = 0.0
    strength = np.maximum(drop, 0.0)
    reach = int(np.ceil(3.0 * sigma)) + 2
    for box in exclude:
        strength[
            max(box.y0 - reach, 0) : box.y1 + reach, max(box.x0 - reach, 0) : box.x1 + reach
        ] = 0.0
    return strength


def _allowed_rows(
    height: int, width: int, around: np.ndarray | None, half_width: float
) -> np.ndarray:
    allowed = np.zeros((width, height), dtype=bool)
    allowed[:, _GUARD_ROWS : height - _GUARD_ROWS] = True
    if around is not None:
        rows = np.arange(height)[None, :]
        centre = np.asarray(around, dtype=float)[:, None]
        banded = np.abs(rows - centre) <= half_width
        known = np.isfinite(centre)  # a column with no expectation is searched in full
        allowed &= banded | ~known
    return allowed


def trace_skyline(
    strength: np.ndarray, *, around: np.ndarray | None = None, half_width: float = 30.0
) -> np.ndarray:
    """The row of the skyline in each column, to a fraction of a pixel, from an edge map.

    The path maximises the summed edge strength with a small cost for moving between rows, and
    may move at most two rows from one column to the next. The integer path is refined with a
    parabola through the three edge values around each row. A column whose band holds no row of
    the frame is NaN.
    """
    height, width = strength.shape
    allowed = _allowed_rows(height, width, around, half_width)
    # A band that lies wholly outside the frame (the expected skyline is below the last row, or
    # above the first) leaves no row to trace. The path passes through such a column at no cost
    # and the column is reported as NaN, so it neither crashes the path nor votes.
    unusable = ~allowed.any(axis=1)
    allowed[unusable, _GUARD_ROWS : height - _GUARD_ROWS] = True
    scale = np.percentile(strength[strength > 0], 99) if np.any(strength > 0) else 1.0
    penalty = _SMOOTHNESS * scale
    gain = np.where(allowed, -strength.T, np.inf)  # columns by rows, lower is better
    gain[unusable] = np.where(allowed[unusable], 0.0, np.inf)
    cost = np.empty((width, height))
    back = np.zeros((width, height), dtype=np.int64)
    cost[0] = gain[0]
    rows = np.arange(height)
    for column in range(1, width):
        previous = cost[column - 1]
        candidates = np.full((2 * _MAX_STEP + 1, height), np.inf)
        for index, step in enumerate(range(-_MAX_STEP, _MAX_STEP + 1)):
            if step > 0:
                candidates[index, step:] = previous[:-step]
            elif step < 0:
                candidates[index, :step] = previous[-step:]
            else:
                candidates[index] = previous
            candidates[index] += penalty * abs(step)
        best = candidates.argmin(axis=0)
        cost[column] = candidates.min(axis=0) + gain[column]
        back[column] = rows - (best - _MAX_STEP)
    path = np.empty(width, dtype=np.int64)
    path[-1] = int(np.argmin(cost[-1]))
    for column in range(width - 1, 0, -1):
        path[column - 1] = back[column, path[column]]
    refined = _refine(strength, path)
    refined[unusable] = np.nan
    return refined


def _refine(strength: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Move each integer row to the vertex of a parabola through its neighbours' edge values."""
    height, width = strength.shape
    columns = np.arange(width)
    above = strength[np.clip(path - 1, 0, height - 1), columns]
    centre = strength[path, columns]
    below = strength[np.clip(path + 1, 0, height - 1), columns]
    curvature = above - 2.0 * centre + below
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(curvature < 0.0, 0.5 * (above - below) / curvature, 0.0)
    return path + np.clip(offset, -0.5, 0.5)


def extract_skyline(
    image: np.ndarray,
    *,
    exclude: Sequence[Box] = (),
    around: np.ndarray | None = None,
    half_width: float = 30.0,
) -> np.ndarray:
    """Row of the sky-to-terrain edge in every column of an RGB or grayscale frame."""
    return trace_skyline(edge_strength(image, exclude), around=around, half_width=half_width)


def combine_skylines(skylines: Sequence[np.ndarray]) -> np.ndarray:
    """Per-column median over several frames' skylines, ignoring missing columns.

    Terrain is the same in every frame and clouds are not, so the median outvotes a frame in
    which a cloud edge took over some columns.
    """
    stacked = np.vstack([np.asarray(skyline, dtype=float) for skyline in skylines])
    with np.errstate(all="ignore"):
        return np.nanmedian(stacked, axis=0)
