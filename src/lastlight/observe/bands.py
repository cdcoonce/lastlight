"""Do a camera's contrast bands lie inside its sky? The check against the frozen fit.

A camera's bands are pixel rows above its skyline (`lastlight.cameras.ContrastBands`). Whether
those rows are in the frame, outside the timestamp and above the ridge is a fact about the
camera's frozen fit, so it is checked here, against the fit's own sky mask
(`lastlight.camgeom.mask.sky_mask_rows`). The rule: every band, and the reference, must lie
inside the mask in at least 95 percent of columns. A column counts as inside only when every
one of the band's rows is inside the frame and the mask there; a column that is not inside is
left out of that band (and counted), never filled.

`validate_contrast` is run by `lastlight observe` before it reads a frame, and by the tests
against every shipped table, so a refit that moves the skyline cannot quietly leave a band
hanging off the top of the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from lastlight.camgeom.fitfile import CameraFit
from lastlight.camgeom.mask import sky_mask_rows, skyline_top_rows
from lastlight.cameras.registry import ContrastBands

MIN_COLUMN_COVERAGE: Final = 0.95
"""The fraction of columns in which each band and the reference must lie inside the sky mask."""


class ContrastConfigError(ValueError):
    """A camera's contrast table does not fit its frozen geometry. The message says how."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """How many columns a band lies wholly inside the sky mask in: `fraction` of the width."""

    rows: tuple[int, int]
    fraction: float
    is_reference: bool


def columns_inside(fit: CameraFit, rows: tuple[int, int], *, margin_rows: int) -> np.ndarray:
    """Boolean per column: True where the rows `[lo, hi)` above the skyline are all in the mask.

    The rows of column `x` are `top(x) - d` for `d` from `lo` to `hi - 1`, where `top` is the
    skyline's whole row (`skyline_top_rows`). Off the frame, in an overlay box, closer than
    `margin_rows` to the skyline, or where the skyline was not seen: not inside.
    """
    low, high = rows
    mask = sky_mask_rows(fit, margin_rows)
    top = skyline_top_rows(fit)
    seen = ~np.isnan(top)
    top_int = np.where(seen, top, 0.0).astype(int)
    row_index = top_int[None, :] - np.arange(low, high)[:, None]
    in_frame = ((row_index >= 0) & (row_index < fit.height)).all(axis=0)
    clipped = np.clip(row_index, 0, fit.height - 1)
    in_mask = mask[clipped, np.arange(fit.width)[None, :]].all(axis=0)
    return seen & in_frame & in_mask


def band_coverage(fit: CameraFit, contrast: ContrastBands) -> tuple[Coverage, ...]:
    """Coverage of each band, lowest first, then of the reference."""
    margin = contrast.skyline_margin_rows
    entries = [(rows, False) for rows in contrast.bands_rows]
    entries.append((contrast.reference_rows, True))
    return tuple(
        Coverage(
            rows=rows,
            fraction=int(columns_inside(fit, rows, margin_rows=margin).sum()) / fit.width,
            is_reference=is_reference,
        )
        for rows, is_reference in entries
    )


def validate_contrast(
    fit: CameraFit, contrast: ContrastBands, *, minimum: float = MIN_COLUMN_COVERAGE
) -> tuple[Coverage, ...]:
    """Coverage of every band, or a `ContrastConfigError` naming each one that falls short.

    Also refuses light-dome column ranges that run past the frame's width.
    """
    coverage = band_coverage(fit, contrast)
    problems = []
    for entry in coverage:
        if entry.fraction < minimum:
            what = "reference" if entry.is_reference else "band"
            hint = " (move it down in 10-row steps)" if entry.is_reference else ""
            problems.append(
                f"{what} {entry.rows} lies inside the sky mask in {entry.fraction:.1%} of "
                f"columns, below the required {minimum:.1%}{hint}"
            )
    for low, high in contrast.light_dome_columns:
        if high > fit.width:
            problems.append(
                f"light_dome_columns {(low, high)} run past the frame's {fit.width} columns"
            )
    if problems:
        raise ContrastConfigError(
            f"the contrast table does not fit the frozen fit of {fit.camera_id}: "
            + "; ".join(problems)
        )
    return coverage
