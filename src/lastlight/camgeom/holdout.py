"""The held-out check: does a frozen skyline match frames that were not used to fit it?

A fit is frozen on a few frames, so a low residual on those frames alone could mean the fit
has memorised them. The check lays the frozen skyline (the curve stored in the fit file) over
other frames of the same camera, traces the skyline in each (`lastlight.camgeom.skyline`) in
a band around the frozen curve, and reports the RMS distance between the two in pixels and in
degrees at the fitted scale.

The band is a setting of the measurement, not a detail: it says how far from the frozen curve
a frame's own skyline may be and still be found. Narrow, it measures how closely the curve
follows the ridge, and a frame whose strongest edge is something else (a cloud base) is not
followed there. Wide, one such edge can take over a whole frame. The default is the width of
the fit's own last pass, so a held-out frame is judged the way a fitted frame is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lastlight.camgeom.fitfile import CameraFit
from lastlight.camgeom.frames import discover_frames, frame_sha256, load_rgb
from lastlight.camgeom.skyline import edge_strength, trace_skyline

DEFAULT_BAND_ROWS = 12.0
LABEL_LENGTH = 8


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """How well the frozen skyline matches one frame."""

    label: str  # the first characters of the frame's SHA-256: never its name or its time
    rms_px: float
    rms_deg: float
    median_abs_px: float
    n_columns: int


def frame_label(sha256: str) -> str:
    return sha256[:LABEL_LENGTH]


def held_out_paths(fit: CameraFit, roots: Iterable[Path]) -> tuple[list[Path], int]:
    """Frames under `roots` that the fit was not frozen on, and how many were left out.

    A frame was used when its SHA-256 is one of the fit's `frames_used`. A frame found under
    two roots counts once.
    """
    used = {record.sha256 for record in fit.frames_used}
    seen: set[str] = set()
    kept: list[Path] = []
    skipped = 0
    for root in roots:
        for frame in discover_frames(root):
            digest = frame_sha256(frame.path)
            if digest in seen:
                continue
            seen.add(digest)
            if digest in used:
                skipped += 1
            else:
                kept.append(frame.path)
    return kept, skipped


def holdout_rms(
    fit: CameraFit,
    image: np.ndarray,
    label: str,
    *,
    band_rows: float = DEFAULT_BAND_ROWS,
) -> HoldoutResult:
    """RMS vertical distance between the frozen skyline and the skyline traced in `image`."""
    if band_rows <= 0 or not math.isfinite(band_rows):
        raise ValueError("the band must be a positive number of rows")
    height, width = image.shape[:2]
    if (width, height) != (fit.width, fit.height):
        raise ValueError(
            f"a frame is {width}x{height} but the fit is for {fit.width}x{fit.height}: "
            "the frozen skyline cannot be laid over it"
        )
    frozen = fit.skyline_y(np.arange(width))
    traced = trace_skyline(
        edge_strength(image, fit.overlay_boxes), around=frozen, half_width=band_rows
    )
    difference = (traced - frozen)[np.isfinite(traced - frozen)]
    if difference.size == 0:
        raise ValueError("the frozen skyline and the traced one share no column")
    rms = float(np.sqrt(np.mean(difference**2)))
    o = fit.orientation
    focal = (width / 2.0) / math.tan(math.radians(o.hfov_deg) / 2.0)
    return HoldoutResult(
        label=label,
        rms_px=rms,
        rms_deg=rms / (focal * math.pi / 180.0),
        median_abs_px=float(np.median(np.abs(difference))),
        n_columns=int(difference.size),
    )


def holdout_paths(
    fit: CameraFit, paths: Iterable[Path], *, band_rows: float = DEFAULT_BAND_ROWS
) -> list[HoldoutResult]:
    """`holdout_rms` for each frame on disk, labelled by the start of its SHA-256."""
    return [
        holdout_rms(
            fit,
            load_rgb(path),
            frame_label(frame_sha256(path)),
            band_rows=band_rows,
        )
        for path in paths
    ]
