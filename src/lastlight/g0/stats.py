"""The G0 statistic and rule, exactly as pre-registered in `docs/gates/G0/criterion.md`.

Contrasts come as an array `contrast[date, elevation, band, channel]`: 29 sweep elevations
(`geometry.ELEVATION_SWEEP`), WHTM1's 5 contrast bands, and the 2 channels log(R/G) and log(B/G).
`band_distances` turns it into D(d, s) for the four twilight bands, `bands_at_threshold` counts
the counted bands (golden, early, mid; never late) in which a date reaches 0.05, and `verdict`
applies the 25 percent rule to the dates used.

Which sweep elevations are inside a band is fixed in the criterion file: a band `(upper, lower)`
holds the elevations `e` with `lower < e <= upper`, and the last band also holds its lower end
(-8.0), so every sweep elevation is in exactly one band: 12, 4, 6 and 7 of them.

Also here, reported but never part of the verdict: the windowed median that controls for the
season, and delta-E 2000 of the absolute curve for tier B.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from lastlight.geometry import ELEVATION_BANDS, ELEVATION_SWEEP
from lastlight.render.colorimetry import delta_e_2000

CRITERION_SENTENCE: Final = (
    "For each date d and each counted sun-elevation band s, let D(d, s) be the mean, over the "
    "sweep elevations inside s, over WHTM1's contrast bands k, and over both channels c in "
    "{log(R/G), log(B/G)}, of |C_d(s, k, c) - median over dates of C(s, k, c)|. Here C is the "
    "rendered within-frame contrast in linear RGB: band k's median minus the reference band's, "
    "computed for WHTM1's frozen geometry. Bands are mapped from rows to view elevation with the "
    "frozen fit's nominal pixel scale, and taken over the fit's sky columns. "
    "**G0 passes if D(d, s) >= 0.05 in at least 2 of the 3 counted bands on at least 25 percent "
    "of dates.**"
)
"""The pre-registered sentence, word for word; a test finds it in the committed criterion file."""

BANDS: Final = tuple(ELEVATION_BANDS)
"""golden, early, mid, late: the order of the columns of `band_distances`."""

COUNTED_BANDS: Final = ("golden", "early", "mid")
THRESHOLD: Final = 0.05
MIN_BANDS: Final = 2
MIN_FRACTION: Final = 0.25

DELTA_E_BANDS: Final = ("early", "mid", "late")
HALF_WINDOW_DAYS: Final = 15
"""The seasonal control takes the median over the dates within this many days of each date."""

_COUNTED_COLUMNS: Final = tuple(BANDS.index(name) for name in COUNTED_BANDS)


def band_members(name: str) -> tuple[int, ...]:
    """Indices into `ELEVATION_SWEEP` of the sweep elevations inside the band `name`."""
    if name not in ELEVATION_BANDS:
        raise ValueError(f"unknown band {name!r}; the bands are {list(ELEVATION_BANDS)}")
    upper, lower = ELEVATION_BANDS[name]
    closed_below = name == BANDS[-1]
    return tuple(
        index
        for index, elevation in enumerate(ELEVATION_SWEEP)
        if elevation <= upper and (elevation >= lower if closed_below else elevation > lower)
    )


# --- D(d, s) ------------------------------------------------------------------------------------


def period_median(contrast: np.ndarray) -> np.ndarray:
    """The median over dates at each elevation, band and channel: shape (elevations, bands, 2)."""
    return np.median(np.asarray(contrast, dtype=float), axis=0)


def windowed_median(
    dates: Sequence[dt.date], contrast: np.ndarray, half_window_days: int = HALF_WINDOW_DAYS
) -> np.ndarray:
    """For each date, the median over the dates within `half_window_days` calendar days of it.

    The date itself is always in its window. Shape (dates, elevations, bands, 2). This is the
    seasonal control: the sun's azimuth and the ozone column change with the season, and a median
    over the whole period counts that as variation between dates.
    """
    array = np.asarray(contrast, dtype=float)
    ordinals = np.array([day.toordinal() for day in dates])
    out = np.empty_like(array)
    for index, ordinal in enumerate(ordinals):
        near = np.abs(ordinals - ordinal) <= half_window_days
        out[index] = np.median(array[near], axis=0)
    return out


def band_distances(contrast: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """D(d, s) for every date and band: shape (dates, 4), columns in `BANDS` order.

    The mean, over the sweep elevations inside the band, the contrast bands and both channels,
    of the absolute difference between the date's contrast and `baseline` (the period median,
    shape (elevations, bands, 2), or one baseline per date, shape of `contrast`).
    """
    difference = np.abs(np.asarray(contrast, dtype=float) - np.asarray(baseline, dtype=float))
    return np.stack(
        [difference[:, list(band_members(name))].mean(axis=(1, 2, 3)) for name in BANDS], axis=1
    )


# --- the rule -----------------------------------------------------------------------------------


def bands_at_threshold(distances: np.ndarray) -> np.ndarray:
    """How many of the counted bands reach the threshold, per date (late is never counted)."""
    return (np.asarray(distances)[:, list(_COUNTED_COLUMNS)] >= THRESHOLD).sum(axis=1)


def meets_rule(distances: np.ndarray) -> np.ndarray:
    """Per date: D reaches the threshold in at least `MIN_BANDS` of the counted bands."""
    return bands_at_threshold(distances) >= MIN_BANDS


@dataclass(frozen=True)
class Verdict:
    """The gate's answer over the dates used: how many meet the rule, and whether that is enough."""

    dates: int
    meeting: int

    @property
    def fraction(self) -> float:
        return self.meeting / self.dates

    @property
    def passed(self) -> bool:
        return self.fraction >= MIN_FRACTION


def verdict(distances: np.ndarray) -> Verdict:
    """G0 passes if at least 25 percent of the dates meet the rule."""
    dates = int(np.asarray(distances).shape[0])
    if dates == 0:
        raise ValueError("no dates to judge: a verdict needs at least one date used")
    return Verdict(dates=dates, meeting=int(meets_rule(distances).sum()))


# --- tier B -------------------------------------------------------------------------------------


def delta_e_by_band(curves: np.ndarray) -> np.ndarray:
    """Delta-E 2000 of each date's absolute curve against the period's median curve, by band.

    `curves[date, elevation, view, (L, a, b)]` is the forecast's own curve in CIELAB. The median
    is taken per component at each elevation and view; the result is, for each date, the mean
    delta-E over the sweep elevations inside each of `DELTA_E_BANDS` and all views.
    """
    array = np.asarray(curves, dtype=float)
    median = np.median(array, axis=0)
    de = delta_e_2000(array, median[None, ...])  # shape (dates, elevations, views)
    return np.stack(
        [de[:, list(band_members(name))].mean(axis=(1, 2)) for name in DELTA_E_BANDS], axis=1
    )
