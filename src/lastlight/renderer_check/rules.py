"""The numbers `docs/gates/renderer-v1/criterion.md` fixes, on plain values.

The rule: renderer v1 is adopted if and only if, over the first 5 counted evenings, the median of
the per-evening ratio r_e lies between 0.5 and 2.0 inclusive. With fewer than 5 counted evenings
the verdict is "not yet decidable". For one evening,

    r_e = median over its used frames of |rendered contrast|
          / median over the same frames of |observed contrast|

(a ratio of two medians, never a median of per-frame ratios), and +infinity when the denominator
is 0, an evening that still takes part in the median. An evening is counted when the mean HRRR
total cloud cover along the sightline is at most 10 percent (exactly 10 counts).
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections.abc import Sequence
from typing import Final

CAMERA_ID: Final = "WHTM1"
FIRST_EVENING: Final = dt.date(2026, 10, 10)
"""The first evening the criterion considers."""
MIN_USED_FRAMES: Final = 3
CLOUD_LIMIT_PCT: Final = 10.0
COUNTED_NEEDED: Final = 5
RATIO_LOW: Final = 0.5
RATIO_HIGH: Final = 2.0

ADOPT: Final = "ADOPT"
KEEP_V0: Final = "KEEP V0"


def not_yet_decidable(counted: int) -> str:
    """The verdict while fewer than 5 evenings are counted."""
    return f"NOT YET DECIDABLE ({counted} of {COUNTED_NEEDED} counted evenings)"


def cloud_mean(values: Sequence[float]) -> float:
    """The arithmetic mean of total cloud cover (percent) over the transect points, exactly."""
    if not values:
        raise ValueError("a cloud mean needs at least one value")
    return math.fsum(values) / len(values)


def is_clear(mean_cloud_pct: float) -> bool:
    """True when the evening is clear enough to count: at most `CLOUD_LIMIT_PCT`, inclusive."""
    return mean_cloud_pct <= CLOUD_LIMIT_PCT


def _median_absolute(values: Sequence[float]) -> float:
    return statistics.median(abs(value) for value in values)


def evening_ratio(rendered: Sequence[float], observed: Sequence[float]) -> float:
    """r_e from the used frames' rendered and observed contrasts, frame by frame.

    A zero median of the absolute observed values gives `math.inf`, whatever the numerator.
    """
    if len(rendered) != len(observed):
        raise ValueError(
            f"{len(rendered)} rendered contrasts but {len(observed)} observed: one per frame"
        )
    if not rendered:
        raise ValueError("an evening ratio needs at least one frame")
    if not all(math.isfinite(value) for value in (*rendered, *observed)):
        raise ValueError("contrasts must be finite numbers")
    denominator = _median_absolute(observed)
    if denominator == 0.0:
        return math.inf
    return _median_absolute(rendered) / denominator


def decide(ratios_in_date_order: Sequence[float]) -> str:
    """The verdict from the v1 ratios of the counted evenings, earliest first.

    Only the first `COUNTED_NEEDED` count, never the latest: the verdict is computed once, on
    them, and later evenings are reported beside it without changing it.
    """
    first = list(ratios_in_date_order)[:COUNTED_NEEDED]
    if len(first) < COUNTED_NEEDED:
        return not_yet_decidable(len(first))
    median = statistics.median(first)
    return ADOPT if RATIO_LOW <= median <= RATIO_HIGH else KEEP_V0
