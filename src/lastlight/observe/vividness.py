"""The v0 observed vividness: a named placeholder scalar over an evening's profile series.

Vividness is the one pre-registered scalar per evening: how much saturated warm colour the
western sky shows during twilight (`docs/design.md` section 1). Its definition is frozen in
`docs/gates/vividness.md` before the first scored evening; that file does not exist yet. Until
it does, the observed side has this stand-in, named as one everywhere it appears. It is the
observed counterpart of `lastlight.vividness.vividness_v0`, which is the forecast's placeholder
and has a different job and signature.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from lastlight.observe.contrast import ContrastProfile

VIVIDNESS_V0_RULE: Final = "v0-placeholder"
"""The rule's name, to be recorded wherever its number is."""


def vividness_v0(profile_series: Iterable[ContrastProfile]) -> float | None:
    """The v0 vividness placeholder: the largest `log(R/G)` contrast of the lowest band.

    Over the frames of an evening (the series `lastlight observe` keeps, sun elevation +6 to -6
    by default), take each frame's lowest band, the one nearest the ridge, where twilight's warm
    colour sits, and return the maximum of its `log_rg` contrast against the reference band.
    A frame whose lowest band is missing is skipped, not counted as zero; with no usable frame
    the value is None. The result is a within-frame log-chromaticity difference, so it is
    unchanged by the camera's exposure and white balance, and it may be negative.

    This is a PLACEHOLDER, not the pre-registered definition (`docs/gates/vividness.md`, due
    before the first scored evening). Nothing may be scored against it.
    """
    values = [
        profile.bands[0].log_rg
        for profile in profile_series
        if profile.bands and profile.bands[0].log_rg is not None
    ]
    return max(values) if values else None
