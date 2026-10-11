"""The v0 vividness rule: a documented placeholder with a deliberately wide spread.

Vividness is the one pre-registered scalar per evening: how much saturated warm colour the
western sky shows during twilight (`docs/design.md` section 1). Its definition is frozen in
`docs/gates/vividness.md` before the first scored evening, and the physics that would predict it
(cloud lit from below, Bruneton scattering) arrives in M4. Until then the daily forecast still
has to seal *a* distribution, and honesty rule 3 forbids pretending to know more than we do.

So the v0 rule is a placeholder, named as one everywhere it appears (`VIVIDNESS_RULE` goes into
the sealed notes), and its spread covers a quarter of the zero-to-one scale on each side:

    mean = 0.45 - 0.30 * (low cloud cover, 0 to 1) + 0.25 * min(AOD / 0.4, 1)

A low cloud deck hides the lit horizon (the first observed evening, 2026-10-10, was a stratiform
overcast and a dull one); moderate aerosol reddens the twilight, and the term saturates at an
optical depth of 0.4. Both weights are guesses, not fits. The sd is 0.25, so any evening's
outcome lies well inside two sd of any mean and the rule is scored as the weak prior it is.
"""

from __future__ import annotations

import math
from typing import Final

from lastlight.forecast import Vividness

VIVIDNESS_RULE: Final = "v0-placeholder"
"""The rule's name, as sealed in the forecast notes."""

VIVIDNESS_SD: Final = 0.25
"""Standard deviation of every v0 vividness forecast: wide on purpose."""

_BASE: Final = 0.45
_CLOUD_WEIGHT: Final = 0.30
_AEROSOL_WEIGHT: Final = 0.25
_AEROSOL_CAP_AOD: Final = 0.4

RULE_SUMMARY: Final = (
    f"Vividness rule {VIVIDNESS_RULE}: mean 0.45 minus 0.30 times low cloud cover (0 to 1) plus "
    f"0.25 times aerosol optical depth over 0.4 (at most 1), sd {VIVIDNESS_SD}, deliberately wide; "
    "a placeholder, not the pre-registered definition."
)
"""One sentence for the forecast notes, so the sealed document says which rule made its number."""


def vividness_v0(low_cloud_pct: float, aod_550: float) -> Vividness:
    """The v0 vividness forecast from low cloud cover (percent) and aerosol optical depth."""
    if not (math.isfinite(low_cloud_pct) and 0.0 <= low_cloud_pct <= 100.0):
        raise ValueError(f"low_cloud_pct must be a percentage from 0 to 100, got {low_cloud_pct!r}")
    if not (math.isfinite(aod_550) and aod_550 >= 0.0):
        raise ValueError(f"aod_550 must be a finite number, not negative, got {aod_550!r}")
    mean = (
        _BASE
        - _CLOUD_WEIGHT * (low_cloud_pct / 100.0)
        + _AEROSOL_WEIGHT * min(aod_550 / _AEROSOL_CAP_AOD, 1.0)
    )
    return Vividness(mean=min(max(mean, 0.0), 1.0), sd=VIVIDNESS_SD)
