"""Observed truth, tier A: frames to within-frame colour contrasts (`docs/design.md` section 1).

- `contrast`: one frame's contrast profile over pixel-row bands above the frozen skyline.
- `bands`: the check that a camera's bands lie inside its frozen fit's sky mask.
- `vividness`: the observed v0 placeholder scalar over an evening's profiles
  (`observed_vividness_v0`, not the forecast's `vividness_v0`).
- `evening`: every saved frame of an evening to the `observed.json` document.
"""

from lastlight.observe.contrast import ContrastProfile, contrast_profile
from lastlight.observe.vividness import OBSERVED_VIVIDNESS_V0_RULE, observed_vividness_v0

__all__ = [
    "OBSERVED_VIVIDNESS_V0_RULE",
    "ContrastProfile",
    "contrast_profile",
    "observed_vividness_v0",
]
