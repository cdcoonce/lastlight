"""Observed truth, tier A: frames to within-frame colour contrasts (`docs/design.md` section 1).

- `contrast`: one frame's contrast profile over pixel-row bands above the frozen skyline.
- `bands`: the check that a camera's bands lie inside its frozen fit's sky mask.
- `vividness`: the v0 placeholder scalar over an evening's profiles.
- `evening`: every saved frame of an evening to the `observed.json` document.
"""

from lastlight.observe.contrast import ContrastProfile, contrast_profile
from lastlight.observe.vividness import VIVIDNESS_V0_RULE, vividness_v0

__all__ = ["VIVIDNESS_V0_RULE", "ContrastProfile", "contrast_profile", "vividness_v0"]
