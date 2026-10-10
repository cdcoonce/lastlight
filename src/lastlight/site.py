"""The sites lastlight knows about: one public reference point, one private camera site.

Everything the repository publishes is modelled at `REFERENCE_POINT`, the public reference
point in downtown Phoenix, never at the camera's real location (docs/design.md section 3).
Sun elevation changes about 0.01 degrees per km, so the exact site is never needed for
modelling.

The camera site's coordinates live only in a private file on the capture host (named by the
`LASTLIGHT_SITE_FILE` environment variable), never in the repository, CI, logs or images.
`PRIVATE_SITE_SENTINEL` is the key that marks such a file's content. `lastlight privacy-scan`
treats any occurrence of it in a tracked or generated file as a leak (rule P005).
"""

from __future__ import annotations

REFERENCE_POINT: tuple[float, float, float] = (33.45, -112.07, 340.0)
"""(latitude, longitude, elevation) of the public reference point: degrees north, degrees east
(negative is west) and metres above sea level.

This is downtown Phoenix. It is public: the privacy scanner lets exactly this point through
(rule P003) and flags every other coordinate pair inside Arizona.
"""

PRIVATE_SITE_SENTINEL = "LASTLIGHT_PRIVATE_SITE"
"""The key that marks private site file content; it must never appear in a tracked file."""
