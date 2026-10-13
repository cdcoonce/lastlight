"""A used frame rendered for the check, with v0 and with v1, through G0's seam.

Criterion definition 6. For each used frame the lowest band's rendered contrast
[log(R/G), log(B/G)] minus the reference's is `lastlight.g0.contrast.rendered_contrasts`
(`sky_xyz`, then `log_chromaticity` with its channel floor) at

- the frame's recorded `sun_elevation_deg`, exactly: it is never snapped to G0's half-degree
  sweep, which is why the sun's azimuth is searched here and not read from the sweep;
- WHTM1's band and reference elevations at the frozen fit's nominal pixel scale for the verdict,
  and at each end of the fit's `uncertainty.px_per_deg` range for the reported-only ratios;
- the relative azimuth, the fit's centre azimuth minus the sun's azimuth when the sun stands at
  that elevation on that evening at the public reference point.

The sun's azimuth stays inside this module and the rendered numbers: no azimuth, clock time or
instant is returned or published.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from lastlight import geometry
from lastlight.g0.contrast import rendered_contrasts
from lastlight.g0.view import SkyView, relative_azimuth
from lastlight.render import Atmosphere
from lastlight.render import v1 as v1_engine
from lastlight.renderer_check.observed import UsedFrame
from lastlight.site import REFERENCE_POINT

_MARGIN: dt.timedelta = dt.timedelta(minutes=45)
"""Slack either side of the sweep (+6 to -8 degrees) when searching for the sun's azimuth, enough
for the table's whole range of sun elevations (+12 to -12) at any date of the year."""
_PRECISION: dt.timedelta = dt.timedelta(milliseconds=1)

LOWEST_BAND: int = 0
RED_GREEN: int = 0
BLUE_GREEN: int = 1


class OutsideTable(ValueError):
    """A frame's sun elevation, or the evening's atmosphere, is outside renderer v1's table."""


@dataclass(frozen=True)
class FramePair:
    """One used frame: what the camera read and what each renderer says (None: not rendered).

    `observed_*` are the file's `bands[0]` values; `v0_*` and `v1_*` are the rendered lowest-band
    contrasts at the nominal pixel scale; `v1_bg_low` and `v1_bg_high` are v1's log(B/G) at the
    ends of the pixel-scale range.
    """

    sun_elevation_deg: float
    observed_bg: float
    observed_rg: float | None
    v0_bg: float | None = None
    v0_rg: float | None = None
    v1_bg: float | None = None
    v1_rg: float | None = None
    v1_bg_low: float | None = None
    v1_bg_high: float | None = None


def _falling(frames: Sequence[UsedFrame]) -> list[UsedFrame]:
    """The frames from the highest sun to the lowest, in file order among equals."""
    return sorted(frames, key=lambda frame: -frame.sun_elevation_deg)


def observed_only(frames: Sequence[UsedFrame]) -> tuple[FramePair, ...]:
    """The pairs of an evening that is not rendered: the observed values alone."""
    return tuple(
        FramePair(frame.sun_elevation_deg, frame.log_bg, frame.log_rg) for frame in _falling(frames)
    )


@cache
def _search_window(evening: dt.date) -> tuple[dt.datetime, dt.datetime]:
    lat, lon, _ = REFERENCE_POINT
    crossings = geometry.elevation_crossings(evening, lat, lon)
    return (
        crossings[geometry.ELEVATION_SWEEP[0]] - _MARGIN,
        crossings[geometry.ELEVATION_SWEEP[-1]] + _MARGIN,
    )


def sun_azimuth_at(evening: dt.date, elevation_deg: float) -> float:
    """The sun's azimuth when it stands at `elevation_deg` on the evening of `evening`.

    Found by bisection on `geometry.solar_position` at the public reference point, to a
    millisecond. INTERNAL USE ONLY, like `geometry.elevation_crossings`: it is a function of the
    time of day, and only the rendered numbers leave this module. Raises `ValueError` when the
    setting sun does not pass through that elevation inside the search window.
    """
    lat, lon, _ = REFERENCE_POINT
    early, late = _search_window(evening)
    if not (
        geometry.solar_position(late, lat, lon).elevation_deg
        <= elevation_deg
        <= geometry.solar_position(early, lat, lon).elevation_deg
    ):
        raise ValueError(f"the sun does not pass through {elevation_deg} degrees that evening")
    while late - early > _PRECISION:
        middle = early + (late - early) / 2
        if geometry.solar_position(middle, lat, lon).elevation_deg > elevation_deg:
            early = middle
        else:
            late = middle
    return geometry.solar_position(early + (late - early) / 2, lat, lon).azimuth_deg


def _in_table(atmosphere: Atmosphere, sun: float, view: SkyView) -> bool:
    try:
        v1_engine.check_domain(atmosphere, sun, view.elevations_deg)
    except ValueError:
        return False
    return True


def render_evening(
    evening: dt.date,
    aod_550: float,
    ozone_du: float,
    frames: Sequence[UsedFrame],
    views: Mapping[str, SkyView],
) -> tuple[FramePair, ...]:
    """Every used frame of an evening rendered, from the highest sun to the lowest.

    The atmosphere is `Atmosphere(aod_550=..., ozone_du=...)` with every other field at its
    default, as G0 builds it. Raises `OutsideTable` (naming the bound) if v1 cannot render the
    atmosphere or any frame at the nominal view; v1 at a pixel-scale end that is outside the
    table gives None for that end only.
    """
    atmosphere = Atmosphere(aod_550=aod_550, ozone_du=ozone_du)
    nominal = views["nominal"]
    ordered = _falling(frames)
    for frame in ordered:
        try:
            v1_engine.check_domain(atmosphere, frame.sun_elevation_deg, nominal.elevations_deg)
        except ValueError as error:
            raise OutsideTable(str(error)) from error
    pairs = []
    for frame in ordered:
        sun = frame.sun_elevation_deg
        azimuth = relative_azimuth(nominal.azimuth_deg, sun_azimuth_at(evening, sun))
        v0, _ = rendered_contrasts(atmosphere, sun, nominal, azimuth, engine="v0")
        v1, _ = rendered_contrasts(atmosphere, sun, nominal, azimuth, engine="v1")
        ends: dict[str, float | None] = {}
        for name in ("low", "high"):
            if _in_table(atmosphere, sun, views[name]):
                end, _ = rendered_contrasts(atmosphere, sun, views[name], azimuth, engine="v1")
                ends[name] = float(end[LOWEST_BAND, BLUE_GREEN])
            else:
                ends[name] = None
        pair = FramePair(
            sun_elevation_deg=sun,
            observed_bg=frame.log_bg,
            observed_rg=frame.log_rg,
            v0_bg=float(v0[LOWEST_BAND, BLUE_GREEN]),
            v0_rg=float(v0[LOWEST_BAND, RED_GREEN]),
            v1_bg=float(v1[LOWEST_BAND, BLUE_GREEN]),
            v1_rg=float(v1[LOWEST_BAND, RED_GREEN]),
            v1_bg_low=ends["low"],
            v1_bg_high=ends["high"],
        )
        pairs.append(pair)
    return tuple(pairs)
