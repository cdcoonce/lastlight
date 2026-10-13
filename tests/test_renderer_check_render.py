"""Rendering a frame for the check: the sun's azimuth at the frame's own elevation, both engines.

The rendered contrast of a used frame goes through G0's seam (`sky_xyz`, then `log_chromaticity`
with its channel floor), at the frame's recorded sun elevation without snapping to G0's sweep, at
WHTM1's nominal view for the verdict and at each end of the pixel-scale range for the
reported-only ratios, with the fit's centre azimuth minus the sun's azimuth at that elevation.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from test_renderer_check_support import LAT, LON, SUNS, expected_rendered

from lastlight import geometry
from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity
from lastlight.g0.view import sun_azimuths
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.render import Atmosphere, sky_xyz
from lastlight.renderer_check import render
from lastlight.renderer_check.observed import UsedFrame

DAY = dt.date(2026, 10, 11)
AOD, OZONE = 0.05, 280.0


@pytest.fixture(scope="module")
def views():
    return build_views()


def frames(*suns: float) -> list[UsedFrame]:
    return [UsedFrame(sun_elevation_deg=s, log_bg=-0.1, log_rg=0.05) for s in suns]


# --- the sun's azimuth at an arbitrary elevation --------------------------------------------------


def test_the_sun_azimuth_agrees_with_g0s_at_every_sweep_elevation() -> None:
    reference = dict(zip(ELEVATION_SWEEP, sun_azimuths(DAY, LAT, LON), strict=True))
    for elevation in (6.0, 3.5, 1.0, 0.0, -0.5, -4.0, -8.0):
        assert render.sun_azimuth_at(DAY, elevation) == pytest.approx(
            reference[elevation], abs=0.02
        )


def test_the_sun_azimuth_at_the_horizon_is_the_sunset_azimuth() -> None:
    assert render.sun_azimuth_at(DAY, 0.0) == pytest.approx(
        geometry.sunset_azimuth(DAY, LAT, LON), abs=0.02
    )


def test_between_two_sweep_elevations_the_azimuth_lies_between_theirs() -> None:
    reference = dict(zip(ELEVATION_SWEEP, sun_azimuths(DAY, LAT, LON), strict=True))
    azimuth = render.sun_azimuth_at(DAY, 1.37)
    low, high = sorted((reference[1.0], reference[1.5]))
    assert low < azimuth < high


def test_the_sun_azimuth_moves_toward_the_west_as_the_sun_sets() -> None:
    values = [render.sun_azimuth_at(DAY, e) for e in (4.0, 2.0, 0.0, -2.0, -4.0)]
    assert values == sorted(values) and len(set(values)) == len(values)


def test_the_sun_azimuth_depends_on_the_date() -> None:
    june = render.sun_azimuth_at(dt.date(2026, 6, 21), 0.0)
    december = render.sun_azimuth_at(dt.date(2026, 12, 21), 0.0)
    assert june - december > 30.0


def test_an_elevation_the_evening_never_reaches_is_an_error() -> None:
    with pytest.raises(ValueError):
        render.sun_azimuth_at(DAY, 40.0)


# --- rendering a frame ----------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["v0", "v1"])
def test_the_rendered_contrast_matches_an_independent_route_at_sweep_elevations(
    engine: str, views
) -> None:
    pairs = render.render_evening(DAY, AOD, OZONE, frames(*SUNS), views)
    want = expected_rendered(engine, DAY, AOD, OZONE, SUNS)
    for pair, row in zip(pairs, want, strict=True):
        assert getattr(pair, f"{engine}_rg") == pytest.approx(row[0], rel=1e-4)
        assert getattr(pair, f"{engine}_bg") == pytest.approx(row[1], rel=1e-4)


def test_the_two_engines_differ_which_proves_each_is_really_used(views) -> None:
    pair = render.render_evening(DAY, AOD, OZONE, frames(0.0), views)[0]
    assert pair.v0_bg < -2.0 and -1.0 < pair.v1_bg < 0.0  # v0 about -2.6 or lower, v1 about -0.4
    assert pair.v1_bg != pair.v0_bg and pair.v1_rg != pair.v0_rg


def test_the_pixel_scale_ends_are_v1_at_the_low_and_high_views(views) -> None:
    pairs = render.render_evening(DAY, AOD, OZONE, frames(*SUNS), views)
    low = expected_rendered("v1", DAY, AOD, OZONE, SUNS, scale="low")
    high = expected_rendered("v1", DAY, AOD, OZONE, SUNS, scale="high")
    nominal = expected_rendered("v1", DAY, AOD, OZONE, SUNS)
    for pair, lo, hi, nom in zip(pairs, low, high, nominal, strict=True):
        assert pair.v1_bg_low == pytest.approx(lo[1], rel=1e-4)
        assert pair.v1_bg_high == pytest.approx(hi[1], rel=1e-4)
        assert pair.v1_bg == pytest.approx(nom[1], rel=1e-4)
        assert abs(pair.v1_bg_low - pair.v1_bg) > 1e-3  # the ends really are other views


def test_the_sun_elevation_is_not_snapped_to_the_sweep(views) -> None:
    sun = 1.37
    sweep = dict(zip(ELEVATION_SWEEP, sun_azimuths(DAY, LAT, LON), strict=True))
    azimuth = sweep[1.0] + (sweep[1.5] - sweep[1.0]) * (sun - 1.0) / 0.5
    relative = (views["nominal"].azimuth_deg - azimuth + 180.0) % 360.0 - 180.0
    xyz = sky_xyz(
        Atmosphere(aod_550=AOD, ozone_du=OZONE),
        sun,
        views["nominal"].elevations_deg,
        relative,
        engine="v1",
    )
    logs, _ = log_chromaticity(xyz)
    want_bg = logs[0, 1] - logs[-1, 1]
    pair = render.render_evening(DAY, AOD, OZONE, frames(sun), views)[0]
    assert pair.v1_bg == pytest.approx(want_bg, rel=2e-3)
    # The two neighbouring sweep elevations are further from it than that tolerance.
    for neighbour in (1.0, 1.5):
        other = render.render_evening(DAY, AOD, OZONE, frames(neighbour), views)[0]
        assert abs(other.v1_bg - want_bg) > 5 * 2e-3 * abs(want_bg)


def test_the_elevation_is_used_to_the_precision_recorded_not_rounded(views) -> None:
    a = render.render_evening(DAY, AOD, OZONE, frames(0.40), views)[0]
    b = render.render_evening(DAY, AOD, OZONE, frames(0.41), views)[0]
    assert a.v1_bg != b.v1_bg and a.v0_bg != b.v0_bg


def test_the_third_decimal_of_the_sun_elevation_reaches_the_render(views) -> None:
    # Both round to 0.40 at two decimals (the precision frames.csv prints), but are other suns.
    a = render.render_evening(DAY, AOD, OZONE, frames(0.4001), views)[0]
    b = render.render_evening(DAY, AOD, OZONE, frames(0.4049), views)[0]
    for name in ("v1_bg", "v0_bg", "v1_rg", "v0_rg"):  # they differ by 1e-4 to 8e-4
        assert abs(getattr(a, name) - getattr(b, name)) > 1e-5, name


def test_pairs_come_in_order_of_falling_sun_elevation_whatever_the_file_order(views) -> None:
    pairs = render.render_evening(DAY, AOD, OZONE, frames(-1.5, 3.0, 0.0, 1.5), views)
    assert [p.sun_elevation_deg for p in pairs] == [3.0, 1.5, 0.0, -1.5]


def test_a_pair_carries_the_observed_values(views) -> None:
    frame = UsedFrame(sun_elevation_deg=1.5, log_bg=-0.123456, log_rg=0.0421)
    pair = render.render_evening(DAY, AOD, OZONE, [frame], views)[0]
    assert (pair.observed_bg, pair.observed_rg) == (-0.123456, 0.0421)


def test_a_null_observed_log_rg_stays_null(views) -> None:
    frame = UsedFrame(sun_elevation_deg=1.5, log_bg=-0.1, log_rg=None)
    pair = render.render_evening(DAY, AOD, OZONE, [frame], views)[0]
    assert pair.observed_rg is None and np.isfinite(pair.v1_rg)


# --- outside v1's table ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("aod", "ozone", "sun"),
    [(1.2, 280.0, 1.0), (0.05, 340.0, 1.0), (0.05, 240.0, 1.0), (0.05, 280.0, 13.0)],
)
def test_an_atmosphere_or_sun_outside_the_table_raises_outside_table(
    aod: float, ozone: float, sun: float, views
) -> None:
    with pytest.raises(render.OutsideTable):
        render.render_evening(DAY, aod, ozone, frames(sun), views)


def test_one_frame_outside_the_table_fails_the_whole_evening(views) -> None:
    with pytest.raises(render.OutsideTable):
        render.render_evening(DAY, AOD, OZONE, frames(1.0, 0.0, -13.0), views)


def test_inside_the_table_edges_is_fine(views) -> None:
    render.render_evening(DAY, 1.0, 330.0, frames(-8.0, 6.0), views)
