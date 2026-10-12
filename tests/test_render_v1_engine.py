"""Renderer v1 behind v0's interface: the engine keyword, the domain and the physical signs (#32).

v1's XYZ is v0's single-scattering XYZ, by the unchanged code path, plus the multiple-scattering
XYZ interpolated from the committed Bruneton table (`lastlight.render.v1`). `sky_xyz`,
`render_row` and `render_curve` take `engine="v0"` (the default) or `"v1"`; anything else is a
`ValueError`. Outside the table's domain, or for an atmosphere whose fixed fields differ from the
values the table bakes, v1 raises `ValueError` naming the bound or the field. It never clamps.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
import time

import numpy as np
import pytest

from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity
from lastlight.g0.view import relative_azimuth, sun_azimuths
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.render import (
    VIEW_ELEVATIONS,
    Atmosphere,
    bruneton,
    render_curve,
    render_row,
    sky_spectrum,
    sky_xyz,
    v1,
)
from lastlight.site import REFERENCE_POINT

ATM = Atmosphere(aod_550=0.05, ozone_du=290.0)
VIEWS = (1.5, 2.0, 2.4, 2.9, 3.3, 4.0)

# --- the engine keyword -------------------------------------------------------------------


def test_the_default_engine_is_v0_and_changes_nothing() -> None:
    np.testing.assert_array_equal(
        sky_xyz(ATM, -2.0, VIEWS, 27.0), sky_xyz(ATM, -2.0, VIEWS, 27.0, engine="v0")
    )
    assert render_row(ATM, 1.0, 30.0) == render_row(ATM, 1.0, 30.0, engine="v0")


def test_the_default_curve_is_the_v0_curve() -> None:
    assert render_curve(ATM) == render_curve(ATM, engine="v0")


@pytest.mark.parametrize("engine", ["v2", "V1", "", "bruneton", "v0 "])
def test_an_unknown_engine_is_refused_by_every_entry_point(engine: str) -> None:
    with pytest.raises(ValueError, match="engine"):
        sky_xyz(ATM, 0.0, VIEWS, engine=engine)
    with pytest.raises(ValueError, match="engine"):
        render_row(ATM, 0.0, engine=engine)
    with pytest.raises(ValueError, match="engine"):
        render_curve(ATM, engine=engine)


@pytest.mark.parametrize("sun", [6.0, 0.0, -3.0, -8.0, -12.0])
@pytest.mark.parametrize("azimuth", [0.0, 27.0, 120.0])
def test_v1_is_v0_plus_the_tables_multiple_scattering(sun: float, azimuth: float) -> None:
    single = sky_xyz(ATM, sun, VIEWS, azimuth, engine="v0")
    extra = v1.multiple_scattering_xyz(ATM, sun, VIEWS, azimuth)
    np.testing.assert_array_equal(sky_xyz(ATM, sun, VIEWS, azimuth, engine="v1"), single + extra)


def test_v1_rows_and_curve_have_v0_s_shape() -> None:
    row = render_row(ATM, -1.0, 27.0, engine="v1")
    assert [point["view_elevation"] for point in row] == list(VIEW_ELEVATIONS)
    curve = render_curve(ATM, engine="v1")
    v0_curve = render_curve(ATM)
    assert [(p["sun_elevation"], p["view_elevation"]) for p in curve] == [
        (p["sun_elevation"], p["view_elevation"]) for p in v0_curve
    ]
    assert all(math.isfinite(p[key]) for p in curve for key in ("L", "a", "b"))
    assert curve != v0_curve


def test_sky_spectrum_stays_v0_only_and_says_so() -> None:
    doc = " ".join((sky_spectrum.__doc__ or "").split())
    assert "v0" in doc and "v1" in doc
    assert "spectra" in doc or "spectrum" in doc


# --- the domain ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("atm", "sun", "views", "named"),
    [
        (Atmosphere(aod_550=1.0001), 0.0, (2.0,), "aod_550"),
        (Atmosphere(aod_550=2.5), 0.0, (2.0,), "aod_550"),
        (Atmosphere(aod_550=0.1, ozone_du=249.9), 0.0, (2.0,), "ozone_du"),
        (Atmosphere(aod_550=0.1, ozone_du=330.1), 0.0, (2.0,), "ozone_du"),
        (Atmosphere(aod_550=0.1, ozone_du=0.0), 0.0, (2.0,), "ozone_du"),
        (ATM, -12.01, (2.0,), "sun"),
        (ATM, 12.5, (2.0,), "sun"),
        (ATM, -30.0, (2.0,), "sun"),
        (ATM, 0.0, (-0.1,), "view"),
        (ATM, 0.0, (2.0, 90.1), "view"),
    ],
)
def test_outside_the_tables_domain_v1_names_the_bound_and_never_clamps(
    atm: Atmosphere, sun: float, views: tuple[float, ...], named: str
) -> None:
    with pytest.raises(ValueError, match=named):
        v1.multiple_scattering_xyz(atm, sun, views, 0.0)
    with pytest.raises(ValueError, match=named):
        sky_xyz(atm, sun, views, 0.0, engine="v1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("angstrom", 1.0),
        ("aerosol_scale_height_km", 2.0),
        ("pressure_hpa", 1013.25),
        ("asymmetry", 0.6),
        ("single_scatter_albedo", 0.9),
    ],
)
def test_an_atmosphere_the_table_does_not_bake_is_refused_by_field_name(
    field: str, value: float
) -> None:
    atm = Atmosphere(aod_550=0.1, **{field: value})
    with pytest.raises(ValueError, match=field):
        v1.multiple_scattering_xyz(atm, 0.0, (2.0,), 0.0)
    with pytest.raises(ValueError, match=field):
        render_curve(atm, engine="v1")


@pytest.mark.parametrize(
    ("atm", "sun", "views"),
    [
        (Atmosphere(aod_550=0.0, ozone_du=250.0), -12.0, (0.0, 90.0)),
        (Atmosphere(aod_550=1.0, ozone_du=330.0), 12.0, (0.0, 90.0)),
    ],
    ids=["low-corner", "high-corner"],
)
def test_the_domain_bounds_are_inclusive(
    atm: Atmosphere, sun: float, views: tuple[float, ...]
) -> None:
    extra = v1.multiple_scattering_xyz(atm, sun, views, 180.0)
    assert np.isfinite(extra).all() and (extra > 0.0).all()


@pytest.mark.parametrize(("azimuth", "folded"), [(200.0, 160.0), (-30.0, 30.0), (390.0, 30.0)])
def test_any_relative_azimuth_is_folded_onto_0_to_180(azimuth: float, folded: float) -> None:
    np.testing.assert_allclose(
        v1.multiple_scattering_xyz(ATM, -2.0, VIEWS, azimuth),
        v1.multiple_scattering_xyz(ATM, -2.0, VIEWS, folded),
        rtol=1e-12,
    )


def test_a_non_finite_relative_azimuth_is_refused() -> None:
    with pytest.raises(ValueError, match="azimuth"):
        v1.multiple_scattering_xyz(ATM, 0.0, VIEWS, math.nan)


# --- physical signs -----------------------------------------------------------------------


@pytest.mark.parametrize("aod", [0.0, 0.03, 0.3, 1.0])
def test_the_extra_light_is_positive_everywhere(aod: float) -> None:
    atm = Atmosphere(aod_550=aod)
    for sun in (12.0, 6.0, 0.0, -6.0, -12.0):
        for azimuth in (0.0, 27.0, 90.0, 180.0):
            extra = v1.multiple_scattering_xyz(atm, sun, (0.0, 1.5, 10.0, 45.0, 90.0), azimuth)
            assert (extra > 0.0).all(), (aod, sun, azimuth)


@pytest.mark.parametrize("aod", [0.03, 0.3])
@pytest.mark.parametrize("azimuth", [0.0, 27.6, 90.0, 180.0])
@pytest.mark.parametrize("view", [1.5, 4.0, 10.0, 45.0])
def test_the_extra_lights_share_of_the_total_rises_as_the_sun_sets(
    aod: float, azimuth: float, view: float
) -> None:
    """From sun +6 to -6 in 1-degree steps the multiply scattered share of Y only grows."""
    atm = Atmosphere(aod_550=aod)
    shares = []
    for sun in np.arange(6.0, -6.5, -1.0):
        extra = v1.multiple_scattering_xyz(atm, float(sun), (view,), azimuth)[0, 1]
        total = sky_xyz(atm, float(sun), (view,), azimuth, engine="v1")[0, 1]
        shares.append(extra / total)
    assert all(later > earlier for earlier, later in zip(shares, shares[1:], strict=False)), shares


def test_a_black_ground_gives_less_extra_light_than_an_albedo_of_0_2() -> None:
    """Light reflected by the ground and scattered back is part of v1; with albedo 0 it is gone.

    Run with the port at its tiny test resolution, where both albedos take a second or two.
    """
    atm = Atmosphere(aod_550=0.1, ozone_du=290.0)
    black = bruneton.precompute(atm, resolution=bruneton.TINY_RESOLUTION, ground_albedo=0.0)
    grey = bruneton.precompute(atm, resolution=bruneton.TINY_RESOLUTION, ground_albedo=0.2)
    assert bruneton.GROUND_ALBEDO == 0.2
    for sun in (6.0, 0.0, -4.0):
        for azimuth in (0.0, 90.0, 180.0):
            dark = black.multiple_xyz(sun, (1.5, 10.0, 45.0), azimuth)
            bright = grey.multiple_xyz(sun, (1.5, 10.0, 45.0), azimuth)
            assert (dark > 0.0).all()
            assert (bright > dark).all(), (sun, azimuth)


def _whtm1_lowest_band_log_bg(engine: str) -> float:
    """WHTM1's lowest band log(B/G) contrast against the reference, sun 0, aod 0.03.

    The bands are G0's nominal view (`build_views()["nominal"]`) and the relative azimuth is
    the camera's centre azimuth minus the sun's on 2026-10-10, when the sun stands at 0.
    """
    view = build_views()["nominal"]
    lat, lon, _ = REFERENCE_POINT
    sun_azimuth = sun_azimuths(dt.date(2026, 10, 10), lat, lon)[ELEVATION_SWEEP.index(0.0)]
    azimuth = relative_azimuth(view.azimuth_deg, sun_azimuth)
    xyz = sky_xyz(Atmosphere(aod_550=0.03), 0.0, view.elevations_deg, azimuth, engine=engine)
    logs, _floored = log_chromaticity(xyz)
    return float(logs[0, 1] - logs[-1, 1])


def test_v1_shrinks_the_lowest_band_log_bg_contrast_at_whtm1_at_sun_0() -> None:
    """Multiply scattered light fills the blue of the low twilight sky (G0's finding)."""
    v0_contrast = _whtm1_lowest_band_log_bg("v0")
    v1_contrast = _whtm1_lowest_band_log_bg("v1")
    assert abs(v1_contrast) < abs(v0_contrast), (v0_contrast, v1_contrast)


# --- speed --------------------------------------------------------------------------------


def _best_of(repeats: int, call) -> float:
    call()  # warm caches: the table, the optics, colour-science's import
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def test_v1_sky_xyz_for_six_view_elevations_takes_under_50_ms() -> None:
    elapsed = _best_of(7, lambda: sky_xyz(ATM, -3.0, VIEWS, 27.0, engine="v1"))
    assert elapsed < 0.050, elapsed


def test_the_full_v1_curve_renders_in_under_ten_seconds() -> None:
    render_curve(ATM, engine="v1")  # warm
    start = time.perf_counter()
    render_curve(Atmosphere(aod_550=0.2, ozone_du=310.0), engine="v1")
    assert time.perf_counter() - start < 10.0
