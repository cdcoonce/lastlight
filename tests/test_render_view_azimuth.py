"""Rendering toward a view that is not the solar vertical (issue #10, conductor decision 4).

WHTM1 looks about 289 degrees while the autumn sun sets near 255, so its sky is seen at a
scattering angle far from the solar vertical, and single-scattering colour depends on that angle.
`render_row` therefore takes the view's azimuth relative to the sun's (default 0: today's
behaviour, which the sealed forecast depends on). Three things are pinned:

* with the default, nothing changes: the output equals the pre-change numbers (copied from the
  code before the parameter existed) and an independent rebuild of the old formula;
* the relative azimuth enters only through its cosine, so +x and -x are the same sky;
* the scattering angle is the spherical-geometry angle between the sun and the view direction,
  checked on hand-computed cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lastlight.render import VIEW_ELEVATIONS, Atmosphere, render_row, sky_spectrum
from lastlight.render import transfer as transfer_module
from lastlight.render.colorimetry import spectrum_to_xyz, xyz_to_lab
from lastlight.render.curve import PEAK_LIGHTNESS, sky_xyz
from lastlight.render.transfer import scattering_cosine

ATMOSPHERE = Atmosphere(aod_550=0.11, ozone_du=300.0)

# render_row(ATMOSPHERE, sun) as the code produced it before `rel_azimuth_deg` existed:
# (view elevation, L, a, b) for the seven standard view elevations.
BEFORE_CHANGE = {
    3.0: [
        (2.0, 66.87275814157205, 22.752892176598593, 64.83230373969909),
        (5.0, 70.0, 14.064123011638952, 53.503303429400994),
        (10.0, 60.99488486938134, 7.61011358692093, 36.041937153141355),
        (15.0, 50.378197439594175, 4.771899414482805, 24.61900873452144),
        (20.0, 41.19444203209464, 3.080414066083448, 16.472442312353387),
        (30.0, 28.343767646418478, 1.1727533360241793, 6.2492767220127865),
        (45.0, 18.129662285012408, 0.013631345297082653, -0.4412013481754351),
    ],
    -2.5: [
        (2.0, 41.22535545149056, 25.80026766729268, 51.92090881637487),
        (5.0, 66.405089420122, 7.083067550998856, 33.82768301493224),
        (10.0, 70.0, -1.7448310374935105, 7.234449087261874),
        (15.0, 65.67037702175449, -2.403253893484003, -3.877796981720172),
        (20.0, 60.5850318946313, -2.093273847809607, -9.133764631486873),
        (30.0, 51.97252219613952, -1.1792945819846246, -12.860661804155082),
        (45.0, 42.34272271234552, -0.5477675285466277, -13.591357150542581),
    ],
}


def _rows_as_tuples(rows):
    return [(r["view_elevation"], r["L"], r["a"], r["b"]) for r in rows]


# --- the default changes nothing -----------------------------------------------------------------


@pytest.mark.parametrize("sun", sorted(BEFORE_CHANGE))
def test_the_default_relative_azimuth_reproduces_the_numbers_from_before_the_parameter(
    sun: float,
) -> None:
    got = _rows_as_tuples(render_row(ATMOSPHERE, sun))
    for (view, light, a, b), (v0, l0, a0, b0) in zip(got, BEFORE_CHANGE[sun], strict=True):
        assert view == v0
        assert (light, a, b) == pytest.approx((l0, a0, b0), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("sun", sorted(BEFORE_CHANGE))
def test_an_explicit_zero_offset_is_the_default_exactly(sun: float) -> None:
    assert render_row(ATMOSPHERE, sun, rel_azimuth_deg=0.0) == render_row(ATMOSPHERE, sun)


def test_the_default_equals_a_rebuild_of_the_old_formula_bit_for_bit() -> None:
    """The old render_row: one spectrum per view along the solar vertical, one exposure."""
    sun = -1.0
    spectra = [sky_spectrum(ATMOSPHERE, sun, view) for view in VIEW_ELEVATIONS]
    xyz = np.array([spectrum_to_xyz(spectrum) for spectrum in spectra])
    peak = ((PEAK_LIGHTNESS + 16.0) / 116.0) ** 3
    lab = xyz_to_lab(xyz * (peak / xyz[:, 1].max()))
    rebuilt = [
        (float(view), float(light), float(a), float(b))
        for view, (light, a, b) in zip(VIEW_ELEVATIONS, lab, strict=True)
    ]
    assert _rows_as_tuples(render_row(ATMOSPHERE, sun)) == rebuilt


# --- the offset is used, and only through its cosine ---------------------------------------------


def test_a_large_offset_changes_the_colours() -> None:
    near = render_row(ATMOSPHERE, 3.0)
    away = render_row(ATMOSPHERE, 3.0, rel_azimuth_deg=60.0)
    assert _rows_as_tuples(near) != _rows_as_tuples(away)
    assert [r["view_elevation"] for r in away] == [r["view_elevation"] for r in near]


@pytest.mark.parametrize("offset", [5.0, 34.29, 90.0, 150.0])
def test_plus_and_minus_the_same_offset_give_the_same_sky(offset: float) -> None:
    for sun in (3.0, -4.0):
        assert render_row(ATMOSPHERE, sun, rel_azimuth_deg=offset) == render_row(
            ATMOSPHERE, sun, rel_azimuth_deg=-offset
        )


def test_an_offset_of_a_full_turn_is_no_offset() -> None:
    plain = sky_xyz(ATMOSPHERE, 3.0, (2.0, 3.5), rel_azimuth_deg=0.0)
    turned = sky_xyz(ATMOSPHERE, 3.0, (2.0, 3.5), rel_azimuth_deg=360.0)
    np.testing.assert_allclose(turned, plain, rtol=1e-12)


def test_sky_xyz_gives_one_xyz_row_per_view_elevation_at_any_height() -> None:
    xyz = sky_xyz(ATMOSPHERE, -2.0, (1.1, 1.6, 3.7), rel_azimuth_deg=34.0)
    assert xyz.shape == (3, 3)
    assert np.isfinite(xyz).all()
    assert (xyz[:, 1] > 0).all()
    one = sky_xyz(ATMOSPHERE, -2.0, (1.6,), rel_azimuth_deg=34.0)
    np.testing.assert_array_equal(one[0], xyz[1])


def test_sky_xyz_default_azimuth_is_the_solar_vertical() -> None:
    np.testing.assert_array_equal(
        sky_xyz(ATMOSPHERE, 3.0, VIEW_ELEVATIONS),
        sky_xyz(ATMOSPHERE, 3.0, VIEW_ELEVATIONS, rel_azimuth_deg=0.0),
    )


# --- the scattering angle is the spherical-geometry angle ----------------------------------------


def _angle_deg(sun: float, view: float, offset: float) -> float:
    return math.degrees(math.acos(scattering_cosine(sun, view, offset)))


@pytest.mark.parametrize(
    ("sun", "view", "offset", "expected"),
    [
        (0.0, 0.0, 0.0, 0.0),  # looking at the sun
        (0.0, 0.0, 90.0, 90.0),  # on the horizon, a quarter turn from the sun
        (0.0, 0.0, 180.0, 180.0),  # the antisolar point
        (0.0, 40.0, 0.0, 40.0),  # straight up the solar vertical
        (30.0, 30.0, 180.0, 120.0),  # over the zenith: 60 + 60 degrees, not 0
        (30.0, 30.0, 90.0, 75.5224878),  # cos = sin30 sin30 + cos30 cos30 cos90 = 1/4
        (-3.0, 2.0, 34.29, None),
    ],
)
def test_the_scattering_angle_follows_spherical_geometry(
    sun: float, view: float, offset: float, expected: float | None
) -> None:
    if expected is None:  # a case with no round number: compare with the haversine form
        s, v, d = (math.radians(x) for x in (sun, view, offset))
        haversine = math.sin((v - s) / 2) ** 2 + math.cos(v) * math.cos(s) * math.sin(d / 2) ** 2
        expected = math.degrees(2 * math.asin(math.sqrt(haversine)))
    assert _angle_deg(sun, view, offset) == pytest.approx(expected, abs=1e-6)


def test_the_hand_computed_case_cosine_is_minus_one_half() -> None:
    # Sun 30 degrees up and the view 30 degrees up on the other side of the sky: the two
    # directions are 120 degrees apart (60 degrees each to the zenith), so cos = -1/2.
    assert scattering_cosine(30.0, 30.0, 180.0) == pytest.approx(-0.5, abs=1e-12)


def test_the_cosine_is_even_in_the_offset_and_periodic() -> None:
    assert scattering_cosine(5.0, 12.0, 70.0) == scattering_cosine(5.0, 12.0, -70.0)
    assert scattering_cosine(5.0, 12.0, 70.0) == pytest.approx(
        scattering_cosine(5.0, 12.0, 430.0), abs=1e-12
    )


def test_sky_spectrum_reads_the_scattering_cosine_from_that_one_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The integral uses `scattering_cosine`, so the tested geometry is the geometry rendered."""
    base = sky_spectrum(ATMOSPHERE, 3.0, 5.0, rel_azimuth_deg=40.0)
    monkeypatch.setattr(transfer_module, "scattering_cosine", lambda *args: 0.99)
    patched = sky_spectrum(ATMOSPHERE, 3.0, 5.0, rel_azimuth_deg=40.0)
    assert not np.allclose(base, patched)


def test_a_view_at_a_wider_scattering_angle_is_dimmer_in_haze() -> None:
    """Forward scattering by aerosol: the same sky 70 degrees round from the sun is dimmer."""
    hazy = Atmosphere(aod_550=0.3)
    near = sky_spectrum(hazy, 5.0, 3.0, rel_azimuth_deg=10.0).sum()
    far = sky_spectrum(hazy, 5.0, 3.0, rel_azimuth_deg=70.0).sum()
    assert near > 2 * far
