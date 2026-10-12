"""Renderer v1, the order-1 oracle: the Bruneton port's own single scattering against v0 (#32).

The port (`lastlight.render.bruneton`) computes every order of scattering on its own grid, with
its own transmittance table, ray sampling and Earth's-shadow test. Its first order, evaluated at
the ground observer at the precomputation's resolution, must agree with v0's `sky_spectrum`,
which is itself checked against a brute-force sum (`tests/test_render_physics.py`). This is the
check that the port's geometry, transmittance and shadow are right before any higher order is
trusted.

The tolerances are the conductor's (issue #32): no looser than 5 percent for a sun at or above
the horizon, and 15 percent at sun -3 and -6, where the lit part of each sightline is a thin
shell high above the observer. Where a sightline lies wholly in Earth's shadow (looking away
from the sun near the horizon at sun -3 and -6) v0 is exactly dark, and so must the port be.

The second atmosphere has every parameter off its default. It is not v0's own test atmosphere
(`UNLIKE` in `tests/test_render_physics.py`): with a 0.6 km aerosol layer of optical depth 0.6,
v0's 50 m shells are optically thick along a 1.5-degree sightline and v0 itself is 11 percent
below the brute-force sum there, so v0 is no oracle at that view. For that atmosphere the port is
checked against the brute-force sum instead.
"""

from __future__ import annotations

import pytest

from lastlight.render import Atmosphere, bruneton, sky_spectrum, v1
from lastlight.render.atmosphere import WAVELENGTHS_NM

HAZY = Atmosphere(aod_550=0.3, ozone_du=300.0)
CHANGED = Atmosphere(  # every parameter off its default, so none can be ignored unnoticed
    aod_550=0.2,
    angstrom=0.6,
    aerosol_scale_height_km=1.0,
    ozone_du=320.0,
    pressure_hpa=930.0,
    asymmetry=0.6,
    single_scatter_albedo=0.85,
)

SUN_ELEVATIONS = (10.0, 2.0, 0.0, -3.0, -6.0)
VIEW_ELEVATIONS = (1.5, 3.0, 10.0, 45.0)
RELATIVE_AZIMUTHS = (0.0, 30.0, 90.0, 180.0)
WAVELENGTHS = (450.0, 550.0, 650.0)


def tolerance(sun: float) -> float:
    """5 percent with the sun at or above the horizon, 15 percent at -3 and -6."""
    return 0.05 if sun >= 0.0 else 0.15


def _index(wavelength: float) -> int:
    return int((wavelength == WAVELENGTHS_NM).nonzero()[0][0])


@pytest.mark.parametrize("atm", [HAZY, CHANGED], ids=["hazy", "every-parameter-changed"])
@pytest.mark.parametrize("sun", SUN_ELEVATIONS)
def test_the_ports_single_scattering_at_the_ground_agrees_with_v0(
    atm: Atmosphere, sun: float
) -> None:
    failures = []
    for view in VIEW_ELEVATIONS:
        for azimuth in RELATIVE_AZIMUTHS:
            port = bruneton.single_scattering_spectrum(
                atm, sun, view, azimuth, resolution=bruneton.RESOLUTION
            )
            reference = sky_spectrum(atm, sun, view, azimuth)
            assert port.shape == WAVELENGTHS_NM.shape
            for wavelength in WAVELENGTHS:
                got, want = port[_index(wavelength)], reference[_index(wavelength)]
                if want == 0.0:  # the whole sightline is in Earth's shadow
                    if got != 0.0:
                        failures.append(f"view {view}, azimuth {azimuth}: v0 dark, port {got}")
                    continue
                error = abs(got / want - 1.0)
                if error > tolerance(sun):
                    failures.append(
                        f"view {view}, azimuth {azimuth}, {wavelength} nm: "
                        f"port {got:.6g} v0 {want:.6g} ({100 * error:.1f} percent)"
                    )
    assert not failures, f"sun {sun}: " + "; ".join(failures)


def test_the_check_reaches_sightlines_that_are_wholly_in_earths_shadow() -> None:
    """Away from the sun, low, at sun -3: v0 is exactly zero, and the port agrees."""
    assert not sky_spectrum(HAZY, -3.0, 1.5, 180.0).any()
    assert not bruneton.single_scattering_spectrum(HAZY, -3.0, 1.5, 180.0).any()


@pytest.mark.parametrize("sun", [10.0, 2.0, 0.0])
def test_for_v0s_thin_dense_layer_the_port_agrees_with_the_brute_force_sum(sun: float) -> None:
    """v0's own test atmosphere at the grazing view, against the sum v0 is itself tested by."""
    from test_render_physics import UNLIKE, _reference_radiance

    port = bruneton.single_scattering_spectrum(UNLIKE, sun, 1.5, 180.0)
    for wavelength in WAVELENGTHS:
        reference = _reference_radiance(UNLIKE, sun, 1.5, 180.0, wavelength)
        assert port[_index(wavelength)] == pytest.approx(reference, rel=tolerance(sun))


def test_the_order_1_check_runs_at_the_resolution_the_table_was_built_with() -> None:
    """The oracle validates the table only if it runs at the table's own resolution."""
    assert v1.default_table().metadata["resolution"] == bruneton.RESOLUTION.as_dict()


def test_the_port_refuses_a_view_below_the_horizon() -> None:
    with pytest.raises(ValueError, match="view"):
        bruneton.single_scattering_spectrum(HAZY, 2.0, -0.5, 0.0)
