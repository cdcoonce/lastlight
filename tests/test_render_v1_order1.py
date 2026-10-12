"""Renderer v1, the order-1 oracle: the Bruneton port's own single scattering against v0 (#32).

The port (`lastlight.render.bruneton`) computes every order of scattering on its own grid, with
its own transmittance table, ray sampling and Earth's-shadow test. Its first order, evaluated at
the ground observer at the precomputation's resolution, must agree with v0's `sky_spectrum`,
which is itself checked against a brute-force sum (`tests/test_render_physics.py`). This is the
check that the port's geometry, transmittance and shadow are right before any higher order is
trusted.

The tolerances are the conductor's (issue #32): no looser than 5 percent for a sun at or above
the horizon, and 15 percent at sun -3 and -6, where the lit part of each sightline is a thin
shell high above the observer.
"""

from __future__ import annotations

import pytest

from lastlight.render import Atmosphere, bruneton, sky_spectrum, v1
from lastlight.render.atmosphere import WAVELENGTHS_NM

HAZY = Atmosphere(aod_550=0.3, ozone_du=300.0)
UNLIKE = Atmosphere(  # every parameter off its default, so none can be ignored unnoticed
    aod_550=0.6,
    angstrom=0.4,
    aerosol_scale_height_km=0.6,
    ozone_du=150.0,
    pressure_hpa=900.0,
    asymmetry=0.4,
    single_scatter_albedo=0.75,
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


@pytest.mark.parametrize("atm", [HAZY, UNLIKE], ids=["hazy", "every-parameter-changed"])
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
                assert want > 0.0, f"v0 is dark at sun {sun}, view {view}, azimuth {azimuth}"
                error = abs(got / want - 1.0)
                if error > tolerance(sun):
                    failures.append(
                        f"view {view}, azimuth {azimuth}, {wavelength} nm: "
                        f"port {got:.6g} v0 {want:.6g} ({100 * error:.1f} percent)"
                    )
    assert not failures, f"sun {sun}: " + "; ".join(failures)


def test_the_order_1_check_runs_at_the_resolution_the_table_was_built_with() -> None:
    """The oracle validates the table only if it runs at the table's own resolution."""
    assert v1.default_table().metadata["resolution"] == bruneton.RESOLUTION.as_dict()


def test_the_port_refuses_a_view_below_the_horizon() -> None:
    with pytest.raises(ValueError, match="view"):
        bruneton.single_scattering_spectrum(HAZY, 2.0, -0.5, 0.0)
