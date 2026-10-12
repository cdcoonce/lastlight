"""Renderer v1, the Bruneton port's orders 2 and up: a pinned regression and a Monte Carlo (#32).

The port (`lastlight.render.bruneton`) is checked against v0 at order 1 (`test_render_v1_order1`),
but the light scattered twice or more, which is what the committed table holds, was tied to
nothing in CI: the table and the off-node fixture are frozen outputs of the port, and the tiny-grid
tests compare the port with itself. Two tests close that, at the tiny resolution, which runs in
a fraction of a second:

* `test_the_port_at_the_tiny_resolution_reproduces_pinned_multiple_scattering` pins numbers. It
  catches any edit that moves orders 2 and up (a dropped ground term, a missing coefficient),
  whether or not the edit is right.
* `test_the_port_at_the_tiny_resolution_agrees_with_the_monte_carlo_for_orders_2_and_3` compares
  the port with an independent method, the tool's own backward Monte Carlo
  (`tools/make_bruneton_table.py --monte-carlo`), which shares only the atmosphere's density
  functions with the port. It catches an edit that moves orders 2 and up grossly wrong.

The pins say the port still computes what it computed. The Monte Carlo says that what it computes
is, within the tiny grid's own error, the physics. Neither is a precision check of the committed
table: that is the convergence report's job (`docs/renderer/bruneton-convergence.md`).
"""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from lastlight.render import Atmosphere, bruneton, sky_spectrum

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "make_bruneton_table.py"


@cache
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_bruneton_table", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_bruneton_table", module)
    spec.loader.exec_module(module)
    return module


@cache
def tiny_field(atm: Atmosphere, ground_albedo: float = bruneton.GROUND_ALBEDO):
    return bruneton.precompute(
        atm, resolution=bruneton.TINY_RESOLUTION, ground_albedo=ground_albedo
    )


# --- a regression pin ---------------------------------------------------------------------------

PINNED_ATMOSPHERE = Atmosphere(aod_550=0.1, ozone_du=290.0)
PINNED_VIEWS = (1.5, 10.0, 45.0)
# (ground albedo, sun elevation, relative azimuth) -> X, Y, Z at PINNED_VIEWS
PINNED = {
    (0.2, 0.0, 30.0): [
        [8.820753e-03, 9.217771e-03, 8.938592e-03],
        [6.909310e-03, 7.474422e-03, 1.113169e-02],
        [1.958333e-03, 2.008373e-03, 4.578247e-03],
    ],
    (0.2, -4.0, 90.0): [
        [1.837604e-04, 1.964208e-04, 2.245849e-04],
        [2.279164e-04, 2.527623e-04, 4.250834e-04],
        [9.871187e-05, 1.026470e-04, 2.506084e-04],
    ],
    (0.2, 6.0, 0.0): [
        [1.507459e-01, 1.541311e-01, 1.015841e-01],
        [7.647826e-02, 8.162678e-02, 8.551827e-02],
        [1.390918e-02, 1.451026e-02, 2.650389e-02],
    ],
    (0.0, 0.0, 30.0): [
        [8.403379e-03, 8.777835e-03, 8.434099e-03],
        [6.651794e-03, 7.195601e-03, 1.071324e-02],
        [1.888414e-03, 1.934423e-03, 4.431844e-03],
    ],
    (0.0, -4.0, 90.0): [
        [1.735010e-04, 1.853593e-04, 2.107858e-04],
        [2.206994e-04, 2.447229e-04, 4.118246e-04],
        [9.638542e-05, 1.001225e-04, 2.451450e-04],
    ],
    (0.0, 6.0, 0.0): [
        [1.457784e-01, 1.490721e-01, 9.733634e-02],
        [7.365537e-02, 7.862984e-02, 8.222912e-02],
        [1.316694e-02, 1.371964e-02, 2.534283e-02],
    ],
}


@pytest.mark.parametrize(
    ("albedo", "sun", "azimuth"),
    PINNED,
    ids=[f"albedo-{a}-sun-{s:+g}-azimuth-{z:g}" for a, s, z in PINNED],
)
def test_the_port_at_the_tiny_resolution_reproduces_pinned_multiple_scattering(
    albedo: float, sun: float, azimuth: float
) -> None:
    """Orders 2 to 3 of the port at `TINY_RESOLUTION` equal the numbers pinned here, to 0.1 percent.

    The numbers are the multiple-scattering CIE XYZ (v0's white-card scale) at views 1.5, 10 and
    45 degrees, from `bruneton.precompute(Atmosphere(aod_550=0.1, ozone_du=290.0),
    resolution=TINY_RESOLUTION, ground_albedo=albedo).multiple_xyz(sun, views, azimuth)`, run on
    this branch with the port as committed in 832768b (numpy 2.5.4, macOS arm64) and printed to
    seven significant figures. They are a regression pin of the port, not an independent physics
    oracle: a pin cannot say the port is right, only that it has not moved. What says it is right
    is the cross-check next to it (the Monte Carlo below, which the port at this resolution
    follows to within the resolution's own error) and the convergence report.

    A deliberate change to the port moves these numbers, the committed table (so its SHA-256
    pin) and `tests/fixtures/bruneton_offnode.json` together: regenerate all of them in one
    change, and say why in its message. The albedo-0 rows leave the ground out, so the ground
    terms and the air's own terms are pinned apart.
    """
    got = tiny_field(PINNED_ATMOSPHERE, albedo).multiple_xyz(sun, PINNED_VIEWS, azimuth)
    np.testing.assert_allclose(got, PINNED[(albedo, sun, azimuth)], rtol=1e-3)


# --- a cross-method oracle ----------------------------------------------------------------------

MONTE_CARLO_ATMOSPHERE = Atmosphere(aod_550=0.03, ozone_du=290.0)
MONTE_CARLO_AZIMUTH = 30.0
MONTE_CARLO_BATCHES = 8
MONTE_CARLO_PHOTONS = 25_000
"""Paths per batch: eight seeded batches, so the standard error comes from their spread."""
# sun elevation -> what the tiny grid itself may be off by there, as a share of the Monte Carlo
RESOLUTION_ALLOWANCE = {0.0: 0.10, -3.0: 0.50}
ORDER_1_ALLOWANCE = 0.04


@pytest.mark.parametrize("sun", RESOLUTION_ALLOWANCE)
def test_the_port_at_the_tiny_resolution_agrees_with_the_monte_carlo_for_orders_2_and_3(
    sun: float,
) -> None:
    """The port's orders 2 and 3 at 550 nm equal the tool's backward Monte Carlo's.

    Both sum orders 2 to `TINY_RESOLUTION.orders` (3), at aod_550 0.03 (the atmosphere of the
    tool's `--monte-carlo` run), WHTM1's lowest band elevation (the tool's `WHTM1_VIEWS[0]`) and
    relative azimuth 30 degrees, at sun 0 and -3. The Monte Carlo runs 8 seeded batches of 25,000
    paths and the tolerance is built from its own spread:

        |port - mean| <= 4 * standard error of the mean + RESOLUTION_ALLOWANCE[sun] * mean

    The allowance is the tiny grid's own error, which is large because it is tiny (8 sun
    elevations from -60 to 90, 4 azimuths): measured here on this branch with 1.6 million paths per
    geometry (standard error 0.19 and 0.33 percent), the port at this resolution is 6.9 percent
    below the Monte Carlo at sun 0 and 40.1 percent above it at sun -3, so the allowances are those
    rounded up (10 and 50 percent). That the gap is the grid's and not the port's: a finer grid
    (12 altitudes, 24 views, 32 sun elevations, 12 azimuths) is within 0.6 percent of the same Monte
    Carlo at both, and the committed report has the full-resolution port within 1.1 percent of it
    at sun 0 (azimuth 27.64, orders 2 to 13). So this check sees an error of the order of the
    allowance or more (a missing coefficient, a dropped order, a sign) and is not a precision
    check; the pin above sees small moves.

    The Monte Carlo's own order 1 is first held to v0's single scattering, which
    `test_render_physics` checks against a brute-force sum, within 4 standard errors and 4 percent:
    the rig reproduces the part of the physics that is known before its orders 2 and 3 are trusted.
    """
    module = tool()
    view = module.WHTM1_VIEWS[0]
    atm = MONTE_CARLO_ATMOSPHERE
    index = int(np.flatnonzero(bruneton.WAVELENGTHS_NM == module.MONTE_CARLO_WAVELENGTH)[0])
    runs = np.array(
        [
            module.monte_carlo(
                atm,
                sun,
                view,
                MONTE_CARLO_AZIMUTH,
                photons=MONTE_CARLO_PHOTONS,
                orders=bruneton.TINY_RESOLUTION.orders,
                seed=seed,
            )
            for seed in range(MONTE_CARLO_BATCHES)
        ]
    )

    def mean_and_error(batches: np.ndarray) -> tuple[float, float]:
        return float(batches.mean()), float(batches.std(ddof=1) / np.sqrt(len(batches)))

    single, single_error = mean_and_error(runs[:, 1])
    v0 = float(sky_spectrum(atm, sun, view, MONTE_CARLO_AZIMUTH)[index])
    assert abs(single - v0) <= 4.0 * single_error + ORDER_1_ALLOWANCE * v0

    multiple, error = mean_and_error(runs[:, 2:].sum(axis=1))
    port = float(tiny_field(atm).multiple_spectrum(sun, [view], MONTE_CARLO_AZIMUTH)[0, index])
    assert abs(port - multiple) <= 4.0 * error + RESOLUTION_ALLOWANCE[sun] * multiple
