"""Renderer v1, the Bruneton port's orders 2 and up: a pinned regression (#32).

The port (`lastlight.render.bruneton`) is checked against v0 at order 1 (`test_render_v1_order1`),
but the light scattered twice or more, which is what the committed table holds, was tied to
nothing in CI: the table and the off-node fixture are frozen outputs of the port, and the tiny-grid
tests compare the port with itself. This pins the port's numbers at the tiny resolution, which
runs in a fraction of a second, so that any edit that moves orders 2 and up (a dropped ground
term, a missing coefficient) turns a test red, whether or not the edit is right. It is not a
precision check of the committed table: that is the convergence report's job
(`docs/renderer/bruneton-convergence.md`).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from lastlight.render import Atmosphere, bruneton


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
    is the convergence report, which compares the port with an independent Monte Carlo.

    A deliberate change to the port moves these numbers, the committed table (so its SHA-256
    pin) and `tests/fixtures/bruneton_offnode.json` together: regenerate all of them in one
    change, and say why in its message. The albedo-0 rows leave the ground out, so the ground
    terms and the air's own terms are pinned apart.
    """
    got = tiny_field(PINNED_ATMOSPHERE, albedo).multiple_xyz(sun, PINNED_VIEWS, azimuth)
    np.testing.assert_allclose(got, PINNED[(albedo, sun, azimuth)], rtol=1e-3)
