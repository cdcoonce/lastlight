"""The two refusals: a verdict is about one renderer, and this check knows which.

Criterion definition 8: the v1 under test is the engine and table registered by the amendment of
2026-10-12, and a later change to v1's physics or table is a different renderer that needs a new
registration. The check therefore refuses to give a verdict (and writes nothing) when

1. **the table's digest differs.** The SHA-256 of the bytes of the table file the renderer
   actually loads is compared with `REGISTERED_TABLE_SHA256`, a constant of this module that is
   also written in the criterion. It is deliberately not compared with `lastlight.render.v1`'s
   own pin: a change to the table and to that pin could be made together, and the check would
   then vouch for a renderer nobody registered.
2. **the v1 physics differs.** Three probes of `sky_xyz(..., engine="v1")` are compared with the
   values registered when v1 was registered (computed on `main` at 807ed34, copied here, not
   regenerated). A relative difference above 1e-6 in any component (absolute floor 1e-12) is a
   different renderer: the code that combines v0 and the table may have changed while the table
   did not.

The same probes for v0 are pinned in the tests, not checked here: a v0 change is not a refusal,
but the report's v0 ratio should be the registered v0.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from importlib import resources
from typing import Final

import numpy as np

from lastlight.render import Atmosphere, sky_xyz
from lastlight.render import v1 as v1_engine

REGISTERED_TABLE_SHA256: Final = "c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368"
"""SHA-256 of `src/lastlight/render/data/bruneton_ms_xyz.bin`, from the criterion's amendment."""

PROBES: Final[tuple[tuple[float, float, float], ...]] = (
    (0.06, 0.0, 20.0),
    (0.03, -3.0, 25.0),
    (0.2, 3.0, 15.0),
)
"""(aod_550, sun elevation, relative azimuth in degrees) of the three fingerprint probes."""

PROBE_OZONE_DU: Final = 272.0
PROBE_VIEWS: Final = (1.5066301585401483, 3.9943811580741357)
"""The view elevations of the probes: WHTM1's lowest band and its reference at the nominal scale."""

REGISTERED_V1_XYZ: Final[tuple[tuple[tuple[float, ...], ...], ...]] = (
    (
        (0.036618204379745126, 0.031229995875955967, 0.012895763795915597),
        (0.042844952295487154, 0.0404837916060278, 0.020678019486736227),
    ),
    (
        (0.005803768231628109, 0.005443700408744306, 0.0028857776736254457),
        (0.006884026755780439, 0.007155401230767449, 0.0052584096899803276),
    ),
    (
        (0.09255670618269227, 0.08138806553723171, 0.032392593837250565),
        (0.14232634249246007, 0.12913720925432798, 0.046845946329301724),
    ),
)
"""CIE XYZ of v1 at each probe, rows [lowest view, highest view], as registered."""

RELATIVE_TOLERANCE: Final = 1e-6
ABSOLUTE_FLOOR: Final = 1e-12


class Refusal(Exception):
    """The check will not give a verdict. The message is one line saying what differs."""


def read_table_bytes() -> bytes:
    """The bytes of the table file `lastlight.render.v1` loads by default."""
    return (resources.files("lastlight.render") / "data" / v1_engine.TABLE_NAME).read_bytes()


def check_table(read: Callable[[], bytes] | None = None) -> None:
    """Raise `Refusal` unless the table's SHA-256 is `REGISTERED_TABLE_SHA256`."""
    data = (read or read_table_bytes)()
    digest = hashlib.sha256(data).hexdigest()
    if digest != REGISTERED_TABLE_SHA256:
        raise Refusal(
            f"the v1 table's SHA-256 is {digest}, not the registered {REGISTERED_TABLE_SHA256}: "
            "a different table is a different renderer and needs a new registration"
        )


def probe_xyz(engine: str) -> np.ndarray:
    """XYZ at the three probes for `engine`: shape (3 probes, 2 views, 3), through `sky_xyz`."""
    return np.array(
        [
            sky_xyz(
                Atmosphere(aod_550=aod, ozone_du=PROBE_OZONE_DU),
                sun,
                list(PROBE_VIEWS),
                azimuth,
                engine=engine,
            )
            for aod, sun, azimuth in PROBES
        ]
    )


def check_fingerprint(evaluate: Callable[[str], np.ndarray] | None = None) -> None:
    """Raise `Refusal` unless v1 reproduces the registered probe values.

    `evaluate(engine)` returns the probe XYZ array; it defaults to `probe_xyz`.
    """
    got = np.asarray((evaluate or probe_xyz)("v1"), dtype=float)
    want = np.array(REGISTERED_V1_XYZ)
    if got.shape != want.shape:
        raise Refusal(f"the v1 probes have shape {got.shape}, not the registered {want.shape}")
    allowed = np.maximum(ABSOLUTE_FLOOR, RELATIVE_TOLERANCE * np.abs(want))
    off = ~(np.abs(got - want) <= allowed)  # also true for a NaN
    if off.any():
        probe, view, component = (int(i) for i in np.argwhere(off)[0])
        aod, sun, azimuth = PROBES[probe]
        raise Refusal(
            f"v1's physics differs from the registered renderer at probe (aod_550 {aod}, sun "
            f"elevation {sun}, relative azimuth {azimuth}), view {PROBE_VIEWS[view]:.4f}, "
            f"{'XYZ'[component]}: {got[probe, view, component]!r} against the registered "
            f"{want[probe, view, component]!r}; a change to v1 is a different renderer"
        )


def check_registered() -> None:
    """Both refusals, the digest first. Raises `Refusal`."""
    check_table()
    check_fingerprint()
