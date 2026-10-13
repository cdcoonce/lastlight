"""The two refusals of the check: the table digest and the v1 physics fingerprint.

A verdict is about one renderer (criterion definition 8). The digest is taken of the bytes of
the table file the renderer actually loads and compared with a constant in the check's own module
(never with the renderer's own pin, which could change together with the table), and three
probes of the engine are compared with the values registered when v1 was registered.
"""

from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
import pytest
from test_renderer_check_support import CRITERION, TABLE_SHA256

from lastlight.render import Atmosphere, sky_xyz, v1
from lastlight.renderer_check import guards

LOW, HIGH = 1.5066301585401483, 3.9943811580741357

# Computed on main at 807ed34 (the registered v1) by the conductor, not by the code under test.
PROBES = [
    (0.06, 0.0, 20.0),
    (0.03, -3.0, 25.0),
    (0.2, 3.0, 15.0),
]
V1_REGISTERED = [
    [
        [0.036618204379745126, 0.031229995875955967, 0.012895763795915597],
        [0.042844952295487154, 0.0404837916060278, 0.020678019486736227],
    ],
    [
        [0.005803768231628109, 0.005443700408744306, 0.0028857776736254457],
        [0.006884026755780439, 0.007155401230767449, 0.0052584096899803276],
    ],
    [
        [0.09255670618269227, 0.08138806553723171, 0.032392593837250565],
        [0.14232634249246007, 0.12913720925432798, 0.046845946329301724],
    ],
]
V0_REGISTERED = [
    [
        [0.022784643320134683, 0.01757923820655384, 0.001470827124095022],
        [0.030757163094225295, 0.028056610958648812, 0.007946208579905665],
    ],
    [
        [0.0037726643085672605, 0.003316645575444221, 0.0005688199397762706],
        [0.005210648253853109, 0.005354349399999691, 0.002715008199005599],
    ],
    [
        [0.04340643459434342, 0.03389462084318697, 0.0033383335171692954],
        [0.08763549519869064, 0.07519843350596886, 0.012676410961999372],
    ],
]


def table_bytes() -> bytes:
    return (resources.files("lastlight.render") / "data" / "bruneton_ms_xyz.bin").read_bytes()


# --- the digest ----------------------------------------------------------------------------------


def test_the_registered_digest_is_the_one_in_the_criterions_amendment() -> None:
    assert guards.REGISTERED_TABLE_SHA256 == TABLE_SHA256
    assert guards.REGISTERED_TABLE_SHA256 in CRITERION.read_text(encoding="utf-8")


def test_the_committed_table_matches_the_registered_digest() -> None:
    assert hashlib.sha256(table_bytes()).hexdigest() == guards.REGISTERED_TABLE_SHA256
    guards.check_table()  # does not raise


def test_the_digest_is_taken_of_the_bytes_the_renderer_loads() -> None:
    assert guards.read_table_bytes() == table_bytes()


def test_one_changed_byte_is_refused_naming_both_digests() -> None:
    data = bytearray(table_bytes())
    data[len(data) // 2] ^= 0x01
    with pytest.raises(guards.Refusal) as caught:
        guards.check_table(lambda: bytes(data))
    message = str(caught.value)
    assert hashlib.sha256(bytes(data)).hexdigest() in message
    assert TABLE_SHA256 in message
    assert "\n" not in message


def test_a_changed_pin_inside_the_renderer_cannot_vouch_for_a_changed_table(
    monkeypatch,
) -> None:
    data = bytes(table_bytes() + b"\x00")
    monkeypatch.setattr(v1, "TABLE_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(guards.Refusal):
        guards.check_table(lambda: data)


def test_check_table_reads_the_file_through_the_module_function_at_call_time(
    monkeypatch,
) -> None:
    monkeypatch.setattr(guards, "read_table_bytes", lambda: b"not the table")
    with pytest.raises(guards.Refusal):
        guards.check_table()


# --- the fingerprint -----------------------------------------------------------------------------


def test_the_probes_are_the_registered_ones() -> None:
    assert [tuple(probe) for probe in guards.PROBES] == PROBES
    assert guards.PROBE_VIEWS == (LOW, HIGH)
    assert guards.PROBE_OZONE_DU == 272.0


def test_the_v1_engine_reproduces_the_registered_probes() -> None:
    got = guards.probe_xyz("v1")
    np.testing.assert_allclose(np.array(got), np.array(V1_REGISTERED), rtol=1e-6, atol=1e-12)
    guards.check_fingerprint()  # does not raise


def test_the_checks_v0_path_reproduces_the_registered_v0_probes() -> None:
    # Not a refusal: this pins that the report's v0 ratio is the registered v0.
    got = guards.probe_xyz("v0")
    np.testing.assert_allclose(np.array(got), np.array(V0_REGISTERED), rtol=1e-6, atol=1e-12)


def test_the_probes_are_evaluated_through_sky_xyz_with_the_stated_engine() -> None:
    for engine, registered in (("v0", V0_REGISTERED), ("v1", V1_REGISTERED)):
        for (aod, sun, azimuth), want in zip(PROBES, registered, strict=True):
            direct = sky_xyz(
                Atmosphere(aod_550=aod, ozone_du=272.0), sun, [LOW, HIGH], azimuth, engine=engine
            )
            np.testing.assert_allclose(direct, np.array(want), rtol=1e-6, atol=1e-12)


def scaled(factor: float, which=(0, 0, 0)):
    """A stand-in for `probe_xyz` that scales one component of the v1 result by `factor`."""

    def evaluate(engine: str):
        values = np.array(V1_REGISTERED)
        probe, view, component = which
        values[probe, view, component] *= factor
        return values

    return evaluate


def test_a_component_off_by_more_than_one_millionth_is_refused() -> None:
    with pytest.raises(guards.Refusal) as caught:
        guards.check_fingerprint(scaled(1.0 + 2e-6))
    message = str(caught.value)
    assert "v1" in message and "0.06" in message and "\n" not in message


def test_a_component_off_by_less_than_one_millionth_is_accepted() -> None:
    guards.check_fingerprint(scaled(1.0 + 5e-7))
    guards.check_fingerprint(scaled(1.0 - 5e-7))


@pytest.mark.parametrize("probe", [0, 1, 2])
@pytest.mark.parametrize("view", [0, 1])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_every_probe_view_and_component_is_compared(probe: int, view: int, component: int) -> None:
    with pytest.raises(guards.Refusal):
        guards.check_fingerprint(scaled(1.01, (probe, view, component)))


def test_the_fingerprint_is_read_through_the_module_function_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(guards, "probe_xyz", scaled(1.001))
    with pytest.raises(guards.Refusal):
        guards.check_fingerprint()


def test_the_registered_check_runs_the_digest_and_then_the_fingerprint(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(guards, "check_table", lambda *a, **k: calls.append("table"))
    monkeypatch.setattr(guards, "check_fingerprint", lambda *a, **k: calls.append("fingerprint"))
    guards.check_registered()
    assert calls == ["table", "fingerprint"]


def test_the_registered_check_passes_on_this_repository() -> None:
    guards.check_registered()
