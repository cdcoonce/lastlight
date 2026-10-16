"""Renderer v1 against Prague at sun 0: what makes the sky-wide offset (#49).

The four candidate rules are pre-registered in `docs/renderer/prague-offset-criterion.md` (landed
in commit fb2fa6d, before Prague was evaluated at sun -1 or +1). This file is written in the order
the work was done, and the order is evidence:

* the rule functions, on synthetic numbers, every branch and every boundary (this section was
  committed red, then the rule code made it green, before any Prague number at sun -1 or +1
  existed);
* the ozone engine, the node check and the two new fixture specs (also red first, on v0 and v1
  alone: no Prague evaluation);
* the two new fixtures, `tests/fixtures/prague/sun-1.json` and `sun1.json`;
* the rules run on the fixtures, the report, pinned by value.

The rules live in `tools/prague_reference.py` (the tool only, never `lastlight.render`).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import zlib
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from lastlight.g0.collect import build_views
from lastlight.render import Atmosphere
from lastlight.render.curve import sky_xyz
from lastlight.render.v1 import DOMAIN, default_table, multiple_scattering_xyz

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prague_reference.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "prague"


def load_tool() -> ModuleType:
    """The tool as a module, shared with the other Prague test files when it is already loaded."""
    existing = sys.modules.get("prague_reference")
    if existing is not None and getattr(existing, "__file__", None) == str(TOOL):
        return existing
    spec = importlib.util.spec_from_file_location("prague_reference", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("prague_reference", module)
    spec.loader.exec_module(module)
    return module


def above(value: float) -> float:
    """The next float up: a value one unit in the last place over a bound."""
    return math.nextafter(value, math.inf)


def below(value: float) -> float:
    return math.nextafter(value, -math.inf)


def pinned(value: float) -> object:
    """A relative pin: pytest's default absolute tolerance would void it on small values."""
    return pytest.approx(value, rel=1e-6, abs=0)


# --- the registered constants and verdict strings -------------------------------------------------

CARRIES = "carries"
CONTRIBUTES = "contributes"
NOT_THIS = "not this"
CANNOT_TELL = "cannot tell"


def test_the_registered_constants_are_the_criterions() -> None:
    tool = load_tool()
    assert tool.OFFSET_AZIMUTHS_DEG == (90.0, 180.0)  # the point set P
    assert tool.OFFSET_RULED_SUNS_DEG == (-1.0, 1.0)
    assert tool.OFFSET_BAR_FACTOR == 2.0  # the bar is 2m
    assert tool.CARRIES_SHARE == 0.5
    assert tool.FIT_FLOOR == 0.05
    assert tool.DEPTH_ERROR == 0.01  # rule 4's reported range, not a rule
    assert tool.DEEPER_SCALE_HEIGHT_KM == 2.0  # rule 4's reported growth, not a rule


def test_the_slab_factor_is_exp_of_the_observer_height_over_the_aerosol_scale_height() -> None:
    """Rule 2's stand-in: v1 with its `aod_550` raised by exp(0.34 / 1.5) = 1.2544."""
    tool = load_tool()
    expected = math.exp(0.34 / 1.5)
    from_atmosphere = math.exp(tool.OBSERVER_ALTITUDE_KM / Atmosphere(0.0).aerosol_scale_height_km)
    slab = tool.SLAB_FACTOR
    assert slab == expected
    assert slab == from_atmosphere
    assert abs(slab - 1.2544) <= 0.5e-4  # the criterion's four digits


def test_the_verdict_strings_are_the_registered_ones() -> None:
    tool = load_tool()
    assert tool.CARRIES == CARRIES
    assert tool.CONTRIBUTES == CONTRIBUTES
    assert tool.NOT_THIS == NOT_THIS
    assert tool.CANNOT_TELL == CANNOT_TELL  # #45's string, kept


# --- the per-condition verdict --------------------------------------------------------------------
#
# A removed part, an offset and a bar: every boundary is tested on exactly representable values
# (eighths), with no rounding between the number and its bound.


def test_a_part_at_least_half_the_offset_and_over_the_bar_carries() -> None:
    tool = load_tool()
    assert tool.condition_verdict(0.5, 1.0, 0.25) == CARRIES
    assert tool.condition_verdict(0.75, 1.0, 0.25) == CARRIES
    assert tool.condition_verdict(2.0, 1.0, 0.25) == CARRIES  # more than the whole offset


def test_boundary_a_part_exactly_half_the_offset_carries() -> None:
    tool = load_tool()
    assert tool.condition_verdict(0.5, 1.0, 0.25) == CARRIES
    assert tool.condition_verdict(below(0.5), 1.0, 0.25) == CONTRIBUTES  # one ulp under half
    assert tool.condition_verdict(0.25, 0.5, 0.125) == CARRIES  # half of another offset


def test_a_part_over_the_bar_but_under_half_the_offset_contributes() -> None:
    tool = load_tool()
    assert tool.condition_verdict(0.375, 1.0, 0.25) == CONTRIBUTES
    assert tool.condition_verdict(0.3125, 1.0, 0.125) == CONTRIBUTES


def test_boundary_a_part_exactly_at_the_bar_is_not_this_even_if_it_is_half_the_offset() -> None:
    """The bar is strict: removed > bar. A part equal to the bar is not over it."""
    tool = load_tool()
    assert tool.condition_verdict(0.5, 1.0, 0.5) == NOT_THIS  # half the offset, but == bar
    assert tool.condition_verdict(above(0.5), 1.0, 0.5) == CARRIES  # one ulp over: both hold
    assert tool.condition_verdict(0.25, 1.0, 0.25) == NOT_THIS
    assert tool.condition_verdict(above(0.25), 1.0, 0.25) == CONTRIBUTES


def test_a_part_over_half_the_offset_but_not_over_the_bar_is_not_this() -> None:
    tool = load_tool()
    assert tool.condition_verdict(0.75, 1.0, 0.875) == NOT_THIS
    assert tool.condition_verdict(0.75, 1.0, 0.75) == NOT_THIS


def test_a_part_under_the_bar_is_not_this() -> None:
    tool = load_tool()
    assert tool.condition_verdict(0.125, 1.0, 0.25) == NOT_THIS
    assert tool.condition_verdict(below(0.25), 1.0, 0.25) == NOT_THIS


def test_a_part_at_or_under_zero_is_not_this_whatever_the_bar() -> None:
    """The candidate moves the models apart, or does not move them at all."""
    tool = load_tool()
    assert tool.condition_verdict(0.0, 1.0, 0.0) == NOT_THIS  # 0 > 0 is false
    assert tool.condition_verdict(-0.125, 1.0, 0.0) == NOT_THIS
    assert tool.condition_verdict(-0.125, 1.0, 0.25) == NOT_THIS
    assert tool.condition_verdict(-2.0, 1.0, 0.0) == NOT_THIS
    assert tool.condition_verdict(above(0.0), 1.0, 0.0) == CONTRIBUTES  # the smallest positive


def test_the_half_is_of_the_offset_not_of_the_bar() -> None:
    tool = load_tool()
    # removed 0.375 is under half of 1.0 but over half of 0.5: the offset decides.
    assert tool.condition_verdict(0.375, 1.0, 0.125) == CONTRIBUTES
    assert tool.condition_verdict(0.375, 0.5, 0.125) == CARRIES


# --- a rule's verdict over its conditions ---------------------------------------------------------


def test_rule_level_branch_1_carries_when_every_condition_carries() -> None:
    tool = load_tool()
    assert tool.rule_verdict([CARRIES]) == (CARRIES, 1)
    assert tool.rule_verdict([CARRIES, CARRIES]) == (CARRIES, 1)
    assert tool.rule_verdict([CARRIES] * 4) == (CARRIES, 1)


def test_rule_level_branch_2_contributes_when_every_condition_carries_or_contributes() -> None:
    tool = load_tool()
    assert tool.rule_verdict([CONTRIBUTES]) == (CONTRIBUTES, 2)
    assert tool.rule_verdict([CONTRIBUTES, CONTRIBUTES]) == (CONTRIBUTES, 2)
    assert tool.rule_verdict([CARRIES, CONTRIBUTES]) == (CONTRIBUTES, 2)
    assert tool.rule_verdict([CONTRIBUTES, CARRIES, CARRIES, CARRIES]) == (CONTRIBUTES, 2)


def test_rule_level_branch_3_not_this_when_every_condition_is_not_this() -> None:
    tool = load_tool()
    assert tool.rule_verdict([NOT_THIS]) == (NOT_THIS, 3)
    assert tool.rule_verdict([NOT_THIS] * 4) == (NOT_THIS, 3)


def test_rule_level_branch_4_cannot_tell_when_the_conditions_disagree() -> None:
    tool = load_tool()
    assert tool.rule_verdict([CARRIES, NOT_THIS]) == (CANNOT_TELL, 4)
    assert tool.rule_verdict([CONTRIBUTES, NOT_THIS]) == (CANNOT_TELL, 4)
    assert tool.rule_verdict([CARRIES, CONTRIBUTES, NOT_THIS]) == (CANNOT_TELL, 4)
    assert tool.rule_verdict([NOT_THIS, NOT_THIS, NOT_THIS, CARRIES]) == (CANNOT_TELL, 4)


def test_one_not_this_among_carries_and_contributes_is_never_contributes() -> None:
    """Branch 2 needs EVERY condition to carry or contribute: one "not this" voids it."""
    tool = load_tool()
    for rest in ([CARRIES] * 3, [CONTRIBUTES] * 3, [CARRIES, CONTRIBUTES, CARRIES]):
        assert tool.rule_verdict([*rest, NOT_THIS]) == (CANNOT_TELL, 4)


def test_the_rule_level_verdict_does_not_depend_on_the_order_of_the_conditions() -> None:
    tool = load_tool()
    kinds = [CARRIES, CONTRIBUTES, NOT_THIS]
    for size in (1, 2, 3, 4):
        for combination in itertools.product(kinds, repeat=size):
            expected = tool.rule_verdict(list(combination))
            for ordering in set(itertools.permutations(combination)):
                assert tool.rule_verdict(list(ordering)) == expected


def test_every_rule_level_branch_is_reachable_and_the_numbers_are_one_to_four() -> None:
    tool = load_tool()
    branches = {
        tool.rule_verdict(list(combination))[1]
        for size in (1, 2)
        for combination in itertools.product([CARRIES, CONTRIBUTES, NOT_THIS], repeat=size)
    }
    assert branches == {1, 2, 3, 4}


def test_a_rule_level_verdict_over_a_tuple_or_a_generator_is_the_same() -> None:
    tool = load_tool()
    assert tool.rule_verdict((CARRIES, CONTRIBUTES)) == (CONTRIBUTES, 2)
    assert tool.rule_verdict(v for v in (NOT_THIS, NOT_THIS)) == (NOT_THIS, 3)


def test_the_rule_level_verdict_refuses_an_empty_sequence() -> None:
    tool = load_tool()
    with pytest.raises(ValueError, match="no conditions"):
        tool.rule_verdict([])
    with pytest.raises(ValueError, match="no conditions"):
        tool.rule_verdict(())


def test_the_rule_level_verdict_refuses_a_string_that_is_no_condition_verdict() -> None:
    tool = load_tool()
    with pytest.raises(ValueError, match="cannot tell"):
        tool.rule_verdict([CARRIES, CANNOT_TELL])  # a rule's verdict is not a condition's
    with pytest.raises(ValueError, match="carry"):
        tool.rule_verdict(["carry"])


# --- the statistics: bend, reading noise, the fit bar, the offset ---------------------------------


def test_the_bend_is_the_centre_minus_the_mean_of_its_neighbours() -> None:
    tool = load_tool()
    assert tool.bend(1.0, 2.0, 3.0) == 0.0  # on the straight line through the neighbours
    assert tool.bend(0.25, 1.0, 0.75) == 0.5  # 1.0 - (0.25 + 0.75) / 2
    assert tool.bend(0.5, 0.5, 0.5) == 0.0
    assert tool.bend(1.0, 0.125, 1.0) == -0.875  # a dip is negative
    assert tool.bend(0.0, 0.0, 0.5) == -0.25


def test_the_bend_takes_its_arguments_in_the_order_minus_zero_plus() -> None:
    tool = load_tool()
    assert tool.bend(1.0, 4.0, 3.0) == 2.0  # 4 - (1 + 3) / 2
    assert tool.bend(3.0, 4.0, 1.0) == 2.0  # the two neighbours enter symmetrically
    assert tool.bend(0.0, 1.0, 0.0) == 1.0
    assert tool.bend(2.0, 0.0, 0.0) == -1.0  # the centre is the second argument


def test_the_reading_noise_is_the_absolute_difference_of_the_two_readings() -> None:
    tool = load_tool()
    assert tool.reading_noise(0.5, 0.625) == 0.125
    assert tool.reading_noise(0.625, 0.5) == 0.125  # either order
    assert tool.reading_noise(0.5, 0.5) == 0.0
    assert tool.reading_noise(-0.25, 0.25) == 0.5


def test_the_fit_bar_is_twice_the_largest_noise_when_that_is_over_the_floor() -> None:
    tool = load_tool()
    assert tool.fit_bar(0.0, 0.03125, 0.0) == 0.0625  # 2 * 0.03125, over the 0.05 floor
    assert tool.fit_bar(0.125, 0.0, 0.0) == 0.25
    assert tool.fit_bar(0.0, 0.0, 0.25) == 0.5


def test_the_fit_bar_takes_the_largest_of_the_three_noises_wherever_it_is() -> None:
    tool = load_tool()
    for position in range(3):
        m = [0.03125, 0.03125, 0.03125]
        m[position] = 0.125
        assert tool.fit_bar(*m) == 0.25, position


def test_the_fit_bar_never_falls_under_the_floor() -> None:
    tool = load_tool()
    assert tool.fit_bar(0.0, 0.0, 0.0) == 0.05
    assert tool.fit_bar(0.0125, 0.0, 0.01) == 0.05  # 2 * 0.0125 = 0.025 < 0.05
    assert tool.fit_bar(0.0072, 0.0036, 0.0036) == 0.05  # sun 0, 0.03 in the pilot is far under


def test_boundary_the_fit_bar_at_twice_the_noise_equal_to_the_floor() -> None:
    tool = load_tool()
    half = tool.FIT_FLOOR / 2  # exact: halving is exact in binary
    assert tool.fit_bar(half, 0.0, 0.0) == tool.FIT_FLOOR
    assert tool.fit_bar(above(half), 0.0, 0.0) == 2 * above(half)
    assert tool.fit_bar(above(half), 0.0, 0.0) > tool.FIT_FLOOR
    assert tool.fit_bar(below(half), 0.0, 0.0) == tool.FIT_FLOOR


def test_the_offset_is_the_mean_of_the_log_ratio_over_the_points_given() -> None:
    tool = load_tool()
    model = np.array([2.0, 8.0])
    prague = np.array([1.0, 2.0])
    # mean(ln 2, ln 4) = 1.5 ln 2
    assert tool.offset_of(model, prague) == pytest.approx(1.5 * math.log(2.0), rel=1e-15)
    assert tool.offset_of([4.0], [1.0]) == pytest.approx(math.log(4.0), rel=1e-15)


def test_the_offset_is_signed_and_zero_when_the_models_agree() -> None:
    tool = load_tool()
    y = np.array([0.5, 1.0, 2.0, 4.0])
    assert tool.offset_of(y, y) == 0.0
    assert tool.offset_of(y, 2 * y) == pytest.approx(-math.log(2.0), rel=1e-15)
    assert tool.offset_of(2 * y, y) == pytest.approx(math.log(2.0), rel=1e-15)


def test_the_offset_is_a_mean_not_a_root_mean_square() -> None:
    """Logs of opposite sign cancel: the root mean square of these would be ln 2, not 0."""
    tool = load_tool()
    assert tool.offset_of([2.0, 1.0], [1.0, 2.0]) == pytest.approx(0.0, abs=1e-15)


def test_the_offset_refuses_what_it_cannot_take_a_logarithm_of() -> None:
    tool = load_tool()
    good = np.array([1.0, 2.0])
    for bad in (math.nan, math.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and above 0"):
            tool.offset_of(np.array([1.0, bad]), good)
        with pytest.raises(ValueError, match="finite and above 0"):
            tool.offset_of(good, np.array([bad, 1.0]))
    with pytest.raises(ValueError, match="same, non-zero number"):
        tool.offset_of(good, np.array([1.0]))
    with pytest.raises(ValueError, match="same, non-zero number"):
        tool.offset_of([], [])


# --- the refusal: no excess to attribute ----------------------------------------------------------


def test_an_offset_over_zero_is_not_refused() -> None:
    tool = load_tool()
    tool.refuse_no_excess(0.125, "sun 1, aod_550 0.03", "xyz")
    tool.refuse_no_excess(above(0.0), "sun 1, aod_550 0.03", "xyz")


@pytest.mark.parametrize(
    "offset", [0.0, -0.0, -0.125, below(0.0), -1.0, math.nan, math.inf, -math.inf]
)
def test_an_offset_at_or_under_zero_is_refused_naming_the_condition_and_the_reading(
    offset: float,
) -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError) as error:
        tool.refuse_no_excess(offset, "sun -1, aod_550 0.1", "xyz_altitude_0")
    message = str(error.value)
    assert "sun -1, aod_550 0.1" in message
    assert "xyz_altitude_0" in message
    assert "no excess to attribute" in message


# --- the ozone engine and the v0/v1-only helpers --------------------------------------------------
#
# No Prague number is used in this section: the engine is v0's single scattering and v1's table, so
# it is graded on v0 and v1 alone, and on the numbers the criterion printed from v0 and v1 alone
# (its "pilot" section, rule 1's bars, rule 2's stand-in, rule 4's deeper layer).

LOADS = (0.03, 0.1)
ALL_AZIMUTHS = (0.0, 30.0, 90.0, 180.0)
SUNS = (-1.0, 0.0, 1.0, 10.0)


def shown(value: float, printed: float, digits: int) -> bool:
    """The criterion's rule: a printed value is reproduced within half a unit of its last digit."""
    return abs(value - printed) <= 0.5 * 10**-digits


def assert_shown(value: float, printed: float, digits: int, what: str) -> None:
    assert shown(value, printed, digits), f"{what}: the tool gives {value!r}, printed {printed}"


def test_the_half_unit_check_accepts_the_edge_and_nothing_beyond_it() -> None:
    assert shown(0.3022, 0.3022, 4)
    assert shown(0.30224, 0.3022, 4)
    assert not shown(0.30226, 0.3022, 4)
    assert not shown(0.30214, 0.3022, 4)
    assert shown(6.684, 6.68, 2)
    assert not shown(6.686, 6.68, 2)


@cache
def all_views() -> tuple[float, ...]:
    return tuple(load_tool().view_elevations())


def v1_y(aod: float, sun: float, azimuth: float) -> np.ndarray:
    return sky_xyz(Atmosphere(aod_550=aod), sun, list(all_views()), azimuth, engine="v1")[:, 1]


@pytest.mark.parametrize("azimuth", ALL_AZIMUTHS)
@pytest.mark.parametrize("aod", LOADS)
@pytest.mark.parametrize("sun", SUNS)
def test_the_ozone_engine_at_280_du_is_v1_to_1e_12(sun: float, aod: float, azimuth: float) -> None:
    """At v1's own ozone the continuation is v1's own interpolation: |Y_oz(280) / Y_v1 - 1|."""
    tool = load_tool()
    ozone = tool.ozone_xyz(aod, 280.0, sun, list(all_views()), azimuth)
    v1 = sky_xyz(Atmosphere(aod_550=aod), sun, list(all_views()), azimuth, engine="v1")
    assert ozone.shape == v1.shape == (len(all_views()), 3)
    assert np.abs(ozone[:, 1] / v1[:, 1] - 1.0).max() <= 1e-12
    assert np.abs(ozone / v1 - 1.0).max() <= 1e-12  # X and Z too


@pytest.mark.parametrize("ozone_du", [250.0, 330.0])
def test_the_ozone_engine_at_the_tables_own_nodes_is_v1_at_those_columns(ozone_du: float) -> None:
    tool = load_tool()
    views_ = list(all_views())
    for sun, aod, azimuth in ((0.0, 0.1, 90.0), (-1.0, 0.03, 0.0), (1.0, 0.1, 180.0)):
        ozone = tool.ozone_xyz(aod, ozone_du, sun, views_, azimuth)
        v1 = sky_xyz(Atmosphere(aod_550=aod, ozone_du=ozone_du), sun, views_, azimuth, engine="v1")
        assert np.abs(ozone / v1 - 1.0).max() <= 1e-12


def test_the_ozone_engine_is_v0_plus_the_tables_log_linear_continuation() -> None:
    """Per channel exp((1 - t) ln MS_250 + t ln MS_330), t = (O - 250) / 80, plus v0 at O."""
    tool = load_tool()
    views_ = list(all_views())
    sun, aod, azimuth = 0.0, 0.1, 90.0
    for ozone_du in (270.0, 345.0):
        t = (ozone_du - 250.0) / 80.0
        single = sky_xyz(Atmosphere(aod_550=aod, ozone_du=ozone_du), sun, views_, azimuth)
        low = multiple_scattering_xyz(Atmosphere(aod_550=aod, ozone_du=250.0), sun, views_, azimuth)
        high = multiple_scattering_xyz(
            Atmosphere(aod_550=aod, ozone_du=330.0), sun, views_, azimuth
        )
        expected = single + np.exp((1.0 - t) * np.log(low) + t * np.log(high))
        got = tool.ozone_xyz(aod, ozone_du, sun, views_, azimuth)
        assert np.abs(got / expected - 1.0).max() <= 1e-13


def test_at_345_du_the_engine_continues_v1s_own_line_15_du_past_the_last_node() -> None:
    tool = load_tool()
    views_ = list(all_views())
    sun, aod, azimuth = 1.0, 0.03, 180.0
    single = sky_xyz(Atmosphere(aod_550=aod, ozone_du=345.0), sun, views_, azimuth)
    multiple = tool.ozone_xyz(aod, 345.0, sun, views_, azimuth) - single
    low = multiple_scattering_xyz(Atmosphere(aod_550=aod, ozone_du=250.0), sun, views_, azimuth)
    high = multiple_scattering_xyz(Atmosphere(aod_550=aod, ozone_du=330.0), sun, views_, azimuth)
    # The slope of ln MS per DU between the nodes, carried on 15 DU past the last one.
    line = np.log(high) + 15.0 * (np.log(high) - np.log(low)) / 80.0
    assert np.abs(np.log(multiple) - line).max() <= 1e-9
    assert (multiple > 0.0).all()


def test_the_engine_never_asks_v1_for_an_ozone_outside_its_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1 refuses 345 DU; the engine reads the table at its two nodes and v0 for the rest."""
    tool = load_tool()
    asked: list[float] = []
    engines: list[tuple[str, dict]] = []
    real_multiple, real_sky = tool.multiple_scattering_xyz, tool.sky_xyz

    def record_multiple(atm, *args, **kwargs):
        asked.append(atm.ozone_du)
        return real_multiple(atm, *args, **kwargs)

    def record_sky(atm, *args, **kwargs):
        engines.append((kwargs.get("engine", "v0"), dict(kwargs)))
        return real_sky(atm, *args, **kwargs)

    monkeypatch.setattr(tool, "multiple_scattering_xyz", record_multiple)
    monkeypatch.setattr(tool, "sky_xyz", record_sky)
    tool.ozone_xyz(0.1, 345.0, 0.0, list(all_views()), 90.0)
    assert sorted(asked) == [250.0, 330.0]
    assert [engine for engine, _ in engines] == ["v0"]
    assert all(kwargs.get("quadrature", "registered") == "registered" for _, kwargs in engines)
    with pytest.raises(ValueError, match="outside renderer v1's table"):
        sky_xyz(Atmosphere(aod_550=0.1, ozone_du=345.0), 0.0, [2.0], 0.0, engine="v1")


def test_the_ozone_engine_asks_the_table_for_the_nodes_the_domain_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nodes are read from the domain, and the table is asked at exactly those columns.

    The continuation is log-linear in ozone, so any two nodes give the same value inside the
    table (the two readings differ by 1e-15). Only the columns the engine asks the table for tell
    a domain read from two numbers written into the code, so they are recorded.
    """
    tool = load_tool()
    assert DOMAIN["ozone_du"] == (250.0, 330.0)  # v1 is unchanged: its domain is registered
    views_ = list(all_views())
    asked: list[float] = []
    real_multiple = tool.multiple_scattering_xyz

    def record_multiple(atm, *args, **kwargs):
        asked.append(atm.ozone_du)
        return real_multiple(atm, *args, **kwargs)

    monkeypatch.setattr(tool, "DOMAIN", {"ozone_du": (260.0, 320.0)})
    monkeypatch.setattr(tool, "multiple_scattering_xyz", record_multiple)
    got = tool.ozone_xyz(0.1, 290.0, 0.0, views_, 90.0)
    assert sorted(asked) == [260.0, 320.0]
    t = (290.0 - 260.0) / 60.0
    low = real_multiple(Atmosphere(aod_550=0.1, ozone_du=260.0), 0.0, views_, 90.0)
    high = real_multiple(Atmosphere(aod_550=0.1, ozone_du=320.0), 0.0, views_, 90.0)
    single = sky_xyz(Atmosphere(aod_550=0.1, ozone_du=290.0), 0.0, views_, 90.0)
    expected = single + np.exp((1.0 - t) * np.log(low) + t * np.log(high))
    assert np.abs(got / expected - 1.0).max() <= 1e-13


def test_more_ozone_means_less_light_at_every_view_and_never_changes_the_geometry() -> None:
    tool = load_tool()
    views_ = list(all_views())
    base = tool.ozone_xyz(0.1, 280.0, 0.0, views_, 90.0)
    more = tool.ozone_xyz(0.1, 345.0, 0.0, views_, 90.0)
    assert more.shape == base.shape
    assert (more[:, 1] < base[:, 1]).all()


# --- Delta_O, the line check and v1's own sky level: pinned to the criterion's printed numbers ----

OZONE_PART_PINS = {  # (sun, aod_550): Delta_O to four digits, from v0 and v1 alone
    (-1.0, 0.03): 0.1084,
    (-1.0, 0.1): 0.1075,
    (0.0, 0.03): 0.0963,
    (0.0, 0.1): 0.0951,
    (1.0, 0.03): 0.0847,
    (1.0, 0.1): 0.0832,
    (10.0, 0.03): 0.0294,
    (10.0, 0.1): 0.0292,
}


@cache
def ozone_part(sun: float, aod: float) -> float:
    return load_tool().ozone_part(aod, sun, list(all_views()), load_tool().OFFSET_AZIMUTHS_DEG)


@pytest.mark.parametrize(("sun", "aod"), sorted(OZONE_PART_PINS))
def test_delta_o_over_p_is_the_criterions_from_v0_and_v1_alone(sun: float, aod: float) -> None:
    assert_shown(ozone_part(sun, aod), OZONE_PART_PINS[sun, aod], 4, f"Delta_O ({sun}, {aod})")


def test_delta_o_is_positive_at_every_condition_it_is_less_light_with_more_ozone() -> None:
    assert all(ozone_part(sun, aod) > 0.0 for sun, aod in OZONE_PART_PINS)


def test_delta_o_is_the_mean_log_ratio_of_v1_to_the_engine_at_345_du_over_p() -> None:
    tool = load_tool()
    views_ = list(all_views())
    logs = []
    for azimuth in (90.0, 180.0):
        v1 = sky_xyz(Atmosphere(aod_550=0.1), 0.0, views_, azimuth, engine="v1")[:, 1]
        engine = tool.ozone_xyz(0.1, tool.US_STANDARD_OZONE_DU, 0.0, views_, azimuth)[:, 1]
        logs.extend(np.log(v1 / engine))
    assert len(logs) == 16
    assert ozone_part(0.0, 0.1) == pytest.approx(float(np.mean(logs)), rel=1e-12)


def test_the_helpers_pick_points_by_value_not_by_position() -> None:
    tool = load_tool()
    views_ = list(all_views())
    forward = tool.ozone_part(0.1, 0.0, views_, (90.0, 180.0))
    swapped = tool.ozone_part(0.1, 0.0, views_[::-1], (180.0, 90.0))
    assert swapped == pytest.approx(forward, rel=1e-12, abs=0)
    one = tool.ozone_part(0.1, 0.0, views_, (90.0,))
    both = tool.ozone_part(0.1, 0.0, views_, (90.0, 180.0))
    assert one != both  # the azimuth set is what it is told, not a constant


def test_the_helpers_refuse_an_empty_set_of_azimuths() -> None:
    tool = load_tool()
    with pytest.raises(ValueError, match="azimuth"):
        tool.ozone_part(0.1, 0.0, list(all_views()), ())


V0_LINE_PINS = {  # (sun, aod_550): the ozone line check, four digits
    (0.0, 0.03): 1.2963,
    (0.0, 0.1): 1.2960,
    (10.0, 0.03): 1.2992,
    (10.0, 0.1): 1.2993,
}


@pytest.mark.parametrize(("sun", "aod"), sorted(V0_LINE_PINS))
def test_the_ozone_line_check_is_the_criterions_and_close_to_a_straight_line(
    sun: float, aod: float
) -> None:
    """mean ln(Y_v0(280) / Y_v0(345)) over mean ln(Y_v0(280) / Y_v0(330)); a line gives 65 / 50."""
    tool = load_tool()
    ratio = tool.ozone_line_ratio(aod, sun, list(all_views()), tool.OFFSET_AZIMUTHS_DEG)
    assert_shown(ratio, V0_LINE_PINS[sun, aod], 4, f"line check ({sun}, {aod})")
    assert abs(ratio - 65.0 / 50.0) < 0.01


def test_the_single_scattering_ozone_part_is_v0s_own_log_ratio_over_p() -> None:
    tool = load_tool()
    views_ = list(all_views())
    logs = []
    for azimuth in (90.0, 180.0):
        at_280 = sky_xyz(Atmosphere(aod_550=0.03, ozone_du=280.0), 1.0, views_, azimuth)[:, 1]
        at_345 = sky_xyz(Atmosphere(aod_550=0.03, ozone_du=345.0), 1.0, views_, azimuth)[:, 1]
        logs.extend(np.log(at_280 / at_345))
    got = tool.ozone_part_single_scattering(0.03, 1.0, views_, (90.0, 180.0))
    assert got == pytest.approx(float(np.mean(logs)), rel=1e-12)
    assert 0.0 < got < ozone_part(1.0, 0.03) * 2.0  # v1's part is the same order, not unrelated


V1_BEND_PINS = {0.03: 0.0996, 0.1: 0.0543}  # v1's own sky: sun 0 above the line through -1 and +1


@pytest.mark.parametrize("aod", LOADS)
def test_v1s_own_sky_bends_at_sun_0_by_the_criterions_numbers(aod: float) -> None:
    tool = load_tool()
    level = {
        sun: tool.v1_level(aod, sun, list(all_views()), tool.OFFSET_AZIMUTHS_DEG)
        for sun in (-1.0, 0.0, 1.0)
    }
    got = tool.bend(level[-1.0], level[0.0], level[1.0])
    assert_shown(got, V1_BEND_PINS[aod], 4, f"v1 bend at {aod}")


def test_v1_level_is_the_mean_of_ln_y_over_the_points() -> None:
    tool = load_tool()
    logs = np.log(np.concatenate([v1_y(0.1, 0.0, azimuth) for azimuth in (90.0, 180.0)]))
    got = tool.v1_level(0.1, 0.0, list(all_views()), (90.0, 180.0))
    assert got == pytest.approx(float(np.mean(logs)), rel=1e-12)


# --- rule 2's stand-in and rule 4's deeper layer: from v1 or v0 alone, reported, never ruled ----

SLAB_PINS = {
    (0.0, 0.03): -0.0684,
    (0.0, 0.1): -0.0894,
    (10.0, 0.03): -0.0265,
    (10.0, 0.1): -0.0597,
}


@pytest.mark.parametrize(("sun", "aod"), sorted(SLAB_PINS))
def test_the_slab_stand_in_moves_the_mean_log_luminance_by_the_criterions_numbers(
    sun: float, aod: float
) -> None:
    tool = load_tool()
    got = tool.slab_change(aod, sun, list(all_views()), tool.OFFSET_AZIMUTHS_DEG)
    assert_shown(got, SLAB_PINS[sun, aod], 4, f"slab stand-in ({sun}, {aod})")


def test_the_slab_stand_in_is_v1_with_its_aod_raised_by_the_slab_factor() -> None:
    tool = load_tool()
    sun, aod = 0.0, 0.1
    raised = np.concatenate([v1_y(aod * tool.SLAB_FACTOR, sun, az) for az in (90.0, 180.0)])
    base = np.concatenate([v1_y(aod, sun, az) for az in (90.0, 180.0)])
    got = tool.slab_change(aod, sun, list(all_views()), (90.0, 180.0))
    assert got == pytest.approx(float(np.mean(np.log(raised) - np.log(base))), rel=1e-12)


def deeper(sun: float, aod: float) -> float:
    tool = load_tool()
    return tool.deeper_layer_change(aod, sun, list(all_views()), tool.OFFSET_AZIMUTHS_DEG)


def test_the_deeper_layer_changes_the_mean_by_the_criterions_numbers_at_aod_0_1() -> None:
    assert_shown(deeper(10.0, 0.1), 0.0405, 4, "deeper layer (sun +10, 0.1)")
    assert_shown(deeper(0.0, 0.1), -0.0576, 4, "deeper layer (sun 0, 0.1)")


@pytest.mark.parametrize(("sun", "printed"), [(-1.0, 3.31), (0.0, 3.68), (1.0, 1.89)])
def test_the_deeper_layer_grows_from_0_03_to_0_1_by_the_criterions_factors(
    sun: float, printed: float
) -> None:
    assert_shown(deeper(sun, 0.1) / deeper(sun, 0.03), printed, 2, f"growth 0.03 to 0.1, sun {sun}")


@pytest.mark.parametrize(("sun", "printed"), [(-1.0, 1.41), (0.0, 1.35), (1.0, 0.80)])
def test_the_deeper_layer_grows_from_0_1_to_0_2_by_the_criterions_factors(
    sun: float, printed: float
) -> None:
    assert_shown(deeper(sun, 0.2) / deeper(sun, 0.1), printed, 2, f"growth 0.1 to 0.2, sun {sun}")


def test_the_deeper_layer_is_v0s_single_scattering_at_two_scale_heights() -> None:
    tool = load_tool()
    views_ = list(all_views())
    logs = []
    for azimuth in (90.0, 180.0):
        deep = sky_xyz(
            Atmosphere(aod_550=0.1, aerosol_scale_height_km=tool.DEEPER_SCALE_HEIGHT_KM),
            0.0,
            views_,
            azimuth,
        )[:, 1]
        base = sky_xyz(Atmosphere(aod_550=0.1), 0.0, views_, azimuth)[:, 1]
        logs.extend(np.log(deep / base))
    assert deeper(0.0, 0.1) == pytest.approx(float(np.mean(logs)), rel=1e-12)


# --- the two new fixture specs --------------------------------------------------------------------


def test_the_sun_minus_1_spec_is_the_registered_one() -> None:
    tool = load_tool()
    spec = tool.SUN_MINUS1
    assert spec.sun_elevation_deg == -1.0
    assert spec.rel_azimuths_deg == (0.0, 30.0, 90.0, 180.0)
    assert spec.path == FIXTURE_DIR / "sun-1.json" == tool.FIXTURE_SUN_MINUS1
    assert spec.command == "uv run python tools/prague_reference.py fixture --sun -1"
    assert spec.sun_must_be_node is True
    assert tool.fixture_spec(-1.0) is spec
    assert tool.fixture_spec(-1) is spec  # the command line says `--sun -1`


def test_the_sun_plus_1_spec_is_the_registered_one() -> None:
    tool = load_tool()
    spec = tool.SUN_PLUS1
    assert spec.sun_elevation_deg == 1.0
    assert spec.rel_azimuths_deg == (0.0, 30.0, 90.0, 180.0)
    assert spec.path == FIXTURE_DIR / "sun1.json" == tool.FIXTURE_SUN_PLUS1
    assert spec.command == "uv run python tools/prague_reference.py fixture --sun 1"
    assert spec.sun_must_be_node is True
    assert tool.fixture_spec(1.0) is spec
    assert tool.fixture_spec(1) is spec


def test_four_fixtures_are_registered_and_the_two_old_specs_are_what_they_were() -> None:
    tool = load_tool()
    assert set(tool.FIXTURE_SPECS) == {10.0, 0.0, -1.0, 1.0}
    assert tool.FIXTURE_SPECS[10.0] is tool.SUN10 and tool.FIXTURE_SPECS[0.0] is tool.SUN0
    assert tool.SUN10.sun_must_be_node is False  # sun +10 is not a node and must keep building
    assert tool.SUN0.sun_must_be_node is False
    assert tool.SUN10.rel_azimuths_deg == (0.0, 90.0, 180.0)
    assert tool.SUN10.command == "uv run python tools/prague_reference.py fixture"
    assert tool.SUN0.command == "uv run python tools/prague_reference.py fixture --sun 0"
    assert tool.SUN0.path == FIXTURE_DIR / "sun0.json" and tool.SUN10.path == tool.FIXTURE


def test_a_spec_made_without_the_flag_does_not_require_a_node() -> None:
    tool = load_tool()
    assert tool.FixtureSpec(5.0, (0.0,), Path("x.json"), "n/a").sun_must_be_node is False


def test_moving_a_registered_fixture_keeps_its_node_requirement() -> None:
    tool = load_tool()
    moved = tool.fixture_spec(-1.0, ROOT / "elsewhere.json")
    assert moved.path == ROOT / "elsewhere.json"
    assert moved.sun_must_be_node is True
    assert moved.command == tool.SUN_MINUS1.command
    assert tool.fixture_spec(0.0, ROOT / "elsewhere.json").sun_must_be_node is False


def test_the_loads_stay_the_two_of_the_earlier_fixtures_there_is_no_0_2() -> None:
    tool = load_tool()
    assert tool.AEROSOL_OPTICAL_DEPTHS == (0.03, 0.1)


def test_the_command_line_takes_a_negative_sun() -> None:
    tool = load_tool()
    parser = tool.build_parser()
    assert parser.parse_args(["fixture", "--sun", "-1"]).sun == -1.0
    assert parser.parse_args(["fixture", "--sun", "1"]).sun == 1.0
    assert parser.parse_args(["fixture", "--sun", "-1", "--check"]).check is True


def test_fixture_check_for_the_sun_minus_1_spec_reaches_the_checker_not_the_parser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    nothing = tmp_path / "nothing.json"
    assert tool.main(["fixture", "--sun", "-1", "--check", "--output", str(nothing)]) == 1
    assert "does not exist: nothing to check" in capsys.readouterr().err
    assert not nothing.exists()  # a check writes nothing


# --- the node check ------------------------------------------------------------------------------

PRAGUE_AXIS = (-4.2, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0, 2.85, 5.23)


def test_a_sun_that_is_a_node_passes() -> None:
    tool = load_tool()
    for sun in (-1.0, 0.0, 1.0):
        tool.check_sun_node(sun, PRAGUE_AXIS, "Prague's sun axis")


def test_a_sun_that_is_not_a_node_is_refused_naming_the_axis() -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match=r"Prague's sun axis") as error:
        tool.check_sun_node(10.0, PRAGUE_AXIS, "Prague's sun axis")
    assert "10" in str(error.value) and "not a node" in str(error.value)
    with pytest.raises(tool.PragueError, match=r"renderer v1's table sun axis"):
        tool.check_sun_node(0.5, PRAGUE_AXIS, "renderer v1's table sun axis")


def test_boundary_a_node_is_matched_to_1e_9_and_no_looser() -> None:
    tool = load_tool()
    tool.check_sun_node(1.0 + 5e-10, PRAGUE_AXIS, "axis")
    tool.check_sun_node(1.0 - 5e-10, PRAGUE_AXIS, "axis")
    with pytest.raises(tool.PragueError, match="not a node"):
        tool.check_sun_node(1.0 + 2e-9, PRAGUE_AXIS, "axis")
    with pytest.raises(tool.PragueError, match="not a node"):
        tool.check_sun_node(1.0 - 2e-9, PRAGUE_AXIS, "axis")


def test_a_sun_that_is_not_finite_is_not_a_node() -> None:
    tool = load_tool()
    for sun in (math.nan, math.inf, -math.inf):
        with pytest.raises(tool.PragueError, match="not a node"):
            tool.check_sun_node(sun, PRAGUE_AXIS, "axis")


def test_the_real_axes_have_minus_1_0_and_1_as_nodes_and_10_is_a_node_of_v1_only() -> None:
    """The reason for the check: sun 10 interpolates in Prague, ±1 do not interpolate anywhere."""
    recorded = json.loads((FIXTURE_DIR / "sun0.json").read_text())
    prague = recorded["provenance"]["dataset_axes"]["elevations_deg"]
    v1 = [float(x) for x in default_table().axes["sun_elevation_deg"]]
    for sun in (-1.0, 0.0, 1.0):
        assert sun in prague and sun in v1
    assert 10.0 in v1 and 10.0 not in prague
    assert 0.5 not in v1  # -0.5 is a Prague node but not v1's, which is why it was not ruled at


def stand_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, elevations=None) -> list[dict]:
    """Stand-ins for everything `build_fixture` reaches for; returns the log of `evaluate` calls.

    No Prague model is run: `fetch_model`, `read_axes`, `evaluate` and the dataset table are
    replaced by deterministic stand-ins that record what they were asked, so these tests grade the
    arguments the builder threads through (the spec's sun and azimuths, the sun axes), not Prague.
    """
    tool = load_tool()
    recorded = json.loads((FIXTURE_DIR / "sun0.json").read_text())
    axes_info = recorded["provenance"]["dataset_axes"]
    centres = recorded["channel_centres_nm"]
    width = centres[1] - centres[0]
    axes = tool.Axes(
        tuple(axes_info["visibilities_km"]),
        tuple(axes_info["albedos"]),
        tuple(axes_info["altitudes_m"]),
        tuple(elevations if elevations is not None else axes_info["elevations_deg"]),
        len(centres),
        centres[0] - width / 2,
        width,
    )
    payload = b"stand-in dataset"
    datasets = {
        key: tool.Dataset(ds.name, ds.drive_id, len(payload), hashlib.sha256(payload).hexdigest())
        for key, ds in tool.DATASETS.items()
    }
    (tmp_path / "dl").mkdir()
    for dataset in datasets.values():
        (tmp_path / "dl" / dataset.name).write_bytes(payload)
    calls: list[dict] = []

    def evaluate(cache, dataset, *, visibility, altitude_m, queries, sun_elevation_deg, mode="sky"):
        calls.append({"sun": sun_elevation_deg, "queries": list(queries), "mode": mode})
        rows = []
        for azimuth, view in queries:
            seed = repr((round(visibility, 6), altitude_m, sun_elevation_deg, azimuth, view, mode))
            if mode == "transmittance":
                rows.append(np.exp(-(0.3 + 5.0 / visibility)) * np.ones(len(centres)))
            else:
                noise = np.random.default_rng(zlib.crc32(seed.encode())).random(len(centres))
                rows.append(0.5 + noise)
        return tool.Evaluation(np.array(rows), axes.channel_start_nm, axes.channel_width_nm)

    monkeypatch.setattr(tool, "DATASETS", datasets)
    monkeypatch.setattr(tool, "fetch_model", lambda cache: cache)
    monkeypatch.setattr(tool, "read_axes", lambda path: axes)
    monkeypatch.setattr(tool, "evaluate", evaluate)
    return calls


@pytest.mark.parametrize("spec_name", ["SUN_MINUS1", "SUN_PLUS1"])
def test_the_builder_builds_a_node_sun_at_that_sun_and_the_four_azimuths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec_name: str
) -> None:
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)
    spec = getattr(tool, spec_name)
    data = tool.build_fixture(tmp_path, spec)
    assert data["sun_elevation_deg"] == spec.sun_elevation_deg
    assert data["rel_azimuths_deg"] == [0.0, 30.0, 90.0, 180.0]
    assert data["provenance"]["command"] == spec.command
    assert set(data["atmospheres"]) == {"0.03", "0.1"}
    assert calls and {call["sun"] for call in calls} == {spec.sun_elevation_deg}
    for entry in data["atmospheres"].values():
        assert list(entry["xyz"]) == ["0", "30", "90", "180"]


def test_the_builder_refuses_a_sun_that_is_not_a_node_of_pragues_axis_before_asking_it_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)
    spec = tool.FixtureSpec(10.0, (0.0, 90.0), tmp_path / "x.json", "n/a", sun_must_be_node=True)
    with pytest.raises(tool.PragueError, match=r"Prague's sun axis"):
        tool.build_fixture(tmp_path, spec)
    assert calls == []


def test_the_builder_refuses_a_sun_that_is_a_prague_node_but_not_a_node_of_v1s_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """-0.5 is a node of Prague's sun axis but not of v1's, so v1 would interpolate there."""
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)
    spec = tool.FixtureSpec(-0.5, (0.0, 90.0), tmp_path / "x.json", "n/a", sun_must_be_node=True)
    with pytest.raises(tool.PragueError, match=r"renderer v1's table sun axis"):
        tool.build_fixture(tmp_path, spec)
    assert calls == []


def test_the_builder_reads_v1s_sun_axis_from_the_table_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A table whose sun axis lacked -1 would be refused, whatever Prague's axis says."""
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)

    class Table:
        axes = {"sun_elevation_deg": np.array([-12.0, 0.0, 12.0])}

    monkeypatch.setattr(tool, "default_table", lambda: Table())
    with pytest.raises(tool.PragueError, match=r"renderer v1's table sun axis"):
        tool.build_fixture(tmp_path, tool.SUN_MINUS1)
    assert calls == []


def test_the_default_builds_a_sun_that_is_no_node_so_sun_10_keeps_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)
    data = tool.build_fixture(tmp_path)  # the default spec is sun +10, which is no node
    assert data["sun_elevation_deg"] == 10.0
    assert data["rel_azimuths_deg"] == [0.0, 90.0, 180.0]
    assert {call["sun"] for call in calls} == {10.0}
    tool.build_fixture(tmp_path, tool.SUN0)  # sun 0 is a node, but is not required to be


def test_a_node_sun_outside_the_datasets_axis_is_still_refused_by_the_range_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = stand_in(monkeypatch, tmp_path)
    spec = tool.FixtureSpec(95.0, (0.0,), tmp_path / "x.json", "n/a", sun_must_be_node=True)
    with pytest.raises(tool.PragueError, match="outside the dataset's sun axis"):
        tool.build_fixture(tmp_path, spec)
    assert calls == []


# --- the two new fixtures -------------------------------------------------------------------------
#
# `sun-1.json` and `sun1.json` were written by the tool after the rule functions, the ozone engine
# and the node check were committed: `fixture --sun -1` and `fixture --sun 1`.

FIXTURE_FILES = {10.0: "sun10.json", 0.0: "sun0.json", -1.0: "sun-1.json", 1.0: "sun1.json"}
FOUR_SUNS = (10.0, 0.0, -1.0, 1.0)
READINGS = (
    "xyz_binned",
    "xyz_inverse_visibility",
    "xyz_altitude_0",
    "xyz_matched_aerosol_depth",
)
"""The fixture's diagnostic readings of Prague, each laid out like the primary `xyz`."""
ALL_READINGS = ("xyz", *READINGS)
NOMINAL_VIEWS = 6
CACHE = Path.home() / ".cache" / "lastlight" / "prague"


@cache
def committed(sun: float) -> dict:
    return json.loads((FIXTURE_DIR / FIXTURE_FILES[sun]).read_text())


def fresh_fixtures() -> dict[float, dict]:
    """Copies of the four committed fixtures, safe to edit."""
    return {sun: copy.deepcopy(committed(sun)) for sun in FOUR_SUNS}


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixture_has_the_documented_layout(sun: float) -> None:
    data = committed(sun)
    assert data["format"] == "lastlight-prague-control"
    assert data["version"] == 1
    assert data["sun_elevation_deg"] == sun  # the fixture records its own sun
    assert data["observer_altitude_km"] == 0.34
    assert data["ground_albedo"] == 0.2
    assert [float(a) for a in data["rel_azimuths_deg"]] == [0.0, 30.0, 90.0, 180.0]
    views_ = data["view_elevations_deg"]
    assert len(views_) == 8 and views_[NOMINAL_VIEWS:] == [30.0, 60.0]
    assert set(data["atmospheres"]) == {"0.03", "0.1"}  # there is no third load
    for aod in LOADS:
        assert data["atmospheres"][f"{aod:g}"]["aod_550"] == aod
        entry = data["atmospheres"][f"{aod:g}"]
        for variant in ALL_READINGS:
            block = entry["xyz"] if variant == "xyz" else entry["diagnostics"][variant]
            assert list(block) == ["0", "30", "90", "180"]
            for key in block:
                xyz = np.array(block[key], dtype=float)
                assert xyz.shape == (8, 3), (aod, key, variant)
                assert np.isfinite(xyz).all() and (xyz > 0.0).all(), (aod, key, variant)


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixture_has_every_key_sun_0_has(sun: float) -> None:
    new, old = committed(sun), committed(0.0)
    assert new.keys() == old.keys()
    for aod in ("0.03", "0.1"):
        assert new["atmospheres"][aod].keys() == old["atmospheres"][aod].keys()
        assert (
            new["atmospheres"][aod]["diagnostics"].keys()
            == old["atmospheres"][aod]["diagnostics"].keys()
        )


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixtures_views_are_the_cameras_nominal_views(sun: float) -> None:
    """A refit of WHTM1 moves its bands: the control must be regenerated, not left stale."""
    nominal = build_views()["nominal"].elevations_deg
    recorded = committed(sun)["view_elevations_deg"][:NOMINAL_VIEWS]
    assert recorded == pytest.approx(list(nominal), abs=1e-9), (
        "the camera's nominal band and reference elevations changed since the fixture was made: "
        f"rerun tools/prague_reference.py fixture --sun {sun:g}"
    )
    assert committed(sun)["view_elevations_deg"] == committed(10.0)["view_elevations_deg"]


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixture_maps_aod_to_visibility_as_the_others_do(sun: float) -> None:
    """Only the sun changed: Koschmieder at 550 nm, v1's own ground extinction."""
    tool = load_tool()
    for aod in LOADS:
        entry = committed(sun)["atmospheres"][f"{aod:g}"]
        assert entry["visibility_km"] == pytest.approx(
            tool.visibility_km(Atmosphere(aod_550=aod)), rel=1e-9
        )
        assert entry["visibility_km"] == committed(10.0)["atmospheres"][f"{aod:g}"]["visibility_km"]


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixtures_matched_reading_is_clamped_at_0_03_and_not_at_0_1(sun: float) -> None:
    thin = committed(sun)["atmospheres"]["0.03"]["diagnostics"]["matched_aerosol_depth"]
    assert thin["clamped_at_largest_visibility"] is True and thin["visibility_km"] == 131.8
    thick = committed(sun)["atmospheres"]["0.1"]["diagnostics"]["matched_aerosol_depth"]
    assert thick["clamped_at_largest_visibility"] is False
    assert thick["prague_aerosol_depth"] == pytest.approx(0.1, abs=1e-4)


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixture_is_the_same_model_and_data_as_the_others(sun: float) -> None:
    tool = load_tool()
    new, old = committed(sun)["provenance"], committed(10.0)["provenance"]
    assert new["model"] == old["model"]
    assert new["datasets"] == old["datasets"]
    assert new["dataset_axes"] == old["dataset_axes"]
    assert new["ground_level_dataset_matches_full_at_altitude_0"] is True
    assert new["command"] == tool.fixture_spec(sun).command
    assert len(new["model"]["commit"]) == 40 and len(new["datasets"]["full"]["sha256"]) == 64


@pytest.mark.parametrize("sun", [-1.0, 1.0])
def test_a_new_fixtures_sun_is_a_node_of_pragues_axis_and_of_v1s_table(sun: float) -> None:
    """So neither model interpolates in the sun there, as at sun 0."""
    tool = load_tool()
    prague = committed(sun)["provenance"]["dataset_axes"]["elevations_deg"]
    tool.check_sun_node(sun, prague, "Prague's sun axis")
    tool.check_sun_node(sun, default_table().axes["sun_elevation_deg"], "v1's table sun axis")


def test_the_two_old_fixtures_are_the_ones_the_earlier_issues_made() -> None:
    assert committed(10.0)["sun_elevation_deg"] == 10.0
    assert committed(10.0)["rel_azimuths_deg"] == [0.0, 90.0, 180.0]
    assert (
        committed(10.0)["provenance"]["command"]
        == "uv run python tools/prague_reference.py fixture"
    )
    assert committed(0.0)["sun_elevation_deg"] == 0.0
    assert committed(0.0)["provenance"]["command"].endswith("fixture --sun 0")


@pytest.mark.parametrize("sun", FOUR_SUNS)
@pytest.mark.parametrize("aod", LOADS)
def test_the_ozone_engine_at_280_du_is_v1_at_every_point_of_every_fixture(
    sun: float, aod: float
) -> None:
    """|Y_oz(280) / Y_v1 - 1| <= 1e-12 at every stored view and relative azimuth."""
    tool = load_tool()
    data = committed(sun)
    views_ = data["view_elevations_deg"]
    for azimuth in data["rel_azimuths_deg"]:
        ozone = tool.ozone_xyz(aod, 280.0, sun, views_, azimuth)[:, 1]
        v1 = sky_xyz(Atmosphere(aod_550=aod), sun, views_, azimuth, engine="v1")[:, 1]
        assert np.abs(ozone / v1 - 1.0).max() <= 1e-12, (sun, aod, azimuth)


@pytest.mark.network
@pytest.mark.parametrize("sun", ["-1", "1"])
def test_a_new_fixture_matches_a_fresh_evaluation_of_the_built_model(sun: str) -> None:
    """`tools/prague_reference.py fixture --sun N --check` re-evaluates Prague and compares."""
    if not (CACHE / "build" / "prague_eval").exists():
        pytest.skip("the Prague model is not built: run tools/prague_reference.py fetch build")
    result = subprocess.run(
        [sys.executable, str(TOOL), "fixture", "--sun", sun, "--check"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_provenance_note_names_all_four_fixtures_and_the_two_new_commands() -> None:
    text = (FIXTURE_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    for name in FIXTURE_FILES.values():
        assert f"`{name}`" in text, name
    for command in (
        "uv run python tools/prague_reference.py fixture --sun -1",
        "uv run python tools/prague_reference.py fixture --sun 1",
        "uv run python tools/prague_reference.py report --offset",
    ):
        assert command in text, command


# --- the rules run on the fixtures, pinned by value -----------------------------------------------
#
# Every pin is `pytest.approx(x, rel=1e-6, abs=0)`: a relative bar. (pytest's default absolute
# tolerance of 1e-12 would void a relative bar on a value this small, so abs is always 0.) The
# numbers were first computed after the rules and both new fixtures were committed, and are
# reproduced by `uv run python tools/prague_reference.py report --offset`. A pin that moves means
# the physics, a fixture or a rule moved: it is not a number to be updated until the reason is
# known.

CONDITION_PINS = {  # (sun, aod_550): the quantities over P, first computed after the fixtures
    (10.0, 0.03): {
        "L": 0.10058602433842617,
        "L_A": 0.11940190714347143,
        "L_inv": 0.09697907095159833,
        "L_binned": 0.12149864916911769,
        "L_matched": 0.08753935378367879,
        "delta_o": 0.029429443852198534,
        "delta_o_v0": 0.029680582746008134,
        "line": 1.2991741194615078,
        "slab": -0.02650692317658665,
        "deeper": 0.016587881434686036,
        "depth": 0.04071429226692963,
        "depth_matched": 0.036851,
    },
    (10.0, 0.1): {
        "L": 0.13249007040044167,
        "L_A": 0.19433922486530908,
        "L_inv": 0.11948789187728692,
        "L_binned": 0.1484080850128942,
        "L_matched": 0.09503966631294729,
        "delta_o": 0.029169770926095566,
        "delta_o_v0": 0.02974183081450626,
        "line": 1.2992644869497179,
        "slab": -0.05965094380013669,
        "deeper": 0.0405226806745258,
        "depth": 0.11366904176346802,
        "depth_matched": 0.0999986,
    },
    (0.0, 0.03): {
        "L": 0.3021636459391136,
        "L_A": 0.37049461987365107,
        "L_inv": 0.29495295296295293,
        "L_binned": 0.30921607438665044,
        "L_matched": 0.27609419631290094,
        "delta_o": 0.0963016603448515,
        "delta_o_v0": 0.10187666960849465,
        "line": 1.2962995547385983,
        "slab": -0.06835381150712472,
        "deeper": -0.015636919781699454,
        "depth": 0.04071429226692963,
        "depth_matched": 0.036851,
    },
    (0.0, 0.1): {
        "L": 0.26750937236089,
        "L_A": 0.3637546212285046,
        "L_inv": 0.24942529038889782,
        "L_binned": 0.28429169433481527,
        "L_matched": 0.2153704766580869,
        "delta_o": 0.09513665716366657,
        "delta_o_v0": 0.10960168269764514,
        "line": 1.2960114162640577,
        "slab": -0.08938229944094278,
        "deeper": -0.05758592619304359,
        "depth": 0.11366904176346802,
        "depth_matched": 0.0999986,
    },
    (-1.0, 0.03): {
        "L": 0.32967599161510547,
        "L_A": 0.3805726045950555,
        "L_inv": 0.3250113610109935,
        "L_binned": 0.3453946195991221,
        "L_matched": 0.3127634650446506,
        "delta_o": 0.10839994882689824,
        "delta_o_v0": 0.1198100844448115,
        "line": 1.2950322771893275,
        "slab": -0.044701985286584645,
        "deeper": -0.04340072701905129,
        "depth": 0.04071429226692963,
        "depth_matched": 0.036851,
    },
    (-1.0, 0.1): {
        "L": 0.3175745049881771,
        "L_A": 0.39609386060234664,
        "L_inv": 0.3037138850846979,
        "L_binned": 0.3390642701767629,
        "L_matched": 0.277448842171621,
        "delta_o": 0.10750743119724851,
        "delta_o_v0": 0.1344679978079286,
        "line": 1.294891642853663,
        "slab": -0.06575910034798937,
        "deeper": -0.14348936197820805,
        "depth": 0.11366904176346802,
        "depth_matched": 0.0999986,
    },
    (1.0, 0.03): {
        "L": 0.26005800082827923,
        "L_A": 0.3313417797087614,
        "L_inv": 0.2521550393652515,
        "L_binned": 0.26597463812223276,
        "L_matched": 0.2315286601199695,
        "delta_o": 0.08474310691696046,
        "delta_o_v0": 0.08867549032270812,
        "line": 1.2971448621954982,
        "slab": -0.0670830133096727,
        "deeper": -0.006105435271164694,
        "depth": 0.04071429226692963,
        "depth_matched": 0.036851,
    },
    (1.0, 0.1): {
        "L": 0.24577917500104757,
        "L_A": 0.3580987475095475,
        "L_inv": 0.22351956379984803,
        "L_binned": 0.2589767988354512,
        "L_matched": 0.18190115669673496,
        "delta_o": 0.08318544532216146,
        "delta_o_v0": 0.0926456024274841,
        "line": 1.2971032279938817,
        "slab": -0.11136725892423271,
        "deeper": -0.011536968513031098,
        "depth": 0.11366904176346802,
        "depth_matched": 0.0999986,
    },
}

AZIMUTH_PINS = {  # (sun, aod_550, rel. azimuth): (L, L_A, L_inv, Delta_O), each azimuth alone
    (10.0, 0.03, 0.0): (
        0.08241356895529747,
        0.052012217986137754,
        0.0901047523644819,
        0.02861600617825584,
    ),
    (10.0, 0.03, 90.0): (
        0.06883924334539189,
        0.0789113400624982,
        0.06747988823637094,
        0.028840240593403316,
    ),
    (10.0, 0.03, 180.0): (
        0.1323328053314604,
        0.1598924742244447,
        0.12647825366682572,
        0.030018647110993755,
    ),
    (10.0, 0.1, 0.0): (
        0.3330892187151742,
        0.37456464738923073,
        0.32876707170219843,
        0.029235163460289166,
    ),
    (10.0, 0.1, 90.0): (
        0.09763018868155235,
        0.14756635620251202,
        0.08997586428859922,
        0.028838372599371987,
    ),
    (10.0, 0.1, 180.0): (
        0.16734995211933096,
        0.24111209352810614,
        0.14899991946597463,
        0.02950116925281914,
    ),
    (0.0, 0.03, 0.0): (
        0.12476933037038054,
        0.21067703981275565,
        0.11304559850728718,
        0.08334629115444733,
    ),
    (0.0, 0.03, 30.0): (
        0.21465308990263113,
        0.2905649468468798,
        0.20452946375238917,
        0.08465476499252401,
    ),
    (0.0, 0.03, 90.0): (
        0.3007196590514416,
        0.36764066533967044,
        0.2929951943605058,
        0.09285471857804718,
    ),
    (0.0, 0.03, 180.0): (
        0.3036076328267856,
        0.37334857440763175,
        0.29691071156540005,
        0.09974860211165584,
    ),
    (0.0, 0.1, 0.0): (
        0.374468540212517,
        0.5641651939748883,
        0.33176753055439034,
        0.0852027795507751,
    ),
    (0.0, 0.1, 30.0): (
        0.31950981248628485,
        0.4654456306046436,
        0.2879656097556397,
        0.08600782941464738,
    ),
    (0.0, 0.1, 90.0): (
        0.2864187152862616,
        0.3874130928057433,
        0.26669520358371734,
        0.0916871957697423,
    ),
    (0.0, 0.1, 180.0): (
        0.24860002943551845,
        0.3400961496512659,
        0.23215537719407825,
        0.09858611855759083,
    ),
    (-1.0, 0.03, 0.0): (
        0.06037733389067981,
        0.15406103027539392,
        0.0480717158770156,
        0.09326495021261494,
    ),
    (-1.0, 0.03, 30.0): (
        0.25690420233932587,
        0.3367072995084679,
        0.24672493813034044,
        0.09604016689730521,
    ),
    (-1.0, 0.03, 90.0): (
        0.36426504399070914,
        0.4263988029268174,
        0.3576444439118121,
        0.10691751362314611,
    ),
    (-1.0, 0.03, 180.0): (
        0.29508693923950186,
        0.33474640626329355,
        0.29237827811017497,
        0.1098823840306504,
    ),
    (-1.0, 0.1, 0.0): (
        0.2665817644664823,
        0.4523450480572543,
        0.22740115980279568,
        0.09542554327895247,
    ),
    (-1.0, 0.1, 30.0): (
        0.31036813947893727,
        0.45199113760831283,
        0.2806986409286557,
        0.09748717062962836,
    ),
    (-1.0, 0.1, 90.0): (
        0.3365729357456508,
        0.42972811722427773,
        0.31892029641034403,
        0.1050308312206687,
    ),
    (-1.0, 0.1, 180.0): (
        0.2985760742307033,
        0.36245960398041555,
        0.28850747375905167,
        0.10998403117382835,
    ),
    (1.0, 0.03, 0.0): (
        0.13816746278920333,
        0.2063456939564635,
        0.13069418946950942,
        0.07511623969459955,
    ),
    (1.0, 0.03, 30.0): (
        0.16592643469554993,
        0.2283698310291843,
        0.15866337034511446,
        0.07512478322535442,
    ),
    (1.0, 0.03, 90.0): (
        0.2447061302240763,
        0.309410125587622,
        0.23732862304364918,
        0.08062895115946209,
    ),
    (1.0, 0.03, 180.0): (
        0.2754098714324822,
        0.3532734338299008,
        0.2669814556868539,
        0.08885726267445884,
    ),
    (1.0, 0.1, 0.0): (
        0.41718359821881945,
        0.6221238066416448,
        0.3725669719651885,
        0.07664642173394907,
    ),
    (1.0, 0.1, 30.0): (
        0.3079922306278796,
        0.4670655495236349,
        0.2742190379165601,
        0.07635963097400005,
    ),
    (1.0, 0.1, 90.0): (
        0.25533560371260877,
        0.36506185586608697,
        0.23303837036260802,
        0.07969193301605185,
    ),
    (1.0, 0.1, 180.0): (
        0.23622274628948642,
        0.35113563915300805,
        0.21400075723708803,
        0.08667895762827105,
    ),
}

RULE1_PINS = {  # (sun, aod_550): (Delta_O, L, bar, left after ozone)
    (-1.0, 0.03): (
        0.10839994882689824,
        0.32967599161510547,
        0.009329261208223971,
        0.22127604278820723,
    ),
    (-1.0, 0.1): (
        0.10750743119724851,
        0.3175745049881771,
        0.027721239806958398,
        0.21006707379092857,
    ),
    (1.0, 0.03): (
        0.08474310691696046,
        0.26005800082827923,
        0.015805922926055427,
        0.17531489391131877,
    ),
    (1.0, 0.1): (0.08318544532216146, 0.24577917500104757, 0.04451922240239908, 0.1625937296788861),
}

RULE3_PINS = {  # aod_550: (L, L_A, m at sun -1, 0, +1; kappa; kappa_A; removed; offset; bar)
    0.03: (
        (0.32967599161510547, 0.3021636459391136, 0.26005800082827923),
        (0.3805726045950555, 0.37049461987365107, 0.3313417797087614),
        (0.004664630604111986, 0.007210692976160671, 0.007902961463027713),
        0.0072966497174212774,
        0.014537427721742602,
        0.0072966497174212774,
        0.3021636459391136,
        0.05,
    ),
    0.1: (
        (0.3175745049881771, 0.26750937236089, 0.24577917500104757),
        (0.39609386060234664, 0.3637546212285046, 0.3580987475095475),
        (0.013860619903479199, 0.018084081971992183, 0.02225961120119954),
        -0.014167467633722342,
        -0.01334168282744247,
        -0.014167467633722342,
        0.26750937236089,
        0.05,
    ),
}

KAPPA_PINS = {  # aod_550: (kappa: Delta_O, L; kappa_A: L_A; kappa: L - Delta_O; kappa_A: L_A - D_O)
    0.03: (
        -0.00026986752707784667,
        0.0072966497174212774,
        0.014537427721742602,
        0.00756651724449911,
        0.014807295248820462,
    ),
    0.1: (
        -0.00020978109603841943,
        -0.014167467633722342,
        -0.01334168282744247,
        -0.013957686537683894,
        -0.01313190173140405,
    ),
}

V1_BEND = {0.03: 0.09962043590903846, 0.1: 0.05427717402958088}
OZONE_COLUMNS = {
    (10.0, 0.03): 295.91100119919986,
    (10.0, 0.1): 337.93891178065303,
    (0.0, 0.03): 289.6692062996262,
    (0.0, 0.1): 304.63625064140814,
}


@cache
def result() -> object:
    return load_tool().evaluate_offset({sun: committed(sun) for sun in FOUR_SUNS})


@cache
def report_text() -> str:
    return load_tool().format_offset(result())


def test_the_rule_set_p_is_16_points_and_each_azimuth_8() -> None:
    for condition in result().conditions.values():
        assert condition.p.points == 16
        assert [point.points for point in condition.by_azimuth.values()] == [8] * len(
            condition.by_azimuth
        )
    assert list(result().conditions[10.0, 0.03].by_azimuth) == [0.0, 90.0, 180.0]  # no 30
    assert list(result().conditions[-1.0, 0.03].by_azimuth) == [0.0, 30.0, 90.0, 180.0]


def test_the_offset_over_p_and_every_reading_carry_the_pinned_numbers() -> None:
    assert set(CONDITION_PINS) == set(result().conditions)
    for key, expected in CONDITION_PINS.items():
        condition = result().conditions[key]
        got = {
            "L": condition.p.L,
            "L_A": condition.p.L_A,
            "L_inv": condition.p.L_inv,
            "L_binned": condition.p.L_binned,
            "L_matched": condition.p.L_matched,
            "delta_o": condition.p.delta_o,
            "delta_o_v0": condition.p.delta_o_v0,
            "line": condition.p.line,
            "slab": condition.slab,
            "deeper": condition.deeper,
            "depth": condition.depth,
            "depth_matched": condition.depth_matched,
        }
        for name, value in expected.items():
            assert got[name] == pinned(value), (key, name)


def test_m_and_delta_h_are_defined_from_those_readings() -> None:
    for condition in result().conditions.values():
        p = condition.p
        assert p.m == pytest.approx(abs(p.L_inv - p.L), rel=1e-12, abs=0)
        assert p.delta_h == pytest.approx(p.L - p.L_A, rel=1e-12, abs=0)


def test_the_offset_at_each_azimuth_alone_carries_the_pinned_numbers() -> None:
    seen = set()
    for (sun, aod), condition in result().conditions.items():
        for azimuth, point in condition.by_azimuth.items():
            seen.add((sun, aod, azimuth))
            expected = AZIMUTH_PINS[sun, aod, azimuth]
            got = (point.L, point.L_A, point.L_inv, point.delta_o)
            assert got == pinned_tuple(expected), (sun, aod, azimuth)
    assert seen == set(AZIMUTH_PINS)


def pinned_tuple(values: tuple[float, ...]) -> object:
    return pytest.approx(values, rel=1e-6, abs=0)


def test_the_offset_over_p_is_the_mean_of_its_azimuths_alone_since_each_has_eight_points() -> None:
    """P is two azimuths of eight views: the mean over 16 points is the mean of the two means."""
    for condition in result().conditions.values():
        a, b = condition.by_azimuth[90.0], condition.by_azimuth[180.0]
        assert pytest.approx((a.L + b.L) / 2, rel=1e-12, abs=0) == condition.p.L
        assert condition.p.delta_o == pytest.approx((a.delta_o + b.delta_o) / 2, rel=1e-12, abs=0)


def test_rule_1_at_sun_minus_1_and_plus_1_is_pinned_and_contributes_at_every_condition() -> None:
    rule = result().rule1
    assert list(rule.conditions) == list(RULE1_PINS)
    for key, (removed, offset, bar, left) in RULE1_PINS.items():
        condition = rule.conditions[key]
        assert condition.removed == pinned(removed)
        assert condition.offset == pinned(offset)
        assert condition.bar == pinned(bar)
        assert condition.share == pinned(removed / offset)
        assert condition.verdict == "contributes", key
        assert result().left_after_ozone[key] == pinned(left)
    assert (rule.verdict, rule.branch) == ("contributes", 2)


def test_rule_1s_bar_is_twice_the_reading_noise_and_its_offset_is_l() -> None:
    for (sun, aod), condition in result().rule1.conditions.items():
        p = result().conditions[sun, aod].p
        assert condition.bar == pytest.approx(2 * p.m, rel=1e-12, abs=0)
        assert condition.offset == p.L and condition.removed == p.delta_o


def test_rule_1_beside_it_the_pilots_two_sun_0_conditions_give_contributes_branch_2() -> None:
    rule = result().rule1
    assert (rule.pilot_verdict, rule.pilot_branch) == ("contributes", 2)
    assert list(rule.pilot) == [(10.0, 0.03), (10.0, 0.1), (0.0, 0.03), (0.0, 0.1)]
    assert [c.verdict for c in rule.pilot.values()] == ["contributes"] * 4


def test_rule_3_is_pinned_and_both_loads_read_not_this() -> None:
    rule = result().rule3
    assert [f.aod for f in rule.conditions] == [0.03, 0.1]
    for fit in rule.conditions:
        l_values, la_values, m_values, kappa, kappa_a, removed, offset, bar = RULE3_PINS[fit.aod]
        assert pinned_tuple(l_values) == fit.L
        assert pinned_tuple(la_values) == fit.L_A
        assert fit.m == pinned_tuple(m_values)
        assert fit.kappa == pinned(kappa)
        assert fit.kappa_a == pinned(kappa_a)
        assert fit.condition.removed == pinned(removed)
        assert fit.condition.offset == pinned(offset)
        assert fit.condition.bar == pinned(bar)
        assert fit.condition.verdict == "not this"
    assert (rule.verdict, rule.branch) == ("not this", 3)


def test_rule_3s_removed_part_is_the_smaller_of_the_two_bends_and_its_bar_the_floor_here() -> None:
    for fit in result().rule3.conditions:
        assert fit.condition.removed == min(fit.kappa, fit.kappa_a)
        assert fit.kappa == pytest.approx(fit.L[1] - (fit.L[0] + fit.L[2]) / 2, rel=1e-12, abs=0)
        assert fit.kappa_a == pytest.approx(
            fit.L_A[1] - (fit.L_A[0] + fit.L_A[2]) / 2, rel=1e-12, abs=0
        )
        assert fit.condition.offset == fit.L[1]  # L at sun 0, primary
        # twice the largest m is under the 0.05 floor at both loads, so the floor is the bar
        assert 2 * max(fit.m) < 0.05 and fit.condition.bar == 0.05


def test_the_reported_bends_and_columns_are_pinned() -> None:
    for aod, expected in KAPPA_PINS.items():
        k = result().kappas[aod]
        assert (k.delta_o, k.L, k.L_A, k.left, k.left_a) == pinned_tuple(expected)
    for aod, expected in V1_BEND.items():
        assert result().v1_bend[aod] == pinned(expected)
    assert set(OZONE_COLUMNS) == set(result().ozone_columns)
    for key, expected in OZONE_COLUMNS.items():
        assert result().ozone_columns[key] == pinned(expected)


# --- the pilot: what the tool gives against what the criterion printed ----------------------------
#
# Each criterion value is checked as "within half a unit of the last digit shown".

PILOT_TABLE = {  # (sun, aod_550): (L, m, 2m, Delta_O, Delta_H), four decimals
    (10.0, 0.03): (0.1006, 0.0036, 0.0072, 0.0294, -0.0188),
    (10.0, 0.1): (0.1325, 0.0130, 0.0260, 0.0292, -0.0618),
    (0.0, 0.03): (0.3022, 0.0072, 0.0144, 0.0963, -0.0683),
    (0.0, 0.1): (0.2675, 0.0181, 0.0362, 0.0951, -0.0962),
}
PILOT_SHARES = {(0.0, 0.03): (0.319, 3), (0.0, 0.1): (0.356, 3), (10.0, 0.03): (0.29, 2)}
PILOT_MARGINS = {(0.0, 0.03): 6.68, (0.0, 0.1): 2.63, (10.0, 0.03): 4.08, (10.0, 0.1): 1.12}
RULE1_BARS = {  # (sun, aod_550): (Delta_O, 2 Delta_O, Delta_O / 2, that m over the pilot's largest)
    (-1.0, 0.03): (0.1084, 0.2168, 0.0542, 3.0),
    (-1.0, 0.1): (0.1075, 0.2150, 0.0538, 3.0),
    (1.0, 0.03): (0.0847, 0.1695, 0.0424, 2.3),
    (1.0, 0.1): (0.0832, 0.1664, 0.0416, 2.3),
}


@pytest.mark.parametrize(("key", "printed"), sorted(PILOT_TABLE.items()))
def test_the_tool_reproduces_the_criterions_pilot_table(
    key: tuple[float, float], printed: tuple[float, ...]
) -> None:
    p = result().conditions[key].p
    got = (p.L, p.m, 2 * p.m, p.delta_o, p.delta_h)
    names = ("L", "m", "2m", "Delta_O", "Delta_H")
    for name, value, expected in zip(names, got, printed, strict=True):
        assert_shown(value, expected, 4, f"{name} at {key}")


def test_the_tool_reproduces_the_criterions_pilot_shares_margins_and_what_is_left() -> None:
    pilot = result().rule1.pilot
    for key, (printed, digits) in PILOT_SHARES.items():
        assert_shown(pilot[key].share, printed, digits, f"share at {key}")
    assert_shown(pilot[10.0, 0.1].share, 0.22, 2, "share at (10, 0.1)")
    for key, printed in PILOT_MARGINS.items():
        assert_shown(pilot[key].removed / pilot[key].bar, printed, 2, f"margin at {key}")
    # After ozone, 0.2059 and 0.1724 of the sun-0 offset is left.
    for aod, printed in ((0.03, 0.2059), (0.1, 0.1724)):
        p = result().conditions[0.0, aod].p
        assert_shown(p.L - p.delta_o, printed, 4, f"L - Delta_O at (0, {aod})")


def test_the_tool_reproduces_the_prague_column_that_turns_sun_10_at_0_1_into_not_this() -> None:
    """A Prague column under 337.94 DU would turn (sun +10, 0.1) into "not this"."""
    column = result().ozone_columns[10.0, 0.1]
    assert_shown(column, 337.94, 2, "the column at (10, 0.1)")
    # It is the column at which Delta_O falls to the bar: just under it the verdict flips.
    tool = load_tool()
    data = committed(10.0)
    views_ = data["view_elevations_deg"]
    bar = result().rule1.pilot[10.0, 0.1].bar

    def part(ozone_du: float) -> float:
        v1 = np.concatenate([v1_y_at(0.1, 10.0, az, views_) for az in (90.0, 180.0)])
        engine = np.concatenate(
            [tool.ozone_xyz(0.1, ozone_du, 10.0, views_, az)[:, 1] for az in (90.0, 180.0)]
        )
        return tool.offset_of(v1, engine)

    assert part(column) == pytest.approx(bar, rel=1e-9)
    assert part(column - 0.5) < bar < part(column + 0.5)
    assert tool.condition_verdict(part(column - 0.5), result().conditions[10.0, 0.1].p.L, bar) == (
        "not this"
    )


def v1_y_at(aod: float, sun: float, azimuth: float, views_: list[float]) -> np.ndarray:
    return sky_xyz(Atmosphere(aod_550=aod), sun, views_, azimuth, engine="v1")[:, 1]


@pytest.mark.parametrize(("key", "printed"), sorted(RULE1_BARS.items()))
def test_the_tool_reproduces_the_criterions_rule_1_bars(
    key: tuple[float, float], printed: tuple[float, ...]
) -> None:
    removed = result().rule1.conditions[key].removed
    biggest = max(result().conditions[k].p.m for k in result().rule1.pilot)
    assert_shown(biggest, 0.0181, 4, "the pilot's largest m")
    got = (removed, 2 * removed, removed / 2, removed / 2 / biggest)
    for name, value, expected, digits in zip(
        ("Delta_O", "2 Delta_O", "Delta_O / 2", "multiple"), got, printed, (4, 4, 4, 1), strict=True
    ):
        assert_shown(value, expected, digits, f"{name} at {key}")


def test_the_tool_reproduces_v1s_own_bend_and_the_ozone_bend() -> None:
    assert_shown(result().v1_bend[0.03], 0.0996, 4, "v1's bend at 0.03")
    assert_shown(result().v1_bend[0.1], 0.0543, 4, "v1's bend at 0.1")
    # "ozone, -0.0003 at 0.03 (0.0963 - (0.1084 + 0.0847) / 2)"
    assert_shown(result().kappas[0.03].delta_o, -0.0003, 4, "kappa for Delta_O at 0.03")


def test_the_tool_reproduces_the_rule_2_stand_in_and_its_bend_from_v1_alone() -> None:
    for (sun, aod), printed in {
        (0.0, 0.03): -0.0684,
        (0.0, 0.1): -0.0894,
        (10.0, 0.03): -0.0265,
        (10.0, 0.1): -0.0597,
    }.items():
        assert_shown(result().conditions[sun, aod].slab, printed, 4, f"stand-in at {(sun, aod)}")
    # The stand-in's bend across sun -1, 0 and +1: -0.0125 at 0.03 and -0.0008 at 0.1.
    for aod, printed in ((0.03, -0.0125), (0.1, -0.0008)):
        slabs = [result().conditions[sun, aod].slab for sun in (-1.0, 0.0, 1.0)]
        assert_shown(load_tool().bend(*slabs), printed, 4, f"the stand-in's bend at {aod}")


def test_the_criterions_delta_h_minus_stand_in_figures_are_differences_of_rounded_values() -> None:
    """The criterion's +0.0001, -0.0068, +0.0077, -0.0021 subtract four-decimal figures.

    From unrounded values the tool gives +0.0000228, -0.0068629, +0.0076910 and -0.0021982: the
    first, second and fourth differ from the criterion's by one unit in the last digit (they are
    not in the pilot section, whose figures all agree to half a unit). Reported as found.
    """
    printed = {(0.0, 0.03): 0.0001, (0.0, 0.1): -0.0068, (10.0, 0.03): 0.0077, (10.0, 0.1): -0.0021}
    for key, value in printed.items():
        c = result().conditions[key]
        difference = c.p.delta_h - c.slab
        assert abs(difference - value) <= 1.0e-4 + 1e-12, key  # within one unit of the last digit
    assert_shown(
        result().conditions[10.0, 0.03].p.delta_h - result().conditions[10.0, 0.03].slab,
        0.0077,
        4,
        "the one that agrees to half a unit",
    )


def test_the_tool_reproduces_rule_4s_matched_depth_figures() -> None:
    tool = load_tool()
    conditions = result().conditions
    assert_shown(conditions[0.0, 0.1].delta_m, 0.0521, 4, "Delta_M at (0, 0.1)")
    assert_shown(conditions[10.0, 0.1].delta_m, 0.0375, 4, "Delta_M at (10, 0.1)")
    assert_shown(conditions[0.0, 0.03].depth, 0.0407, 4, "Prague's depth at 0.03")
    assert_shown(conditions[0.0, 0.1].depth, 0.1137, 4, "Prague's depth at 0.1")
    assert_shown(conditions[0.0, 0.03].depth_matched, 0.0369, 4, "the clamped depth")
    assert all(conditions[sun, 0.03].clamped for sun in FOUR_SUNS)
    assert not any(conditions[sun, 0.1].clamped for sun in FOUR_SUNS)
    assert conditions[0.0, 0.03].delta_m is None  # clamped: no Delta_M
    c = conditions[0.0, 0.1]
    gap = c.depth - c.depth_matched
    assert_shown(gap, 0.0137, 4, "g, the gap")
    low, high = tool.depth_range(c.delta_m, gap)
    assert_shown(low, 0.0140, 4, "Delta_M's range at +-0.01, low")
    assert_shown(high, 0.0903, 4, "Delta_M's range at +-0.01, high")
    assert_shown(2 * c.p.m, 0.0362, 4, "the bar at (0, 0.1)")  # 0.0140 to 0.0903 straddles it
    assert low < 2 * c.p.m < high
    low5, high5 = tool.depth_range(c.delta_m, gap, 0.005)
    assert_shown(low5, 0.0331, 4, "Delta_M's range at +-0.005, low")
    assert_shown(high5, 0.0712, 4, "Delta_M's range at +-0.005, high")


def test_the_tool_reproduces_rule_4s_deeper_layer_figures() -> None:
    assert_shown(result().conditions[10.0, 0.1].deeper, 0.0405, 4, "deeper layer at (10, 0.1)")
    assert_shown(result().conditions[0.0, 0.1].deeper, -0.0576, 4, "deeper layer at (0, 0.1)")
    for sun, printed in ((-1.0, 3.31), (0.0, 3.68), (1.0, 1.89)):
        ratio = result().conditions[sun, 0.1].deeper / result().conditions[sun, 0.03].deeper
        assert_shown(ratio, printed, 2, f"deeper layer growth 0.03 to 0.1 at sun {sun}")


def test_the_tool_reproduces_the_ozone_line_check_of_the_pilot() -> None:
    for key, printed in {
        (0.0, 0.03): 1.2963,
        (0.0, 0.1): 1.2960,
        (10.0, 0.03): 1.2992,
        (10.0, 0.1): 1.2993,
    }.items():
        assert_shown(result().conditions[key].p.line, printed, 4, f"line check at {key}")


def test_the_criterions_one_sided_remark_about_sun_plus_1_does_not_come_true() -> None:
    """Rule 1 would carry a condition at sun +1 only if L(+1) were at most 0.1695 or 0.1664."""
    for aod, bound in ((0.03, 0.1695), (0.1, 0.1664)):
        c = result().rule1.conditions[1.0, aod]
        assert_shown(2 * c.removed, bound, 4, f"the bar on L at (1, {aod})")
        assert c.offset > 2 * c.removed  # L(+1) is over it: the condition cannot carry


# --- the pipeline can say each thing: stand-ins for Prague ----------------------------------------
#
# Rules 1 and 3 are run end to end on copies of the fixtures whose Prague numbers are replaced, so
# that "carries", "contributes", "not this" and "cannot tell" are each shown able to come out.


def set_prague(data: dict, aod: float, rows_for) -> None:
    """In place: Prague's XYZ at `aod` (every azimuth, every reading) is `rows_for(reading, az)`."""
    entry = data["atmospheres"][f"{aod:g}"]
    for azimuth in data["rel_azimuths_deg"]:
        for reading in ALL_READINGS:
            block = entry["xyz"] if reading == "xyz" else entry["diagnostics"][reading]
            block[f"{azimuth:g}"] = np.asarray(rows_for(reading, azimuth)).tolist()


def uniform(offset: float, **others: float) -> dict[str, float]:
    """One offset under every reading, with some readings (by name) set to another value."""
    out = dict.fromkeys(ALL_READINGS, offset)
    out.update(others)
    return out


def prague_below_v1(data: dict, aod: float, offsets: dict[str, float]) -> None:
    """Prague := v1 * exp(-offset) under each reading: L under that reading is its offset."""
    tool = load_tool()
    sun, views_ = data["sun_elevation_deg"], data["view_elevations_deg"]
    set_prague(
        data,
        aod,
        lambda reading, azimuth: (
            tool.engine_xyz(aod, sun, views_, azimuth, "v1") * math.exp(-offsets[reading])
        ),
    )


def prague_below_ozone(data: dict, aod: float, residual: dict[str, float]) -> None:
    """Prague := the ozone engine at 345 DU * exp(-residual): L is Delta_O plus the residual."""
    tool = load_tool()
    sun, views_ = data["sun_elevation_deg"], data["view_elevations_deg"]
    set_prague(
        data,
        aod,
        lambda reading, azimuth: (
            tool.ozone_xyz(aod, tool.US_STANDARD_OZONE_DU, sun, views_, azimuth)
            * math.exp(-residual[reading])
        ),
    )


def rule3_stand_in(table: dict[float, tuple[float, float, float]], **per_reading) -> dict:
    """Fixtures whose offset at sun -1, 0 and +1 is `table[aod]`; `per_reading` edits some readings.

    `per_reading` maps a reading name to a function (sun, aod) -> its offset there, overriding
    the uniform one.
    """
    fixtures = fresh_fixtures()
    for aod, offsets in table.items():
        for sun, offset in zip((-1.0, 0.0, 1.0), offsets, strict=True):
            overrides = {reading: function(sun, aod) for reading, function in per_reading.items()}
            prague_below_v1(fixtures[sun], aod, uniform(offset, **overrides))
    return fixtures


def rule3(table: dict[float, tuple[float, float, float]], **per_reading) -> object:
    return load_tool().evaluate_offset(rule3_stand_in(table, **per_reading)).rule3


CARRIES_LINE = (0.1, 0.5, 0.1)  # kappa 0.4 over L(0) 0.5: carries
CONTRIBUTES_LINE = (0.25, 0.4, 0.25)  # kappa 0.15 over L(0) 0.4, under half of it: contributes
FLAT_LINE = (0.3, 0.3, 0.3)  # kappa 0: not this


def test_rule_3_carries_when_sun_0_sits_far_above_the_line_through_its_neighbours() -> None:
    rule = rule3({0.03: CARRIES_LINE, 0.1: CARRIES_LINE})
    assert [f.condition.verdict for f in rule.conditions] == ["carries", "carries"]
    assert (rule.verdict, rule.branch) == ("carries", 1)
    fit = rule.conditions[0]
    assert fit.kappa == pytest.approx(0.4, abs=1e-12) and fit.kappa_a == fit.kappa
    assert fit.condition.offset == pytest.approx(0.5, abs=1e-12)
    assert fit.condition.bar == 0.05  # m is 0 here: the floor is the bar


def test_rule_3_contributes_when_the_bend_is_over_the_bar_but_under_half_the_offset() -> None:
    rule = rule3({0.03: CONTRIBUTES_LINE, 0.1: CONTRIBUTES_LINE})
    assert [f.condition.verdict for f in rule.conditions] == ["contributes", "contributes"]
    assert (rule.verdict, rule.branch) == ("contributes", 2)
    assert rule.conditions[0].kappa == pytest.approx(0.15, abs=1e-12)


def test_rule_3_says_not_this_when_sun_0_is_on_the_line() -> None:
    rule = rule3({0.03: FLAT_LINE, 0.1: FLAT_LINE})
    assert [f.condition.verdict for f in rule.conditions] == ["not this", "not this"]
    assert (rule.verdict, rule.branch) == ("not this", 3)
    assert rule.conditions[0].kappa == pytest.approx(0.0, abs=1e-12)


def test_rule_3_says_cannot_tell_when_the_two_loads_disagree() -> None:
    rule = rule3({0.03: CARRIES_LINE, 0.1: FLAT_LINE})
    assert [f.condition.verdict for f in rule.conditions] == ["carries", "not this"]
    assert (rule.verdict, rule.branch) == ("cannot tell", 4)


def test_rule_3_over_a_carries_and_a_contributes_is_contributes() -> None:
    rule = rule3({0.03: CARRIES_LINE, 0.1: CONTRIBUTES_LINE})
    assert [f.condition.verdict for f in rule.conditions] == ["carries", "contributes"]
    assert (rule.verdict, rule.branch) == ("contributes", 2)


def test_rule_3_reads_a_dip_at_sun_0_as_not_this() -> None:
    """A negative kappa (sun 0 below its neighbours' line) reads "not this"."""
    rule = rule3({0.03: (0.5, 0.1, 0.5), 0.1: (0.5, 0.1, 0.5)})
    assert all(f.kappa < 0 for f in rule.conditions)
    assert (rule.verdict, rule.branch) == ("not this", 3)


def test_rule_3_floor_makes_a_small_bend_not_this_even_with_no_noise_at_all() -> None:
    """kappa 0.04 is over the noise bar (0) and under the 0.05 floor: not this."""
    rule = rule3({0.03: (0.2, 0.24, 0.2), 0.1: (0.2, 0.24, 0.2)})
    fit = rule.conditions[0]
    assert fit.kappa == pytest.approx(0.04, abs=1e-12)
    assert fit.condition.bar == 0.05 and fit.condition.verdict == "not this"
    over = rule3({0.03: (0.2, 0.26, 0.2), 0.1: (0.2, 0.26, 0.2)})  # kappa 0.06: over the floor
    assert over.conditions[0].condition.verdict == "contributes"


def test_rule_3_calls_a_fault_only_if_both_readings_show_it() -> None:
    """The removed part is min(kappa, kappa_A): a bend under one reading alone is not a fault."""
    table = {0.03: CARRIES_LINE, 0.1: CARRIES_LINE}
    only_primary = rule3(
        table, xyz_altitude_0=lambda sun, aod: 0.3
    )  # altitude 0: flat at 0.3, so kappa_A = 0
    assert only_primary.conditions[0].kappa == pytest.approx(0.4, abs=1e-12)
    assert only_primary.conditions[0].kappa_a == pytest.approx(0.0, abs=1e-12)
    assert only_primary.conditions[0].condition.removed == pytest.approx(0.0, abs=1e-12)
    assert (only_primary.verdict, only_primary.branch) == ("not this", 3)
    only_altitude = rule3(
        {0.03: FLAT_LINE, 0.1: FLAT_LINE},
        xyz_altitude_0=lambda sun, aod: {-1.0: 0.1, 0.0: 0.5, 1.0: 0.1}[sun],
    )
    assert only_altitude.conditions[0].kappa_a == pytest.approx(0.4, abs=1e-12)
    assert (only_altitude.verdict, only_altitude.branch) == ("not this", 3)


@pytest.mark.parametrize("noisy_sun", [-1.0, 0.0, 1.0])
def test_rule_3s_bar_is_twice_the_largest_reading_noise_at_any_of_the_three_suns(
    noisy_sun: float,
) -> None:
    """m_F is the largest of m(-1), m(0) and m(+1): noise at either neighbour counts too."""
    table = {0.03: (0.1, 0.4, 0.1), 0.1: (0.1, 0.4, 0.1)}  # kappa 0.3: carries over a 0.05 floor
    quiet = rule3(table)
    assert quiet.conditions[0].condition.verdict == "carries"

    def inverse(sun: float, aod: float) -> float:
        offset = dict(zip((-1.0, 0.0, 1.0), table[aod], strict=True))[sun]
        return offset + (0.2 if sun == noisy_sun else 0.0)

    loud = rule3(table, xyz_inverse_visibility=inverse)
    fit = loud.conditions[0]
    assert fit.m[(-1.0, 0.0, 1.0).index(noisy_sun)] == pytest.approx(0.2, abs=1e-12)
    assert fit.condition.bar == pytest.approx(0.4, abs=1e-12)  # 2 * 0.2, over the floor
    assert fit.kappa == pytest.approx(0.3, abs=1e-12)  # kappa uses the primary reading only
    assert fit.condition.verdict == "not this"  # 0.3 is under the 0.4 bar
    assert (loud.verdict, loud.branch) == ("not this", 3)


def rule1_stand_in(residual, shift=None) -> dict:
    """Fixtures whose Prague at sun -1 and +1 is the ozone engine at 345 DU, less a residual.

    `residual(sun, aod)` is L - Delta_O there; `shift(sun, aod)` (default 0) is how far the
    inverse-visibility reading is from the primary one, which makes the reading noise m.
    """
    fixtures = fresh_fixtures()
    for sun in (-1.0, 1.0):
        for aod in LOADS:
            base = residual(sun, aod)
            moved = base + (shift(sun, aod) if shift else 0.0)
            prague_below_ozone(fixtures[sun], aod, uniform(base, xyz_inverse_visibility=moved))
    return fixtures


@cache
def delta_o(sun: float, aod: float) -> float:
    return result().conditions[sun, aod].p.delta_o


def rule1(fixtures: dict) -> object:
    return load_tool().evaluate_offset(fixtures).rule1


def test_rule_1_carries_when_prague_is_v1_with_the_ozone_column_alone() -> None:
    rule = rule1(rule1_stand_in(lambda sun, aod: 0.0))
    assert [c.verdict for c in rule.conditions.values()] == ["carries"] * 4
    assert (rule.verdict, rule.branch) == ("carries", 1)
    for key, c in rule.conditions.items():
        assert c.offset == pytest.approx(delta_o(*key), rel=1e-9)  # L is Delta_O: share 1
        assert c.share == pytest.approx(1.0, rel=1e-9)


def test_rule_1_contributes_when_ozone_is_a_third_of_the_offset() -> None:
    rule = rule1(rule1_stand_in(lambda sun, aod: 2 * delta_o(sun, aod)))
    assert [c.verdict for c in rule.conditions.values()] == ["contributes"] * 4
    assert (rule.verdict, rule.branch) == ("contributes", 2)
    assert rule.conditions[-1.0, 0.03].share == pytest.approx(1 / 3, rel=1e-9)


def test_rule_1_over_a_carries_and_a_contributes_is_contributes_not_cannot_tell() -> None:
    """The rule-level verdict is registered: carries and contributes together are "contributes".

    (The criterion's prediction paragraph names "cannot tell" for sun +1 carrying and sun -1
    contributing; its rule-level verdict, which the tool follows, says branch 2.)
    """
    rule = rule1(rule1_stand_in(lambda sun, aod: 0.0 if sun == 1.0 else 2 * delta_o(sun, aod)))
    verdicts = [c.verdict for c in rule.conditions.values()]
    assert verdicts == ["contributes", "contributes", "carries", "carries"]
    assert (rule.verdict, rule.branch) == ("contributes", 2)


def test_rule_1_says_cannot_tell_when_one_condition_is_not_this() -> None:
    """A reading noise over half of Delta_O (the bar is 2m) makes a condition "not this"."""
    rule = rule1(
        rule1_stand_in(
            lambda sun, aod: 2 * delta_o(sun, aod),
            shift=lambda sun, aod: 0.1 if (sun, aod) == (1.0, 0.1) else 0.0,
        )
    )
    verdicts = [c.verdict for c in rule.conditions.values()]
    assert verdicts == ["contributes", "contributes", "contributes", "not this"]
    assert (rule.verdict, rule.branch) == ("cannot tell", 4)
    assert rule.conditions[1.0, 0.1].bar == pytest.approx(0.2, abs=1e-12)  # 2m


def test_rule_1_code_path_not_this_exists_though_it_cannot_fire_on_prague() -> None:
    """The criterion: Delta_O is fixed and positive, so "not this" would need a huge m."""
    rule = rule1(rule1_stand_in(lambda sun, aod: 0.0, shift=lambda sun, aod: 0.2))
    assert [c.verdict for c in rule.conditions.values()] == ["not this"] * 4
    assert (rule.verdict, rule.branch) == ("not this", 3)


def test_the_pilot_verdict_beside_rule_1_is_over_the_sun_0_conditions_only() -> None:
    """Sun +10 stays "contributes" while sun 0 is made to carry: the verdict follows sun 0."""
    fixtures = fresh_fixtures()
    for aod in LOADS:
        prague_below_ozone(fixtures[0.0], aod, uniform(0.0))
    rule = rule1(fixtures)
    assert [rule.pilot[0.0, aod].verdict for aod in LOADS] == ["carries", "carries"]
    assert [rule.pilot[10.0, aod].verdict for aod in LOADS] == ["contributes", "contributes"]
    assert (rule.pilot_verdict, rule.pilot_branch) == ("carries", 1)
    assert (rule.verdict, rule.branch) == ("contributes", 2)  # the ruled conditions are untouched


def test_what_is_left_after_ozone_is_l_less_delta_o_unless_the_condition_is_not_this() -> None:
    fixtures = rule1_stand_in(
        lambda sun, aod: 2 * delta_o(sun, aod),
        shift=lambda sun, aod: 0.1 if (sun, aod) == (-1.0, 0.03) else 0.0,
    )
    outcome = load_tool().evaluate_offset(fixtures)
    rule = outcome.rule1
    assert rule.conditions[-1.0, 0.03].verdict == "not this"
    assert outcome.left_after_ozone[-1.0, 0.03] == rule.conditions[-1.0, 0.03].offset  # L
    key = (1.0, 0.1)
    assert rule.conditions[key].verdict == "contributes"
    left = rule.conditions[key].offset - rule.conditions[key].removed
    assert outcome.left_after_ozone[key] == pytest.approx(left, rel=1e-12)


# --- selection by value, end to end ---------------------------------------------------------------


def permuted(data: dict) -> dict:
    """The same fixture with its azimuths and views listed in the opposite order."""
    new = copy.deepcopy(data)
    new["rel_azimuths_deg"] = list(reversed(data["rel_azimuths_deg"]))
    new["view_elevations_deg"] = list(reversed(data["view_elevations_deg"]))
    for entry in new["atmospheres"].values():
        for block in (entry["xyz"], *(entry["diagnostics"][v] for v in READINGS)):
            for key in block:
                block[key] = list(reversed(block[key]))
    return new


@cache
def permuted_result() -> object:
    return load_tool().evaluate_offset({sun: permuted(committed(sun)) for sun in FOUR_SUNS})


def test_both_rules_do_not_depend_on_the_order_of_azimuths_and_views() -> None:
    here, there = result(), permuted_result()
    for key, condition in here.conditions.items():
        other = there.conditions[key]
        for name in ("L", "L_A", "L_inv", "L_binned", "L_matched", "delta_o", "delta_o_v0", "line"):
            assert getattr(other.p, name) == pytest.approx(
                getattr(condition.p, name), rel=1e-12, abs=0
            ), (key, name)
        for name in ("slab", "deeper", "depth", "depth_matched"):
            assert getattr(other, name) == pytest.approx(getattr(condition, name), rel=1e-12, abs=0)
        for azimuth, point in condition.by_azimuth.items():
            assert pytest.approx(point.L, rel=1e-12, abs=0) == other.by_azimuth[azimuth].L
    for key, c in here.rule1.conditions.items():
        assert there.rule1.conditions[key].verdict == c.verdict
        assert there.rule1.conditions[key].removed == pytest.approx(c.removed, rel=1e-12, abs=0)
        assert there.rule1.conditions[key].bar == pytest.approx(c.bar, rel=1e-12, abs=0)
    assert (there.rule1.verdict, there.rule1.branch) == (here.rule1.verdict, here.rule1.branch)
    for a, b in zip(here.rule3.conditions, there.rule3.conditions, strict=True):
        assert b.kappa == pytest.approx(a.kappa, rel=1e-12, abs=0)
        assert b.kappa_a == pytest.approx(a.kappa_a, rel=1e-12, abs=0)
        assert b.condition.verdict == a.condition.verdict
    assert (there.rule3.verdict, there.rule3.branch) == (here.rule3.verdict, here.rule3.branch)
    assert there.left_after_ozone == pytest.approx(here.left_after_ozone, rel=1e-12, abs=0)


def test_the_order_of_the_report_rows_follows_the_fixtures_but_the_numbers_do_not() -> None:
    """Rows are keyed by the azimuth's value, not by where it is listed."""
    there = permuted_result()
    assert list(there.conditions[-1.0, 0.03].by_azimuth) == [180.0, 90.0, 30.0, 0.0]
    assert (
        pytest.approx(result().conditions[-1.0, 0.03].by_azimuth[30.0].L, rel=1e-12, abs=0)
        == there.conditions[-1.0, 0.03].by_azimuth[30.0].L
    )


# --- no excess to attribute: the refusal ----------------------------------------------------------

REFUSALS = [  # (sun, aod_550, reading): an offset a rule uses, under the reading it uses it under
    (sun, aod, reading)
    for aod in LOADS
    for sun in (-1.0, 0.0, 1.0)
    for reading in ("xyz", "xyz_altitude_0")
]


@pytest.mark.parametrize(("sun", "aod", "reading"), REFUSALS)
def test_an_offset_at_or_under_zero_that_a_rule_uses_stops_the_report_naming_it(
    sun: float, aod: float, reading: str
) -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    data = fixtures[sun]
    views_ = data["view_elevations_deg"]
    entry = data["atmospheres"][f"{aod:g}"]
    existing = {
        rd: copy.deepcopy(entry["xyz"] if rd == "xyz" else entry["diagnostics"][rd])
        for rd in ALL_READINGS
    }

    def brighter(rd: str, azimuth: float) -> np.ndarray:
        if rd == reading:  # Prague brighter than v1 under this one reading only
            return tool.engine_xyz(aod, sun, views_, azimuth, "v1") * math.exp(0.1)
        return np.array(existing[rd][f"{azimuth:g}"])

    set_prague(data, aod, brighter)
    label = f"sun {'0' if sun == 0 else f'{sun:+g}'}, aod_550 {aod:g}"
    with pytest.raises(tool.PragueError) as error:
        tool.evaluate_offset(fixtures)
    message = str(error.value)
    assert label in message
    assert f"reading {reading}" in message
    assert "no excess to attribute" in message


def test_an_offset_of_exactly_zero_is_refused_too() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    prague_below_v1(fixtures[1.0], 0.03, uniform(0.0))
    with pytest.raises(tool.PragueError, match=r"sun \+1, aod_550 0.03"):
        tool.evaluate_offset(fixtures)


def test_an_offset_no_rule_uses_does_not_stop_the_report() -> None:
    """Sun +10 is not an input of either rule, and neither is L_A at the ruled suns for rule 1."""
    tool = load_tool()
    fixtures = fresh_fixtures()
    prague_below_v1(fixtures[10.0], 0.03, uniform(-0.1))  # Prague brighter at sun +10: negative L
    outcome = tool.evaluate_offset(fixtures)
    assert outcome.conditions[10.0, 0.03].p.L < 0.0
    assert tool.format_offset(outcome)  # printed as found, signed


def test_the_command_line_writes_nothing_and_exits_1_when_there_is_no_excess(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    prague_below_v1(fixtures[-1.0], 0.1, uniform(-0.05))
    monkeypatch.setattr(tool, "read_offset_fixtures", lambda: fixtures)
    assert tool.main(["report", "--offset"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "sun -1, aod_550 0.1" in captured.err and "reading xyz" in captured.err


# --- another tool wrote these files: nothing is used before it is checked -------------------------


def corrupt(sun: float, edit) -> dict:
    fixtures = fresh_fixtures()
    edit(fixtures[sun])
    return fixtures


def edit_luminance(value: object):
    def edit(data: dict) -> None:
        data["atmospheres"]["0.1"]["xyz"]["90"][2][1] = value

    return edit


BAD_FIXTURES = {
    "a sun recorded as a string": lambda d: d.update(sun_elevation_deg="0"),
    "a sun recorded as a bool": lambda d: d.update(sun_elevation_deg=True),
    "a sun that is not finite": lambda d: d.update(sun_elevation_deg=math.nan),
    "a sun that is another fixture's": lambda d: d.update(sun_elevation_deg=5.0),
    "the wrong format": lambda d: d.update(format="something-else"),
    "views that are not a list": lambda d: d.update(view_elevations_deg="1.5"),
    "a view that is not finite": lambda d: d["view_elevations_deg"].__setitem__(2, math.inf),
    "a view that is a bool": lambda d: d["view_elevations_deg"].__setitem__(0, False),
    "no azimuth 90": lambda d: d.update(rel_azimuths_deg=[0.0, 30.0, 180.0, 45.0]),
    "an azimuth listed twice": lambda d: d.update(rel_azimuths_deg=[0.0, 30.0, 90.0, 90.0, 180.0]),
    "an azimuth that is not finite": lambda d: d["rel_azimuths_deg"].__setitem__(0, math.nan),
    "channel centres that are not numbers": lambda d: d.update(channel_centres_nm=["a", "b"]),
    "no atmospheres": lambda d: d.update(atmospheres=[]),
    "no 0.1 atmosphere": lambda d: d["atmospheres"].pop("0.1"),
    "an aod that does not match its key": lambda d: d["atmospheres"]["0.1"].update(aod_550=0.2),
    "no diagnostics": lambda d: d["atmospheres"]["0.1"].pop("diagnostics"),
    "a missing reading": lambda d: d["atmospheres"]["0.1"]["diagnostics"].pop("xyz_altitude_0"),
    "a missing azimuth block": lambda d: d["atmospheres"]["0.1"]["xyz"].pop("90"),
    "a block of the wrong length": lambda d: d["atmospheres"]["0.1"]["xyz"]["90"].pop(),
    "a row of two numbers": lambda d: d["atmospheres"]["0.1"]["xyz"]["90"][1].pop(),
    "a luminance that is NaN": edit_luminance(math.nan),
    "a luminance that is infinite": edit_luminance(math.inf),
    "a luminance of zero": edit_luminance(0.0),
    "a negative luminance": edit_luminance(-1.0),
    "a luminance that is a string": edit_luminance("0.5"),
    "a luminance that is a bool": edit_luminance(True),
    "a diagnostic luminance that is NaN": lambda d: d["atmospheres"]["0.03"]["diagnostics"][
        "xyz_inverse_visibility"
    ]["180"][0].__setitem__(1, math.nan),
    "a transmittance of zero": lambda d: d["atmospheres"]["0.1"]["diagnostics"][
        "zenith_transmittance"
    ].__setitem__(3, 0.0),
    "a transmittance list of the wrong length": lambda d: d["atmospheres"]["0.1"]["diagnostics"][
        "zenith_transmittance"
    ].pop(),
    "no matched depth": lambda d: d["atmospheres"]["0.1"]["diagnostics"].pop(
        "matched_aerosol_depth"
    ),
    "a clamp flag that is a number": lambda d: d["atmospheres"]["0.1"]["diagnostics"][
        "matched_aerosol_depth"
    ].update(clamped_at_largest_visibility=1),
    "a matched depth that is NaN": lambda d: d["atmospheres"]["0.1"]["diagnostics"][
        "matched_aerosol_depth"
    ].update(prague_aerosol_depth=math.nan),
}


@pytest.mark.parametrize("name", sorted(BAD_FIXTURES))
def test_a_malformed_fixture_is_refused_before_any_value_is_used(name: str) -> None:
    tool = load_tool()
    fixtures = corrupt(-1.0, BAD_FIXTURES[name])
    with pytest.raises(tool.PragueError, match="malformed"):
        tool.evaluate_offset(fixtures)


def test_a_fixture_that_is_not_an_object_is_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    fixtures[-1.0] = [1, 2, 3]
    with pytest.raises(tool.PragueError, match="not a JSON object"):
        tool.evaluate_offset(fixtures)


def test_a_missing_fixture_is_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    del fixtures[1.0]
    with pytest.raises(tool.PragueError, match="need the fixtures"):
        tool.evaluate_offset(fixtures)


def test_the_committed_fixtures_pass_the_check() -> None:
    tool = load_tool()
    for sun in FOUR_SUNS:
        tool.check_offset_fixture(committed(sun), sun)


def test_an_unreadable_fixture_file_is_an_error_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    moved = {}
    for sun, spec in tool.FIXTURE_SPECS.items():
        path = tmp_path / spec.path.name
        path.write_text(spec.path.read_text())
        moved[sun] = tool.fixture_spec(sun, path)
    (tmp_path / "sun1.json").write_text("{ this is not json")
    monkeypatch.setattr(tool, "FIXTURE_SPECS", moved)
    assert tool.main(["report", "--offset"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "sun1.json" in captured.err and "cannot be read" in captured.err


# --- the report -----------------------------------------------------------------------------------

HEADINGS = (
    "### Rule 1: ozone",
    "### Rule 2: no test",
    "### Rule 3: Prague's fit near the horizon sun",
    "### Rule 4: no test",
    "### Reported, never ruled",
)


def test_the_report_has_the_pilot_then_rules_1_to_4_then_what_is_reported_in_that_order() -> None:
    text = report_text()
    headings = [line for line in text.splitlines() if line.startswith("### ")]
    assert "pilot" in headings[0].lower()
    assert len(headings) == 1 + len(HEADINGS)
    for found, wanted in zip(headings[1:], HEADINGS, strict=True):
        assert found.startswith(wanted), (found, wanted)
    positions = [text.index(heading) for heading in HEADINGS]
    assert text.index(headings[0]) < positions[0]
    assert positions == sorted(positions)


def test_the_report_prints_two_verdicts_and_nothing_else_that_reads_as_one() -> None:
    text = report_text()
    assert "Verdict: **contributes** (branch 2)." in text
    assert "Verdict: **not this** (branch 3)." in text
    assert text.count("Verdict:") == 2  # rules 1 and 3; rules 2 and 4 are "no test"
    assert text.count("No verdict is printed.") == 2


def test_the_report_prints_the_pilots_rule_level_verdict_beside_rule_1s() -> None:
    text = report_text()
    assert "under the same rule-level verdict (not a ruling): contributes (branch 2)." in text


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    following = [text.find("\n### ", start + 1)]
    end = following[0] if following[0] != -1 else len(text)
    return text[start:end]


def test_the_report_prints_every_item_the_criterion_lists_for_the_pilot() -> None:
    pilot = section(report_text(), "### The pilot")
    for needle in (
        "| condition | L | m | 2m | Delta_O, ozone | Delta_H, altitude 0 |",
        "(sun +10, 0.03)",
        "(sun +10, 0.1)",
        "(sun 0, 0.03)",
        "(sun 0, 0.1)",
        "share Delta_O / L",
        "margin over the bar, Delta_O / 2m",
        "a Prague column under (DU) turns it into",
        "left after ozone, L - Delta_O",
        "carries only if L is at most 2 Delta_O",
        '"not this" only if m is at least Delta_O / 2',
        "the pilot's largest m (0.0181)",
        "v1's own sky across sun -1, 0 and +1",
    ):
        assert needle in pilot, needle


def test_the_report_prints_every_item_the_criterion_lists_for_rule_1() -> None:
    rule_1 = section(report_text(), "### Rule 1: ozone")
    for needle in (
        "| condition | Delta_O | L | bar 2m | share Delta_O / L | verdict |",
        "(sun -1, 0.03)",
        "(sun -1, 0.1)",
        "(sun +1, 0.03)",
        "(sun +1, 0.1)",
        "Verdict:",
        "the pilot's two sun-0 conditions",
    ):
        assert needle in rule_1, needle


def test_the_report_prints_every_item_the_criterion_lists_for_rule_2() -> None:
    rule_2 = section(report_text(), "### Rule 2: no test")
    for needle in (
        "Delta_H = L - L_A",
        "1.2544",
        "slab stand-in",
        "Delta_H minus stand-in",
        "libRadtran",
        "never vendored or imported",
        "ground raised to the observer",
    ):
        assert needle in rule_2, needle
    for sun in ("+10", "0", "-1", "+1"):
        for aod in ("0.03", "0.1"):
            assert f"| (sun {sun}, {aod}) |" in rule_2, (sun, aod)


def test_the_report_prints_every_item_the_criterion_lists_for_rule_3() -> None:
    rule_3 = section(report_text(), "### Rule 3: Prague's fit near the horizon sun")
    for needle in (
        "kappa(a) = L(0, a) - (L(-1, a) + L(+1, a)) / 2",
        "kappa_A",
        "| `aod_550` | L(-1) | L(0) | L(+1) | L_A(-1) | L_A(0) | L_A(+1) | m(-1) | m(0) | m(+1) |",
        "removed, min(kappa, kappa_A)",
        "bar max(2 m_F, 0.05)",
        "share removed / L(0)",
        'A negative kappa reads "not this".',
        "Verdict: **not this** (branch 3).",
    ):
        assert needle in rule_3, needle
    assert "kink" not in rule_3  # that sentence is for "carries" or "contributes" only


def test_the_report_prints_every_item_the_criterion_lists_for_rule_4() -> None:
    rule_4 = section(report_text(), "### Rule 4: no test")
    for needle in (
        "Delta_M = L - L_matched",
        "Prague's depth, primary",
        "Prague's depth at the match",
        "0.01 off either way",
        "clamped (no range)",
        "load step L(0.1) - L(0.03)",
        "deeper layer at 0.03",
        "deeper layer at 0.1",
        "ratio 0.1 to 0.03",
        "libRadtran",
    ):
        assert needle in rule_4, needle
    for sun in ("+10", "0", "-1", "+1"):
        for aod in ("0.03", "0.1"):
            assert f"| (sun {sun}, {aod}) |" in rule_4, (sun, aod)
        assert f"| {sun} | " in rule_4  # the load step and the deeper layer, at each sun


def test_the_report_prints_every_item_the_criterion_lists_as_reported_never_ruled() -> None:
    reported = section(report_text(), "### Reported, never ruled")
    for needle in (
        "| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L",
        "Delta_O (v0)",
        "line check",
        "L_binned",
        "never part of m",
        "kappa for Delta_O",
        "kappa for L",
        "kappa_A for L_A",
        "kappa for L - Delta_O",
        "kappa_A for L_A - Delta_O",
        "What is left of L after ozone",
    ):
        assert needle in reported, needle


def test_the_per_azimuth_rows_are_only_for_the_azimuths_a_fixture_has() -> None:
    reported = section(report_text(), "### Reported, never ruled")
    pieces = reported.split("#### Sun ")[1:]
    by_sun = {piece.split(" ", 1)[0]: piece for piece in pieces}
    assert set(by_sun) == {"+10", "0", "-1", "+1"}
    assert "azimuth 30 alone" not in by_sun["+10"]  # sun10.json has no azimuth 30
    for sun in ("0", "-1", "+1"):
        assert "azimuth 30 alone" in by_sun[sun]
    for sun, piece in by_sun.items():
        for azimuth in ("0", "90", "180"):
            assert f"azimuth {azimuth} alone (8)" in piece, (sun, azimuth)
        assert "P (azimuths 90, 180) (16)" in piece
    assert "`sun-1.json`" in by_sun["-1"] and "`sun1.json`" in by_sun["+1"]


def test_there_is_no_third_load_anywhere_in_the_report() -> None:
    text = report_text()
    for sun in ("+10", "0", "-1", "+1"):
        assert f"(sun {sun}, 0.2)" not in text


def test_the_report_rows_carry_the_pinned_numbers() -> None:
    """The printed tables are what goes in the build log: their numbers are pinned as text."""
    text = report_text()
    for row in (
        "| (sun +10, 0.03) | 0.1006 | 0.0036 | 0.0072 | 0.0294 | -0.0188 |",
        "| (sun 0, 0.1) | 0.2675 | 0.0181 | 0.0362 | 0.0951 | -0.0962 |",
        "| (sun +10, 0.1) | 0.220 | 1.12 | contributes | 337.94 | 0.1033 |",
        "| (sun 0, 0.03) | 0.319 | 6.68 | contributes | 289.67 | 0.2059 |",
        "| (sun -1, 0.03) | 0.1084 | 0.3297 | 0.0093 | 0.329 | contributes |",
        "| (sun -1, 0.1) | 0.1075 | 0.3176 | 0.0277 | 0.339 | contributes |",
        "| (sun +1, 0.03) | 0.0847 | 0.2601 | 0.0158 | 0.326 | contributes |",
        "| (sun +1, 0.1) | 0.0832 | 0.2458 | 0.0445 | 0.338 | contributes |",
        "| 0.03 | 0.3297 | 0.3022 | 0.2601 | 0.3806 | 0.3705 | 0.3313 | 0.0047 | 0.0072 | 0.0079 |",
        "| 0.1 | 0.3176 | 0.2675 | 0.2458 | 0.3961 | 0.3638 | 0.3581 | 0.0139 | 0.0181 | 0.0223 |",
        "| 0.03 | +0.0073 | +0.0145 | +0.0073 | 0.3022 | 0.0500 | +0.024 | not this |",
        "| 0.1 | -0.0142 | -0.0133 | -0.0142 | 0.2675 | 0.0500 | -0.053 | not this |",
        "| (sun 0, 0.1) | +0.0521 | 0.1137 | 0.1000 | 0.0140 to 0.0903 |",
        "| (sun 0, 0.03) | clamped | 0.0407 | 0.0369 | clamped (no range) |",
        "| 0.03 | -0.0003 | +0.0073 | +0.0145 | +0.0076 | +0.0148 |",
        "| 0.1 | -0.0002 | -0.0142 | -0.0133 | -0.0140 | -0.0131 |",
    ):
        assert row in text, row
    assert "0.03: +0.0996; 0.1: +0.0543" in text


def test_the_report_is_complete_and_deterministic() -> None:
    tool = load_tool()
    again = tool.report_offset({sun: committed(sun) for sun in FOUR_SUNS})
    assert again == report_text()
    assert report_text().endswith("\n") and not report_text().endswith("\n\n")


def test_the_command_line_report_for_the_offset_runs_from_the_committed_fixtures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["report", "--offset"]) == 0
    assert capsys.readouterr().out == report_text()


def test_the_offset_flag_belongs_to_report_alone() -> None:
    tool = load_tool()
    assert tool.build_parser().parse_args(["report", "--offset"]).offset is True
    assert tool.build_parser().parse_args(["report"]).offset is False
    with pytest.raises(SystemExit):
        tool.main(["fixture", "--offset"])


def test_a_report_for_one_of_the_new_suns_alone_points_at_the_offset_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["report", "--sun", "-1"]) == 1
    assert "report --offset" in capsys.readouterr().err


def test_the_older_reports_still_run_and_the_34_and_45_reports_are_unchanged_in_kind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Byte identity with the earlier reports is checked against origin/main's output (the log)."""
    tool = load_tool()
    assert tool.main(["report"]) == 0
    first = capsys.readouterr().out
    assert first.startswith("### Sun +10, `aod_550` 0.03")
    assert tool.main(["report", "--sun", "0"]) == 0
    second = capsys.readouterr().out
    assert second.startswith("### Rule A at sun 0")
    assert "Rule 1" not in first and "Rule 1" not in second


def test_the_older_reports_are_written_to_stdout_exactly_as_the_report_functions_build_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command line adds nothing and drops nothing: stdout is the report function's text.

    The byte identity of the #34 and #45 reports with origin/main's is checked against main's own
    output in the build log. What a test can hold is the thing the command line could break: the
    text the report function builds is the text written, with one newline at the end and no more.
    This compares the command line with the function that builds the report, never with a stored
    copy of report text or a hash of it (the project pins numbers by value).
    """
    tool = load_tool()
    cases = {
        ("report",): tool.report(committed(10.0)),
        ("report", "--sun", "0"): tool.report_sun0(committed(10.0), committed(0.0)),
        ("report", "--offset"): tool.report_offset({sun: committed(sun) for sun in FOUR_SUNS}),
    }
    for argv, built in cases.items():
        assert tool.main(list(argv)) == 0, argv
        written = capsys.readouterr().out
        assert written == built, argv
        assert written.endswith("\n") and not written.endswith("\n\n"), argv
