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

import hashlib
import importlib.util
import itertools
import json
import math
import sys
import zlib
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

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


def test_the_ozone_engine_reads_the_tables_ozone_nodes_from_its_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    assert DOMAIN["ozone_du"] == (250.0, 330.0)  # v1 is unchanged: its domain is registered
    views_ = list(all_views())
    monkeypatch.setattr(tool, "DOMAIN", {"ozone_du": (260.0, 320.0)})
    t = (290.0 - 260.0) / 60.0
    low = multiple_scattering_xyz(Atmosphere(aod_550=0.1, ozone_du=260.0), 0.0, views_, 90.0)
    high = multiple_scattering_xyz(Atmosphere(aod_550=0.1, ozone_du=320.0), 0.0, views_, 90.0)
    single = sky_xyz(Atmosphere(aod_550=0.1, ozone_du=290.0), 0.0, views_, 90.0)
    expected = single + np.exp((1.0 - t) * np.log(low) + t * np.log(high))
    assert np.abs(tool.ozone_xyz(0.1, 290.0, 0.0, views_, 90.0) / expected - 1.0).max() <= 1e-13


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
