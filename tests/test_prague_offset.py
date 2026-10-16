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

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from lastlight.render import Atmosphere

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
