"""Renderer v1 against the Prague Sky Model at sun 0, and the forward-lobe attribution (#45).

The two decision rules are pre-registered in `docs/renderer/prague-sun0-criterion.md` (landed in
commit fabbbdc, before any sun-0 number existed). This file is written in the order the work was
done, and the order is evidence:

* the rules themselves, on synthetic inputs, every branch and every boundary (this section was
  committed red, then the rule code made it green, before any Prague number at sun 0 existed);
* the lobe diagnostic engine;
* the sun-0 fixture, `tests/fixtures/prague/sun0.json`;
* the rules run on the fixtures, pinned by value.

The rules live in `tools/prague_reference.py` (the tool only, never `lastlight.render`).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
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
from lastlight.render.v1 import multiple_scattering_xyz

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prague_reference.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "prague"
FIXTURE_SUN0 = FIXTURE_DIR / "sun0.json"
FIXTURE_SUN10 = FIXTURE_DIR / "sun10.json"
CACHE = Path.home() / ".cache" / "lastlight" / "prague"


def load_tool() -> ModuleType:
    """The tool as a module, shared with the other Prague test file when it is already loaded."""
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


# --- the registered constants ---------------------------------------------------------------------


def test_the_registered_constants_are_the_criterions() -> None:
    tool = load_tool()
    assert tool.MONTE_CARLO_EXCESS == 0.066
    assert tool.UNCERTAINTY_BAR_FACTOR == 2.0  # the bar is 2U
    assert tool.STD_DDOF == 1
    assert tool.EXCESS_FLOOR == 0.05
    assert tool.LOBE_FACTOR == 0.5
    assert tool.V1_ASYMMETRY_K == 70
    assert tool.RULE_A_AZIMUTHS_DEG == (90.0, 180.0)
    assert tool.RULE_A_ALTERNATIVE_READINGS == (
        "xyz_inverse_visibility",
        "xyz_matched_aerosol_depth",
    )


def test_the_asymmetry_grid_is_61_integer_hundredths_from_30_to_90() -> None:
    tool = load_tool()
    grid = tool.ASYMMETRY_GRID_K
    assert list(grid) == list(range(30, 91))
    assert len(grid) == 61
    assert all(isinstance(k, int) for k in grid)
    assert [tool.asymmetry_of(k) for k in grid] == [k / 100 for k in grid]
    assert tool.asymmetry_of(tool.V1_ASYMMETRY_K) == 0.7  # v1's own asymmetry, exactly


# --- rule A: which side of the aod 0.1 gap Prague is on -------------------------------------------
#
# The verdict is a function of three numbers, D, E and U, so every boundary is tested on exactly
# representable values (eighths), with no rounding between the number and its bound.

CANNOT_TELL = "cannot tell"
PORT = "port"
MONTE_CARLO = "Monte Carlo"


def test_rule_a_branch_1_cannot_tell_when_the_predictions_are_not_twice_u_apart() -> None:
    tool = load_tool()
    assert tool.rule_a_verdict(0.0, 0.2, 0.125) == (CANNOT_TELL, 1)
    assert tool.rule_a_verdict(0.2, 0.2, 0.125) == (CANNOT_TELL, 1)


def test_rule_a_branch_1_beats_branch_2_even_when_d_is_dead_on_the_port() -> None:
    """E <= 2U is decided first: a D of 0 is "port" only when the predictions are separated."""
    tool = load_tool()
    assert tool.rule_a_verdict(0.0, 0.1, 0.125) == (CANNOT_TELL, 1)


def test_rule_a_boundary_e_exactly_2u_is_cannot_tell() -> None:
    tool = load_tool()
    u = 0.125
    assert tool.rule_a_verdict(0.0, 2 * u, u) == (CANNOT_TELL, 1)
    assert tool.rule_a_verdict(2 * u, 2 * u, u) == (CANNOT_TELL, 1)  # nor Monte Carlo
    assert tool.rule_a_verdict(0.0, above(2 * u), u) == (PORT, 2)  # one ulp over: separated


def test_rule_a_branch_2_port_when_d_is_within_u_of_zero() -> None:
    tool = load_tool()
    assert tool.rule_a_verdict(0.0, 0.5, 0.125) == (PORT, 2)
    assert tool.rule_a_verdict(0.0625, 0.5, 0.125) == (PORT, 2)
    assert tool.rule_a_verdict(-0.0625, 0.5, 0.125) == (PORT, 2)


def test_rule_a_boundary_d_exactly_u_from_zero_is_port_on_either_side() -> None:
    tool = load_tool()
    u = 0.125
    assert tool.rule_a_verdict(u, 0.5, u) == (PORT, 2)
    assert tool.rule_a_verdict(-u, 0.5, u) == (PORT, 2)


def test_rule_a_just_beyond_u_from_zero_is_no_longer_port() -> None:
    tool = load_tool()
    u = 0.125
    assert tool.rule_a_verdict(above(u), 0.5, u) == (CANNOT_TELL, 4)
    assert tool.rule_a_verdict(below(-u), 0.5, u) == (CANNOT_TELL, 4)


def test_rule_a_branch_3_monte_carlo_when_d_is_within_u_of_e() -> None:
    tool = load_tool()
    assert tool.rule_a_verdict(0.5, 0.5, 0.125) == (MONTE_CARLO, 3)
    assert tool.rule_a_verdict(0.4375, 0.5, 0.125) == (MONTE_CARLO, 3)
    assert tool.rule_a_verdict(0.5625, 0.5, 0.125) == (MONTE_CARLO, 3)  # above E counts too


def test_rule_a_boundary_d_exactly_u_from_e_is_monte_carlo_on_either_side() -> None:
    tool = load_tool()
    u, e = 0.125, 0.5
    assert tool.rule_a_verdict(e - u, e, u) == (MONTE_CARLO, 3)
    assert tool.rule_a_verdict(e + u, e, u) == (MONTE_CARLO, 3)


def test_rule_a_just_beyond_u_from_e_is_no_longer_monte_carlo() -> None:
    tool = load_tool()
    u, e = 0.125, 0.5
    assert tool.rule_a_verdict(below(e - u), e, u) == (CANNOT_TELL, 4)
    assert tool.rule_a_verdict(above(e + u), e, u) == (CANNOT_TELL, 4)


def test_rule_a_branch_4_cannot_tell_when_d_is_between_the_predictions() -> None:
    tool = load_tool()
    assert tool.rule_a_verdict(0.25, 0.5, 0.125) == (CANNOT_TELL, 4)
    assert tool.rule_a_verdict(-0.5, 0.5, 0.125) == (CANNOT_TELL, 4)  # outside, on the far side
    assert tool.rule_a_verdict(1.0, 0.5, 0.125) == (CANNOT_TELL, 4)


def test_rule_a_branches_2_and_3_never_both_hold_when_e_exceeds_2u() -> None:
    """With E > 2U, a D within U of one prediction is more than U from the other."""
    tool = load_tool()
    u = 0.125
    rng = np.random.default_rng(0)
    for e in [above(2 * u), 0.3, 0.5, 2.0]:
        for d in rng.uniform(-1.0, 3.0, 400):
            near_port = abs(d) <= u
            near_mc = abs(d - e) <= u
            assert not (near_port and near_mc), (d, e)
            verdict, branch = tool.rule_a_verdict(float(d), e, u)
            assert branch in (2, 3, 4)
            assert (branch == 2) == near_port
            assert (branch == 3) == near_mc


def test_rule_a_verdict_strings_are_the_registered_ones() -> None:
    tool = load_tool()
    assert tool.CANNOT_TELL == CANNOT_TELL
    assert tool.PORT == PORT
    assert tool.MONTE_CARLO == MONTE_CARLO


# --- rule A's statistics ------------------------------------------------------------------------


def alternatives(**shifts: float) -> dict[str, np.ndarray]:
    """Per-reading d arrays: each a constant, so its mean is the value."""
    return {name: np.full(4, value) for name, value in shifts.items()}


def test_rule_a_d_and_e_are_means_over_the_points() -> None:
    tool = load_tool()
    result = tool.rule_a(
        d=np.array([0.0, 0.25, 0.5, 0.75]),
        e=np.array([1.0, 2.0, 3.0, 6.0]),
        d_alternatives=alternatives(a=0.375, b=0.375),
    )
    assert result.D == 0.375
    assert result.E == 3.0


def test_rule_a_s_is_the_sample_standard_deviation_ddof_1() -> None:
    tool = load_tool()
    result = tool.rule_a(
        d=np.array([0.0, 1.0]),
        e=np.array([5.0, 5.0]),
        d_alternatives={"a": np.full(2, 0.5), "b": np.full(2, 0.5)},
    )
    # ddof = 1: sqrt(0.5); the population value (ddof = 0) would be 0.5.
    assert result.s == pytest.approx(math.sqrt(0.5), rel=1e-12, abs=0)
    assert result.s != 0.5


def test_rule_a_m_is_the_largest_absolute_shift_of_d_over_the_alternative_readings() -> None:
    tool = load_tool()
    d = np.full(4, 0.5)
    result = tool.rule_a(
        d=d,
        e=np.full(4, 9.0),
        d_alternatives={"inv": np.full(4, 0.625), "matched": np.full(4, 0.125)},
    )
    assert result.D == 0.5
    assert result.m == 0.375  # |0.125 - 0.5|, the negative shift, is larger than |0.625 - 0.5|
    assert result.s == 0.0
    assert result.U == 0.375


def test_rule_a_m_takes_the_absolute_value_of_a_positive_shift_too() -> None:
    tool = load_tool()
    result = tool.rule_a(
        d=np.full(4, 0.5),
        e=np.full(4, 9.0),
        d_alternatives={"inv": np.full(4, 0.875), "matched": np.full(4, 0.375)},
    )
    assert result.m == 0.375  # |0.875 - 0.5|


def test_rule_a_u_is_the_larger_of_s_and_m() -> None:
    tool = load_tool()
    spread = np.array([0.0, 1.0])  # s = sqrt(0.5) = 0.707
    big_shift = tool.rule_a(
        d=spread,
        e=np.full(2, 9.0),
        d_alternatives={"a": np.full(2, 0.5 + 1.0), "b": np.full(2, 0.5)},
    )
    assert big_shift.m == 1.0
    assert big_shift.s < big_shift.m
    assert big_shift.m == big_shift.U
    big_spread = tool.rule_a(
        d=spread,
        e=np.full(2, 9.0),
        d_alternatives={"a": np.full(2, 0.5 + 0.125), "b": np.full(2, 0.5)},
    )
    assert big_spread.m == 0.125
    assert big_spread.s > big_spread.m
    assert big_spread.s == big_spread.U


def test_rule_a_uses_every_alternative_reading_it_is_given() -> None:
    tool = load_tool()
    result = tool.rule_a(
        d=np.full(2, 0.0),
        e=np.full(2, 9.0),
        d_alternatives={"a": np.full(2, 0.25), "b": np.full(2, -0.5), "c": np.full(2, 0.125)},
    )
    assert result.m == 0.5


def test_rule_a_end_to_end_boundary_e_exactly_2u() -> None:
    """With s = 0 and one reading 0.25 away, U = 0.25; an E of exactly 0.5 is not above 2U."""
    tool = load_tool()
    flat = np.full(4, 0.5)
    result = tool.rule_a(
        d=flat,
        e=np.full(4, 0.5),
        d_alternatives={"a": np.full(4, 0.75), "b": flat},
    )
    assert (result.D, result.E, result.s, result.m, result.U) == (0.5, 0.5, 0.0, 0.25, 0.25)
    assert (result.verdict, result.branch) == (CANNOT_TELL, 1)


def test_rule_a_end_to_end_each_verdict() -> None:
    tool = load_tool()
    zero = np.zeros(4)

    def run(d_value: float) -> tuple[str, int]:
        result = tool.rule_a(
            d=np.full(4, d_value),
            e=np.full(4, 1.0),
            d_alternatives={"a": np.full(4, d_value + 0.125), "b": np.full(4, d_value)},
        )
        assert result.U == 0.125
        return result.verdict, result.branch

    assert run(0.0) == (PORT, 2)
    assert run(1.0) == (MONTE_CARLO, 3)
    assert run(0.5) == (CANNOT_TELL, 4)
    low_power = tool.rule_a(
        d=zero, e=np.full(4, 0.125), d_alternatives={"a": zero + 0.125, "b": zero}
    )
    assert (low_power.verdict, low_power.branch) == (CANNOT_TELL, 1)


def test_rule_a_rejects_inputs_it_cannot_rule_on() -> None:
    tool = load_tool()
    ok = np.full(4, 0.5)
    with pytest.raises(ValueError, match="same number of points"):
        tool.rule_a(d=ok, e=np.full(3, 0.5), d_alternatives={"a": ok})
    with pytest.raises(ValueError, match="same number of points"):
        tool.rule_a(d=ok, e=ok, d_alternatives={"a": np.full(3, 0.5)})
    with pytest.raises(ValueError, match="finite"):
        tool.rule_a(d=np.array([0.5, math.nan, 0.5, 0.5]), e=ok, d_alternatives={"a": ok})
    with pytest.raises(ValueError, match="alternative"):
        tool.rule_a(d=ok, e=ok, d_alternatives={})
    with pytest.raises(ValueError, match="at least two"):
        tool.rule_a(d=np.array([0.5]), e=np.array([0.5]), d_alternatives={"a": np.array([0.5])})


# --- rule B: what the toward-sun excess comes from ----------------------------------------------

NO_EXCESS = "no excess at sun 0"
AEROSOL_LOBE = "aerosol lobe"
NEITHER = "neither"
FLOOR = 0.05


def test_rule_b_verdict_strings_are_the_registered_ones() -> None:
    tool = load_tool()
    assert tool.NO_EXCESS == NO_EXCESS
    assert tool.AEROSOL_LOBE == AEROSOL_LOBE
    assert tool.NEITHER == NEITHER


def test_rule_b_branch_1_no_excess_when_no_condition_counts() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[0.04, 0.03], r_fit=[0.5, 0.5])
    assert (result.verdict, result.branch) == (NO_EXCESS, 1)
    assert result.counted == (False, False)


def test_rule_b_boundary_r_exactly_at_the_floor_counts() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[FLOOR, 0.0], r_fit=[FLOOR / 2, 0.0])
    assert result.counted == (True, False)
    assert (result.verdict, result.branch) == (AEROSOL_LOBE, 2)


def test_rule_b_boundary_r_one_ulp_under_the_floor_does_not_count() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[below(FLOOR), below(FLOOR)], r_fit=[0.0, 0.0])
    assert result.counted == (False, False)
    assert (result.verdict, result.branch) == (NO_EXCESS, 1)


def test_rule_b_branch_2_aerosol_lobe_when_every_counted_load_is_at_least_halved() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[0.2, 0.4], r_fit=[0.05, 0.1])
    assert result.counted == (True, True)
    assert result.halved == (True, True)
    assert (result.verdict, result.branch) == (AEROSOL_LOBE, 2)


def test_rule_b_boundary_r_fit_exactly_half_of_r_base_is_a_lobe() -> None:
    tool = load_tool()
    r_base = [0.25, 0.5]  # halves are exact in binary
    result = tool.rule_b(r_base=r_base, r_fit=[0.5 * r for r in r_base])
    assert result.halved == (True, True)
    assert (result.verdict, result.branch) == (AEROSOL_LOBE, 2)


def test_rule_b_boundary_r_fit_one_ulp_over_half_is_neither() -> None:
    tool = load_tool()
    for i in range(2):
        r_base = [0.25, 0.5]
        r_fit = [0.125, 0.25]
        r_fit[i] = above(r_fit[i])
        result = tool.rule_b(r_base=r_base, r_fit=r_fit)
        assert result.halved[i] is False
        assert (result.verdict, result.branch) == (NEITHER, 3)


def test_rule_b_branch_3_neither_when_a_counted_load_is_not_halved() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[0.2, 0.4], r_fit=[0.05, 0.3])
    assert result.counted == (True, True)
    assert result.halved == (True, False)
    assert (result.verdict, result.branch) == (NEITHER, 3)


def test_rule_b_one_load_under_the_floor_while_the_other_halves_is_a_lobe() -> None:
    """A load with no excess at sun 0 has nothing for the lobe to explain: only the other counts."""
    tool = load_tool()
    for counted in (0, 1):
        r_base = [0.01, 0.01]
        r_fit = [0.5, 0.5]  # the uncounted load's fit is worse: it must not matter
        r_base[counted] = 0.4
        r_fit[counted] = 0.1
        result = tool.rule_b(r_base=r_base, r_fit=r_fit)
        assert result.counted == tuple(i == counted for i in range(2))
        assert (result.verdict, result.branch) == (AEROSOL_LOBE, 2)


def test_rule_b_one_load_under_the_floor_and_the_other_not_halved_is_neither() -> None:
    tool = load_tool()
    for counted in (0, 1):
        r_base = [0.01, 0.01]
        r_fit = [0.0, 0.0]  # the uncounted load is "fixed" completely: it must not rescue the rule
        r_base[counted] = 0.4
        r_fit[counted] = 0.3
        result = tool.rule_b(r_base=r_base, r_fit=r_fit)
        assert (result.verdict, result.branch) == (NEITHER, 3)


def test_rule_b_a_fit_that_makes_the_excess_worse_is_neither() -> None:
    tool = load_tool()
    result = tool.rule_b(r_base=[0.122], r_fit=[0.284])
    assert (result.verdict, result.branch) == (NEITHER, 3)


def test_rule_b_rejects_inputs_it_cannot_rule_on() -> None:
    tool = load_tool()
    with pytest.raises(ValueError, match="same number"):
        tool.rule_b(r_base=[0.2, 0.2], r_fit=[0.1])
    with pytest.raises(ValueError, match="finite"):
        tool.rule_b(r_base=[0.2, math.nan], r_fit=[0.1, 0.1])
    with pytest.raises(ValueError, match="negative"):
        tool.rule_b(r_base=[0.2], r_fit=[-0.1])  # r is a root mean square: unsigned


# --- the grid fit: g*(a) is the minimiser of r on the grid ----------------------------------------


def r_of(overrides: dict[int, float], default: float = 1.0) -> dict[int, float]:
    """r over the whole grid: `default` everywhere except at the overridden k."""
    tool = load_tool()
    return {k: overrides.get(k, default) for k in tool.ASYMMETRY_GRID_K}


def test_the_fit_picks_the_grid_value_with_the_smallest_r() -> None:
    tool = load_tool()
    assert tool.best_asymmetry_k(r_of({60: 0.035})) == 60
    assert tool.best_asymmetry_k(r_of({30: 0.5, 31: 0.6}, default=2.0)) == 30
    assert tool.best_asymmetry_k(r_of({90: 0.0})) == 90


def test_a_strictly_smaller_r_beats_a_tie_even_one_ulp_smaller_and_farther_from_70() -> None:
    tool = load_tool()
    r = r_of({70: 0.25, 40: below(0.25)})
    assert tool.best_asymmetry_k(r) == 40


def test_a_two_way_tie_goes_to_the_k_nearest_70() -> None:
    tool = load_tool()
    assert tool.best_asymmetry_k(r_of({66: 0.1, 55: 0.1})) == 66  # 4 away against 15
    assert tool.best_asymmetry_k(r_of({50: 0.1, 85: 0.1})) == 85  # the nearer one is the upper
    assert tool.best_asymmetry_k(r_of({70: 0.1, 40: 0.1})) == 70
    assert tool.best_asymmetry_k(r_of({30: 0.1, 90: 0.1})) == 90  # 40 away against 20


def test_an_equal_distance_tie_goes_to_the_lower_k() -> None:
    tool = load_tool()
    assert tool.best_asymmetry_k(r_of({65: 0.1, 75: 0.1})) == 65
    assert tool.best_asymmetry_k(r_of({60: 0.1, 80: 0.1})) == 60
    assert tool.best_asymmetry_k(r_of({50: 0.1, 90: 0.1})) == 50


def test_a_three_way_tie_is_resolved_by_the_same_two_steps() -> None:
    tool = load_tool()
    assert tool.best_asymmetry_k(r_of({50: 0.1, 66: 0.1, 74: 0.1})) == 66  # 66 and 74: equal, lower
    assert tool.best_asymmetry_k(r_of({50: 0.1, 66: 0.1, 73: 0.1})) == 73


def test_a_tie_that_is_not_at_the_minimum_is_ignored() -> None:
    tool = load_tool()
    assert tool.best_asymmetry_k(r_of({68: 0.3, 72: 0.3, 31: 0.2})) == 31


def test_the_fit_needs_the_whole_grid() -> None:
    tool = load_tool()
    short = r_of({}, default=1.0)
    del short[55]
    with pytest.raises(ValueError, match="grid"):
        tool.best_asymmetry_k(short)
    extra = r_of({}, default=1.0)
    extra[95] = 0.0
    with pytest.raises(ValueError, match="grid"):
        tool.best_asymmetry_k(extra)
    with pytest.raises(ValueError, match="finite"):
        tool.best_asymmetry_k(r_of({60: math.nan}))


# --- the excess: r and its signed mean ------------------------------------------------------------


def test_r_is_the_root_mean_square_of_the_log_ratio_and_the_mean_is_signed() -> None:
    tool = load_tool()
    model = np.array([2.0, 0.5, 2.0, 0.5])  # ln 2, -ln 2, ln 2, -ln 2
    prague = np.ones(4)
    r, mean = tool.excess(model, prague)
    assert r == pytest.approx(math.log(2.0), rel=1e-12, abs=0)
    assert mean == pytest.approx(0.0, abs=1e-15)
    r, mean = tool.excess(np.array([1.0, 1.0, 1.0, 1.0]) * math.exp(-0.25), prague)
    assert r == pytest.approx(0.25, rel=1e-12, abs=0)  # r is unsigned...
    assert mean == pytest.approx(-0.25, rel=1e-12, abs=0)  # ...the mean carries the sign


def test_r_squares_before_it_averages() -> None:
    """Unequal logs tell a root mean square from a mean absolute value."""
    tool = load_tool()
    model = np.exp(np.array([0.1, -0.3]))
    r, mean = tool.excess(model, np.ones(2))
    assert r == pytest.approx(math.sqrt((0.01 + 0.09) / 2), rel=1e-12, abs=0)
    assert r != pytest.approx(0.2, rel=1e-3, abs=0)  # the mean absolute log
    assert mean == pytest.approx(-0.1, rel=1e-12, abs=0)


def test_r_is_zero_when_the_model_is_prague() -> None:
    tool = load_tool()
    y = np.array([0.8, 0.1, 0.03])
    assert tool.excess(y, y) == (0.0, 0.0)


# --- the port bound -------------------------------------------------------------------------------


def test_the_port_bound_is_the_log_of_total_over_total_less_6_6_percent_of_ms() -> None:
    tool = load_tool()
    total, multiple = np.array([1.0, 0.5]), np.array([0.5, 0.25])
    expected = np.log(total / (total - multiple * 0.066 / 1.066))
    assert tool.port_bound(total, multiple) == pytest.approx(expected, rel=1e-12, abs=0)


def test_the_port_bound_is_never_negative_and_is_zero_without_multiple_scattering() -> None:
    tool = load_tool()
    total = np.array([1.0, 0.3, 0.02, 0.9])
    multiple = np.array([0.0, 0.1, 0.0199, 0.9 * 0.999])
    b = tool.port_bound(total, multiple)
    assert (b >= 0.0).all()
    assert b[0] == 0.0
    assert (b[1:] > 0.0).all()


def test_the_port_bound_grows_with_the_multiple_scattering_share() -> None:
    tool = load_tool()
    total = np.full(3, 1.0)
    b = tool.port_bound(total, np.array([0.1, 0.2, 0.4]))
    assert b[0] < b[1] < b[2]


def test_B_is_the_root_mean_square_of_b_over_the_views() -> None:
    tool = load_tool()
    total, multiple = np.array([1.0, 1.0]), np.array([0.2, 0.4])
    b = tool.port_bound(total, multiple)
    assert tool.port_bound_rms(total, multiple) == pytest.approx(
        math.sqrt(float(np.mean(b**2))), rel=1e-12, abs=0
    )
    assert tool.port_bound_rms(total, multiple) != pytest.approx(float(np.mean(b)), rel=1e-6, abs=0)


# --- the lobe diagnostic engine -------------------------------------------------------------------
#
# XYZ_g = v0's single scattering at asymmetry g plus v1's multiple scattering (asymmetry baked at
# 0.7). v1 refuses any other asymmetry, so it is never called with one: the tool builds XYZ_g from
# `sky_xyz(..., engine="v0")` and `multiple_scattering_xyz`.

SUNS = (10.0, 0.0)
LOADS = (0.03, 0.1)
ALL_AZIMUTHS = (0.0, 30.0, 90.0, 180.0)
GAP_VIEW = 1.51


def v1_reference(aod: float, sun: float, views: list[float], azimuth: float) -> np.ndarray:

    return sky_xyz(Atmosphere(aod_550=aod), sun, views, azimuth, engine="v1")


@pytest.mark.parametrize("azimuth", ALL_AZIMUTHS)
@pytest.mark.parametrize("aod", LOADS)
@pytest.mark.parametrize("sun", SUNS)
def test_the_lobe_engine_at_g_0_7_is_v1_exactly(sun: float, aod: float, azimuth: float) -> None:
    tool = load_tool()
    views = tool.view_elevations()
    lobe = tool.lobe_xyz(aod, 0.7, sun, views, azimuth)
    assert np.array_equal(lobe, v1_reference(aod, sun, views, azimuth))


def test_the_lobe_engine_is_the_registered_composition() -> None:
    """v0 at asymmetry g, plus the table's multiple scattering at v1's own atmosphere."""

    tool = load_tool()
    views = tool.view_elevations()
    lobe = tool.lobe_xyz(0.1, 0.55, 0.0, views, 0.0)
    expected = sky_xyz(
        Atmosphere(aod_550=0.1, asymmetry=0.55), 0.0, views, 0.0, engine="v0"
    ) + multiple_scattering_xyz(Atmosphere(aod_550=0.1), 0.0, views, 0.0)
    assert np.array_equal(lobe, expected)


@pytest.mark.parametrize("sun", SUNS)
@pytest.mark.parametrize("aod", LOADS)
def test_a_lower_asymmetry_lowers_luminance_at_a_toward_sun_low_view(
    sun: float, aod: float
) -> None:
    """The positive control: the diagnostic can move the quantity it is meant to explain."""
    tool = load_tool()
    views = tool.view_elevations()[:1]  # WHTM1's lowest band, 1.51 degrees
    y = [tool.lobe_xyz(aod, g, sun, views, 0.0)[0, 1] for g in (0.5, 0.6, 0.7, 0.8)]
    assert y[0] < y[1] < y[2] < y[3]


def test_a_lower_asymmetry_barely_moves_the_sky_away_from_the_sun() -> None:
    """Behind the observer (azimuth 180) the forward lobe is not what is being changed."""
    tool = load_tool()
    views = tool.view_elevations()[:1]
    toward = (
        tool.lobe_xyz(0.1, 0.6, 10.0, views, 0.0)[0, 1] / v1_reference(0.1, 10.0, views, 0.0)[0, 1]
    )
    away = (
        tool.lobe_xyz(0.1, 0.6, 10.0, views, 180.0)[0, 1]
        / v1_reference(0.1, 10.0, views, 180.0)[0, 1]
    )
    assert toward < 0.7  # toward the sun, the lobe carries most of the light
    assert abs(away - 1.0) < abs(toward - 1.0)


def test_the_engine_never_hands_v1_an_asymmetry_other_than_0_7() -> None:
    """v1 refuses one: the diagnostic is built from v0 and the table, so v1 is untouched."""
    with pytest.raises(ValueError):
        sky_xyz(Atmosphere(aod_550=0.1, asymmetry=0.6), 0.0, [1.5], 0.0, engine="v1")
    tool = load_tool()
    assert tool.lobe_xyz(0.1, 0.6, 0.0, [1.5], 0.0).shape == (1, 3)


# --- the sun elevation comes from the fixture, never from the module's +10 ------------------------


def test_engine_xyz_takes_the_sun_it_is_given() -> None:
    tool = load_tool()
    views = tool.view_elevations()
    for sun in SUNS:
        assert np.array_equal(
            tool.engine_xyz(0.1, sun, views, 90.0, "v1"), v1_reference(0.1, sun, views, 90.0)
        )
    assert not np.array_equal(
        tool.engine_xyz(0.1, 0.0, views, 90.0, "v1"), tool.engine_xyz(0.1, 10.0, views, 90.0, "v1")
    )


def test_e_over_p_at_sun_0_is_the_criterions_0_044527_from_v1_alone() -> None:
    tool = load_tool()
    e = tool.predicted_if_monte_carlo(0.0, tool.view_elevations(), tool.RULE_A_AZIMUTHS_DEG)
    assert e.shape == (16,)
    assert float(np.mean(e)) == pytest.approx(0.044527, rel=1e-4, abs=0)


def test_e_over_p_at_sun_10_is_what_the_pilot_log_reports() -> None:
    """The #34 log: expected if Monte Carlo, azimuths 90 and 180, is +0.0341."""
    tool = load_tool()
    e = tool.predicted_if_monte_carlo(10.0, tool.view_elevations(), tool.RULE_A_AZIMUTHS_DEG)
    assert float(np.mean(e)) == pytest.approx(0.0341, abs=5e-5)


def test_e_depends_on_the_sun_elevation() -> None:
    tool = load_tool()
    views = tool.view_elevations()
    sun0 = tool.predicted_if_monte_carlo(0.0, views, (90.0, 180.0))
    sun10 = tool.predicted_if_monte_carlo(10.0, views, (90.0, 180.0))
    assert not np.allclose(sun0, sun10, rtol=1e-2)


def test_e_is_the_port_bound_in_other_clothes() -> None:
    """e = exp(b) - 1: the same 6.6 percent of the multiple scattering, as a ratio."""
    tool = load_tool()
    views = tool.view_elevations()
    e = tool.predicted_if_monte_carlo(0.0, views, (90.0,))
    total = v1_reference(0.1, 0.0, views, 90.0)[:, 1]

    multiple = multiple_scattering_xyz(Atmosphere(aod_550=0.1), 0.0, views, 90.0)[:, 1]
    assert e == pytest.approx(np.expm1(tool.port_bound(total, multiple)), rel=1e-12, abs=0)


# --- selecting points by value, never by position -------------------------------------------------


def test_points_are_selected_by_azimuth_and_view_value() -> None:
    tool = load_tool()
    azimuth, view = tool.point_table([0.0, 30.0, 90.0, 180.0], [1.5, 2.5, 30.0])
    assert azimuth.tolist() == [0.0] * 3 + [30.0] * 3 + [90.0] * 3 + [180.0] * 3
    assert view.tolist() == [1.5, 2.5, 30.0] * 4
    mask = tool.mask_points(azimuth, view, azimuths=(90.0, 180.0))
    assert mask.sum() == 6
    assert set(azimuth[mask]) == {90.0, 180.0}
    only = tool.mask_points(azimuth, view, azimuths=(30.0,), views=(2.5,))
    assert only.sum() == 1


def test_a_permuted_point_table_selects_the_same_points() -> None:
    tool = load_tool()
    azimuths, views = [180.0, 0.0, 90.0, 30.0], [30.0, 1.5, 2.5]
    azimuth, view = tool.point_table(azimuths, views)
    mask = tool.mask_points(azimuth, view, azimuths=(90.0, 180.0))
    chosen = sorted(zip(azimuth[mask].tolist(), view[mask].tolist(), strict=True))
    assert chosen == sorted((a, v) for a in (90.0, 180.0) for v in views)


def test_a_view_is_matched_within_a_tolerance_not_by_exact_equality() -> None:
    tool = load_tool()
    azimuth, view = tool.point_table([0.0], [1.5066301585401483, 2.0])
    assert tool.mask_points(azimuth, view, views=(1.51,), tol=0.005).tolist() == [True, False]
    assert tool.mask_points(azimuth, view, views=(1.51,), tol=0.001).tolist() == [False, False]


def test_the_nominal_views_are_found_by_value_in_any_order() -> None:
    tool = load_tool()

    nominal = [float(x) for x in build_views()["nominal"].elevations_deg]
    stored = [60.0, *reversed(nominal), 30.0]
    indices = tool.nominal_indices(stored)
    assert [stored[i] for i in indices] == nominal  # in the camera's order, whatever the file's


def test_a_missing_nominal_view_is_refused_by_name() -> None:
    tool = load_tool()

    nominal = [float(x) for x in build_views()["nominal"].elevations_deg]
    with pytest.raises(ValueError, match=r"nominal view .* not in the fixture"):
        tool.nominal_indices([*nominal[:-1], 30.0, 60.0])
    with pytest.raises(ValueError, match="more than once"):
        tool.nominal_indices([*nominal, nominal[0], 30.0])


# --- the grid fit on a planted asymmetry (a positive control) -------------------------------------


def test_the_excess_curve_recovers_an_asymmetry_planted_in_a_stand_in_for_prague() -> None:
    """If Prague were XYZ_g at g = 0.55, r is exactly 0 there and large at v1's 0.7."""
    tool = load_tool()
    views = tool.view_elevations()[: tool.NOMINAL_VIEWS]
    planted = tool.lobe_xyz(0.1, 0.55, 0.0, views, 0.0)[:, 1]
    ks = (45, 50, 55, 60, 70)
    curve = tool.lobe_curve(0.1, 0.0, views, planted, ks=ks)
    assert curve.ks == ks
    assert curve.r_at(55) == 0.0
    assert curve.signed_mean_at(55) == 0.0
    assert curve.r_at(70) > tool.EXCESS_FLOOR
    assert curve.signed_mean_at(70) > 0.0  # v1 is brighter than the planted, weaker lobe
    assert min(ks, key=curve.r_at) == 55


def test_the_excess_curve_at_v1_itself_is_zero_at_k_70() -> None:
    tool = load_tool()
    views = tool.view_elevations()[: tool.NOMINAL_VIEWS]
    v1_y = v1_reference(0.03, 10.0, views, 0.0)[:, 1]
    curve = tool.lobe_curve(0.03, 10.0, views, v1_y, ks=(60, 70, 80))
    assert curve.r_at(70) == 0.0
    assert curve.r_at(60) > 0.0 and curve.r_at(80) > 0.0


# --- the fixture builder: sun, azimuths and path are parameters -----------------------------------

DATASET_SUN_AXIS = (-4.2, -4.0, 0.0, 10.0, 90.0)


def synthetic_axes() -> object:
    tool = load_tool()
    return tool.Axes(
        visibilities_km=(20.0, 131.8),
        albedos=(0.0, 1.0),
        altitudes_m=(0.0, 405.0),
        elevations_deg=DATASET_SUN_AXIS,
        channels=11,
        channel_start_nm=320.0,
        channel_width_nm=40.0,
    )


def test_the_default_spec_is_the_sun_10_fixture_exactly() -> None:
    tool = load_tool()
    spec = tool.SUN10
    assert spec.sun_elevation_deg == 10.0 == tool.SUN_ELEVATION_DEG
    assert spec.rel_azimuths_deg == (0.0, 90.0, 180.0) == tool.REL_AZIMUTHS_DEG
    assert spec.path == ROOT / "tests" / "fixtures" / "prague" / "sun10.json" == tool.FIXTURE
    assert spec.command == "uv run python tools/prague_reference.py fixture"
    assert tool.fixture_spec(10.0) is spec


def test_the_sun_0_spec_has_the_azimuth_30_the_gap_was_measured_at() -> None:
    tool = load_tool()
    spec = tool.SUN0
    assert spec.sun_elevation_deg == 0.0
    assert spec.rel_azimuths_deg == (0.0, 30.0, 90.0, 180.0)
    assert spec.path == ROOT / "tests" / "fixtures" / "prague" / "sun0.json" == tool.FIXTURE_SUN0
    assert spec.command == "uv run python tools/prague_reference.py fixture --sun 0"
    assert tool.fixture_spec(0.0) is spec
    assert tool.fixture_spec(0) is spec  # the command line says `--sun 0`


def test_an_unregistered_sun_has_no_fixture_and_an_output_path_moves_a_registered_one() -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match=r"no fixture is registered for sun elevation 5"):
        tool.fixture_spec(5.0)
    moved = tool.fixture_spec(0.0, ROOT / "elsewhere.json")
    assert moved.path == ROOT / "elsewhere.json"
    assert (moved.sun_elevation_deg, moved.rel_azimuths_deg) == (0.0, tool.SUN0.rel_azimuths_deg)
    assert moved.command == tool.SUN0.command  # how to remake it is the same either way


def test_the_command_line_defaults_to_the_sun_10_fixture() -> None:
    tool = load_tool()
    parser = tool.build_parser()
    assert parser.parse_args(["fixture"]).sun == 10.0
    assert parser.parse_args(["fixture", "--sun", "0"]).sun == 0.0
    assert parser.parse_args(["fixture", "--sun", "0", "--check"]).check is True
    assert parser.parse_args(["report"]).output is None


def test_an_unregistered_sun_on_the_command_line_is_an_error_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["fixture", "--sun", "5"]) == 1
    assert "no fixture is registered" in capsys.readouterr().err


@pytest.mark.parametrize("sun", [-4.2, -1.0, 0.0, 10.0, 90.0])
def test_a_sun_inside_the_datasets_axis_is_accepted_including_the_ends(sun: float) -> None:
    tool = load_tool()
    tool.check_sun_elevation(sun, synthetic_axes())


@pytest.mark.parametrize("sun", [-4.2000001, -10.0, 90.0000001, 120.0, math.nan, math.inf])
def test_a_sun_outside_the_datasets_axis_is_refused(sun: float) -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match=r"outside the dataset's sun axis, -4.2 to 90"):
        tool.check_sun_elevation(sun, synthetic_axes())


def channel_rows(count: int, value: float = 1.0) -> np.ndarray:
    return np.full((count, 11), value)


def test_xyz_block_lays_out_rows_azimuth_major_keyed_by_the_azimuth() -> None:
    tool = load_tool()
    channels = np.vstack([channel_rows(2, 1.0), channel_rows(2, 2.0), channel_rows(2, 3.0)])
    block = tool.xyz_block(channels, 320.0, 40.0, [1.5, 3.0], [0.0, 30.0, 180.0], where="test")
    assert list(block) == ["0", "30", "180"]
    assert all(len(rows) == 2 and all(len(row) == 3 for row in rows) for rows in block.values())
    # Y scales with the radiance: the second azimuth is twice, the third three times the first.
    first = block["0"][0][1]
    assert block["30"][0][1] == pytest.approx(2 * first, rel=1e-8)
    assert block["180"][1][1] == pytest.approx(3 * first, rel=1e-8)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_a_luminance_that_is_not_finite_and_above_zero_is_refused_naming_the_point(
    bad: float,
) -> None:
    tool = load_tool()
    channels = channel_rows(4)
    channels[3] = bad  # the second view of the second azimuth (30 degrees, view 3)
    with pytest.raises(tool.PragueError) as error:
        tool.xyz_block(
            channels, 320.0, 40.0, [1.5, 3.0], [0.0, 30.0], where="aod_550 0.03, reading xyz"
        )
    message = str(error.value)
    assert "aod_550 0.03, reading xyz" in message
    assert "relative azimuth 30, view 3 degrees" in message
    assert "not finite and above 0" in message


def test_the_binned_reading_is_refused_too_when_it_leaves_no_light() -> None:
    """Binned Prague holds zero outside its channels: a spectrum of zeros is no luminance."""
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="relative azimuth 0, view 1.5 degrees"):
        tool.xyz_block(channel_rows(1, 0.0), 320.0, 40.0, [1.5], [0.0], binned=True, where="x")


# --- the fixture builder is handed the spec's sun and azimuths ------------------------------------
#
# No Prague model is run here. `fetch_model`, `read_axes`, `evaluate` and the dataset table are
# replaced by deterministic stand-ins that record what they were asked, so these tests grade the
# arguments the builder threads through (the spec's sun and relative azimuths, the dataset's sun
# axis), not Prague. Without them nothing in the default run called `build_fixture` at all.


def install_prague_stand_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """Stand-ins for everything `build_fixture` reaches for; returns the log of `evaluate` calls."""
    tool = load_tool()
    recorded = json.loads(FIXTURE_SUN0.read_text())
    axes_info = recorded["provenance"]["dataset_axes"]
    centres = recorded["channel_centres_nm"]
    width = centres[1] - centres[0]
    axes = tool.Axes(
        tuple(axes_info["visibilities_km"]),
        tuple(axes_info["albedos"]),
        tuple(axes_info["altitudes_m"]),
        tuple(axes_info["elevations_deg"]),
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


def test_the_builder_evaluates_prague_at_the_specs_sun_and_azimuths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = install_prague_stand_in(monkeypatch, tmp_path)
    data = tool.build_fixture(tmp_path, tool.SUN0)
    assert data["sun_elevation_deg"] == 0.0
    assert data["rel_azimuths_deg"] == [0.0, 30.0, 90.0, 180.0]
    assert data["provenance"]["command"] == tool.SUN0.command
    assert calls, "the builder evaluated nothing"
    assert {call["sun"] for call in calls} == {0.0}  # every evaluation, not just the first
    sky = [call for call in calls if call["mode"] == "sky"]
    assert sky
    assert {azimuth for call in sky for azimuth, _ in call["queries"]} == {0.0, 30.0, 90.0, 180.0}
    for call in sky:
        assert len(call["queries"]) == 4 * 8  # four azimuths, eight views, none left out
    for entry in data["atmospheres"].values():
        assert list(entry["xyz"]) == ["0", "30", "90", "180"]
        for reading in READINGS:
            assert list(entry["diagnostics"][reading]) == ["0", "30", "90", "180"]


def test_the_default_spec_still_builds_the_sun_10_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = install_prague_stand_in(monkeypatch, tmp_path)
    data = tool.build_fixture(tmp_path)
    assert data["sun_elevation_deg"] == 10.0
    assert data["rel_azimuths_deg"] == [0.0, 90.0, 180.0]
    assert data["provenance"]["command"] == tool.SUN10.command
    assert {call["sun"] for call in calls} == {10.0}
    asked = {azimuth for call in calls if call["mode"] == "sky" for azimuth, _ in call["queries"]}
    assert asked == {0.0, 90.0, 180.0}
    assert list(data["atmospheres"]["0.1"]["xyz"]) == ["0", "90", "180"]


def test_the_builder_refuses_a_sun_outside_the_datasets_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prague would clamp 95 degrees to 90 without a word: the builder stops, naming the axis."""
    tool = load_tool()
    calls = install_prague_stand_in(monkeypatch, tmp_path)
    spec = tool.FixtureSpec(95.0, (0.0, 90.0), tmp_path / "x.json", "n/a")
    with pytest.raises(tool.PragueError, match="outside the dataset's sun axis"):
        tool.build_fixture(tmp_path, spec)
    assert calls == []  # refused before Prague was asked anything


# --- the sun-0 fixture ----------------------------------------------------------------------------

READINGS = (
    "xyz_binned",
    "xyz_inverse_visibility",
    "xyz_altitude_0",
    "xyz_matched_aerosol_depth",
)
"""The fixture's diagnostic readings of Prague, each laid out like the primary `xyz`."""
NOMINAL_VIEWS = 6
DIAGNOSTIC_ELEVATIONS = (30.0, 60.0)


@cache
def fixture_sun0() -> dict:
    return json.loads(FIXTURE_SUN0.read_text())


@cache
def fixture_sun10() -> dict:
    return json.loads(FIXTURE_SUN10.read_text())


def prague_block(data: dict, aod: float, variant: str = "xyz") -> dict:
    entry = data["atmospheres"][f"{aod:g}"]
    return entry["xyz"] if variant == "xyz" else entry["diagnostics"][variant]


def test_the_sun_0_fixture_has_the_documented_layout() -> None:
    data = fixture_sun0()
    assert data["format"] == "lastlight-prague-control"
    assert data["version"] == 1
    assert data["sun_elevation_deg"] == 0.0  # the fixture records its own sun
    assert data["observer_altitude_km"] == 0.34
    assert data["ground_albedo"] == 0.2
    assert [float(a) for a in data["rel_azimuths_deg"]] == [0.0, 30.0, 90.0, 180.0]
    views = data["view_elevations_deg"]
    assert len(views) == NOMINAL_VIEWS + len(DIAGNOSTIC_ELEVATIONS) == 8
    assert views[NOMINAL_VIEWS:] == list(DIAGNOSTIC_ELEVATIONS)
    assert set(data["atmospheres"]) == {"0.03", "0.1"}
    for aod in LOADS:
        assert data["atmospheres"][f"{aod:g}"]["aod_550"] == aod
        for variant in ("xyz", *READINGS):
            block = prague_block(data, aod, variant)
            assert list(block) == ["0", "30", "90", "180"]
            for azimuth in ("0", "30", "90", "180"):
                xyz = np.array(block[azimuth], dtype=float)
                assert xyz.shape == (8, 3), (aod, azimuth, variant)
                assert np.isfinite(xyz).all() and (xyz > 0.0).all(), (aod, azimuth, variant)


def test_the_sun_0_fixture_has_every_diagnostic_reading_sun_10_has() -> None:
    new, old = fixture_sun0(), fixture_sun10()
    assert new.keys() == old.keys()
    for aod in ("0.03", "0.1"):
        assert new["atmospheres"][aod].keys() == old["atmospheres"][aod].keys()
        assert (
            new["atmospheres"][aod]["diagnostics"].keys()
            == old["atmospheres"][aod]["diagnostics"].keys()
        )


def test_the_sun_0_fixture_views_are_the_cameras_nominal_views() -> None:
    """A refit of WHTM1 moves its bands: the control must be regenerated, not left stale."""
    nominal = build_views()["nominal"].elevations_deg
    recorded = fixture_sun0()["view_elevations_deg"][:NOMINAL_VIEWS]
    assert len(nominal) == NOMINAL_VIEWS
    assert recorded == pytest.approx(list(nominal), abs=1e-9), (
        "the camera's nominal band and reference elevations changed since the fixture was made: "
        "rerun tools/prague_reference.py fixture --sun 0"
    )
    assert fixture_sun0()["view_elevations_deg"] == fixture_sun10()["view_elevations_deg"]


def test_the_sun_0_fixture_maps_aod_to_visibility_as_sun_10_does() -> None:
    """Only the sun changed: Koschmieder at 550 nm, v1's own ground extinction."""
    tool = load_tool()
    for aod in LOADS:
        entry = fixture_sun0()["atmospheres"][f"{aod:g}"]
        assert entry["visibility_km"] == pytest.approx(
            tool.visibility_km(Atmosphere(aod_550=aod)), rel=1e-9
        )
        assert entry["visibility_km"] == fixture_sun10()["atmospheres"][f"{aod:g}"]["visibility_km"]
    low, high = fixture_sun0()["prague_visibility_range_km"]
    assert all(
        low <= fixture_sun0()["atmospheres"][a]["visibility_km"] <= high for a in ("0.03", "0.1")
    )


def test_the_matched_reading_is_clamped_at_0_03_as_the_criterion_says() -> None:
    """No visibility in Prague's range has an aerosol as thin as v1's 0.03: 131.8 km, kept."""
    thin = fixture_sun0()["atmospheres"]["0.03"]["diagnostics"]["matched_aerosol_depth"]
    assert thin["clamped_at_largest_visibility"] is True
    assert thin["visibility_km"] == 131.8
    thick = fixture_sun0()["atmospheres"]["0.1"]["diagnostics"]["matched_aerosol_depth"]
    assert thick["clamped_at_largest_visibility"] is False
    assert thick["prague_aerosol_depth"] == pytest.approx(0.1, abs=1e-4)


def test_the_sun_0_fixture_is_the_same_model_and_data_as_sun_10() -> None:
    new, old = fixture_sun0()["provenance"], fixture_sun10()["provenance"]
    assert new["model"] == old["model"]
    assert new["datasets"] == old["datasets"]
    assert new["dataset_axes"] == old["dataset_axes"]
    assert new["ground_level_dataset_matches_full_at_altitude_0"] is True
    assert new["command"] == "uv run python tools/prague_reference.py fixture --sun 0"
    assert len(new["model"]["commit"]) == 40 and len(new["datasets"]["full"]["sha256"]) == 64


def test_the_sun_0_sun_is_inside_the_datasets_sun_axis() -> None:
    axis = fixture_sun0()["provenance"]["dataset_axes"]["elevations_deg"]
    assert axis[0] <= fixture_sun0()["sun_elevation_deg"] <= axis[-1]
    assert -4.2 in axis and 0.0 in axis  # the model's sun axis reaches -4.2, and has a node at 0


def test_the_sun_10_fixture_is_untouched() -> None:
    """#34's fixture is pinned by its own control; this one only checks the sun it records."""
    assert fixture_sun10()["sun_elevation_deg"] == 10.0
    assert fixture_sun10()["rel_azimuths_deg"] == [0.0, 90.0, 180.0]


@pytest.mark.network
def test_the_sun_0_fixture_matches_a_fresh_evaluation_of_the_built_model() -> None:
    """`tools/prague_reference.py fixture --sun 0 --check` re-evaluates Prague and compares."""
    if not (CACHE / "build" / "prague_eval").exists():
        pytest.skip("the Prague model is not built: run tools/prague_reference.py fetch build")
    result = subprocess.run(
        [sys.executable, str(TOOL), "fixture", "--sun", "0", "--check"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- the rules run on the fixtures, pinned by value -----------------------------------------------
#
# Every pin is `pytest.approx(x, rel=1e-6, abs=0)`: a relative bar. (pytest's default absolute
# tolerance of 1e-12 would void a relative bar on a value this small, so abs is always 0.) The
# numbers were first computed after the rules and the fixture were committed, and are reproduced by
# `uv run python tools/prague_reference.py report --sun 0`. A pin that moves means the physics, the
# fixture or the rule moved: it is not a number to be updated until the reason is known.


def pinned(value: float) -> object:
    return pytest.approx(value, rel=1e-6, abs=0)


@cache
def rule_a_sun0() -> object:
    return load_tool().evaluate_rule_a(fixture_sun0())


@cache
def rule_a_sun10() -> object:
    return load_tool().evaluate_rule_a(fixture_sun10())


@cache
def rule_b() -> object:
    return load_tool().evaluate_rule_b(fixture_sun10(), fixture_sun0())


def test_rule_a_at_sun_0_statistics_verdict_and_branch() -> None:
    a = rule_a_sun0()
    r = a.result
    assert int(a.in_p.sum()) == 16
    assert pinned(-0.03370715339) == r.D
    assert pinned(0.04452681248) == r.E
    assert r.s == pinned(0.02687968216)
    assert r.m == pinned(0.02473118902)
    assert pinned(0.02687968216) == r.U
    assert r.s == r.U  # s is the larger of the two here
    assert r.D_alternatives["xyz_inverse_visibility"] == pinned(-0.04412763309)
    assert r.D_alternatives["xyz_matched_aerosol_depth"] == pinned(-0.05843834241)
    assert a.D_reported["xyz_binned"] == pinned(-0.02431444962)
    assert a.D_reported["xyz_altitude_0"] == pinned(-0.006056026403)
    assert r.verdict == "cannot tell"
    assert r.branch == 1
    assert r.E <= 2 * r.U  # branch 1: the comparison lacks the power to separate the predictions


def test_rule_a_reported_not_ruled_items_at_sun_0() -> None:
    tool = load_tool()
    a = rule_a_sun0()
    assert a.d.shape == (32,)
    assert float(a.d.mean()) == pinned(0.08478889701)
    assert float(a.d.std(ddof=1)) == pinned(0.1628234777)
    assert float(a.d.min()) == pinned(-0.0761619868)
    assert float(a.d.max()) == pinned(0.4118543606)
    assert float(a.e.mean()) == pinned(0.03631939902)
    groups = {
        0.0: (0.2942464439, 0.1686403943, 0.02242686327),
        30.0: (0.1123234509, 0.06572154008, 0.03379710783),
        90.0: (-0.0141346633, 0.01205125974, 0.04408326721),
        180.0: (-0.05327964347, 0.02296522768, 0.04497035775),
    }
    for azimuth, (mean_d, sd_d, mean_e) in groups.items():
        mask = tool.mask_points(a.azimuth, a.view, azimuths=(azimuth,))
        assert int(mask.sum()) == 8
        assert float(a.d[mask].mean()) == pinned(mean_d)
        assert float(a.d[mask].std(ddof=1)) == pinned(sd_d)
        assert float(a.e[mask].mean()) == pinned(mean_e)
    gap = tool._gap_mask(a)
    assert a.azimuth[gap][0] == 30.0 and a.view[gap][0] == pytest.approx(1.51, abs=0.005)
    assert a.d[gap][0] == pinned(0.1109421706)
    assert a.e[gap][0] == pinned(0.04846668752)


def test_rule_a_at_sun_10_reproduces_the_pilot_numbers_the_criterion_quotes() -> None:
    """The same code over sun10.json gives #34's log: s 0.019, m 0.025, E 0.034, U 0.025."""
    r = rule_a_sun10().result
    assert r.s == pytest.approx(0.019, abs=5e-4)
    assert r.m == pytest.approx(0.025, abs=5e-4)
    assert pytest.approx(0.034, abs=5e-4) == r.E
    assert pytest.approx(0.025, abs=5e-4) == r.U
    assert pytest.approx(0.0326, abs=5e-5) == r.D  # the log's "mean d +0.0326" (azimuths 90, 180)
    assert r.D_alternatives["xyz_inverse_visibility"] == pytest.approx(0.0228, abs=5e-5)
    assert r.D_alternatives["xyz_matched_aerosol_depth"] == pytest.approx(0.0076, abs=5e-5)
    assert pinned(0.03257880062) == r.D
    assert pinned(0.03413468513) == r.E
    assert r.s == pinned(0.01877069571)
    assert r.m == pinned(0.02498906954)
    assert pinned(0.02498906954) == r.U


def test_rule_a_at_sun_0_e_is_the_criterions_0_0445_from_v1_alone() -> None:
    assert pytest.approx(0.044527, rel=1e-4, abs=0) == rule_a_sun0().result.E


def test_rule_b_fits_g_star_at_sun_10_the_pilot_values_the_criterion_quotes() -> None:
    b = rule_b()
    assert b.k_star == {0.03: 67, 0.1: 60}
    low, high = b.pilot[0.03], b.pilot[0.1]
    assert low.r_at(70) == pytest.approx(0.122, abs=5e-4)  # "r at 0.7 is 0.122" at 0.03
    assert high.r_at(70) == pytest.approx(0.409, abs=5e-4)  # "falls from 0.409 ..."
    assert high.r_at(60) == pytest.approx(0.035, abs=5e-4)  # "... to 0.035"
    assert low.r_at(60) == pytest.approx(0.284, abs=5e-4)  # "0.60 applied at 0.03 ... r 0.284"
    assert low.r_at(70) == pinned(0.1220474098)
    assert low.r_at(67) == pinned(0.04283475751)
    assert high.r_at(70) == pinned(0.4088037393)
    assert high.r_at(60) == pinned(0.0352887899)
    assert low.signed_mean_at(70) == pinned(0.1162912794)
    assert low.signed_mean_at(67) == pinned(-0.01562037652)
    assert high.signed_mean_at(70) == pinned(0.4083876742)
    assert high.signed_mean_at(60) == pinned(-0.01326411714)


def test_the_fitted_asymmetry_depends_on_the_load() -> None:
    """One g cannot serve both loads: the criterion calls that evidence against a phase error."""
    b = rule_b()
    assert b.k_star[0.03] != b.k_star[0.1]
    cross_low = b.pilot[0.03].r_at(b.k_star[0.1])  # 0.60 applied at 0.03: worse than v1's own
    cross_high = b.pilot[0.1].r_at(b.k_star[0.03])
    assert cross_low > b.pilot[0.03].r_at(70)
    assert cross_low == pinned(0.2839919473)
    assert cross_high == pinned(0.2705908726)
    assert cross_high < b.pilot[0.1].r_at(70)  # but 0.67 applied at 0.1 still helps


def test_rule_b_at_sun_0_conditions_and_verdict() -> None:
    b = rule_b()
    low, high = b.sun0[0.03], b.sun0[0.1]
    assert low.r_at(70) == pinned(0.1253466303)
    assert low.r_at(67) == pinned(0.05601825059)
    assert high.r_at(70) == pinned(0.4322458811)
    assert high.r_at(60) == pinned(0.1785046002)
    assert low.signed_mean_at(70) == pinned(0.1061068562)
    assert low.signed_mean_at(67) == pinned(-0.001170362906)
    assert high.signed_mean_at(70) == pinned(0.4298290092)
    assert high.signed_mean_at(60) == pinned(0.1701009294)
    assert b.port_bound[(0.0, 0.03)] == pinned(0.0104572589)
    assert b.port_bound[(0.0, 0.1)] == pinned(0.02380760363)
    assert 2 * b.port_bound[(0.0, 0.03)] == pinned(2 * 0.0104572589)
    assert 2 * b.port_bound[(0.0, 0.1)] == pinned(2 * 0.02380760363)
    assert b.result.counted == (True, True)  # r at 0.7 is at least 0.05 at both loads
    assert b.result.halved == (True, True)
    assert b.result.verdict == "aerosol lobe"
    assert b.result.branch == 2


def test_the_port_bound_cannot_carry_half_of_any_excess_large_enough_to_count() -> None:
    """2B is under the 0.05 floor at both sun-0 conditions, and under half the sun +10 excess."""
    tool = load_tool()
    b = rule_b()
    assert 2 * b.port_bound[(0.0, 0.03)] < tool.EXCESS_FLOOR
    assert 2 * b.port_bound[(0.0, 0.1)] < tool.EXCESS_FLOOR
    assert b.port_bound[(0.0, 0.03)] == pytest.approx(0.0105, abs=5e-5)  # the criterion's figures
    assert b.port_bound[(0.0, 0.1)] == pytest.approx(0.0238, abs=5e-5)
    assert b.port_bound[(10.0, 0.03)] == pytest.approx(0.0058, abs=5e-5)
    assert b.port_bound[(10.0, 0.1)] == pytest.approx(0.0094, abs=5e-5)
    assert b.port_bound[(10.0, 0.03)] == pinned(0.005796576675)
    assert b.port_bound[(10.0, 0.1)] == pinned(0.009374451592)
    for aod in (0.03, 0.1):  # against half the sun +10 excess: 0.061 and 0.204
        assert b.port_bound[(10.0, aod)] < 0.5 * b.pilot[aod].r_at(70)


def test_the_grid_value_that_minimises_r_at_each_sun_0_condition_alone() -> None:
    b = rule_b()
    assert b.k_best_sun0 == {0.03: 67, 0.1: 48}
    assert b.sun0[0.03].r_at(67) == pinned(0.05601825059)
    assert b.sun0[0.1].r_at(48) == pinned(0.06398375669)


def test_the_side_and_back_sky_cost_of_fixing_the_lobe() -> None:
    tool = load_tool()
    b = rule_b()
    pins = {
        ("sun10", 0.03): (0.1090276495, 0.1158503825),
        ("sun10", 0.1): (0.1426222844, 0.1936970755),
        ("sun0", 0.03): (0.3046067259, 0.30686188),
        ("sun0", 0.1): (0.2698850025, 0.2751393217),
    }
    for (label, aod), (at_v1, at_fit) in pins.items():
        data = fixture_sun10() if label == "sun10" else fixture_sun0()
        assert tool.side_and_back_r(data, aod, 70) == pinned(at_v1)
        assert tool.side_and_back_r(data, aod, b.k_star[aod]) == pinned(at_fit)


def test_the_sun_10_control_recomputed_with_the_fitted_asymmetry() -> None:
    tool = load_tool()
    rows = tool.control_with_lobe(fixture_sun10(), rule_b().k_star[0.03])
    assert [row.azimuth for row in rows] == [0.0, 90.0, 180.0]
    expected = [
        (
            (0.01031409744, -0.04895865486),
            (0.01546092952, -0.05360277616),
            0.004930054482,
            0.003568036888,
        ),
        (
            (0.01948547443, -0.03821735726),
            (0.02010055397, -0.04040632428),
            0.00258655263,
            0.002518440931,
        ),
        (
            (0.02612575002, -0.05617136565),
            (0.02646713095, -0.05690095228),
            0.005213757498,
            0.005316384317,
        ),
    ]
    for row, (v1_contrast, lobe_contrast, v1_distance, lobe_distance) in zip(
        rows, expected, strict=True
    ):
        assert row.v1_contrast == pinned_pair(v1_contrast)
        assert row.lobe_contrast == pinned_pair(lobe_contrast)
        assert row.v1_distance == pinned(v1_distance)
        assert row.lobe_distance == pinned(lobe_distance)
        # The control's own tolerances, #34's: 0.10 per log ratio, 0.02 in u'v'.
        assert max(abs(x) for x in row.lobe_contrast) <= tool.CONTRAST_TOLERANCE
        assert row.lobe_distance <= tool.CHROMATICITY_TOLERANCE


def pinned_pair(pair: tuple[float, float]) -> object:
    return pytest.approx(pair, rel=1e-6, abs=0)


# --- the pipeline can say each thing: positive controls on synthetic stand-ins for Prague ---------


def with_prague_rows(
    data: dict, aod: float, azimuth: float, rows: np.ndarray, *, readings=True
) -> dict:
    """A copy of a fixture whose Prague at (aod, azimuth) is `rows`, in every reading."""
    new = copy.deepcopy(data)
    entry = new["atmospheres"][f"{aod:g}"]
    entry["xyz"][f"{azimuth:g}"] = rows.tolist()
    if readings:
        for variant in READINGS:
            entry["diagnostics"][variant][f"{azimuth:g}"] = rows.tolist()
    return new


def prague_that_is(data: dict, side: str) -> dict:
    """A fixture whose Prague has d = 0 (the port) or d = e (the Monte Carlo) at every point."""
    tool = load_tool()
    views, sun, azimuths = (
        data["view_elevations_deg"],
        data["sun_elevation_deg"],
        data["rel_azimuths_deg"],
    )
    e = tool.predicted_if_monte_carlo(sun, views, azimuths).reshape(len(azimuths), len(views))
    new = data
    for aod in LOADS:
        for i, azimuth in enumerate(azimuths):
            rows = tool.engine_xyz(aod, sun, views, azimuth, "v1")
            if aod == 0.1 and side == "monte_carlo":
                rows = rows / (1.0 + e[i])[:, None]
            new = with_prague_rows(new, aod, azimuth, rows)
    return new


def test_rule_a_says_port_when_prague_is_the_port() -> None:
    tool = load_tool()
    a = tool.evaluate_rule_a(prague_that_is(fixture_sun0(), "port"))
    assert a.result.D == 0.0 and a.result.s == 0.0 and a.result.m == 0.0
    assert (a.result.verdict, a.result.branch) == ("port", 2)


def test_rule_a_says_monte_carlo_when_prague_is_the_monte_carlo() -> None:
    tool = load_tool()
    a = tool.evaluate_rule_a(prague_that_is(fixture_sun0(), "monte_carlo"))
    assert pytest.approx(a.result.E, rel=1e-12, abs=0) == a.result.D
    assert a.result.E > 2 * a.result.U
    assert (a.result.verdict, a.result.branch) == ("Monte Carlo", 3)


def test_rule_a_says_cannot_tell_by_branch_4_when_prague_is_neither() -> None:
    """Prague 3 percent brighter at 0.1 than the Monte Carlo predicts: D falls between."""
    tool = load_tool()
    data = fixture_sun0()
    views, sun, azimuths = data["view_elevations_deg"], 0.0, data["rel_azimuths_deg"]
    e = tool.predicted_if_monte_carlo(sun, views, azimuths).reshape(len(azimuths), len(views))
    new = prague_that_is(data, "port")
    for i, azimuth in enumerate(azimuths):
        rows = tool.engine_xyz(0.1, sun, views, azimuth, "v1") / (1.0 + e[i] / 2.0)[:, None]
        new = with_prague_rows(new, 0.1, azimuth, rows)
    a = tool.evaluate_rule_a(new)
    assert a.result.E > 2 * a.result.U
    assert (a.result.verdict, a.result.branch) == ("cannot tell", 4)


def test_rule_b_says_no_excess_when_prague_is_v1_toward_the_sun() -> None:
    tool = load_tool()
    data = fixture_sun0()
    index = tool.nominal_indices(data["view_elevations_deg"])
    for aod in LOADS:
        rows = np.array(data["atmospheres"][f"{aod:g}"]["xyz"]["0"], dtype=float)
        rows[index] = tool.engine_xyz(
            aod, 0.0, [data["view_elevations_deg"][i] for i in index], 0.0, "v1"
        )
        data = with_prague_rows(data, aod, 0.0, rows, readings=False)
    result = tool.evaluate_rule_b(fixture_sun10(), data).result
    assert (result.verdict, result.branch) == ("no excess at sun 0", 1)


def test_rule_b_says_aerosol_lobe_when_prague_is_the_fitted_lobe() -> None:
    tool = load_tool()
    b = rule_b()
    data = fixture_sun0()
    index = tool.nominal_indices(data["view_elevations_deg"])
    views = [data["view_elevations_deg"][i] for i in index]
    for aod in LOADS:
        rows = np.array(data["atmospheres"][f"{aod:g}"]["xyz"]["0"], dtype=float)
        rows[index] = tool.lobe_xyz(aod, tool.asymmetry_of(b.k_star[aod]), 0.0, views, 0.0)
        data = with_prague_rows(data, aod, 0.0, rows, readings=False)
    result = tool.evaluate_rule_b(fixture_sun10(), data)
    assert [result.sun0[a].r_at(b.k_star[a]) for a in LOADS] == [0.0, 0.0]
    assert (result.result.verdict, result.result.branch) == ("aerosol lobe", 2)


def test_rule_b_says_neither_when_prague_is_brighter_than_v1_uniformly() -> None:
    """A lobe that is too strong cannot be the cause if v1 is too faint: g* only makes it worse."""
    tool = load_tool()
    data = fixture_sun0()
    index = tool.nominal_indices(data["view_elevations_deg"])
    views = [data["view_elevations_deg"][i] for i in index]
    for aod in LOADS:
        rows = np.array(data["atmospheres"][f"{aod:g}"]["xyz"]["0"], dtype=float)
        rows[index] = 1.2 * tool.engine_xyz(aod, 0.0, views, 0.0, "v1")
        data = with_prague_rows(data, aod, 0.0, rows, readings=False)
    result = tool.evaluate_rule_b(fixture_sun10(), data).result
    assert result.counted == (True, True) and result.halved == (False, False)
    assert (result.verdict, result.branch) == ("neither", 3)


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


def test_rule_a_does_not_depend_on_the_order_of_azimuths_and_views() -> None:
    tool = load_tool()
    here, there = rule_a_sun0().result, tool.evaluate_rule_a(permuted(fixture_sun0())).result
    for name in ("D", "E", "s", "m", "U"):
        assert getattr(there, name) == pytest.approx(getattr(here, name), rel=1e-12, abs=0), name
    assert (there.verdict, there.branch) == (here.verdict, here.branch)


def test_rule_b_does_not_depend_on_the_order_of_azimuths_and_views() -> None:
    tool = load_tool()
    there = tool.evaluate_rule_b(permuted(fixture_sun10()), permuted(fixture_sun0()))
    here = rule_b()
    assert there.k_star == here.k_star
    for aod in LOADS:
        assert there.sun0[aod].r_at(70) == pytest.approx(here.sun0[aod].r_at(70), rel=1e-12, abs=0)
        assert there.sun0[aod].r_at(60) == pytest.approx(here.sun0[aod].r_at(60), rel=1e-12, abs=0)
    assert (there.result.verdict, there.result.branch) == (here.result.verdict, here.result.branch)


def test_rule_b_refuses_fixtures_at_the_wrong_suns() -> None:
    """The fit is at sun +10 and the ruling at sun 0: swapping or repeating them is an error."""
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="rules at sun 0"):
        tool.evaluate_rule_b(fixture_sun10(), fixture_sun10())
    with pytest.raises(tool.PragueError, match="fits g.*at sun .10"):
        tool.evaluate_rule_b(fixture_sun0(), fixture_sun0())


# --- the report -----------------------------------------------------------------------------------


@cache
def sun0_report() -> str:
    return load_tool().report_sun0(fixture_sun10(), fixture_sun0())


def test_the_report_prints_both_verdicts_and_branches() -> None:
    text = sun0_report()
    assert "Verdict: **cannot tell** (branch 1)." in text
    assert "Verdict: **aerosol lobe** (branch 2)." in text


def test_the_report_prints_every_item_the_criterion_lists_for_rule_a() -> None:
    text = sun0_report()
    for needle in (
        "D, mean of d over P",
        "E, mean of e over P",
        "s, sd of d over P (ddof 1)",
        "m, largest shift of D",
        "U, the larger of s and m",
        "2U, the bar E must exceed",
        "`xyz_binned` (reported, not part of m)",
        "`xyz_altitude_0` (reported, not part of m)",
        "all 32",
        "azimuth 0 alone",
        "azimuth 30 alone",
        "azimuth 90 alone",
        "azimuth 180 alone",
        "the gap's own point (azimuth 30, view 1.51)",
    ):
        assert needle in text, needle


def test_the_report_says_branch_1_is_a_lack_of_power_and_what_would_tell() -> None:
    text = sun0_report()
    assert "This says the comparison lacks power, not anything about either model." in text
    assert "libRadtran" in text and "never vendored or imported" in text


def test_the_report_prints_every_item_the_criterion_lists_for_rule_b() -> None:
    text = sun0_report()
    for needle in (
        "#### The fit at sun +10 (the pilot)",
        "#### The ruled conditions: sun 0",
        "| B | 2B | counts (r at 0.7 >= 0.05) |",
        "signed mean at 0.7",
        "The sun +10 cross-load pilot",
        "The grid value that minimises r at each sun-0 condition alone",
        "Whether fixing the lobe costs the side and back sky",
        "#34's control at sun +10",
        "XYZ_g* within both",
        "which is itself evidence against a pure phase-function error",
        "a new renderer with its own registration, which the conductor files",
    ):
        assert needle in text, needle
    assert "Renderer v1 is unchanged by this." in text


def test_the_report_does_not_print_34s_old_aod_0_1_verdict() -> None:
    text = sun0_report()
    for old in (  # the wording of the #34 rule's output line and its verdicts
        "**sides with",
        "result: **",
        "predicted gap",
        "Rule (written before",
        "Which model does Prague side with",
    ):
        assert old not in text, old
    assert text.count("Verdict:") == 2  # rule A's and rule B's, and nothing else that reads as one


def test_the_34_report_refuses_the_sun_0_fixture() -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="not for sun 0"):
        tool.report(fixture_sun0())


def test_the_command_line_report_for_sun_0_runs_from_the_committed_fixtures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["report", "--sun", "0"]) == 0
    assert capsys.readouterr().out == sun0_report()


def test_the_report_rows_carry_the_pinned_numbers() -> None:
    """The printed tables are what goes in the build log: their numbers are pinned as text."""
    text = sun0_report()
    for row in (
        "| D, mean of d over P | -0.0337 |",
        "| E, mean of e over P | +0.0445 |",
        "| s, sd of d over P (ddof 1) | 0.0269 |",
        "| D under `xyz_inverse_visibility` (part of m) | -0.0441 |",
        "| D under `xyz_matched_aerosol_depth` (part of m) | -0.0584 |",
        "| m, largest shift of D over those two readings | 0.0247 |",
        "| U, the larger of s and m | 0.0269 |",
        "| 2U, the bar E must exceed | 0.0538 |",
        "| D under `xyz_binned` (reported, not part of m) | -0.0243 |",
        "| D under `xyz_altitude_0` (reported, not part of m) | -0.0061 |",
        "| all 32 | 32 | +0.0848 | 0.1628 | -0.0762 to +0.4119 | +0.0363 |",
        "| azimuth 0 alone | 8 | +0.2942 | 0.1686 | -0.0072 to +0.4119 | +0.0224 |",
        "| azimuth 30 alone | 8 | +0.1123 | 0.0657 | -0.0104 to +0.1597 | +0.0338 |",
        "| azimuth 90 alone | 8 | -0.0141 | 0.0121 | -0.0326 to +0.0035 | +0.0441 |",
        "| azimuth 180 alone | 8 | -0.0533 | 0.0230 | -0.0762 to -0.0222 | +0.0450 |",
        "| the gap's own point (azimuth 30, view 1.51) | 1 | +0.1109 | n/a | n/a | +0.0485 |",
        "| 0.03 | 0.67 | 0.1220 | 0.0428 | +0.1163 | -0.0156 |",
        "| 0.1 | 0.60 | 0.4088 | 0.0353 | +0.4084 | -0.0133 |",
        "| 0.03 | 0.1253 | 0.0560 | 0.447 | +0.1061 | -0.0012 | 0.0105 | 0.0209 | yes |",
        "| 0.1 | 0.4322 | 0.1785 | 0.413 | +0.4298 | +0.1701 | 0.0238 | 0.0476 | yes |",
        "| (sun 0, 0.03) | 0.3046 | 0.3069 |",
        "| 0 | +0.010, -0.049 | +0.015, -0.054 | 0.0049 | 0.0036 | yes |",
    ):
        assert row in text.splitlines(), row
    for row in (  # the exploratory section: every figure the build log quotes beyond the rulings
        "| 0.03 | 0 | 1.024 to 1.249 | 1.136 | 0.16 to 0.22 |",
        "| 0.03 | 90 | 1.254 to 1.400 | 1.352 | 0.25 to 0.58 |",
        "| 0.1 | 0 | 1.223 to 1.625 | 1.462 | 0.28 to 0.49 |",
        "| 0.1 | 180 | 1.240 to 1.308 | 1.282 | 0.29 to 0.97 |",
        "| 0.03 | 0.2252 | 0.2296 | +0.2236 | +0.2280 |",
        "| 0.1 | 0.3604 | 0.3625 | +0.3600 | +0.3622 |",
        "| (sun 0, 0.03) | 0.67 (g*(0.03)) | +0.097 +0.042 -0.000 -0.029 -0.049 -0.067 |",
        "| (sun 0, 0.1) | 0.48 (the sun-0 minimiser) | +0.108 +0.056 +0.008 -0.027 -0.054 -0.078 |",
    ):
        assert row in text.splitlines(), row
    assert "the rule would fall through to branch 4 (cannot tell) at the largest such U" in text
    assert "D +0.0326, E +0.0341, s 0.0188, m 0.0250, U 0.0250" in text
    assert "r 0.2840 against 0.1220 at 0.7 and 0.0428 at its own g*(0.03)" in text
    assert "(sun 0, 0.1): g = 0.48, r 0.0640" in text


# --- the #34 report at sun +10: its numbers, pinned by value --------------------------------------
#
# `report(sun10.json)` printed these tables before #45 touched the tool, and the #34 build log
# quotes them (0.1380 is the spread of d over the 24 points, the number its aod 0.1 rule is decided
# on).
# Nothing graded them, so a changed number went unseen (mutation E1: the table's sd with ddof 0).
# They are pinned by value, never by a hash of the printed text. The quantities of the aod 0.1 rule
# are pinned at full precision with `pinned` and the report must print exactly those, to its four
# decimals; the other tables are pinned at the precision they print, number by number.

RULE_34_ALL = "all 24"
RULE_34_AWAY = "azimuth 90, 180 (16)"
RULE_34_READINGS = (
    "Koschmieder mapping (the rule's reading)",
    "visibility read in 1 / V",
    "aerosol depth matched",
)
# (reading, points) -> mean d, sd of d (ddof 1), min d, max d, mean of the expected column
RULE_34_PINS = {
    (RULE_34_READINGS[0], RULE_34_ALL): (
        0.11812996731091342,
        0.13801360494415948,
        -0.0020563629284355844,
        0.372533895185271,
        0.02630972164683536,
    ),
    (RULE_34_READINGS[0], RULE_34_AWAY): (
        0.03257880062457766,
        0.018770695710166655,
        -0.0020563629284355844,
        0.062088876067345256,
        0.03413468512840513,
    ),
    (RULE_34_READINGS[1], RULE_34_ALL): (
        0.10623795586989511,
        0.13211946207300945,
        0.0013458651080480166,
        0.35337644987362715,
        0.02630972164683536,
    ),
    (RULE_34_READINGS[1], RULE_34_AWAY): (
        0.02283951512227264,
        0.012825539286916944,
        0.0013458651080480166,
        0.0439373345432712,
        0.03413468512840513,
    ),
    (RULE_34_READINGS[2], RULE_34_ALL): (
        0.083628438422715,
        0.11792143326862765,
        -0.018767334117661738,
        0.30782642449041897,
        0.02630972164683536,
    ),
    (RULE_34_READINGS[2], RULE_34_AWAY): (
        0.007589731079940473,
        0.011445232096974707,
        -0.018767334117661738,
        0.026890806252619415,
        0.03413468512840513,
    ),
}
RULE_34_PREDICTED = 0.02630972164683536  # mean of the expected column over all 24 points
RULE_34_SPREAD = 0.13801360494415948  # sd of d over all 24 points: "0.1380" in the #34 log
RULE_34_SHIFT = 0.014092184014259093  # mean |d - d in 1 / V| over all 24 points


@cache
def sun10_report() -> str:
    return load_tool().report(fixture_sun10())


def sun10_rule_34_d() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """d under each reading, the expected column and the azimuth 90, 180 mask, from the tool."""
    tool = load_tool()
    data = fixture_sun10()
    azimuth, view = tool.point_table(data["rel_azimuths_deg"], data["view_elevations_deg"])
    away = tool.mask_points(azimuth, view, azimuths=tool.RULE_A_AZIMUTHS_DEG)
    d = {
        RULE_34_READINGS[0]: tool._ratio_of_ratios(data) - 1.0,
        RULE_34_READINGS[1]: tool._ratio_of_ratios(data, "xyz_inverse_visibility") - 1.0,
        RULE_34_READINGS[2]: tool._ratio_of_ratios(data, "xyz_matched_aerosol_depth") - 1.0,
    }
    return d, tool._predicted_if_monte_carlo(data), away


def test_the_34_rule_quantities_are_what_the_34_log_reports() -> None:
    d, expected, away = sun10_rule_34_d()
    assert int(away.sum()) == 16 and d[RULE_34_READINGS[0]].size == 24
    for (reading, points), pin in RULE_34_PINS.items():
        mask = away if points == RULE_34_AWAY else np.ones_like(away)
        x = d[reading][mask]
        stats = (x.mean(), x.std(ddof=1), x.min(), x.max(), expected[mask].mean())
        assert stats == pytest.approx(pin, rel=1e-6, abs=0), (reading, points)
    primary = d[RULE_34_READINGS[0]]
    assert float(np.std(primary, ddof=1)) == pinned(RULE_34_SPREAD)
    assert float(np.mean(np.abs(primary - d[RULE_34_READINGS[1]]))) == pinned(RULE_34_SHIFT)
    assert float(np.mean(expected)) == pinned(RULE_34_PREDICTED)
    assert float(np.std(primary, ddof=1)) == pytest.approx(0.1380, abs=5e-5)  # the #34 log's


def test_the_34_report_prints_the_pinned_rule_table_to_four_decimals() -> None:
    lines = sun10_report().splitlines()
    assert len(RULE_34_PINS) == 6
    for (reading, points), (mean, sd, low, high, expected) in RULE_34_PINS.items():
        row = (
            f"| {reading} | {points} | {mean:+.4f} | {sd:.4f} | {low:+.4f} to {high:+.4f} "
            f"| {expected:+.4f} |"
        )
        assert row in lines, row


def test_the_34_report_prints_the_rule_sentence_with_the_pinned_numbers_and_its_verdict() -> None:
    text = sun10_report()
    assert (
        f"the predicted gap ({RULE_34_PREDICTED:+.4f}) exceeds both the spread of d across the "
        f"views and azimuths ({RULE_34_SPREAD:.4f}) and the shift from reading Prague's "
        f"visibility in 1 / V ({RULE_34_SHIFT:.4f}); result: **cannot tell**."
    ) in text
    assert text.count("result: **") == 1  # one verdict, and it is the registered one


def report_tables(text: str) -> list[tuple[str, list[tuple[float, ...]]]]:
    """Every pipe table of a report but the aod 0.1 rule's: heading, then the numbers per row."""
    number = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?!\w)")
    tables: list[tuple[str, list[tuple[float, ...]]]] = []
    heading, rows, seen = "", [], 0  # `seen` counts the table's lines: header, rule, then rows
    for line in [*text.splitlines(), ""]:
        if line.startswith("### "):
            heading = line
        if line.startswith("|"):
            seen += 1
            if seen > 2:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                rows.append(tuple(float(x) for cell in cells for x in number.findall(cell)))
        elif seen:
            if not heading.startswith("### Which model"):
                tables.append((heading, rows))
            rows, seen = [], 0
    return tables


SUN10_REPORT_TABLES: dict[str, tuple[str, list[tuple[float, ...]]]] = {
    # azimuth; Prague log(R/G), log(B/G); v1 minus Prague; v0 minus Prague; u'v' v1; v0
    "control_0_03": (
        "### Sun +10, `aod_550` 0.03,",
        [
            (0.0, 0.134, -0.174, 0.01, -0.049, 0.008, -0.051, 0.0049, 0.0103),
            (90.0, 0.182, -0.149, 0.019, -0.038, 0.046, -0.121, 0.0026, 0.024),
            (180.0, 0.172, -0.122, 0.026, -0.056, 0.056, -0.135, 0.0052, 0.0218),
        ],
    ),
    # the same columns, the diagnostic load
    "control_0_1": (
        "### Sun +10, `aod_550` 0.1,",
        [
            (0.0, 0.107, -0.118, 0.01, -0.029, 0.015, -0.048, 0.0088, 0.0136),
            (90.0, 0.114, -0.061, 0.021, -0.014, 0.075, -0.116, 0.0027, 0.0306),
            (180.0, 0.087, -0.005, 0.026, -0.018, 0.096, -0.114, 0.0049, 0.031),
        ],
    ),
    # azimuth; view; Prague Y; v1 / Prague; v0 / Prague; MS share
    "totals_0_03": (
        "### `aod_550` 0.03 (visibility",
        [
            (0.0, 1.51, 0.81624, 1.19, 1.077, 0.095),
            (0.0, 1.96, 0.79816, 1.158, 1.049, 0.094),
            (0.0, 2.41, 0.77669, 1.132, 1.026, 0.094),
            (0.0, 2.86, 0.75416, 1.11, 1.007, 0.093),
            (0.0, 3.32, 0.73192, 1.091, 0.99, 0.093),
            (0.0, 3.99, 0.70048, 1.064, 0.967, 0.092),
            (0.0, 30.0, 0.10272, 0.958, 0.829, 0.134),
            (0.0, 60.0, 0.02794, 1.005, 0.784, 0.219),
            (90.0, 1.51, 0.10201, 1.091, 0.624, 0.428),
            (90.0, 1.96, 0.10246, 1.089, 0.636, 0.416),
            (90.0, 2.41, 0.10215, 1.086, 0.644, 0.407),
            (90.0, 2.86, 0.10117, 1.081, 0.65, 0.399),
            (90.0, 3.32, 0.09971, 1.077, 0.655, 0.392),
            (90.0, 3.99, 0.09692, 1.071, 0.66, 0.384),
            (90.0, 30.0, 0.03149, 1.03, 0.701, 0.319),
            (90.0, 60.0, 0.01883, 1.046, 0.742, 0.29),
            (180.0, 1.51, 0.12819, 1.183, 0.773, 0.347),
            (180.0, 1.96, 0.13166, 1.177, 0.788, 0.331),
            (180.0, 2.41, 0.13383, 1.168, 0.796, 0.319),
            (180.0, 2.86, 0.13471, 1.159, 0.801, 0.309),
            (180.0, 3.32, 0.13456, 1.15, 0.804, 0.301),
            (180.0, 3.99, 0.1329, 1.138, 0.806, 0.291),
            (180.0, 30.0, 0.0402, 1.076, 0.799, 0.258),
            (180.0, 60.0, 0.01868, 1.085, 0.773, 0.287),
        ],
    ),
    # the same columns
    "totals_0_1": (
        "### `aod_550` 0.1 (visibility",
        [
            (0.0, 1.51, 0.79304, 1.544, 1.294, 0.162),
            (0.0, 1.96, 0.81453, 1.529, 1.289, 0.157),
            (0.0, 2.41, 0.82902, 1.513, 1.283, 0.152),
            (0.0, 2.86, 0.83658, 1.498, 1.276, 0.148),
            (0.0, 3.32, 0.83964, 1.483, 1.269, 0.144),
            (0.0, 3.99, 0.8383, 1.461, 1.256, 0.14),
            (0.0, 30.0, 0.15818, 1.179, 0.977, 0.171),
            (0.0, 60.0, 0.03678, 1.051, 0.749, 0.288),
            (90.0, 1.51, 0.07814, 1.116, 0.413, 0.63),
            (90.0, 1.96, 0.07998, 1.121, 0.431, 0.616),
            (90.0, 2.41, 0.08153, 1.123, 0.448, 0.601),
            (90.0, 2.86, 0.08267, 1.124, 0.463, 0.589),
            (90.0, 3.32, 0.08343, 1.122, 0.475, 0.576),
            (90.0, 3.99, 0.08391, 1.12, 0.491, 0.562),
            (90.0, 30.0, 0.0372, 1.048, 0.6, 0.428),
            (90.0, 60.0, 0.02282, 1.05, 0.654, 0.377),
            (180.0, 1.51, 0.07239, 1.205, 0.455, 0.622),
            (180.0, 1.96, 0.07572, 1.217, 0.491, 0.597),
            (180.0, 2.41, 0.07905, 1.222, 0.523, 0.572),
            (180.0, 2.86, 0.08213, 1.222, 0.549, 0.55),
            (180.0, 3.32, 0.08482, 1.217, 0.571, 0.531),
            (180.0, 3.99, 0.08797, 1.209, 0.595, 0.507),
            (180.0, 30.0, 0.04186, 1.094, 0.679, 0.379),
            (180.0, 60.0, 0.02126, 1.083, 0.665, 0.386),
        ],
    ),
    # per reading (xyz, binned, 1 / V, altitude 0, matched): dR/G, dB/G, u'v' at azimuth 0, 90, 180
    "other_readings": (
        "### The other readings of Prague",
        [
            (0.01, -0.049, 0.0049, 0.019, -0.038, 0.0026, 0.026, -0.056, 0.0052),
            (0.013, -0.048, 0.0081, 0.022, -0.032, 0.0055, 0.028, -0.049, 0.0095),
            (0.01, -0.048, 0.0051, 0.019, -0.037, 0.0025, 0.025, -0.054, 0.0053),
            (0.008, -0.045, 0.0021, 0.024, -0.045, 0.0045, 0.028, -0.066, 0.0032),
            (0.008, -0.045, 0.0057, 0.016, -0.032, 0.0024, 0.022, -0.048, 0.0055),
        ],
    ),
    # aod_550; mapped visibility; Prague's depth there; matched visibility; its depth there
    "aerosol_depth": (
        "### Prague's own vertical aerosol",
        [
            (0.03, 123.82, 0.041, 131.8, 0.037),
            (0.1, 49.99, 0.114, 55.53, 0.1),
        ],
    ),
}


def test_the_34_report_tables_carry_the_pinned_numbers() -> None:
    tables = report_tables(sun10_report())
    assert len(tables) == len(SUN10_REPORT_TABLES) == 6
    for (heading, rows), (label, (needle, pinned_rows)) in zip(
        tables, SUN10_REPORT_TABLES.items(), strict=True
    ):
        assert heading.startswith(needle), label
        assert len(rows) == len(pinned_rows), label
        for row, pinned_row in zip(rows, pinned_rows, strict=True):
            assert row == pytest.approx(pinned_row, rel=1e-6, abs=0), (label, row)


def test_the_report_parser_reads_the_numbers_of_a_row_and_skips_names() -> None:
    text = "### H\n\n| a | b |\n| --- | --- |\n| xyz_altitude_0 | +0.010, -0.049, 0.0049 | 3 |\n"
    assert report_tables(text) == [("### H", [(0.01, -0.049, 0.0049, 3.0)])]
