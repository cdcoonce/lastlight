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

import importlib.util
import json
import math
import subprocess
import sys
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
