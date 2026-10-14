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
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prague_reference.py"


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
        d_alternatives=alternatives(a=0.5, b=0.5),
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
