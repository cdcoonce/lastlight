"""Prague's aerosol amount from Prague's own gas depths: no test (#55).

The route is registered as "no test" in `docs/renderer/prague-aerosol-depth-criterion.md` (landed
in PR #56). The tool prints the evidence that data in hand can give, under a "pilot" heading: the
eight gas estimates fitted to the two committed spectra, the crossings, and the invariance checks
of the criterion's problem 1. This file pins that output:

* every number the criterion shows, within half a unit of its last printed digit, computed from the
  tool's unrounded values (`assert_shown`);
* the same quantities by value, `pytest.approx(ref, rel=1e-6, abs=0)`, against references computed
  in an independent prototype (never a hash of float output or of report text);
* the fit's definitions (channels, model, start, settings, standard error, refusals);
* the command: `report --depth` writes exactly what the report function returns, runs no Prague
  evaluation and writes no file.

No Prague model is run anywhere: the tests read the four committed fixtures and v1 only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import pytest
from scipy.optimize import least_squares

import lastlight.render as render_package
from lastlight.render import Atmosphere
from lastlight.render.atmosphere import ozone_absorption_per_atm_cm, rayleigh_optical_depth

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prague_reference.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "prague"
CRITERION = ROOT / "docs" / "renderer" / "prague-aerosol-depth-criterion.md"


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


def pinned(value: float) -> object:
    """A relative pin: pytest's default absolute tolerance would void it on small values."""
    return pytest.approx(value, rel=1e-6, abs=0)


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
    assert shown(14.1475, 14.1, 1)
    assert not shown(14.1501, 14.1, 1)
    assert shown(14_699.4, 1.470e4, -1)  # a last digit in the tens: half a unit is 5
    assert not shown(14_706.0, 1.470e4, -1)


# --- the fixtures ---------------------------------------------------------------------------------

FIXTURE_FILES = {10.0: "sun10.json", 0.0: "sun0.json", -1.0: "sun-1.json", 1.0: "sun1.json"}
FOUR_SUNS = (10.0, 0.0, -1.0, 1.0)
CRITERION_SUNS = (-1.0, 0.0, 1.0, 10.0)
"""The order the criterion lists the suns in: its crossings are printed at -1, 0, +1, +10."""
LOADS = (0.03, 0.1)


@cache
def _committed(sun: float) -> dict:
    return json.loads((FIXTURE_DIR / FIXTURE_FILES[sun]).read_text())


def committed(sun: float) -> dict:
    return copy.deepcopy(_committed(sun))


def fresh_fixtures() -> dict[float, dict]:
    """Copies of the four committed fixtures, safe to edit."""
    return {sun: committed(sun) for sun in FOUR_SUNS}


def spectrum(data: dict, aod: float) -> list[float]:
    return data["atmospheres"][f"{aod:g}"]["diagnostics"]["zenith_transmittance"]


@cache
def pilot() -> object:
    """The tool's pilot on the four committed fixtures (the tool must not edit its input)."""
    return load_tool().depth_pilot({sun: _committed(sun) for sun in FOUR_SUNS})


@cache
def report_text() -> str:
    return load_tool().report_depth({sun: _committed(sun) for sun in FOUR_SUNS})


# --- what the criterion shows ---------------------------------------------------------------------
#
# The pilot table, as printed: G, 2SE, g, g - 2SE, g + 2SE, residual rms, a_R, ozone (DU) and the
# condition number, each with the digits it is printed to (the condition number's last digit is in
# the ones for 882 and 912, the tenths for 17.5, 14.1, 87.2 and 76.9, the tens for 1.470e4 and
# 1.033e4).

SHOWN_DIGITS = (4, 4, 4, 4, 4, 5, 4, 1)
SHOWN_CONDITION_DIGITS = {1: 0, 2: 1, 3: 0, 4: 1, 5: -1, 6: 1, 7: -1, 8: 1}
SHOWN = {
    1: (0.1149, 0.0032, 0.0208, 0.0176, 0.0241, 0.00096, 0.0859, 327.4, 882),
    2: (0.1233, 0.0022, 0.0124, 0.0102, 0.0147, 0.00182, 0.0930, 341.0, 17.5),
    3: (0.1171, 0.0030, 0.0186, 0.0156, 0.0216, 0.00114, 0.0866, 345.0, 912),
    4: (0.1237, 0, 0.0121, 0.0121, 0.0121, 0.00182, 0.0930, 345.0, 14.1),
    5: (0.1144, 0.0280, 0.0214, -0.0066, 0.0494, 0.00081, 0.0853, 327.9, 1.470e4),
    6: (0.1214, 0.0013, 0.0144, 0.0131, 0.0156, 0.00083, 0.0930, 318.5, 87.2),
    7: (0.0990, 0.0601, 0.0367, -0.0234, 0.0969, 0.00092, 0.0686, 345.0, 1.033e4),
    8: (0.1237, 0, 0.0121, 0.0121, 0.0121, 0.00117, 0.0930, 345.0, 76.9),
}
"""The pilot's eight estimates, as the criterion's table prints them (2SE twice the SE)."""


class Pinned(NamedTuple):
    """One estimate's quantities by value, from an independent prototype (not the tool)."""

    gas: float
    se: float
    excess: float
    rms: float
    a_r: float
    ozone_du: float
    condition: float


PINNED = {
    1: Pinned(
        0.11493432685488959,
        0.0016186808945007141,
        0.020816290397279683,
        0.0009625468969361734,
        0.08589673829033148,
        327.44155953972506,
        881.7487722572441,
    ),
    2: Pinned(
        0.12332746339937095,
        0.001116241321336982,
        0.012423153852798324,
        0.0018174048406101077,
        0.09304979922170653,
        341.0337059072076,
        17.45724551926898,
    ),
    3: Pinned(
        0.11714334309054616,
        0.0014991434883228974,
        0.01860727416162311,
        0.0011352061483916238,
        0.08658657106453596,
        345.0,
        911.6684594987447,
    ),
    4: Pinned(
        0.12366912514595116,
        0.0,
        0.012081492106218117,
        0.0018231290768816546,
        0.09304979922170653,
        345.0,
        14.147488915084997,
    ),
    5: Pinned(
        0.11437405051098207,
        0.014012243279039965,
        0.02137656674118721,
        0.0008061695237377817,
        0.08530373861675154,
        327.88806584136205,
        14699.405174565132,
    ),
    6: Pinned(
        0.12139025160395789,
        0.0006308959634504234,
        0.014360365648211387,
        0.0008262066129804541,
        0.09304979922170653,
        318.54493806140306,
        87.23557981233105,
    ),
    7: Pinned(
        0.09903218286431412,
        0.030066926788235047,
        0.03671843438785516,
        0.0009163513336969127,
        0.06864901817978118,
        345.0,
        10331.964297897275,
    ),
    8: Pinned(
        0.12366912514595116,
        0.0,
        0.012081492106218117,
        0.0011723577779113022,
        0.09304979922170653,
        345.0,
        76.90275342471637,
    ),
}

FITTING_GAS = (1, 2, 3, 5, 6, 7)
"""The six estimates that fit a gas parameter: every one but 4 and 8."""
FITTING_NO_GAS = (4, 8)

# The tool's crossings at each sun, by value (an independent prototype on the committed fixtures).
# sun: (g_old, delta_m, b, L, g*, carrying excess, delta_m / b, delta_m g_4 / (g_old b))
CROSSING = {
    -1.0: (
        0.013670441763468016,
        0.04012566281655605,
        0.027721239806958398,
        0.3175745049881771,
        0.009444369707353337,
        0.054097346576562726,
        1.4474699939821587,
        1.2792269342030873,
    ),
    0.0: (
        0.013670441763468016,
        0.05213889570280311,
        0.036168163943984366,
        0.26750937236089,
        0.009483031280641796,
        0.03506951237408755,
        1.441568772568978,
        1.274011626522913,
    ),
    1.0: (
        0.013670441763468016,
        0.06387801830431261,
        0.04451922240239908,
        0.24577917500104757,
        0.009527494016917382,
        0.026299421833333524,
        1.4348412855672499,
        1.2680660921713265,
    ),
    10.0: (
        0.013670441763468016,
        0.03745040408749438,
        0.02600435704630949,
        0.13249007040044167,
        0.009492315430495191,
        0.024181285032540134,
        1.4401588172628672,
        1.2727655538505271,
    ),
}
GAS_OLD = 0.12208157548870124
GAS_PAPER = 0.12366912514595116
GAS_INTERPOLATION = 0.0012914879790091593
GAS_PRESSURE = 0.00029606167824075547
Y_BAR_CLEAREST = 0.15893257548870124
"""The 0.03 load's stored clamped depth, 0.036851, plus the old gas."""

# --- the model, written out again from the criterion ----------------------------------------------

CENTRES = np.array([380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0, 700.0, 740.0])
X = np.log(CENTRES / 550.0)
RHO = rayleigh_optical_depth(CENTRES, 1013.25) / rayleigh_optical_depth(550.0, 1013.25)
KAPPA = ozone_absorption_per_atm_cm(CENTRES)
P_PAPER = 1013.25 * (1 - 0.0065 * 340 / 288.15) ** 5.25588
R_PAPER = float(rayleigh_optical_depth(550.0, P_PAPER))
RHO_BAR = float(np.interp(550.0, CENTRES, RHO))
KAPPA_BAR = float(np.interp(550.0, CENTRES, KAPPA))
OZONE_PAPER = 0.345

ESTIMATES = {  # number: (curved, ozone free, Rayleigh free), the form varying slowest
    1: (False, True, True),
    2: (False, True, False),
    3: (False, False, True),
    4: (False, False, False),
    5: (True, True, True),
    6: (True, True, False),
    7: (True, False, True),
    8: (True, False, False),
}


def model_residuals(number: int, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The model minus the data for estimate `number` at `theta`, one visibility after another."""
    curved, ozone_free, rayleigh_free = ESTIMATES[number]
    width = 3 if curved else 2
    index = 0
    a_r, a_o = R_PAPER, OZONE_PAPER
    if rayleigh_free:
        a_r, index = theta[index], index + 1
    if ozone_free:
        a_o, index = theta[index], index + 1
    blocks = theta[index:].reshape(-1, width)
    rows = []
    for (u, alpha, *gamma), row in zip(blocks, y, strict=True):
        curvature = gamma[0] if gamma else 0.0
        aerosol = np.exp(u - alpha * X + curvature * X**2)
        rows.append(a_r * RHO + a_o * KAPPA + aerosol - row)
    return np.concatenate(rows)


def finite_difference_jacobian(number: int, theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    columns = []
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6
        columns.append(
            (model_residuals(number, theta + step, y) - model_residuals(number, theta - step, y))
            / 2e-6
        )
    return np.column_stack(columns)


def pilot_y() -> np.ndarray:
    """-ln T over the ten fitted channels, one row per load (the 0.03 load first)."""
    data = _committed(10.0)
    return np.array([-np.log(spectrum(data, aod))[1:] for aod in LOADS])


def synthetic_rows(tau_550: float = 0.2, noise: float = 0.0) -> list[list[float]]:
    """Eleven transmittances: a power-law aerosol plus the paper's gas, with a fixed wobble."""
    centres = np.concatenate([[340.0], CENTRES])
    x = np.log(centres / 550.0)
    rho = rayleigh_optical_depth(centres, 1013.25) / rayleigh_optical_depth(550.0, 1013.25)
    tau = R_PAPER * rho + OZONE_PAPER * ozone_absorption_per_atm_cm(centres)
    tau = tau + tau_550 * np.exp(-1.3 * x)
    wobble = noise * np.sin(np.arange(centres.size) * 2.0)
    return [float(v) for v in np.exp(-(tau + wobble))]


def tool_data(rows: list[list[float]] | None = None):
    tool = load_tool()
    centres = _committed(10.0)["channel_centres_nm"]
    if rows is None:
        rows = [spectrum(_committed(10.0), aod) for aod in LOADS]
    return tool.gas_data(centres, rows)


# --- the registered definitions -------------------------------------------------------------------


def test_the_eight_estimates_are_numbered_with_the_aerosol_form_varying_slowest() -> None:
    tool = load_tool()
    assert [e.number for e in tool.GAS_ESTIMATES] == list(range(1, 9))
    assert {
        e.number: (e.curved, e.ozone_free, e.rayleigh_free) for e in tool.GAS_ESTIMATES
    } == ESTIMATES


def test_the_four_estimates_with_a_fixed_gas_parameter_keep_it_exactly() -> None:
    tool = load_tool()
    data = tool_data()
    fits = {e.number: tool.fit_gas_estimate(e, data) for e in tool.GAS_ESTIMATES}
    for number in (2, 4, 6, 8):  # Rayleigh at the paper's
        assert fits[number].a_r == R_PAPER
    for number in (3, 4, 7, 8):  # ozone at the U.S. Standard column
        assert fits[number].a_o == OZONE_PAPER
        assert fits[number].ozone_du == pytest.approx(345.0, rel=1e-12)
    for number in (1, 2, 5, 6):  # ozone free: moved off the paper's
        assert fits[number].a_o != OZONE_PAPER
    for number in (1, 3, 5, 7):  # Rayleigh free: moved off the paper's
        assert fits[number].a_r != R_PAPER


def test_the_paper_pressure_is_the_standard_atmosphere_at_340_m_unrounded() -> None:
    tool = load_tool()
    assert tool.PAPER_PRESSURE_HPA == P_PAPER
    assert_shown(P_PAPER, 973.0664, 4, "the paper's pressure")
    assert tool.PAPER_PRESSURE_HPA != 973.0664  # the criterion's figure is a printout
    assert tool_data().r_paper == R_PAPER
    assert float(rayleigh_optical_depth(550.0, 973.0664)) != R_PAPER


def test_the_gas_shapes_are_v1s_rayleigh_ratio_and_ozone_table_at_the_channel_centres() -> None:
    data = tool_data()
    assert data.centres.tolist() == CENTRES.tolist()
    np.testing.assert_allclose(data.x, X, rtol=1e-14)
    np.testing.assert_allclose(data.rho, RHO, rtol=1e-14)
    np.testing.assert_allclose(data.kappa, KAPPA, rtol=1e-14)


def test_the_gas_means_at_550_nm_interpolate_between_the_540_and_580_channels() -> None:
    data = tool_data()
    assert data.rho_bar == pinned(RHO_BAR)
    assert data.kappa_bar == pinned(KAPPA_BAR)
    assert_shown(data.rho_bar, 1.00968, 5, "rho bar")
    assert_shown(data.kappa_bar, 0.086141, 6, "kappa bar")
    by_hand = RHO[4] + (RHO[5] - RHO[4]) * (550.0 - 540.0) / (580.0 - 540.0)
    assert data.rho_bar == pytest.approx(by_hand, rel=1e-12)


def test_the_340_nm_channel_is_not_one_of_the_ten_fitted_channels() -> None:
    data = tool_data()
    assert data.centres.size == 10 and data.centres[0] == 380.0 and data.centres[-1] == 740.0
    assert data.y.shape == (2, 10)
    np.testing.assert_allclose(data.y, pilot_y(), rtol=1e-14)


def test_perturbing_the_340_nm_transmittance_changes_no_fit_at_all() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    for data in fixtures.values():  # the four fixtures keep holding the same spectra
        for aod in LOADS:
            spectrum(data, aod)[0] *= 0.5
    perturbed = tool.depth_pilot(fixtures)
    reference = pilot()
    for a, b in zip(reference.fits, perturbed.fits, strict=True):
        assert np.array_equal(a.theta, b.theta) and a.gas == b.gas and a.se == b.se
        assert a.condition == b.condition
    assert reference.y_bar_p == perturbed.y_bar_p


def test_perturbing_a_fitted_channel_does_change_the_fits() -> None:
    """The control for the test above: the rig can move when a fitted channel moves."""
    tool = load_tool()
    fixtures = fresh_fixtures()
    for data in fixtures.values():
        for aod in LOADS:
            spectrum(data, aod)[1] *= 0.99  # the 380 nm channel
    perturbed = tool.depth_pilot(fixtures)
    assert abs(perturbed.fits[0].gas - pilot().fits[0].gas) > 1e-4


@pytest.mark.parametrize("number", range(1, 9))
def test_the_model_residuals_and_the_analytic_jacobian_are_the_criterions(number: int) -> None:
    tool = load_tool()
    fit = tool.fit_gas_estimate(tool.GAS_ESTIMATES[number - 1], tool_data())
    y = pilot_y()
    residuals = model_residuals(number, fit.theta, y)
    assert float(residuals @ residuals) == pytest.approx(fit.ssr, rel=1e-9)
    np.testing.assert_allclose(
        fit.jac, finite_difference_jacobian(number, fit.theta, y), rtol=1e-6, atol=1e-8
    )
    assert fit.values == 20 and fit.parameters == fit.theta.size


@pytest.mark.parametrize("number", range(1, 9))
def test_the_solver_is_called_as_the_criterion_says_from_the_registered_start(
    monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    tool = load_tool()
    calls: list[tuple[tuple, dict]] = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return least_squares(*args, **kwargs)

    monkeypatch.setattr(tool, "least_squares", spy)
    tool.fit_gas_estimate(tool.GAS_ESTIMATES[number - 1], tool_data())
    assert len(calls) == 1
    ((args, kwargs),) = calls
    assert len(args) == 2  # the residual function and the start; everything else by keyword
    assert set(kwargs) == {"jac", "method", "xtol", "ftol", "gtol", "max_nfev"}
    assert callable(kwargs["jac"])
    assert kwargs["method"] == "trf"
    assert kwargs["xtol"] == kwargs["ftol"] == kwargs["gtol"] == 1e-15
    assert kwargs["max_nfev"] == 20000
    curved, ozone_free, rayleigh_free = ESTIMATES[number]
    expected: list[float] = []
    if rayleigh_free:
        expected.append(R_PAPER)
    if ozone_free:
        expected.append(0.345)
    g_paper = R_PAPER * RHO_BAR + 0.345 * KAPPA_BAR
    for aod in LOADS:
        y_bar = float(
            np.interp(
                550.0,
                _committed(10.0)["channel_centres_nm"],
                -np.log(spectrum(_committed(10.0), aod)),
            )
        )
        expected += [math.log(max(y_bar - g_paper, 0.001)), 1.3] + ([0.0] if curved else [])
    assert list(args[1]) == pytest.approx(expected, rel=1e-12)


def test_a_start_whose_aerosol_would_be_under_a_thousandth_is_floored_at_0_001() -> None:
    tool = load_tool()
    data = tool_data(
        [[math.exp(-0.05)] * 11, spectrum(_committed(10.0), 0.1)]  # y-bar 0.05 under the gas 0.1237
    )
    assert data.y_bar[0] == pytest.approx(0.05, rel=1e-12)
    start = tool.gas_start(tool.GAS_ESTIMATES[3], data)
    assert start[0] == math.log(0.001)
    assert start[2] == math.log(float(data.y_bar[1]) - (R_PAPER * RHO_BAR + 0.345 * KAPPA_BAR))


def test_the_pilot_fits_every_one_of_the_eight_estimates() -> None:
    result = pilot()
    assert [f.estimate.number for f in result.fits] == list(range(1, 9))
    assert all(f.status > 0 for f in result.fits)


# --- the gas, its error, the excess ---------------------------------------------------------------


@pytest.mark.parametrize("number", range(1, 9))
def test_every_pilot_quantity_is_the_criterions_to_half_a_unit(number: int) -> None:
    result = pilot()
    fit = result.fits[number - 1]
    values = (
        fit.gas,
        2 * fit.se,
        result.excess(fit),
        result.low(fit),
        result.high(fit),
        fit.rms,
        fit.a_r,
        fit.ozone_du,
    )
    for value, printed, digits, name in zip(
        values,
        SHOWN[number][:8],
        SHOWN_DIGITS,
        ("G", "2SE", "g", "g - 2SE", "g + 2SE", "residual rms", "a_R", "ozone"),
        strict=True,
    ):
        assert_shown(value, printed, digits, f"estimate {number}, {name}")
    assert_shown(
        fit.condition, SHOWN[number][8], SHOWN_CONDITION_DIGITS[number], f"estimate {number}, cond"
    )


@pytest.mark.parametrize("number", range(1, 9))
def test_every_pilot_quantity_is_pinned_by_value(number: int) -> None:
    result = pilot()
    fit, ref = result.fits[number - 1], PINNED[number]
    assert fit.gas == pinned(ref.gas)
    assert fit.excess_over(result.y_bar_p) == pinned(ref.excess)
    assert fit.rms == pinned(ref.rms)
    assert fit.a_r == pinned(ref.a_r)
    assert fit.ozone_du == pinned(ref.ozone_du)
    assert fit.condition == pinned(ref.condition)
    if ref.se == 0.0:
        assert fit.se == 0.0
    else:
        assert fit.se == pinned(ref.se)
    assert result.low(fit) == pytest.approx(ref.excess - 2 * ref.se, rel=1e-6, abs=1e-12)
    assert result.high(fit) == pytest.approx(ref.excess + 2 * ref.se, rel=1e-6, abs=1e-12)


def test_the_gas_is_the_fitted_rayleigh_and_ozone_at_the_interpolated_550_nm_shapes() -> None:
    for fit in pilot().fits:
        assert fit.gas == fit.a_r * RHO_BAR + fit.a_o * KAPPA_BAR


def test_the_excess_is_prague_at_its_visibility_minus_the_gas_minus_v1s_tenth() -> None:
    result = pilot()
    y_bar_p = float(
        np.interp(
            550.0, _committed(10.0)["channel_centres_nm"], -np.log(spectrum(_committed(10.0), 0.1))
        )
    )
    assert result.y_bar_p == y_bar_p
    for fit in result.fits:
        assert result.excess(fit) == y_bar_p - fit.gas - 0.1


def test_the_standard_error_has_no_variance_to_report_where_no_gas_parameter_is_fitted() -> None:
    result = pilot()
    for number in FITTING_NO_GAS:
        fit = result.fits[number - 1]
        assert fit.se == 0.0
        assert result.low(fit) == result.high(fit) == result.excess(fit)
    for number in FITTING_GAS:
        assert result.fits[number - 1].se > 0.0


@pytest.mark.parametrize("number", FITTING_GAS)
def test_the_standard_error_divides_by_values_minus_fitted_parameters(number: int) -> None:
    """s2 = SSR / (values - parameters): neither SSR / values nor the one-parameter form."""
    fit = pilot().fits[number - 1]
    curved, ozone_free, rayleigh_free = ESTIMATES[number]
    parameters = int(ozone_free) + int(rayleigh_free) + (3 if curved else 2) * 2
    assert fit.parameters == parameters and fit.values == 20
    s2 = fit.ssr / (20 - parameters)
    assert fit.s2 == s2
    q = np.zeros(parameters)
    index = 0
    if rayleigh_free:
        q[index], index = RHO_BAR, index + 1
    if ozone_free:
        q[index] = KAPPA_BAR
    covariance = s2 * np.linalg.inv(fit.jac.T @ fit.jac)
    assert fit.se == pytest.approx(math.sqrt(q @ covariance @ q), rel=1e-12)
    for wrong in (fit.ssr / 20, fit.ssr / (20 - 1)):
        assert fit.se != pytest.approx(
            math.sqrt(q @ (wrong * np.linalg.inv(fit.jac.T @ fit.jac)) @ q), rel=1e-6
        )


def test_the_standard_error_formula_on_a_matrix_worked_by_hand() -> None:
    tool = load_tool()
    jac = np.array([[2.0, 0.0], [0.0, 1.0]])  # J'J = diag(4, 1), (J'J)^-1 = diag(1/4, 1)
    assert tool.standard_error(jac, np.array([1.0, 0.0]), 4.0) == 1.0  # sqrt(4 * 1/4)
    assert tool.standard_error(jac, np.array([0.0, 1.0]), 4.0) == 2.0  # sqrt(4 * 1)
    assert tool.standard_error(jac, np.array([1.0, 1.0]), 4.0) == math.sqrt(5.0)


def test_the_residual_variance_of_a_fit_with_no_degrees_of_freedom_is_infinite() -> None:
    tool = load_tool()
    assert tool.residual_variance(2.0, values=10, parameters=8) == 1.0
    assert tool.residual_variance(2.0, values=8, parameters=8) == math.inf
    assert tool.residual_variance(2.0, values=6, parameters=8) == math.inf


def test_the_standard_error_is_infinite_where_the_normal_matrix_is_singular() -> None:
    """Two identical columns: numpy.linalg.inv raises, and the error is infinite, never a guess."""
    tool = load_tool()
    jac = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(jac.T @ jac)
    assert tool.standard_error(jac, np.array([1.0, 0.0]), 1.0) == math.inf


@pytest.mark.parametrize(
    ("q", "s2"),
    [
        (np.array([0.0, 0.0]), 1.0),  # a variance of exactly 0
        (np.array([1.0, 0.0]), 0.0),  # a perfect fit
        (np.array([1.0, 0.0]), -1.0),  # a negative variance
        (np.array([1.0, 0.0]), math.inf),
        (np.array([1.0, 0.0]), math.nan),
    ],
)
def test_the_standard_error_is_infinite_where_the_variance_is_not_positive_and_finite(
    q: np.ndarray, s2: float
) -> None:
    tool = load_tool()
    assert tool.standard_error(np.eye(2), q, s2) == math.inf


# --- refusals -------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [0, -1])
def test_a_fit_that_did_not_converge_is_refused_naming_the_estimate(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    tool = load_tool()

    def failing(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.status = status
        return result

    monkeypatch.setattr(tool, "least_squares", failing)
    with pytest.raises(tool.PragueError, match="estimate 3"):
        tool.fit_gas_estimate(tool.GAS_ESTIMATES[2], tool_data())


@pytest.mark.parametrize("status", [1, 2, 3, 4])
def test_a_converged_status_is_not_refused(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    tool = load_tool()

    def converged(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.status = status
        return result

    monkeypatch.setattr(tool, "least_squares", converged)
    fit = tool.fit_gas_estimate(tool.GAS_ESTIMATES[0], tool_data())
    assert fit.status == status


def test_a_refusal_stops_the_command_with_exit_1_and_the_estimate_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    calls = []

    def fifth_fails(*args, **kwargs):
        calls.append(1)
        result = least_squares(*args, **kwargs)
        if len(calls) == 5:
            result.status = 0
        return result

    monkeypatch.setattr(tool, "least_squares", fifth_fails)
    assert tool.main(["report", "--depth"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # the text is complete before anything is written
    assert "estimate 5" in captured.err and captured.err.startswith("error:")


def spectrum_changed(sun: float, aod: float, channel: int = 3):
    fixtures = fresh_fixtures()
    spectrum(fixtures[sun], aod)[channel] *= 1.001
    return fixtures


@pytest.mark.parametrize("sun", [10.0, 0.0, -1.0, 1.0])
@pytest.mark.parametrize("aod", LOADS)
def test_fixtures_that_do_not_hold_identical_spectra_are_refused(sun: float, aod: float) -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="identical") as raised:
        tool.depth_pilot(spectrum_changed(sun, aod))
    label = "0" if sun == 0 else f"{sun:+g}"
    assert f"{aod:g} spectrum" in str(raised.value)
    assert f"sun {label}" in str(raised.value)


def test_fixtures_with_different_channel_centres_are_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    fixtures[0.0]["channel_centres_nm"][4] += 1.0
    with pytest.raises(tool.PragueError, match="channel"):
        tool.depth_pilot(fixtures)


REGISTERED_CENTRES = [340.0 + 40.0 * i for i in range(11)]
"""The eleven channel centres the fixtures hold: 340 to 740 nm by 40, ascending."""


def committed_centres_and_rows() -> tuple[list[float], list[list[float]]]:
    return (
        list(_committed(10.0)["channel_centres_nm"]),
        [list(spectrum(_committed(10.0), aod)) for aod in LOADS],
    )


def fixtures_with_channels(centres: list[float], keep: list[int]) -> dict[float, dict]:
    """All four fixtures given these centres, and only the spectra columns `keep` (in this order),
    so the four still agree with one another and only the channels have changed."""
    fixtures = fresh_fixtures()
    for data in fixtures.values():
        data["channel_centres_nm"] = list(centres)
        for aod in LOADS:
            diagnostics = data["atmospheres"][f"{aod:g}"]["diagnostics"]
            row = diagnostics["zenith_transmittance"]
            diagnostics["zenith_transmittance"] = [row[i] for i in keep]
    return fixtures


def test_the_registered_centres_are_what_the_committed_fixtures_hold() -> None:
    assert _committed(10.0)["channel_centres_nm"] == REGISTERED_CENTRES
    assert tool_data().centres.tolist() == REGISTERED_CENTRES[1:]


@pytest.mark.parametrize("shift", [1.0, -1.0, 40.0, 2e-9])
def test_centres_that_are_not_the_registered_ones_are_refused_not_fitted(shift: float) -> None:
    """The fit is the channels centred at 380, 420, ..., 740 nm: a +1 nm shift is not them."""
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.gas_data([c + shift for c in centres], rows)
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.depth_pilot(fixtures_with_channels([c + shift for c in centres], list(range(11))))


@pytest.mark.parametrize("shift", [0.0, 5e-10, -5e-10])
def test_a_centre_within_a_billionth_of_a_nanometre_is_the_same_centre(shift: float) -> None:
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    data = tool.gas_data([c + shift for c in centres], rows)
    assert data.centres.size == 10
    np.testing.assert_allclose(data.centres, CENTRES, rtol=0, atol=1e-9)
    np.testing.assert_allclose(data.y, pilot_y(), rtol=1e-14)


@pytest.mark.parametrize("dropped", range(11))
def test_a_dropped_channel_is_refused_whichever_channel_it_is(dropped: int) -> None:
    """Including the 340 nm one: ten channels are not the fit's ten unless they are 380 to 740."""
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    keep = [i for i in range(11) if i != dropped]
    kept = [centres[i] for i in keep]
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.gas_data(kept, [[row[i] for i in keep] for row in rows])
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.depth_pilot(fixtures_with_channels(kept, keep))


def test_an_extra_channel_is_refused() -> None:
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.gas_data([*centres, 780.0], [[*row, row[-1]] for row in rows])


def test_descending_centres_are_refused_not_fitted_in_the_order_given() -> None:
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    backwards = list(range(10, -1, -1))
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.gas_data(centres[::-1], [row[::-1] for row in rows])
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.depth_pilot(fixtures_with_channels(centres[::-1], backwards))


def test_a_centre_that_is_not_a_number_is_refused() -> None:
    tool = load_tool()
    centres, rows = committed_centres_and_rows()
    centres[3] = math.nan
    with pytest.raises(tool.PragueError, match="registered eleven"):
        tool.gas_data(centres, rows)


V_003_KM = 123.8187372
V_P_KM = 49.9864309567


def with_visibility(aod: float, km: float) -> dict[float, dict]:
    """All four fixtures with the `aod` load's visibility set to `km` (they still agree)."""
    fixtures = fresh_fixtures()
    for data in fixtures.values():
        data["atmospheres"][f"{aod:g}"]["visibility_km"] = km
    return fixtures


def test_the_committed_loads_have_the_criterions_visibilities() -> None:
    for aod, km in zip(LOADS, (V_003_KM, V_P_KM), strict=True):
        assert _committed(10.0)["atmospheres"][f"{aod:g}"]["visibility_km"] == pytest.approx(
            km, abs=1e-6, rel=0
        )


@pytest.mark.parametrize("aod", LOADS)
@pytest.mark.parametrize("shift", [2e-6, -2e-6, 1e-3, -5.0])
def test_a_load_whose_visibility_is_not_the_registered_one_is_refused(
    aod: float, shift: float
) -> None:
    tool = load_tool()
    registered = dict(zip(LOADS, (V_003_KM, V_P_KM), strict=True))[aod]
    with pytest.raises(tool.PragueError, match=f"the {aod:g} load's visibility") as raised:
        tool.depth_pilot(with_visibility(aod, registered + shift))
    assert f"{registered + shift!r}" in str(raised.value)


def test_the_two_visibilities_swapped_between_the_loads_are_refused() -> None:
    tool = load_tool()
    fixtures = with_visibility(0.03, V_P_KM)
    for data in fixtures.values():
        data["atmospheres"]["0.1"]["visibility_km"] = V_003_KM
    with pytest.raises(tool.PragueError, match="load's visibility"):
        tool.depth_pilot(fixtures)


@pytest.mark.parametrize("aod", LOADS)
@pytest.mark.parametrize("shift", [0.0, 5e-7, -5e-7])
def test_a_visibility_within_a_millionth_of_a_kilometre_is_the_registered_one(
    aod: float, shift: float
) -> None:
    tool = load_tool()
    registered = dict(zip(LOADS, (V_003_KM, V_P_KM), strict=True))[aod]
    result = tool.depth_pilot(with_visibility(aod, registered + shift))
    assert [f.estimate.number for f in result.fits] == list(range(1, 9))


def test_fixtures_with_different_clamped_matches_for_the_clearest_load_are_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    fixtures[1.0]["atmospheres"]["0.03"]["diagnostics"]["matched_aerosol_depth"][
        "prague_aerosol_depth"
    ] = 0.036852
    with pytest.raises(tool.PragueError, match="0.03"):
        tool.depth_pilot(fixtures)


def test_a_clearest_load_that_is_not_clamped_has_no_depth_at_131_8_km_and_is_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    for data in fixtures.values():
        data["atmospheres"]["0.03"]["diagnostics"]["matched_aerosol_depth"][
            "clamped_at_largest_visibility"
        ] = False
    with pytest.raises(tool.PragueError, match="clamp"):
        tool.depth_pilot(fixtures)


def test_fixtures_that_disagree_on_a_visibility_are_refused() -> None:
    """A spectrum is Prague's at a visibility: the same numbers at another is another tool's."""
    tool = load_tool()
    fixtures = fresh_fixtures()
    fixtures[-1.0]["atmospheres"]["0.1"]["visibility_km"] += 0.001
    with pytest.raises(tool.PragueError, match="identical") as raised:
        tool.depth_pilot(fixtures)
    assert "0.1 spectrum" in str(raised.value) and "sun -1" in str(raised.value)


def test_a_fixture_without_a_visibility_is_malformed() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    del fixtures[0.0]["atmospheres"]["0.03"]["visibility_km"]
    with pytest.raises(tool.PragueError, match="malformed"):
        tool.depth_pilot(fixtures)


def test_the_fixtures_may_be_given_in_any_order() -> None:
    """Every sun is found by its value: the order of the mapping decides nothing."""
    tool = load_tool()
    reversed_order = tool.depth_pilot({sun: _committed(sun) for sun in reversed(FOUR_SUNS)})
    for ours, theirs in zip(reversed_order.suns, pilot().suns, strict=True):
        assert ours == theirs
    assert reversed_order.y_bar_clearest == pilot().y_bar_clearest


def test_the_two_spectra_belong_to_the_two_loads_visibilities() -> None:
    low, p = pilot().visibilities_km
    assert_shown(low, 123.8187372, 7, "the 0.03 load's visibility")
    assert_shown(p, 49.9864309567, 10, "V_p")
    assert "123.8187" in report_text() and "49.9864" in report_text()


def test_a_malformed_fixture_is_refused_before_anything_is_computed() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    del fixtures[-1.0]["atmospheres"]["0.1"]["diagnostics"]["zenith_transmittance"]
    with pytest.raises(tool.PragueError, match="malformed"):
        tool.depth_pilot(fixtures)


def test_a_missing_fixture_is_refused() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    del fixtures[1.0]
    with pytest.raises(tool.PragueError, match="fixtures"):
        tool.depth_pilot(fixtures)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, clamped=True),  # no amount's part at a clamped match
        lambda c: replace(c, p=replace(c.p, L_matched=c.p.L)),  # delta_m is 0
        lambda c: replace(c, p=replace(c.p, L_inv=c.p.L)),  # the reading noise is 0: no bar
        lambda c: replace(c, depth=c.depth_matched),  # no old excess to scale
    ],
    ids=["clamped", "no amount", "no bar", "no excess"],
)
def test_a_crossing_that_has_no_meaning_is_refused_naming_the_sun(
    monkeypatch: pytest.MonkeyPatch, change
) -> None:
    tool = load_tool()
    real = tool.offset_condition
    monkeypatch.setattr(tool, "offset_condition", lambda data, aod: change(real(data, aod)))
    with pytest.raises(tool.PragueError, match="sun -1"):
        tool.depth_pilot({sun: _committed(sun) for sun in FOUR_SUNS})


# --- the old estimate and the old gas -------------------------------------------------------------


def test_the_old_estimate_is_unchanged_by_the_shared_gas_helper() -> None:
    """`aerosol_depth_from_transmittance` is bit for bit what it was: total - Rayleigh - ozone."""
    tool = load_tool()
    centres = _committed(10.0)["channel_centres_nm"]
    for aod in LOADS:
        t = spectrum(_committed(10.0), aod)
        total = float(np.interp(550.0, centres, -np.log(np.asarray(t, dtype=float))))
        rayleigh = float(rayleigh_optical_depth(550.0, Atmosphere(0.0).pressure_hpa))
        ozone = float(ozone_absorption_per_atm_cm(550.0)) * 345.0 / 1e3
        assert tool.aerosol_depth_from_transmittance(t, centres) == total - rayleigh - ozone


def test_the_gas_helper_is_v1s_rayleigh_at_550_nm_and_the_345_du_ozone_depth() -> None:
    tool = load_tool()
    rayleigh, ozone = tool.v1_gas_depth_at_550(970.0)
    assert rayleigh == float(rayleigh_optical_depth(550.0, 970.0))
    assert ozone == float(ozone_absorption_per_atm_cm(550.0)) * 345.0 / 1e3
    assert tool.v1_gas_depth_at_550(Atmosphere(0.0).pressure_hpa) == (rayleigh, ozone)
    assert_shown(rayleigh + ozone, 0.12208, 5, "the old gas")


def test_the_old_gas_and_the_paper_gas_and_the_split_of_their_difference() -> None:
    result = pilot()
    assert result.gas_old == pinned(GAS_OLD)
    assert result.gas_paper == pinned(GAS_PAPER)
    assert result.gas_paper == result.fits[3].gas  # the paper's gas is estimate 4's
    assert result.gas_interpolation == pinned(GAS_INTERPOLATION)
    assert result.gas_pressure == pinned(GAS_PRESSURE)
    assert result.gas_paper - result.gas_old == pytest.approx(
        result.gas_interpolation + result.gas_pressure, rel=1e-12
    )
    assert_shown(result.gas_paper, 0.12367, 5, "G_paper")
    assert_shown(result.gas_old, 0.12208, 5, "G_old")
    assert_shown(result.gas_interpolation, 0.0013, 4, "the interpolation part")
    assert_shown(result.gas_pressure, 0.0003, 4, "the pressure part")


def test_the_interpolation_part_is_v1s_970_hpa_gas_interpolated_like_g_minus_the_old_gas() -> None:
    result = pilot()
    r_970 = float(rayleigh_optical_depth(550.0, 970.0))
    interpolated = r_970 * RHO_BAR + 0.345 * KAPPA_BAR
    assert result.gas_interpolation == pytest.approx(interpolated - result.gas_old, rel=1e-9)
    assert result.gas_pressure == pytest.approx(result.gas_paper - interpolated, rel=1e-9)


def test_the_old_excess_is_not_the_papers() -> None:
    result = pilot()
    assert_shown(result.suns[0].g_old, 0.0137, 4, "the old excess")
    assert_shown(result.suns[0].g_old, 0.01367, 5, "the old excess")
    assert_shown(result.excess(result.fits[3]), 0.0121, 4, "the paper's excess")


# --- the crossings --------------------------------------------------------------------------------


def sun_row(sun: float):
    (row,) = [s for s in pilot().suns if s.sun == sun]
    return row


def test_the_crossings_are_listed_at_the_four_suns_in_the_criterions_order_by_value() -> None:
    assert [s.sun for s in pilot().suns] == list(CRITERION_SUNS)


@pytest.mark.parametrize("sun", CRITERION_SUNS)
def test_each_sun_takes_its_own_fixtures_numbers(sun: float) -> None:
    """The crossing at a sun is that sun's `offset_condition`, found by the sun's value."""
    tool = load_tool()
    row = sun_row(sun)
    c = tool.offset_condition(_committed(sun), 0.1)
    assert row.L == c.p.L and row.m == c.p.m and row.delta_m == c.delta_m
    assert row.g_old == c.depth - c.depth_matched
    assert row.bar == 2 * c.p.m


@pytest.mark.parametrize("sun", CRITERION_SUNS)
def test_the_crossing_quantities_are_pinned_by_value(sun: float) -> None:
    row = sun_row(sun)
    g_old, delta_m, bar, level, g_star, carrying, over_bar, at_paper = CROSSING[sun]
    assert row.g_old == pinned(g_old)
    assert row.delta_m == pinned(delta_m)
    assert row.bar == pinned(bar)
    assert row.L == pinned(level)  # noqa: SIM300
    assert row.g_star == pinned(g_star)
    assert row.carrying == pinned(carrying)
    assert row.delta_m_over_bar == pinned(over_bar)
    assert row.at_paper == pinned(at_paper)


@pytest.mark.parametrize("sun", CRITERION_SUNS)
def test_the_crossing_quantities_are_their_definitions(sun: float) -> None:
    result = pilot()
    row = sun_row(sun)
    g_4 = result.excess(result.fits[3])
    assert row.g_star == row.g_old * row.bar / row.delta_m
    assert row.carrying == row.g_old * (row.L / 2) / row.delta_m
    assert row.delta_m_over_bar == row.delta_m / row.bar
    assert row.at_paper == row.delta_m * g_4 / (row.g_old * row.bar)


def test_the_crossing_numbers_are_the_criterions_to_half_a_unit() -> None:
    delta = {-1.0: 0.0401, 0.0: 0.0521, 1.0: 0.0639, 10.0: 0.0375}
    star = {-1.0: 0.0094, 0.0: 0.0095, 1.0: 0.0095, 10.0: 0.0095}
    carrying = {-1.0: 0.0541, 0.0: 0.0351, 1.0: 0.0263, 10.0: 0.0242}
    for sun in CRITERION_SUNS:
        row = sun_row(sun)
        assert_shown(row.delta_m, delta[sun], 4, f"delta_M at sun {sun:+g}")
        assert_shown(row.g_star, star[sun], 4, f"g* at sun {sun:+g}")
        assert_shown(row.carrying, carrying[sun], 4, f"the carrying excess at sun {sun:+g}")
        assert_shown(row.g_old, 0.01367, 5, f"g_old at sun {sun:+g}")
    over_bar = [sun_row(s).delta_m_over_bar for s in CRITERION_SUNS]
    at_paper = [sun_row(s).at_paper for s in CRITERION_SUNS]
    assert_shown(min(over_bar), 1.435, 3, "delta_M / b, lowest")
    assert_shown(max(over_bar), 1.447, 3, "delta_M / b, highest")
    assert_shown(min(at_paper), 1.268, 3, "delta_M g_4 / (g_old b), lowest")
    assert_shown(max(at_paper), 1.279, 3, "delta_M g_4 / (g_old b), highest")


def test_the_clearest_node_from_the_recorded_clamped_match() -> None:
    result = pilot()
    stored = _committed(10.0)["atmospheres"]["0.03"]["diagnostics"]["matched_aerosol_depth"]
    assert stored["clamped_at_largest_visibility"] is True
    assert result.y_bar_clearest == stored["prague_aerosol_depth"] + result.gas_old
    assert result.y_bar_clearest == pinned(Y_BAR_CLEAREST)
    assert result.g_max == result.y_bar_p - result.y_bar_clearest
    assert result.threshold_003 == result.y_bar_clearest - 0.03
    assert_shown(result.y_bar_clearest, 0.15893, 5, "y-bar(131.8)")
    assert_shown(result.g_max, 0.0768, 4, "g_max")
    assert_shown(result.threshold_003, 0.1289, 4, "the 0.03 load's G threshold")
    assert all(f.gas < result.threshold_003 for f in result.fits)  # "no estimate comes near it"


# --- problem 1's invariance checks ----------------------------------------------------------------


def test_the_invariance_covers_the_six_estimates_that_fit_a_gas_parameter() -> None:
    inv = pilot().invariance
    assert tuple(inv.estimates) == FITTING_GAS
    assert (
        set(inv.scale_ratios) == set(inv.alpha_ratios) == set(inv.triple_ratios) == set(FITTING_GAS)
    )


def test_scaling_every_nodes_amount_by_a_power_of_two_leaves_the_standard_error_alone() -> None:
    inv = pilot().invariance
    for number in FITTING_GAS:
        assert set(inv.scale_ratios[number]) == {0.5, 2.0}
        for factor, ratio in inv.scale_ratios[number].items():
            assert abs(ratio - 1.0) < 1e-9, f"estimate {number}, factor {factor}"
    assert inv.scale_largest < 1e-9
    assert inv.scale_largest == max(
        abs(r - 1.0) for ratios in inv.scale_ratios.values() for r in ratios.values()
    )


def test_the_control_that_must_move_is_the_first_visibilitys_alpha_raised_by_0_5() -> None:
    inv = pilot().invariance
    assert_shown(inv.alpha_smallest, 0.0180, 4, "the control")
    assert inv.alpha_smallest == pinned(0.018021599218259077)
    assert inv.alpha_smallest == min(abs(r - 1.0) for r in inv.alpha_ratios.values())
    assert inv.alpha_smallest > 1e-3  # it moves by orders of magnitude more than the checks
    expected = {
        1: 0.10120115585365075,
        2: 0.018021599218259077,
        3: 0.1684168384275433,
        5: -0.061701470516756673,
        6: -0.01863337740693438,
        7: -0.03404471046429858,
    }
    for number, change in expected.items():
        assert inv.alpha_ratios[number] - 1.0 == pytest.approx(change, rel=1e-6)


def test_three_copies_of_each_spectrum_divide_every_standard_error_by_root_3() -> None:
    inv = pilot().invariance
    for number in FITTING_GAS:
        ratio = inv.triple_ratios[number]
        assert abs(ratio - 1 / math.sqrt(3)) < 1e-9, f"estimate {number}"
        assert_shown(ratio, 0.5774, 4, f"estimate {number}'s ratio to its pilot SE")
    assert inv.triple_largest < 1e-9
    assert inv.triple_largest == max(abs(r - 1 / math.sqrt(3)) for r in inv.triple_ratios.values())


def test_the_six_node_jacobian_gives_each_copy_its_own_aerosol_parameters() -> None:
    """Six visibilities: the gas columns are shared, the aerosol columns are block diagonal."""
    tool = load_tool()
    data = tool_data()
    for estimate in tool.GAS_ESTIMATES:
        fit = tool.fit_gas_estimate(estimate, data)
        width = 3 if estimate.curved else 2
        gas = estimate.gas_parameters
        blocks = fit.theta[gas:].reshape(2, width)
        six = np.concatenate([fit.theta[:gas], np.repeat(blocks, 3, axis=0).ravel()])
        jac = tool.gas_jacobian(estimate, data, six, 6)
        assert jac.shape == (60, gas + 6 * width)
        for i in range(6):
            rows = slice(10 * i, 10 * i + 10)
            others = np.ones(jac.shape[1], dtype=bool)
            others[gas + width * i : gas + width * (i + 1)] = False
            others[:gas] = False
            assert not jac[rows][:, others].any(), "a copy's aerosol columns are its own"
            assert jac[rows][:, gas + width * i].any()
        if estimate.rayleigh_free:
            assert np.array_equal(jac[:, 0], np.tile(data.rho, 6))
        # copies of one spectrum have equal blocks: copy 1 and copy 2 of the first spectrum
        assert np.array_equal(
            jac[0:10, gas : gas + width], jac[10:20, gas + width : gas + 2 * width]
        )


def test_the_invariance_holds_the_pilots_residual_variance() -> None:
    """If s2 were refitted for six nodes the ratio would not be 1 / sqrt(3): it is held."""
    tool = load_tool()
    data = tool_data()
    estimate = tool.GAS_ESTIMATES[0]
    fit = tool.fit_gas_estimate(estimate, data)
    q = np.array([data.rho_bar, data.kappa_bar] + [0.0] * 4)
    width = 2
    six = np.concatenate(
        [fit.theta[:2], np.repeat(fit.theta[2:].reshape(2, width), 3, axis=0).ravel()]
    )
    jac = tool.gas_jacobian(estimate, data, six, 6)
    q6 = np.concatenate([q[:2], np.zeros(12)])
    held = tool.standard_error(jac, q6, fit.s2)
    refitted = tool.standard_error(jac, q6, tool.residual_variance(3 * fit.ssr, 60, 14))
    assert held / fit.se == pytest.approx(1 / math.sqrt(3), rel=1e-12)
    assert abs(refitted / fit.se - 1 / math.sqrt(3)) > 1e-3


def test_the_invariance_numbers_are_the_same_whatever_the_order_of_the_copies() -> None:
    tool = load_tool()
    data = tool_data()
    estimate = tool.GAS_ESTIMATES[4]
    fit = tool.fit_gas_estimate(estimate, data)
    gas, width = estimate.gas_parameters, 3
    blocks = fit.theta[gas:].reshape(2, width)
    q = np.concatenate([[data.rho_bar, data.kappa_bar], np.zeros(18)])
    ratios = []
    for order in ([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]):
        theta = np.concatenate([fit.theta[:gas], blocks[order].ravel()])
        jac = tool.gas_jacobian(estimate, data, theta, 6)
        ratios.append(tool.standard_error(jac, q, fit.s2) / fit.se)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)


class StandardErrorCall(NamedTuple):
    """The three arguments one call of `standard_error` received (copies, taken at the call)."""

    jac: np.ndarray
    q: np.ndarray
    s2: float


def spy_on_standard_error(
    monkeypatch: pytest.MonkeyPatch, tool: ModuleType
) -> list[StandardErrorCall]:
    """Record what the tool hands `standard_error`, and still return its answer."""
    calls: list[StandardErrorCall] = []
    real = tool.standard_error

    def spy(jac: np.ndarray, q: np.ndarray, s2: float) -> float:
        calls.append(StandardErrorCall(np.array(jac, copy=True), np.array(q, copy=True), s2))
        return real(jac, q, s2)

    monkeypatch.setattr(tool, "standard_error", spy)
    return calls


def expected_q(number: int, parameters: int) -> np.ndarray:
    """q from the criterion: rho-bar for a_R, kappa-bar for a_O, 0 for each aerosol parameter."""
    _, ozone_free, rayleigh_free = ESTIMATES[number]
    q = np.zeros(parameters)
    index = 0
    if rayleigh_free:
        q[index], index = RHO_BAR, index + 1
    if ozone_free:
        q[index] = KAPPA_BAR
    return q


@pytest.mark.parametrize("factor", [0.5, 2.0])
@pytest.mark.parametrize("number", FITTING_GAS)
def test_the_amount_scaling_check_scales_the_pilots_aerosol_columns_and_only_those(
    monkeypatch: pytest.MonkeyPatch, number: int, factor: float
) -> None:
    """Scaling by a power of two is exact, so a check that scaled nothing would still pass the
    1e-9 bound on the ratio: this reads the Jacobian the check built, not the ratio it returned."""
    tool = load_tool()
    result = pilot()
    fit = result.fits[number - 1]
    gas = fit.estimate.gas_parameters
    assert not np.array_equal(fit.jac[:, gas:], factor * fit.jac[:, gas:])  # scaling can be seen
    calls = spy_on_standard_error(monkeypatch, tool)
    tool.invariance_checks([fit], result.data)
    scaled = [
        c
        for c in calls
        if c.jac.shape == fit.jac.shape
        and np.array_equal(c.jac[:, :gas], fit.jac[:, :gas])
        and np.array_equal(c.jac[:, gas:], factor * fit.jac[:, gas:])
    ]
    assert len(scaled) == 1, (
        f"estimate {number}: no standard error was computed from the pilot's Jacobian with its "
        f"aerosol columns x {factor:g} and its gas columns unchanged "
        f"({len(calls)} calls, shapes {[c.jac.shape for c in calls]})"
    )
    (call,) = scaled
    assert call.s2 == fit.s2, "the pilot's s2 is held"
    assert np.array_equal(call.q, expected_q(number, fit.theta.size))


@pytest.mark.parametrize("number", FITTING_GAS)
def test_the_three_copy_check_hands_the_standard_error_a_block_diagonal_sixty_row_jacobian(
    monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    """Each copy has its own aerosol parameters (the criterion's construction): six nodes' aerosol
    columns, each copy's rows nonzero in its own columns only, the pilot's gas columns tiled."""
    tool = load_tool()
    result = pilot()
    fit = result.fits[number - 1]
    gas, width = fit.estimate.gas_parameters, fit.estimate.aerosol_width
    calls = spy_on_standard_error(monkeypatch, tool)
    tool.invariance_checks([fit], result.data)
    sixty = [c for c in calls if c.jac.shape[0] == 60]
    assert len(sixty) == 1, f"estimate {number}: {[c.jac.shape for c in calls]}"
    (call,) = sixty
    jac = call.jac
    assert jac.shape == (60, gas + 6 * width), (
        f"estimate {number}: the six-node J is {jac.shape}: each of the six copies needs its own "
        f"{width} aerosol columns after the {gas} gas columns"
    )
    assert np.array_equal(jac[:, :gas], np.tile(fit.jac[:, :gas], (3, 1))), "the gas columns"
    pilot_blocks = [
        fit.jac[10 * v : 10 * v + 10, gas + width * v : gas + width * (v + 1)] for v in (0, 1)
    ]
    copies_of_each_spectrum = [0, 0]
    for node in range(6):
        rows = slice(10 * node, 10 * node + 10)
        own = slice(gas + width * node, gas + width * (node + 1))
        elsewhere = np.ones(jac.shape[1], dtype=bool)
        elsewhere[:gas] = False
        elsewhere[own] = False
        assert not jac[rows][:, elsewhere].any(), f"copy {node} has aerosol columns of its own"
        spectra = [
            v for v in (0, 1) if np.allclose(jac[rows, own], pilot_blocks[v], rtol=1e-12, atol=0)
        ]
        assert len(spectra) == 1, f"copy {node} is a copy of one pilot spectrum's aerosol"
        copies_of_each_spectrum[spectra[0]] += 1
    assert copies_of_each_spectrum == [3, 3], "three copies of each pilot spectrum"
    assert call.s2 == fit.s2, "the pilot's s2 is held"
    assert np.array_equal(call.q, expected_q(number, jac.shape[1]))


# --- the report -----------------------------------------------------------------------------------


def table(text: str, header: str) -> list[list[str]]:
    """The cells of the rows under the table whose header row starts with `header`."""
    lines = text.splitlines()
    (start,) = [i for i, line in enumerate(lines) if line.startswith(header)]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def number(cell: str) -> float:
    return float(cell.replace("+", ""))


def headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


def test_the_report_has_the_issue_heading_and_a_pilot_heading_above_the_numbers() -> None:
    text = report_text()
    found = headings(text)
    assert [h for h in found if "#55" in h and "no test" in h]
    pilot_headings = [
        i for i, line in enumerate(text.splitlines()) if line.startswith("#") and "pilot" in line
    ]
    assert pilot_headings
    first_table = next(
        i for i, line in enumerate(text.splitlines()) if line.startswith("| estimate | G |")
    )
    assert pilot_headings[0] < first_table
    first_number = next(
        i for i, line in enumerate(text.splitlines()) if line.startswith("|") and "0.1149" in line
    )
    assert pilot_headings[0] < first_number


def test_the_report_prints_the_eight_row_pilot_table_with_the_criterions_columns() -> None:
    rows = table(report_text(), "| estimate | G |")
    header = [
        c.strip()
        for c in next(
            line for line in report_text().splitlines() if line.startswith("| estimate | G |")
        )
        .strip()
        .strip("|")
        .split("|")
    ]
    assert header == [
        "estimate",
        "G",
        "2SE",
        "g",
        "g - 2SE",
        "g + 2SE",
        "residual rms",
        "a_R",
        "ozone (DU)",
        "condition number",
    ]
    assert [r[0] for r in rows] == [str(k) for k in range(1, 9)]


def test_every_printed_pilot_number_reproduces_the_criterions_to_half_a_unit() -> None:
    rows = table(report_text(), "| estimate | G |")
    for row in rows:
        k = int(row[0])
        for cell, printed, digits, name in zip(
            row[1:9],
            SHOWN[k][:8],
            SHOWN_DIGITS,
            ("G", "2SE", "g", "g - 2SE", "g + 2SE", "residual rms", "a_R", "ozone"),
            strict=True,
        ):
            assert_shown(number(cell), printed, digits, f"printed estimate {k}, {name}")
        assert_shown(
            number(row[9]), SHOWN[k][8], SHOWN_CONDITION_DIGITS[k], f"printed estimate {k}, cond"
        )


def test_a_standard_error_of_zero_prints_as_0_and_an_infinite_one_as_inf() -> None:
    tool = load_tool()
    rows = {r[0]: r for r in table(report_text(), "| estimate | G |")}
    assert rows["4"][2] == "0" and rows["8"][2] == "0"
    assert tool.format_two_se(0.0) == "0"
    assert tool.format_two_se(math.inf) == "inf"
    assert tool.format_two_se(0.0016) == "0.0032"


def test_a_condition_number_that_is_not_finite_prints_as_inf_or_nan_and_does_not_raise() -> None:
    tool = load_tool()
    assert tool.format_condition(math.inf) == "inf"
    assert tool.format_condition(math.nan) == "nan"
    assert tool.format_condition(float(np.float64("inf"))) == "inf"
    for value, printed in ((14.1475, "14.1"), (882.2, "882"), (14_699.4, "1.470e4")):
        assert tool.format_condition(value) == printed  # the finite ones are as they were


def test_the_report_names_each_estimates_assumptions() -> None:
    rows = table(report_text(), "| estimate | aerosol form |")
    assert [r[0] for r in rows] == [str(k) for k in range(1, 9)]
    for row in rows:
        curved, ozone_free, rayleigh_free = ESTIMATES[int(row[0])]
        assert row[1] == ("curved" if curved else "power law")
        assert row[2] == ("free" if ozone_free else "paper")
        assert row[3] == ("free" if rayleigh_free else "paper")


GAS_HEADER = "| gas depth at 550 nm | value |"
NODE_HEADER = "| the clearest node | value |"


def labelled(text: str, header: str) -> dict[str, float]:
    return {row[0]: number(row[1]) for row in table(text, header)}


def test_the_report_prints_the_gas_bullets() -> None:
    values = labelled(report_text(), GAS_HEADER)
    assert_shown(values["G_paper"], 0.12367, 5, "printed G_paper")
    assert_shown(values["G_old"], 0.12208, 5, "printed G_old")
    assert_shown(values["of which the interpolation"], 0.0013, 4, "printed interpolation part")
    assert_shown(values["of which the pressure"], 0.0003, 4, "printed pressure part")
    assert_shown(values["G_paper - G_old"], 0.0016, 4, "printed difference")


def test_the_printed_sentence_gives_the_papers_excess_and_the_old_one() -> None:
    """'The paper's gas puts the excess at g_4 = ..., not the old g_old = ...': 0.0121, 0.0137."""
    found = re.search(
        r"The paper's gas puts the excess at g_4 = (\S+), not the old g_old = (\S+) \(Prague's",
        report_text(),
    )
    assert found, "the sentence about g_4 and g_old is not in the report"
    g_4, g_old = float(found.group(1)), float(found.group(2))
    result = pilot()
    assert_shown(g_4, 0.0121, 4, "the criterion's g_4 in the printed sentence")
    assert_shown(g_old, 0.0137, 4, "the criterion's g_old in the printed sentence")
    assert_shown(result.excess(result.fits[3]), g_4, 4, "estimate 4's excess, unrounded")
    assert_shown(result.suns[0].g_old, g_old, 4, "the old excess, unrounded")


def test_the_prose_labels_the_two_visibilities_with_the_criterions_values() -> None:
    """V is the 0.03 load's visibility (123.8187 km) and V_p the 0.1 load's (49.9864 km)."""
    found = re.search(
        r"at V = (\S+) km \(the 0\.03 load's visibility\) and V_p = (\S+) km \(the 0\.1 load's\)",
        report_text(),
    )
    assert found, "the sentence naming V and V_p is not in the report"
    v, v_p = float(found.group(1)), float(found.group(2))
    assert_shown(v, 123.8187, 4, "the criterion's V in the prose")
    assert_shown(v_p, 49.9864, 4, "the criterion's V_p in the prose")
    low, p = pilot().visibilities_km
    assert_shown(low, v, 4, "the 0.03 load's visibility, unrounded")
    assert_shown(p, v_p, 4, "V_p, unrounded")


def test_the_prose_names_the_papers_pressure_not_v1s() -> None:
    found = re.search(r"R_paper = (\S+) at (\S+) hPa \(the U\.S\. Standard pressure", report_text())
    assert found, "the sentence naming the paper's Rayleigh and pressure is not in the report"
    r_paper, hpa = float(found.group(1)), float(found.group(2))
    assert_shown(hpa, 973.0664, 4, "the criterion's pressure in the prose")
    assert_shown(P_PAPER, hpa, 4, "the paper's pressure, unrounded")
    assert_shown(R_PAPER, r_paper, 5, "the paper's Rayleigh depth, unrounded")
    assert hpa != Atmosphere(0.0).pressure_hpa  # 970 is v1's, the old estimate's


def test_the_largest_of_the_eight_g_is_the_maximum() -> None:
    found = re.search(r"The largest of the eight G is (\S+)\.", report_text())
    assert found, "the sentence about the largest G is not in the report"
    largest = float(found.group(1))
    assert_shown(max(row[0] for row in SHOWN.values()), largest, 4, "the criterion's largest G")
    assert_shown(max(f.gas for f in pilot().fits), largest, 4, "the largest G, unrounded")
    assert not shown(min(f.gas for f in pilot().fits), largest, 4)


def test_the_report_prints_the_crossings_at_each_sun() -> None:
    text = report_text()
    rows = table(text, "| sun | L | m | bar b = 2m |")
    assert [r[0] for r in rows] == ["-1", "0", "+1", "+10"]
    header = next(line for line in text.splitlines() if line.startswith("| sun | L | m |"))
    for name in (
        "Delta_M",
        "g_old",
        "g*",
        "carrying excess",
        "Delta_M / b",
        "Delta_M g_4 / (g_old b)",
    ):
        assert name in header
    expected = {
        "-1": (0.0401, 0.0094, 0.0541),
        "0": (0.0521, 0.0095, 0.0351),
        "+1": (0.0639, 0.0095, 0.0263),
        "+10": (0.0375, 0.0095, 0.0242),
    }
    cells = {r[0]: r for r in rows}
    columns = [c.strip() for c in header.strip().strip("|").split("|")]
    at = {name: columns.index(name) for name in columns}
    for sun, (delta_m, g_star, carrying) in expected.items():
        assert_shown(number(cells[sun][at["Delta_M"]]), delta_m, 4, f"printed Delta_M, sun {sun}")
        assert_shown(number(cells[sun][at["g*"]]), g_star, 4, f"printed g*, sun {sun}")
        assert_shown(
            number(cells[sun][at["carrying excess"]]), carrying, 4, f"printed carry, {sun}"
        )
        assert_shown(number(cells[sun][at["g_old"]]), 0.01367, 5, f"printed g_old, sun {sun}")
    over_bar = [number(r[at["Delta_M / b"]]) for r in rows]
    at_paper = [number(r[at["Delta_M g_4 / (g_old b)"]]) for r in rows]
    assert_shown(min(over_bar), 1.435, 3, "printed Delta_M / b, lowest")
    assert_shown(max(over_bar), 1.447, 3, "printed Delta_M / b, highest")
    assert_shown(min(at_paper), 1.268, 3, "printed ratio at the paper's excess, lowest")
    assert_shown(max(at_paper), 1.279, 3, "printed ratio at the paper's excess, highest")


def test_the_report_prints_the_clearest_node() -> None:
    text = report_text()
    rows = labelled(text, NODE_HEADER)
    assert_shown(rows["ybar(131.8)"], 0.15893, 5, "printed y-bar(131.8)")
    assert_shown(rows["g_max"], 0.0768, 4, "printed g_max")
    assert_shown(rows["0.03 load's G threshold"], 0.1289, 4, "printed threshold")


def test_the_report_prints_the_invariance_checks_and_the_control() -> None:
    text = report_text()
    rows = table(text, "| estimate | aerosol x 0.5")
    assert [r[0] for r in rows] == [str(k) for k in FITTING_GAS]
    for row in rows:
        assert float(row[1]) < 1e-9 and float(row[2]) < 1e-9
        assert row[4] == "0.5774"  # each estimate's ratio to its pilot SE, to 4 decimals
    largest = re.search(
        r"Largest \|ratio - 1\| over the six estimates and both factors: (\S+)", text
    )
    control = re.search(
        r"Smallest \|ratio - 1\| of the control over the six estimates: (\S+)", text
    )
    triple = re.search(r"Largest \|ratio - 1/sqrt\(3\)\| over the six estimates: (\S+)", text)
    assert largest and control and triple
    assert float(largest.group(1)) < 1e-9
    assert_shown(float(control.group(1)), 0.0180, 4, "printed control")
    assert float(triple.group(1)) < 1e-9
    assert float(largest.group(1)) == pytest.approx(
        pilot().invariance.scale_largest, rel=0.06, abs=1e-16
    )


def test_the_report_prints_no_verdict() -> None:
    """The route is registered as no test: no verdict word of #49's rules, and no 'spans'."""
    text = report_text()
    for word in ("carries", "contributes", "not this", "spans", "cannot tell"):
        assert word not in text.lower(), word
    assert "no test" in text.lower()


def test_the_report_is_complete_and_deterministic() -> None:
    tool = load_tool()
    again = tool.report_depth({sun: _committed(sun) for sun in FOUR_SUNS})
    assert again == report_text()
    assert report_text().endswith("\n") and not report_text().endswith("\n\n")


def test_the_report_function_does_not_edit_the_fixtures_it_is_given() -> None:
    tool = load_tool()
    fixtures = fresh_fixtures()
    tool.report_depth(fixtures)
    assert fixtures == {sun: _committed(sun) for sun in FOUR_SUNS}


# --- the command ----------------------------------------------------------------------------------


def test_the_depth_flag_belongs_to_report_alone_and_not_with_offset() -> None:
    tool = load_tool()
    assert tool.build_parser().parse_args(["report", "--depth"]).depth is True
    assert tool.build_parser().parse_args(["report"]).depth is False
    assert tool.build_parser().parse_args(["report", "--offset"]).depth is False
    with pytest.raises(SystemExit):
        tool.main(["fixture", "--depth"])
    with pytest.raises(SystemExit):
        tool.main(["report", "--depth", "--offset"])


def test_the_command_line_writes_exactly_the_report_functions_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pinned to the function that builds the report, never to stored text or a hash of it."""
    tool = load_tool()
    built = tool.report_depth({sun: _committed(sun) for sun in FOUR_SUNS})
    assert tool.main(["report", "--depth"]) == 0
    written = capsys.readouterr().out
    assert written == built
    assert written.endswith("\n") and not written.endswith("\n\n")


def test_the_command_line_reads_the_four_committed_fixtures_and_v1_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    read: list[Path] = []
    real = Path.read_text

    def recording(self: Path, *args, **kwargs) -> str:
        read.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    assert tool.main(["report", "--depth"]) == 0
    capsys.readouterr()
    fixtures = {p for p in read if FIXTURE_DIR in p.parents}
    assert fixtures == {FIXTURE_DIR / name for name in FIXTURE_FILES.values()}
    data = Path(render_package.__file__).parent / "data"
    assert all(p in fixtures or data in p.parents for p in read), read


def test_the_command_line_runs_no_prague_evaluation_and_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()

    def refuse(*args, **kwargs):
        raise AssertionError("the depth report must not evaluate Prague or write a file")

    for name in ("evaluate", "build_fixture", "matched_visibility_km", "cmd_fetch", "cmd_build"):
        monkeypatch.setattr(tool, name, refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(Path, "write_text", refuse)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    monkeypatch.chdir(tmp_path)
    before = {p.name: p.read_bytes() for p in FIXTURE_DIR.iterdir()}
    assert tool.main(["report", "--depth"]) == 0
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []
    assert {p.name: p.read_bytes() for p in FIXTURE_DIR.iterdir()} == before


def test_the_command_line_reports_a_disagreement_between_the_fixtures_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    fixtures = spectrum_changed(0.0, 0.1)
    monkeypatch.setattr(tool, "read_offset_fixtures", lambda: fixtures)
    assert tool.main(["report", "--depth"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:") and "0.1" in captured.err


def test_the_older_reports_are_still_reached_by_their_own_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["report"]) == 0
    assert capsys.readouterr().out.startswith("### Sun +10, `aod_550` 0.03")
    assert tool.main(["report", "--sun", "0"]) == 0
    assert capsys.readouterr().out.startswith("### Rule A at sun 0")


def test_the_module_docstring_lists_the_new_command() -> None:
    assert "report --depth" in (load_tool().__doc__ or "")


def test_the_criterion_file_is_the_one_the_tool_reproduces() -> None:
    """The contract is a committed file; its pilot table must hold the numbers this file pins."""
    text = CRITERION.read_text()
    assert (
        "| 7 | 0.0990 | 0.0601 | 0.0367 | -0.0234 | 0.0969 | 0.00092 | 0.0686 | 345.0 | 1.033e4 |"
        in text
    )
    assert "No six-node fit is run under it" in text


def test_the_provenance_note_lists_the_depth_report_beside_the_other_reports() -> None:
    text = (FIXTURE_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "uv run python tools/prague_reference.py report --depth" in text
    assert "`report --depth` prints #55's evidence (no test) from all four fixtures" in text
