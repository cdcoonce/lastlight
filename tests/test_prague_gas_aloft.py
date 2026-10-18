"""Prague's aerosol amount from Prague's own ozone aloft and the paper's Rayleigh (#57).

The rule is registered in `docs/renderer/prague-gas-aloft-criterion.md` (landed in PR #60), with a
power check on synthetic truths, `tools/issue57_power.py`. This file grades the tool that carries
the rule out, `tools/prague_reference.py`, on synthetic spectra only (no Prague binary, no network):

* the agreement test: the tool's envelope (g_lo and g_hi) and its verdict line equal the power
  check's `estimate(...).envelope()` and `verdict(...)` to 1e-6 on at least 20 synthetic pairs;
* each step's pieces against hand-computed values and an independent re-derivation written here
  from the criterion's text (the power law, the passes, the six-node fit, the gradients);
* the verdict at each boundary, and "mixed";
* the refusals, each exiting 1 with the reason on stderr;
* the fixture builder (`fixture --aloft`) against stand-ins for the model, and the report command
  (`report --aloft`) against synthetic aloft spectra.

The tests that pin the committed `aloft.json` and the printed quantities by value are at the end of
the file.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import math
import re
import subprocess
import sys
import zlib
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import least_squares

from lastlight.render.atmosphere import ozone_absorption_per_atm_cm, rayleigh_optical_depth

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prague_reference.py"
POWER = ROOT / "tools" / "issue57_power.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "prague"
CRITERION = ROOT / "docs" / "renderer" / "prague-gas-aloft-criterion.md"


def _load(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def load_tool() -> ModuleType:
    """The tool as a module, shared with the other Prague test files when it is already loaded."""
    return _load("prague_reference", TOOL)


def load_power() -> ModuleType:
    """The registered power check, which this file reads and never edits."""
    return _load("issue57_power", POWER)


def pinned(value: float) -> object:
    """A relative pin: pytest's default absolute tolerance would void it on small values."""
    return pytest.approx(value, rel=1e-6, abs=0)


# --- an independent re-derivation of the criterion's definitions ---------------------------------
#
# Written here from the criterion's text, in plain numpy and scipy, so the tool's pieces are graded
# against something that does not share their code.

LAM = np.arange(380.0, 741.0, 40.0)
X = np.log(LAM / 550.0)
P_034 = 1013.25 * (1.0 - 0.0065 * 340.0 / 288.15) ** 5.25588
P_15 = 121.11
R_PAPER = float(rayleigh_optical_depth(550.0, P_034))
RHO = np.asarray(rayleigh_optical_depth(LAM, P_034)) / R_PAPER
R15 = np.asarray(rayleigh_optical_depth(LAM, P_15))
K460, K500, K550 = (float(ozone_absorption_per_atm_cm(w)) for w in (460.0, 500.0, 550.0))
OMEGA_420 = K460**2 / (K500 * K550)
OMEGA_380 = K460**3 / (K500**2 * K550)
LOG_420_380 = math.log(420.0 / 380.0)
KS = (0.99, 1.00, 1.01)
MS = (0.0, 1.0, 2.0)
MEMBERS = [(k, m) for k in KS for m in MS]
NODES = (20.0, 27.6, 40.0, 59.4, 90.0, 131.8)
ALTITUDES = (340.0, 15000.0)
CENTRES = [340.0 + 40.0 * i for i in range(11)]
CRITERION_SUNS = (-1.0, 0.0, 1.0, 10.0)
FIT_SETTINGS = {"method": "trf", "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15, "max_nfev": 20000}
FD = 1e-5
SOLVES_PER_MEMBER = 1 + 20 + 20
"""One six-node fit, then the central differences: 10 channels, two signs, for each gradient."""


def at550(v: np.ndarray) -> np.ndarray:
    return 0.75 * v[..., 4] + 0.25 * v[..., 5]


def ref_power_law(z380: float, z420: float) -> tuple[float, float, str]:
    """(beta, eta, regime), the criterion's two branches."""
    if z380 > 0.0 and z420 > 0.0:
        eta = math.log(z380 / z420) / LOG_420_380
        if 0.0 <= eta <= 3.0:
            return z380 * (380.0 / 550.0) ** eta, eta, "exact"
    raw = math.log(max(z380, 1e-12) / max(z420, 1e-12)) / LOG_420_380
    eta = min(max(raw, 0.0), 3.0)
    w380, w420 = (380.0 / 550.0) ** (-eta), (420.0 / 550.0) ** (-eta)
    beta = (w380 * z380 + w420 * z420) / (w380**2 + w420**2)
    return max(beta, 0.0), eta, "guard"


def ref_ozone(y15_mean: np.ndarray, k: float, m: float) -> dict:
    """The criterion's passes, from no tail, stopping at the first that moves neither tail value
    by more than 1e-15."""
    r = y15_mean - k * R15
    t380 = t420 = 0.0
    for passes in range(1, 201):
        beta, eta, regime = ref_power_law(float(r[0]) - t380, float(r[1]) - t420)
        o = r - beta * (LAM / 550.0) ** (-eta)
        o_bar = float(at550(o))
        n380 = m * OMEGA_380 * o_bar if o_bar > 0.0 else 0.0
        n420 = m * OMEGA_420 * o_bar if o_bar > 0.0 else 0.0
        done = abs(n380 - t380) <= 1e-15 and abs(n420 - t420) <= 1e-15
        t380, t420 = n380, n420
        if done:
            o[0], o[1] = t380, t420
            return {"o": o, "regime": regime, "passes": passes, "t380": t380, "t420": t420,
                    "eta": eta, "beta": beta}  # fmt: skip
    raise AssertionError("the reference tail did not settle")


def ref_aerosol(p: np.ndarray) -> np.ndarray:
    u, al, ga = p[1::3], p[2::3], p[3::3]
    return np.exp(u[:, None] - al[:, None] * X[None, :] + ga[:, None] * X[None, :] ** 2)


def ref_sixnode(y0: np.ndarray, o: np.ndarray, k: float, start: np.ndarray | None = None) -> dict:
    """The six-node fit: 19 parameters, the criterion's start and settings."""
    target = y0 - k * R_PAPER * RHO[None, :]
    o_bar = float(at550(o))
    if start is None:
        rho_bar = float(at550(RHO))
        total = k * R_PAPER * rho_bar + o_bar
        start = np.empty(19)
        start[0] = 1.0
        start[1::3] = np.log(np.maximum(at550(y0) - total, 0.001))
        start[2::3] = 1.3
        start[3::3] = 0.0

    def residual(p: np.ndarray) -> np.ndarray:
        return (p[0] * o[None, :] + ref_aerosol(p) - target).ravel()

    def jacobian(p: np.ndarray) -> np.ndarray:
        a = ref_aerosol(p)
        j = np.zeros((60, 19))
        for i in range(6):
            rows = slice(i * 10, (i + 1) * 10)
            j[rows, 0] = o
            j[rows, 1 + 3 * i] = a[i]
            j[rows, 2 + 3 * i] = -X * a[i]
            j[rows, 3 + 3 * i] = X**2 * a[i]
        return j

    result = least_squares(residual, start, jac=jacobian, **FIT_SETTINGS)
    final = residual(result.x)
    gas = k * R_PAPER * float(at550(RHO)) + float(result.x[0]) * o_bar
    return {
        "x": result.x,
        "status": result.status,
        "gas": gas,
        "ssr": float(final @ final),
        "jac": jacobian(result.x),
        "o_bar": o_bar,
    }


def ref_member(y0: np.ndarray, y15: np.ndarray, ybar_p: float, k: float, m: float) -> dict:
    """One member's quantities by the criterion's steps 1 to 5, using the helpers above."""
    y_mean = y15.mean(axis=0)
    ozone = ref_ozone(y_mean, k, m)
    fit = ref_sixnode(y0, ozone["o"], k)
    s = math.sqrt(fit["ssr"] / 41.0)
    cov00 = s**2 * np.linalg.inv(fit["jac"].T @ fit["jac"])[0, 0]
    se_fit = fit["o_bar"] * math.sqrt(cov00)
    grad15 = np.empty(10)
    grad0 = np.empty(10)
    for c in range(10):
        step = np.zeros(10)
        step[c] = FD
        plus = ref_sixnode(y0, ref_ozone(y_mean + step, k, m)["o"], k, start=fit["x"])
        minus = ref_sixnode(y0, ref_ozone(y_mean - step, k, m)["o"], k, start=fit["x"])
        grad15[c] = (plus["gas"] - minus["gas"]) / (2 * FD)
        plus = ref_sixnode(y0 + step[None, :], ozone["o"], k, start=fit["x"])
        minus = ref_sixnode(y0 - step[None, :], ozone["o"], k, start=fit["x"])
        grad0[c] = (plus["gas"] - minus["gas"]) / (2 * FD)
    s15 = ref_aloft_scale(y15)
    n15, n0 = float(np.linalg.norm(grad15)), float(np.linalg.norm(grad0))
    se_ext = n15 * math.sqrt(s15**2 / 6 + s**2)
    se_0 = n0 * s
    se = math.sqrt(se_fit**2 + se_ext**2 + se_0**2)
    return {
        "ozone": ozone, "fit": fit, "s": s, "se_fit": se_fit, "grad15": n15, "grad0": n0,
        "se_ext": se_ext, "se_0": se_0, "se": se, "gas": fit["gas"],
        "g": ybar_p - fit["gas"] - 0.1,
    }  # fmt: skip


def ref_aloft_scale(y15: np.ndarray) -> float:
    d = y15 - y15.mean(axis=0)
    basis = np.stack([np.ones(10), X, X**2], axis=1)
    coef, *_ = np.linalg.lstsq(basis, d.T, rcond=None)
    r = d.T - basis @ coef
    return math.sqrt(float((r**2).sum()) / 35.0)


# --- synthetic spectra ----------------------------------------------------------------------------


def synthetic(cell: str, index: int = 0, g_true: float = 0.0121) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic truth's -ln T at the six nodes, 0.34 km and 15 km, from the power check."""
    power = load_power()
    truth = replace(power.cells()[cell], g_true=g_true)
    rng = np.random.default_rng(zlib.crc32(f"{cell}|{index}".encode()))
    perturbation = power.PERTURBATIONS[index % len(power.PERTURBATIONS)]
    return power.draw(rng, truth, perturbation)


@cache
def base_pair() -> tuple[np.ndarray, np.ndarray]:
    return synthetic("base_s1", 0)


YBAR_P = 0.235751
"""The power check's ȳ(V_p), passed to both implementations."""


@cache
def base_estimate() -> object:
    y0, y15 = base_pair()
    return load_tool().aloft_estimate(y0, y15, YBAR_P)


def aloft_dict(y0: np.ndarray, y15: np.ndarray, y340: tuple[float, float] = (0.45, 0.012)) -> dict:
    """An aloft fixture as the builder writes it, from -ln T at the ten fitted channels (6 x 10)
    and, for the 340 nm channel, one value of -ln T per altitude."""

    def block(y: np.ndarray, first: float) -> list[list[float]]:
        return [[float(f"{math.exp(-v):.9g}") for v in (first, *row)] for row in y]

    return {
        "format": "lastlight-prague-aloft",
        "version": 1,
        "sun_elevation_deg": 10.0,
        "ground_albedo": 0.2,
        "visibility_nodes_km": list(NODES),
        "altitudes_m": list(ALTITUDES),
        "channel_centres_nm": list(CENTRES),
        "transmittance": [block(y0, y340[0]), block(y15, y340[1])],
        "provenance": {"command": "uv run python tools/prague_reference.py fixture --aloft"},
    }


FIXTURE_FILES = {10.0: "sun10.json", 0.0: "sun0.json", -1.0: "sun-1.json", 1.0: "sun1.json"}


@cache
def _committed(sun: float) -> dict:
    return json.loads((FIXTURE_DIR / FIXTURE_FILES[sun]).read_text())


def four_fixtures() -> dict[float, dict]:
    return {sun: copy.deepcopy(_committed(sun)) for sun in FIXTURE_FILES}


@cache
def g_stars() -> dict[float, float]:
    """#55's crossings at the four suns, unrounded, as `report --depth` computes them."""
    pilot = load_tool().depth_pilot({sun: _committed(sun) for sun in FIXTURE_FILES})
    return {row.sun: row.g_star for row in pilot.suns}


# --- the agreement test ---------------------------------------------------------------------------

AGREEMENT_CELLS = (
    "base_s1",
    "base_s3",
    "base_s10",
    "base_s20",
    "tail05_s1",
    "tail1_s1",
    "tail2_s1",
    "tail3_s1",
    "tail420_s3",
    "dr_m2_s1",
    "dr_p2_s3",
    "asteep_s1",
    "aflat_s3",
    "bigaloft_s3",
    "band_s1",
    "topcommon_s3",
)
AGREEMENT_PAIRS = [(cell, i, 0.0121) for cell in AGREEMENT_CELLS for i in range(2)] + [
    # True excesses far from the paper's, so "not this" and the high "contributes" occur.
    ("base_s1", 2, 0.004),
    ("base_s3", 2, 0.006),
    ("tail1_s1", 2, 0.0),
    ("base_s1", 3, 0.022),
    ("tail2_s1", 2, 0.0095),
]


@cache
def _pair_estimates(cell: str, index: int, g_true: float) -> tuple[object, object]:
    y0, y15 = synthetic(cell, index, g_true)
    power = load_power()
    return load_tool().aloft_estimate(y0, y15, power.YBAR_P), power.estimate(y0, y15, power.YBAR_P)


def test_the_agreement_test_covers_at_least_twenty_pairs_and_both_kinds_of_cell() -> None:
    assert len(AGREEMENT_PAIRS) >= 20
    assert len(set(AGREEMENT_PAIRS)) == len(AGREEMENT_PAIRS)
    names = {cell for cell, _, _ in AGREEMENT_PAIRS}
    assert any(n.startswith("tail") for n in names) and any(n.startswith("band") for n in names)
    assert any(n.startswith("base") for n in names)


@pytest.mark.parametrize(("cell", "index", "g_true"), AGREEMENT_PAIRS)
def test_the_envelope_and_the_verdict_agree_with_the_power_checks_to_1e_minus_6(
    cell: str, index: int, g_true: float
) -> None:
    tool, power = load_tool(), load_power()
    ours, theirs = _pair_estimates(cell, index, g_true)
    lo, hi = theirs.envelope()
    assert ours.g_lo == pytest.approx(lo, abs=1e-6)
    assert ours.g_hi == pytest.approx(hi, abs=1e-6)
    # The same envelope into both verdict functions: the lines must be the same.
    for end_lo, end_hi in ((ours.g_lo, ours.g_hi), (lo, hi)):
        line = tool.aloft_verdict(end_lo, end_hi, power.G_STARS)
        assert line.verdict == power.verdict(end_lo, end_hi)
        for sun, g_star in power.G_STARS.items():
            assert line.outcome_at(sun) == power.outcome(end_lo, end_hi, g_star)


@pytest.mark.parametrize(("cell", "index", "g_true"), AGREEMENT_PAIRS)
def test_every_member_agrees_with_the_power_check_not_only_the_envelope(
    cell: str, index: int, g_true: float
) -> None:
    ours, theirs = _pair_estimates(cell, index, g_true)
    assert [(m.k, m.m) for m in ours.members] == MEMBERS
    np.testing.assert_allclose([m.g for m in ours.members], theirs.g, rtol=0, atol=1e-8)
    np.testing.assert_allclose([m.fit.s for m in ours.members], theirs.s_fit, rtol=1e-8, atol=0)
    np.testing.assert_allclose([m.grad15 for m in ours.members], theirs.grad15, rtol=1e-4, atol=0)
    np.testing.assert_allclose([m.grad0 for m in ours.members], theirs.grad0, rtol=1e-4, atol=0)
    np.testing.assert_allclose([m.se for m in ours.members], theirs.se_tot, rtol=1e-4, atol=1e-9)
    assert ours.scale.s15 == pytest.approx(theirs.s15, rel=1e-9, abs=0)


def test_the_agreement_pairs_reach_both_regimes_of_the_power_law() -> None:
    regimes: set[str] = set()
    for pair in AGREEMENT_PAIRS:
        ours, _ = _pair_estimates(*pair)
        regimes |= {m.ozone.regime for m in ours.members}
    assert regimes == {"exact", "guard"}


def test_the_agreement_pairs_include_the_guard_and_the_exact_branch_in_the_same_member() -> None:
    """Both branches for the plain member (k = 1, m = 0): the power check's `regime`."""
    seen = set()
    for pair in AGREEMENT_PAIRS:
        ours, theirs = _pair_estimates(*pair)
        plain = next(m for m in ours.members if (m.k, m.m) == (1.0, 0.0))
        assert plain.ozone.regime == theirs.regime
        seen.add(plain.ozone.regime)
    assert seen == {"exact", "guard"}


def test_the_agreement_pairs_reach_every_outcome_a_verdict_can_take_but_mixed() -> None:
    tool, power = load_tool(), load_power()
    outcomes: set[str] = set()
    for pair in AGREEMENT_PAIRS:
        ours, _ = _pair_estimates(*pair)
        outcomes.add(tool.aloft_verdict(ours.g_lo, ours.g_hi, power.G_STARS).verdict)
    assert {"contributes", "spans", "not this"} <= outcomes


# --- step 1: the power law ------------------------------------------------------------------------


def test_the_power_law_is_exact_through_both_points_when_both_are_positive_and_in_range() -> None:
    tool = load_tool()
    beta, eta, regime = tool.aloft_power_law(0.02, 0.015)
    assert regime == "exact"
    assert eta == pinned(2.8744217735839825)  # ln(0.02 / 0.015) / ln(420 / 380)
    assert beta == pinned(0.006909680315843062)  # 0.02 (380 / 550)^eta
    assert beta * (380.0 / 550.0) ** (-eta) == pinned(0.02)
    assert beta * (420.0 / 550.0) ** (-eta) == pinned(0.015)


def test_eta_equal_to_zero_is_inside_the_guard_and_exact() -> None:
    beta, eta, regime = load_tool().aloft_power_law(0.01, 0.01)
    assert (regime, eta) == ("exact", 0.0)
    assert beta == pinned(0.01)


def test_an_eta_above_three_is_clipped_and_the_amplitude_is_the_least_squares_one() -> None:
    beta, eta, regime = load_tool().aloft_power_law(0.05, 0.01)  # eta_2 = 16.08
    assert (regime, eta) == ("guard", 3.0)
    assert beta == pinned(0.012226465637933137)  # (w380 z380 + w420 z420) / (w380^2 + w420^2)


def test_an_eta_below_zero_is_clipped_to_zero_and_the_amplitude_is_the_mean() -> None:
    beta, eta, regime = load_tool().aloft_power_law(0.01, 0.02)  # eta_2 = -6.9
    assert (regime, eta) == ("guard", 0.0)
    assert beta == pinned(0.015)


def test_a_residual_at_or_under_zero_takes_the_guard_whichever_end_it_is() -> None:
    tool = load_tool()
    beta, eta, regime = tool.aloft_power_law(-0.001, 0.01)
    assert (regime, eta) == ("guard", 0.0)  # ln(1e-12 / 0.01) is far below 0
    assert beta == pinned(0.0045)
    beta, eta, regime = tool.aloft_power_law(0.01, -0.002)
    assert (regime, eta) == ("guard", 3.0)  # ln(0.01 / 1e-12) is far above 3
    assert beta == pinned(0.0018143298606346793)
    beta, eta, regime = tool.aloft_power_law(0.01, 0.0)  # exactly 0 is not positive
    assert (regime, eta) == ("guard", 3.0)


def test_the_amplitude_is_floored_at_zero_and_both_residuals_under_zero_give_eta_zero() -> None:
    beta, eta, regime = load_tool().aloft_power_law(-0.004, -0.003)
    assert (beta, eta, regime) == (0.0, 0.0, "guard")  # the two floors are 1e-12 each: ln 1 = 0
    beta, _, _ = load_tool().aloft_power_law(-0.004, 0.001)
    assert beta == 0.0  # w = 1: (-0.004 + 0.001) / 2 is under 0, floored


def test_the_residuals_are_floored_at_1e_minus_12_inside_the_logarithm_only() -> None:
    beta, eta, regime = load_tool().aloft_power_law(1.2e-12, 3e-13)
    assert regime == "guard"  # eta_2 = ln 4 / ln(420 / 380) = 13.85
    assert eta == pinned(1.8216952074068224)  # ln(1.2e-12 / 1e-12) / ln(420 / 380): the floor
    assert beta == pinned(4.3632942735514586e-13)  # the amplitude uses the unfloored residuals


def test_the_guard_bounds_are_inclusive_at_both_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eta_2 exactly on a bound of the guard is exact; one ulp outside it is the guard."""
    tool = load_tool()
    _, eta, regime = tool.aloft_power_law(0.02, 0.015)
    assert regime == "exact"
    for bounds, expected in (
        ((eta, 3.0), "exact"),
        ((0.0, eta), "exact"),
        ((math.nextafter(eta, math.inf), 3.0), "guard"),
        ((0.0, math.nextafter(eta, 0.0)), "guard"),
    ):
        monkeypatch.setattr(tool, "ALOFT_ETA_GUARD", bounds)
        assert tool.aloft_power_law(0.02, 0.015)[2] == expected, bounds


def test_the_registered_guard_is_zero_to_three_and_the_floor_is_1e_minus_12() -> None:
    tool = load_tool()
    assert tool.ALOFT_ETA_GUARD == (0.0, 3.0)
    assert tool.ALOFT_RESIDUAL_FLOOR == 1e-12
    assert tool.ALOFT_TAIL_TOLERANCE == 1e-15 and tool.ALOFT_TAIL_MAX_PASSES == 200
    assert tool.ALOFT_STEP == 1e-5


@pytest.mark.parametrize(
    ("z380", "z420"),
    [(0.02, 0.015), (0.05, 0.01), (0.01, 0.02), (-0.001, 0.01), (0.01, -0.002), (0.013, 0.0117)],
)
def test_the_power_law_matches_the_independent_derivation(z380: float, z420: float) -> None:
    ours = load_tool().aloft_power_law(z380, z420)
    theirs = ref_power_law(z380, z420)
    assert ours[2] == theirs[2]
    assert ours[0] == pinned(theirs[0]) and ours[1] == pinned(theirs[1])


# --- step 1: the ozone tail and its passes --------------------------------------------------------


def test_the_tail_ratios_are_the_log_linear_continuation_of_the_coefficients_at_460_and_500() -> (
    None
):
    tool = load_tool()
    assert pytest.approx(0.014118, abs=5e-7) == OMEGA_420  # the criterion's printed values
    assert pytest.approx(0.0028235, abs=5e-8) == OMEGA_380
    assert pinned(OMEGA_420) == tool.ALOFT_TAIL_RATIO_420
    assert pinned(OMEGA_380) == tool.ALOFT_TAIL_RATIO_380
    assert (
        pinned(0.006**2 / (0.03 * 0.085)) == tool.ALOFT_TAIL_RATIO_420
    )  # SPCTRL2: 0.006, 0.03, 0.085
    assert pinned(0.006**3 / (0.03**2 * 0.085)) == tool.ALOFT_TAIL_RATIO_380


def test_the_rayleigh_at_15_km_is_computed_at_121_11_hpa_exactly() -> None:
    tool = load_tool()
    assert tool.ALOFT_PRESSURE_HPA == 121.11
    np.testing.assert_array_equal(tool.ALOFT_R15, R15)
    assert tool.ALOFT_R_PAPER == R_PAPER
    assert pytest.approx(0.093050, abs=5e-7) == tool.ALOFT_R_PAPER  # as the criterion prints it
    np.testing.assert_array_equal(tool.ALOFT_RHO, RHO)
    assert float(at550(RHO)) == pytest.approx(1.00968, abs=5e-6)  # the criterion's rho-bar


def test_the_members_are_the_nine_pairs_of_rayleigh_factor_and_tail_multiplier() -> None:
    tool = load_tool()
    assert list(tool.ALOFT_MEMBERS) == MEMBERS
    assert tool.ALOFT_RAYLEIGH_FACTORS == KS and tool.ALOFT_TAIL_MULTIPLIERS == MS


def y15_mean_for_tail() -> np.ndarray:
    """A mean spectrum at 15 km: Rayleigh, 0.016 of ozone at 550 nm, a weak aerosol aloft."""
    ozone = np.array([0.0, 0.0, 0.0065, 0.0345, 0.0975, 0.0924, 0.0765, 0.0405, 0.0113, 0.0035])
    return R15 + ozone * 0.5 + 0.004 * (LAM / 550.0) ** (-1.2)


@pytest.mark.parametrize(("k", "m"), MEMBERS)
def test_the_passes_and_the_ozone_match_the_independent_derivation(k: float, m: float) -> None:
    tool = load_tool()
    y_mean = y15_mean_for_tail()
    ours = tool.aloft_ozone(y_mean, k, m)
    theirs = ref_ozone(y_mean, k, m)
    np.testing.assert_allclose(ours.o, theirs["o"], rtol=1e-10, atol=1e-14)
    assert ours.passes == theirs["passes"]
    assert ours.regime == theirs["regime"]
    assert ours.t380 == pytest.approx(theirs["t380"], abs=1e-14)
    assert ours.t420 == pytest.approx(theirs["t420"], abs=1e-14)
    assert ours.eta == pinned(theirs["eta"]) and ours.beta == pinned(theirs["beta"])


def test_without_a_tail_the_first_pass_settles_and_the_ozone_at_380_and_420_is_zero() -> None:
    ours = load_tool().aloft_ozone(y15_mean_for_tail(), 1.0, 0.0)
    assert ours.passes == 1
    assert (ours.t380, ours.t420) == (0.0, 0.0)
    assert ours.o[0] == 0.0 and ours.o[1] == 0.0


def test_with_a_tail_the_passes_stop_at_a_fixed_point_tied_to_the_ozone_at_550_nm() -> None:
    tool = load_tool()
    for m in (1.0, 2.0):
        ours = tool.aloft_ozone(y15_mean_for_tail(), 1.0, m)
        assert 2 <= ours.passes <= 200
        o_bar = float(at550(ours.o))
        assert o_bar > 0.0
        assert ours.t380 == pytest.approx(m * OMEGA_380 * o_bar, abs=1e-14)
        assert ours.t420 == pytest.approx(m * OMEGA_420 * o_bar, abs=1e-14)
        assert ours.o[0] == ours.t380 and ours.o[1] == ours.t420  # the new values
        assert ours.o_bar == pinned(o_bar)


def test_the_passes_stop_at_the_first_one_that_moves_neither_tail_by_more_than_1e_minus_15() -> (
    None
):
    """One pass fewer than the independent derivation counts would leave the tail unsettled; one
    more would not be the first. The count is the pass count, settling pass included."""
    tool = load_tool()
    ours = tool.aloft_ozone(y15_mean_for_tail(), 1.0, 2.0)
    theirs = ref_ozone(y15_mean_for_tail(), 1.0, 2.0)
    assert ours.passes == theirs["passes"] > 2


def test_each_pass_starts_from_the_residuals_less_the_latest_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    seen: list[tuple[float, float]] = []
    real = tool.aloft_power_law

    def spy(z380: float, z420: float):
        seen.append((z380, z420))
        return real(z380, z420)

    monkeypatch.setattr(tool, "aloft_power_law", spy)
    y_mean = y15_mean_for_tail()
    ours = tool.aloft_ozone(y_mean, 1.0, 1.0)
    r = y_mean - R15
    assert seen[0] == (float(r[0]), float(r[1]))  # the first pass: no tail
    assert len(seen) == ours.passes
    # The last pass was taken with the tail the previous one set (within 1e-15 of the new one).
    assert seen[-1][0] == pytest.approx(float(r[0]) - ours.t380, abs=1e-14)
    assert seen[-1][1] == pytest.approx(float(r[1]) - ours.t420, abs=1e-14)


def test_the_tail_is_zero_where_the_ozone_at_550_nm_is_not_positive() -> None:
    y_mean = R15 - 0.01  # every residual is -0.01: no ozone, no aerosol
    ours = load_tool().aloft_ozone(y_mean, 1.0, 2.0)
    assert (ours.t380, ours.t420) == (0.0, 0.0)
    assert ours.passes == 1
    assert ours.o_bar <= 0.0
    assert ours.regime == "guard"


def test_a_tail_that_has_not_settled_in_200_passes_is_a_refusal_naming_the_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    count = []
    real = tool.aloft_power_law

    def counting(z380: float, z420: float):
        count.append(1)
        return real(z380, z420)

    monkeypatch.setattr(tool, "aloft_power_law", counting)
    monkeypatch.setattr(tool, "ALOFT_TAIL_TOLERANCE", -1.0)  # no pass can settle
    with pytest.raises(tool.PragueError, match=re.escape("member (k=1.01, m=2)")):
        tool.aloft_ozone(y15_mean_for_tail(), 1.01, 2.0)
    assert len(count) == 200


def test_the_ozone_aloft_uses_the_six_node_mean_less_k_times_the_15_km_rayleigh() -> None:
    tool = load_tool()
    y_mean = y15_mean_for_tail()
    for k in KS:
        ours = tool.aloft_ozone(y_mean, k, 0.0)
        theirs = ref_ozone(y_mean, k, 0.0)
        np.testing.assert_allclose(ours.o, theirs["o"], rtol=1e-10, atol=1e-14)
    low, high = tool.aloft_ozone(y_mean, 0.99, 0.0), tool.aloft_ozone(y_mean, 1.01, 0.0)
    assert high.o_bar < low.o_bar  # more Rayleigh aloft leaves less for the ozone


# --- step 2: the six-node fit ---------------------------------------------------------------------


def spy_solver(monkeypatch: pytest.MonkeyPatch, tool: ModuleType, fail_calls: tuple[int, ...] = ()):
    """Wrap the tool's `least_squares`; the six-node fits (19 parameters) are counted from 1 and
    the ones in `fail_calls` report status 0. Other fits (#55's) pass through uncounted."""
    calls: list[dict] = []

    def spy(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        if np.size(args[1]) != 19:
            return result
        calls.append({"args": args, "kwargs": kwargs, "x": np.array(result.x)})
        if len(calls) in fail_calls:
            result.status = 0
        return result

    monkeypatch.setattr(tool, "least_squares", spy)
    return calls


def test_the_solver_is_called_with_the_registered_settings_and_the_registered_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.01, 0.0)["o"]
    calls = spy_solver(monkeypatch, tool)
    tool.aloft_fit(y0, o, 1.01)
    (call,) = calls
    assert len(call["args"]) == 2  # the residual function and the start; the rest by keyword
    assert set(call["kwargs"]) == {"jac", "method", "xtol", "ftol", "gtol", "max_nfev"}
    assert callable(call["kwargs"]["jac"])
    assert call["kwargs"]["method"] == "trf"
    assert call["kwargs"]["xtol"] == call["kwargs"]["ftol"] == call["kwargs"]["gtol"] == 1e-15
    assert call["kwargs"]["max_nfev"] == 20000
    start = call["args"][1]
    assert start.shape == (19,)
    assert start[0] == 1.0
    total = 1.01 * R_PAPER * float(at550(RHO)) + float(at550(o))
    np.testing.assert_allclose(
        start[1::3], np.log(np.maximum(at550(y0) - total, 0.001)), rtol=1e-14
    )
    np.testing.assert_array_equal(start[2::3], np.full(6, 1.3))
    np.testing.assert_array_equal(start[3::3], np.zeros(6))


def test_a_start_depth_under_0_001_is_raised_to_it_before_the_logarithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]
    flat = y0 - 5.0  # every node's ybar less the gas is far below 0
    calls = spy_solver(monkeypatch, tool)
    with contextlib.suppress(tool.PragueError):  # whether it then converges is not graded
        tool.aloft_fit(flat, o, 1.0)
    np.testing.assert_allclose(calls[0]["args"][1][1::3], np.full(6, math.log(0.001)), rtol=1e-14)


def test_the_fit_matches_the_independent_six_node_fit() -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    for k, m in ((0.99, 0.0), (1.0, 1.0), (1.01, 2.0)):
        o = ref_ozone(y15.mean(axis=0), k, m)["o"]
        ours = tool.aloft_fit(y0, o, k)
        theirs = ref_sixnode(y0, o, k)
        assert ours.status > 0 and theirs["status"] > 0
        assert ours.gas == pytest.approx(theirs["gas"], abs=1e-10)
        assert ours.ssr == pytest.approx(theirs["ssr"], rel=1e-8)
        assert ours.x.size == 19
        assert ours.gas == pinned(k * R_PAPER * float(at550(RHO)) + ours.a * float(at550(o)))


def test_the_residual_scale_has_41_degrees_of_freedom_and_the_rms_is_over_the_60_values() -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]
    ours = tool.aloft_fit(y0, o, 1.0)
    theirs = ref_sixnode(y0, o, 1.0)
    assert ours.x.size == 19 and 60 - ours.x.size == 41
    assert ours.s == pinned(math.sqrt(theirs["ssr"] / 41.0))
    assert ours.rms == pinned(math.sqrt(theirs["ssr"] / 60.0))
    assert ours.s**2 * 41.0 == pinned(ours.ssr)


def test_the_standard_error_takes_q_equal_to_the_ozone_at_550_nm_for_a_alone() -> None:
    """SE_fit = sqrt(q' C q), C = s_k^2 (J'J)^-1, q = O-bar for a and 0 for the aerosol."""
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 1.0)["o"]
    ours = tool.aloft_fit(y0, o, 1.0)
    theirs = ref_sixnode(y0, o, 1.0)
    s2 = theirs["ssr"] / 41.0
    inverse = np.linalg.inv(theirs["jac"].T @ theirs["jac"])
    expected = float(at550(o)) * math.sqrt(s2 * inverse[0, 0])
    assert ours.se == pinned(expected)
    q = np.zeros(19)
    q[0] = float(at550(o))
    assert ours.se == pinned(math.sqrt(q @ (s2 * inverse) @ q))
    assert ours.se != pinned(math.sqrt(s2 * inverse[0, 0]))  # q = 1 would be a different number


def test_the_condition_number_is_that_of_the_unscaled_analytic_jacobian_at_the_solution() -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]
    ours = tool.aloft_fit(y0, o, 1.0)
    theirs = ref_sixnode(y0, o, 1.0)
    assert ours.condition == pinned(float(np.linalg.cond(theirs["jac"])))
    assert ours.jac.shape == (60, 19)


def test_the_standard_error_is_infinite_where_the_jacobian_is_singular() -> None:
    tool = load_tool()
    y0, _ = base_pair()
    fit = tool.aloft_fit(y0, np.zeros(10), 1.0)  # no ozone: the column of a is 0, J'J is singular
    assert fit.se == math.inf
    assert math.isfinite(fit.gas)


def test_the_standard_error_is_infinite_where_inverse_raises_or_the_variance_is_not_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]

    def raises(a):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "inv", raises)
    assert tool.aloft_fit(y0, o, 1.0).se == math.inf
    monkeypatch.setattr(np.linalg, "inv", lambda a: -np.eye(a.shape[0]))  # a negative variance
    assert tool.aloft_fit(y0, o, 1.0).se == math.inf
    monkeypatch.setattr(np.linalg, "inv", lambda a: np.zeros_like(a))  # a zero variance
    assert tool.aloft_fit(y0, o, 1.0).se == math.inf
    monkeypatch.setattr(np.linalg, "inv", lambda a: np.full_like(a, np.nan))  # not finite
    assert tool.aloft_fit(y0, o, 1.0).se == math.inf


def test_a_fit_with_a_status_at_or_under_zero_is_refused_naming_the_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    spy_solver(monkeypatch, tool, fail_calls=(1,))
    with pytest.raises(tool.PragueError, match=re.escape("member (k=0.99, m=0)")):
        tool.aloft_estimate(y0, y15, YBAR_P)


@pytest.mark.parametrize("status", [1, 2, 3, 4])
def test_a_converged_status_is_not_refused(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]

    def converged(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.status = status
        return result

    monkeypatch.setattr(tool, "least_squares", converged)
    assert tool.aloft_fit(y0, o, 1.0).status == status


@pytest.mark.parametrize("status", [0, -1])
def test_a_status_of_zero_or_under_is_refused(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    o = ref_ozone(y15.mean(axis=0), 1.0, 0.0)["o"]

    def failing(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.status = status
        return result

    monkeypatch.setattr(tool, "least_squares", failing)
    with pytest.raises(tool.PragueError, match="did not converge"):
        tool.aloft_fit(y0, o, 1.0, where="member (k=1, m=0)")


# --- step 3: the node-to-node scale at 15 km ------------------------------------------------------


def unit_cubic() -> np.ndarray:
    """A unit vector over the ten channels orthogonal to 1, x and x^2: no quadratic can fit it."""
    basis = np.stack([np.ones(10), X, X**2, X**3], axis=1)
    q, _ = np.linalg.qr(basis)
    u = q[:, 3]
    assert abs(np.linalg.norm(u) - 1.0) < 1e-12
    assert np.abs(u @ basis[:, :3]).max() < 1e-12
    return u


def test_the_scale_aloft_has_35_degrees_of_freedom() -> None:
    """Node 0 and node 1 differ from the mean by +d u and -d u, u orthogonal to the quadratics: the
    quadratic removes nothing, SSR_15 = 2 d^2, and s_15 = d sqrt(2 / 35)."""
    d = 0.01
    y15 = np.tile(R15 + 0.03 * np.linspace(1, 2, 10), (6, 1))
    u = unit_cubic()
    y15[0] += d * u
    y15[1] -= d * u
    scale = load_tool().aloft_scale(y15)
    assert scale.s15 == pinned(d * math.sqrt(2.0 / 35.0))
    assert scale.ssr == pinned(2 * d**2)
    assert scale.s15 != pinned(d * math.sqrt(2.0 / 36.0))  # not 60 - 10 - 14
    assert scale.s15 != pinned(d * math.sqrt(2.0 / 30.0))  # not 60 - 30


def test_a_difference_that_is_a_quadratic_in_the_log_wavelength_leaves_nothing() -> None:
    rng = np.random.default_rng(5)
    base = R15 + 0.03 * np.linspace(1, 2, 10)
    c = rng.normal(0.0, 0.002, (6, 3))
    c -= c.mean(axis=0)  # the six nodes' differences sum to zero
    y15 = base[None, :] + c[:, 0:1] + c[:, 1:2] * X[None, :] + c[:, 2:3] * X[None, :] ** 2
    scale = load_tool().aloft_scale(y15)
    assert scale.s15 < 1e-12
    np.testing.assert_allclose(scale.d550, c[:, 0], atol=1e-12)  # the constant coefficient
    np.testing.assert_allclose(scale.coefficients, c, atol=1e-12)  # 1, x, x^2 in that order


def test_the_scale_matches_the_independent_derivation_on_noisy_spectra() -> None:
    _, y15 = base_pair()
    scale = load_tool().aloft_scale(y15)
    assert scale.s15 == pinned(ref_aloft_scale(y15))
    # D at 550 nm is the constant coefficient of each node's fitted difference.
    d = y15 - y15.mean(axis=0)
    basis = np.stack([np.ones(10), X, X**2], axis=1)
    coef, *_ = np.linalg.lstsq(basis, d.T, rcond=None)
    np.testing.assert_allclose(scale.d550, coef[0], rtol=1e-9, atol=1e-14)
    assert scale.d550.shape == (6,)


# --- step 4: the gradients and the errors ---------------------------------------------------------


def test_each_member_runs_41_fits_and_each_perturbed_fit_starts_at_the_unperturbed_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    calls = spy_solver(monkeypatch, tool)
    tool.aloft_estimate(y0, y15, YBAR_P)
    assert len(calls) == 9 * SOLVES_PER_MEMBER
    for member in range(9):
        block = calls[member * SOLVES_PER_MEMBER : (member + 1) * SOLVES_PER_MEMBER]
        solution = block[0]["x"]
        for later in block[1:]:
            np.testing.assert_array_equal(later["args"][1], solution)


def test_the_gradients_are_central_differences_with_a_step_of_1e_minus_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """15 km: the mean spectrum is raised and lowered by 1e-5 in one channel at a time and the
    ozone is redone from no tail; 0.34 km: one channel at all six nodes at once, O held."""
    tool = load_tool()
    y0, y15 = base_pair()
    ozone_inputs: list[np.ndarray] = []
    fit_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    real_ozone, real_fit = tool.aloft_ozone, tool.aloft_fit

    def ozone_spy(y_mean, k, m, **kwargs):
        ozone_inputs.append(np.array(y_mean))
        return real_ozone(y_mean, k, m, **kwargs)

    def fit_spy(y, o, k, **kwargs):
        fit_inputs.append((np.array(y), np.array(o)))
        return real_fit(y, o, k, **kwargs)

    monkeypatch.setattr(tool, "aloft_ozone", ozone_spy)
    monkeypatch.setattr(tool, "aloft_fit", fit_spy)
    tool.aloft_estimate(y0, y15, YBAR_P)
    mean = y15.mean(axis=0)
    first = ozone_inputs[:21]  # the member's own, then the 20 raised and lowered
    assert np.array_equal(first[0], mean)
    steps = [np.array(a) - mean for a in first[1:21]]
    for c in range(10):
        up, down = steps[2 * c], steps[2 * c + 1]
        assert up[c] == pytest.approx(FD, rel=1e-9) and down[c] == pytest.approx(-FD, rel=1e-9)
        assert (
            np.count_nonzero(np.abs(up) > 1e-12) == 1
            and np.count_nonzero(np.abs(down) > 1e-12) == 1
        )
    # 0.34 km: the 21st to 40th perturbed fits of the member take y0 +- 1e-5 in one channel, at
    # all six nodes, and the member's own unperturbed O.
    own = fit_inputs[0]
    np.testing.assert_array_equal(own[0], y0)
    for c in range(10):
        up, down = fit_inputs[21 + 2 * c], fit_inputs[22 + 2 * c]
        np.testing.assert_array_equal(up[1], own[1])  # O held at its unperturbed value
        np.testing.assert_array_equal(down[1], own[1])
        raised, lowered = up[0] - y0, down[0] - y0
        np.testing.assert_allclose(raised[:, c], FD, rtol=1e-9)
        np.testing.assert_allclose(lowered[:, c], -FD, rtol=1e-9)
        assert np.count_nonzero(np.abs(raised) > 1e-12) == 6


@pytest.mark.parametrize(("k", "m"), [(0.99, 0.0), (1.0, 1.0), (1.01, 2.0)])
def test_the_member_matches_the_independent_derivation_quantity_by_quantity(
    k: float, m: float
) -> None:
    y0, y15 = base_pair()
    theirs = ref_member(y0, y15, YBAR_P, k, m)
    ours = next(x for x in base_estimate().members if (x.k, x.m) == (k, m))
    assert ours.fit.gas == pytest.approx(theirs["gas"], abs=1e-10)
    assert ours.g == pytest.approx(theirs["g"], abs=1e-10)
    assert ours.fit.s == pinned(theirs["s"])
    assert ours.fit.se == pytest.approx(theirs["se_fit"], rel=1e-5, abs=0)
    assert ours.grad15 == pytest.approx(theirs["grad15"], rel=1e-5, abs=0)
    assert ours.grad0 == pytest.approx(theirs["grad0"], rel=1e-5, abs=0)
    assert ours.se_ext == pytest.approx(theirs["se_ext"], rel=1e-5, abs=0)
    assert ours.se_0 == pytest.approx(theirs["se_0"], rel=1e-5, abs=0)
    assert ours.se == pytest.approx(theirs["se"], rel=1e-5, abs=0)


def test_the_extraction_error_carries_s15_over_root_six_and_s_k_inside_the_gradient_norm() -> None:
    for member in base_estimate().members:
        s15 = base_estimate().scale.s15
        assert member.se_ext == pinned(member.grad15 * math.sqrt(s15**2 / 6.0 + member.fit.s**2))
        assert member.se_0 == pinned(member.grad0 * member.fit.s)


def test_the_error_adds_the_fit_the_extraction_and_the_shared_tolerance_in_quadrature() -> None:
    for member in base_estimate().members:
        assert member.se == pinned(math.sqrt(member.fit.se**2 + member.se_ext**2 + member.se_0**2))
        assert member.se > max(member.fit.se, member.se_ext, member.se_0) * 0.999


def test_the_excess_is_prague_at_vp_less_the_gas_less_v1s_one_tenth() -> None:
    for member in base_estimate().members:
        assert member.g == pinned(YBAR_P - member.fit.gas - 0.1)
        assert member.fit.gas == pinned(
            member.k * R_PAPER * float(at550(RHO)) + member.fit.a * member.ozone.o_bar
        )


def test_the_envelope_is_the_lowest_and_highest_of_g_minus_and_plus_two_se_over_the_nine() -> None:
    estimate = base_estimate()
    lows = [m.g - 2.0 * m.se for m in estimate.members]
    highs = [m.g + 2.0 * m.se for m in estimate.members]
    assert len(lows) == 9
    assert estimate.g_lo == min(lows) and estimate.g_hi == max(highs)
    assert estimate.g_lo < min(m.g for m in estimate.members)
    assert estimate.g_hi > max(m.g for m in estimate.members)


def test_the_envelope_function_takes_the_minimum_and_maximum_over_all_members_by_hand() -> None:
    tool = load_tool()
    members = [
        SimpleNamespace(g=0.010, se=0.001),  # 0.008 .. 0.012
        SimpleNamespace(g=0.020, se=0.0005),  # 0.019 .. 0.021
        SimpleNamespace(g=0.005, se=0.004),  # -0.003 .. 0.013
        SimpleNamespace(g=0.030, se=0.0001),  # 0.0298 .. 0.0302  (the highest)
        SimpleNamespace(g=0.015, se=0.0),
    ]
    lo, hi = tool.aloft_envelope(members)
    assert lo == pytest.approx(-0.003, abs=1e-15)
    assert hi == pytest.approx(0.0302, abs=1e-15)
    lo, hi = tool.aloft_envelope(members[:2])
    assert (lo, hi) == (pytest.approx(0.008, abs=1e-15), pytest.approx(0.021, abs=1e-15))


def test_an_infinite_error_widens_the_envelope_without_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    monkeypatch.setattr(tool, "standard_error", lambda jac, q, s2: math.inf)
    estimate = tool.aloft_estimate(y0, y15, YBAR_P)
    assert estimate.g_lo == -math.inf and estimate.g_hi == math.inf
    verdict = tool.aloft_verdict(estimate.g_lo, estimate.g_hi, load_power().G_STARS)
    assert verdict.verdict == "spans"


def test_an_error_that_is_not_a_number_is_a_refusal_naming_the_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    monkeypatch.setattr(tool, "standard_error", lambda jac, q, s2: math.nan)
    with pytest.raises(tool.PragueError, match=re.escape("member (k=0.99, m=0)")):
        tool.aloft_estimate(y0, y15, YBAR_P)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_excess_is_a_refusal_naming_the_member(bad: float) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    with pytest.raises(tool.PragueError, match=re.escape("member (k=0.99, m=0)")):
        tool.aloft_estimate(y0, y15, bad)


@pytest.mark.parametrize("member", range(9))
def test_a_failed_main_fit_names_each_of_the_nine_members(
    monkeypatch: pytest.MonkeyPatch, member: int
) -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    k, m = MEMBERS[member]
    spy_solver(monkeypatch, tool, fail_calls=(member * SOLVES_PER_MEMBER + 1,))
    with pytest.raises(tool.PragueError, match=re.escape(f"member (k={k:g}, m={m:g})")):
        tool.aloft_estimate(y0, y15, YBAR_P)


@pytest.mark.parametrize("call", [2, 3, 21, 22, 23, 41, 42 + 5])
def test_a_failed_perturbed_fit_is_a_refusal_naming_the_member(
    monkeypatch: pytest.MonkeyPatch, call: int
) -> None:
    """Calls 2 to 21 are the 15 km differences, 22 to 41 the 0.34 km ones; 47 is the second
    member's fifth perturbed fit."""
    tool = load_tool()
    y0, y15 = base_pair()
    spy_solver(monkeypatch, tool, fail_calls=(call,))
    member = (call - 1) // SOLVES_PER_MEMBER
    k, m = MEMBERS[member]
    with pytest.raises(tool.PragueError, match=re.escape(f"member (k={k:g}, m={m:g})")) as error:
        tool.aloft_estimate(y0, y15, YBAR_P)
    assert "perturbed" in str(error.value)


# --- the 340 nm channel enters nothing ------------------------------------------------------------


def test_the_inputs_are_the_ten_channels_from_380_nm_up_and_leave_the_340_nm_channel_out() -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    plain = tool.aloft_inputs(aloft_dict(y0, y15))
    wild = tool.aloft_inputs(aloft_dict(y0, y15, y340=(3.0, 7.0)))
    for a, b in zip(plain, wild, strict=True):
        assert a.shape == (6, 10)
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(plain[0], y0, rtol=0, atol=2e-9)  # T is kept to nine figures
    np.testing.assert_allclose(plain[1], y15, rtol=0, atol=2e-9)


def test_perturbing_the_340_nm_channel_moves_neither_end_of_the_envelope() -> None:
    tool = load_tool()
    y0, y15 = base_pair()
    ends = []
    for y340 in ((0.45, 0.012), (0.9, 0.5), (0.05, 0.0001)):
        data = aloft_dict(y0, y15, y340=y340)
        a, b = tool.aloft_inputs(data)
        estimate = tool.aloft_estimate(a, b, YBAR_P)
        ends.append((estimate.g_lo, estimate.g_hi))
    assert ends[0] == ends[1] == ends[2]


# --- step 6: the verdict --------------------------------------------------------------------------

G_STAR = {-1.0: 0.0094444, 0.0: 0.0094830, 1.0: 0.0095275, 10.0: 0.0094923}
"""The four crossings in the order of increasing excess: -1, then +10 and 0, then +1."""


def test_the_outcome_at_one_sun_is_contributes_above_not_this_at_or_under_and_spans_between() -> (
    None
):
    outcome = load_tool().aloft_outcome
    assert outcome(0.0100, 0.0200, 0.0095) == "contributes"
    assert outcome(0.0001, 0.0090, 0.0095) == "not this"
    assert outcome(0.0050, 0.0200, 0.0095) == "spans"


def test_g_lo_exactly_at_the_crossing_does_not_contribute() -> None:
    outcome = load_tool().aloft_outcome
    assert outcome(0.0095, 0.0200, 0.0095) == "spans"  # g_lo > g* is strict
    assert outcome(math.nextafter(0.0095, 1.0), 0.0200, 0.0095) == "contributes"
    assert outcome(math.nextafter(0.0095, 0.0), 0.0200, 0.0095) == "spans"


def test_g_hi_exactly_at_the_crossing_is_not_this() -> None:
    outcome = load_tool().aloft_outcome
    assert outcome(0.0010, 0.0095, 0.0095) == "not this"  # g_hi <= g* is inclusive
    assert outcome(0.0010, math.nextafter(0.0095, 1.0), 0.0095) == "spans"
    assert outcome(0.0010, math.nextafter(0.0095, 0.0), 0.0095) == "not this"


def test_the_two_ends_at_the_crossing_together_are_not_this_not_contributes() -> None:
    assert load_tool().aloft_outcome(0.0095, 0.0095, 0.0095) == "not this"


def test_the_four_sun_lines_are_printed_in_the_order_minus_one_zero_plus_one_plus_ten() -> None:
    tool = load_tool()
    verdict = tool.aloft_verdict(0.0100, 0.0200, G_STAR)
    assert verdict.lines == (
        "sun -1: g* 0.0094444, contributes",
        "sun 0: g* 0.0094830, contributes",
        "sun +1: g* 0.0095275, contributes",
        "sun +10: g* 0.0094923, contributes",
        "verdict: contributes",
    )


def test_the_crossings_are_printed_to_five_significant_figures_from_the_unrounded_value() -> None:
    tool = load_tool()
    precise = {-1.0: 0.009444369707353337, 0.0: 0.009483031280641796,
               1.0: 0.009527494016917382, 10.0: 0.009492315430495191}  # fmt: skip
    lines = tool.aloft_verdict(0.0, 1.0, precise).lines
    assert lines[0] == "sun -1: g* 0.0094444, spans"
    assert lines[1] == "sun 0: g* 0.0094830, spans"
    assert lines[2] == "sun +1: g* 0.0095275, spans"
    assert lines[3] == "sun +10: g* 0.0094923, spans"


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.0100, 0.0200, "contributes"),
        (0.0001, 0.0090, "not this"),
        (0.0050, 0.0200, "spans"),
    ],
)
def test_the_four_suns_agree_and_the_verdict_is_their_common_outcome(
    lo: float, hi: float, expected: str
) -> None:
    verdict = load_tool().aloft_verdict(lo, hi, G_STAR)
    assert verdict.verdict == expected
    assert verdict.lines[-1] == f"verdict: {expected}"
    assert set(verdict.outcomes) == {expected}


def test_mixed_when_g_lo_lies_between_two_suns_crossings() -> None:
    tool = load_tool()
    # 0.00946 is above sun -1's crossing (0.0094444) and under sun 0's (0.0094830).
    verdict = tool.aloft_verdict(0.00946, 0.0200, G_STAR)
    assert verdict.outcomes == ("contributes", "spans", "spans", "spans")
    assert verdict.verdict == "mixed"
    assert verdict.lines[-1] == "verdict: mixed"
    # 0.00950 is above sun 0's, +10's and -1's but under +1's (0.0095275).
    verdict = tool.aloft_verdict(0.00950, 0.0200, G_STAR)
    assert verdict.outcomes == ("contributes", "contributes", "spans", "contributes")
    assert verdict.verdict == "mixed"


def test_mixed_when_g_hi_lies_between_two_suns_crossings() -> None:
    tool = load_tool()
    verdict = tool.aloft_verdict(0.0010, 0.00948, G_STAR)  # between sun -1's and sun 0's
    assert verdict.outcomes == ("spans", "not this", "not this", "not this")
    assert verdict.verdict == "mixed"
    verdict = tool.aloft_verdict(0.0010, 0.00950, G_STAR)  # between sun 0's and sun +1's
    assert verdict.outcomes == ("spans", "spans", "not this", "spans")
    assert verdict.verdict == "mixed"


def test_an_envelope_that_straddles_every_crossing_is_spans_not_mixed() -> None:
    verdict = load_tool().aloft_verdict(0.0050, 0.0120, G_STAR)
    assert verdict.verdict == "spans" and set(verdict.outcomes) == {"spans"}


def test_the_verdict_needs_the_four_suns_and_refuses_otherwise() -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="sun"):
        tool.aloft_verdict(0.0, 1.0, {-1.0: 0.0094, 0.0: 0.0095})


def test_the_outcomes_are_looked_up_by_sun_in_the_criterion_order() -> None:
    verdict = load_tool().aloft_verdict(0.00946, 0.0200, G_STAR)
    assert [verdict.outcome_at(s) for s in CRITERION_SUNS] == list(verdict.outcomes)


# --- the fixture builder, against stand-ins for the model -----------------------------------------


def full_axes(tool: ModuleType, **changes) -> object:
    recorded = json.loads((FIXTURE_DIR / "sun10.json").read_text())["provenance"]["dataset_axes"]
    axes = tool.Axes(
        tuple(recorded["visibilities_km"]),
        tuple(recorded["albedos"]),
        tuple(recorded["altitudes_m"]),
        tuple(recorded["elevations_deg"]),
        11,
        320.0,
        40.0,
    )
    return replace(axes, **changes)


def model_t(visibility: float, altitude_m: float) -> np.ndarray:
    """A smooth stand-in for Prague's transmittance: eleven channels, rising with the wavelength."""
    base = 0.30 + 0.45 * np.linspace(0.0, 1.0, 11) ** 0.5
    return np.exp(-(0.4 + 6.0 / visibility) * (1.0 - base) * (1.0 - 0.8 * altitude_m / 15000.0))


def aloft_stand_in(
    monkeypatch: pytest.MonkeyPatch,
    tool: ModuleType,
    *,
    axes: object | None = None,
    reported: float | None = None,
    reported_shift: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    """Stand-ins for everything `build_aloft_fixture` reaches for; returns the log of `evaluate`."""
    calls: list[dict] = []
    the_axes = axes if axes is not None else full_axes(tool)

    def evaluate(cache, dataset, *, visibility, altitude_m, queries, sun_elevation_deg, mode="sky"):
        calls.append(
            {
                "dataset": dataset,
                "visibility": visibility,
                "altitude_m": altitude_m,
                "queries": list(queries),
                "sun": sun_elevation_deg,
                "mode": mode,
            }
        )
        row = scale * model_t(visibility, altitude_m)
        said = altitude_m + reported_shift if reported is None else reported
        return tool.Evaluation(np.array([row]), 320.0, 40.0, np.array([said]))

    monkeypatch.setattr(tool, "fetch_model", lambda cache: cache)
    monkeypatch.setattr(tool, "verify_dataset", lambda path, dataset: None)
    monkeypatch.setattr(tool, "read_axes", lambda path: the_axes)
    monkeypatch.setattr(tool, "evaluate", evaluate)
    return calls


def test_the_builder_evaluates_each_node_at_each_altitude_the_way_the_criterion_says(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = aloft_stand_in(monkeypatch, tool)
    data = tool.build_aloft_fixture(tmp_path)
    assert len(calls) == 12
    assert [(c["altitude_m"], c["visibility"]) for c in calls] == [
        (altitude, node) for altitude in ALTITUDES for node in NODES
    ]
    for call in calls:
        assert call["mode"] == "transmittance"
        assert call["queries"] == [(0.0, 90.0)]
        assert call["sun"] == 10.0
        assert call["dataset"] == tool.DATASETS["full"]
    assert data["altitudes_m"] == list(ALTITUDES)
    assert data["visibility_nodes_km"] == list(NODES)
    assert data["channel_centres_nm"] == CENTRES
    assert data["format"] == "lastlight-prague-aloft" and data["version"] == 1


def test_the_builder_stores_t_to_nine_significant_figures_by_altitude_then_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    data = tool.build_aloft_fixture(tmp_path)
    table = data["transmittance"]
    assert len(table) == 2 and all(len(block) == 6 for block in table)
    assert all(len(row) == 11 for block in table for row in block)
    for a, altitude in enumerate(ALTITUDES):
        for n, node in enumerate(NODES):
            expected = [float(f"{t:.9g}") for t in model_t(node, altitude)]
            assert table[a][n] == expected


def test_the_builder_records_the_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    provenance = tool.build_aloft_fixture(tmp_path)["provenance"]
    assert provenance["model"]["commit"] == tool.MODEL_COMMIT
    assert provenance["dataset"]["sha256"] == tool.DATASETS["full"].sha256
    assert provenance["dataset"]["file"] == tool.DATASETS["full"].name
    assert provenance["dataset_axes"]["altitudes_m"][-1] == 15000.0
    assert provenance["dataset_axes"]["visibilities_km"] == list(NODES)
    assert provenance["command"] == "uv run python tools/prague_reference.py fixture --aloft"
    assert [(e["altitude_m"], e["visibility_km"]) for e in provenance["evaluations"]] == [
        (altitude, node) for altitude in ALTITUDES for node in NODES
    ]
    assert all(e["reported_altitude_m"] == e["altitude_m"] for e in provenance["evaluations"])


def test_the_builder_checks_the_dataset_before_it_evaluates_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    calls = aloft_stand_in(monkeypatch, tool)
    checked: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        tool, "verify_dataset", lambda path, dataset: checked.append((path, dataset))
    )
    tool.build_aloft_fixture(tmp_path)
    assert checked == [(tmp_path / "dl" / tool.DATASETS["full"].name, tool.DATASETS["full"])]
    assert calls


def test_the_builder_refuses_an_altitude_above_the_datasets_highest_before_evaluating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    axes = full_axes(tool, altitudes_m=(0.0, 340.0, 14000.0))
    calls = aloft_stand_in(monkeypatch, tool, axes=axes)
    with pytest.raises(tool.PragueError, match="15000"):
        tool.build_aloft_fixture(tmp_path)
    assert calls == []


def test_the_altitude_check_allows_one_part_in_a_million_and_no_more(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    within = full_axes(tool, altitudes_m=(0.0, 340.0, 15000.0 / (1.0 + 5e-7)))
    aloft_stand_in(monkeypatch, tool, axes=within)
    tool.build_aloft_fixture(tmp_path)
    beyond = full_axes(tool, altitudes_m=(0.0, 340.0, 15000.0 / (1.0 + 2e-6)))
    aloft_stand_in(monkeypatch, tool, axes=beyond)
    with pytest.raises(tool.PragueError, match="altitude"):
        tool.build_aloft_fixture(tmp_path)


@pytest.mark.parametrize(
    ("visibilities", "node"),
    [((21.0, 131.8), 20.0), ((20.0, 130.0), 131.8), ((20.0 * (1 + 2e-6), 131.8), 20.0)],
)
def test_the_builder_refuses_a_visibility_outside_the_datasets_range_before_evaluating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, visibilities: tuple, node: float
) -> None:
    tool = load_tool()
    axes = full_axes(tool, visibilities_km=visibilities)
    calls = aloft_stand_in(monkeypatch, tool, axes=axes)
    with pytest.raises(tool.PragueError, match=f"{node:g}"):
        tool.build_aloft_fixture(tmp_path)
    assert calls == []


def test_the_visibility_check_allows_one_part_in_a_million(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    axes = full_axes(tool, visibilities_km=(20.0 * (1 + 5e-7), 131.8 / (1 + 5e-7)))
    aloft_stand_in(monkeypatch, tool, axes=axes)
    tool.build_aloft_fixture(tmp_path)


def test_the_builder_refuses_a_reported_altitude_that_differs_from_the_one_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool, reported=290.0)  # the model's 50 m safety lift not undone
    with pytest.raises(tool.PragueError, match="290"):
        tool.build_aloft_fixture(tmp_path)


def test_the_builder_refuses_a_transmittance_that_is_not_finite_and_above_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    for bad in (0.0, -0.1, math.nan, math.inf):
        calls = aloft_stand_in(monkeypatch, tool)
        real = tool.evaluate

        def broken(*args, _real=real, _bad=bad, **kwargs):
            out = _real(*args, **kwargs)
            channels = np.array(out.channels)
            channels[0, 4] = _bad
            return tool.Evaluation(channels, out.start_nm, out.width_nm, out.altitudes_m)

        monkeypatch.setattr(tool, "evaluate", broken)
        with pytest.raises(tool.PragueError, match="transmittance"):
            tool.build_aloft_fixture(tmp_path)
        assert calls  # it did evaluate before it refused


def test_the_reported_altitude_is_checked_to_a_millionth_of_a_metre(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool, reported_shift=5e-7)
    data = tool.build_aloft_fixture(tmp_path)
    reported = [e["reported_altitude_m"] for e in data["provenance"]["evaluations"]]
    assert reported[0] == pytest.approx(340.0, abs=1e-6) and reported[-1] == pytest.approx(
        15000.0, abs=1e-6
    )
    assert reported[0] != 340.0  # what the model reported is what is recorded
    aloft_stand_in(monkeypatch, tool, reported_shift=2e-6)
    with pytest.raises(tool.PragueError, match="altitude"):
        tool.build_aloft_fixture(tmp_path)


def test_an_evaluation_that_does_not_say_its_altitude_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    monkeypatch.setattr(
        tool,
        "evaluate",
        lambda *a, **k: tool.Evaluation(np.array([model_t(20.0, 340.0)]), 320.0, 40.0),
    )
    with pytest.raises(tool.PragueError, match="altitude"):
        tool.build_aloft_fixture(tmp_path)


def test_evaluate_hands_back_the_altitudes_the_model_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = load_tool()
    binary = tool.binary_path(tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    stdout = "# available channels=11 start=320 width=40\n0 90 340 10 0 " + " ".join(["0.5"] * 11)
    seen: dict = {}

    def run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = tool.evaluate(
        tmp_path,
        tool.DATASETS["full"],
        visibility=20.0,
        altitude_m=340.0,
        queries=[(0.0, 90.0)],
        sun_elevation_deg=10.0,
        mode="transmittance",
    )
    assert list(result.altitudes_m) == [340.0]
    assert result.channels.shape == (1, 11)
    assert seen["command"][4] == repr(340.0 - tool.SAFETY_ALTITUDE_M)
    assert seen["command"][-1] == "transmittance"


def test_verify_dataset_refuses_a_missing_short_or_altered_file(tmp_path: Path) -> None:
    import hashlib

    tool = load_tool()
    payload = b"a stand-in dataset"
    dataset = tool.Dataset("d.dat", "id", len(payload), hashlib.sha256(payload).hexdigest())
    path = tmp_path / "d.dat"
    with pytest.raises(tool.PragueError, match="is missing"):
        tool.verify_dataset(path, dataset)
    path.write_bytes(payload[:-1])
    with pytest.raises(tool.PragueError, match="is missing"):
        tool.verify_dataset(path, dataset)
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(tool.PragueError, match="pinned SHA-256"):
        tool.verify_dataset(path, dataset)
    path.write_bytes(payload)
    tool.verify_dataset(path, dataset)


def test_the_command_writes_the_fixture_and_check_passes_against_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    target = tmp_path / "aloft.json"
    assert tool.main(["fixture", "--aloft", "--cache", str(tmp_path), "--output", str(target)]) == 0
    assert "wrote" in capsys.readouterr().out
    written = json.loads(target.read_text())
    assert written == tool.build_aloft_fixture(tmp_path)
    assert target.read_text() == tool.dump(written)
    assert tool.main(["fixture", "--aloft", "--check", "--cache", str(tmp_path),
                      "--output", str(target)]) == 0  # fmt: skip
    assert "0 difference(s)" in capsys.readouterr().out


def test_check_returns_1_on_any_difference_beyond_one_part_in_a_million(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    target = tmp_path / "aloft.json"
    assert tool.main(["fixture", "--aloft", "--cache", str(tmp_path), "--output", str(target)]) == 0
    capsys.readouterr()
    before = target.read_bytes()
    aloft_stand_in(monkeypatch, tool, scale=1.0 + 5e-8)  # inside the tolerance
    assert tool.main(["fixture", "--aloft", "--check", "--cache", str(tmp_path),
                      "--output", str(target)]) == 0  # fmt: skip
    capsys.readouterr()
    aloft_stand_in(monkeypatch, tool, scale=0.999)  # a difference
    assert tool.main(["fixture", "--aloft", "--check", "--cache", str(tmp_path),
                      "--output", str(target)]) == 1  # fmt: skip
    out = capsys.readouterr().out
    assert "difference(s)" in out and "0 difference(s)" not in out
    assert target.read_bytes() == before  # check wrote nothing


def test_check_of_a_fixture_that_does_not_exist_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    aloft_stand_in(monkeypatch, tool)
    status = tool.main(["fixture", "--aloft", "--check", "--cache", str(tmp_path),
                        "--output", str(tmp_path / "missing.json")])  # fmt: skip
    assert status == 1
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    ("kind", "needle"),
    [("altitude", "15000"), ("visibility", "131.8"), ("reported", "290")],
)
def test_each_builder_refusal_exits_1_with_the_reason_on_stderr_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    needle: str,
) -> None:
    tool = load_tool()
    if kind == "altitude":
        aloft_stand_in(monkeypatch, tool, axes=full_axes(tool, altitudes_m=(0.0, 14000.0)))
    elif kind == "visibility":
        aloft_stand_in(monkeypatch, tool, axes=full_axes(tool, visibilities_km=(20.0, 100.0)))
    else:
        aloft_stand_in(monkeypatch, tool, reported=290.0)
    target = tmp_path / "aloft.json"
    status = tool.main(["fixture", "--aloft", "--cache", str(tmp_path), "--output", str(target)])
    captured = capsys.readouterr()
    assert status == 1
    assert captured.err.startswith("error:") and needle in captured.err
    assert not target.exists()


def test_fixture_without_aloft_is_unchanged_it_still_picks_the_sun_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_tool()
    assert tool.fixture_spec(10.0).path == FIXTURE_DIR / "sun10.json"
    assert not tool.build_parser().parse_args(["fixture"]).aloft
    called: list[object] = []
    monkeypatch.setattr(tool, "cmd_fixture", lambda cache, spec, *, check: called.append(spec) or 0)
    monkeypatch.setattr(
        tool, "build_aloft_fixture", lambda cache: pytest.fail("not for plain fixture")
    )
    assert tool.main(["fixture", "--sun", "0"]) == 0
    assert called == [tool.SUN0]


# --- the command line -----------------------------------------------------------------------------


def test_the_aloft_flag_belongs_to_fixture_and_report_and_not_with_offset_or_depth() -> None:
    tool = load_tool()
    parser = tool.build_parser()
    assert parser.parse_args(["report", "--aloft"]).aloft is True
    assert parser.parse_args(["fixture", "--aloft"]).aloft is True
    assert parser.parse_args(["report"]).aloft is False
    for bad in (
        ["report", "--aloft", "--depth"],
        ["report", "--aloft", "--offset"],
        ["fetch", "--aloft"],
        ["build", "--aloft"],
    ):
        with pytest.raises(SystemExit):
            tool.main(bad)


def test_the_module_docstring_lists_the_two_new_commands() -> None:
    doc = load_tool().__doc__ or ""
    assert "fixture --aloft" in doc and "report --aloft" in doc


# --- number formatting and the fixture's layout --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.009483031280641796, "0.0094830"),
        (0.0094444, "0.0094444"),
        (-0.0012345678, "-0.0012346"),
        (0.12345, "0.12345"),
        (1.0, "1.0000"),
        (87.2, "87.200"),
        (14699.6, "14700"),
        (99999.4, "99999"),
        (123456.0, "1.2346e+05"),
        (9.99996, "10.000"),
        (0.00012345, "0.00012345"),
        (0.000012345, "1.2345e-05"),
        (0.0, "0"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (math.nan, "nan"),
    ],
)
def test_numbers_are_printed_to_five_significant_figures_trailing_zeros_kept(
    value: float, expected: str
) -> None:
    assert load_tool().format_significant(value) == expected


def test_the_digits_can_be_changed_and_the_default_is_five() -> None:
    tool = load_tool()
    assert tool.format_significant(0.0094830, 3) == "0.00948"
    assert tool.format_significant(0.0094830) == "0.0094830"


def test_the_aloft_fixture_must_be_the_registered_format_and_version() -> None:
    tool = load_tool()
    good = synthetic_aloft()
    tool.check_aloft_fixture(good)
    for key, value in (("format", "lastlight-prague-control"), ("version", 2), ("version", None)):
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(tool.PragueError, match=key):
            tool.check_aloft_fixture(bad)
    with pytest.raises(tool.PragueError, match="not an object"):
        tool.check_aloft_fixture([good])
    short = copy.deepcopy(good)
    short["transmittance"][0].pop()
    with pytest.raises(tool.PragueError, match="transmittance"):
        tool.check_aloft_fixture(short)
    short = copy.deepcopy(good)
    short["transmittance"][1][0].pop()
    with pytest.raises(tool.PragueError, match="transmittance"):
        tool.check_aloft_fixture(short)


def test_a_fixture_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    tool = load_tool()
    with pytest.raises(tool.PragueError, match="cannot be read"):
        tool.read_aloft_fixture(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(tool.PragueError, match="cannot be read"):
        tool.read_aloft_fixture(broken)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(synthetic_aloft()))
    assert tool.read_aloft_fixture(good) == synthetic_aloft()


# --- the report, on synthetic aloft spectra -------------------------------------------------------


def synthetic_aloft(cell: str = "base_s3", index: int = 1) -> dict:
    y0, y15 = synthetic(cell, index)
    return aloft_dict(y0, y15)


@cache
def _synthetic_report() -> str:
    return load_tool().report_aloft(four_fixtures(), synthetic_aloft())


def test_the_report_is_headed_by_the_issue_number() -> None:
    first = _synthetic_report().splitlines()[0]
    assert "#57" in first and first.startswith("### ")


def test_the_report_prints_the_132_transmittances_as_minus_log_t_to_five_decimals() -> None:
    text = _synthetic_report()
    data = synthetic_aloft()
    for altitude, block in zip(ALTITUDES, data["transmittance"], strict=True):
        for node, row in zip(NODES, block, strict=True):
            expected = " | ".join(f"{-math.log(t):.5f}" for t in row)
            assert expected in text, (altitude, node)


def test_the_report_prints_the_sun_lines_and_the_verdict_line_exactly_as_step_6_gives_them() -> (
    None
):
    text = _synthetic_report()
    lines = text.splitlines()
    pattern = re.compile(r"^sun (-1|0|\+1|\+10): g\* (\d\.\d+), (contributes|not this|spans)$")
    suns = [pattern.match(ln) for ln in lines if ln.startswith("sun ")]
    assert [m.group(1) for m in suns if m] == ["-1", "0", "+1", "+10"]
    assert len(suns) == 4 and all(suns)
    verdicts = [ln for ln in lines if ln.startswith("verdict: ")]
    assert len(verdicts) == 1
    assert re.fullmatch(r"verdict: (contributes|not this|spans|mixed)", verdicts[0])
    for m in suns:
        assert len(m.group(2).replace(".", "").lstrip("0")) == 5  # five significant figures
        assert float(m.group(2)) == pytest.approx(g_stars()[float(m.group(1))], rel=1e-4)
    assert (
        lines.index(verdicts[0])
        == max(lines.index(ln) for ln in lines if ln.startswith("sun ")) + 1
    )


def test_the_sun_lines_compare_the_envelope_with_the_unrounded_crossings_of_report_depth() -> None:
    tool = load_tool()
    text = _synthetic_report()
    y0, y15 = tool.aloft_inputs(synthetic_aloft())
    estimate = tool.aloft_estimate(y0, y15, tool.depth_pilot(four_fixtures()).y_bar_p)
    expected = tool.aloft_verdict(estimate.g_lo, estimate.g_hi, g_stars())
    assert list(expected.lines) == [
        ln for ln in text.splitlines() if ln.startswith(("sun ", "verdict: "))
    ]


def test_the_report_prints_every_member_and_g_lo_g_hi_and_g_seg() -> None:
    text = _synthetic_report()
    for k, m in MEMBERS:
        assert re.search(rf"^\| {k:g} \| {m:g} \|", text, re.M), (k, m)
    assert re.search(r"^g_lo = -?\d", text, re.M)
    assert re.search(r"^g_hi = -?\d", text, re.M)
    assert re.search(r"^g_seg = -?\d", text, re.M)
    assert "s_15" in text


def test_g_seg_is_the_excess_at_the_59_4_km_node() -> None:
    tool = load_tool()
    data = synthetic_aloft()
    y0, _ = tool.aloft_inputs(data)
    y_bar_p = tool.depth_pilot(four_fixtures()).y_bar_p
    expected = y_bar_p - float(at550(y0[3]))  # the 59.4 km node, 540 and 580 nm
    shown = re.search(r"^g_seg = (-?[\d.]+)", _synthetic_report(), re.M)
    assert shown is not None
    assert float(shown.group(1)) == pytest.approx(expected, rel=1e-4)


def test_the_report_runs_no_evaluation_and_writes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    monkeypatch.setattr(tool, "read_aloft_fixture", lambda *a, **k: synthetic_aloft())

    def refuse(*args, **kwargs):
        raise AssertionError("the aloft report must not evaluate Prague or write a file")

    for name in ("evaluate", "build_fixture", "build_aloft_fixture", "cmd_fetch", "cmd_build",
                 "matched_visibility_km"):  # fmt: skip
        monkeypatch.setattr(tool, name, refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(Path, "write_text", refuse)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    monkeypatch.chdir(tmp_path)
    before = {p.name: p.read_bytes() for p in FIXTURE_DIR.iterdir()}
    assert tool.main(["report", "--aloft"]) == 0
    capsys.readouterr()
    assert list(tmp_path.iterdir()) == []
    assert {p.name: p.read_bytes() for p in FIXTURE_DIR.iterdir()} == before


def test_the_command_line_writes_exactly_the_report_functions_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    monkeypatch.setattr(tool, "read_aloft_fixture", lambda *a, **k: synthetic_aloft())
    assert tool.main(["report", "--aloft"]) == 0
    written = capsys.readouterr().out
    assert written == _synthetic_report()
    assert written.endswith("\n") and not written.endswith("\n\n")


def test_the_report_does_not_edit_its_inputs() -> None:
    tool = load_tool()
    fixtures, aloft = four_fixtures(), synthetic_aloft()
    before = copy.deepcopy((fixtures, aloft))
    tool.report_aloft(fixtures, aloft)
    assert (fixtures, aloft) == before


def test_a_refusal_in_the_report_exits_1_with_the_member_on_stderr_and_nothing_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    monkeypatch.setattr(tool, "read_aloft_fixture", lambda *a, **k: synthetic_aloft())
    spy_solver(monkeypatch, tool, fail_calls=(1,))
    assert tool.main(["report", "--aloft"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:") and "member (k=0.99, m=0)" in captured.err


@pytest.mark.parametrize("call", [5, 30])
def test_a_failed_perturbed_fit_in_the_report_exits_1_with_the_member_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], call: int
) -> None:
    tool = load_tool()
    monkeypatch.setattr(tool, "read_aloft_fixture", lambda *a, **k: synthetic_aloft())
    spy_solver(monkeypatch, tool, fail_calls=(call,))
    assert tool.main(["report", "--aloft"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "member (k=0.99, m=0)" in captured.err and "perturbed" in captured.err


def test_the_report_refuses_a_malformed_aloft_fixture(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = load_tool()
    broken = synthetic_aloft()
    broken["transmittance"][1][2][4] = -0.5  # not above 0
    monkeypatch.setattr(tool, "read_aloft_fixture", lambda *a, **k: broken)
    assert tool.main(["report", "--aloft"]) == 1
    assert capsys.readouterr().err.startswith("error:")
    wrong_nodes = synthetic_aloft()
    wrong_nodes["visibility_nodes_km"] = [20.0, 27.6, 40.0, 59.4, 90.0, 120.0]
    with pytest.raises(tool.PragueError, match="visibility"):
        tool.report_aloft(four_fixtures(), wrong_nodes)
    wrong_altitudes = synthetic_aloft()
    wrong_altitudes["altitudes_m"] = [340.0, 14000.0]
    with pytest.raises(tool.PragueError, match="altitude"):
        tool.report_aloft(four_fixtures(), wrong_altitudes)
    wrong_centres = synthetic_aloft()
    wrong_centres["channel_centres_nm"] = [c + 1.0 for c in CENTRES]
    with pytest.raises(tool.PragueError, match="channel"):
        tool.report_aloft(four_fixtures(), wrong_centres)


def test_the_older_reports_are_still_reached_by_their_own_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = load_tool()
    assert tool.main(["report"]) == 0
    assert capsys.readouterr().out.startswith("### Sun +10, `aod_550` 0.03")
    assert tool.main(["report", "--depth"]) == 0
    assert "#55" in capsys.readouterr().out.splitlines()[0]


def test_the_criterion_file_is_the_one_the_tool_implements() -> None:
    text = CRITERION.read_text()
    assert (
        "Registered in the commit that lands this file on `main`, on 2026-10-17, for issue #57."
        in text
    )
    assert "p_15 = 121.11 hPa exactly" in text
    assert "`fixture --aloft`" in text and "`report --aloft`" in text
