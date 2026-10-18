"""The verdict-level power check registered with issue #57, on synthetic truths only.

`docs/renderer/prague-gas-aloft-criterion.md` registers a rule for Prague's aerosol amount: the
paper's Rayleigh within a tolerance, Prague's own ozone shape from its zenith transmittance at
15 km, and the ozone column at 0.34 km fitted across the six visibility nodes. Its pilot must show,
before any six-node or high-altitude Prague number exists, that the rule's "contributes" becomes
more likely as the true excess grows. This script is that check. It draws synthetic truths for the
six nodes at 0.34 km and for the spectrum at 15 km, runs the registered estimator on them exactly
as the criterion defines it, and prints how often each verdict comes out at each true excess.

It reads no Prague data. Everything is synthetic, plus v1's pure functions
(`rayleigh_optical_depth`, `ozone_absorption_per_atm_cm`). Every cell is seeded from its name, so a
rerun prints the same tables.

Usage, from the repository root:

    uv run --frozen python tools/issue57_power.py [--n N] [--procs P] [--cells NAME,...]
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib
from dataclasses import dataclass, replace
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from lastlight.render.atmosphere import ozone_absorption_per_atm_cm, rayleigh_optical_depth

# --- the registered constants --------------------------------------------------------------------

LAM = np.arange(380.0, 741.0, 40.0)
"""The ten fitted channel centres, 380 to 740 nm (Prague's 340 nm channel is left out)."""
NCH = LAM.size
X = np.log(LAM / 550.0)
I540, I580 = 4, 5
W540, W580 = 0.75, 0.25
"""Linear interpolation to 550 nm between the 540 and 580 nm channel centres."""

P_034 = 1013.25 * (1.0 - 0.0065 * 340.0 / 288.15) ** 5.25588
P_15 = 120.45
R_PAPER = float(rayleigh_optical_depth(550.0, P_034))
RHO = np.asarray(rayleigh_optical_depth(LAM, P_034)) / R_PAPER
R15 = np.asarray(rayleigh_optical_depth(LAM, P_15))

V_NODES = np.array([20.0, 27.6, 40.0, 59.4, 90.0, 131.8])
NN = V_NODES.size
V_P = 49.9864309567
YBAR_P = 0.235751
"""Prague's -ln T at V_p, interpolated at 550 nm, from the committed fixtures (rounded)."""
G_STARS = {
    -1.0: 0.009444369707353337,
    0.0: 0.009483031280641796,
    1.0: 0.009527494016917382,
    10.0: 0.009492315430495191,
}
"""#55's crossing excess g* at each sun, unrounded, as `report --depth` computes it."""

RAYLEIGH_FACTORS = (0.99, 1.00, 1.01)
ALPHA_GUARD = (0.0, 3.0)
FD_STEP = 1e-5
SE_MULTIPLE = 2.0
FIT_SETTINGS = {"method": "trf", "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15, "max_nfev": 20000}
START_S, START_ALPHA, START_GAMMA, START_FLOOR = 1.0, 1.3, 0.0, 0.001

DATA_VALUES = NN * NCH
FIT_PARAMETERS = 1 + 3 * NN
ALOFT_DOF = DATA_VALUES - NCH - 3 * (NN - 1)
"""Degrees of freedom of the across-node quadratic at 15 km: 60 values, 10 + 15 parameters."""


def at550(v: np.ndarray) -> np.ndarray:
    """Linear at 550 nm between the 540 and 580 nm channel centres (last axis = channels)."""
    return W540 * v[..., I540] + W580 * v[..., I580]


RHO_BAR = float(at550(RHO))

# --- the registered estimator --------------------------------------------------------------------


def aloft_power_law(r380: float, r420: float) -> tuple[float, float, str]:
    """The aerosol aloft, b (lambda / 550)^-alpha, from the residuals at 380 and 420 nm.

    Exact through both points when both are positive and alpha lies in the guard [0, 3].
    Otherwise alpha is the two-point value clipped to the guard (each residual floored at 1e-12
    before the logarithm) and b is the least-squares amplitude on the two points, floored at 0.
    """
    f380, f420 = 380.0 / 550.0, 420.0 / 550.0
    if r380 > 0.0 and r420 > 0.0:
        a = float(np.log(r380 / r420) / np.log(420.0 / 380.0))
        if ALPHA_GUARD[0] <= a <= ALPHA_GUARD[1]:
            return float(r380 * f380**a), a, "exact"
    a_raw = np.log(max(r380, 1e-12) / max(r420, 1e-12)) / np.log(420.0 / 380.0)
    a = float(np.clip(a_raw, *ALPHA_GUARD))
    p380, p420 = f380 ** (-a), f420 ** (-a)
    b = (p380 * r380 + p420 * r420) / (p380**2 + p420**2)
    return float(max(b, 0.0)), a, "guard"


def ozone_aloft(y15_mean: np.ndarray, k: float) -> tuple[np.ndarray, str]:
    """Prague's ozone depth above 15 km per channel, from the six-node mean of -ln T at 15 km."""
    r = y15_mean - k * R15
    b, a, regime = aloft_power_law(float(r[0]), float(r[1]))
    o = r - b * (LAM / 550.0) ** (-a)
    o[:2] = 0.0
    return o, regime


def _aerosol(p: np.ndarray) -> np.ndarray:
    u, al, ga = p[1::3], p[2::3], p[3::3]
    return np.exp(u[:, None] - al[:, None] * X[None, :] + ga[:, None] * X[None, :] ** 2)


@dataclass
class Fit:
    gas: float
    se: float
    s: float
    ssr: float
    p: np.ndarray
    status: int


def sixnode_fit(y0: np.ndarray, o: np.ndarray, k: float, start: np.ndarray | None = None) -> Fit:
    """The six-node fit at 0.34 km with the Rayleigh fixed at k R_paper and the shape o."""
    target = y0 - k * R_PAPER * RHO[None, :]
    o_bar = float(at550(o))
    if start is None:
        g0 = k * R_PAPER * RHO_BAR + START_S * o_bar
        start = np.empty(FIT_PARAMETERS)
        start[0] = START_S
        start[1::3] = np.log(np.maximum(at550(y0) - g0, START_FLOOR))
        start[2::3] = START_ALPHA
        start[3::3] = START_GAMMA

    def resid(p: np.ndarray) -> np.ndarray:
        return (p[0] * o[None, :] + _aerosol(p) - target).ravel()

    def jac(p: np.ndarray) -> np.ndarray:
        a = _aerosol(p)
        j = np.zeros((DATA_VALUES, FIT_PARAMETERS))
        for i in range(NN):
            rows = slice(i * NCH, (i + 1) * NCH)
            j[rows, 0] = o
            j[rows, 1 + 3 * i] = a[i]
            j[rows, 2 + 3 * i] = -X * a[i]
            j[rows, 3 + 3 * i] = X**2 * a[i]
        return j

    res = least_squares(resid, start, jac=jac, **FIT_SETTINGS)
    ssr = float(res.fun @ res.fun)
    s2 = ssr / (DATA_VALUES - FIT_PARAMETERS)
    q = np.zeros(FIT_PARAMETERS)
    q[0] = o_bar
    try:
        var = float(q @ (s2 * np.linalg.inv(res.jac.T @ res.jac)) @ q)
        se = float(np.sqrt(var)) if np.isfinite(var) and var > 0.0 else np.inf
    except np.linalg.LinAlgError:
        se = np.inf
    gas = k * R_PAPER * RHO_BAR + float(res.x[0]) * o_bar
    return Fit(gas, se, float(np.sqrt(s2)), ssr, res.x, int(res.status))


def aloft_scale(y15: np.ndarray) -> float:
    """s_15: the residual scale at 15 km after each node's quadratic difference from the mean."""
    d = y15 - y15.mean(axis=0)
    a = np.stack([np.ones(NCH), X, X**2], axis=1)
    coef, *_ = np.linalg.lstsq(a, d.T, rcond=None)
    r = d.T - a @ coef
    return float(np.sqrt(float((r**2).sum()) / ALOFT_DOF))


@dataclass
class Estimate:
    """The registered quantities, by Rayleigh factor (in the order of RAYLEIGH_FACTORS)."""

    g: np.ndarray
    se_fit: np.ndarray
    grad15: np.ndarray  # |dG/d(mean 15 km spectrum)|, Euclidean over the channels
    grad0: np.ndarray  # |dG/d(0.34 km spectrum, each channel raised at all six nodes)|
    s_fit: np.ndarray
    s15: float
    regime: str
    status_ok: bool

    @property
    def se_ext(self) -> np.ndarray:
        """The extraction's noise and the tolerance for an error shared by the nodes aloft."""
        return self.grad15 * np.sqrt(self.s15**2 / NN + self.s_fit**2)

    @property
    def se_shared0(self) -> np.ndarray:
        """The tolerance for an error at 0.34 km shared by the six nodes."""
        return self.grad0 * self.s_fit

    @property
    def se_tot(self) -> np.ndarray:
        return np.sqrt(self.se_fit**2 + self.se_ext**2 + self.se_shared0**2)

    def envelope(self, design: str = "registered") -> tuple[float, float]:
        """g_lo and g_hi. Designs other than "registered" are the rejected forms of the error."""
        if design == "registered":
            se = self.se_tot
        elif design == "noise only":  # the extraction's noise averaged over the nodes, no tolerance
            se = np.sqrt(self.se_fit**2 + (self.grad15 * self.s15 / np.sqrt(NN)) ** 2)
        elif design == "fit only":  # the day-5 estimator's error: the six-node fit's SE alone
            se = self.se_fit
        else:
            raise ValueError(design)
        return float(np.min(self.g - SE_MULTIPLE * se)), float(np.max(self.g + SE_MULTIPLE * se))

    @property
    def g_lo(self) -> float:
        return self.envelope()[0]

    @property
    def g_hi(self) -> float:
        return self.envelope()[1]


def _gradient(y0: np.ndarray, y15_mean: np.ndarray, k: float, f: Fit, aloft: bool):
    """dG/dy by central differences, each perturbed fit starting at the unperturbed solution."""
    grad = np.empty(NCH)
    ok = True
    for c in range(NCH):
        step = np.zeros(NCH)
        step[c] = FD_STEP
        if aloft:
            gp = sixnode_fit(y0, ozone_aloft(y15_mean + step, k)[0], k, start=f.p)
            gm = sixnode_fit(y0, ozone_aloft(y15_mean - step, k)[0], k, start=f.p)
        else:
            o = ozone_aloft(y15_mean, k)[0]
            gp = sixnode_fit(y0 + step[None, :], o, k, start=f.p)
            gm = sixnode_fit(y0 - step[None, :], o, k, start=f.p)
        ok = ok and gp.status > 0 and gm.status > 0
        grad[c] = (gp.gas - gm.gas) / (2.0 * FD_STEP)
    return float(np.sqrt(grad @ grad)), ok


def estimate(y0: np.ndarray, y15: np.ndarray, ybar_p: float = YBAR_P) -> Estimate:
    """Steps 1 to 5 of the criterion, for each of the three Rayleigh factors."""
    y15_mean = y15.mean(axis=0)
    s15 = aloft_scale(y15)
    g, se_fit, grad15, grad0, s_fit = [], [], [], [], []
    ok = True
    regime = ""
    for k in RAYLEIGH_FACTORS:
        o, reg = ozone_aloft(y15_mean, k)
        if k == 1.00:
            regime = reg
        f = sixnode_fit(y0, o, k)
        n15, ok15 = _gradient(y0, y15_mean, k, f, aloft=True)
        n0, ok0 = _gradient(y0, y15_mean, k, f, aloft=False)
        ok = ok and f.status > 0 and ok15 and ok0
        g.append(ybar_p - f.gas - 0.1)
        se_fit.append(f.se)
        grad15.append(n15)
        grad0.append(n0)
        s_fit.append(f.s)
    return Estimate(
        g=np.array(g),
        se_fit=np.array(se_fit),
        grad15=np.array(grad15),
        grad0=np.array(grad0),
        s_fit=np.array(s_fit),
        s15=s15,
        regime=regime,
        status_ok=ok,
    )


def outcome(g_lo: float, g_hi: float, g_star: float) -> str:
    """At one sun: contributes if the whole envelope is above the crossing, not this if none is."""
    if g_lo > g_star:
        return "contributes"
    if g_hi <= g_star:
        return "not this"
    return "spans"


def verdict(g_lo: float, g_hi: float) -> str:
    """The verdict line: the four suns' common outcome, or "mixed" if they disagree."""
    each = {outcome(g_lo, g_hi, g) for g in G_STARS.values()}
    return each.pop() if len(each) == 1 else "mixed"


# --- synthetic truths ----------------------------------------------------------------------------

PERTURBATIONS = ("none", "bump", "tiltp", "tiltm", "shift")


@dataclass(frozen=True)
class Truth:
    sigma: float = 0.001
    g_true: float = 0.0095
    d_r: float | None = None  # None: uniform in [-0.01, 0.01]
    r15_rel: float = 0.0  # the 15 km Rayleigh's own departure from k R15, beyond d_r
    aloft: str = "same"  # same | amp20
    aloft_a_lo: float = 0.5
    aloft_a_hi: float = 1.5
    aloft_b_hi: float = 0.005
    top_common: bool = False  # the 15 km noise is one vector shared by the six nodes
    top_smooth: float = 0.0  # a smooth error at 15 km shared by the nodes: A x / max|x|
    ground_tilt: float = 0.0  # ozone shape at 0.34 km differs from aloft by this tilt per 100 nm
    f_lo: float = 0.80
    f_hi: float = 0.90


def true_ozone_shape(pert: str) -> np.ndarray:
    base = np.asarray(ozone_absorption_per_atm_cm(LAM))
    if pert == "none":
        k = base
    elif pert == "bump":
        k = base * (1.0 + 0.10 * np.exp(-0.5 * ((LAM - 600.0) / 40.0) ** 2))
    elif pert == "tiltp":
        k = base * (1.0 + 0.05 * (LAM - 550.0) / 100.0)
    elif pert == "tiltm":
        k = base * (1.0 - 0.05 * (LAM - 550.0) / 100.0)
    elif pert == "shift":
        k = np.asarray(ozone_absorption_per_atm_cm(LAM - 5.0))
    else:
        raise ValueError(pert)
    k = np.array(k, dtype=float)
    k[LAM <= 440.0] = 0.0
    return k


def draw(rng: np.random.Generator, t: Truth, pert: str) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic truth's -ln T at the six nodes, at 0.34 km and at 15 km, with noise."""
    d_r = rng.uniform(-0.01, 0.01) if t.d_r is None else t.d_r
    kap = true_ozone_shape(pert)
    kap_g = kap * (1.0 + t.ground_tilt * (LAM - 550.0) / 100.0)
    r_true = R_PAPER * (1.0 + d_r)
    gas = YBAR_P - 0.1 - t.g_true
    o_true = (gas - r_true * RHO_BAR) / float(at550(kap_g))
    f_top = rng.uniform(t.f_lo, t.f_hi)
    tau550 = 6.0 / V_NODES * rng.uniform(0.8, 1.25, NN)
    alpha = 1.6 - 0.8 * np.log(131.8 / V_NODES) / np.log(131.8 / 20.0)
    alpha = alpha + rng.uniform(-0.15, 0.15, NN)
    gamma = rng.uniform(-0.3, 0.3, NN)
    tau = tau550[:, None] * np.exp(-alpha[:, None] * X[None, :] + gamma[:, None] * X[None, :] ** 2)
    b = rng.uniform(0.0, t.aloft_b_hi)
    a = rng.uniform(t.aloft_a_lo, t.aloft_a_hi)
    amp = 1.0 + 0.2 * rng.uniform(-1.0, 1.0, NN) if t.aloft == "amp20" else np.ones(NN)
    aloft = (b * amp)[:, None] * (LAM[None, :] / 550.0) ** (-a)
    y0 = r_true * RHO[None, :] + o_true * kap_g[None, :] + tau
    r15 = R15 * (1.0 + d_r) * (1.0 + t.r15_rel)
    y15 = r15[None, :] + f_top * o_true * kap[None, :] + aloft
    y0 = y0 + rng.normal(0.0, t.sigma, (NN, NCH))
    if t.top_smooth:
        y15 = y15 + t.top_smooth * (X / np.max(np.abs(X)))[None, :]
    if t.top_common:
        y15 = y15 + rng.normal(0.0, t.sigma, NCH)[None, :]
    else:
        y15 = y15 + rng.normal(0.0, t.sigma, (NN, NCH))
    return y0, y15


# --- cells ---------------------------------------------------------------------------------------

GRID = (0.000, 0.004, 0.006, 0.008, 0.009, 0.0095, 0.010, 0.011, 0.0121, 0.013, 0.015, 0.018, 0.022)


def cells() -> dict[str, Truth]:
    out: dict[str, Truth] = {}
    for sigma, tag in ((0.0003, "s3"), (0.001, "s10")):
        base = Truth(sigma=sigma)
        out[f"base_{tag}"] = base
        out[f"dr_m2_{tag}"] = replace(base, d_r=-0.02)
        out[f"dr_p2_{tag}"] = replace(base, d_r=0.02)
        out[f"topcommon_{tag}"] = replace(base, top_common=True)
        out[f"asteep_{tag}"] = replace(base, aloft_a_lo=1.5, aloft_a_hi=2.5)
        out[f"aflat_{tag}"] = replace(base, aloft_a_lo=0.0, aloft_a_hi=0.5)
        out[f"amp20_{tag}"] = replace(base, aloft="amp20")
        out[f"bigaloft_{tag}"] = replace(base, aloft_b_hi=0.02)
        out[f"tiltp_{tag}"] = replace(base, ground_tilt=0.03)
        out[f"tiltm_{tag}"] = replace(base, ground_tilt=-0.03)
        out[f"r15_{tag}"] = replace(base, r15_rel=0.005)
        out[f"smooth_{tag}"] = replace(base, top_smooth=0.001)
    out["base_s1"] = Truth(sigma=0.0001)
    out["base_s20"] = Truth(sigma=0.002)
    return out


def run_cell(task: tuple[str, Truth, float, int]) -> list[dict]:
    name, truth, g_true, n = task
    t = replace(truth, g_true=g_true)
    rng = np.random.default_rng(zlib.crc32(f"{name}|{g_true:.4f}".encode()))
    rows = []
    for trial in range(n):
        y0, y15 = draw(rng, t, PERTURBATIONS[trial % len(PERTURBATIONS)])
        e = estimate(y0, y15)
        row = {"cell": name, "g_true": g_true}
        for design, tag in DESIGNS:
            lo, hi = e.envelope(design)
            row[f"verdict{tag}"] = verdict(lo, hi) if e.status_ok else "refused"
            row[f"covered{tag}"] = lo <= g_true <= hi
            row[f"g_lo{tag}"] = lo
            row[f"g_hi{tag}"] = hi
        row.update(
            {
                "g_mid": float(e.g[1]),
                "se_fit": float(e.se_fit[1]),
                "se_ext": float(e.se_ext[1]),
                "se_shared0": float(e.se_shared0[1]),
                "grad15": float(e.grad15[1]),
                "grad0": float(e.grad0[1]),
                "s_fit": float(e.s_fit[1]),
                "s15": e.s15,
                "regime": e.regime,
            }
        )
        rows.append(row)
    return rows


DESIGNS = (("registered", ""), ("noise only", "_noise"), ("fit only", "_fit"))
"""The registered error and the two rejected forms, run on the same trials."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=400, help="trials per (cell, true excess)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cells", default="", help="comma-separated cell names (default: all)")
    ap.add_argument("--csv", default="", help="also write every trial to this CSV file")
    args = ap.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")  # inherited by the spawned workers before numpy loads
    every = cells()
    names = [c for c in args.cells.split(",") if c] or list(every)
    tasks = [(name, every[name], g, args.n) for name in names for g in GRID]
    with Pool(args.procs) as pool:
        rows = [r for chunk in pool.imap(run_cell, tasks) for r in chunk]
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(headline(rows, names))
    print(designs(rows, [n for n in names if n.startswith("base_")]))
    print(summarise(rows, names))
    return 0


def designs(rows: list[dict], names: list[str]) -> str:
    """P(contributes) / P(not this) / coverage under each form of the error, on the same trials."""
    if not names:
        return ""
    lines = [
        "\nThe registered error against the rejected forms (P(contributes) / P(not this) / "
        "coverage):",
        "| cell | design | " + " | ".join(f"{g:.4f}" for g in GRID) + " |",
        "|---|---|" + "---|" * len(GRID),
    ]
    for name in names:
        for design, tag in DESIGNS:
            cells = []
            for g in GRID:
                sel = [r for r in rows if r["cell"] == name and r["g_true"] == g]
                if not sel:
                    cells.append("")
                    continue
                pc = sum(r[f"verdict{tag}"] == "contributes" for r in sel) / len(sel)
                pn = sum(r[f"verdict{tag}"] == "not this" for r in sel) / len(sel)
                cv = sum(r[f"covered{tag}"] for r in sel) / len(sel)
                cells.append(f"{pc:.2f}/{pn:.2f}/{cv:.2f}")
            lines.append(f"| {name} | {design} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _crossing(gs: list[float], ps: list[float], rising: bool) -> float:
    """The true excess where P crosses 0.5, by linear interpolation on the grid (nan if never)."""
    for i in range(len(gs) - 1):
        g0, g1, p0, p1 = gs[i], gs[i + 1], ps[i], ps[i + 1]
        if (p0 < 0.5 <= p1) if rising else (p0 >= 0.5 > p1):
            return g0 + (0.5 - p0) * (g1 - g0) / (p1 - p0)
    return float("nan")


def headline(rows: list[dict], names: list[str]) -> str:
    """One line per cell: 50 percent points, power at the paper's excess, monotonicity, errors."""
    lines = [
        "| cell | c50 | n50 | P(C) at 0.0121 | P(N) at 0.0121 | worst step in P(C), in SE | "
        "worst step in P(N), in SE | false C (g <= 0.008) | false N (g >= 0.011) | "
        "lowest coverage | refused |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        gs, pc, pn, cov, n_ = [], [], [], [], []
        refused = 0
        for g in GRID:
            sel = [r for r in rows if r["cell"] == name and r["g_true"] == g]
            if not sel:
                continue
            gs.append(g)
            n_.append(len(sel))
            pc.append(sum(r["verdict"] == "contributes" for r in sel) / len(sel))
            pn.append(sum(r["verdict"] == "not this" for r in sel) / len(sel))
            cov.append(sum(r["covered"] for r in sel) / len(sel))
            refused += sum(r["verdict"] == "refused" for r in sel)

        def worst(ps: list[float], rising: bool, n_: list[int] = n_) -> float:
            out = 0.0
            for i in range(len(ps) - 1):
                step = (ps[i + 1] - ps[i]) if rising else (ps[i] - ps[i + 1])
                var = ps[i] * (1 - ps[i]) / n_[i] + ps[i + 1] * (1 - ps[i + 1]) / n_[i + 1]
                se = np.sqrt(max(var, 1e-12))
                out = min(out, step / se)
            return out

        at = gs.index(0.0121)
        lines.append(
            f"| {name} | {_crossing(gs, pc, True):.4f} | {_crossing(gs, pn, False):.4f} | "
            f"{pc[at]:.3f} | {pn[at]:.3f} | {worst(pc, True):.2f} | {worst(pn, False):.2f} | "
            f"{max(p for g, p in zip(gs, pc, strict=True) if g <= 0.008):.3f} | "
            f"{max(p for g, p in zip(gs, pn, strict=True) if g >= 0.011):.3f} | "
            f"{min(cov):.3f} | {refused} |"
        )
    return "\n".join(lines)


def summarise(rows: list[dict], names: list[str]) -> str:
    lines = []
    for name in names:
        lines.append(f"\n{name}")
        lines.append(
            "| g_true | n | P(contributes) | P(spans) | P(not this) | P(mixed) | refused | "
            "covered | median width | median SE_fit | median SE_ext |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for g in GRID:
            sel = [r for r in rows if r["cell"] == name and r["g_true"] == g]
            n = len(sel)
            if not n:
                continue

            def share(v: str, sel: list[dict] = sel) -> float:
                return sum(r["verdict"] == v for r in sel) / len(sel)

            width = float(np.median([r["g_hi"] - r["g_lo"] for r in sel]))
            lines.append(
                f"| {g:.4f} | {n} | {share('contributes'):.3f} | {share('spans'):.3f} | "
                f"{share('not this'):.3f} | {share('mixed'):.3f} | {share('refused'):.3f} | "
                f"{sum(r['covered'] for r in sel) / n:.3f} | {width:.4f} | "
                f"{float(np.median([r['se_fit'] for r in sel])):.5f} | "
                f"{float(np.median([r['se_ext'] for r in sel])):.5f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
