# Renderer v1 against Prague at sun 0: the two decision rules (pre-registered)

Registered in the commit that adds this file, on 2026-10-13, for issue #45. At that commit neither model has been evaluated against the other at a sun elevation of 0: the only Prague comparison is #34's at sun +10 (`tests/fixtures/prague/sun10.json`, `docs/log/2026-10-12-prague-control.md`), and the lobe diagnostic below has never been run at any sun elevation. Both rules are fixed here, with every constant, before either is computed. The builder implements them in `tools/prague_reference.py` exactly as written; a disagreement between this file and the tool is a defect in the tool.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8): whatever these rules find, a change to v1's aerosol is a new renderer with its own registration.

## What is compared

- **Prague:** the same reference implementation, dataset, driver and primary reading as `sun10.json`: commit `2385c912e9051c1258013ff8c3ce2e19e10fb917`, the full dataset (SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`), observer altitude 0.34 km, ground albedo 0.2, visibility by Koschmieder from v1's ground extinction, the model's own interpolation in visibility, and linear interpolation between channel centres (`xyz`). The alternative readings `xyz_inverse_visibility` and `xyz_matched_aerosol_depth` are computed as the tool computes them for `sun10.json`.
- **The only change is the sun:** elevation 0 degrees instead of +10. The new fixture is `tests/fixtures/prague/sun0.json`; `sun10.json` is not regenerated or edited.
- **Points:** relative azimuths 0, 30, 90 and 180 degrees; view elevations WHTM1's six nominal band elevations (`build_views()["nominal"]`) plus 30 and 60; `aod_550` 0.03 and 0.1. Azimuth 30 is new: it is where the port's 6.6 percent excess over the tool's Monte Carlo was measured (sun 0, view 1.51), and it is reported, never ruled on.
- **v1:** `sky_xyz(Atmosphere(aod_550=a), 0.0, views, azimuth, engine="v1")` with the registered quadrature (the default), the table whose SHA-256 is `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368`, and every other `Atmosphere` field at its default. Y is luminance on v1's scale, as in `sun10.json`.

## Rule A: which side of the `aod_550` 0.1 gap Prague is on

The question is #34's: at `aod_550` 0.1 the port's orders 2 and up sit 6.6 percent above the tool's Monte Carlo, and at 0.03 they agree. Does Prague side with the port or with the Monte Carlo?

**Rule set P:** relative azimuths 90 and 180, all eight views: 16 points.

**Statistic.** At each point i,

d_i = (Y_v1(0.1) / Y_Prague(0.1)) / (Y_v1(0.03) / Y_Prague(0.03)) - 1.

If Prague sides with the port, d_i = 0. If it sides with the Monte Carlo, d_i = e_i, where e_i is the existing `_predicted_if_monte_carlo` formula evaluated at sun 0: e_i = T_i / (T_i - M_i * 0.066 / 1.066) - 1, with T_i v1's total Y and M_i v1's multiple-scattering Y (`multiple_scattering_xyz`) at `aod_550` 0.1.

Let D = mean of d_i over P and E = mean of e_i over P. The uncertainty is U = max(s, m), where

- s is the standard deviation of d_i over P (ddof = 1): how much the two models' disagreement varies across the sky for reasons neither hypothesis predicts;
- m is the largest |D_r - D| over the two alternative readings r (`xyz_inverse_visibility`, `xyz_matched_aerosol_depth`), with D_r the mean of d_i over P under reading r: how much the answer moves with the way Prague's aerosol is set.

**Verdict,** in this order:

1. If E <= 2U: **cannot tell** (the two predictions are not separated by twice the uncertainty).
2. Else if |D| <= U: **port**.
3. Else if |D - E| <= U: **Monte Carlo**.
4. Else: **cannot tell** (neither prediction lies within U of D).

The report prints D, E, s, m, U, the verdict and the branch that produced it. It also reports, without ruling, d and e over all 32 points, at each azimuth alone, and at the gap's own point (azimuth 30, view 1.51).

**Why P leaves out the azimuths toward the sun.** At sun +10 the azimuth-0 points carried a single-scattering excess that grows with aerosol (mean d 0.29 at azimuth 0, against 0.03 to 0.04 at 90 and 180), which is rule B's subject and swamps the multiple-scattering question. That was seen at sun +10, before any sun 0 number. Rejected: all points (#34's choice, which the aureole dominated); azimuth 0 corrected by rule B's fitted asymmetry (it would make rule A depend on rule B's fit).

**Why U is the larger of the two and the bar is 2U.** Both sources were measured at sun +10 and both are of the same size as the effect (s 0.019, m 0.025, E 0.034 over the same 16 points), so either alone could fake or hide it. With E > 2U the midpoint between the predictions is at least U from each, so a D inside U of one is outside U of the other. Rejected: #34's bar (E above each source separately), under which a D halfway between the predictions could still be called; a standard error s / sqrt(16) (the 16 points are not independent draws, they are one smooth sky).

## Rule B: what the toward-sun luminance excess comes from

At sun +10 v1's luminance is 1.19 times Prague's at `aod_550` 0.03 and 1.54 times at 0.1 at the lowest view toward the sun, where v1's multiply scattered share is 9 and 16 percent. The suspect is the aerosol's forward lobe: v1 fixes a Henyey-Greenstein asymmetry of 0.7.

**The diagnostic engine** (in the tool only, never in `lastlight.render`): Y_g = the Y of v0's single scattering (`lastlight.render.transfer`, registered quadrature) with `Atmosphere(aod_550=a, asymmetry=g)`, plus v1's multiple-scattering Y at `aod_550` a from the committed table (whose asymmetry is baked at 0.7). At g = 0.7 it equals v1 exactly; a test pins that.

**Excess.** For a condition c (a sun elevation and an `aod_550`), r_c(g) = the root mean square, over the six nominal views at relative azimuth 0, of ln(Y_g / Y_Prague) under the primary reading.

**Fit.** g* is the value on the grid 0.30, 0.31, ..., 0.90 (61 values) that minimises r_F, where F is (sun +10, `aod_550` 0.1), the largest excess, from `sun10.json`. A tie goes to the value nearest 0.7.

**Held out.** H = (sun +10, `aod_550` 0.03), (sun 0, 0.03) and (sun 0, 0.1).

**Port bound.** At each of those views and conditions, the most the port can add to ln(Y_v1 / Y_Prague) is b = ln(T / (T - M * 0.066 / 1.066)), with T and M v1's total and multiple-scattering Y at that point: 6.6 percent of the multiple scattering, the gap at `aod_550` 0.1, charged at 0.03 too, where the port and the Monte Carlo agree, so the bound is generous to the port. R_c is the root mean square of b over the six views.

**Verdict,** in this order:

1. **Aerosol lobe** if 0.50 <= g* <= 0.80 and, for every c in H, r_c(g*) <= 0.5 * r_c(0.7).
2. Else **port** if, for every c in F and H, R_c >= 0.5 * r_c(0.7).
3. Else **neither**.

The report prints g*, r_c(0.7) and r_c(g*) for every condition, R_c, the verdict and its branch. It also reports, without ruling: the g that minimises r in each held-out condition alone; the root mean square of ln(Y_g / Y_Prague) at azimuths 90 and 180 for g* and for 0.7 (whether fixing the lobe costs the side and back sky); and #34's contrast and u'v' control at sun +10 recomputed with g* (whether a renderer with g* would still pass it).

**Why these constants.** The fit uses the one condition where the excess is largest and already measured, so the three held-out conditions, two of them at a sun no one has compared at, test whether one asymmetry transfers across aerosol load and sun elevation: an explanation that only works where it was fitted is not one. "At least half" asks the lobe to carry most of the excess, not all of it, because rule A's question (the multiple scattering) and Prague's different aerosol profile also move the level. 0.50 to 0.80 is the range of a single Henyey-Greenstein asymmetry for urban, continental and desert aerosol near 550 nm; a g* outside it would mean a phase-function shape, not the lobe's strength, is wrong, which the lobe hypothesis as stated does not cover. Rejected: Prague's own asymmetry parameters as g (they are per particle type and wavelength inside its simulator, not in the published model, and #34 did not extract them); fitting g on all conditions at once (nothing would be held out); a finer grid (a step of 0.01 already moves the phase function 4 to 7 percent at 1.5 to 12 degrees from the sun, and the verdict's bars are a factor of two and a range 0.30 wide, so a finer step buys nothing but a second look at a borderline).

## What is not decided here

- Whether the sun +10 control still passes with g*, and what a renderer with g* would do to the camera magnitude check, are reported, never used: v1 is pre-registered.
- If rule B says "aerosol lobe", the conductor files the change as a new renderer registration. If rule A says "cannot tell", the report states what would tell (a reference with v1's own aerosol: libRadtran, run offline, GPL, never vendored or imported).
