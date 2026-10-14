# Renderer v1 against Prague at sun 0: the two decision rules (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-13, for issue #45. At that commit nobody has evaluated Prague at a sun elevation of 0, so no sun-0 comparison of the two models exists. The only Prague fixture is #34's at sun +10 (`tests/fixtures/prague/sun10.json`, `docs/log/2026-10-12-prague-control.md`). The builder implements both rules in `tools/prague_reference.py` exactly as written; a disagreement between this file and the tool is a defect in the tool.

**What was seen before this file landed, said plainly.** An independent cold read of the first draft of this file (on its pull request, before it landed) ran the rule B diagnostic below at sun +10, read-only, from `sun10.json` and v1. Its numbers, which the builder's tool must reproduce and report: the asymmetry that best fits the toward-sun excess is 0.60 at `aod_550` 0.1 (the root mean square excess r falls from 0.409 to 0.035) and 0.67 at 0.03 (where r at 0.7 is 0.122); 0.60 applied at 0.03 makes the excess worse (r 0.284). The first draft held out (sun +10, 0.03) after fitting at (sun +10, 0.1), so its verdict was already "neither" before any sun-0 number existed. Rule B was therefore revised after that look: it now fits one asymmetry per aerosol load at sun +10 and rules only on the sun-0 sky, which nobody has computed. The sun +10 finding stands as a pilot result in its own right: one Henyey-Greenstein asymmetry does not fit both aerosol loads. Rule A was not changed by the cold read, apart from the clarifications listed under it.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8): whatever these rules find, a change to v1's aerosol is a new renderer with its own registration.

## What is compared

- **Prague:** the same reference implementation, dataset, driver and primary reading as `sun10.json`: commit `2385c912e9051c1258013ff8c3ce2e19e10fb917`, the full dataset (SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`), observer altitude 0.34 km, ground albedo 0.2, visibility by Koschmieder from v1's ground extinction, the model's own interpolation in visibility, and linear interpolation between channel centres (`xyz`). The alternative readings `xyz_inverse_visibility` and `xyz_matched_aerosol_depth` are computed as the tool computes them for `sun10.json`, including the clamp: at `aod_550` 0.03 no visibility in Prague's range matches v1's depth, so the matched reading is the largest visibility, 131.8 km, as `sun10.json` records (`clamped_at_largest_visibility`). The clamped reading is kept; it is a real alternative setting of Prague's aerosol, not a match.
- **The only change is the sun:** elevation 0 degrees instead of +10. The new fixture is `tests/fixtures/prague/sun0.json` and records its sun elevation. The sun elevation, the azimuth list and the fixture path become per-fixture parameters whose defaults reproduce `sun10.json` byte for byte; `sun10.json` is not regenerated or edited, and `fixture --check` for it must still pass. The tool refuses a sun elevation outside the dataset's sun axis and any Prague luminance that is not finite and above 0, naming the point.
- **Points:** relative azimuths 0, 30, 90 and 180 degrees; the eight view elevations stored in the fixture, which are WHTM1's six nominal band elevations (`build_views()["nominal"]`, guarded to 1e-9 as in `sun10.json`) plus 30 and 60; `aod_550` 0.03 and 0.1. Azimuth 30 is new: it is where the port's 6.6 percent excess over the tool's Monte Carlo was measured (sun 0, view 1.51), and it is reported, never ruled on. Every selection of points (the rule set P, the six nominal views at azimuth 0) is made by azimuth and view value, never by position in an array.
- **v1:** `sky_xyz(Atmosphere(aod_550=a), 0.0, views, azimuth, engine="v1")` at the fixture's stored views, with the registered quadrature (the default), the table whose SHA-256 is `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368`, and every other `Atmosphere` field at its default. Y is luminance on v1's scale, as in `sun10.json`.
- **#34's own aod 0.1 rule** (the `predicted > max(spread, reading shift)` verdict in `report`) is not applied to `sun0.json`: the report prints rules A and B for sun 0 and nothing else that reads as a verdict.

## Rule A: which side of the `aod_550` 0.1 gap Prague is on

The question is #34's: at `aod_550` 0.1 the port's orders 2 and up sit 6.6 percent above the tool's Monte Carlo, and at 0.03 they agree. Does Prague side with the port or with the Monte Carlo?

**Rule set P:** relative azimuths 90 and 180, all eight views: 16 points.

**Statistic.** At each point i,

d_i = (Y_v1(0.1) / Y_Prague(0.1)) / (Y_v1(0.03) / Y_Prague(0.03)) - 1.

If Prague sides with the port, d_i = 0. If it sides with the Monte Carlo, d_i = e_i, where e_i is the existing `_predicted_if_monte_carlo` formula evaluated at sun 0: e_i = T_i / (T_i - M_i * 0.066 / 1.066) - 1, with T_i v1's total Y and M_i v1's multiple-scattering Y (`multiple_scattering_xyz`) at `aod_550` 0.1 and sun 0.

Let D = mean of d_i over P and E = mean of e_i over P. The uncertainty is U = max(s, m), where

- s is the standard deviation of d_i over P with ddof = 1: how much the two models' disagreement varies across the sky for reasons neither hypothesis predicts;
- m is the largest |D_r - D| over the two alternative readings r (`xyz_inverse_visibility`, `xyz_matched_aerosol_depth`), with D_r the mean of d_i over P when Prague's Y at both loads comes from reading r: how much the answer moves with the way Prague's aerosol is set.

**Verdict,** in this order:

1. If E <= 2U: **cannot tell** (the two predictions are not separated by twice the uncertainty).
2. Else if |D| <= U: **port**.
3. Else if |D - E| <= U: **Monte Carlo**.
4. Else: **cannot tell** (neither prediction lies within U of D).

When E > 2U, branches 2 and 3 cannot both hold. The report prints D, E, s, m, U, the verdict and the branch number. It also reports, without ruling, d and e over all 32 points, at each azimuth alone, and at the gap's own point (azimuth 30, view 1.51).

**What to expect, computed from v1 alone before any Prague number at sun 0.** E over P at sun 0 is 0.0445 (the cold read's figure, from v1 alone; the tool reproduces it). Branch 1 therefore fires unless U < 0.0223, and at sun +10 U over the same points was 0.025. "Cannot tell" from branch 1 is the likely outcome; it would say the comparison lacks power, not anything about either model, and the report says so in those words.

**Why P leaves out the azimuths toward the sun.** At sun +10 the azimuth-0 points carried a single-scattering excess that grows with aerosol (mean d 0.29 at azimuth 0, against 0.03 to 0.04 at 90 and 180), which is rule B's subject and swamps the multiple-scattering question. That was seen at sun +10, before any sun-0 number. Rejected: all points (#34's choice, which the aureole dominated); azimuth 0 corrected by rule B's fitted asymmetry (it would make rule A depend on rule B's fit).

**Why U is the larger of the two and the bar is 2U.** Both sources were measured at sun +10 over the same 16 points and both are the size of the effect (s 0.019, m 0.025, E 0.034), so either alone could fake or hide it. With E > 2U the midpoint between the predictions is at least U from each, so a D within U of one is more than U from the other. Rejected: #34's bar (E above each source separately), under which a D halfway between the predictions could still be called; a standard error s / sqrt(16) (the 16 points are not independent draws, they are one smooth sky).

## Rule B: what the toward-sun luminance excess comes from

At sun +10 v1's luminance is 1.19 times Prague's at `aod_550` 0.03 and 1.54 times at 0.1 at the lowest view toward the sun, where v1's multiply scattered share is 9.5 and 16 percent. The suspect is the aerosol's forward lobe: v1 fixes a Henyey-Greenstein asymmetry of 0.7.

**The diagnostic engine** (in the tool only, never in `lastlight.render`): XYZ_g = `sky_xyz(Atmosphere(aod_550=a, asymmetry=g), sun, views, azimuth, engine="v0")` (v0's single scattering, the registered quadrature) plus `multiple_scattering_xyz(Atmosphere(aod_550=a), sun, views, azimuth)` (v1's table, asymmetry baked at 0.7). v1 itself refuses an asymmetry other than 0.7, so it is never called with one. At g = 0.7, XYZ_g equals v1's XYZ exactly; a test pins that. Y_g is its Y.

**Grid.** g takes the 61 values k / 100 for integer k from 30 to 90. Comparisons with the bounds below are made on k (in integer hundredths), and g = 0.7 means k = 70.

**Excess.** For a condition c (a sun elevation and an `aod_550`), r_c(g) = the root mean square, over the six nominal views at relative azimuth 0, of ln(Y_g / Y_Prague) under the primary reading. It is unsigned. The signed mean of the same logarithms is reported beside it.

**Fit, at sun +10 (the pilot fixture).** For each load a in {0.03, 0.1}, g*(a) is the grid value that minimises r at (sun +10, a). If two grid values give exactly the same r, the one nearer k = 70 wins, and at equal distance the lower one.

**Ruled conditions: sun 0 only.** S_a = (sun 0, a) for a in {0.03, 0.1}. A condition counts only if r_{S_a}(0.7) >= 0.05: below that there is no excess at sun 0 worth attributing.

**Port bound.** At each of the six views of a condition, the most the port can add to ln(Y_v1 / Y_Prague) is b = ln(T / (T - M * 0.066 / 1.066)), with T and M v1's total and multiple-scattering Y there: 6.6 percent of the multiple scattering, the gap at `aod_550` 0.1, charged at 0.03 too, where the port and the Monte Carlo agree, so the bound is generous to the port. B_c is the root mean square of b over the six views. b is never negative: the port can only make v1 too bright.

**Verdict,** in this order:

1. If no S_a counts: **no excess at sun 0**.
2. Else **aerosol lobe** if, for every counted S_a, 50 <= k*(a) <= 80 and r_{S_a}(g*(a)) <= 0.5 * r_{S_a}(0.7).
3. Else **port** if, for every counted S_a, the signed mean of ln(Y_v1 / Y_Prague) is above 0 and B_{S_a} >= 0.5 * r_{S_a}(0.7).
4. Else **neither**.

The report prints, for both loads, g*(a) and r at (sun +10, a) for 0.7 and g*(a); for both sun-0 conditions, r(0.7), r(g*(a)), the signed means, B and whether it counts; the verdict; and the branch number. It also reports, without ruling:

- the sun +10 cross-load pilot (g*(0.1) applied at (sun +10, 0.03), and g*(0.03) applied at (sun +10, 0.1));
- the grid value that minimises r at each sun-0 condition alone;
- the root mean square of ln(Y_g / Y_Prague) over the eight views at azimuths 90 and 180 together, at each of the four conditions, for g*(a) and for 0.7 (whether fixing the lobe costs the side and back sky);
- #34's control at sun +10, `aod_550` 0.03 (lowest-band contrasts against the reference band, and the reference band's u'v' distance), recomputed with XYZ_{g*(0.03)} in place of v1 (whether a renderer with that asymmetry would still pass it).

**Why these constants.** The hypothesis is that v1's single-scattering lobe is too strong. If so, an asymmetry fitted to the excess at sun +10, where the lowest views toward the sun sit 6 to 8.5 degrees from the sun, must also remove most of the excess at sun 0, where they sit 1.5 to 4 degrees from it and the light crosses a far longer slant path: a different part of the lobe under a different extinction. One g per load is fitted because the pilot showed a single g cannot serve both loads; that dependence on load is itself evidence against a pure phase-function error, and the report says so whatever the sun-0 verdict. "At least half" asks the lobe to carry most of the excess, not all of it, because rule A's question (the multiple scattering) and Prague's different aerosol profile also move the level. 50 to 80 hundredths is the range of a single Henyey-Greenstein asymmetry for urban, continental and desert aerosol near 550 nm; outside it the lobe hypothesis as stated does not apply. The 0.05 floor (a 5 percent root mean square excess) is under half the smallest toward-sun excess at sun +10 (0.122 at 0.03), so it sets aside only a condition where the excess has all but vanished, where halving it could not be told from the 3 to 22 percent level offset v1 has against Prague away from the sun.

Rejected: the first draft's hold-out at (sun +10, 0.03) after a fit at (sun +10, 0.1), whose verdict the cold read showed was already "neither" (above); Prague's own asymmetry parameters as g (they are per particle type and wavelength inside its simulator, not in the published model, and #34 did not extract them); fitting g at sun 0 (nothing would be held out); a finer grid (a step of 0.01 already moves the phase function 4 to 7 percent at 1.5 to 12 degrees from the sun, and the verdict's bars are a factor of two and a range 0.30 wide, so a finer step buys nothing but a second look at a borderline); a "port" branch over the sun +10 conditions (there B is at most about 0.009 against a half-excess of about 0.2, so the branch could not fire).

## What is not decided here

- Whether the sun +10 control still passes with an asymmetry of g*(0.03), and what such a renderer would do to the camera magnitude check, are reported, never used: v1 is pre-registered.
- If rule B says "aerosol lobe", the conductor files the change as a new renderer registration. If rule A says "cannot tell", the report states what would tell: a reference with v1's own aerosol, from libRadtran, run offline (GPL, never vendored or imported).
