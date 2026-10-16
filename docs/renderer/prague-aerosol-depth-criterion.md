# Prague's aerosol amount, from Prague's own gas depths: no test (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-16, for issue #55.

At that commit Prague's zenith transmittance has been recorded at two visibilities only, 123.8187 and 49.9864 km, both between its nodes (the same eleven numbers in all four committed fixtures). #49's fixture builds also evaluated it inside the matched-depth search. That search starts at Prague's clearest and haziest nodes (131.8 and 20 km), and two of its results were recorded: the 0.1 load's match at 55.5320 km (depth 0.0999986), and the 0.03 load's clamped match at 131.8 km (depth 0.036851), both under the old estimate. Nobody has fitted Prague's transmittance at its six visibility nodes, and nothing under this file does. The builder implements in `tools/prague_reference.py` exactly what this file says the tool prints; a disagreement between this file and the tool is a defect in the tool.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

**The question.** #49 registered the aerosol amount as "no test" (rule 4 of `docs/renderer/prague-offset-criterion.md`). At the 0.1 load's visibility (Koschmieder, 49.9864 km) Prague's aerosol is deeper than v1's 0.1, and moving Prague to the visibility where the two match removes part of v1's offset over Prague: Δ_M is 0.0401, 0.0521, 0.0639 and 0.0375 at sun -1, 0, +1 and +10. The match rests on an estimate of Prague's aerosol depth, its zenith transmittance minus its gas. The tool subtracts v1's gas, not Prague's, so the excess it finds (0.0137) carries an error never derived, put at about ±0.01. #55 proposed estimating Prague's gas from its own transmittance fitted across its six visibility nodes, "where the aerosol varies and the gas does not", deriving the estimate's error, and ruling.

**What this file registers.** That route has no test of the amount. A rule was drafted for it. Its pilot, and two cold reads of the draft on this file's pull request (none running Prague), showed three things:

- the six nodes add no information about the gas beyond repeating the same spectral fit;
- the drafted rule's verdict was set by its design, not by the sky;
- the ozone cross-section used as a stand-in can by itself move single estimates by several times the margin.

This file says so, with the evidence, instead of registering a rule whose outcome is already fixed. No six-node fit is run under it. The tool prints the evidence that data in hand can give (below). A route that may have a test is filed as #57.

## The route that was drafted

The definitions below are kept because the evidence and the tool's printed pilot refer to them.

**Data.** y(c, V) = -ln T(c, V), where T is Prague's zenith transmittance in channel c at visibility V, computed as the tool computes `zenith_transmittance` for the fixtures: `evaluate` in transmittance mode, the zenith query, observer altitude 0.34 km. Transmittance does not depend on the sun, and the four fixtures hold the same spectra. The fit uses the ten channels centred at 380, 420, ..., 740 nm. The 340 nm channel is left out: SPCTRL2's ozone table starts at 380 nm, while the paper's ozone absorbs strongly in the Huggins band across 320 to 360 nm.

**Model.** ŷ(c, V_i) = a_R ρ_c + a_O κ_c + exp(u_i - α_i x_c + γ_i x_c²), with x_c = ln(λ_c / 550 nm), where:

- ρ_c is v1's Rayleigh depth at λ_c over its depth at 550 nm (`rayleigh_optical_depth`);
- κ_c is v1's ozone coefficient at λ_c per atm cm (`ozone_absorption_per_atm_cm`, the SPCTRL2 table);
- a_R (the Rayleigh depth at 550 nm) and a_O (the ozone column in atm cm) are shared by every visibility fitted;
- u_i, α_i and γ_i are the aerosol at visibility i: exp(u_i) is its depth at 550 nm, α_i its Ångström exponent and γ_i its curvature in log wavelength.

**The eight estimates.** Three assumptions, each either fitted or fixed:

- the aerosol's form: a power law (every γ_i = 0) or curved (every γ_i free);
- the ozone column: free, or the paper's U.S. Standard column, a_O = 0.345 atm cm (the tool's `US_STANDARD_OZONE_DU`);
- the Rayleigh depth: free, or the paper's, a_R = R_paper = `rayleigh_optical_depth(550, p)`, with p = 1013.25 (1 - 0.0065 × 340 / 288.15)^5.25588 computed unrounded (973.0664 hPa), the U.S. Standard pressure at 0.34 km.

They are numbered with the form varying slowest:

| estimate | aerosol form | ozone | Rayleigh |
| --- | --- | --- | --- |
| 1 | power law | free | free |
| 2 | power law | free | paper |
| 3 | power law | paper | free |
| 4 | power law | paper | paper |
| 5 | curved | free | free |
| 6 | curved | free | paper |
| 7 | curved | paper | free |
| 8 | curved | paper | paper |

**The fit.** Unweighted least squares of ŷ - y, with `scipy.optimize.least_squares`: method `trf`, the analytic Jacobian, `xtol`, `ftol` and `gtol` all 1e-15, at most 20000 evaluations, and every other argument at its default. The start is a_R = R_paper and a_O = 0.345, and at each visibility u_i = ln max(ȳ(V_i) - G_paper, 0.001), α_i = 1.3 and γ_i = 0. A fit with `status` ≤ 0 is a refusal: the tool prints which estimate failed to standard error and exits 1.

**Gas, error, excess.**

- **The gas.** G_k = a_R ρ̄ + a_O κ̄ is estimate k's gas depth at 550 nm. ρ̄ and κ̄ are ρ and κ interpolated linearly at 550 nm between the 540 and 580 nm channel centres, the interpolation the tool already applies to y: 1.00968 and 0.086141 per atm cm. G_paper, estimate 4's gas, is 0.12367.
- **Its error.** SE_k = sqrt(qᵀ C q), where:
  - C = s² (JᵀJ)⁻¹, with J the Jacobian at the solution;
  - s² is the sum of squared residuals over (number of values minus number of fitted parameters);
  - q is the derivative of G with respect to the parameters: ρ̄ for a_R, κ̄ for a_O, and 0 for every aerosol parameter.

  SE is 0 when no gas parameter is fitted (estimates 4 and 8). It is infinite if `numpy.linalg.inv` raises, or if qᵀ C q is not a positive finite number.
- **Residuals and conditioning.** The residual rms is sqrt(sum of squared residuals / number of values). The condition number is the 2-norm condition number of the unscaled J.
- **The excess.** ȳ(V) is y interpolated linearly at 550 nm between channel centres. V_p = 49.9864309567 km and V_0.03 = 123.8187372 km are the two loads' visibilities. g_k = ȳ(V_p) - G_k - 0.1 is Prague's aerosol depth at V_p minus v1's, under estimate k.

**The drafted rule** (not registered). With six nodes, the envelope g_lo = min over k of (g_k - 2 SE_k) and g_hi = max over k of (g_k + 2 SE_k) would have been carried into Prague as matched skies at the excesses g_lo and g_hi. The amount's part Δ_M at those ends would have been ruled against #49's bar 2m at the 0.1 load at sun -1, 0, +1 and +10, on #49's ladder plus a fourth per-condition outcome, "spans", for a bar inside the amount's range.

## What was seen before this file landed, said plainly

Everything in this section is computed from data in hand: the two transmittance spectra in the committed fixtures, the fixtures' skies through #49's `offset_condition`, and v1. The fixtures hold two visibilities, not six, so each of the eight estimates is fitted to those two (20 values) with the model, start and settings above. The tool prints the numbers in this section's table and in the bullets marked "printed" under a heading containing the word "pilot", each within half a unit in its last digit. The synthetic checks and the cold reads' figures quoted further down are not printed by the tool.

The pilot's eight estimates:

| estimate | G | 2SE | g | g - 2SE | g + 2SE | residual rms | a_R | ozone (DU) | condition number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1149 | 0.0032 | 0.0208 | 0.0176 | 0.0241 | 0.00096 | 0.0859 | 327.4 | 882 |
| 2 | 0.1233 | 0.0022 | 0.0124 | 0.0102 | 0.0147 | 0.00182 | 0.0930 | 341.0 | 17.5 |
| 3 | 0.1171 | 0.0030 | 0.0186 | 0.0156 | 0.0216 | 0.00114 | 0.0866 | 345.0 | 912 |
| 4 | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00182 | 0.0930 | 345.0 | 14.1 |
| 5 | 0.1144 | 0.0280 | 0.0214 | -0.0066 | 0.0494 | 0.00081 | 0.0853 | 327.9 | 1.470e4 |
| 6 | 0.1214 | 0.0013 | 0.0144 | 0.0131 | 0.0156 | 0.00083 | 0.0930 | 318.5 | 87.2 |
| 7 | 0.0990 | 0.0601 | 0.0367 | -0.0234 | 0.0969 | 0.00092 | 0.0686 | 345.0 | 1.033e4 |
| 8 | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00117 | 0.0930 | 345.0 | 76.9 |

What it shows:

- **The old estimate is not consistent with itself** (printed). It subtracts v1's gas at 550 nm from a total interpolated between the 540 and 580 nm channel centres. The same gas interpolated the same way is 0.12367, not 0.12208. Of the difference, 0.0013 is the interpolation (neither λ⁻⁴ nor the Chappuis band is straight between 540 and 580 nm) and 0.0003 is the pressure (v1's 970 hPa against 973.07). The paper's gas puts the excess at 0.0121, not 0.0137.
- **The power law's low Rayleigh comes from the form.** Under a power law the fit wants a Rayleigh depth 8 percent under the paper's (0.0859 against 0.0930), and fixing the paper's doubles the residual. Under the curved form the paper's Rayleigh fits as well as a free one (0.00083 against 0.00081).
- **Two spectra cannot separate the gas from a curved aerosol** when the Rayleigh is free: estimates 5 and 7 have 2SE 0.0280 and 0.0601.
- **The crossing** (printed). Scaled linearly in g from #49's matched reading (Δ_M at the old excess 0.01367), the amount's part equals the bar at g* = 0.01367 b / Δ_M: 0.0094, 0.0095, 0.0095 and 0.0095 at sun -1, 0, +1 and +10. #49's Δ_M is 1.435 to 1.447 times the bar at all four suns, so the four conditions move together. A condition would carry only if the excess reached 0.0541, 0.0351, 0.0263 or 0.0242. At the paper's excess, Δ_M scaled linearly is 1.268 to 1.279 times the bar.
- **The clearest node** (printed). From the recorded clamped match, ȳ(131.8) = 0.036851 + 0.12208 = 0.15893, so g_max = ȳ(V_p) - ȳ(131.8) = 0.0768. A match at the 0.03 load would need G ≥ ȳ(131.8) - 0.03 = 0.1289. No estimate comes near it.
- The 340 nm channel, if included, would move estimate 1's G from 0.1149 to 0.1285 and triple its residual (0.0010 to 0.0030).

Earlier looks at the same question, disclosed: #34's "about ±0.01" (from v1's gas, never derived); #49's first cold read, ±0.003 to ±0.005; #49's third cold read, a gas fit to one visibility's eleven channels (estimator error 0.003 to 0.006, fit rms 0.0034; method not recorded).

## Why the route has no test

**1. The six nodes add no information about the gas beyond repeating the fit.** With each visibility's u_i, α_i and γ_i free, the columns of J that belong to visibility i are A_i, x A_i and x² A_i (A_i its aerosol depth by channel). Their span does not depend on u_i. So the precision of a_R and a_O does not depend on how much aerosol each node holds, only on each node's spectral shape. More nodes of the same shape only repeat the information: three copies of each pilot spectrum (six nodes) divide the SE of every estimate that fits a gas parameter by exactly √3 at the same s.

#55's mechanism, gas and aerosol separating because the amount varies and the gas does not, needs a constraint that ties the nodes' aerosol shapes together; one shared shape is the simplest. Prague's do not: the paper's Fig. 4 labels 131.8 km continental clean, 59.4 km continental average and 27.6 km continental polluted, with the others interpolated or extrapolated between them. A family that imposed one shape, or one curvature, across the nodes would impose what Prague's documentation says is not there. A family without it has only the spectral slopes, which is #49's problem again.

The tool prints (under "pilot"):

- for every estimate that fits a gas parameter (all but 4 and 8), its 2SE recomputed from J at the pilot solution with every node's amount multiplied by 0.5, and by 2, at the pilot's s. Each equals the pilot's 2SE to within 1e-9 relative;
- for the same six estimates, the 2SE with each pilot spectrum's fitted aerosol taken three times (six nodes) at the pilot's s. Each, divided by the pilot's 2SE, is 1/√3 = 0.5774.

**2. The drafted rule's verdict was set by its design.**

- **Three of its outcomes could not fire.**
  - Estimates 4 and 8 fit no gas, so their excess, 0.0121, lies in every envelope. That gives g_lo ≤ 0.0121 and g_hi ≥ 0.0121.
  - Inside Prague's visibility segment from 40 to 59.4 km, which holds V_p and every visibility of excess 0 to about 0.023, Prague interpolates its radiance coefficients and √T linearly in visibility (the reference implementation's source). So Δ_M is close to linear in g there, and the scaled figures above hold.
  - "Not this" needed Δ_M at g_hi to be at most the bar. Δ_M(0.0121) is 1.27 times it, so it was dead. With it went the rule's "cannot tell", which needs a "not this" condition.
  - "Carries" needed Δ_M at g_lo to be at least L / 2. Even Δ_M(0.01367) is under L / 2 at every sun, so it was dead too.
  - Estimate 8 duplicates estimate 4 by construction. The upper end mattered only to the dead branches.
- **"Contributes" needed every estimate's lower end above about 0.0095.** Estimate 4 sits only 0.0026 above that. A synthetic six-node check by the author (six power-law aerosols, Ångström exponents from 0.8 to 1.6, residual rms 0.001, linearised at the truth) puts estimate 5's 2SE at 0.010 to 0.014. Cold read 2, linearising at the pilot's fitted shapes, puts it at 0.0155. So estimate 5 alone gives "spans" unless it lands 0.004 to 0.009 below the paper's gas, and the curved form gives no reason to expect it to land low.
- **"Contributes" was not more likely as the true excess grew.** Cold read 2 generated six-node data from the model's own families, 100 trials per cell. It found P(contributes) of 0.00 to 0.06 when the truth's Rayleigh is the paper's or up to 5 percent under it, 0.23 at 7 percent under with power-law aerosol (0 with curved), and 0 at 10 percent under, where the true excess is largest (0.0215). As the truth's Rayleigh falls, estimate 2 (paper Rayleigh, free ozone) is misspecified and its lower end falls from 0.0116 to 0.0083 while the true excess rises. Where it had any power, the power came from the residual scale (P of 0.33 to 0.67 at rms 0.0003, for power-law and two-mode aerosols only), that is from how smoothly Prague stores its transmittance, not from the aerosol amount.

**3. The ozone stand-in is neither small nor common to the estimates.** SPCTRL2's table stands in for the paper's Gorshelev cross-sections at 223 K. Cold read 2 checked plausible departures of the true ozone shape from SPCTRL2's on noise-free data: 10 percent more at 600 nm, a 5 percent tilt, or the band 5 nm redder. These moved single estimates' G by -0.016 to +0.007, up to six times the 0.0026 margin, while their residuals stayed at the pilot's level. No fit statistic would flag it. An earlier draft listed the ozone shape among the things common to all estimates that no envelope could see. That was wrong: it moves the estimates differently, and on its own it could set the verdict.

**Rescues considered, each rejected or a different test.**

- **Power laws only (estimates 1 to 4).** The pilot's envelope would be 0.0102 to 0.0241, and the pilot's verdict "contributes" by 0.0007. But the power law's low Rayleigh belongs to the form, not to the data, and when the truth's Rayleigh is low the binding estimate's lower end falls as the excess grows (problem 2).
- **One shape, or one curvature, shared across the nodes.** It would restore #55's mechanism by asserting what the paper says is not so (problem 1).
- **A gate that drops poorly fitting estimates.** The verdict would then turn on Prague's residual scale, which is what set the power (problem 2). The ozone stand-in's bias leaves the residuals unchanged (problem 3).
- **A physical tolerance on the Rayleigh depth instead of free or fixed** (the paper's air is Bodhaine's N2 and O2 for U.S. Standard air, known to about 1 percent). That is a different test: the paper's Rayleigh with Prague's own ozone. It is only a test if Prague's own ozone shape replaces SPCTRL2's (problem 3), and Prague can give that from its transmittance aloft. That is #57.

## What the tool prints

Under a heading containing "#55" and "no test", with a "pilot" heading for its numbers:

- the eight-row pilot table above;
- the bullets marked "printed": G_paper and the old G with the split, the crossings, the carrying excesses, the two ratios, ȳ(131.8), g_max and the 0.03 load's G threshold;
- the two invariance checks under problem 1.

It runs no Prague evaluation, writes no fixture, and reads only the four committed fixtures and v1. The four fixtures are not regenerated or edited, and `fixture --check` for each must still pass. The report command's output is the report function's output.

## What would tell

- **#57, the paper's Rayleigh with Prague's own ozone:**
  - Rayleigh from the paper's air within a stated tolerance;
  - Prague's own ozone shape from its zenith transmittance at its highest observer altitude, with any aerosol aloft bounded from the channels where ozone does not absorb;
  - the ozone column at 0.34 km fitted across the six nodes with the Rayleigh fixed (estimate 6's analogue: condition number 87 and 2SE 0.0013 on two spectra);
  - a rule whose "contributes" grows more likely with the true excess, checked at the verdict level in its pilot before it is registered.
- The paper's own gas, computed per channel from its cross-sections (a table not in hand).
- A reference with v1's own atmosphere (libRadtran, run offline; GPL, never vendored or imported).

## What is not decided here

- Whether the amount contributes: no rule is registered. #49's "no test" for the aerosol stands, now with this file's evidence beside rule 4's.
- Which model is right: both are models.
- The depths compared, if a successor rules, are Prague's above its 0.34 km observer and v1's whole column above its ground observer, as in #34 and #49: the air above the two observers matches (#49, rule 2).
- A change to v1 for any of this is a new renderer with its own registration (#50), never a change to v1.

## Decisions, each with its rejected alternative

- **The six-node route is registered as "no test", with its evidence.** Rejected:
  - landing the drafted envelope rule. Its verdict would be "spans" whatever the sky, nearly, and its "contributes" would not become more likely as the excess grew;
  - restoring its power by a smaller factor than 2 or by dropping estimates. Either would be chosen with the pilot and cold read 2's power table in view;
  - imposing a shared aerosol shape across the nodes (problem 1).
- **No six-node fit is run, even as a diagnostic.** Its numbers are what #57 needs, and computing them first would make them #57's pilot. Rejected: running it because the transmittance is cheap to evaluate.
- **The tool prints the pilot, the crossings and the invariance; cold read 2's verdict-level and ozone-shape synthetics are quoted, not rebuilt.** The truths and noise in those synthetics were the reader's choices, and pinning them in the tool would register a reader's design as the project's. Both reads are posted on this file's pull request. Rejected: committing them as tests.
- **The 340 nm channel is left out of every fit, and the fit is unweighted in -ln T.** These are kept from the draft as definitions of the printed pilot. Rejected: weights from the transmittance's error, which nobody has per channel.
- **#57 is filed, not registered here.** Its pilot must include a verdict-level power check, the lesson of this file, and that is a design of its own. Rejected: rewriting this file into #57 on the day its draft failed, with the failure's numbers in view.
