# Prague's aerosol amount, from Prague's own gas depths (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-16, for issue #55. At that commit Prague's zenith transmittance has been recorded at two visibilities only, 123.8187 and 49.9864 km, both between its nodes (the same eleven numbers in all four committed fixtures). #49's fixture builds also evaluated it inside the matched-depth search, which starts at Prague's clearest and haziest nodes (131.8 and 20 km). Only one of those evaluations was recorded: the aerosol depth at 131.8 km under the old estimate, 0.036851, as the 0.03 load's clamped match. Nobody has fitted Prague's transmittance at its six visibility nodes, or evaluated Prague's sky at a visibility matched under any estimate in this file. The builder implements this file in `tools/prague_reference.py` exactly as written; a disagreement between this file and the tool is a defect in the tool.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

**The question.** #49 registered the aerosol amount as "no test" (rule 4 of `docs/renderer/prague-offset-criterion.md`). At the 0.1 load's visibility (Koschmieder, 49.9864 km) Prague's aerosol is deeper than v1's 0.1, and moving Prague to the visibility where the two match removes part of v1's offset over Prague: Δ_M, 0.0401, 0.0521, 0.0639 and 0.0375 at sun -1, 0, +1 and +10. The match rests on an estimate of Prague's aerosol depth, its zenith transmittance minus its gas. The tool subtracts v1's gas, not Prague's, so the excess it finds (0.0137) carries an error that was never derived and was put at about ±0.01. This file estimates Prague's gas from its own transmittance at its six visibility nodes, derives the estimate's error, and rules on whether the amount contributes to the offset.

**The answer this can give, in one line.** The amount contributes if every one of the eight gas estimates below, after twice its own fitting error, leaves Prague's aerosol at the 0.1 load's visibility more than about 0.0095 deeper than v1's (the excess at which the amount's part equals the bar). Otherwise the derived error still spans the bar. The two other outcomes, "carries" and "not this", cannot fire on data already in hand (see the pilot).

## Definitions

### Prague's gas, estimated eight ways

**Data.** y(c, V) = -ln T(c, V), where T is Prague's zenith transmittance in channel c at visibility V, computed exactly as the tool computes `zenith_transmittance` for the fixtures: `evaluate` in transmittance mode, the zenith query, observer altitude 0.34 km. The fit uses the ten channels centred at 380, 420, ..., 740 nm and Prague's six visibility nodes, 20, 27.6, 40, 59.4, 90 and 131.8 km, as read from the dataset's header: 60 values.

**Model.** ŷ(c, V_i) = a_R ρ_c + a_O κ_c + exp(u_i - α_i x_c + γ_i x_c²), with x_c = ln(λ_c / 550 nm), where:

- ρ_c is v1's Rayleigh depth at λ_c over its depth at 550 nm (`rayleigh_optical_depth`, which is proportional to pressure, so any one pressure gives the same shape);
- κ_c is v1's ozone coefficient at λ_c per atm cm (`ozone_absorption_per_atm_cm`, the SPCTRL2 table);
- a_R (the Rayleigh depth at 550 nm) and a_O (the ozone column in atm cm) are shared by all six nodes;
- u_i, α_i and γ_i are node i's aerosol: exp(u_i) is its depth at 550 nm, α_i its Ångström exponent, γ_i its curvature in log wavelength.

**The eight estimates.** Three assumptions, each either fitted or fixed:

- the aerosol's form: a power law (every γ_i = 0) or curved (every γ_i free);
- the ozone column: free, or the paper's U.S. Standard column, a_O = 0.345 atm cm (the tool's `US_STANDARD_OZONE_DU`);
- the Rayleigh depth: free, or the paper's, a_R = R_paper = `rayleigh_optical_depth(550, p)` at p = 1013.25 (1 - 0.0065 × 340 / 288.15)^5.25588 = 973.07 hPa, the U.S. Standard pressure at 0.34 km.

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

Estimate 1 is Prague's own gas, the issue's estimate. Estimates 4 and 8 are the paper's gas: they fit no gas parameter.

**The fit.** Unweighted least squares of ŷ - y over the 60 values, with `scipy.optimize.least_squares` (method `trf`), the analytic Jacobian, `xtol`, `ftol` and `gtol` all 1e-15, and at most 20000 evaluations. The start is a_R = R_paper, a_O = 0.345, and at each node u_i = ln max(ȳ(V_i) - G_paper, 0.001), α_i = 1.3 and γ_i = 0. A fit whose `status` is not positive is a refusal (below).

**G and its error.** G_k = a_R ρ̄ + a_O κ̄ is estimate k's gas depth at 550 nm. The bar is the interpolation the tool already applies to y: linear at 550 nm between the 540 and 580 nm channel centres, so ρ̄ = 1.00968 and κ̄ = 0.086141 per atm cm. G_paper is G for estimate 4, 0.12367.

SE_k = sqrt(qᵀ C q), where:

- C = s² (JᵀJ)⁻¹, with J the Jacobian at the solution;
- s² is the sum of squared residuals over (60 minus the number of fitted parameters);
- q is the derivative of G with respect to the parameters: ρ̄ for a_R, κ̄ for a_O, and 0 for every aerosol parameter.

SE is 0 for estimates 4 and 8. If JᵀJ cannot be inverted, SE is infinite.

**The excess.** ȳ(V) is y interpolated linearly at 550 nm between the channel centres, over all eleven channels, as the tool does. V_p = 49.9864309567 km is the 0.1 load's visibility; T(V_p) is the 0.1 `zenith_transmittance` of the committed fixtures. Then g_k = ȳ(V_p) - G_k - 0.1: Prague's aerosol depth at V_p minus v1's, under estimate k.

**The envelope.** g_lo = the smallest of g_k - 2 SE_k, and g_hi = the largest of g_k + 2 SE_k, over the eight estimates.

### The amount's part of the offset

For an excess g, V_g is the visibility where ȳ(V) = ȳ(V_p) - g. That is Prague's air with g less aerosol depth than at V_p, whatever its gas, because the gas cancels in the difference. The tool finds V_g as `matched_visibility_km` does: 16 bisections over [20, 131.8] km, with the model's own interpolation in visibility.

Δ_M(g), at a sun s and the 0.1 load, is the mean over P of ln(Y_Prague(V_g) / Y_Prague(V_p)). That is L - L_matched with the match at V_g. Both are Prague's primary reading, at the stored views and azimuths of the sun's fixture, observer 0.34 km, albedo 0.2. P is #49's 16 points: relative azimuths 90 and 180, all eight views, selected by value. Δ_M(0) = 0, and #49's Δ_M is Δ_M at the old estimate's excess, 0.01367.

g_max = ȳ(V_p) - ȳ(131.8) is the largest excess Prague's clearest node can remove.

**The ends of the amount's range.**

- Δ_lo is 0 if g_lo ≤ 0, with no Prague run. Otherwise it is Δ_M(min(g_lo, g_max)). When g_lo > g_max the clamped value is below the unclamped one, so the clamp can only understate.
- Δ_hi is Δ_M(g_hi) if 0 < g_hi ≤ g_max, 0 if g_hi ≤ 0, and unbounded if g_hi > g_max.

Between the ends Δ_M is taken to increase with g, the premise #49's linear scaling also used. The report prints Δ_M at the excesses of estimates 1 and 4, which lie inside the envelope, so a departure would show.

### Conditions, bar and verdicts

**Conditions:** (-1, 0.1), (0, 0.1), (+1, 0.1) and (+10, 0.1). The offset L and the reading noise m at each come from the committed fixtures, exactly as #49 computes them (`offset_condition`). **Bar:** b = 2m.

**The per-condition verdict:**

1. **carries** if Δ_lo ≥ L / 2 and Δ_lo > b;
2. else **contributes** if Δ_lo > b;
3. else **not this** if Δ_hi ≤ b;
4. else **spans**: the bar lies inside the amount's range.

**The rule's verdict over its conditions,** in this order:

1. If every condition carries: **carries**.
2. Else if every condition carries or contributes: **contributes**.
3. Else if every condition is not this: **not this**.
4. Else if any condition spans: **the derived error still spans the bar**.
5. Else: **cannot tell**, because the conditions disagree.

For every condition the report prints Δ_lo, Δ_hi, L, b, the share Δ_lo / L and the condition's verdict. It then prints the rule's verdict and its branch number (1 to 5, above). Every comparison uses unrounded values computed by the tool; the four-digit values in this file are what it must reproduce, never thresholds it reads.

**Refusal.** The report is not written, and the tool prints the reason to standard error and exits 1, if any of the eight fits ends with a status that is not positive, or if the targets stored in the new fixture differ from the ones the tool recomputes from the fixture's transmittance. No estimate and no condition is ever dropped.

## What was seen before this file landed, said plainly

Everything in this section is computed from data in hand: the two transmittance spectra in the committed fixtures (at V_p and at the 0.03 load's 123.8187 km), the fixtures' skies and v1. The fixtures hold two visibilities, not six, so the pilot fits each of the eight estimates to those two (20 values), with the model, start and settings above. It is a **pilot, not a test**: no ruled number uses it. The report prints every number in this section under a heading containing the word "pilot", and each must agree with the value shown here to within half a unit in its last digit.

The pilot's eight estimates:

| estimate | G | 2SE | g | g - 2SE | g + 2SE | residual rms | a_R | ozone (DU) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1149 | 0.0032 | 0.0208 | 0.0176 | 0.0241 | 0.00096 | 0.0859 | 327.4 |
| 2 | 0.1233 | 0.0022 | 0.0124 | 0.0102 | 0.0147 | 0.00182 | 0.0930 | 341.0 |
| 3 | 0.1171 | 0.0030 | 0.0186 | 0.0156 | 0.0216 | 0.00114 | 0.0866 | 345.0 |
| 4 | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00182 | 0.0930 | 345.0 |
| 5 | 0.1144 | 0.0280 | 0.0214 | -0.0066 | 0.0494 | 0.00081 | 0.0853 | 327.9 |
| 6 | 0.1214 | 0.0013 | 0.0144 | 0.0131 | 0.0156 | 0.00083 | 0.0930 | 318.5 |
| 7 | 0.0990 | 0.0601 | 0.0367 | -0.0234 | 0.0969 | 0.00092 | 0.0686 | 345.0 |
| 8 | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00117 | 0.0930 | 345.0 |

What it shows:

- **The old estimate is not consistent with itself.** It subtracts v1's gas at 550 nm from a total interpolated between the 540 and 580 nm channel centres. The same gas interpolated the same way is 0.12367, not 0.12208. Of the difference, 0.0013 is the interpolation (neither λ⁻⁴ nor the Chappuis band is straight between 540 and 580 nm) and 0.0003 is the pressure (v1's 970 hPa against 973.07). So the paper's gas puts the excess at 0.0121 (estimates 4 and 8), not 0.0137.
- **The power law's low Rayleigh comes from the form.** Under a power law the fit wants a Rayleigh depth 8 percent under the paper's (0.0859 against 0.0930), and fixing the paper's doubles the residual (0.00182 against 0.00096). Under the curved form the paper's Rayleigh fits as well as a free one (0.00083 against 0.00081).
- **Two visibilities cannot separate the gas from a curved aerosol.** Estimates 5 and 7, curved with a free Rayleigh, have Jacobian condition numbers of 1.5e4 and 1.0e4 and 2SE of 0.0280 and 0.0601. The other six have condition numbers of 14 to 910.
- The pilot's envelope runs from -0.0234 to 0.0969, both ends from estimate 7.

**The crossing.** Scaled linearly in g from #49's matched reading (Δ_M at 0.01367), the amount's part equals the bar at an excess g* = 0.01367 b / Δ_M: 0.0094 at sun -1 and 0.0095 at sun 0, +1 and +10. It is nearly the same at every sun because Δ_M and m both measure how Prague responds to visibility: #49's Δ_M is 1.435 to 1.447 times the bar at all four. So the four conditions move together. Scaled the same way, a condition carries only if g_lo is at least 0.0541, 0.0351, 0.0263 or 0.0242 (sun -1, 0, +1, +10).

**What the pilot predicts, branch by branch.**

- **"Not this" cannot fire.** Estimates 4 and 8 fit no gas parameter, so their excess, 0.0121, is fixed by data in hand and lies inside every envelope: g_hi ≥ 0.0121. Scaled linearly, Δ_M(0.0121) is 1.268 to 1.279 times the bar at the four suns, so Δ_hi > b at every condition. Branch 3 is dead, per condition and for the rule, and so is the rule's branch 5, which needs a "not this" condition beside a contributing one.
- **"Carries" cannot fire.** For the same reason g_lo ≤ 0.0121, which is under every condition's linear carrying excess (0.0242 at the least). No condition carries, so branch 1 is dead too.
- **The live outcomes are "contributes" (branch 2) and "spans" (branch 4).** The rule says "contributes" only if g_lo exceeds about 0.0095: all eight estimates, after twice their errors, must leave Prague at least that much deeper than v1.
- **On the pilot's two visibilities the rule would say "spans"** (g_lo = -0.0234).
- **With six nodes the outcome is open.** A synthetic check (six power-law aerosols with depths near 6 / V at 550 nm and Ångström exponents from 0.8 to 1.6, residuals of rms 0.001 as in the pilot, linearised at the truth) puts 2SE at 0.010 to 0.014 for estimate 5 and 0.006 to 0.008 for estimate 7. Estimate 5's pilot margin over the crossing is 0.012. So "contributes" is reachable but is not the expected outcome. Estimate 2's pilot margin is only 0.0007 (its lower end is 0.0102).
- **The family decides the pilot's verdict, and the author saw that before fixing it.** With power laws only (estimates 1 to 4) the pilot's envelope would be 0.0102 to 0.0241 and the pilot's verdict "contributes", by 0.0007. The curved estimates are in the family for two reasons. Prague's aerosol changes type across its nodes: the paper's Fig. 4 labels 131.8 km continental clean, 59.4 km continental average and 27.6 km continental polluted, with the others interpolated or extrapolated between them, and the depth of a changing mixture of types is not in general a power law in wavelength. And the power law's low Rayleigh is a property of the form (above). A family of power laws alone would assume the separation that the issue asks the six nodes to show.

Earlier looks at the same question, also disclosed: #34's "about ±0.01" (from v1's gas, never derived); #49's first cold read, ±0.003 to ±0.005; and #49's third cold read, which fitted gas depths to one visibility's eleven channels (an estimator error of 0.003 to 0.006, fit rms 0.0034; its method is not recorded).

## What is compared

- **Prague:** the reference implementation, dataset and driver pinned for #49: commit `2385c912e9051c1258013ff8c3ce2e19e10fb917`, the full dataset (SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`), observer altitude 0.34 km, ground albedo 0.2, the model's own interpolation in visibility, and the primary reading (`xyz`, linear interpolation between channel centres).
- **One new fixture,** `tests/fixtures/prague/aerosol-depth.json`. It holds:
  - the zenith transmittance at the six nodes, all eleven channels, so ȳ(131.8) is in it;
  - the four targets g_lo, g_hi, g_1 and g_4;
  - for each target that the rule or the report evaluates, the matched visibility V_g, whether the match is clamped, and Prague's `xyz` at V_g at each of the four suns, at the stored views and azimuths of that sun's fixture.

  A target is evaluated only where it is needed: g_lo only if g_lo > 0, and g_hi, g_1 and g_4 only if they lie in (0, g_max]. The tool refuses to write the fixture unless the six visibilities in the dataset's header are the six above. The four existing fixtures are not regenerated or edited, and `fixture --check` for each must still pass.
- **v1** enters only through L and m, which come from the committed fixtures as #49 computes them. The amount's part Δ_M compares Prague with Prague.

## Reported, never ruled

- The pilot section, recomputed.
- For each of the eight estimates on the six nodes:
  - a_R, a_O in DU, and each node's exp(u_i), α_i and γ_i;
  - the residual rms, the largest residual, and the condition number of J;
  - G_k, SE_k, g_k, g_k - 2SE_k and g_k + 2SE_k.
- Prague's aerosol depth at each node, ȳ(V_i) - G_k, under estimates 1 and 4.
- g_max, with the old estimate's excess (0.01367) and the paper's (0.0121) beside the envelope.
- At each condition:
  - Δ_M at the excesses of estimates 1 and 4, beside its linear scaling from #49's matched reading;
  - the shares Δ / L and Δ / (L - Δ_O), where Δ_O is #49's ozone part.
- At the 0.03 load, each estimate's excess there, ȳ(V_0.03) - G_k - 0.03, and whether a match exists (whether that excess is at most ȳ(V_0.03) - ȳ(131.8)). No matched sky is evaluated at 0.03.

## What is not decided here, and what this cannot tell

- This attributes v1's offset from Prague. It does not say which model is right: both are models.
- These are common to all eight estimates, so no envelope can see them:
  - the ozone cross-section's shape: SPCTRL2's table (v1's) stands in for the paper's Gorshelev cross-sections at 223 K;
  - the Rayleigh cross-section's shape: v1's (with argon and CO2) stands in for the paper's Bodhaine cross-section for N2 and O2. Only its amplitude is fitted or fixed;
  - the channels: everything is read at the channel centres, as the tool reads Prague;
  - Prague's fit of its own transmittance. The supplement gives that rank-12 fit's largest error over the whole dataset as 0.014, without saying whether in T or in its square root, and nothing for the zenith. A smooth part of that error biases every estimate alike.
- The depths compared are Prague's above its 0.34 km observer and v1's whole column above its ground observer, as in #34 and #49: the air above the two observers matches (#49, rule 2).
- Δ_M moves Prague along its own visibility axis, so it changes Prague's aerosol type with its depth (the nodes change type). It measures "Prague with as much aerosol as v1", not "Prague with v1's aerosol".
- The aerosol's profile and the observer and ground stay as #49 registered them (no test), with everything else #49 listed as untested.
- If the error still spans the bar, what would tell is the paper's own gas computed per channel from its cross-sections (a table not in hand), or a reference with v1's own atmosphere (libRadtran, run offline; GPL, never vendored or imported).
- A change to v1 for any of this is a new renderer with its own registration (#50), never a change to v1.

## Decisions, each with its rejected alternative

- **The error is the envelope of eight estimates, each widened by twice its own fitting error.** Rejected:
  - one estimate with its fitting error alone. In the pilot estimate 1's 2SE is 0.0032 while the estimates' excesses differ by 0.025;
  - a centre plus or minus one error, which privileges one estimate. Among the power laws alone the centre's choice moves the excess from 0.0121 to 0.0208;
  - the issue's own proposal, a fraction of the gap fixed before any fit. Any fraction chosen now would be chosen with the pilot's numbers in view, while the envelope takes its threshold from the bar, which has a reason of its own;
  - the spread over the six nodes left out one at a time, which cannot see a form error shared by every node;
  - the spread of the eight excesses without their fitting errors, which would make an undetermined fit look certain.
- **The family is three assumptions the transmittance cannot settle by itself, each fitted or fixed.** Rejected:
  - power laws only (see the pilot);
  - a free shape at each node (ten numbers per node, in which the gas can hide);
  - one shape, or one curvature, shared by all nodes, because Prague's aerosol changes type across its nodes;
  - the paper's own ozone cross-sections, which are not in hand. That table is what would tell.
- **The 340 nm channel is left out.** SPCTRL2's table starts at 380 nm (v1's function gives 0 below it), while the paper's ozone absorbs strongly in the Huggins band, across 320 to 360 nm. The fit would read that absorption as Rayleigh. In the pilot, including the channel moved estimate 1's G from 0.1149 to 0.1285 and tripled its residual, from 0.0010 to 0.0030. Rejected: also leaving out 740 nm, where the paper lists no absorber.
- **The paper's gas uses v1's cross-sections at the paper's column and pressure.** Rejected: v1's own 970 hPa, which is v1's air, not Prague's air above 0.34 km.
- **The amount's part is evaluated in Prague at the envelope's ends, not scaled.** Rejected: #49's linear scaling, which holds near the old match only and has no account of the clamp or of an excess at or below zero.
- **The rule rules at the 0.1 load, at all four suns.** Rejected:
  - the 0.03 load, where whether a match exists at all depends on the estimate, so the set of ruled conditions would depend on the fit. #49's match there was clamped;
  - sun 0 alone. The four suns move together, but each is a separate Prague evaluation, and a fault at one would show as a condition that spans.
- **The bar is 2m, as in #49.** Rejected: m plus the matched reading's own inverse-visibility shift. The tool defines no inverse-visibility matched reading, and its match would sit at a different visibility.
- **The fit is unweighted, in -ln T.** Rejected: weights from the transmittance's error, which nobody has per channel.
- **Twice the fitting error,** the same factor of two as every bar since #45. Rejected: a factor chosen against the pilot's margins.
- **An excess at or below zero gives Δ_lo = 0 without a run, and a clamp at the upper end leaves Δ_hi unbounded.** Rejected: evaluating Prague at negative excesses, which costs runs for a branch whose outcome the sign already sets.
