# Prague's aerosol amount, from Prague's own ozone aloft and the paper's Rayleigh (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-17, for issue #57.

At that commit Prague's zenith transmittance has been recorded at two visibilities only, 123.8187 and 49.9864 km, both between its nodes and both at the 0.34 km observer (the same eleven numbers in all four committed fixtures). #49's matched-depth search also evaluated it at 0.34 km, starting at Prague's clearest and haziest nodes, and recorded two results: the 0.1 load's match at 55.5320 km, and the 0.03 load's clamped match at 131.8 km. That clamped match puts one node quantity in hand: Prague's -ln T at 131.8 km, interpolated at 550 nm, is 0.15893 (`docs/renderer/prague-aerosol-depth-criterion.md`). No channel of Prague's transmittance at any visibility node has been recorded or printed, and nor has any transmittance at an observer altitude other than 0.34 km (the fixtures' altitude-0 reading is radiance, not transmittance). Nothing under this file computes one before it lands. The builder implements in `tools/prague_reference.py` exactly what this file says the tool prints; a disagreement between this file and the tool is a defect in the tool.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

**The question.** At the 0.1 load's visibility (Koschmieder, V_p = 49.9864 km), is Prague's aerosol deeper than v1's 0.1 by enough that matching it removes more than #49's bar? #55 put the crossing, the excess at which the amount's part Δ_M equals the bar, at g* = 0.0094 at sun -1 and 0.0095 at sun 0, +1 and +10 (`report --depth`), and showed that the paper's gas computed with v1's cross-sections puts the excess at 0.0121. In gas terms: is Prague's gas depth at 550 nm, channel-interpolated, below ȳ(V_p) - 0.1 - g*, about 0.1263? Which side it falls on turns mostly on Prague's ozone near 540 to 580 nm. #55 could not take that from SPCTRL2's table, the stand-in that can by itself move a single gas estimate by several times the margin (its problem 3).

**The route.** The Rayleigh is the paper's, fixed, within a registered tolerance. The ozone's spectral shape is Prague's own: its zenith transmittance seen from 15 km, Prague's highest observer altitude, minus the paper's Rayleigh there, minus a power-law aerosol aloft fitted through the two channels where ozone does not absorb. The ozone's amount at 0.34 km is fitted across Prague's six visibility nodes, with the Rayleigh fixed and a curved aerosol at each node. The gas's error carries four things: the six-node fit's error, the extraction's error, a tolerance for an error in Prague's own transmittance that is shared by the six nodes (at 15 km and at 0.34 km), and the Rayleigh tolerance. The verdict compares the resulting envelope of the excess with the crossing, at each sun.

## Definitions

**Prague's spectra.** For each visibility node V_i in (20, 27.6, 40, 59.4, 90, 131.8) km and each of two observer altitudes h, 0.34 km and 15 km:

- T_h(c, V_i) is Prague's transmittance in channel c, computed as the tool computes `zenith_transmittance` for the fixtures: `evaluate` with the full dataset, `mode="transmittance"`, the zenith query (0, 90), ground albedo 0.2 and sun elevation 10 degrees (transmittance depends on neither), at visibility V_i and `altitude_m` 340 or 15000. 15000 m is the last node of the dataset's altitude axis, and each V_i is a node of its visibility axis. `evaluate` already refuses an altitude the model reports differently from the one asked for.
- y_h(c, V_i) = -ln T_h(c, V_i). Every fit below uses the ten channels centred 380, 420, ..., 740 nm. The 340 nm channel is left out, as in #55: the paper's ozone absorbs in the Huggins band there.
- The 132 transmittances (eleven channels, six nodes, two altitudes) are written to a new committed fixture, `tests/fixtures/prague/aloft.json`, by `fixture --aloft`, with its provenance (the dataset's SHA-256, the reference commit, and the altitude and visibility range the driver reported for each evaluation). `fixture --aloft --check` rebuilds it and compares, as `fixture --check` does for the other four. `report --aloft` reads that fixture and the four committed ones, and runs no Prague evaluation.

**Constants.**

- x_c = ln(λ_c / 550 nm).
- p_0.34 = 1013.25 (1 - 0.0065 × 340 / 288.15)^5.25588 hPa, unrounded (973.0664; the tool's `PAPER_PRESSURE_HPA`), and p_15 = 120.45 hPa, the U.S. Standard Atmosphere's pressure at 15 km.
- R_paper = `rayleigh_optical_depth(550, p_0.34)` (0.093050), ρ_c = `rayleigh_optical_depth(λ_c, p_0.34)` / R_paper, and R15_c = `rayleigh_optical_depth(λ_c, p_15)`.
- For any per-channel quantity v, v̄ is v interpolated linearly at 550 nm between the 540 and 580 nm channel centres: v̄ = 0.75 v(540) + 0.25 v(580). ρ̄ = 1.00968.
- ȳ(V_p) is Prague's -ln T at V_p = 49.9864309567 km and 0.34 km, interpolated linearly at 550 nm between channel centres, from the committed fixtures (the tool's `y_bar_p`, 0.23575).
- g*_s is #55's crossing at sun s, as `report --depth` computes it (`depth_crossing`), unrounded.
- K = (0.99, 1.00, 1.01), the Rayleigh factors.

**Step 1, Prague's ozone aloft.** For each k in K:

- ŷ15(c) is the mean of y_15(c, V_i) over the six nodes, and r(c) = ŷ15(c) - k R15_c.
- The aerosol aloft is β (λ / 550 nm)^-α, from r(380) and r(420):
  - if both are positive and α_2 = ln(r(380) / r(420)) / ln(420 / 380) lies in [0, 3], then α = α_2 and β = r(380) (380 / 550)^α, exact through both points (the tool prints "exact");
  - otherwise α is ln(max(r(380), 1e-12) / max(r(420), 1e-12)) / ln(420 / 380) clipped to [0, 3], and β is the least-squares amplitude on the two points, (p_380 r(380) + p_420 r(420)) / (p_380² + p_420²) with p_λ = (λ / 550)^-α, floored at 0 (the tool prints "guard").
- O(c) = r(c) - β (λ_c / 550)^-α for the channels from 460 nm up, and O(380) = O(420) = 0. O is Prague's ozone depth above 15 km by channel, and Ō its value interpolated at 550 nm.

**Step 2, the six-node fit.** For each k in K:

- The model is ŷ(c, V_i) = k R_paper ρ_c + a O(c) + exp(u_i - α_i x_c + γ_i x_c²). It has 19 parameters: a, the ozone at 0.34 km as a multiple of the ozone above 15 km, and u_i, α_i and γ_i for each node's aerosol (its depth at 550 nm is exp(u_i)).
- The fit is unweighted least squares of ŷ - y_0.34 over the 60 values, with `scipy.optimize.least_squares` and #55's settings: method `trf`, the analytic Jacobian, `xtol`, `ftol` and `gtol` all 1e-15, at most 20000 evaluations, every other argument at its default. The start is a = 1, and at each node u_i = ln max(ȳ_i - (k R_paper ρ̄ + Ō), 0.001), α_i = 1.3 and γ_i = 0, where ȳ_i is y_0.34(·, V_i) interpolated at 550 nm. A fit with `status` ≤ 0 is a refusal: the tool prints which factor failed to standard error and exits 1.
- G_k = k R_paper ρ̄ + a Ō is the gas depth at 550 nm.
- s_k = sqrt(SSR / 41) is the residual scale (41 = 60 values - 19 parameters).
- SE_fit,k = sqrt(qᵀ C q), with C = s_k² (JᵀJ)⁻¹ at the solution and q = Ō for a and 0 for every aerosol parameter. It is infinite if `numpy.linalg.inv` raises or if qᵀ C q is not a positive finite number.

**Step 3, the node-to-node scale at 15 km.**

- D(c, V_i) = y_15(c, V_i) - ŷ15(c).
- Each node's D(·, V_i) is fitted by linear least squares with 1, x_c and x_c², three coefficients per node, with `numpy.linalg.lstsq`.
- s_15 = sqrt(SSR_15 / 35), where SSR_15 is the sum of squared residuals over the 60 values and 35 = 60 - 10 - 15 (the ten channel means and the fifteen independent node coefficients, since the six nodes' differences sum to zero).

**Step 4, the extraction's error and the tolerance for a shared error.** For each k in K:

- ∇_k(c) = ∂G_k / ∂ŷ15(c), by central differences: steps 1 and 2 are repeated with ŷ15(c) raised by 1e-5 and lowered by 1e-5, each perturbed fit starting at the unperturbed fit's solution, and ∇_k(c) is the difference of the two G_k over 2e-5.
- ∇⁰_k(c) = ∂G_k / ∂y_0.34(c, ·), the same way, with y_0.34(c, V_i) raised and lowered by 1e-5 at all six nodes at once and O held at its unperturbed value (step 2 only).
- A perturbed fit with `status` ≤ 0 is a refusal.
- SE_ext,k = |∇_k| sqrt(s_15² / 6 + s_k²), with |∇_k| the Euclidean norm over the ten channels. The first term is the noise of the six-node mean aloft spectrum, from its own scatter (choice 3 below). The second is the tolerance for an error in Prague's transmittance at 15 km that is the same at all six nodes (choice 4).
- SE_0,k = |∇⁰_k| s_k: the same tolerance for an error at 0.34 km shared by the six nodes, which the six-node fit's SE, treating the 60 values as independent, does not carry (choice 4).
- SE_k = sqrt(SE_fit,k² + SE_ext,k² + SE_0,k²).

**Step 5, the envelope.**

- g_k = ȳ(V_p) - G_k - 0.1, Prague's aerosol depth at V_p minus v1's, under factor k.
- g_lo = min over k of (g_k - 2 SE_k), and g_hi = max over k of (g_k + 2 SE_k).

**Step 6, the verdict.** At each sun s in (-1, 0, +1, +10):

- **contributes** if g_lo > g*_s;
- **not this** if g_hi ≤ g*_s;
- **spans** otherwise.

The verdict line is the outcome if all four suns agree, and otherwise "mixed", followed by each sun's outcome. g*_s differs between the suns by less than 0.0001, so "mixed" needs g_lo or g_hi to fall between them.

What each outcome says:

- **contributes**: at every excess in the envelope, the amount's part Δ_M, scaled linearly from #49's matched reading as g* is, is over the bar: matching Prague's aerosol depth would remove more than #49's bar from v1's offset.
- **not this**: at every excess in the envelope it is at or under the bar.
- **spans**: the envelope holds excesses on both sides of the crossing. This is the outcome when the error, not the sky, decides.

"Carries", #49's fourth outcome, is not ruled: its excesses (0.0242 and up) are linear extrapolations far past the segment where Δ_M is known to be nearly linear, and #55 showed Δ_M is under L / 2 even at the old excess.

## The five design choices, each with its reason

The day-5 power study (below) found five places where, left open, the design rather than the sky would set the verdict. Each is registered here with a reason that does not depend on the pilot. Each also makes the verdict conditional on something Prague cannot show by itself, and that is said with it.

**1. The Rayleigh: the paper's, within ±1 percent, as an envelope.** The paper's Rayleigh is Bodhaine et al.'s formula for N2 and O2 only ("only takes N2 and O2 into account", main article §3.2; supplement §3.1, Equation 23), over the U.S. Standard Atmosphere. The tool's `rayleigh_optical_depth` uses Peck and Reeder's refractive index for standard air with Bodhaine's King factor, a column set by pressure and standard gravity. The known differences are:

- leaving out argon and CO2 lowers the depth by about 1.0 percent (their share of air's scattering at 550 nm, from Bodhaine's refractive indices and King factors for each gas);
- the tool's 0.0969 at 550 nm and 1013.25 hPa sits 0.4 percent under Bucholtz's 0.0973 for the same air;
- a column that allows for gravity falling with height, as Bodhaine's does, is larger, by about 0.2 percent above 0.34 km and about 0.5 percent above 15 km.

None is over 1 percent, and they partly cancel: together they put the paper's Rayleigh about 0.4 percent under the tool's above 0.34 km and about 0.1 percent under it above 15 km. The tolerance enters as an envelope over three fixed factors, never as a free parameter: with the Rayleigh free, the gas cannot be told from a curved aerosol (#55, estimates 5 and 7). One factor k applies at both altitudes, since the composition and the cross-section are the same there. **Conditional on:** the paper's Rayleigh lying within ±1 percent of the tool's at both altitudes, by a common factor. Outside the tolerance the verdict follows the assumed Rayleigh instead of the sky (the study's Rayleigh axis: P(contributes) stays flat while the true excess grows), and each 1 percent moves the 50 percent point by about 0.0007 to 0.0008 (the study, and the power check below). The 0.3 percent difference between the two altitudes' gravity corrections is carried by neither the envelope nor the error; in the power check, a truth whose 15 km Rayleigh is 0.5 percent over its 0.34 km factor moves the 50 percent point by 0.0001 or less (0.0129 against 0.0128 at rms 0.0003, 0.0180 against 0.0181 at 0.001).

**2. The aerosol aloft: removed, by a two-point power law with its exponent guarded to [0, 3].** Ignoring it biases G by +0.0032 and gives false "not this" in up to 0.75 of trials (the study's estimator ENO). Only 380 and 420 nm are nearly free of ozone among the channels used (a weak ozone tail there was not material in the study), so two points determine the power law and nothing checks it. The guard keeps α inside the range of real aerosol Ångström exponents: about 0 for coarse dust, up to about 2.5 for fine smoke and sulfate. A residual at or under 0, or an exponent outside the guard, means the two points cannot be fitted exactly by a physical power law, and the clipped least-squares fit is the nearest one that is. The paper's aerosols are WASO, INSO and SOOT in every configuration (main article §3.2), so nothing fixes the exponent aloft. A fixed exponent of 1 halves the extraction's noise but needs a prior the paper does not give; with a true exponent of 1.5 to 2.5 it biases G by -0.0011, against 0.0000 for the guarded two-point law (the study's F2). **Conditional on:** Prague's aerosol aloft following a power law between 380 and 420 nm and on to 740 nm. The study's variants (an exponent of 0 to 0.5 or 1.5 to 2.5, an amount that varies by ±20 percent across the nodes, a depth aloft up to 0.02) are in the power check below.

**3. The extraction's error: in the envelope, from the aloft spectrum's own scatter.** The six-node fit's SE treats Prague's ozone shape as exact. In the study it is not: the shape extracted from a noisy aloft spectrum adds an error that the fit's 2SE omitted, so the envelope covered the truth in only 92 percent (rms 0.0003) and 81 percent (rms 0.001) of trials. The registered term propagates the noise of the six-node mean aloft spectrum, s_15 / √6 per channel, through the whole extraction and fit, by the gradient |∇_k|. s_15 is measured from Prague's own six aloft spectra after each node's smooth difference from their mean is removed, so it is the part of Prague's error that differs between the nodes and is not smooth across the channels. Most of |∇_k| comes from 380 and 420 nm, which the power law extrapolates from (the study's J1: dG/dy of 1.36 and -2.51 there, under 0.8 elsewhere).

**4. Prague's own error, where the six nodes share it: a tolerance equal to what the six-node fit leaves, at both altitudes.** Prague's transmittance is not stored; it is a rank-12 reconstruction from one singular value decomposition per altitude across every atmosphere configuration, with "a maximum absolute error over the dataset of 0.014" (supplement §2). At 15 km the six visibility nodes differ only by their thin aerosol aloft, so the reconstruction errors there are likely to be shared by the six nodes. Averaging over the nodes does not reduce such an error, and s_15 cannot see it. The registered tolerance takes it to be no larger, per channel, than s_k, the scale of what the six-node fit at 0.34 km leaves. The same holds at 0.34 km, where the six nodes also share one reconstruction per altitude: the six-node fit's SE treats its 60 values as independent and so divides a shared error by √6 that it should not, and SE_0,k carries it at the same scale. That fit sees the same reconstruction's error wherever it is not smooth across channels, whether or not it is shared by the nodes: a smooth per-node aerosol cannot absorb a channel-to-channel error. It is also the term that grows when Prague's ozone shape from 15 km fits the 0.34 km data badly. The dataset-wide maximum of 0.014 is not used: it bounds the worst point of every configuration, altitude and distance, not the zenith from 15 km, the shortest path the dataset holds, and as a tolerance it would put the envelope's half-width near 0.09 (twice 3.1 times 0.014), so the verdict would be "spans" whatever the sky. **Conditional on:** Prague's error that is shared by the six nodes, at either altitude, being no larger than s_k per channel and independent from channel to channel. An error with the worst sign in every channel would move G by up to the sum of |∇_k(c)|, about twice |∇_k| (the study's J2: 6.6 against 3.1). An error that is smooth across the channels and shared by the nodes can move the extracted shape while the curved aerosols at 0.34 km absorb most of its trace, so neither s_k nor s_15 grows; in the power check, such an error (a tilt in log wavelength reaching ±0.001 at the ends of the band, the same at all six nodes) left the 50 percent points within 0.0015 of the base truths' and the coverage at 0.995 or more, but a larger or differently shaped one could do more. Neither is carried by the envelope.

**5. One ozone shape at both altitudes.** The paper uses one ozone cross-section, Gorshelev et al.'s at 223 K, at every height, and its only absorber is O3: "the only molecules taken into account are N2, O2 and O3" (supplement §3.1; main article §3.2). So the ozone above 15 km and the ozone above 0.34 km have the same spectral shape in Prague by construction, and the shape from 15 km may stand for the shape at 0.34 km. **Conditional on:** that being so in the dataset as built. A shape at 0.34 km tilted by ±3 percent per 100 nm against the shape aloft moves the 50 percent point of "contributes" by +0.0009 and 0.0000 at rms 0.0003 (+3 and -3 percent), and by -0.0001 and -0.0007 at rms 0.001, in the power check.

## What was seen before this file landed, said plainly

Three sets of numbers were in view: #55's two-spectra pilot, the day-5 power study of an earlier form of this route, and the power check of the rule above. None evaluates Prague at a node or at 15 km.

**#55's two-spectra pilot** (`docs/renderer/prague-aerosol-depth-criterion.md`, printed by `report --depth`) is this issue's pilot too. The quantities this route leans on:

- the paper's gas, G_paper = 0.12367, puts the excess at 0.0121, 0.0026 above the crossing;
- estimate 6 (curved aerosol, ozone free with SPCTRL2's shape, the paper's Rayleigh), this route's analogue, fits two spectra with condition number 87.2, 2SE 0.0013 and residual rms 0.00083. ±1 percent in the Rayleigh moves its G by ±0.0008. Its ozone is 318.5 DU against the paper's 345, with a shape the paper does not use;
- the curved form leaves the paper's Rayleigh fitting as well as a free one (rms 0.00083 against 0.00081), while the power-law form wants a Rayleigh 8 percent low. That is why the aerosol at each node is curved here;
- Prague's -ln T at its clearest node, interpolated at 550 nm, is 0.15893.

**The day-5 power study** (machine-local, `~/.workshop/lastlight/research/2026-10-16-issue57-power/`, headline on #57). It studied an earlier estimator: the same extraction with the aloft Rayleigh always at k = 1, no extraction-error term, no tolerance aloft. Its truths: six nodes at 0.34 km with curved aerosol (depth near 6 / V, Ångström falling from 1.6 at 131.8 km to 0.8 at 20 km, curvature ±0.3), ozone of SPCTRL2's shape or a perturbed one (10 percent more at 600 nm, a ±5 percent tilt, or the band 5 nm redder), and a Rayleigh of the paper's times (1 + δ_R). At 15 km: Rayleigh at 120.45 hPa, 0.80 to 0.90 of the ozone, and a power-law aerosol aloft of depth 0 to 0.005 at 550 nm. White noise of rms 0.0003 or 0.001. It found:

- the route has power. With δ_R within ±1 percent, P(contributes) is 0.5 at a true excess of 0.0106 (rms 0.0003) and 0.0111 (rms 0.001), and monotone in every variant;
- the ozone-shape perturbations no longer bias G (|mean error| ≤ 0.00006), because the shape comes from Prague's own spectrum aloft. #55's problem 3 is gone;
- the five places where the design would set the verdict, registered above;
- not material: aerosol aloft that varies across the nodes, two-mode aerosol, curvature ±0.6, an aloft fraction of 0.6 to 1.0, a weak ozone tail at 380 to 440 nm, a Rayleigh exponent off by ±0.05, correlated channel noise, and the start values.

**The power check of the registered rule** is `tools/issue57_power.py`, committed with this file. It draws the day-5 study's truths, adds the variants below, runs exactly the estimator of steps 1 to 6 (each sun's unrounded g*_s, 0.009444, 0.009483, 0.009527 and 0.009492 at sun -1, 0, +1 and +10, and ȳ(V_p) rounded to 0.235751), and prints how often each verdict line comes out at each true excess. It reads no Prague data. Every cell is seeded from its name, so `uv run --frozen python tools/issue57_power.py --n 400 --cells base_s1,base_s3,base_s10,base_s20` and the same with `--n 200 --cells` and the variant cells' names reprint these tables. Its trials cycle through the five ozone shapes, and δ_R is uniform in ±1 percent unless a row says otherwise. Coverage is the share of trials whose envelope contains the true excess.

**Headline, base truths.**

| residual rms | 50 percent point of "contributes" | 50 percent point of "not this" | P(contributes) at 0.0121 | median envelope width | lowest coverage | share of trials with the guard active |
|---|---|---|---|---|---|---|
| 0.0001 | 0.0112 | 0.0076 | 0.98 | 0.0032 | 1.00 | 0.10 |
| 0.0003 | 0.0128 | 0.0064 | 0.23 | 0.0066 | 1.00 | 0.19 |
| 0.001 | 0.0181 | 0.0020 | 0.02 | 0.0155 | 1.00 | 0.46 |
| 0.002 | 0.0199 | never (0.45 at 0) | 0.00 | 0.0183 | 1.00 | 0.68 |

**The registered rule on the base truths** (400 trials per cell): P(contributes) / P(not this), by the true excess.

| residual rms | 0 | 0.004 | 0.006 | 0.008 | 0.009 | 0.0095 | 0.010 | 0.011 | 0.0121 | 0.013 | 0.015 | 0.018 | 0.022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.00/0.39 | 0.00/0.00 | 0.00/0.00 | 0.01/0.00 | 0.39/0.00 | 0.98/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 |
| 0.0003 | 0.00/1.00 | 0.00/0.99 | 0.00/0.60 | 0.00/0.04 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.01/0.00 | 0.23/0.00 | 0.57/0.00 | 0.99/0.00 | 1.00/0.00 | 1.00/0.00 |
| 0.001 | 0.00/0.71 | 0.00/0.28 | 0.00/0.09 | 0.00/0.01 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.02/0.00 | 0.06/0.00 | 0.29/0.00 | 0.48/0.00 | 0.96/0.00 |
| 0.002 | 0.00/0.45 | 0.00/0.04 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.01/0.00 | 0.06/0.00 | 0.34/0.00 | 0.67/0.00 |

What it shows:

- **Monotone in every cell.** In all 26 cells, P(contributes) never falls and P(not this) never rises as the true excess grows (no adverse step between neighbouring excesses), and each outcome fires somewhere on the grid. No outcome is fixed by the design.
- **No false verdict, even outside the conditions.** In all 78,000 trials, no "contributes" came out at a true excess under the lowest crossing (0.009444) and no "not this" over the highest (0.009527), in every cell, including the cells outside the conditions.
- **Coverage.** The envelope held the true excess in every trial at the base truths, and in at least 0.990 of trials in every variant inside the conditions, including the one built for the shared tolerance (15 km noise the same at all six nodes: 0.995 and 0.990).
- **Outside the Rayleigh tolerance** (the truth's Rayleigh 2 percent under or over the paper's), coverage falls to 0.915 at rms 0.0003 and the 50 percent point of "contributes" moves to 0.0141 and 0.0113 (base 0.0128): the verdict follows the assumed Rayleigh, about 0.0007 per percent.
- **The aerosol aloft** (exponents of 1.5 to 2.5 or 0 to 0.5, ±20 percent across the nodes, depths up to 0.02) moves the 50 percent points by at most 0.0013.
- **A shared error is what the shared tolerance is for.** With the 15 km noise the same at all six nodes, the nodes do not differ, so s_15 sees nothing and the "noise only" form's extraction term vanishes. Its coverage falls to 0.735 (rms 0.0003) and 0.605 (0.001), with 13 and 33 false "contributes" and 6 and 15 false "not this" among 2,600 trials each. The registered form holds 0.995 and 0.990, with none. (The rig records every form on every trial; these numbers come from the registered run, after the form was chosen.)
- **Refusals:** none in 78,000 trials.

The variants, 200 trials per cell (at the base truths, for comparison: 50 percent points 0.0128 and 0.0064 and P(contributes) at 0.0121 of 0.225 at rms 0.0003; 0.0181, 0.0020 and 0.020 at 0.001):

| variant | what differs from the base truth | rms | 50 percent point of "contributes" | 50 percent point of "not this" | P(contributes) at 0.0121 | lowest coverage |
|---|---|---|---|---|---|---|
| dr_m2 | the truth's Rayleigh 2 percent under the paper's (outside the tolerance) | 0.0003 / 0.001 | 0.0141 / 0.0177 | 0.0080 / 0.0052 | 0.000 / 0.000 | 0.915 / 0.990 |
| dr_p2 | 2 percent over (outside the tolerance) | 0.0003 / 0.001 | 0.0113 / 0.0163 | 0.0050 / 0.0004 | 0.980 / 0.105 | 0.975 / 0.995 |
| topcommon | the 15 km noise the same at all six nodes | 0.0003 / 0.001 | 0.0130 / 0.0166 | 0.0060 / 0.0025 | 0.300 / 0.060 | 0.995 / 0.990 |
| asteep | aloft exponent 1.5 to 2.5 | 0.0003 / 0.001 | 0.0124 / 0.0168 | 0.0068 / 0.0022 | 0.355 / 0.005 | 1.000 / 1.000 |
| aflat | aloft exponent 0 to 0.5 | 0.0003 / 0.001 | 0.0127 / 0.0171 | 0.0064 / 0.0027 | 0.325 / 0.005 | 1.000 / 1.000 |
| amp20 | aloft amount ±20 percent across the nodes | 0.0003 / 0.001 | 0.0128 / 0.0180 | 0.0063 / 0.0021 | 0.210 / 0.020 | 1.000 / 1.000 |
| bigaloft | aloft depth up to 0.02 at 550 nm | 0.0003 / 0.001 | 0.0130 / 0.0192 | 0.0063 / 0.0008 | 0.100 / 0.015 | 1.000 / 1.000 |
| tiltp | ozone shape at 0.34 km +3 percent per 100 nm against aloft (choice 5 broken) | 0.0003 / 0.001 | 0.0137 / 0.0180 | 0.0062 / 0.0024 | 0.055 / 0.010 | 1.000 / 0.995 |
| tiltm | -3 percent per 100 nm (choice 5 broken) | 0.0003 / 0.001 | 0.0128 / 0.0174 | 0.0053 / 0.0015 | 0.260 / 0.010 | 1.000 / 1.000 |
| r15 | the truth's 15 km Rayleigh 0.5 percent over its 0.34 km factor | 0.0003 / 0.001 | 0.0129 / 0.0180 | 0.0065 / 0.0018 | 0.170 / 0.015 | 1.000 / 1.000 |
| smooth | a smooth error at 15 km shared by the nodes, ±0.001 at the band's ends (choice 4 broken) | 0.0003 / 0.001 | 0.0128 / 0.0170 | 0.0068 / 0.0035 | 0.220 / 0.010 | 0.995 / 1.000 |

With 200 trials, a 50 percent point is uncertain by about ±0.0002 and a share near 0.5 by about ±0.035.

**The form of the error was chosen with power numbers in view, against power.** Before it was fixed, a scratch run (200 trials per cell, an earlier form without SE_0,k and with one crossing of 0.0095) compared three forms on the same trials: the six-node fit's SE alone (the day-5 estimator), that plus the extraction's noise averaged over the nodes, and that plus a shared error aloft at the scale of the fit's residual. All three were monotone; over rms 0.0001 to 0.001 the coverage was 0.80 to 0.98, 0.96 to 1.00 and 0.99 to 1.00, and P(contributes) at an excess of 0.0121 was 0.99, 0.94 and 0.30 at rms 0.0003. The rig now prints the same comparison for the registered rule (the forms "fit only", "noise only" and "registered", on the same trials):

P(contributes) / P(not this) / coverage, 400 trials per cell:

| residual rms | form | 0 | 0.004 | 0.006 | 0.008 | 0.009 | 0.0095 | 0.010 | 0.011 | 0.0121 | 0.013 | 0.015 | 0.018 | 0.022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0003 | registered | 0.00/1.00/1.00 | 0.00/0.99/1.00 | 0.00/0.60/1.00 | 0.00/0.04/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.01/0.00/1.00 | 0.23/0.00/1.00 | 0.57/0.00/1.00 | 0.99/0.00/1.00 | 1.00/0.00/1.00 | 1.00/0.00/1.00 |
| 0.0003 | noise only | 0.00/1.00/0.97 | 0.00/1.00/0.98 | 0.00/0.98/0.96 | 0.00/0.43/0.99 | 0.00/0.04/0.99 | 0.00/0.00/0.99 | 0.03/0.00/0.98 | 0.43/0.00/0.99 | 0.91/0.00/0.98 | 1.00/0.00/0.99 | 1.00/0.00/0.97 | 1.00/0.00/0.98 | 1.00/0.00/0.99 |
| 0.0003 | fit only | 0.00/1.00/0.90 | 0.00/1.00/0.90 | 0.00/1.00/0.87 | 0.00/0.75/0.88 | 0.00/0.21/0.89 | 0.05/0.03/0.92 | 0.23/0.00/0.90 | 0.77/0.00/0.88 | 0.99/0.00/0.88 | 1.00/0.00/0.90 | 1.00/0.00/0.86 | 1.00/0.00/0.90 | 1.00/0.00/0.88 |
| 0.001 | registered | 0.00/0.71/1.00 | 0.00/0.28/1.00 | 0.00/0.09/1.00 | 0.00/0.01/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.02/0.00/1.00 | 0.06/0.00/1.00 | 0.29/0.00/1.00 | 0.48/0.00/1.00 | 0.96/0.00/1.00 |
| 0.001 | noise only | 0.00/1.00/0.97 | 0.00/0.91/0.97 | 0.00/0.62/0.97 | 0.00/0.17/0.97 | 0.00/0.03/0.98 | 0.01/0.02/0.97 | 0.03/0.01/0.98 | 0.12/0.00/0.98 | 0.34/0.00/0.98 | 0.61/0.00/0.97 | 0.94/0.00/0.98 | 1.00/0.00/0.98 | 1.00/0.00/0.98 |
| 0.001 | fit only | 0.00/1.00/0.84 | 0.00/1.00/0.83 | 0.00/0.93/0.84 | 0.01/0.44/0.83 | 0.04/0.14/0.86 | 0.08/0.08/0.83 | 0.16/0.03/0.82 | 0.50/0.00/0.84 | 0.81/0.00/0.81 | 0.95/0.00/0.85 | 1.00/0.00/0.82 | 1.00/0.00/0.85 | 1.00/0.00/0.85 |

The registered form has the least power and the only coverage that holds when the error is shared by the nodes. It was chosen for the reason under choice 4, not for its numbers, and the more powerful forms are rejected below.

## What the pilot predicts, branch by branch

- **All three outcomes are live.** On synthetic truths each fires: "not this" when the true excess is well under the crossing, "contributes" when it is well over, "spans" near the crossing. Which one fires grows with the true excess at every residual scale tried, and no branch is fixed by the design.
- **What decides the resolution is Prague's own smoothness,** through s_k and s_15. The envelope's median width is 0.0032, 0.0066, 0.0155 and 0.0183 at residual rms 0.0001, 0.0003, 0.001 and 0.002, mostly from the extraction term: |∇_k| is about 3.7 when the two-point law is exact and about 1.4 when the guard is active, and |∇⁰_k| about 0.8. So "contributes" for an excess near the paper's needs Prague's residual scale near 0.0001. At #55's pilot scale (rms 0.0008, with SPCTRL2's shape; Prague's own shape may fit better), an excess near 0.0121 would most likely read "spans", "contributes" would need an excess near 0.018 or more, and "not this" an excess near 0.002 or less. That is a limit set by Prague's own data, stated before they exist; it is not a branch the design closes.
- **At the paper's gas** (true excess 0.0121), P(contributes) is 0.98, 0.23, 0.02 and 0.00 at residual rms 0.0001, 0.0003, 0.001 and 0.002. An excess of 0.0121 is no prediction of Prague's: Prague's gas is what this issue measures.
- **"Mixed"** needs an end of the envelope between the lowest crossing (sun -1, 0.009444) and the highest (sun +1, 0.009527). In the base trials it came out in 0.6 percent of trials (at most 3.3 percent in a cell, near an excess of 0.012 to 0.013 at rms 0.0003).
- **Refusals** come only from a fit that does not converge. In 78,000 synthetic trials there were none.
- **Outside the conditions** (a Rayleigh off by 2 percent, a smooth error aloft, a different ozone shape at 0.34 km) the verdict can be wrong without any sign in the printed numbers. That is why each condition is stated with the verdict.

## What the tool prints

`report --aloft` (a new flag, like `--offset` and `--depth`; `--sun` is not used) prints, under a heading containing "#57":

1. the 132 transmittances as y_h(c, V_i), to five decimals, one table per altitude;
2. for each k: r(380), r(420), the regime ("exact" or "guard"), α and β; O(c) for the channels from 460 nm up, and Ō;
3. s_15, and for each node its fitted D at 550 nm (the constant coefficient): how much Prague's aerosol aloft changes with visibility, a diagnostic that rules nothing;
4. for each k: a, 1 / a, G_k, s_k, the residual rms sqrt(SSR / 60), the condition number of J (the 2-norm condition number of the unscaled analytic Jacobian at the solution, as in #55), SE_fit,k, |∇_k|, SE_ext,k, |∇⁰_k|, SE_0,k, SE_k, g_k, g_k - 2 SE_k and g_k + 2 SE_k;
5. g_lo and g_hi; and g_seg, the excess at Prague's 59.4 km node, ȳ(V_p) - ȳ_0.34(59.4) with ȳ_0.34(59.4) the 59.4 km node's y_0.34 interpolated at 550 nm: up to g_seg, Δ_M is nearly linear in the excess (Prague interpolates linearly in visibility between 40 and 59.4 km); past it the linear scaling is an assumption;
6. at each sun, g*_s and the outcome, then the verdict line.

Numbers are printed to five significant figures unless stated, computed from unrounded values. The report writes no file, and the four committed fixtures are not regenerated or edited (`fixture --check` for each must still pass). The tool's estimator and the power check's `estimate` must agree on g_lo and g_hi to 1e-9 on any input, and a test checks that on synthetic spectra.

## What would tell instead

- The paper's own gas, computed per channel from its cross-sections (Gorshelev et al. at 223 K, a table not in hand).
- A reference with v1's own atmosphere (libRadtran, run offline; GPL, never vendored or imported).

## What is not decided here

- Which model is right: both are models.
- The depths compared are Prague's above its 0.34 km observer and v1's whole column above its ground observer, as in #34, #49 and #55: the air above the two observers matches (#49, rule 2).
- Whether v1 should change. A change to v1 for any of this is a new renderer with its own registration (#50), never a change to v1.
- The carrying outcome (above).

## Decisions, each with its rejected alternative

- **The route is #57's, with the five choices registered.** Rejected:
  - a smoothly varying aerosol shape across the nodes, or a basis of the paper's species (WASO, INSO, SOOT) with fitted mixing (#57's optional parts). Both would assert a model the paper does not state, and with the Rayleigh fixed the curved per-node fit is well conditioned without them;
  - a free Rayleigh. The gas cannot then be told from a curved aerosol (#55).
- **The error carries s_15 / √6 for the extraction's noise, plus s_k for an error shared by the nodes at either altitude.** Rejected:
  - the six-node fit's SE alone (the day-5 estimator, "fit only"). Its envelope misses the truth in up to one trial in five even when the noise differs between the nodes;
  - the extraction's noise without the shared tolerance ("noise only"). It assumes Prague's error averages down over the nodes, which a reconstruction shared across configurations need not do, and its coverage falls when the error is shared;
  - a fixed numeric tolerance, either the supplement's 0.014 (a dataset-wide maximum; it would make the verdict "spans" whatever the sky) or a smaller number chosen with the power tables in view.
- **The aerosol aloft is removed by the guarded two-point power law.** Rejected: a fixed exponent of 1 (a prior the paper does not give, biased when the truth is steep); no removal (biased by +0.0032).
- **The verdict compares the envelope with #55's linearly scaled crossing, not with Δ_M carried into Prague at the envelope's ends.** The power check can then check the rule as registered, and no new matched-depth search is needed. Inside Prague's 40 to 59.4 km segment the two agree to the few percent by which Δ_M departs from linear there (#55's confirming read: 1 to 3 percent), about 0.0003 in the excess at the crossing. Rejected: carrying the ends into Prague, whose verdict the power check could not reproduce.
- **The six aloft spectra are averaged before the extraction.** Prague's ozone above 15 km does not depend on visibility, and averaging divides the part of the noise that differs between the nodes by √6. Rejected: the clearest node alone (it discards five spectra), and a separate extraction at each node (six noisier two-point fits).
- **The spectra go into a committed fixture written by `fixture --aloft`, and `report --aloft` reads only fixtures.** So CI can run the report, as it runs `report --offset` and `report --depth`. Rejected: evaluating Prague inside the report.
- **The power check is committed with this file, before any six-node or high-altitude number exists.** #57's acceptance asks for the rules, the power check included, on `main` first. Rejected: committing it with the build, as the day-5 note on #57 planned.
