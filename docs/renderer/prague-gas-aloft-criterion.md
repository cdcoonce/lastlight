# Prague's aerosol amount, from Prague's own ozone aloft and the paper's Rayleigh (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-17, for issue #57.

At that commit Prague's zenith transmittance has been recorded at two visibilities only, 123.8187 and 49.9864 km, both between its nodes and both at the 0.34 km observer (the same eleven numbers in all four committed fixtures). #49's matched-depth search also evaluated it at 0.34 km, starting at Prague's clearest and haziest nodes, and recorded two results: the 0.1 load's match at 55.5320 km, and the 0.03 load's clamped match at 131.8 km. That clamped match puts one node quantity in hand: Prague's -ln T at 131.8 km, interpolated at 550 nm, is 0.15893 (`docs/renderer/prague-aerosol-depth-criterion.md`). The 0.1 load's match gives the same quantity between nodes, 0.22208 at 55.532 km. No channel of Prague's transmittance at any visibility node has been recorded or printed, and nor has any transmittance at an observer altitude other than 0.34 km (the fixtures' altitude-0 reading is radiance, not transmittance). Nothing under this file computes one before it lands. The builder implements in `tools/prague_reference.py` exactly what this file says the tool prints; a disagreement between this file and the tool is a defect in the tool.

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

**The question.** At the 0.1 load's visibility (Koschmieder, V_p = 49.9864 km), is Prague's aerosol deeper than v1's 0.1 by enough that matching it removes more than #49's bar? #55 put the crossing, the excess at which the amount's part Δ_M equals the bar, at g* = 0.0094 at sun -1 and 0.0095 at sun 0, +1 and +10 (`report --depth`), and showed that the paper's gas computed with v1's cross-sections puts the excess at 0.0121. In gas terms: is Prague's gas depth at 550 nm, channel-interpolated, below ȳ(V_p) - 0.1 - g*, about 0.1263? Which side it falls on turns mostly on Prague's ozone near 540 to 580 nm. #55 could not take that from SPCTRL2's table, the stand-in that can by itself move a single gas estimate by several times the margin (its problem 3).

**The route.** The Rayleigh is the paper's, fixed, within a registered tolerance. The ozone's spectral shape is Prague's own: its zenith transmittance seen from 15 km, Prague's highest observer altitude, minus the paper's Rayleigh there, minus a power-law aerosol aloft fitted through the two channels where ozone barely absorbs, with an allowance for the ozone it does absorb there. The ozone's amount at 0.34 km is fitted across Prague's six visibility nodes, with the Rayleigh fixed and a curved aerosol at each node. The gas's error carries five things: the six-node fit's error, the extraction's error, a tolerance for an error in Prague's own transmittance that is shared by the six nodes (at 15 km and at 0.34 km), the Rayleigh tolerance, and the allowance for the ozone at 380 and 420 nm. The verdict compares the resulting envelope of the excess with the crossing, at each sun.

## Definitions

**Prague's spectra.** For each visibility node V_i in (20, 27.6, 40, 59.4, 90, 131.8) km and each of two observer altitudes h, 0.34 km and 15 km:

- T_h(c, V_i) is Prague's transmittance in channel c, computed as the tool computes `zenith_transmittance` for the fixtures: `evaluate` with the full dataset, `mode="transmittance"`, the zenith query (0, 90), ground albedo 0.2 and sun elevation 10 degrees (transmittance depends on neither), at visibility V_i and `altitude_m` 340 or 15000. Each V_i is a node of the dataset's visibility axis, and 15000 m is the last of the altitudes at which its transmittance is computed ("the same set of altitudes as the in-scattered radiance model", supplement §2). `evaluate` already refuses a viewpoint altitude the model reports differently from the one asked for. The transmittance's own interpolation clamps silently on its own axes, which are private to the reference implementation (`altitudesTrans` and `visibilitiesTrans`, stored as float32). By the paper's statement just quoted, the radiance altitude axis stands for the transmittance's; no statement covers the visibility axis. The tool reads the dataset's radiance axes with `read_axes`, as the other fixtures record them, and refuses an altitude above their highest altitude, or a V_i outside their visibility range, by more than 1e-6 relative. (The model does not clamp the altitude at the top, so this check is not redundant.) If a V_i is not also a node of the transmittance's own visibility axis, the model interpolates there; that is not checked.
- y_h(c, V_i) = -ln T_h(c, V_i). Every fit below uses the ten channels centred 380, 420, ..., 740 nm. The 340 nm channel is left out, as in #55: the paper's ozone absorbs in the Huggins band there.
- The 132 transmittances (eleven channels, six nodes, two altitudes), stored as T to nine significant figures as the other fixtures store theirs, are written to a new committed fixture, `tests/fixtures/prague/aloft.json`, by `fixture --aloft`, with its provenance (the dataset's SHA-256, the reference commit, the dataset's radiance axes from `read_axes`, and the viewpoint altitude the model reported for each evaluation). `fixture --aloft --check` rebuilds it and compares, as `fixture --check` does for the other four. `report --aloft` reads that fixture and the four committed ones, and runs no Prague evaluation.

**Constants.**

- x_c = ln(λ_c / 550 nm).
- p_0.34 = 1013.25 (1 - 0.0065 × 340 / 288.15)^5.25588 hPa, unrounded (973.0664; the tool's `PAPER_PRESSURE_HPA`), and p_15 = 121.11 hPa exactly, as written: the U.S. Standard Atmosphere's pressure at a geometric altitude of 15 km, truncated (Prague's altitudes are geometric; the 1976 formulas give 121.117 to 121.118; 120.45 hPa, 0.55 percent lower, is the pressure at a geopotential altitude of 15 km). A builder must use the literal: computing it instead moves g_k by up to about 2.5e-6, and g_lo and g_hi past the 1e-6 agreement in 66 of 180, 47 of 60 and 186 of 240 trials in three readers' reproductions (by up to about 1.9e-4), through the finite-difference gradient.
- R_paper = `rayleigh_optical_depth(550, p_0.34)` (0.093050), ρ_c = `rayleigh_optical_depth(λ_c, p_0.34)` / R_paper, and R15_c = `rayleigh_optical_depth(λ_c, p_15)`.
- For any per-channel quantity v, v̄ is v interpolated linearly at 550 nm between the 540 and 580 nm channel centres: v̄ = 0.75 v(540) + 0.25 v(580). ρ̄ = 1.00968.
- ȳ(V_p) is Prague's -ln T at V_p = 49.9864309567 km and 0.34 km, interpolated linearly at 550 nm between channel centres, from the committed fixtures (the tool's `y_bar_p`, 0.23575).
- g*_s is #55's crossing at sun s, as `report --depth` computes it (`depth_crossing`), unrounded.
- K = (0.99, 1.00, 1.01), the Rayleigh factors, and M = (0, 1, 2), the ozone-tail multipliers. The nine members (k, m) of K × M each give one gas estimate.

**Step 1, Prague's ozone aloft.** For each member (k, m):

- Y(c) is the mean of y_15(c, V_i) over the six nodes, and r(c) = Y(c) - k R15_c.
- **The power law.** From two residuals z_380 and z_420, the aerosol aloft is β (λ / 550 nm)^-η:
  - if both are positive and η_2 = ln(z_380 / z_420) / ln(420 / 380) lies in [0, 3], then η = η_2 and β = z_380 (380 / 550)^η, exact through both points ("exact");
  - otherwise η is ln(max(z_380, 1e-12) / max(z_420, 1e-12)) / ln(420 / 380) clipped to [0, 3], and β is the least-squares amplitude on the two points, (w_380 z_380 + w_420 z_420) / (w_380² + w_420²) with w_λ = (λ / 550)^-η, floored at 0 ("guard").
- **The ozone tail.** The ozone at 380 and 420 nm is t_380 = m ω_380 Ō and t_420 = m ω_420 Ō, with fixed ratios to the ozone at 550 nm: ω_420 = κ(460)² / (κ(500) κ(550)) and ω_380 = κ(460)³ / (κ(500)² κ(550)), with κ the tool's `ozone_absorption_per_atm_cm` (SPCTRL2's coefficients at 460 and 500 nm continued log-linearly to 420 and 380 nm, the channels being 40 nm apart, over its coefficient at 550 nm: ω_420 = 0.014118 and ω_380 = 0.0028235, unrounded). If Ō is not positive, the tail is 0.
- **The passes.** Start with t_380 = t_420 = 0. Each pass takes the power law from z_380 = r(380) - t_380 and z_420 = r(420) - t_420, sets O(c) = r(c) - β (λ_c / 550)^-η for the channels from 460 nm up, computes Ō, and from it the new t_380 and t_420. The passes stop at the first pass whose new t_380 and t_420 each differ from the old by at most 1e-15; that pass's O, with O(380) = t_380 and O(420) = t_420 (the new values), is the result, and its regime ("exact" or "guard") is printed. A tail that has not settled in 200 passes is a refusal. For m = 0 the first pass settles.
- O is Prague's ozone depth above 15 km by channel under member (k, m), and Ō its value interpolated at 550 nm.

**Step 2, the six-node fit.** For each member (k, m), with that member's O:

- The model is f(c, V_i) = k R_paper ρ_c + a O(c) + exp(u_i - α_i x_c + γ_i x_c²). It has 19 parameters: a, the ozone at 0.34 km as a multiple of the ozone above 15 km, and u_i, α_i and γ_i for each node's aerosol (its depth at 550 nm is exp(u_i)).
- The fit is unweighted least squares of f - y_0.34 over the 60 values, with `scipy.optimize.least_squares` and #55's settings: method `trf`, the analytic Jacobian, `xtol`, `ftol` and `gtol` all 1e-15, at most 20000 evaluations, every other argument at its default. The start is a = 1, and at each node u_i = ln max(ȳ_i - (k R_paper ρ̄ + Ō), 0.001), α_i = 1.3 and γ_i = 0, where ȳ_i is y_0.34(·, V_i) interpolated at 550 nm. A fit with `status` ≤ 0 is a refusal: the tool prints which member failed to standard error and exits 1.
- G_k = k R_paper ρ̄ + a Ō is the gas depth at 550 nm.
- SSR is the fit's sum of squared residuals and J its analytic Jacobian (60 by 19) at the solution. s_k = sqrt(SSR / 41) is the residual scale (41 = 60 values - 19 parameters).
- SE_fit,k = sqrt(qᵀ C q), with C = s_k² (JᵀJ)⁻¹ at the solution and q = Ō for a and 0 for every aerosol parameter. It is infinite if `numpy.linalg.inv` raises or if qᵀ C q is not a positive finite number.

**Step 3, the node-to-node scale at 15 km.**

- D(c, V_i) = y_15(c, V_i) - Y(c).
- Each node's D(·, V_i) is fitted by linear least squares with 1, x_c and x_c², three coefficients per node, with `numpy.linalg.lstsq` (`rcond=None`).
- s_15 = sqrt(SSR_15 / 35), where SSR_15 is the sum of squared residuals over the 60 values and 35 = 60 - 10 - 15 (the ten channel means and the fifteen independent node coefficients, since the six nodes' differences sum to zero).

**Step 4, the extraction's error and the tolerance for a shared error.** For each member (k, m):

- ∇_k(c) = ∂G_k / ∂Y(c), by central differences: steps 1 and 2 (all passes, the tail recomputed from no tail) are repeated with Y(c) raised by 1e-5 and lowered by 1e-5, each perturbed fit starting at the unperturbed fit's solution, and ∇_k(c) is the difference of the two G_k over 2e-5.
- ∇⁰_k(c) = ∂G_k / ∂y_0.34(c, ·), the same way, with y_0.34(c, V_i) raised and lowered by 1e-5 at all six nodes at once and O held at its unperturbed value (step 2 only).
- A perturbed fit with `status` ≤ 0 is a refusal.
- SE_ext,k = |∇_k| sqrt(s_15² / 6 + s_k²), with |∇_k| the Euclidean norm over the ten channels. The first term is the noise of the six-node mean aloft spectrum, from its own scatter (choice 3 below). The second is the tolerance for an error in Prague's transmittance at 15 km that is the same at all six nodes (choice 4).
- SE_0,k = |∇⁰_k| s_k, with |∇⁰_k| the Euclidean norm of ∇⁰_k over the ten channels: the same tolerance for an error at 0.34 km shared by the six nodes, which the six-node fit's SE, treating the 60 values as independent, does not carry (choice 4).
- SE_k = sqrt(SE_fit,k² + SE_ext,k² + SE_0,k²). A non-finite g_k (step 5) or a SE_k that is not a number is a refusal; an infinite SE_k is not, and gives "spans".

**Step 5, the envelope.**

- g_k = ȳ(V_p) - G_k - 0.1, Prague's aerosol depth at V_p minus v1's, under member (k, m) (the subscript k stands for the member throughout).
- g_lo = min over the nine members of (g_k - 2 SE_k), and g_hi = max over the nine members of (g_k + 2 SE_k).

**Step 6, the verdict.** At each sun s in (-1, 0, +1, +10):

- **contributes** if g_lo > g*_s;
- **not this** if g_hi ≤ g*_s;
- **spans** otherwise.

The tool prints one line per sun, `sun <s>: g* <g*_s>, <outcome>`, with `<s>` written -1, 0, +1 and +10 in that order, then the verdict line: `verdict: <outcome>` if all four suns agree, and otherwise `verdict: mixed`. g*_s differs between the suns by less than 0.0001, so "mixed" needs g_lo or g_hi to fall between them.

What each outcome says:

- **contributes**: at every excess in the envelope, the amount's part Δ_M, scaled linearly from #49's matched reading as g* is, is over the bar: matching Prague's aerosol depth would remove more than #49's bar from v1's offset.
- **not this**: at every excess in the envelope it is at or under the bar.
- **spans**: the envelope holds excesses on both sides of the crossing: the error is too wide to tell, or the excess sits near the crossing.

"Carries", #49's fourth outcome, is not ruled. Its excesses (0.0242 and up at linear scaling) lie past the segment where Δ_M is known to be nearly linear (which ends near an excess of 0.023, #56's confirming read), and g_lo ≥ 0.0242 would need a gas of 0.1116 or less, against the paper's 0.1237.

## The six design choices, each with its reason

The day-5 power study (below) found five places where, left open, the design rather than the sky would set the verdict. Cold read 2 of this file's pull request found a sixth, the ozone at 380 and 420 nm. Each is registered here with a reason that does not depend on the pilot. Each also makes the verdict conditional on something Prague cannot show by itself, and that is said with it.

**1. The Rayleigh: the paper's, within ±1 percent, as an envelope.** The paper's Rayleigh is Bodhaine et al.'s formula for N2 and O2 only ("only takes N2 and O2 into account", main article §4.2; supplement §3.1, Equation 23), over the U.S. Standard Atmosphere. The tool's `rayleigh_optical_depth` uses Peck and Reeder's refractive index for standard air with Bodhaine's King factor and a column set by pressure and standard gravity, which equals cold read 1's reconstruction of Bodhaine's formula with a standard-gravity column (0.0969 at 550 nm and 1013.25 hPa; Bucholtz's 0.0973 is 0.4 percent higher). The known differences between the paper's and the tool's:

- leaving out argon and CO2 lowers the depth by about 0.9 percent (their share of air's scattering at 550 nm, from each gas's refractive index and King factor; cold read 1: 0.88 to 0.90);
- a column that allows for gravity falling with height is larger than the tool's pressure-over-standard-gravity column, by 0.2 percent above 0.34 km and 0.5 to 0.7 percent above geometric 15 km (0.18 and 0.52 by Bodhaine's mass-weighted height, 0.24 and 0.67 by integrating the U.S. Standard column; cold read 1);
- whatever atmo_sim's own implementation of Bodhaine's Equation 23 does differently, which the files in hand cannot show.

Together the first two put the paper's Rayleigh about 0.7 percent under the tool's above 0.34 km and about 0.2 to 0.4 percent under it above 15 km, inside ±1 percent at both. The tolerance enters as an envelope over three fixed factors, never as a free parameter: with the Rayleigh free, the gas cannot be told from a curved aerosol (#55, estimates 5 and 7). One factor k applies at both altitudes, since the composition and the cross-section are the same there. **Conditional on:** the paper's Rayleigh lying within ±1 percent of the tool's at both altitudes, by a factor common to the two altitudes within about 0.5 percent. Outside the tolerance the verdict follows the assumed Rayleigh instead of the sky (the study's Rayleigh axis: P(contributes) stays flat while the true excess grows), and at rms 0.0003 each 1 percent moves the 50 percent point by about 0.0008 (the study) to 0.0010 (the power check below: 0.0159 and 0.0118 against 0.0138 for a Rayleigh 2 percent under and over). At rms 0.001 the shifts are not proportional: +0.0007 for a Rayleigh 2 percent under and -0.0025 for 2 percent over (0.0199 and 0.0167 against 0.0192), with the guard active in 0.64 and 0.35 of trials against 0.46. The 0.3 to 0.5 percent difference between the two altitudes is carried by neither the envelope nor the error; in the power check, a truth whose 15 km Rayleigh is 0.5 percent over its 0.34 km factor leaves the 50 percent point of "contributes" where it was (0.0124 and 0.0138 at rms 0.0001 and 0.0003), and cold read 1 found the same at 1 percent. **Also conditional on** Prague's channels being values at their centre wavelengths, as the tool evaluates them, not 40 nm band averages. The paper speaks both of "11 wavelengths" and of "fairly broad spectral bands" (main article §4.3, §4.4). Band averages over 40 nm would raise the Rayleigh by 1.08 percent at 380 nm and 0.26 percent at 740 nm against centre values (+0.59 and -0.22 percent relative to 550 nm), which no common factor removes; in the power check, band-averaged truths gave no false outcome and a coverage of 1.000 at every scale, with 50 percent points within 0.0013 of the base truths' (0.0123, 0.0135 and 0.0179 against 0.0124, 0.0138 and 0.0192).

**2. The aerosol aloft: removed, by a two-point power law with its exponent guarded to [0, 3].** Ignoring it biases G by +0.0032 and gives false "not this" in 0.57 and 0.50 of trials at an excess of 0.011, at rms 0.0003 and 0.001 (the study's estimator ENO). Only 380 and 420 nm are nearly free of ozone among the channels used, so two points determine the power law and nothing checks it. The ozone they do hold is choice 6. The guard keeps η inside the range of real aerosol Ångström exponents: about 0 for coarse dust, up to about 2.5 for fine smoke and sulfate. A residual at or under 0, or an exponent outside the guard, means the two points cannot be fitted exactly by a physical power law, and the clipped least-squares fit is the nearest one that is. The paper's aerosols are WASO, INSO and SOOT in every configuration (main article §4.2), so nothing fixes the exponent aloft. A fixed exponent of 1 halves the extraction's noise but needs a prior the paper does not give; with a true exponent of 1.5 to 2.5 it biases G by -0.0011, against 0.0000 for the guarded two-point law (the study's F2). **Conditional on:** Prague's aerosol aloft following a power law between 380 and 420 nm and on to 740 nm. The study's variants (an exponent of 0 to 0.5 or 1.5 to 2.5, an amount that varies by ±20 percent across the nodes, a depth aloft up to 0.02) are in the power check below.

**3. The extraction's error: in the envelope, from the aloft spectrum's own scatter.** The six-node fit's SE treats Prague's ozone shape as exact. In the study it is not: the shape extracted from a noisy aloft spectrum adds an error that the fit's 2SE omitted, so the envelope covered the truth in only 92 percent (rms 0.0003) and 81 percent (rms 0.001) of trials. The registered term propagates the noise of the six-node mean aloft spectrum, s_15 / √6 per channel, through the whole extraction and fit, by the gradient |∇_k|. s_15 is measured from Prague's own six aloft spectra after each node's smooth difference from their mean is removed, so it is the part of Prague's error that differs between the nodes and is not smooth across the channels. Most of |∇_k| comes from 380 and 420 nm, which the power law extrapolates from (the study's J1: dG/dy of 1.36 and -2.51 there, under 0.8 elsewhere).

**4. Prague's own error, where the six nodes share it: a tolerance at the scale of what the six-node fit leaves, at both altitudes.** Prague's transmittance is not stored; it is a rank-12 reconstruction of √T from one singular value decomposition per altitude across every atmosphere configuration and wavelength, with "a maximum absolute error over the dataset of 0.014" (supplement §2). At 15 km the six visibility nodes differ only by their thin aerosol aloft, so part of the reconstruction error there is likely to be shared by the six nodes. Averaging over the nodes does not reduce that part, and s_15 cannot see it (the part that differs between the nodes, s_15 does see). The same holds at 0.34 km, where the six-node fit's SE treats its 60 values as independent and so divides a shared error by √6 that it should not.

The registered tolerance is s_k per channel at both altitudes. That is a registered scale, not a measured bound, and its reasons are these:

- s_k is the only measure of Prague's reconstruction error available when the rule runs. It is measured at 0.34 km, and nothing in the reconstruction ties the error at 15 km to it: each altitude has its own decomposition.
- A shared error at 15 km in the channels from 460 nm up changes the extracted ozone shape. The 0.34 km data then fit that shape worse, which raises s_k. So the tolerance grows with the kind of error it is for. Cold read 2 checked this: with a shared 15 km error at two and three times the 0.34 km noise, coverage stayed at 0.970 or more, with at most one false outcome in 200 trials, and the envelope widened from 0.0066 to 0.0080 and 0.0093. The power check carries the same test: a shared 15 km noise of three times the 0.34 km noise gave no false outcome in 3,900 trials, and a coverage of 0.987 at rms 0.0001 and 0.0003.
- What s_k cannot see is a shared error confined to 380 and 420 nm. Choice 6's allowance covers one such error, a positive ozone tail proportional to Ō in a fixed ratio between the two channels. Any other shared error there is bounded only by this condition.

The dataset-wide maximum of 0.014 is not used. It bounds the worst point of every configuration, altitude and distance, not the zenith from 15 km, a path of optical depth about 0.04 at 550 nm. As a tolerance it would put the envelope's half-width near 0.09 (twice 3.1 times 0.014), so the verdict would be "spans" whatever the sky. The scale of one s_k was not varied with power numbers in view. **Conditional on:** Prague's error shared by the six nodes, at either altitude, being no larger than s_k per channel, taken as independent from channel to channel. The reconstruction may well make it smooth across the channels instead. A smooth shared error can move the extracted shape while the curved aerosols at 0.34 km absorb most of its trace. In the power check such an error (a tilt in log wavelength reaching ±0.001 at the ends of the band, the same at all six nodes) gave 3 false outcomes in 1,950 trials at rms 0.0001 (coverage 0.973) and none at 0.0003 (coverage 0.993). A larger or differently shaped one could do more. An error with the worst sign in every channel would move G by up to the sum of |∇_k(c)|, about twice |∇_k| (the study's J2: 6.6 against 3.1). Neither is carried by the envelope.

**5. One ozone shape at both altitudes.** The paper uses one ozone cross-section, Gorshelev et al.'s at 223 K (main article §4.2), at every height, and its simulator takes absorption into account for O3 only ("only take absorption into account for O3", supplement §3.1, describing atmo_sim). So the ozone above 15 km and the ozone above 0.34 km have the same spectral shape in Prague by construction, and the shape from 15 km may stand for the shape at 0.34 km. **Conditional on:** that being so in the dataset as built. A shape at 0.34 km tilted by ±3 percent per 100 nm against the shape aloft moves the 50 percent point of "contributes" by +0.0012 and +0.0002 at rms 0.0001 (+3 and -3 percent) and by +0.0004 and 0.0000 at rms 0.0003, with no false outcome, in the power check.

**6. The ozone at 380 and 420 nm: an allowance tied to the ozone at 550 nm, as an envelope over 0, 1 and 2 times a fixed ratio.** The paper's ozone is Gorshelev et al.'s at 223 K. Unlike SPCTRL2's table, which is 0 at 440 nm and below, it absorbs across the onset of the Chappuis band at 400 to 440 nm and weakly at 380 nm. Without an allowance, step 1 reads that ozone as aerosol aloft. The extracted ozone at 460 nm and above then comes out too small, so the gas comes out too small and the excess too large. That pushes the verdict toward "contributes", and nothing in s_k or s_15 sees it, because the error is shared by the nodes and confined to the two channels the power law rests on. Cold read 2 put the bias at +0.0007 for a 420 nm coefficient of 8.1e-4 per atm cm (+0.0004 to +0.0013 for half and double that, +0.0010 with band-averaged channels), and found false "contributes" at rms 0.0001.

Gorshelev's table is not in hand, so the tail's size relative to the band is taken from SPCTRL2's coefficients at 460 and 500 nm continued log-linearly: a 420 nm coefficient of 1.2e-3 per atm cm, 0.014 of the 550 nm one. That sits inside the reader's range from memory (about 4e-4 to 1.6e-3), and twice it covers the top of that range. The envelope runs over 0, 1 and 2 times the ratio, because the onset's real shape is not known to be log-linear: it may be steeper (less ozone at 420 nm) or softer (more). The tail is tied to Ō, the ozone at 550 nm, because Ō is large and changes little when the aerosol aloft changes, so the passes settle fast on one tail.

A first form, disclosed: the tail continued log-linearly from Prague's own O(460) and O(500) after one pass. In the rerun power check it failed at its own boundary. A truth at twice the continuation gave false outcomes in 51 of 1,950 trials at rms 0.0001 (coverage 0.673). A first pass that reads the tail as aerosol leaves O(460) and O(500) too small, so the continuation is short. Iterating it to a fixed point did not cure this: with aerosol aloft present, a tail continued from 460 and 500 nm and a flat aerosol trade off through the two channels, so the fixed point is not unique and the iteration settled on too small a tail. Tying the tail to Ō instead converged in at most 12 passes in noise-free tests. There, the m = 1 member was unbiased within 0.0004 for a true tail of 1 times the ratio, and the m = 2 member overcorrected slightly for a true tail of 2 times (mean -0.00015).

**Conditional on:** the ozone at 380 and 420 nm lying on the allowance's line, in the ratio ω_380 : ω_420 (0.2) at 0 to 2 times ω_380 Ō and ω_420 Ō, or close to it: within 12 percent of the line's ratio and up to 2.15 times, as the power check's on-line tails (0.5, 1 and 2 times) are. The nine members span that line and nothing off it.

- **On the line.** In the power check, truths with 0.5, 1 and 2 times the log-linear continuation of their own shape gave no false outcome and a coverage of 1.000 at rms 0.0001 and 0.0003. Their ratios to the rule's run from 1.38 to 2.11 times at 420 nm (1.22 to 2.15 at 380 nm) at the nominal 2, so the tail2 cell reaches slightly past the allowance's 2, and their 380 : 420 ratios lie within 12 percent of the line's, because the five ozone shapes differ at 460 and 500 nm. P(contributes) at an excess of 0.0121 was 0.473, 0.533 and 0.827 at rms 0.0001, against 0.380 with no tail. Cold read 3 found no false outcome in 3,500 trials at rms 0.00003 for tails on the line.
- **Off the line.** A tail of the allowance's size in another ratio can be missed. Cold read 3 put tails at the corners (0, 2) and (2, 0), the first number at 380 nm and the second at 420 nm, in units of ω_380 Ō and ω_420 Ō. Noise-free, the envelope missed the truth in up to 22 percent of trials, by up to 0.0004. Through the full rule, in the readers' scratch runs at excesses of 0.0085, 0.0090, 0.0094, 0.0095, 0.0100 and 0.0105 (100 trials each), corner (0, 2) gave 1 false outcome at rms 0.0001 and 5 to 12 at rms 0.00003, and corner (2, 0) gave none and 4 to 5. All of them fell at 0.0094 or 0.0095, within 0.0001 of a crossing. The committed rig has no corner cells. The power check's tail of another shape (8.1e-4 per atm cm at 420 nm and 0 at 380 nm, about 0.65 times ω_420 Ō at 420 nm) is off the line too, and gave no false outcome at rms 0.0001 and 0.0003 (P(contributes) at 0.0121 of 0.513 at 0.0001). Widening the members to the corners was not done: it would be chosen with these numbers in view.
- **Outside the allowance.** A truth at 3 times gave 15 false outcomes in 1,950 trials at rms 0.0001 (coverage 0.733) and none at 0.0003.
- **The residual scales tested.** The power check covers rms 0.0001 to 0.002. Below 0.0001, cold read 3's runs at 0.00003 are the only evidence: none false on the line, the corner results above off it.

**When the multipliers were fixed, and what they cost.** M = (0, 1, 2) was fixed with the first form, before any power number of a tail allowance existed, and kept when the tail was tied to Ō. The first form's 51 false outcomes in 1,950 trials came from the second run of the rig, with that form; the committed rig no longer holds it. Its cost is part of the floor described under "What the pilot predicts": noise-free, the envelope's low end sits a median 0.0023 below the truth when there is no tail, against about 0.0016 to 0.0017 with M = (0, 1) (cold reads 3 and 4), so the m = 2 member adds about 0.0006 to 0.0007.

**The verdict's conditions, together.** The verdict holds if:

1. the paper's Rayleigh is within ±1 percent of the tool's at both altitudes, by a factor common to the two altitudes within about 0.5 percent;
2. Prague's channels are centre-wavelength values;
3. its aerosol aloft follows a power law across the channels;
4. its error shared by the six nodes is no larger than s_k per channel at either altitude, and independent from channel to channel;
5. one ozone shape holds at both altitudes;
6. the ozone at 380 and 420 nm lies on the allowance's line, t_380 and t_420 in the ratio ω_380 : ω_420 at 0 to 2 times ω_380 Ō and ω_420 Ō, or close to it (within 12 percent in the ratio, up to 2.15 times);
7. Δ_M is linear in the excess near the crossing. It departs from linear by 1 to 3 percent inside the 40 to 59.4 km segment (#55's confirming read), about 0.0003 in the excess at the crossing, and this is not carried.

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
- not material: aerosol aloft that varies across the nodes, two-mode aerosol, curvature ±0.6, an aloft fraction of 0.6 to 1.0, a Rayleigh exponent off by ±0.05, correlated channel noise, and the start values. It also found a weak ozone tail at 380 to 440 nm not material, but its tail fell from 380 to 420 nm, shaped like aerosol; a tail shaped like the Chappuis onset is material (choice 6);
- its table G compared envelopes of 1, 2 and 3 SE and Rayleigh tolerances of ±1 and ±2 percent, with power numbers in view. The 2 SE and the ±1 percent registered here predate it (#55's drafted rule, and #57's text).

**The power check of the registered rule** is `tools/issue57_power.py`, committed with this file. It reads no Prague data. It runs exactly the estimator of steps 1 to 6, with each sun's unrounded g*_s (0.009444, 0.009483, 0.009527 and 0.009492 at sun -1, 0, +1 and +10) and ȳ(V_p) rounded to 0.235751. It prints how often each verdict line comes out at each true excess, the trials with a false outcome at some sun, the coverage (the share of trials whose envelope contains the true excess), the median width and the share of trials with the guard active, and it scores the two rejected forms of the error on the same trials.

Its truths are the day-5 study's:

- aerosol at each node of depth 6 / V_i times a factor from 0.8 to 1.25, Ångström exponent falling from 1.6 at 131.8 km to 0.8 at 20 km with ±0.15 of scatter, and curvature ±0.3;
- aerosol aloft of depth 0 to 0.005 at 550 nm and exponent 0.5 to 1.5;
- 0.80 to 0.90 of the ozone above 15 km;
- trials cycling through the five ozone shapes;
- δ_R uniform in ±1 percent unless a row says otherwise;
- white noise at the stated rms, independent between the nodes unless a row says otherwise.

Every cell is seeded from its name, so these two commands reprint the tables below: `uv run --frozen python tools/issue57_power.py --n 400 --cells base_s1,base_s3,base_s10,base_s20`, and the same with `--n 150 --cells` and the variant cells' names (`tools/issue57_power.py`'s `cells()`). Under this lockfile, the two took 2.3 and 8.3 CPU-hours on the build machine.

**Headline, base truths.**

| residual rms | 50 percent point of "contributes" | 50 percent point of "not this" | P(contributes) at 0.0121 | median envelope width | trials with a false outcome | lowest coverage | share of trials with the guard active (member k = 1, m = 0) |
|---|---|---|---|---|---|---|---|
| 0.0001 | 0.0124 | 0.0076 | 0.380 | 0.0045 | 0 of 5200 | 1.000 | 0.10 |
| 0.0003 | 0.0138 | 0.0063 | 0.087 | 0.0075 | 0 of 5200 | 1.000 | 0.19 |
| 0.001 | 0.0192 | 0.0010 | 0.003 | 0.0179 | 0 of 5200 | 1.000 | 0.46 |
| 0.002 | 0.0215 | never (0.37 at 0) | 0.000 | 0.0244 | 0 of 5200 | 1.000 | 0.68 |

**The registered rule on the base truths** (400 trials per cell): P(contributes) / P(not this), by the true excess.

| residual rms | 0 | 0.004 | 0.006 | 0.008 | 0.009 | 0.0095 | 0.010 | 0.011 | 0.0121 | 0.013 | 0.015 | 0.018 | 0.022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.00/0.39 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.05/0.00 | 0.38/0.00 | 0.75/0.00 | 1.00/0.00 | 1.00/0.00 | 1.00/0.00 |
| 0.0003 | 0.00/1.00 | 0.00/0.99 | 0.00/0.60 | 0.00/0.04 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.09/0.00 | 0.25/0.00 | 0.91/0.00 | 1.00/0.00 | 1.00/0.00 |
| 0.001 | 0.00/0.59 | 0.00/0.22 | 0.00/0.08 | 0.00/0.01 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.01/0.00 | 0.14/0.00 | 0.35/0.00 | 0.84/0.00 |
| 0.002 | 0.00/0.37 | 0.00/0.04 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.01/0.00 | 0.20/0.00 | 0.54/0.00 |

What it shows:

- **Monotone in every cell.** In all 45 cells, P(contributes) never falls and P(not this) never rises between neighbouring true excesses, and each outcome fires somewhere on the grid. No outcome is fixed by the design.
- **No false outcome inside the conditions.** A "contributes" at a sun is false if the true excess is at or under that sun's crossing, and a "not this" if it is over. In every cell inside the conditions (the base truths, the shared noise at the tolerance, the aerosol aloft variants, the 15 km Rayleigh, and tails of 0.5, 1 and 2 times) there was none, at any residual scale tested. The lowest coverage there was 0.993.
- **Outside the conditions it can be wrong.** A Rayleigh 2 percent under gave false outcomes in 52 of 1,950 trials at rms 0.0001 (42 "not this" and 10 "mixed", at true excesses of 0.0095 and 0.010; coverage 0.613) and 3 at 0.0003. A tail of 3 times gave 15 at rms 0.0001 (coverage 0.733), and a smooth shared error aloft gave 3. Band-averaged channels, the tilted ozone shapes, a Rayleigh 2 percent over, a shared error at 3 times the tolerance and the tail of another shape (off the line) gave none; cold read 3's tails at the line's corners gave a few, within 0.0001 of a crossing (choice 6).
- **A shared error is what the shared tolerance is for.** With the 15 km noise the same at all six nodes, the nodes do not differ, so s_15 sees nothing and the "noise only" form's extraction term vanishes. Its coverage falls to 0.913 and 0.813 (rms 0.0001 and 0.0003), with 11 and 20 false outcomes in 1,950 trials each, and at three times that noise to 0.820 and 0.700, with 28 and 72. The registered form holds 0.993 at the tolerance and 0.987 at three times, with none. That the registered form holds at the tolerance is built in, since the rig sets the shared noise equal to it; the case at three times is not.
- **Refusals:** none in 100,750 trials.

The variants, 150 trials per cell. For comparison, the base truths' 50 percent points are 0.0124, 0.0138 and 0.0192 for "contributes" and 0.0076, 0.0063 and 0.0010 for "not this" at rms 0.0001, 0.0003 and 0.001, with P(contributes) at 0.0121 of 0.380, 0.087 and 0.003:

| variant | what differs from the base truth | rms | 50 percent point of "contributes" | 50 percent point of "not this" | P(contributes) at 0.0121 | trials with a false outcome | lowest coverage |
|---|---|---|---|---|---|---|---|
| dr_m2 | the truth's Rayleigh 2 percent under the paper's (outside condition 1) | 0.0001 | 0.0140 | 0.0093 | 0.000 | 52 of 1950 | 0.613 |
| dr_m2 | the truth's Rayleigh 2 percent under the paper's (outside condition 1) | 0.0003 | 0.0159 | 0.0078 | 0.000 | 3 of 1950 | 0.927 |
| dr_m2 | the truth's Rayleigh 2 percent under the paper's (outside condition 1) | 0.001 | 0.0199 | 0.0042 | 0.000 | 0 of 1950 | 0.987 |
| dr_p2 | 2 percent over (outside condition 1) | 0.0001 | 0.0108 | 0.0069 | 1.000 | 0 of 1950 | 0.900 |
| dr_p2 | 2 percent over (outside condition 1) | 0.0003 | 0.0118 | 0.0049 | 0.627 | 0 of 1950 | 1.000 |
| dr_p2 | 2 percent over (outside condition 1) | 0.001 | 0.0167 | nan | 0.013 | 0 of 1950 | 1.000 |
| topcommon | the 15 km noise the same at all six nodes | 0.0001 | 0.0125 | 0.0074 | 0.347 | 0 of 1950 | 0.993 |
| topcommon | the 15 km noise the same at all six nodes | 0.0003 | 0.0141 | 0.0059 | 0.093 | 0 of 1950 | 0.993 |
| topcommon | the 15 km noise the same at all six nodes | 0.001 | 0.0184 | 0.0018 | 0.007 | 0 of 1950 | 0.993 |
| topcommon3 | the same, at 3 times the 0.34 km noise (outside condition 4) | 0.0001 | 0.0136 | 0.0067 | 0.127 | 0 of 1950 | 0.987 |
| topcommon3 | the same, at 3 times the 0.34 km noise (outside condition 4) | 0.0003 | 0.0157 | 0.0045 | 0.080 | 0 of 1950 | 0.987 |
| asteep | aloft exponent 1.5 to 2.5 | 0.0001 | 0.0119 | 0.0079 | 0.587 | 0 of 1950 | 1.000 |
| asteep | aloft exponent 1.5 to 2.5 | 0.0003 | 0.0131 | 0.0068 | 0.213 | 0 of 1950 | 1.000 |
| aflat | aloft exponent 0 to 0.5 | 0.0001 | 0.0126 | 0.0074 | 0.280 | 0 of 1950 | 1.000 |
| aflat | aloft exponent 0 to 0.5 | 0.0003 | 0.0141 | 0.0063 | 0.093 | 0 of 1950 | 1.000 |
| amp20 | aloft amount ±20 percent across the nodes | 0.0001 | 0.0124 | 0.0075 | 0.387 | 0 of 1950 | 1.000 |
| amp20 | aloft amount ±20 percent across the nodes | 0.0003 | 0.0138 | 0.0062 | 0.053 | 0 of 1950 | 1.000 |
| bigaloft | aloft depth up to 0.02 at 550 nm | 0.0001 | 0.0133 | 0.0075 | 0.140 | 0 of 1950 | 1.000 |
| bigaloft | aloft depth up to 0.02 at 550 nm | 0.0003 | 0.0144 | 0.0062 | 0.020 | 0 of 1950 | 1.000 |
| tiltp | ozone shape at 0.34 km +3 percent per 100 nm against aloft (outside condition 5) | 0.0001 | 0.0136 | 0.0070 | 0.093 | 0 of 1950 | 1.000 |
| tiltp | ozone shape at 0.34 km +3 percent per 100 nm against aloft (outside condition 5) | 0.0003 | 0.0142 | 0.0062 | 0.020 | 0 of 1950 | 1.000 |
| tiltm | -3 percent per 100 nm (outside condition 5) | 0.0001 | 0.0126 | 0.0066 | 0.287 | 0 of 1950 | 1.000 |
| tiltm | -3 percent per 100 nm (outside condition 5) | 0.0003 | 0.0138 | 0.0053 | 0.113 | 0 of 1950 | 1.000 |
| r15 | the truth's 15 km Rayleigh 0.5 percent over its 0.34 km factor | 0.0001 | 0.0124 | 0.0077 | 0.353 | 0 of 1950 | 1.000 |
| r15 | the truth's 15 km Rayleigh 0.5 percent over its 0.34 km factor | 0.0003 | 0.0138 | 0.0065 | 0.087 | 0 of 1950 | 1.000 |
| smooth | a smooth error at 15 km shared by the nodes, ±0.001 at the band's ends (outside condition 4) | 0.0001 | 0.0128 | 0.0080 | 0.200 | 3 of 1950 | 0.973 |
| smooth | a smooth error at 15 km shared by the nodes, ±0.001 at the band's ends (outside condition 4) | 0.0003 | 0.0138 | 0.0068 | 0.080 | 0 of 1950 | 0.993 |
| tail05 | ozone tail 0.5 times the log-linear continuation of its own shape | 0.0001 | 0.0122 | 0.0071 | 0.473 | 0 of 1950 | 1.000 |
| tail05 | ozone tail 0.5 times the log-linear continuation of its own shape | 0.0003 | 0.0136 | 0.0052 | 0.113 | 0 of 1950 | 1.000 |
| tail1 | tail 1 times | 0.0001 | 0.0120 | 0.0069 | 0.533 | 0 of 1950 | 1.000 |
| tail1 | tail 1 times | 0.0003 | 0.0137 | 0.0049 | 0.087 | 0 of 1950 | 1.000 |
| tail1 | tail 1 times | 0.001 | 0.0193 | nan | 0.000 | 0 of 1950 | 1.000 |
| tail2 | tail 2 times its own shape's continuation (1.38 to 2.11 times the rule's ratio) | 0.0001 | 0.0115 | 0.0061 | 0.827 | 0 of 1950 | 1.000 |
| tail2 | tail 2 times its own shape's continuation (1.38 to 2.11 times the rule's ratio) | 0.0003 | 0.0130 | 0.0038 | 0.107 | 0 of 1950 | 1.000 |
| tail3 | tail 3 times (outside condition 6) | 0.0001 | 0.0105 | 0.0055 | 1.000 | 15 of 1950 | 0.733 |
| tail3 | tail 3 times (outside condition 6) | 0.0003 | 0.0121 | 0.0039 | 0.493 | 0 of 1950 | 0.987 |
| tail420 | tail of another shape: 8.1e-4 per atm cm at 420 nm, 0 at 380 nm (off the line, outside condition 6) | 0.0001 | 0.0121 | 0.0070 | 0.513 | 0 of 1950 | 1.000 |
| tail420 | tail of another shape: 8.1e-4 per atm cm at 420 nm, 0 at 380 nm (off the line, outside condition 6) | 0.0003 | 0.0137 | 0.0052 | 0.073 | 0 of 1950 | 1.000 |
| band | 40 nm band-averaged channels (outside condition 2) | 0.0001 | 0.0123 | 0.0076 | 0.387 | 0 of 1950 | 1.000 |
| band | 40 nm band-averaged channels (outside condition 2) | 0.0003 | 0.0135 | 0.0067 | 0.153 | 0 of 1950 | 1.000 |
| band | 40 nm band-averaged channels (outside condition 2) | 0.001 | 0.0179 | 0.0017 | 0.000 | 0 of 1950 | 1.000 |

With 150 trials, a 50 percent point is uncertain by about ±0.0002 and a share near 0.5 by about ±0.04.

**The form of the error was chosen with power numbers in view, against power.** Before it was fixed, a scratch run (200 trials per cell, an earlier form without SE_0,k and with one crossing of 0.0095) compared three forms on the same trials: the six-node fit's SE alone (the day-5 estimator), that plus the extraction's noise averaged over the nodes, and that plus a shared error aloft at the scale of the fit's residual. All three were monotone; over rms 0.0001 to 0.001 the coverage was 0.80 to 0.98, 0.96 to 1.00 and 0.99 to 1.00, and P(contributes) at an excess of 0.0121 was 0.99, 0.94 and 0.30 at rms 0.0003. The rig now prints the same comparison for the registered rule (the forms "fit only", "noise only" and "registered", on the same trials):

P(contributes) / P(not this) / coverage, 400 trials per cell:

| residual rms | form | 0 | 0.004 | 0.006 | 0.008 | 0.009 | 0.0095 | 0.010 | 0.011 | 0.0121 | 0.013 | 0.015 | 0.018 | 0.022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0003 | registered | 0.00/1.00/1.00 | 0.00/0.99/1.00 | 0.00/0.60/1.00 | 0.00/0.04/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.09/0.00/1.00 | 0.25/0.00/1.00 | 0.91/0.00/1.00 | 1.00/0.00/1.00 | 1.00/0.00/1.00 |
| 0.0003 | noise only | 0.00/1.00/0.99 | 0.00/1.00/0.99 | 0.00/0.98/0.98 | 0.00/0.43/0.99 | 0.00/0.04/0.99 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.08/0.00/0.99 | 0.36/0.00/0.99 | 0.76/0.00/1.00 | 1.00/0.00/0.99 | 1.00/0.00/0.99 | 1.00/0.00/0.99 |
| 0.0003 | fit only | 0.00/1.00/0.96 | 0.00/1.00/0.95 | 0.00/1.00/0.94 | 0.00/0.75/0.94 | 0.00/0.21/0.94 | 0.00/0.03/0.97 | 0.00/0.00/0.95 | 0.11/0.00/0.96 | 0.54/0.00/0.94 | 0.89/0.00/0.94 | 1.00/0.00/0.94 | 1.00/0.00/0.95 | 1.00/0.00/0.94 |
| 0.001 | registered | 0.00/0.59/1.00 | 0.00/0.22/1.00 | 0.00/0.08/1.00 | 0.00/0.01/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.00/0.00/1.00 | 0.01/0.00/1.00 | 0.14/0.00/1.00 | 0.35/0.00/1.00 | 0.84/0.00/1.00 |
| 0.001 | noise only | 0.00/1.00/0.98 | 0.00/0.86/0.98 | 0.00/0.59/0.98 | 0.00/0.17/0.98 | 0.00/0.03/0.98 | 0.00/0.02/0.98 | 0.00/0.01/0.99 | 0.02/0.00/0.99 | 0.15/0.00/0.99 | 0.30/0.00/0.98 | 0.81/0.00/0.98 | 1.00/0.00/0.98 | 1.00/0.00/0.99 |
| 0.001 | fit only | 0.00/1.00/0.91 | 0.00/1.00/0.93 | 0.00/0.93/0.90 | 0.00/0.44/0.91 | 0.00/0.14/0.94 | 0.00/0.08/0.91 | 0.03/0.03/0.91 | 0.14/0.00/0.92 | 0.46/0.00/0.93 | 0.78/0.00/0.92 | 0.99/0.00/0.91 | 1.00/0.00/0.93 | 1.00/0.00/0.93 |

The registered form has the least power and the only coverage that holds when the error is shared by the nodes. It was chosen for the reason under choice 4, not for its numbers, and the more powerful forms are rejected below.

## What the pilot predicts, branch by branch

- **All three outcomes fire on synthetic truths,** and which one fires grows with the true excess at every residual scale tried. No branch is fixed by the design.
- **The outcomes to expect from Prague are "contributes" or "spans".** "Not this" at even odds needs Prague's gas over the paper's (computed with SPCTRL2's ozone) by 0.0045, 0.0058 or 0.0111 at rms 0.0001, 0.0003 and 0.001: its ozone at 550 nm 15, 20 or 37 percent above SPCTRL2's. None of #55's eight estimates put the gas above the paper's 0.1237. So "not this" is live in the rule but would need an ozone far from the paper's.
- **What decides between those two is the design's floor and then Prague's own smoothness.** The nine members spread the estimate even with no noise. Noise-free, g_lo sits a median 0.0023 below the true excess with no tail, 0.0017 with a tail of 1 times and 0.0009 with 2 times, and g_hi above it by 0.0007, 0.0017 and 0.0026 (cold read 3, 300 trials; the largest shortfall with no tail was 0.0037 there and 0.0040 to 0.0043 in cold read 4's 600-trial runs). At rms 0.0001, at an excess of 0.0121, that spread is most of the width: 0.0033 of a median 0.0046, against 0.0014 for four SE (cold read 3, 90 trials). So no smoothness removes the floor. At rms 0.00003 (cold read 4, 600 trials per point), P(contributes) with no tail is 0.000 at 0.0100, 0.002 at 0.0105, 0.067 at 0.011, 0.180 at 0.0115 and 0.553 at 0.0121, with even odds near 0.0120. With a tail of 1 times it is 0.000, 0.042, 0.208, 0.520 and 0.840, with even odds near 0.0115. So with no tail or a tail of 1 times, "contributes" is rare below an excess of about 0.011 even with very smooth data. With a tail of 2 times, the top of the allowance, it comes sooner: P(contributes) 0.085, 0.33, 0.655 and 0.865 at 0.0100, 0.0105, 0.011 and 0.0115, with even odds near 0.0107 to 0.0108 (cold reads 5 and 6, 200 trials per point each). At rms 0.0001 and above, the noise adds to the floor: the median widths over all trials are 0.0045, 0.0075, 0.0179 and 0.0244 at rms 0.0001, 0.0003, 0.001 and 0.002, the extraction term growing with the noise. |∇_k| is about 3.7 with the two-point law exact and 1.4 with the guard active for m = 0, 3.2 and 1.2 for m = 1, and 2.9 and 1.2 for m = 2; |∇⁰_k| is about 0.8. At #55's pilot scale (rms 0.0008, with SPCTRL2's shape; Prague's own shape may fit better), an excess near 0.0121 would most likely read "spans". "Contributes" would need an excess of about 0.018 or more and "not this" one of about 0.0025 or less, interpolating linearly in rms between 0.0003 and 0.001. That is a limit set by the registered design and by Prague's own data, stated before the data exist.
- **At the paper's gas** (true excess 0.0121), P(contributes) is 0.380, 0.087, 0.003 and 0.000 at residual rms 0.0001, 0.0003, 0.001 and 0.002 on truths with no ozone tail, and 0.533 and 0.087 at rms 0.0001 and 0.0003 with a tail of 1 times the ratio. An excess of 0.0121 is no prediction of Prague's: Prague's gas is what this issue measures.
- **"Mixed"** needs an end of the envelope between the lowest crossing (sun -1, 0.009444) and the highest (sun +1, 0.009527). In the base trials it came out in 0.6 percent of trials, at most 5.0 percent in a cell (rms 0.0001, excess 0.008) and at most 2.8 percent at rms 0.0003.
- **Refusals** come from a fit or a perturbed fit that does not converge, a tail that does not settle in 200 passes, a non-finite g_k or a SE_k that is not a number, a V_i or altitude outside the dataset's radiance axes, and a reported viewpoint altitude other than the one asked for. In 100,750 synthetic trials there were none of the estimator's refusals; the last two belong to the fixture writer, which the synthetic trials do not exercise.
- **Outside the conditions** (a Rayleigh off by 2 percent, band-averaged channels, a tail over twice the ratio or off the allowance's line, a smooth error aloft, a different ozone shape at 0.34 km) the verdict can be wrong without any sign in the printed numbers. In the power check that happened for a Rayleigh 2 percent under, a tail of 3 times and a smooth shared error aloft, and in the readers' runs for tails at the line's corners (above). That is why each condition is stated with the verdict.

## What the tool prints

`report --aloft` (a new flag, like `--offset` and `--depth`; `--sun` is not used) prints, under a heading containing "#57":

1. the 132 transmittances as y_h(c, V_i), to five decimals, one table per altitude;
2. for each member (k, m): t_380, t_420, the number of passes, the last pass's regime ("exact" or "guard"), η and β; O(c) for every channel, and Ō;
3. s_15, and for each node its fitted D at 550 nm (the constant coefficient): how much Prague's aerosol aloft changes with visibility, a diagnostic that rules nothing;
4. for each member (k, m): a, 1 / a, G_k, s_k, the residual rms sqrt(SSR / 60), the condition number of J (the 2-norm condition number of the unscaled analytic Jacobian at the solution, as in #55), SE_fit,k, |∇_k|, SE_ext,k, |∇⁰_k|, SE_0,k, SE_k, g_k, g_k - 2 SE_k and g_k + 2 SE_k;
5. g_lo and g_hi; and g_seg, the excess at Prague's 59.4 km node, ȳ(V_p) - ȳ_0.34(59.4) with ȳ_0.34(59.4) the 59.4 km node's y_0.34 interpolated at 550 nm: up to g_seg, Δ_M is nearly linear in the excess (Prague interpolates linearly in visibility between 40 and 59.4 km); past it the linear scaling is an assumption;
6. the four sun lines and the verdict line, exactly as step 6 gives them.

Numbers are printed to five significant figures unless stated, computed from unrounded values. The report writes no file, and the four committed fixtures are not regenerated or edited (`fixture --check` for each must still pass). The tool's estimator and the power check's `estimate` must agree on g_lo and g_hi to 1e-6 on any input, and a test checks that on synthetic spectra (an implementation that is correct but orders its arithmetic differently differs by up to about 4e-8, cold read 1).

## What would tell instead

- The paper's own gas, computed per channel from its cross-sections (Gorshelev et al. at 223 K). The table is published (Serdyuchenko et al. 2014, *Atmos. Meas. Tech.* 7, 625) but not in hand here, and its use is #61's, with its own registration.
- A reference with v1's own atmosphere (libRadtran, run offline; GPL, never vendored or imported).

## What is not decided here

- Which model is right: both are models.
- The depths compared are Prague's above its 0.34 km observer and v1's whole column above its ground observer, as in #34, #49 and #55: the air above the two observers matches (#49, rule 2).
- Whether v1 should change. A change to v1 for any of this is a new renderer with its own registration (#50), never a change to v1.
- The carrying outcome (above).

## Decisions, each with its rejected alternative

- **The route is #57's, with the six choices registered.** Rejected:
  - a smoothly varying aerosol shape across the nodes, or a basis of the paper's species (WASO, INSO, SOOT) with fitted mixing (#57's optional parts). Both would assert a model the paper does not state, and with the Rayleigh fixed the curved per-node fit is well conditioned without them;
  - a free Rayleigh. The gas cannot then be told from a curved aerosol (#55).
- **The error carries s_15 / √6 for the extraction's noise, plus s_k for an error shared by the nodes at either altitude.** Rejected:
  - the six-node fit's SE alone (the day-5 estimator, "fit only"). Its envelope misses the truth in up to about one trial in seven inside the conditions (coverage 0.863), and more when the error is shared;
  - the extraction's noise without the shared tolerance ("noise only"). It assumes Prague's error averages down over the nodes, which a reconstruction shared across configurations need not do, and its coverage falls when the error is shared;
  - a fixed numeric tolerance, either the supplement's 0.014 (a dataset-wide maximum; it would make the verdict "spans" whatever the sky) or a smaller number chosen with the power tables in view.
- **The ozone at 380 and 420 nm gets an allowance tied to Ō, over 0, 1 and 2 times SPCTRL2's log-linear ratio** (cold read 2's finding 1). Rejected:
  - the reader's envelope over no tail and one continuation. The continuation is not known to be an upper bound; the reader's own range from memory runs to about 1.33 times it;
  - the tail continued from Prague's own O(460) and O(500), in one pass or iterated. It failed at its own boundary in the power check (above);
  - Gorshelev's table, fetched, to set the tail. The rule would then rest on a table the paper's channels may band-average, and it would turn this issue into the route under "What would tell instead";
  - dropping 380 and 420 nm from the extraction. With no channel left nearly free of ozone, the aerosol aloft could not be removed.
- **The 15 km pressure is the U.S. Standard pressure at geometric 15 km** (both reads). Rejected: 120.45 hPa, the pressure at geopotential 15 km.
- **The tool must agree with the power check's `estimate` to 1e-6** (cold read 1: a correct implementation that orders its arithmetic differently differs by up to about 4e-8). Rejected: 1e-9, and one shared code path (the tool must stand alone, and two implementations that agree are the check).
- **The aerosol aloft is removed by the guarded two-point power law.** Rejected: a fixed exponent of 1 (a prior the paper does not give, biased when the truth is steep); no removal (biased by +0.0032).
- **The verdict compares the envelope with #55's linearly scaled crossing, not with Δ_M carried into Prague at the envelope's ends.** The power check can then check the rule as registered, and no new matched-depth search is needed. Inside Prague's 40 to 59.4 km segment the two agree to the few percent by which Δ_M departs from linear there (#55's confirming read: 1 to 3 percent), about 0.0003 in the excess at the crossing. Rejected: carrying the ends into Prague, whose verdict the power check could not reproduce.
- **The six aloft spectra are averaged before the extraction.** Prague's ozone above 15 km does not depend on visibility, and averaging divides the part of the noise that differs between the nodes by √6. Rejected: the clearest node alone (it discards five spectra), and a separate extraction at each node (six noisier two-point fits).
- **The spectra go into a committed fixture written by `fixture --aloft`, and `report --aloft` reads only fixtures.** So CI can run the report, as it runs `report --offset` and `report --depth`. Rejected: evaluating Prague inside the report.
- **The power check is committed with this file, before any six-node or high-altitude number exists.** #57's acceptance asks for the rules, the power check included, on `main` first. Rejected: committing it with the build, as the day-5 note on #57 planned.
