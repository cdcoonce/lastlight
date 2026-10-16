# Renderer v1 against Prague at sun 0: what makes the sky-wide offset (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-15, for issue #49. At that commit Prague has been evaluated at two sun elevations only, +10 and 0 (`tests/fixtures/prague/sun10.json` and `sun0.json`). Nobody has evaluated Prague at sun -1 or +1. The builder implements this file in `tools/prague_reference.py` exactly as written; a disagreement between this file and the tool is a defect in the tool.

**The offset.** At sun 0, away from the sun (relative azimuths 90 and 180, all eight views), v1's luminance is 1.35 times Prague's at `aod_550` 0.03 and 1.31 times at 0.1: the mean of ln(Y_v1 / Y_Prague) is 0.3022 and 0.2675. At sun +10 it was 1.11 and 1.14 (0.1006 and 0.1325). Nothing in #45 explained it (`docs/log/2026-10-13-prague-sun0.md`).

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

**The four candidates, and what became of each before any ruled number existed.** The issue names four. Two have a test (rules 1 and 3). Two turned out, on data already in hand, to have no test that Prague can give (rules 2 and 4), and this file says so instead of registering a rule that could not fire or whose outcome was already set:

| rule | candidate | status |
| --- | --- | --- |
| 1 | ozone, 280 DU against about 345 DU | a test at sun -1 and +1 of whether ozone carries the offset or only contributes; the pilot predicts "contributes" |
| 2 | observer altitude and ground | no test: the only reading that moves Prague's observer measures 340 m of air and aerosol, not the ground |
| 3 | Prague's fit at its sun-0 node | a test, the only one whose outcome is open |
| 4 | the aerosol, its amount and profile | no test: the amount's reading error is nearly the whole gap, and the load response does not separate the aerosol from the air |

## What was seen before this file landed, said plainly

Every candidate except Prague's fit can be computed at sun 0 and sun +10 from data already committed: the two fixtures, v0 and v1. The conductor computed all of it before writing the rules, read-only, so the sun 0 and sun +10 numbers in this section are a **pilot, not a test**. The rules rule only on Prague at sun -1 and +1, which nobody has computed. Three independent cold reads (on this file's pull request, before it landed, none running Prague) recomputed every pilot number. They found three that were wrong, all corrected here. The first two reads showed that the first draft's rules 2 and 4 could not test their candidates (the evidence is under each rule). The third read the revision and asked for the disclosures under rules 1, 2, 3 and 4 that cite it.

The report prints every number in this section under a heading containing the word "pilot". Each must agree with the value shown here to within half a unit in its last digit.

The pilot, over the 16 points P defined below (the symbols are defined under "Statistics"):

| condition | L | m | 2m | Δ_O, ozone | Δ_H, altitude 0 |
| --- | --- | --- | --- | --- | --- |
| sun +10, 0.03 | 0.1006 | 0.0036 | 0.0072 | 0.0294 | -0.0188 |
| sun +10, 0.1 | 0.1325 | 0.0130 | 0.0260 | 0.0292 | -0.0618 |
| sun 0, 0.03 | 0.3022 | 0.0072 | 0.0144 | 0.0963 | -0.0683 |
| sun 0, 0.1 | 0.2675 | 0.0181 | 0.0362 | 0.0951 | -0.0962 |

- **Ozone contributes** at all four, under the per-condition verdict below: a third of the offset at sun 0 (shares 0.319 and 0.356), and 0.29 and 0.22 at sun +10. Its margin over the bar is 6.68 and 2.63 times at sun 0 and 4.08 at (sun +10, 0.03), but only 1.12 times at (sun +10, 0.1), where a Prague column under 337.94 DU would turn it into "not this".
- After ozone, 0.2059 (at 0.03) and 0.1724 (at 0.1) of the sun-0 offset is left.
- The evidence that rules 2 and 4 have no test is under each rule.

Also computed before landing, from v0 and v1 alone (no Prague):

- Δ_O at the ruled suns is 0.1084 and 0.1075 at sun -1 (loads 0.03 and 0.1), and 0.0847 and 0.0832 at sun +1.
- v1's own sky curves across sun -1, 0 and +1. The mean over P of ln Y_v1 at sun 0, minus the mean of its values at -1 and +1, is +0.0996 at 0.03 and +0.0543 at 0.1. This is why rule 3 has a floor beside its noise bar.

**What the pilot predicts.** If the sky at sun -1 and +1 behaves like the sky at sun 0, rule 1 says "contributes". The margin at sun +1 is not wide. Rule 1 carries a condition there if L(+1) is at most 0.1695 (0.03) or 0.1664 (0.1). A third cold read modelled the offset's change with the sun as a shift of the sun, which puts L(+1) near 0.215 and 0.222, 0.045 and 0.056 above those bars. So "cannot tell" (sun +1 carries, sun -1 contributes) is a realistic outcome with no fault anywhere. Beside rule 1's verdict the report prints the verdict the pilot's two sun-0 conditions give under rule 1's rule-level verdict.

## What is compared

- **Prague:** the reference implementation, dataset, driver and primary reading of `sun0.json`: commit `2385c912e9051c1258013ff8c3ce2e19e10fb917`, the full dataset (SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`), observer altitude 0.34 km, ground albedo 0.2, visibility by Koschmieder from v1's ground extinction, the model's own interpolation in visibility, linear interpolation between channel centres (`xyz`). Every diagnostic reading is computed exactly as the tool computes it for `sun0.json`, including the matched reading's clamp: `xyz_inverse_visibility`, `xyz_altitude_0`, `xyz_matched_aerosol_depth`, `xyz_binned`, the zenith transmittance, and the ground-level dataset check.
- **Two new fixtures,** `tests/fixtures/prague/sun-1.json` and `tests/fixtures/prague/sun1.json`. They differ from `sun0.json` only in the sun elevation, -1 and +1 degrees: the same relative azimuths (0, 30, 90 and 180), the same eight views and the same two loads (0.03 and 0.1). Both suns must be nodes of Prague's sun axis (as read from the dataset's header) and of v1's table (as read from the table's header), and the tool refuses to write either fixture otherwise, naming the axis. So neither model interpolates in the sun, as at sun 0. `sun10.json` and `sun0.json` are not regenerated or edited, and `fixture --check` for each must still pass.
- **v1:** `engine_xyz(a, s, views, azimuth, "v1")` at the fixture's stored views and recorded sun, with the registered quadrature and the table whose SHA-256 is `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368`. Y is luminance on v1's scale, as in the fixtures.
- **The ozone engine** (in the tool only, never in `lastlight.render`): XYZ_oz(O) is v0's single scattering at `Atmosphere(aod_550=a, ozone_du=O)`, with the registered quadrature (the default), plus v1's multiple scattering continued in ozone by the table's own rule. Per channel, MS(O) = exp((1 - t) ln MS_250 + t ln MS_330), with t = (O - 250) / 80, where MS_250 and MS_330 are `multiple_scattering_xyz` at ozone 250 and 330, the table's two ozone nodes.
  - v1 interpolates the table log-linearly in every axis, and its ozone axis has exactly those two nodes, so at O = 280 this is v1's XYZ. A test pins |Y_oz(280) / Y_v1 - 1| <= 1e-12 at every point of every fixture.
  - At O = 345, Prague's U.S. Standard column, it continues v1's own line 15 DU past the table's last node.
  - v0's single scattering takes any ozone exactly, so it checks the line: the mean over P of ln(Y_v0(280) / Y_v0(345)), divided by the mean over P of ln(Y_v0(280) / Y_v0(330)). A straight line gives 65 / 50 = 1.3. The pilot gives 1.2963 and 1.2960 at sun 0 and 1.2992 and 1.2993 at sun +10. The report prints this ratio at every condition; it is a check, not a rule.
  - For the multiple scattering, `docs/renderer/bruneton-convergence.md` bounds the table's ozone interpolation error at 0.49 percent.
- **Points.** P is relative azimuths 90 and 180, all eight views: 16 points, selected by azimuth and view value, never by position in an array.
- **The ruled conditions** are at sun -1 and +1. Sun 0 and sun +10 enter only as the pilot, except that rule 3 tests sun 0's offset against its two new neighbours.

## Statistics

At a condition c = (sun s, load a):

- **L(c)**, the offset: the mean over P of ln(Y_v1 / Y_Prague), under the primary reading. **L_A(c)** is the same under `xyz_altitude_0`.
- **m(c)**, the reading noise: |L_inv(c) - L(c)|, where L_inv uses `xyz_inverse_visibility`. It is how far the offset moves when Prague's aerosol is interpolated between its visibility nodes in 1 / V instead of V: the one reading that says only how precisely Prague's aerosol is set at a given load. Its size depends on where v1's visibility falls between Prague's nodes; it is the reading's noise at that load, not a property of the physics.
- **Δ_O(c)** = the mean over P of ln(Y_v1 / Y_oz(345)): the part of the offset that goes when v1 is given Prague's ozone.
- **Δ_H(c)** = L(c) - L_A(c) = the mean over P of ln(Y_Prague,alt0 / Y_Prague) (reported; see rule 2).

Every comparison uses unrounded values, computed by the tool; the four-digit values in this file are what it must reproduce, never thresholds it reads.

**Refusal.** If any L or L_A a rule uses is <= 0 (in rule 3, at sun -1, 0 and +1 under both readings), the report is not written: the tool prints the condition and reading to standard error and exits 1, because there is no excess to attribute. No condition is ever dropped. From the pilot this can happen at sun -1 or +1 only if both sun 0 and sun +10 are unlike their neighbours, so it is a refusal, not a branch.

**The per-condition verdict.** For a removed part Δ, an offset L and a bar b:

1. **carries** if Δ >= L / 2 and Δ > b;
2. else **contributes** if Δ > b;
3. else **not this**: Δ <= b, which includes every Δ <= 0 (the candidate moves the models apart, or does not move them by more than the noise).

**A rule's verdict over its conditions,** in this order:

1. If every condition carries: **carries**.
2. Else if every condition carries or contributes: **contributes**.
3. Else if every condition is not this: **not this**.
4. Else: **cannot tell**, because the conditions disagree. The report lists each condition's verdict.

For every ruled condition, the report prints Δ, L, the bar, the share Δ / L and the condition's verdict. It then prints the rule's verdict and the rule's branch number (1 to 4, above).

## Rule 1: ozone (v1 280 DU, Prague about 345 DU)

At sun 0 the direct beam crosses the ozone layer on a long slant path, so the Chappuis absorption differs far more than at sun +10.

**Conditions:** (-1, 0.03), (-1, 0.1), (+1, 0.03), (+1, 0.1). **Removed part:** Δ_O. **Offset:** L. **Bar:** 2m.

From v1 alone, Δ_O at these conditions is 0.1084, 0.1075, 0.0847 and 0.0832.

- Ozone carries a condition only if its L is at most twice that: 0.2168, 0.2150, 0.1695 or 0.1664, against 0.30 and 0.27 at sun 0.
- It is "not this" only if m is at least half of Δ_O: 0.0542, 0.0538, 0.0424 or 0.0416. That is 3.0, 3.0, 2.3 and 2.3 times the largest m in the pilot.

**Rule 1's "not this" cannot fire in practice.** Δ_O is fixed and positive from v1 alone. m is about 0.05 (at 0.03) and 0.1 (at 0.1) times the change in Prague's luminance between its two bracketing visibility nodes, which is 0.14 and 0.19 in the pilot. Reaching m = Δ_O / 2 would need that change to be 0.81 to 1.04 at 0.03 and 0.43 to 0.56 at 0.1. The rule-level "not this" (branch 3) is therefore dead too. Rule 1's live outcomes are "carries", "contributes" and "cannot tell"; it asks whether ozone carries the offset or only contributes to it, never whether ozone is part of it. Prague's column is the U.S. Standard profile's, about 345 DU. If it is 10 DU either side, Δ_O moves by about 15 percent: no sun-0 pilot verdict changes, and (sun +10, 0.1) is the pilot condition whose verdict could.

## Rule 2: observer altitude and ground: no test, and why

Prague puts a 340 m observer above a sea-level ground; v1's observer is on the ground, under 970 hPa of air. At 0.34 km the air above Prague's observer matches v1's (about 973 hPa), and its ground is 340 m below.

The only reading that moves Prague's observer is `xyz_altitude_0`, which puts it on its ground, as in v1. But it also puts the observer under 340 m more air and, with it, the bottom of Prague's aerosol layer. That slab dominates the reading:

- Under an exponential aerosol with v1's 1.5 km scale height, the slab holds 1 - exp(-0.34 / 1.5) of the column. So it is v1 with its `aod_550` raised by the factor exp(0.34 / 1.5) = 1.2544 that stands in for the slab.
- That stand-in moves the mean over P of ln Y_v1 by -0.0684 and -0.0894 at sun 0 (loads 0.03 and 0.1), and -0.0265 and -0.0597 at sun +10.
- The reading's own Δ_H was -0.0683 and -0.0962 at sun 0, and -0.0188 and -0.0618 at sun +10.

Δ_H minus the stand-in leaves +0.0001 and -0.0068 at sun 0, and +0.0077 and -0.0021 at sun +10 (cold read 3). Δ_H is the slab's extinction, not the ground's position, so no rule on it can attribute anything to the ground.

The ground's position by itself works the other way. Raising Prague's ground to its observer would remove the slab below the observer and the light it scatters upward. That makes Prague's sky no brighter, so the offset cannot shrink. The geometry can hide part of the offset, not cause it.

The report prints, under "Rule 2: no test", Δ_H and the stand-in's change at every condition of all four fixtures.

What would tell: Prague with its ground raised to the observer, which the model cannot do; or a reference with v1's own geometry (libRadtran, run offline; GPL, never vendored or imported).

## Rule 3: Prague's fit near the horizon sun

The dataset's sun nodes near the horizon are -1.0, -0.5, 0 and 1.0. The fitted model's accuracy there is not documented, and its fit at each node is separate. A cause in the physics changes smoothly with the sun. A fault in the fit at the sun-0 node would put sun 0's offset off the line through its neighbours.

**Two readings, because the geometry has a kink of its own.** v1's observer is on the ground, so its horizon, and with it the edge of Earth's shadow on the air beside it, is at sun 0. Prague's observer at 0.34 km sees the sun until -0.59 degrees. Between -0.59 and 0 the air beside Prague's observer is lit and the air beside v1's is not, so under the primary reading the offset can bend at sun 0 for a reason that is not a fault. Under `xyz_altitude_0` both observers are on the ground and the two horizons coincide, at the cost of the slab (rule 2), whose stand-in bends by -0.0125 at 0.03 and -0.0008 at 0.1 (from v1 alone). A fault in the fit at the sun-0 node shows under both readings. A kink from the observer's height shows under the primary reading only. A third cold read emulated the elevated observer in v0's single scattering and put its effect on κ at only +0.004 (0.03) and +0.002 (0.1), so the second reading is a guard more than a correction.

**Statistic.** κ(a) = L(0, a) - (L(-1, a) + L(+1, a)) / 2, and κ_A(a), the same from L_A: how far sun 0's offset sits above the straight line through its two neighbours, 1 degree on either side.

**Conditions:** a in {0.03, 0.1}. **Removed part:** min(κ(a), κ_A(a)), so a fault is called only if both readings show it. **Offset:** L(0, a). **Bar:** max(2 m_F(a), 0.05), where m_F(a) is the largest of m(-1, a), m(0, a) and m(+1, a). Twice m_F bounds how far the inverse-visibility reading could move κ. There is no inverse-visibility reading at altitude 0, so the same bar serves κ_A.

**Why the floor of 0.05.** A smooth cause can still bend the offset, because each model's sky bends on its own (v1's by +0.0996 at 0.03 and +0.0543 at 0.1). The cold reads computed the bend that each smooth difference they could model puts on κ, in v1's sky:

- a deeper aerosol layer (scale height), at most 0.025;
- Prague's extra aerosol amount, +0.0167 at 0.03 and +0.0006 at 0.1;
- ozone, -0.0003 at 0.03 (0.0963 - (0.1084 + 0.0847) / 2).

The floor is twice the largest of them. Stacked in the same direction they reach at most 0.042 at 0.03, still under it. The noise bar alone (2m is 0.0144 at 0.03 in the pilot) would read such a bend as a fault.

Refraction does not enter. Neither model refracts: v1's sun elevation is geometric by design, and Prague's reference implementation (cached source at the pinned commit) builds the sun's direction straight from the elevation it is given, with no refraction term anywhere in its source. An earlier draft listed "refraction in one model and not the other" as a confound; it is not one between these two models.

**What it cannot see.**

- A fault shared by every near-horizon node bends nothing.
- A fault in the fit at one observer altitude and not the other is not called.
- κ cannot tell a fault in Prague's fit from a kink in v1 at its own sun-0 node: v1's table has a node at 0 too, and its multiple scattering makes 0.065 of v1's 0.0996 bend at 0.03 (cold read 3). "Carries" or "contributes" says one of the two models has a kink at sun 0, most likely Prague's (v1's single scattering is exact in the sun, and its table interpolates log-linearly between nodes 1 degree apart), but not which.
- A negative κ (sun 0 below its neighbours' line) reads "not this".

The report prints κ and κ_A with their signs.

## Rule 4: the aerosol, its amount and profile: no test, and why

Prague's aerosol at the Koschmieder visibilities is deeper than v1's, 0.0407 and 0.1137 against 0.03 and 0.1 (from its zenith transmittance, as the tool computes it), in a layer with an inversion, not an exponential with a 1.5 km scale height.

- **The amount.** Matching the depth (`xyz_matched_aerosol_depth`) removes Δ_M = 0.0521 of the offset at (sun 0, 0.1) and 0.0375 at (sun +10, 0.1); at 0.03 the match is clamped at Prague's clearest visibility (depth 0.0369).
  - The tool's depth estimate subtracts v1's Rayleigh and ozone depths, not Prague's. #34 put its error at about ±0.01, nearly the whole gap of 0.0137.
  - Scaled linearly, ±0.01 puts Δ_M at (sun 0, 0.1) anywhere from 0.0140 to 0.0903, on both sides of the bar 0.0362. A rule on Δ_M would be decided by that ratio, not by the sky.
  - The ±0.01 is #34's statement and was never derived. Cold read 1 estimated ±0.003 to ±0.005. At ±0.003 Δ_M's whole range would clear the bar; at ±0.005 it would not (0.0331 to 0.0712). Choosing the error now, with this in view, would decide the verdict, so it is not chosen.
  - Cold read 3 fitted Prague's own gas depths from the fixtures' 11-channel zenith transmittance (fit rms 0.0034). It put the estimator's error at 0.003 to 0.006, against the crossing of 0.0042 at (sun 0, 0.1): a derived error would still decide the verdict. That look was taken before this file landed and is disclosed here; it is why no error is chosen.
- **The profile.** A first draft tested it by the offset's growth when the load doubles (from 0.1 to 0.2), on the premise that an aerosol-made offset grows in proportion to the load. Both cold reads showed the premise false in v1's own sky.
  - A deeper aerosol layer (scale height 1.5 to 2.0 km, v0's single scattering) changes the mean over P of ln Y by an amount that grows ×1.41, ×1.35 and ×0.80 from 0.1 to 0.2 at sun -1, 0 and +1. From 0.03 to 0.1 it grows ×3.31, ×3.68 and ×1.89. At sun +10 it has the other sign (+0.0405 at 0.1, against -0.0576 at sun 0).
  - A constant extra aerosol amount makes a part of the offset that shrinks with the load, of the size of the bar.
  - The growth also carries every other load-dependent difference: the phase function, the single-scattering albedo, the Angström exponent and v1's multiple scattering.

  The growth could never reach "carries", and at sun +1 not even "contributes", for any aerosol change tried.

The report prints, under "Rule 4: no test", at every condition of all four fixtures:

- Δ_M = L - L_matched, or "clamped" where the match is clamped (no range then);
- Prague's aerosol depth at the primary reading and at the match;
- Δ_M's range if the depth estimate is 0.01 off either way: Δ_M × (1 - 0.01 / g) to Δ_M × (1 + 0.01 / g), where g is Prague's depth at the primary reading minus its depth at the match, printed as computed even where the lower end is negative;
- at each sun, the load step L(s, 0.1) - L(s, 0.03);
- at each sun, the change the deeper layer makes to the mean over P of ln Y at both loads, and its ratio between them.

What would tell: the depth estimate's error, derived from Prague's own gas depths (fitted from its spectral transmittance), which may make the amount testable; and a reference with v1's own aerosol (libRadtran, offline).

## Reported, never ruled

- The pilot section, recomputed.
- For every condition of both new fixtures and both pilot fixtures, these quantities, and the same for each relative azimuth the fixture has, so the offset toward the sun is in the same table as away from it:
  - L, L_A, m, Δ_O, Δ_H, and the shares of L;
  - Δ_O from v0's single scattering alone, the mean over P of ln(Y_v0(280) / Y_v0(345));
  - the ozone line check.
- At 0.03 and 0.1: κ for Δ_O; κ and κ_A for L; and κ for L - Δ_O with κ_A for L_A - Δ_O.
- L under `xyz_binned` (a spectral reading). It is never part of m.
- At each ruled condition of rule 1, what is left: L minus Δ_O where that condition's verdict is "carries" or "contributes", otherwise L.

## What is not decided here, and what this cannot tell

- These rules attribute v1's offset from Prague. They do not say which model is right: both are models.
- No rule tests these:
  - the observer and ground (rule 2) and the aerosol (rule 4);
  - Prague's Rayleigh profile against v1's;
  - refraction, which neither model includes (it matters against the camera, not between these two models);
  - the extraterrestrial solar spectrum each model uses (a common scale factor at every sun);
  - water vapour;
  - Prague's ozone column and profile (taken as the U.S. Standard 345 DU) and its ozone cross-sections, against v1's Gaussian layer at 22 km and v1's own cross-sections, which Δ_O uses;
  - v1's own multiple scattering. Its share of v1's luminance over P at the ruled conditions averages 0.38 to 0.73 (0.19 to 1.00 point by point, from v1 alone), and #45 found its port 6.6 percent above a Monte Carlo at 0.1.

  Whatever the rules leave unexplained may be any of these.
- A change to v1 for any of this is a new renderer with its own registration (#50 is the first), never a change to v1.

## Decisions, each with its rejected alternative

- **The rules rule at sun -1 and +1, not at sun 0.** Every candidate but the fit is already computed at sun 0 (the pilot). Rejected:
  - ruling at sun 0, because the outcome is known;
  - sun -0.5, which the issue proposed. It is a node of Prague's sun axis but not of v1's table, so v1 would interpolate in the sun there, and the curvature test would carry v1's interpolation error.
- **Rules 2 and 4 are registered as "no test", with their evidence.** Rejected:
  - the first draft's rule 2 on Δ_H, which measures the slab, so its "not this" was set by physics before any ruled number;
  - the first draft's rule 4 on the load step from 0.1 to 0.2 (its premise is false in v1's own sky), and the third load of 0.2 it needed, which the new fixtures therefore do not carry;
  - a rule on Δ_M with a bar of 2m alone (it ignores the matched reading's error), or with a bar of twice the error's effect (larger than Δ_M whenever the gap is under 0.02, so it could never be cleared);
  - an albedo-0 reading to bound the ground's light (it bounds the ground's reflection but not the slab, and the ground's position can only hide the offset);
  - fitting v1's aerosol scale height to Prague (Prague's profile is not an exponential, and a fitted height would absorb every other difference into one number, as #45's fitted asymmetry did).
- **The bar is 2m from inverse visibility alone.** Rejected:
  - #45 rule A's m over inverse visibility and matched depth (the matched depth's own error is larger than the effect it measures);
  - the spread over the 16 points (they are one smooth sky, not independent draws);
  - a share floor such as 0.2 of L. Every share floor would be chosen with the pilot's shares in view, and a noise bar has a reason of its own.
- **Rule 3 calls a fault only if both observer heights show it, above a floor of 0.05.** Rejected:
  - the primary reading alone, whose observer's horizon is at -0.59 degrees and not at v1's 0;
  - the altitude-0 reading alone, which carries the slab;
  - the noise bar alone, which would read a smooth bend as a fault;
  - a floor tied to v1's own bend at each load (half of it is 0.050 at 0.03 and 0.027 at 0.1). That is near the smooth bends computed above at 0.1, and the question is the size of a fault, which does not depend on the load;
  - a ladder of suns with a fitted curve, which needs more Prague runs and a curve family chosen after the fact.
