# Renderer v1 against Prague at sun 0: what makes the sky-wide offset (pre-registered)

Registered in the commit that lands this file on `main`, on 2026-10-15, for issue #49. At that commit Prague has been evaluated at two sun elevations only, +10 and 0 (`tests/fixtures/prague/sun10.json` and `sun0.json`), and at two aerosol loads only, `aod_550` 0.03 and 0.1. Nobody has evaluated Prague at sun -1 or +1, or at `aod_550` 0.2. The builder implements the four rules below in `tools/prague_reference.py` exactly as written; a disagreement between this file and the tool is a defect in the tool.

**The offset.** At sun 0, away from the sun (relative azimuths 90 and 180, all eight views), v1's luminance is 1.35 times Prague's at `aod_550` 0.03 and 1.31 times at 0.1: the mean of ln(Y_v1 / Y_Prague) is 0.3022 and 0.2675. At sun +10 it was 1.11 and 1.14 (0.1006 and 0.1325). Nothing in #45 explained it (`docs/log/2026-10-13-prague-sun0.md`).

Nothing in this issue changes renderer v1. v1's physics is pre-registered (`docs/gates/renderer-v1/criterion.md`, definition 8).

## What was seen before this file landed, said plainly

Every candidate this file rules on, except Prague's fit, can be computed at sun 0 and sun +10 from data already committed: the two fixtures, v0 and v1. The conductor computed all of it before writing the rules, read-only, so the sun 0 and sun +10 numbers in this section are a **pilot, not a test**. The rules therefore rule only on Prague at sun -1 and +1, and on the load 0.2, which nobody has computed. The tool must reproduce every number in this section to the digits shown and print it under a heading that says it is the pilot.

The pilot, over the 16 points P defined below (the symbols are defined under "Statistics"):

| condition | L | m | 2m | Δ_O, ozone | Δ_H, altitude 0 |
| --- | --- | --- | --- | --- | --- |
| sun +10, 0.03 | 0.1006 | 0.0036 | 0.0072 | 0.0294 | -0.0188 |
| sun +10, 0.1 | 0.1325 | 0.0130 | 0.0260 | 0.0292 | -0.0618 |
| sun 0, 0.03 | 0.3022 | 0.0072 | 0.0144 | 0.0963 | -0.0683 |
| sun 0, 0.1 | 0.2675 | 0.0181 | 0.0362 | 0.0951 | -0.0962 |

Read with the per-condition verdict defined below:

- **Ozone contributes** at all four: a third of the offset at sun 0 (shares 0.319 and 0.356), and 0.29 and 0.22 at sun +10.
- **Putting Prague's observer on its ground moves Prague away from v1** at all four (not this).
- **The aerosol's load dependence:** from 0.03 to 0.1 the offset changes by -0.0347 at sun 0 and +0.0319 at sun +10. Under rule 4's first-order reading (scaled by 0.1 / 0.07 to the part of L at 0.1 that the aerosol makes), that is -0.0495 at sun 0 (not this) and +0.0456 at sun +10, against a bar of 0.0260 (contributes). The aerosol's share grows away from the horizon sun, not toward it.
- **The aerosol amount cannot be settled by the matched-depth reading.** Prague's own aerosol depth at the Koschmieder visibility is 0.0407 at 0.03 and 0.1137 at 0.1 (from its zenith transmittance, as the tool computes it), so it carries 0.011 and 0.014 more aerosol than v1. Matching the depth removes Δ_M = 0.0521 of the offset at (sun 0, 0.1), and 0.0375 at (sun +10, 0.1); at 0.03 the match is clamped at Prague's clearest visibility (depth 0.0369). But the tool's depth estimate carries about ±0.01 of its own (it subtracts v1's Rayleigh and ozone depths, not Prague's), nearly the whole gap. Scaled linearly, ±0.01 puts Δ_M at (sun 0, 0.1) anywhere from 0.0140 to 0.0903, on both sides of the bar 0.0362. Any rule on Δ_M would be decided by that ratio and not by the sky, so there is none. Δ_M is reported with its range.
- After ozone, 0.2059 (at 0.03) and 0.1724 (at 0.1) of the sun-0 offset is left. Of the second, the amount can claim anywhere from 0.0140 to 0.0903.

Also computed before landing, from v0 and v1 alone (no Prague):

- Δ_O at the ruled suns is 0.1084 and 0.1075 at sun -1 (loads 0.03 and 0.1), and 0.0847 and 0.0832 at sun +1.
- v1's own sky curves across sun -1, 0 and +1: the mean over P of ln Y_v1 at sun 0, minus the mean of its values at -1 and +1, is +0.0996 at 0.03 and +0.0543 at 0.1. This is why rule 3 has a floor beside its noise bar.

**What this means for the verdicts before any ruled number exists.** Suppose the sky at sun -1 and +1 behaves like the sky at sun 0. Then rule 1 says "contributes", rule 2 "not this" and rule 4 "not this". They can say otherwise only if sun 0 is unlike its neighbours, which is rule 3's question, and nothing in hand answers it. So rules 1, 2 and 4 are close to confirmations of the pilot at new suns, and the report must call them that. Rule 3 is the only test whose outcome is open.

## What is compared

- **Prague:** the reference implementation, dataset, driver and primary reading of `sun0.json`: commit `2385c912e9051c1258013ff8c3ce2e19e10fb917`, the full dataset (SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`), observer altitude 0.34 km, ground albedo 0.2, visibility by Koschmieder from v1's ground extinction, the model's own interpolation in visibility, linear interpolation between channel centres (`xyz`). The diagnostic readings (`xyz_inverse_visibility`, `xyz_altitude_0`, `xyz_matched_aerosol_depth`, `xyz_binned`, the zenith transmittance and the ground-level dataset check) are computed exactly as the tool computes them for `sun0.json`, including the matched reading's clamp.
- **Two new fixtures,** `tests/fixtures/prague/sun-1.json` and `tests/fixtures/prague/sun1.json`: sun elevation -1 and +1 degrees, relative azimuths 0, 30, 90 and 180, the same eight views as `sun0.json`, and three loads, `aod_550` 0.03, 0.1 and 0.2. Both suns are nodes of Prague's sun axis and of v1's table, so neither model interpolates in the sun, as at sun 0. The load list becomes a per-fixture parameter whose default reproduces `sun10.json` and `sun0.json` byte for byte; neither is regenerated or edited, and `fixture --check` for each must still pass. At 0.2, v1's ground extinction maps to a visibility of 26.99 km, inside Prague's range (20 to 131.8 km); the tool refuses a load whose visibility falls outside it, naming the load.
- **v1:** `engine_xyz(a, s, views, azimuth, "v1")` at the fixture's stored views and recorded sun, with the registered quadrature and the table whose SHA-256 is `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368`. Y is luminance on v1's scale, as in the fixtures.
- **The ozone engine** (in the tool only, never in `lastlight.render`): XYZ_oz(O) is v0's single scattering at `Atmosphere(aod_550=a, ozone_du=O)` plus v1's multiple scattering continued in ozone by the table's own rule. Per channel, MS(O) = exp((1 - t) ln MS_250 + t ln MS_330), with t = (O - 250) / 80, where MS_250 and MS_330 are `multiple_scattering_xyz` at ozone 250 and 330, the table's two ozone nodes. v1 interpolates the table log-linearly between exactly those nodes, so at O = 280 this is v1's XYZ: a test pins Y_oz(280) / Y_v1 - 1 within 1e-12 at every point of every fixture. At O = 345, Prague's U.S. Standard column, it continues v1's own line 15 DU past the table's last node. v0's single scattering, which takes any ozone exactly, checks that line: the ratio of its log response at 345 to its log response at 330 (both against 280) is 1.298 to 1.300 at every pilot condition, and a straight line gives 65 / 50 = 1.300.
- **Points.** P is relative azimuths 90 and 180, all eight views: 16 points, selected by azimuth and view value, never by position in an array.
- **The ruled conditions** are at sun -1 and +1, with the loads each rule names. Sun 0 and sun +10 enter only as the pilot, except that rule 3 tests sun 0's offset against its two new neighbours.

## Statistics

At a condition c = (sun s, load a):

- **L(c)** is the offset: the mean over P of ln(Y_v1 / Y_Prague), under the primary reading. The tool refuses to rule on a condition with L(c) <= 0, naming it, because there is no excess to attribute. This is a refusal and not a branch: from the pilot it can happen at sun -1 or +1 only if both sun 0 and sun +10 are unlike their neighbours.
- **m(c)** is the reading noise: |L_inv(c) - L(c)|, where L_inv uses `xyz_inverse_visibility`. It is how far the offset moves when Prague's aerosol is interpolated between its visibility nodes in 1 / V instead of V, the one reading that says only how precisely Prague's aerosol is set at a given load.
- **Δ_O(c)** = the mean over P of ln(Y_v1 / Y_oz(345)): the part of the offset that goes when v1 is given Prague's ozone.
- **Δ_H(c)** = L(c) - L_alt0(c) = the mean over P of ln(Y_Prague,alt0 / Y_Prague): the part that goes when Prague's observer is put on its ground (`xyz_altitude_0`).

Every comparison uses unrounded values.

**The per-condition verdict.** For a removed part Δ, an offset L and a bar b:

1. **carries** if Δ >= L / 2 and Δ > b;
2. else **contributes** if Δ > b;
3. else **not this**: Δ <= b, which includes every Δ <= 0 (the candidate moves the models apart, or does not move them by more than the noise).

**A rule's verdict over its conditions,** in this order:

1. If every condition carries: **carries**.
2. Else if every condition carries or contributes: **contributes**.
3. Else if every condition is not this: **not this**.
4. Else: **cannot tell**, because the conditions disagree. The report lists each condition's verdict.

For every ruled condition, the report prints Δ, L, the bar, the share Δ / L and the condition's verdict, then the rule's verdict and branch number.

## Rule 1: ozone (v1 280 DU, Prague about 345 DU)

At sun 0 the direct beam crosses the ozone layer on a long slant path, so the Chappuis absorption differs far more than at sun +10.

**Conditions:** (-1, 0.03), (-1, 0.1), (+1, 0.03), (+1, 0.1). **Removed part:** Δ_O. **Offset:** L. **Bar:** 2m.

From v1 alone, Δ_O at these conditions is 0.1084, 0.1075, 0.0847 and 0.0832.

- Ozone carries a condition only if its L is at most twice that: 0.2168, 0.2150, 0.1694 or 0.1664, against 0.30 and 0.27 at sun 0.
- It is "not this" only if m is at least half of Δ_O: 0.0542, 0.0538, 0.0424 or 0.0416, about three times the largest m in the pilot.

Either needs sun 0's neighbours to be unlike it. If Prague's column is 10 DU either side of 345, Δ_O moves by about 15 percent, which changes no pilot verdict.

## Rule 2: observer altitude and ground

Prague puts a 340 m observer above a sea-level ground; v1's observer is on the ground, under 970 hPa of air. Neither Prague reading is v1's geometry:

- at 0.34 km, the air above Prague's observer matches v1's (about 973 hPa), and its ground is 340 m below;
- at altitude 0, the ground is at the observer, as in v1, and the column above is about 4.4 percent heavier.

The issue names the altitude-0 reading as this candidate's diagnostic. The rule asks only whether that reading brings Prague toward v1.

**Conditions:** (-1, 0.03), (-1, 0.1), (+1, 0.03), (+1, 0.1). **Removed part:** Δ_H. **Offset:** L. **Bar:** 2m.

## Rule 3: Prague's fit near the horizon sun

The dataset's sun nodes near the horizon are -1.0, -0.5, 0 and 1.0. The fitted model's accuracy there is not documented, and the fit is made separately at each node. A cause in the physics (ozone, air, aerosol, geometry) changes smoothly with the sun. A fault in the fit at the sun-0 node would put sun 0's offset off the line through its neighbours.

**Statistic.** κ(a) = L(0, a) - (L(-1, a) + L(+1, a)) / 2: how far sun 0's offset sits above the straight line through its two neighbours, 1 degree on either side.

**Conditions:** a in {0.03, 0.1} (`sun0.json` has no 0.2). **Removed part:** κ(a). **Offset:** L(0, a). **Bar:** max(2 m_F(a), 0.05), where m_F(a) is the largest of m(-1, a), m(0, a) and m(+1, a).

**Why the floor of 0.05.** A smooth cause can still bend the offset. Over these three suns v1's own sky bends by +0.0996 at 0.03 and +0.0543 at 0.1 (above). A smooth difference in how fast the two models darken, a fraction of that bend, could put a few hundredths of κ on sun 0 with no fault in the fit. In the pilot, 2m at sun 0 is 0.0144 at 0.03, inside that range, so the noise bar alone would read a modest smooth bend as a fault. The floor 0.05 is half of v1's bend at 0.03 and the floor #45 used for an excess worth attributing. Ozone is a smooth cause: from v1 alone its own part bends by 0.0963 - (0.1084 + 0.0847) / 2 = -0.0003 over these suns at 0.03, so the floor is not there for ozone.

**What it cannot see.** A fault shared by every near-horizon node bends nothing and is not seen. A negative κ (sun 0 below its neighbours' line) reads "not this"; the report prints κ's sign.

## Rule 4: the aerosol's vertical profile

Prague's aerosol sits in a layer with an inversion, not in an exponential with a 1.5 km scale height. A difference in where the aerosol sits makes an offset that grows with the aerosol: to first order in proportion to the load, so going from 0.1 to 0.2 doubles it. A cause in the air or the geometry does not grow at all.

**Statistic.** Λ(s) = L(s, 0.2) - L(s, 0.1). To first order, that is the part of L(s, 0.1) that grows in proportion to the load, which is where a difference in the profile shows.

**Conditions:** s in {-1, +1}. **Removed part:** Λ(s). **Offset:** L(s, 0.1). **Bar:** 2 max(m(s, 0.1), m(s, 0.2)).

The pilot's version of the same reading (0.03 to 0.1, above) gives "not this" at sun 0 and "contributes" at sun +10.

**What it cannot see.**

- **Saturation.** At sun 0 the light reaching the sky away from the sun has crossed the lowest air at a grazing angle, where aerosol extinction can saturate, so a profile effect can grow more slowly than the load. "Not this" says the offset did not grow with the aerosol by more than the noise, not that a profile effect saturating already at 0.1 is excluded.
- **The amount.** Prague's extra aerosol is nearly the same depth at both pilot loads (0.011 and 0.014), so its part of the offset need not grow with the load and is not what Λ measures. It cannot be settled by the matched reading (above). It is reported, never ruled.

## Reported, never ruled

- The pilot section, recomputed.
- For every condition of both new fixtures and both pilot fixtures:
  - L, m, Δ_O, Δ_O from v0's single scattering alone (the mean over P of ln(Y_v0(280) / Y_v0(345))), Δ_H, and their shares of L;
  - Δ_M = L - L_matched, from `xyz_matched_aerosol_depth`, or "clamped" where the match is clamped, with Prague's depth at the primary reading and at the match, and Δ_M's range if the depth estimate is 0.01 off either way (Δ_M scaled by 1 ± 0.01 / the depth gap);
  - the same quantities per relative azimuth (0, 30, 90 and 180, all eight views), so the offset toward the sun is in the same table as away from it.
- κ at 0.03 and 0.1 for Δ_O and for L - Δ_O.
- L under `xyz_binned` (a spectral reading). It is never part of m.
- At each ruled condition: the sum of the parts that rules 1 and 2 found carrying or contributing, and L minus that sum.

## What is not decided here, and what this cannot tell

- These rules attribute v1's offset from Prague. They do not say which model is right: both are models.
- Δ_H mixes the ground's position with a heavier column. "Not this" says the reading that puts Prague's ground where v1's is does not bring Prague toward v1. It does not say the ground's position alone cannot matter.
- No rule tests these: Prague's Rayleigh profile against v1's; refraction; the extraterrestrial solar spectrum each model uses (a common scale factor at every sun); water vapour; or the shape of Prague's ozone profile against v1's Gaussian layer at 22 km. Whatever the rules leave unexplained may be any of these.
- The aerosol amount needs a depth estimate better than ±0.01, from Prague's own gas depths rather than v1's.
- What would tell, for anything left: a reference with v1's own atmosphere, from libRadtran, run offline (GPL, never vendored or imported).
- A change to v1 for any of this is a new renderer with its own registration (#50 is the first), never a change to v1.

## Decisions, each with its rejected alternative

- **The rules rule at sun -1 and +1, not at sun 0.** Every candidate but the fit is already computed at sun 0 (the pilot). Rejected:
  - ruling at sun 0, because the outcome is known;
  - sun -0.5, which the issue proposed. It is a node of Prague's sun axis but not of v1's table, so v1 would interpolate in the sun there, and the curvature test would carry v1's interpolation error.
- **The bar is 2m from inverse visibility alone.** Rejected:
  - #45 rule A's m over inverse visibility and matched depth (the matched depth's own error is larger than the effect it measures);
  - the spread over the 16 points (they are one smooth sky, not independent draws);
  - a share floor such as 0.2 of L. Every share floor would be chosen with the pilot's shares in view, and a noise bar has a reason of its own.
- **Rule 3 has a floor of 0.05 beside its noise bar** (see the rule). Rejected:
  - the noise bar alone, which would read a smooth bend as a fault;
  - a ladder of suns with a fitted curve, which needs more Prague runs and a curve family chosen after the fact.
- **No rule on the aerosol amount.** Its diagnostic's own error (±0.01 in depth) is nearly the whole gap it corrects (0.014), so its verdict would be decided by that ratio and not by the sky. Rejected:
  - a bar of 2m alone, which ignores the matched reading's error;
  - a bar of 2 × (the error's effect on Δ_M), which is larger than Δ_M whenever the gap is under 0.02 and so can never be cleared.
- **The profile is tested by doubling the load from 0.1 to 0.2.** Rejected:
  - the pilot's 0.03 to 0.1 (already computed, so a pilot);
  - the rest after the matched depth, which would inherit the depth's ±0.01;
  - fitting v1's aerosol scale height to Prague. Prague's profile is not an exponential, and a fitted height would absorb every other difference into one number, as #45's fitted asymmetry did.
- **No albedo-0 reading to separate the ground from the column in rule 2.** The pilot finds the altitude-0 reading pushing Prague away from v1 at every condition, and at sun 0 the ground is lit only by the sky. Rejected: a new Prague reading for a candidate the pilot already finds pointing the wrong way. The limit is stated instead.
