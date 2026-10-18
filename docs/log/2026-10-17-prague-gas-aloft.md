# Prague's aerosol amount contributes, under the registered conditions (#57)

**Result: contributes, at all four suns.** The rule pre-registered in `docs/renderer/prague-gas-aloft-criterion.md` (landed in #60, before any six-node or 15 km Prague number existed) was built as written and run on Prague's zenith transmittance at its six visibility nodes and two observer altitudes, evaluated once after it landed.

- The envelope of the excess is g_lo = 0.011984 to g_hi = 0.019657.
- #55's crossings are 0.009444, 0.009483, 0.009527 and 0.009492 at sun -1, 0, +1 and +10.
- The whole envelope lies over every crossing, so the verdict line reads `verdict: contributes`.

Matching Prague's aerosol depth at the 0.1 load's visibility removes more than #49's bar from v1's offset over Prague. This holds on the linear scaling of #49's matched reading, under the criterion's seven conditions. #49's rule 4 ("no test" for the aerosol amount) and #55's "no test" for the six-node route stand for their routes; this is #57's route, with its own test.

Renderer v1 is unchanged.

## What the numbers say

From `report --aloft` (full output in the PR):

- **Prague's gas at 550 nm** is 0.12023 under the paper's Rayleigh (0.11953 to 0.12085 across the nine members), against the paper's 0.12367 computed with SPCTRL2's ozone. Prague's ozone at 550 nm above 0.34 km, a Ō, is 0.0263, against the paper's 0.0297 with SPCTRL2's coefficient and 345 DU: about 11 percent less. At k = 1 the Rayleigh parts are the same, so that is the whole difference in the gas, and it is why the excess (0.0155 at k = 1 and m = 0) sits 0.0034 over the paper's 0.0121.
- **The fraction of the 0.34 km ozone above 15 km,** 1 / a, is 0.838 to 0.852, in the range the power check assumed (0.80 to 0.90).
- **The error.** SE_k is 0.0013 to 0.0017. Of it, the extraction term is 0.0010 to 0.0015, the shared tolerance at 0.34 km 0.0007 to 0.0009, and the fit's own SE 0.0003. s_k is 0.00089 to 0.00107 and s_15 is 0.00011.
- **The envelope sits inside Prague's linear segment.** g_seg, the excess at the 59.4 km node, is 0.02315, above g_hi. Δ_M is nearly linear in the excess over the whole envelope (condition 7), and no part of it rests on extrapolating past 59.4 km.
- **The aerosol aloft barely changes with visibility.** Each node's difference from the mean at 550 nm, at 15 km, is -8e-5 to +1.4e-4.

## What the data say about the conditions (diagnostics, not part of the rule)

These were computed after the verdict, from the committed fixture. None of them changes it.

- **380 nm reads low at both altitudes.** At 15 km, Y(380) less k R15(380) is -0.00081, -0.00134 and -0.00188 for k = 0.99, 1.00 and 1.01. Prague's extinction there is about 2.5 percent under the tool's Rayleigh, while ozone and aerosol can only add to it. At 420 nm the same residual is +0.00095, +0.00060 and +0.00025, the ozone tail's 0.0003 and a little more. At 0.34 km the six-node fit's residual at 380 nm is -0.0007 at every node. So the deficit is at 380 nm, not in the Rayleigh's amplitude across the band. It is 0.8 to 1.9 times s_k, at the edge of condition 4's scale for an error shared by the nodes.
- **Its effect: the aerosol aloft was not removed.** With r(380) negative, the power law fell to its guard in all nine members (η = 0), and β came out 0 in eight of them. The extracted ozone then carries whatever aerosol is aloft. The power check's ENO estimator showed that this biases the gas up and the excess down (+0.0032 in G for aloft depths up to 0.005). So the deficit does not push the verdict toward "contributes"; if anything the excess is understated.
- **If instead Prague's Rayleigh is low** (condition 1 broken toward a lower Rayleigh), the power check's dr_m2 cells show the same direction. Their false outcomes were "not this" or "mixed", never "contributes", and their 50 percent points moved up. A Rayleigh higher than the tool's, the direction that would favour "contributes", is what the 380 nm channel argues against.
- **The six-node fit's residual is shared by the nodes.** In every channel the mean residual over the six nodes equals its rms: -0.0007, +0.0010, +0.0011, -0.0009, -0.0012, +0.0003, +0.0007, -0.0004, -0.0005 and +0.0006 from 380 to 740 nm (k = 1, m = 1). An error or shape mismatch shared by the nodes is what SE_0 was added to carry (the criterion's choice 4, after its first power grid). Without it, the fit's SE would have treated the residual as independent across 60 values and divided it by √6.

## How it was built and reviewed

- **Tests first, the Prague numbers last.** The order was:
  1. `875e418` holds the synthetic-spectra tests alone, red (197 failed, 3 passed, because the tool had no aloft code);
  2. `866bcb3` holds the tool;
  3. `3ca0737` holds `aloft.json`, the first evaluation of Prague at its six nodes or at 15 km, made after the criterion landed in #60;
  4. `5772f41` holds the tests that pin the report's quantities on that fixture.
- **The tool agrees with the registered power check.** Over 36 synthetic pairs from 16 of its cells, the tool's g_lo and g_hi equal `tools/issue57_power.py`'s to at most 2.8e-9 (registered: 1e-6). On `aloft.json` they agree exactly.
- **A blind mutation review** (`tests/teeth/feat-prague-gas-aloft.teeth.json`) wrote 106 rows: the 58 intents the conductor authored before this PR existed, 47 of the reviewer's own, and one control.
  - 92 were killed, each with its predicted test among the failures, and all 58 intents were killed. 13 survived and the control held.
  - Four survivors were blocking. X1: `fixture --aloft --check` never noticed the 15 km half going uncompared. X16 and X17: the verdict could compare rounded crossings or rounded envelope ends. X39: the upper visibility tolerance was untested.
  - The reviewer disclosed that its first form of row I45 let `report --aloft` reach the Prague binary during tests, a few times. Nothing was recorded, and the registration had landed hours before.
- **The fix round** (`dfcd4b5`, `15d5654`):
  - An autouse stub makes `evaluate` raise in the #57 tests, so no test can reach the binary.
  - It closed X1 and its mirror (the 0.34 km half), X16, X17 and X39, plus X20, X35, X42 and X45, each RED-confirmed on its mutant. Three by-hand mutants were added as rows F1 to F3.
  - The spec now holds 109 rows: 103 killed, 5 survived, 0 broken, and the control held. The survivors (X15, X21, X27, X36, X37) are below every tolerance, still refused, or message order only.
- **The conductor re-injected six rows** from the committed spec: X1, X16, X17, X39, I53 and I54. Each anchor was asserted to occur once, and only the predicted test was run. All six were killed by it. Eight rows the fix round closed were relabelled to name the fix round's tests: X1, X16, X17, X20, X35, X39, X42 and X45.
- **The gate** (`uv sync --frozen && uv run ruff check && uv run ruff format --check && uv run pytest && uv run lastlight privacy-scan`) passed: 4,310 passed, 10 deselected, privacy clean.
- **CI failed once, on the platform, not the rule.** The first landing attempt stopped when CI failed two value-pin tests that pass on the build machine. CI's Linux runner and the macOS build machine differ by about 1.1e-6 relative in |∇_k| on the committed spectra, because the registered step of 1e-5 differences fits converged to 1e-15. The gradient-built quantities (|∇_k|, |∇⁰_k|, SE_ext, SE_0, SE_k, g_k ± 2 SE_k, g_lo and g_hi) are now pinned at 1e-4 relative instead of 1e-6, and their printed digits are checked against the same run's values (`a2b360c`). The same-platform agreement with the power check (1e-6) still binds them. After the change the full spec was rerun: 103 killed, the same 5 survived, 0 broken, and the control held. No kill depended on the tighter pins.
- **Found in passing, not changed.** `evaluate`'s refusal of a reported altitude uses `np.allclose` with its default relative tolerance, so its bound is about 3.4 mm at 340 m and 0.15 m at 15 km, not the 1e-6 m it reads as. The #57 fixture writer checks the altitude itself at 1e-6 m, so this instrument does not rest on it.

## What this does not decide

- Which model is right. Prague and v1 are both models, and "contributes" says that matching Prague's aerosol depth would remove more than #49's bar from v1's offset over Prague. It does not say v1's aerosol is wrong.
- Whether v1 should change. Any change is a new renderer with its own registration (#50), never a change to v1.
- The paper's own gas from Gorshelev's 223 K table. That is #61, which can now disclose this verdict as its pilot. The 11 percent lower ozone at 550 nm is a prediction #61 can check.
- The 380 nm channel's deficit. Whether it is Prague's reconstruction, its spectral sampling or its Rayleigh at short wavelengths is not settled here.
