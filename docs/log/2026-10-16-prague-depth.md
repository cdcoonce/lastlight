# Prague's aerosol amount from its own gas depths: no test (#55)

**Result: no test.** The route #55 proposed was to fit Prague's gas (Rayleigh and ozone) to its zenith transmittance across its six visibility nodes, derive the estimate's error, and rule on whether the aerosol amount explains part of v1's offset over Prague. That route cannot settle the amount, and the criterion registers it as "no test" with its evidence (`docs/renderer/prague-aerosol-depth-criterion.md`, landed in #56 before any six-node number existed). #49's "no test" for the aerosol stands, now with this evidence beside it. No six-node fit was run, and no Prague evaluation happened in this issue. Renderer v1 is unchanged. In the words of #55's acceptance: the derived error still spans the bar, and the six-node route cannot make it otherwise. A route that may have a test is filed as #57.

## What the criterion found before landing

The draft criterion fitted Prague's gas eight ways: the aerosol's form (power law or curved), times the ozone column (free or the paper's 345 DU), times the Rayleigh depth (free or the paper's, Bodhaine's for U.S. Standard air above 0.34 km). The amount's part of the offset was then ruled through the envelope of the eight excesses, each widened by twice its fitting error. The conductor computed the pilot first, from the two transmittance spectra the fixtures already hold. Two cold reads of the draft followed, a confirming read of the revision, and a fourth read of its corrections. None ran the Prague binary, and all four are posted on #56. The second read showed the route has no test:

1. **The amounts carry no information about the gas** when each node's aerosol shape is free. Node i's Jacobian columns span {A_i, x A_i, x² A_i}, whatever its amount. Six nodes of the pilot's shapes only repeat the fit, which divides its error by exactly √3. #55's mechanism needs the nodes' shapes tied together. Prague's are not: its nodes run from continental clean to continental polluted.
2. **The drafted rule's verdict was set by its design.**
   - "Not this", "carries" and "cannot tell" could not fire, because the paper's gas fixes an excess of 0.0121 inside every envelope.
   - "Contributes" was not more likely as the true excess grew. In the cold read's verdict-level synthetic, P(contributes) was at most 0.23, and 0 at the largest true excess.
3. **The ozone stand-in moves single estimates by up to 0.016.** SPCTRL2's table stands in for the paper's Gorshelev cross-sections at 223 K, and the residuals stay within a factor of about two of the pilot's.

The confirming read corrected one overstatement. A smooth tie between the nodes' shapes, or a basis of the paper's species, would restore the mechanism. With SPCTRL2's ozone, though, it still falls to problem 3. #57 takes that up with Prague's own ozone.

## What the tool prints, and its numbers

`uv run python tools/prague_reference.py report --depth` reads only the four committed fixtures and v1, runs no Prague evaluation and writes nothing. The tables are copied from its output.

| estimate | aerosol form | ozone | Rayleigh | G | 2SE | g | g - 2SE | g + 2SE | residual rms | a_R | ozone (DU) | condition number |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | power law | free | free | 0.1149 | 0.0032 | 0.0208 | 0.0176 | 0.0241 | 0.00096 | 0.0859 | 327.4 | 882 |
| 2 | power law | free | paper | 0.1233 | 0.0022 | 0.0124 | 0.0102 | 0.0147 | 0.00182 | 0.0930 | 341.0 | 17.5 |
| 3 | power law | paper | free | 0.1171 | 0.0030 | 0.0186 | 0.0156 | 0.0216 | 0.00114 | 0.0866 | 345.0 | 912 |
| 4 | power law | paper | paper | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00182 | 0.0930 | 345.0 | 14.1 |
| 5 | curved | free | free | 0.1144 | 0.0280 | 0.0214 | -0.0066 | 0.0494 | 0.00081 | 0.0853 | 327.9 | 1.470e4 |
| 6 | curved | free | paper | 0.1214 | 0.0013 | 0.0144 | 0.0131 | 0.0156 | 0.00083 | 0.0930 | 318.5 | 87.2 |
| 7 | curved | paper | free | 0.0990 | 0.0601 | 0.0367 | -0.0234 | 0.0969 | 0.00092 | 0.0686 | 345.0 | 1.033e4 |
| 8 | curved | paper | paper | 0.1237 | 0 | 0.0121 | 0.0121 | 0.0121 | 0.00117 | 0.0930 | 345.0 | 76.9 |

The pilot's eight gas estimates, fitted to the two committed spectra. G is the gas depth at 550 nm, and g = ȳ(V_p) - G - 0.1 is Prague's aerosol depth at the 0.1 load's visibility minus v1's.

- **The old estimate is not consistent with itself.** It subtracts v1's gas at 550 nm, 0.12208, from a total interpolated between the 540 and 580 nm channel centres. The same gas interpolated the same way, at the paper's pressure, is 0.12367: 0.00129 of the difference is the interpolation and 0.00030 the pressure. The paper's gas puts Prague's excess over v1's 0.1 at 0.0121, not 0.0137.
- **The crossings.** Scaled linearly from #49's matched reading, the amount's part would equal the bar at an excess of 0.0094 to 0.0095 at every sun, and half the offset at 0.0242 to 0.0541. At the paper's excess the part is 1.268 to 1.279 times the bar.
- **The clearest node.** ȳ(131.8) is 0.15893 and g_max is 0.0768. A match at the 0.03 load would need a gas of at least 0.1289, and the largest of the eight is 0.1237.
- **The invariance.**
  - Scaling every node's amount by 0.5 or by 2 leaves the SE unchanged: the largest |ratio - 1| is 4.4e-16.
  - The control, raising the first visibility's α by 0.5, moves it by at least 0.0180.
  - Three copies of each spectrum divide it by √3, with a largest deviation of 5.7e-13.

## How it was built and reviewed

- **Tests first.** Commit `6b0d2dd` holds the tests alone, red: 127 of 131 then failed on the missing tool attribute or the unknown flag. `719fceb` holds the tool. v1's gas at 550 nm moved into one shared helper; `aerosol_depth_from_transmittance` computes exactly what it did, and a test pins that.
- **A blind mutation review** wrote 63 rows (`tests/teeth/feat-prague-depth.teeth.json`, `0178d88`): the conductor's 35 intents, authored before the pull request existed, 27 of its own, and one inert control.
  - 54 were killed, 53 by the predicted test; the remaining one was killed by another test first under `-x`, and its predicted test is red on its own. 8 survived, all of them predicted. The control held.
  - Intent I20 (an envelope taken the wrong way) has no site: the tool has no envelope, which belongs to the unregistered draft. I30 is equivalent: the mutant was run, and every output was bit-identical.
- **Blocking survivors.**
  - The printed sentence "g_4 = ..., not the old g_old = ..." was read by no test (X11, X12).
  - The amount-scaling invariance check could be a no-op (X5). Scaling by a power of two is exact in floating point, so the 1e-9 bound cannot tell a real check from nothing.
  - The three-copy check's construction, each copy with its own aerosol parameters, was graded only at its helper (I31). Its output is equivalent to within 1.5e-12, because exact copies give 1/√3 either way, but the criterion states the construction.
- **The fix round** (`d40e5f4`, `fea6912`, `171e01f`):
  - A test reads both numbers of that sentence.
  - Spy tests check the Jacobians the two invariance checks build.
  - The review's untrusted-input probes are closed. Fit channels are chosen by exact centre: a consistent +1 nm shift, a dropped channel or descending centres were accepted before, and are refused now. The loads' visibilities must be the criterion's. A condition number that is infinite or NaN prints instead of raising.
  - Three prose lines are pinned, and `PROVENANCE.md` lists `report --depth`.
  - Afterwards the full spec killed 62 of 63 rows and the control held.
- **The conductor re-injected** six rows from the committed spec: the four blocking ones (X5, X11, X12, I31), the s² denominator (I3) and the channel guard (X19). Each was applied as an exact find and replace with its anchor asserted to occur once, and only its predicted test was run. All six were killed by the predicted test, there was no broken mutant (pytest exit 2), and the tree was clean afterwards.
- **The gate** (`uv sync --frozen && uv run ruff check && uv run ruff format --check && uv run pytest && uv run lastlight privacy-scan`) passed: 3995 passed, 10 network-marked deselected, privacy-scan clean. Nothing under `src/lastlight/render` or the fixtures' JSON changed.

## What this does not decide

- Whether the aerosol amount contributes to the offset: no rule was registered.
- What is left after ozone, 0.17 to 0.21 of the sun-0 offset, still has no attribution.
- What would tell:
  - #57: the paper's Rayleigh, with Prague's own ozone taken from its transmittance aloft and fitted across the six nodes, under a rule whose positive branch is shown in its pilot to grow with the true excess;
  - the paper's own cross-sections;
  - libRadtran, run offline (GPL, never vendored or imported).
