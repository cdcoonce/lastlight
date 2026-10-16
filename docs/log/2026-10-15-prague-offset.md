# Renderer v1 against Prague at sun 0: what makes the sky-wide offset (#49)

**Result:** the four rules pre-registered in `docs/renderer/prague-offset-criterion.md` (landed in commit fb2fa6d, before Prague had been evaluated at sun -1 or +1) were implemented as written and run on the four committed fixtures. **Rule 1, ozone: contributes, branch 2.** All four ruled conditions (sun -1 and sun +1, at `aod_550` 0.03 and 0.1) read "contributes": Delta_O is 0.1084, 0.1075, 0.0847 and 0.0832, its share of the offset L is 0.329, 0.339, 0.326 and 0.338, and each Delta_O is over its bar (0.0093, 0.0277, 0.0158 and 0.0445) and under half of L. The pilot's two sun-0 conditions give "contributes", branch 2, beside it. **Rule 2, observer altitude and ground: no test.** Delta_H, the only reading that moves Prague's observer, is within 0.0128 of the stand-in for the 340 m slab of air and aerosol under it at every one of the eight conditions (Delta_H minus the stand-in runs from -0.0128 to +0.0077), so it measures the slab and not the ground. **Rule 3, Prague's fit near the horizon sun: not this, branch 3.** The bend kappa of the offset across sun -1, 0 and +1 is +0.0073 at 0.03 and -0.0142 at 0.1, and kappa_A under `xyz_altitude_0` is +0.0145 and -0.0133; the bar is the 0.05 floor at both loads. Sun 0's offset lies on the line through its neighbours, so a fault in Prague's fit at its sun-0 node does not explain it. **Rule 4, the aerosol's amount and profile: no test.** Matching Prague's aerosol depth removes Delta_M of +0.0401 at (sun -1, 0.1) and +0.0639 at (sun +1, 0.1), with ranges 0.0108 to 0.0695 and 0.0172 to 0.1106 if the depth estimate is 0.01 off; at 0.03 the match is clamped at every sun. What this attributes: ozone carries about a third of the offset and no more (shares 0.319 to 0.356 across sun -1, 0 and +1 at both loads), Prague's fit at its sun-0 node does not carry it, and the observer and the aerosol cannot be tested with Prague. After ozone, 0.2059 and 0.1724 of the sun-0 offset is left (0.2213, 0.2101, 0.1753 and 0.1626 at the four ruled conditions). That remainder has no attribution here. What would tell is a reference with v1's own geometry and aerosol (libRadtran, run offline; GPL, never vendored or imported). Renderer v1 is unchanged. The tool reproduced every number in the criterion's pilot section (table below); three of the four figures in the criterion's rule 2 evidence differ by one unit in the last digit, because the criterion subtracted rounded values (see "What broke, and what surprised").

## What was run, and how to reproduce it

- **Rules in the tool, tests first.** `tools/prague_reference.py` gained the rules as pure functions of numbers (`condition_verdict`, `rule_verdict`, `bend`, `reading_noise`, `fit_bar`, `offset_of`, `refuse_no_excess`) with every registered constant named (`OFFSET_AZIMUTHS_DEG`, `OFFSET_RULED_SUNS_DEG`, `OFFSET_BAR_FACTOR`, `CARRIES_SHARE`, `FIT_FLOOR`, `DEPTH_ERROR`, `SLAB_FACTOR`, `DEEPER_SCALE_HEIGHT_KM`). `tests/test_prague_offset.py` tests every branch and every boundary on exactly representable values. Its first commit is red (the tool had none of it); the next is green.
- **The ozone engine, the node check and the two specs, tests first.** `ozone_xyz` is v0's single scattering at the given ozone plus v1's multiple scattering continued by the table's own log-linear rule; it lives in the tool only and never asks v1 for an ozone outside its table. The tests for it use v0 and v1 alone, and they already reproduce the criterion's v0-and-v1-only numbers (Delta_O at sun -1 and +1, v1's own bend, the ozone line check, the slab stand-in, the deeper layer). `FixtureSpec` gained `sun_must_be_node` (False by default); the two new specs set it, and `build_fixture` then refuses a sun that is not a node of Prague's sun axis and of v1's table, naming the axis. This commit and the one before it contain no Prague number at sun -1 or +1.
- **The fixtures** `tests/fixtures/prague/sun-1.json` and `sun1.json` were written after the rules, the engine and the node check were committed: `uv run python tools/prague_reference.py fixture --sun -1` and `fixture --sun 1` (about 30 seconds each here, the model and datasets already in `~/.cache/lastlight/prague/`; nothing was downloaded). Relative azimuths 0, 30, 90 and 180, the same eight views and the same two loads as `sun0.json`, every diagnostic reading, the sun recorded.
- **The numbers** are `uv run python tools/prague_reference.py report --offset`, from the four committed fixtures alone (about 3 seconds). The tables below are copied from its output. Nothing in this log was computed by hand.
- **Commit order is the evidence.**

  | commit | what it holds |
  | --- | --- |
  | `5405f0d` | red tests for the rule functions |
  | `9557fa3` | the rule functions |
  | `2aa2411` | red tests for the ozone engine, the node check and the two specs |
  | `b3bbabe` | the ozone engine, the node check, the two specs |
  | `2680f90` | the two fixtures, generated: the first Prague numbers at sun -1 and +1 |
  | `b899cbc` | `report --offset`, the tests that pin its numbers, and `PROVENANCE.md` |

- **`sun10.json` and `sun0.json` are unchanged.** Each was regenerated to a scratch path with `--output` and compared with `cmp`: both are identical to the committed files. `git diff` against `origin/main` shows neither file, and nothing under `src/`. The network-marked `fixture --check` for each of the four fixtures passes (4 passed, using the cached model; no download).
- **The earlier reports are unchanged.** `report` and `report --sun 0` were run before the first edit (on `origin/main`) and after the last; `cmp` finds each output byte-identical.
- **Checks:** the gate (`uv sync --frozen && uv run ruff check && uv run ruff format --check && uv run pytest && uv run lastlight privacy-scan`) passed before this log was added: 3795 passed, 10 network-marked deselected, privacy-scan clean. The same gate is run again with this log in the tree.
- **A mutation pass** applied 34 single-line mutants to the tool, one at a time, and ran the offset tests against each. 33 were killed at once. One survived: the verdict printed beside rule 1, which must come from the pilot's sun-0 conditions, could be computed from sun +10 and no test noticed, because all four pilot conditions read "contributes". A test was added in which sun 0 is made to carry while sun +10 does not, and that mutant is now killed.

## Rule 1: ozone

Delta_O is the part of the offset that goes when v1 is given Prague's ozone: v1 against the ozone engine at 345 DU, the U.S. Standard column Prague uses, instead of v1's 280 DU. The bar is 2m, with m the shift of the offset when Prague's aerosol is read in 1 / V.

| condition | Delta_O | L | bar 2m | share Delta_O / L | verdict |
| --- | --- | --- | --- | --- | --- |
| (sun -1, 0.03) | 0.1084 | 0.3297 | 0.0093 | 0.329 | contributes |
| (sun -1, 0.1) | 0.1075 | 0.3176 | 0.0277 | 0.339 | contributes |
| (sun +1, 0.03) | 0.0847 | 0.2601 | 0.0158 | 0.326 | contributes |
| (sun +1, 0.1) | 0.0832 | 0.2458 | 0.0445 | 0.338 | contributes |

**Verdict: contributes (branch 2).** The pilot's two sun-0 conditions, under the same rule-level verdict, give contributes (branch 2). Every condition has its Delta_O between 0.0832 and 0.1084, over a bar of at most 0.0445 and under half of L, so none carries and none is "not this". The criterion said the rule's "not this" cannot fire in practice; the reading noise is at most 0.0223 in the ruled conditions, against Delta_O / 2 of at least 0.0416. "Not this" and "cannot tell" were reached on stand-ins only (see below).

Left after ozone at each ruled condition (from the report):

| condition | left |
| --- | --- |
| (sun -1, 0.03) | 0.2213 |
| (sun -1, 0.1) | 0.2101 |
| (sun +1, 0.03) | 0.1753 |
| (sun +1, 0.1) | 0.1626 |

## Rule 2: no test

The criterion registers no rule for the observer's altitude and ground, and says why: `xyz_altitude_0` also puts Prague's observer under 340 m more air and aerosol, so Delta_H measures that slab. The slab's stand-in is v1 with `aod_550` raised by exp(0.34 / 1.5), a factor of 1.2544 (printed in the report's rule 2 text). Delta_H and the stand-in at every condition of the four fixtures:

| condition | Delta_H | slab stand-in, change in the mean of ln Y_v1 | Delta_H minus stand-in |
| --- | --- | --- | --- |
| (sun +10, 0.03) | -0.0188 | -0.0265 | +0.0077 |
| (sun +10, 0.1) | -0.0618 | -0.0597 | -0.0022 |
| (sun 0, 0.03) | -0.0683 | -0.0684 | +0.0000 |
| (sun 0, 0.1) | -0.0962 | -0.0894 | -0.0069 |
| (sun -1, 0.03) | -0.0509 | -0.0447 | -0.0062 |
| (sun -1, 0.1) | -0.0785 | -0.0658 | -0.0128 |
| (sun +1, 0.03) | -0.0713 | -0.0671 | -0.0042 |
| (sun +1, 0.1) | -0.1123 | -0.1114 | -0.0010 |

Delta_H and the stand-in agree to within 0.0128 at all eight conditions, which is the criterion's point: Delta_H is the slab's extinction, not the ground's position. No verdict is printed, and none is claimed. What would tell: Prague with its ground raised to the observer, which the model cannot do, or a reference with v1's own geometry (libRadtran, run offline; GPL, never vendored or imported).

## Rule 3: Prague's fit near the horizon sun

kappa is how far sun 0's offset sits above the straight line through its neighbours at sun -1 and +1; kappa_A is the same under `xyz_altitude_0`. The removed part is the smaller of the two, so a fault is called only if both readings show it. The bar is the larger of twice the largest reading noise and 0.05.

| `aod_550` | L(-1) | L(0) | L(+1) | L_A(-1) | L_A(0) | L_A(+1) | m(-1) | m(0) | m(+1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | 0.3297 | 0.3022 | 0.2601 | 0.3806 | 0.3705 | 0.3313 | 0.0047 | 0.0072 | 0.0079 |
| 0.1 | 0.3176 | 0.2675 | 0.2458 | 0.3961 | 0.3638 | 0.3581 | 0.0139 | 0.0181 | 0.0223 |

| `aod_550` | kappa | kappa_A | removed, min(kappa, kappa_A) | offset L(0) | bar max(2 m_F, 0.05) | share removed / L(0) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | +0.0073 | +0.0145 | +0.0073 | 0.3022 | 0.0500 | +0.024 | not this |
| 0.1 | -0.0142 | -0.0133 | -0.0142 | 0.2675 | 0.0500 | -0.053 | not this |

**Verdict: not this (branch 3).** At 0.03 the bend is +0.0073 (kappa_A +0.0145), well under the 0.05 floor. At 0.1 it is negative under both readings (-0.0142 and -0.0133): sun 0 sits slightly below the line through its neighbours. The largest m is 0.0079 at 0.03 and 0.0223 at 0.1, so twice it is under the floor and the floor is the bar at both loads. The offset falls smoothly with the sun at both loads (0.3297, 0.3022, 0.2601 at 0.03; 0.3176, 0.2675, 0.2458 at 0.1), which is what a cause in the physics does, and not what a fault in the fit at one node does.

The bends reported beside it (kappa for Delta_O, for L, kappa_A for L_A, then the same after removing ozone), from the report:

| `aod_550` | kappa for Delta_O | kappa for L | kappa_A for L_A | kappa for L - Delta_O | kappa_A for L_A - Delta_O |
| --- | --- | --- | --- | --- | --- |
| 0.03 | -0.0003 | +0.0073 | +0.0145 | +0.0076 | +0.0148 |
| 0.1 | -0.0002 | -0.0142 | -0.0133 | -0.0140 | -0.0131 |

Removing ozone leaves the bend where it was (+0.0076 and +0.0148 at 0.03, -0.0140 and -0.0131 at 0.1): ozone's own bend across the three suns is -0.0003 and -0.0002.

## Rule 4: no test

The criterion registers no rule for the aerosol's amount and profile, and says why: the depth estimate's error is nearly the whole gap, and the load response does not separate the aerosol from the air. At every condition of the four fixtures:

| condition | Delta_M = L - L_matched | Prague's depth, primary | Prague's depth at the match | Delta_M if the depth estimate is 0.01 off either way |
| --- | --- | --- | --- | --- |
| (sun +10, 0.03) | clamped | 0.0407 | 0.0369 | clamped (no range) |
| (sun +10, 0.1) | +0.0375 | 0.1137 | 0.1000 | 0.0101 to 0.0648 |
| (sun 0, 0.03) | clamped | 0.0407 | 0.0369 | clamped (no range) |
| (sun 0, 0.1) | +0.0521 | 0.1137 | 0.1000 | 0.0140 to 0.0903 |
| (sun -1, 0.03) | clamped | 0.0407 | 0.0369 | clamped (no range) |
| (sun -1, 0.1) | +0.0401 | 0.1137 | 0.1000 | 0.0108 to 0.0695 |
| (sun +1, 0.03) | clamped | 0.0407 | 0.0369 | clamped (no range) |
| (sun +1, 0.1) | +0.0639 | 0.1137 | 0.1000 | 0.0172 to 0.1106 |

| sun | load step L(0.1) - L(0.03) | deeper layer at 0.03 | deeper layer at 0.1 | ratio 0.1 to 0.03 |
| --- | --- | --- | --- | --- |
| +10 | +0.0319 | +0.0166 | +0.0405 | 2.44 |
| 0 | -0.0347 | -0.0156 | -0.0576 | 3.68 |
| -1 | -0.0121 | -0.0434 | -0.1435 | 3.31 |
| +1 | -0.0143 | -0.0061 | -0.0115 | 1.89 |

At 0.03 the match is clamped at Prague's clearest visibility at all four suns (Prague's depth there is 0.0369, against v1's 0.03), so no Delta_M exists. At 0.1, Delta_M is +0.0375 to +0.0639 across the four suns and every range at +-0.01 is wide: at (sun 0, 0.1) it runs from 0.0140 to 0.0903 around the bar of 0.0362, as the criterion said. The deeper layer changes the mean of ln Y by an amount that differs in sign between sun +10 (+0.0405 at 0.1) and sun 0 (-0.0576), and its growth from 0.03 to 0.1 is 3.31, 3.68 and 1.89 at sun -1, 0 and +1. No verdict is printed. What would tell: the depth estimate's error, derived from Prague's own gas depths (fitted from its spectral transmittance), which may make the amount testable; and a reference with v1's own aerosol (libRadtran, run offline).

## Reported, never ruled

For every condition of the four fixtures, over P (relative azimuths 90 and 180, all eight views: 16 points) and at each relative azimuth the fixture has, so the azimuths toward the sun are in the same table as those away from it. `sun10.json` has no azimuth 30.

### Sun +10 (`sun10.json`)

| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L | Delta_O (v0) | line check | L_binned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | P (azimuths 90, 180) (16) | 0.1006 | 0.1194 | 0.0036 | 0.0294 | -0.0188 | 0.293 | -0.187 | 0.0297 | 1.2992 | 0.1215 |
| 0.03 | azimuth 0 alone (8) | 0.0824 | 0.0520 | 0.0077 | 0.0286 | +0.0304 | 0.347 | +0.369 | 0.0286 | 1.2994 | 0.0881 |
| 0.03 | azimuth 90 alone (8) | 0.0688 | 0.0789 | 0.0014 | 0.0288 | -0.0101 | 0.419 | -0.146 | 0.0291 | 1.2992 | 0.0889 |
| 0.03 | azimuth 180 alone (8) | 0.1323 | 0.1599 | 0.0059 | 0.0300 | -0.0276 | 0.227 | -0.208 | 0.0303 | 1.2991 | 0.1541 |
| 0.1 | P (azimuths 90, 180) (16) | 0.1325 | 0.1943 | 0.0130 | 0.0292 | -0.0618 | 0.220 | -0.467 | 0.0297 | 1.2993 | 0.1484 |
| 0.1 | azimuth 0 alone (8) | 0.3331 | 0.3746 | 0.0043 | 0.0292 | -0.0415 | 0.088 | -0.125 | 0.0293 | 1.2994 | 0.3325 |
| 0.1 | azimuth 90 alone (8) | 0.0976 | 0.1476 | 0.0077 | 0.0288 | -0.0499 | 0.295 | -0.511 | 0.0294 | 1.2993 | 0.1110 |
| 0.1 | azimuth 180 alone (8) | 0.1673 | 0.2411 | 0.0184 | 0.0295 | -0.0738 | 0.176 | -0.441 | 0.0301 | 1.2992 | 0.1858 |

### Sun 0 (`sun0.json`)

| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L | Delta_O (v0) | line check | L_binned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | P (azimuths 90, 180) (16) | 0.3022 | 0.3705 | 0.0072 | 0.0963 | -0.0683 | 0.319 | -0.226 | 0.1019 | 1.2963 | 0.3092 |
| 0.03 | azimuth 0 alone (8) | 0.1248 | 0.2107 | 0.0117 | 0.0833 | -0.0859 | 0.668 | -0.689 | 0.0835 | 1.2982 | 0.1055 |
| 0.03 | azimuth 30 alone (8) | 0.2147 | 0.2906 | 0.0101 | 0.0847 | -0.0759 | 0.394 | -0.354 | 0.0852 | 1.2980 | 0.2077 |
| 0.03 | azimuth 90 alone (8) | 0.3007 | 0.3676 | 0.0077 | 0.0929 | -0.0669 | 0.309 | -0.223 | 0.0974 | 1.2972 | 0.3077 |
| 0.03 | azimuth 180 alone (8) | 0.3036 | 0.3733 | 0.0067 | 0.0997 | -0.0697 | 0.329 | -0.230 | 0.1063 | 1.2955 | 0.3107 |
| 0.1 | P (azimuths 90, 180) (16) | 0.2675 | 0.3638 | 0.0181 | 0.0951 | -0.0962 | 0.356 | -0.360 | 0.1096 | 1.2960 | 0.2843 |
| 0.1 | azimuth 0 alone (8) | 0.3745 | 0.5642 | 0.0427 | 0.0852 | -0.1897 | 0.228 | -0.507 | 0.0855 | 1.2982 | 0.3587 |
| 0.1 | azimuth 30 alone (8) | 0.3195 | 0.4654 | 0.0315 | 0.0860 | -0.1459 | 0.269 | -0.457 | 0.0872 | 1.2980 | 0.3190 |
| 0.1 | azimuth 90 alone (8) | 0.2864 | 0.3874 | 0.0197 | 0.0917 | -0.1010 | 0.320 | -0.353 | 0.1017 | 1.2972 | 0.3021 |
| 0.1 | azimuth 180 alone (8) | 0.2486 | 0.3401 | 0.0164 | 0.0986 | -0.0915 | 0.397 | -0.368 | 0.1175 | 1.2950 | 0.2665 |

### Sun -1 (`sun-1.json`)

| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L | Delta_O (v0) | line check | L_binned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | P (azimuths 90, 180) (16) | 0.3297 | 0.3806 | 0.0047 | 0.1084 | -0.0509 | 0.329 | -0.154 | 0.1198 | 1.2950 | 0.3454 |
| 0.03 | azimuth 0 alone (8) | 0.0604 | 0.1541 | 0.0123 | 0.0933 | -0.0937 | 1.545 | -1.552 | 0.0932 | 1.2974 | 0.0440 |
| 0.03 | azimuth 30 alone (8) | 0.2569 | 0.3367 | 0.0102 | 0.0960 | -0.0798 | 0.374 | -0.311 | 0.0968 | 1.2972 | 0.2538 |
| 0.03 | azimuth 90 alone (8) | 0.3643 | 0.4264 | 0.0066 | 0.1069 | -0.0621 | 0.294 | -0.171 | 0.1146 | 1.2959 | 0.3745 |
| 0.03 | azimuth 180 alone (8) | 0.2951 | 0.3347 | 0.0027 | 0.1099 | -0.0397 | 0.372 | -0.134 | 0.1250 | 1.2942 | 0.3163 |
| 0.1 | P (azimuths 90, 180) (16) | 0.3176 | 0.3961 | 0.0139 | 0.1075 | -0.0785 | 0.339 | -0.247 | 0.1345 | 1.2949 | 0.3391 |
| 0.1 | azimuth 0 alone (8) | 0.2666 | 0.4523 | 0.0392 | 0.0954 | -0.1858 | 0.358 | -0.697 | 0.0952 | 1.2975 | 0.2560 |
| 0.1 | azimuth 30 alone (8) | 0.3104 | 0.4520 | 0.0297 | 0.0975 | -0.1416 | 0.314 | -0.456 | 0.0990 | 1.2973 | 0.3141 |
| 0.1 | azimuth 90 alone (8) | 0.3366 | 0.4297 | 0.0177 | 0.1050 | -0.0932 | 0.312 | -0.277 | 0.1199 | 1.2959 | 0.3545 |
| 0.1 | azimuth 180 alone (8) | 0.2986 | 0.3625 | 0.0101 | 0.1100 | -0.0639 | 0.368 | -0.214 | 0.1491 | 1.2941 | 0.3236 |

### Sun +1 (`sun1.json`)

| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L | Delta_O (v0) | line check | L_binned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.03 | P (azimuths 90, 180) (16) | 0.2601 | 0.3313 | 0.0079 | 0.0847 | -0.0713 | 0.326 | -0.274 | 0.0887 | 1.2971 | 0.2660 |
| 0.03 | azimuth 0 alone (8) | 0.1382 | 0.2063 | 0.0075 | 0.0751 | -0.0682 | 0.544 | -0.493 | 0.0754 | 1.2985 | 0.1204 |
| 0.03 | azimuth 30 alone (8) | 0.1659 | 0.2284 | 0.0073 | 0.0751 | -0.0624 | 0.453 | -0.376 | 0.0758 | 1.2984 | 0.1582 |
| 0.03 | azimuth 90 alone (8) | 0.2447 | 0.3094 | 0.0074 | 0.0806 | -0.0647 | 0.329 | -0.264 | 0.0837 | 1.2979 | 0.2512 |
| 0.03 | azimuth 180 alone (8) | 0.2754 | 0.3533 | 0.0084 | 0.0889 | -0.0779 | 0.323 | -0.283 | 0.0937 | 1.2965 | 0.2807 |
| 0.1 | P (azimuths 90, 180) (16) | 0.2458 | 0.3581 | 0.0223 | 0.0832 | -0.1123 | 0.338 | -0.457 | 0.0926 | 1.2971 | 0.2590 |
| 0.1 | azimuth 0 alone (8) | 0.4172 | 0.6221 | 0.0446 | 0.0766 | -0.2049 | 0.184 | -0.491 | 0.0772 | 1.2986 | 0.3954 |
| 0.1 | azimuth 30 alone (8) | 0.3080 | 0.4671 | 0.0338 | 0.0764 | -0.1591 | 0.248 | -0.516 | 0.0775 | 1.2985 | 0.2998 |
| 0.1 | azimuth 90 alone (8) | 0.2553 | 0.3651 | 0.0223 | 0.0797 | -0.1097 | 0.312 | -0.430 | 0.0859 | 1.2980 | 0.2673 |
| 0.1 | azimuth 180 alone (8) | 0.2362 | 0.3511 | 0.0222 | 0.0867 | -0.1149 | 0.367 | -0.486 | 0.0994 | 1.2963 | 0.2507 |

What the tables show, without ruling on any of it:

- Toward the sun the share is not steady. At azimuth 0 alone, Delta_O / L is 1.545 at (sun -1, 0.03), where L is 0.0604 and Delta_O is 0.0933 (Prague's ozone alone would put v1 below Prague there), and 0.088 at (sun +10, 0.1).
- Away from the sun the share is steady: Delta_O / L is between 0.294 and 0.372 at azimuths 90 and 180 at sun -1, and between 0.309 and 0.329 at sun 0, 0.03.
- Delta_O from v0's single scattering alone (the column "Delta_O (v0)") is larger than the full Delta_O in the P rows of all eight conditions (0.1345 against 0.1075 at (sun -1, 0.1)): the multiple scattering is less sensitive to ozone than the single scattering.
- The ozone line check, the ratio of v0's response to 345 DU over its response to 330 DU, is between 1.2941 and 1.2994 in every row, against 1.3 for a straight line. The continuation is a straight line in v0 to within that.

## Did the tool reproduce the pilot?

Yes, every number of the criterion's pilot section, to within half a unit of its last digit (the tests check each one that way, and the report prints them under its "pilot" heading). The report's three pilot tables:

| condition | L | m | 2m | Delta_O, ozone | Delta_H, altitude 0 |
| --- | --- | --- | --- | --- | --- |
| (sun +10, 0.03) | 0.1006 | 0.0036 | 0.0072 | 0.0294 | -0.0188 |
| (sun +10, 0.1) | 0.1325 | 0.0130 | 0.0260 | 0.0292 | -0.0618 |
| (sun 0, 0.03) | 0.3022 | 0.0072 | 0.0144 | 0.0963 | -0.0683 |
| (sun 0, 0.1) | 0.2675 | 0.0181 | 0.0362 | 0.0951 | -0.0962 |

| condition | share Delta_O / L | margin over the bar, Delta_O / 2m | verdict | a Prague column under (DU) turns it into "not this" | left after ozone, L - Delta_O |
| --- | --- | --- | --- | --- | --- |
| (sun +10, 0.03) | 0.293 | 4.08 | contributes | 295.91 | 0.0712 |
| (sun +10, 0.1) | 0.220 | 1.12 | contributes | 337.94 | 0.1033 |
| (sun 0, 0.03) | 0.319 | 6.68 | contributes | 289.67 | 0.2059 |
| (sun 0, 0.1) | 0.356 | 2.63 | contributes | 304.64 | 0.1724 |

| condition | Delta_O | carries only if L is at most 2 Delta_O | "not this" only if m is at least Delta_O / 2 | that m in multiples of the pilot's largest m (0.0181) |
| --- | --- | --- | --- | --- |
| (sun -1, 0.03) | 0.1084 | 0.2168 | 0.0542 | 3.0 |
| (sun -1, 0.1) | 0.1075 | 0.2150 | 0.0538 | 3.0 |
| (sun +1, 0.03) | 0.0847 | 0.1695 | 0.0424 | 2.3 |
| (sun +1, 0.1) | 0.0832 | 0.1664 | 0.0416 | 2.3 |

The criterion against the tool, number by number:

| quantity | the criterion | the tool |
| --- | --- | --- |
| L, m, 2m, Delta_O, Delta_H at (sun +10, 0.03) | 0.1006, 0.0036, 0.0072, 0.0294, -0.0188 | 0.1006, 0.0036, 0.0072, 0.0294, -0.0188 |
| the same at (sun +10, 0.1) | 0.1325, 0.0130, 0.0260, 0.0292, -0.0618 | 0.1325, 0.0130, 0.0260, 0.0292, -0.0618 |
| the same at (sun 0, 0.03) | 0.3022, 0.0072, 0.0144, 0.0963, -0.0683 | 0.3022, 0.0072, 0.0144, 0.0963, -0.0683 |
| the same at (sun 0, 0.1) | 0.2675, 0.0181, 0.0362, 0.0951, -0.0962 | 0.2675, 0.0181, 0.0362, 0.0951, -0.0962 |
| share of the offset at sun 0 (0.03, 0.1) | 0.319, 0.356 | 0.319, 0.356 |
| share of the offset at sun +10 (0.03, 0.1) | 0.29, 0.22 | 0.293, 0.220 |
| margin over the bar at sun 0 (0.03, 0.1) | 6.68, 2.63 | 6.68, 2.63 |
| margin over the bar at sun +10 (0.03, 0.1) | 4.08, 1.12 | 4.08, 1.12 |
| the Prague column under which (sun +10, 0.1) turns into "not this" | 337.94 DU | 337.94 DU |
| left after ozone at sun 0 (0.03, 0.1) | 0.2059, 0.1724 | 0.2059, 0.1724 |
| Delta_O at sun -1 (0.03, 0.1), from v0 and v1 alone | 0.1084, 0.1075 | 0.1084, 0.1075 |
| Delta_O at sun +1 (0.03, 0.1), from v0 and v1 alone | 0.0847, 0.0832 | 0.0847, 0.0832 |
| v1's own bend at sun 0 (0.03, 0.1) | +0.0996, +0.0543 | +0.0996, +0.0543 |
| rule 1 carries a condition only if L is at most (sun -1: 0.03, 0.1; sun +1: 0.03, 0.1) | 0.2168, 0.2150, 0.1695, 0.1664 | 0.2168, 0.2150, 0.1695, 0.1664 |
| "not this" only if m is at least (the same four) | 0.0542, 0.0538, 0.0424, 0.0416 | 0.0542, 0.0538, 0.0424, 0.0416 |
| that m in multiples of the pilot's largest m, 0.0181 | 3.0, 3.0, 2.3, 2.3 | 3.0, 3.0, 2.3, 2.3 |

Other figures the criterion quotes, outside its pilot section:

| quantity | the criterion | the tool |
| --- | --- | --- |
| the slab stand-in at (sun 0, 0.03), (sun 0, 0.1), (sun +10, 0.03), (sun +10, 0.1) | -0.0684, -0.0894, -0.0265, -0.0597 | -0.0684, -0.0894, -0.0265, -0.0597 |
| Delta_H minus the stand-in, same four | +0.0001, -0.0068, +0.0077, -0.0021 | +0.0000, -0.0069, +0.0077, -0.0022 |
| ozone's own bend at 0.03 | -0.0003 | -0.0003 |
| Delta_M at (sun 0, 0.1), (sun +10, 0.1) | 0.0521, 0.0375 | +0.0521, +0.0375 |
| Prague's aerosol depth at 0.03 and 0.1; the clamped depth at 0.03 | 0.0407, 0.1137; 0.0369 | 0.0407, 0.1137; 0.0369 |
| Delta_M's range at (sun 0, 0.1) if the estimate is 0.01 off | 0.0140 to 0.0903 | 0.0140 to 0.0903 |
| the deeper layer at (sun +10, 0.1) and (sun 0, 0.1) | +0.0405, -0.0576 | +0.0405, -0.0576 |
| the deeper layer's growth from 0.03 to 0.1 at sun -1, 0, +1 | 3.31, 3.68, 1.89 | 3.31, 3.68, 1.89 |
| the ozone line check at (sun 0, 0.03), (sun 0, 0.1), (sun +10, 0.03), (sun +10, 0.1) | 1.2963, 1.2960, 1.2992, 1.2993 | 1.2963, 1.2960, 1.2992, 1.2993 |

Not tool numbers, so not reproduced: the third cold read's estimate of L(+1) (near 0.215 and 0.222), the bends it modelled for smooth differences (at most 0.025, +0.0167 and +0.0006, and the ozone one reproduced above), its emulation of the elevated observer, and its fit of Prague's gas depths. The estimate of L(+1) turned out low: the tool gives 0.2601 at 0.03 and 0.2458 at 0.1 (rule 1 table), against carries bars of 0.1695 and 0.1664.

## Decisions, each with its rejected alternative

- **The rules are pure functions of numbers.** `condition_verdict` takes a removed part, an offset and a bar; `rule_verdict` takes the conditions' verdicts. Every boundary is then tested on exactly representable values: a part exactly half the offset (carries), exactly at the bar (not this, even if it is half the offset), a bar of exactly 0.05, a noise of exactly half the floor. Rejected: running the rules only through the fixtures, where no real number lands on a bound.
- **Commit order is evidence.** Red rule tests, rule code, red engine tests, engine, fixtures, report and pins, log. Rejected: fixtures first, which would put a ruled number in view of the bars.
- **The refusal is its own function and runs before any verdict.** `refuse_no_excess` names the condition and the reading, and `evaluate_offset` checks every L and L_A a rule uses (L at the four ruled conditions, L and L_A at sun -1, 0 and +1 under both loads) before computing anything else. The report text is complete before anything is written, so a refusal leaves standard output empty and exits 1. Rejected: dropping a condition (the criterion forbids it), and a check inside `condition_verdict`, which has no condition or reading to name.
- **The ozone engine is in the tool only and reads the table's ozone nodes from v1's `DOMAIN`.** Rejected: an ozone parameter on v1, or relaxing v1's domain check (v1's physics is pre-registered, definition 8 of `docs/gates/renderer-v1/criterion.md`).
- **The node check is a field on the spec, `sun_must_be_node`, False by default.** Rejected: checking every spec, which would refuse sun +10 (not a node of Prague's axis); hard-coding the two suns; and checking only Prague's axis (sun -0.5 is one of Prague's nodes and not one of v1's).
- **Points are selected by value.** P is "azimuth in {90, 180}", found in the fixture by value to 1e-9. A test runs both rules on copies of all four fixtures with the azimuths and views listed in the opposite order and gets the same numbers. Rejected: positional masks.
- **Evaluation and printing are separate** (`evaluate_offset` returns the numbers, `format_offset` prints them), so the tests pin numbers by value and the log copies the printed tables. Rejected: pinning text only.
- **Another tool wrote the fixtures, so `check_offset_fixture` runs before any field is read.** It checks types (a bool is not a number), finiteness, shapes, the recorded sun, the azimuths and every luminance (finite and above 0). 33 malformed variants are tested. Rejected: trusting a file because the same tool wrote it.
- **Every verdict is reached on a stand-in for Prague.** For rule 3, Prague is v1 scaled by exp(-offset) at each of the three suns, under every reading, so L is the chosen offset and m is 0; the stand-ins show carries (branch 1), contributes (2), not this (3) and cannot tell (4), the floor binding on a small bend, the smaller of kappa and kappa_A deciding, and the noise at sun -1, 0 or +1 each raising the bar. For rule 1, Prague is the ozone engine at 345 DU scaled by exp(-residual), so L is Delta_O plus the residual: carries (1), contributes (2), cannot tell (4, one condition's m over half of Delta_O), and not this (3, as a code path only). Rejected: replacing only the primary reading, which would leave m nonzero and unknown.
- **A carries and a contributes together are "contributes", as the rule-level verdict says.** The criterion's prediction paragraph calls "sun +1 carries, sun -1 contributes" a realistic "cannot tell"; its rule-level verdict says every condition carrying or contributing is branch 2. The tool follows the rule-level verdict and a test says so. Rejected: reading the prediction paragraph as a fifth branch.
- **`report --offset` is a flag of `report` and reads all four fixtures.** `--sun` is not used with it; `report --sun -1` stops with a message pointing at `--offset`. Rejected: a per-sun report for the new suns (the rules need all four at once).
- **The Prague column that turns (sun +10, 0.1) into "not this" is found by root finding** (`scipy.optimize.brentq` on Delta_O against the column), so the criterion's 337.94 DU is a computed number. Rejected: quoting it.
- **Pins are by value** with `pytest.approx(x, rel=1e-6, abs=0)`, plus exact verdict strings and branch numbers; the criterion's printed values are checked as "within half a unit of the last digit shown". Rejected: a hash of float output, and pytest's default absolute tolerance, which would void a relative bar on values this small.

## What broke, and what surprised

- **The first red commit carried a lint error.** `ruff` reads `tool.SLAB_FACTOR` (an upper-case attribute) as a constant and flags the comparison as a Yoda condition. A `;` after a failed `&&` chain let the commit through; it was amended before anything else existed. The same rule flagged five more comparisons later, all fixed by `ruff --fix`.
- **One mutant survived**, described above: the pilot's verdict beside rule 1 could come from the wrong sun, because all four pilot conditions read "contributes". Fixed with a test that makes sun 0 carry.
- **The criterion's Delta_H minus the stand-in are differences of rounded values.** It prints +0.0001, -0.0068, +0.0077 and -0.0021; the tool, from unrounded values, gives +0.0000, -0.0069, +0.0077 and -0.0022. Three differ by one unit in the fourth decimal. These figures are in the rule 2 section, not in the pilot section, whose numbers all agree to half a unit. A test records the one-unit agreement.
- **The criterion's prediction paragraph and its rule-level verdict disagree** about "cannot tell" (see Decisions). It does not matter for the result, because every ruled condition contributes.
- **The cold-read estimate of L(+1) was low** (0.215 and 0.222 against the tool's 0.2601 and 0.2458). The "cannot tell" outcome it warned of did not happen: at sun +1 neither condition is near carrying (L(+1) would have to be at most 0.1695 or 0.1664).
- **Rule 3 came out clean.** The offset falls with the sun at both loads (0.3297, 0.3022, 0.2601 at 0.03), and sun 0 sits on the line at 0.03 and slightly below it at 0.1. The criterion built a floor of 0.05 so that a smooth bend would not be read as a fault; the largest kappa is +0.0145, against that floor.
- **Ozone over-explains the offset toward the sun at sun -1**: the share is 1.545 at azimuth 0, 0.03. Away from the sun it stays between 0.294 and 0.372 at sun -1. This is reported, not ruled.
- **Everything was fast.** The whole report takes about 3 seconds and each fixture about 30 seconds.
- **The host's load stayed low** (a one-minute average under 6 at each check), so the suite ran in one piece: 211 seconds.

## Known limits

- **Rules 2 and 4 have no test.** The observer's height and the aerosol are not attributed. Of the sun-0 offset, 0.2059 and 0.1724 remain after ozone, and no rule here can say what they are.
- **Prague's ozone is assumed to be the U.S. Standard profile's column, 345 DU, and was not measured.** Delta_O moves with it (the report prints the column at which each pilot condition flips). The continuation of v1's multiple scattering 15 DU past its last node is a straight line in the logarithm; the criterion bounds the table's interpolation error at 0.49 percent, and the line check above shows v0's own response is straight to within the printed ratios. Nothing checks the multiple scattering's response beyond 330 DU.
- **Rule 3 cannot see** a fault shared by every near-horizon node, a fault at one observer altitude and not the other, or a kink in v1 at its own sun-0 node (v1's table has a node at 0 too). "Not this" says sun 0 is on the line, not that Prague is right.
- **Two loads, one albedo (0.2), one ozone.** The loads are v1's 0.03 and 0.1; Prague at its Koschmieder visibilities holds aerosol depths of 0.0407 and 0.1137.
- **Sun -1, 0 and +1 are nodes of both models; sun -0.5 is not a node of v1's table**, so it was not ruled at.
- **Reading noise m is the shift under one reading** (inverse visibility), as registered; the spectral reading `xyz_binned` is reported and is never part of it.
- **The datasets carry no licence statement of their own that I could find;** nothing of them is committed. `tests/fixtures/prague/PROVENANCE.md` describes all four fixtures.

## Next

The conductor decides what "contributes" means for a new renderer: v1 is unchanged, and a change to its ozone (a column of 345 DU, a continuation past the table) would be a new registration (#50 is the first). The larger part of the sun-0 offset is still unattributed. The one follow-up that could tell is a reference with v1's own geometry and aerosol, libRadtran run offline, which would also give rule 2 and rule 4 their tests; it is GPL and is never vendored or imported.
