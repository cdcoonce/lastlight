# Prague Sky Model daylight control for renderer v1 (#34)

**Result:** renderer v1 **passes** the Prague control at all three relative azimuths, with room to spare. At sun +10 degrees in the low-aerosol atmosphere (`aod_550` 0.03), v1's lowest-band `log(R/G)` and `log(B/G)` contrast against the reference band is within 0.056 of Prague's (the tolerance is 0.10) and the reference elevation's chromaticity is within 0.0052 of Prague's in CIE 1976 u'v' (the tolerance is 0.02). No comparison needed the `xfail(strict=True)` the issue prescribes for a miss; the renderer and the tolerances were not touched. v0 against the same fixture is worse, and misses 0.10 and 0.02 at azimuths 90 and 180 (reported, never a pass/fail), so the control can fail. The absolute level is a separate matter: in the low-aerosol atmosphere v1's luminance runs from 4 percent below to 19 percent above Prague's (v0's from 38 percent below to 8 percent above), which the contrast control, being scale-free, does not see (see "What the control does not cover"). The `aod_550` 0.1 question (does Prague side with the Bruneton port or with the tool's Monte Carlo in the 6.6 percent gap?) **cannot be told** from this comparison, and the section "The `aod_550` 0.1 question" says why and what would tell.

## What was obtained, and how

- **Model:** the authors' reference implementation, https://github.com/PetrVevoda/pragueskymodel at commit `2385c912e9051c1258013ff8c3ce2e19e10fb917` (2023-08-27, the repository's last commit; no release tags). Apache-2.0, Copyright 2022 Charles University. Cloned outside the repository into `~/.cache/lastlight/prague/src/`.
- **Datasets:** from the Google Drive links in that repository's README. No account, no payment, no acceptance of terms beyond the model's Apache-2.0 (the README states no separate licence for the datasets). Google Drive shows a "can't scan this file for viruses" page for files this large; the tool submits that page's form, which is what its "Download anyway" button does.
  - `SkyModelDataset.dat`, the full dataset: 2397370196 bytes (2.2 GiB), SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`.
  - `SkyModelDatasetGround.dat`, the ground-level dataset: 107632816 bytes (103 MB), SHA-256 `76bd619dc6dfcbc900c2996436dd6cf68197c03e3982fe5580f2b109ce1c71c2`.
- **Build:** no CMake on this Mac, and none needed: `PragueSkyModel.cpp` and `.h` are "all you need to use the model", so the tool compiles them with the system `c++` (Apple clang 21, C++17) together with a 70-line driver of ours that calls the model's public API. The driver is a string inside `tools/prague_reference.py`; the Prague source is never copied into the repository. The GPL `skylight` port was neither used nor imported; libRadtran was not used.
- **Total spend:** zero. `tools/prague_reference.py fetch build fixture` rebuilds everything (about 4 minutes after the 2.5 GB of downloads, which took 4 more); `fixture --check` re-evaluates and compares (the network-marked test `test_the_fixture_matches_a_fresh_evaluation_of_the_built_model` runs it in about 3 minutes); `report` prints every table below from the fixture alone.

## The mapping from `aod_550` to Prague's visibility

Koschmieder at 550 nm, V = 3.912 / beta, with beta v1's extinction at the ground, read from `lastlight.render.atmosphere` and not hard-coded: the Rayleigh vertical optical depth at v1's 970 hPa (0.0927566) over its 8 km scale height, 0.0115946 per km, plus `aod_550` over its 1.5 km aerosol scale height.

| `aod_550` | aerosol per km | beta per km | visibility | dataset nodes around it |
| --- | --- | --- | --- | --- |
| 0.03 | 0.0200000 | 0.0315946 | 123.8187 km | 90 and 131.8 |
| 0.1 | 0.0666667 | 0.0782612 | 49.9864 km | 40 and 59.4 |

Both are inside Prague's valid range, 20 to 131.8 km. The model's documents call its parameter "horizontal viewing distance at ground level" and do not say whether Rayleigh extinction is part of it; counting it is the only reading under which `aod_550` 0.03 is in range at all (aerosol alone would be 195.6 km).

## Results

All numbers are from `uv run python tools/prague_reference.py report`, which reads the committed fixture. "Lowest band" is WHTM1's first band (1.5066 degrees), "reference" its reference elevation (3.9944 degrees), from `build_views()["nominal"]`. XYZ is on v1's scale (a white card under the extraterrestrial sun has Y = 1). The multiply scattered share is `multiple_scattering_xyz` over v1's total, in Y.

### Sun +10, `aod_550` 0.03, lowest band against the reference band (the control)

| rel. azimuth | Prague log(R/G), log(B/G) | v1 minus Prague | v0 minus Prague | v1 u'v' distance | v0 u'v' distance |
| --- | --- | --- | --- | --- | --- |
| 0 | +0.134, -0.174 | +0.010, -0.049 | +0.008, -0.051 | 0.0049 | 0.0103 |
| 90 | +0.182, -0.149 | +0.019, -0.038 | +0.046, -0.121 | 0.0026 | 0.0240 |
| 180 | +0.172, -0.122 | +0.026, -0.056 | +0.056, -0.135 | 0.0052 | 0.0218 |

### Sun +10, `aod_550` 0.1, lowest band against the reference band (a diagnostic, no tolerance)

| rel. azimuth | Prague log(R/G), log(B/G) | v1 minus Prague | v0 minus Prague | v1 u'v' distance | v0 u'v' distance |
| --- | --- | --- | --- | --- | --- |
| 0 | +0.107, -0.118 | +0.010, -0.029 | +0.015, -0.048 | 0.0088 | 0.0136 |
| 90 | +0.114, -0.061 | +0.021, -0.014 | +0.075, -0.116 | 0.0027 | 0.0306 |
| 180 | +0.087, -0.005 | +0.026, -0.018 | +0.096, -0.114 | 0.0049 | 0.0310 |

Tolerances of the control: 0.10 per log ratio, 0.02 in u'v'.

### `aod_550` 0.03 (visibility 123.82 km): totals against Prague, and v1's multiply scattered share

| rel. azimuth | view elevation | Prague Y | v1 / Prague | v0 / Prague | MS share of v1 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.51 | 0.81624 | 1.190 | 1.077 | 0.095 |
| 0 | 1.96 | 0.79816 | 1.158 | 1.049 | 0.094 |
| 0 | 2.41 | 0.77669 | 1.132 | 1.026 | 0.094 |
| 0 | 2.86 | 0.75416 | 1.110 | 1.007 | 0.093 |
| 0 | 3.32 | 0.73192 | 1.091 | 0.990 | 0.093 |
| 0 | 3.99 | 0.70048 | 1.064 | 0.967 | 0.092 |
| 0 | 30.00 | 0.10272 | 0.958 | 0.829 | 0.134 |
| 0 | 60.00 | 0.02794 | 1.005 | 0.784 | 0.219 |
| 90 | 1.51 | 0.10201 | 1.091 | 0.624 | 0.428 |
| 90 | 1.96 | 0.10246 | 1.089 | 0.636 | 0.416 |
| 90 | 2.41 | 0.10215 | 1.086 | 0.644 | 0.407 |
| 90 | 2.86 | 0.10117 | 1.081 | 0.650 | 0.399 |
| 90 | 3.32 | 0.09971 | 1.077 | 0.655 | 0.392 |
| 90 | 3.99 | 0.09692 | 1.071 | 0.660 | 0.384 |
| 90 | 30.00 | 0.03149 | 1.030 | 0.701 | 0.319 |
| 90 | 60.00 | 0.01883 | 1.046 | 0.742 | 0.290 |
| 180 | 1.51 | 0.12819 | 1.183 | 0.773 | 0.347 |
| 180 | 1.96 | 0.13166 | 1.177 | 0.788 | 0.331 |
| 180 | 2.41 | 0.13383 | 1.168 | 0.796 | 0.319 |
| 180 | 2.86 | 0.13471 | 1.159 | 0.801 | 0.309 |
| 180 | 3.32 | 0.13456 | 1.150 | 0.804 | 0.301 |
| 180 | 3.99 | 0.13290 | 1.138 | 0.806 | 0.291 |
| 180 | 30.00 | 0.04020 | 1.076 | 0.799 | 0.258 |
| 180 | 60.00 | 0.01868 | 1.085 | 0.773 | 0.287 |

### `aod_550` 0.1 (visibility 49.99 km): totals against Prague, and v1's multiply scattered share

| rel. azimuth | view elevation | Prague Y | v1 / Prague | v0 / Prague | MS share of v1 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.51 | 0.79304 | 1.544 | 1.294 | 0.162 |
| 0 | 1.96 | 0.81453 | 1.529 | 1.289 | 0.157 |
| 0 | 2.41 | 0.82902 | 1.513 | 1.283 | 0.152 |
| 0 | 2.86 | 0.83658 | 1.498 | 1.276 | 0.148 |
| 0 | 3.32 | 0.83964 | 1.483 | 1.269 | 0.144 |
| 0 | 3.99 | 0.83830 | 1.461 | 1.256 | 0.140 |
| 0 | 30.00 | 0.15818 | 1.179 | 0.977 | 0.171 |
| 0 | 60.00 | 0.03678 | 1.051 | 0.749 | 0.288 |
| 90 | 1.51 | 0.07814 | 1.116 | 0.413 | 0.630 |
| 90 | 1.96 | 0.07998 | 1.121 | 0.431 | 0.616 |
| 90 | 2.41 | 0.08153 | 1.123 | 0.448 | 0.601 |
| 90 | 2.86 | 0.08267 | 1.124 | 0.463 | 0.589 |
| 90 | 3.32 | 0.08343 | 1.122 | 0.475 | 0.576 |
| 90 | 3.99 | 0.08391 | 1.120 | 0.491 | 0.562 |
| 90 | 30.00 | 0.03720 | 1.048 | 0.600 | 0.428 |
| 90 | 60.00 | 0.02282 | 1.050 | 0.654 | 0.377 |
| 180 | 1.51 | 0.07239 | 1.205 | 0.455 | 0.622 |
| 180 | 1.96 | 0.07572 | 1.217 | 0.491 | 0.597 |
| 180 | 2.41 | 0.07905 | 1.222 | 0.523 | 0.572 |
| 180 | 2.86 | 0.08213 | 1.222 | 0.549 | 0.550 |
| 180 | 3.32 | 0.08482 | 1.217 | 0.571 | 0.531 |
| 180 | 3.99 | 0.08797 | 1.209 | 0.595 | 0.507 |
| 180 | 30.00 | 0.04186 | 1.094 | 0.679 | 0.379 |
| 180 | 60.00 | 0.02126 | 1.083 | 0.665 | 0.386 |

### The other readings of Prague (`aod_550` 0.03, v1 minus Prague, lowest band)

Each cell: log(R/G) and log(B/G) deviation, then the u'v' distance at the reference elevation.

| reading | az 0 | az 90 | az 180 |
| --- | --- | --- | --- |
| xyz | +0.010, -0.049, 0.0049 | +0.019, -0.038, 0.0026 | +0.026, -0.056, 0.0052 |
| xyz_binned | +0.013, -0.048, 0.0081 | +0.022, -0.032, 0.0055 | +0.028, -0.049, 0.0095 |
| xyz_inverse_visibility | +0.010, -0.048, 0.0051 | +0.019, -0.037, 0.0025 | +0.025, -0.054, 0.0053 |
| xyz_altitude_0 | +0.008, -0.045, 0.0021 | +0.024, -0.045, 0.0045 | +0.028, -0.066, 0.0032 |
| xyz_matched_aerosol_depth | +0.008, -0.045, 0.0057 | +0.016, -0.032, 0.0024 | +0.022, -0.048, 0.0055 |

Every other reading of Prague leaves v1 inside both tolerances: the worst contrast deviation is 0.066 (observer at altitude 0, azimuth 180) and the worst u'v' distance is 0.0095 (piecewise-constant channels, azimuth 180).

## What the control does not cover

The control is a contrast and a chromaticity, both independent of overall level. The totals tables show the level differs: at `aod_550` 0.03 v1's luminance is 0.96 to 1.19 times Prague's (mean 1.10; v0's is 0.62 to 1.08, mean 0.81), and at 0.1 it is 1.05 to 1.54 times (mean 1.23; v0's is 0.41 to 1.29, mean 0.75). The largest excess is toward the sun at the lowest elevations (azimuth 0, 1.5 degrees: 1.19 and 1.54), where v0 is already high (1.08 and 1.29) and the multiply scattered share is only 9 and 16 percent, so it comes from single scattering through the aerosol's forward lobe, not from the Bruneton port. A likely cause, not tested (Prague's asymmetry parameters were not extracted): v1 fixes a Henyey-Greenstein asymmetry of 0.7 for the whole aerosol, while Prague fits a wavelength-dependent one per particle type to OPAC's phase functions (and uses deliberately blurrier ones for the coarse particles), so the glow 6 to 8.5 degrees from the sun differs. Away from the sun (azimuths 90 and 180) v1 sits 3 to 22 percent above Prague and v0 20 to 59 percent below, so the Bruneton term moves v1 most of the way. #35's camera magnitude check, not this control, is where level enters.

## The `aod_550` 0.1 question

The port's orders 2 and up sit 6.6 percent above the tool's Monte Carlo at `aod_550` 0.1 (sun 0, view 1.51, azimuth 30) and agree at 0.03. If Prague sides with the Monte Carlo, v1 at 0.1 must be high against Prague by about 6.6 percent of its multiply scattered share, relative to how it stands at 0.03 (the "expected if Monte Carlo" column: mean +0.026, from +0.009 to +0.041). If it sides with the port, v1 matches Prague as well at 0.1 as at 0.03 (0).

**Rule, written into the tool before the numbers were seen:** the two are told apart only if the predicted gap exceeds both the spread across the 24 views and azimuths and the shift from reading Prague's visibility in 1 / V instead of V. The predicted gap is 0.026, the spread (sd) 0.138 and the shift 0.014. The spread is five times the gap, so the answer is **cannot tell**.

### Which model does Prague side with at `aod_550` 0.1?

d = (v1 / Prague in luminance at 0.1) divided by (v1 / Prague at 0.03), minus 1, at each of the 24 views and azimuths. If Prague sides with the port, d is 0; if with the Monte Carlo, d is the expected column (6.6 percent of v1's multiply scattered share).

| reading of Prague | points | mean d | sd of d | min to max | expected if Monte Carlo |
| --- | --- | --- | --- | --- | --- |
| Koschmieder mapping (the rule's reading) | all 24 | +0.1181 | 0.1380 | -0.0021 to +0.3725 | +0.0263 |
| Koschmieder mapping (the rule's reading) | azimuth 90, 180 (16) | +0.0326 | 0.0188 | -0.0021 to +0.0621 | +0.0341 |
| visibility read in 1 / V | all 24 | +0.1062 | 0.1321 | +0.0013 to +0.3534 | +0.0263 |
| visibility read in 1 / V | azimuth 90, 180 (16) | +0.0228 | 0.0128 | +0.0013 to +0.0439 | +0.0341 |
| aerosol depth matched | all 24 | +0.0836 | 0.1179 | -0.0188 to +0.3078 | +0.0263 |
| aerosol depth matched | azimuth 90, 180 (16) | +0.0076 | 0.0114 | -0.0188 to +0.0269 | +0.0341 |

Rule (written before the numbers were seen): the two are told apart only if the predicted gap (+0.0263) exceeds both the spread of d across the views and azimuths (0.1380) and the shift from reading Prague's visibility in 1 / V (0.0141); result: **cannot tell**. The other rows and the azimuth 90, 180 subset are exploratory context, chosen after the first look.

What the numbers say beyond the rule (exploratory, chosen after the first look, so not evidence for either side):

- The mean of d over all 24 points is +0.118, far above both predictions, because v1's aureole excess grows with the aerosol (it is the single-scattering mismatch above, 1.19 at 0.03 and 1.54 at 0.1 at the lowest sunward view), not because of the multiple scattering. Azimuth 0 alone has a mean d of 0.29; azimuths 90 and 180 have 0.03 and 0.04.
- Away from the aureole (azimuths 90 and 180, 16 points) d is small, and which hypothesis it favours depends on how Prague's aerosol is set: +0.033 with the Koschmieder mapping (the Monte Carlo's +0.034), +0.023 with visibility read in 1 / V, +0.008 with Prague's own aerosol depth matched to v1's (the port's 0). The three readings of the same model differ by 0.025, three quarters of the 0.034 gap between the hypotheses, so the comparison cannot separate them.
- The mapping is itself part of the problem. Prague's aerosol profile is not exponential, so a horizontal visibility does not fix a vertical depth (its paper says so); the table above shows Prague at the Koschmieder visibilities has aerosol optical depths of 0.041 and 0.114 where v1 has 0.03 and 0.1 (each uncertain by about 0.01, as v1's Rayleigh and a 345 DU ozone column are subtracted), and at 0.03 no visibility in its range matches (131.8 km still has 0.037).
- Prague's own accuracy cannot resolve a 2.6 percent effect either: its paper gives no error figure for the fit (its supplement shows normalised errors only as box plots, and reports its brute-force simulator agreeing with libRadtran's MYSTIC in one validation atmosphere on average to a ratio of 1.0025 at sun 45 degrees and 1.0002 at 5 degrees, with more variance at the lower sun).

**What would tell.** (1) Run this comparison where the gap was measured: sun 0, view 1.51 degrees, azimuth 30. Prague's range reaches -4.2 degrees, so it covers that point, and there the multiply scattered share is 0.48 to 0.97 instead of 0.14 to 0.63 at sun +10, so a 6.6 percent excess is 3 to 6 percent of the total. A first look with the tool's functions (not committed, not in the fixture) at sun 0, observer 340 m, views 1.51 and 3.99 degrees: v1 is 1.2 to 1.45 times Prague in luminance at both atmospheres, and d is +0.11 and +0.15 at azimuth 30 with the Koschmieder mapping (the Monte Carlo predicts +0.05 and +0.03) and +0.05 and +0.08 with the aerosol depth matched, but +0.00 to -0.10 at azimuths 90 and 180 where the Monte Carlo predicts +0.04 to +0.06: it points both ways, which is the same answer. (2) A reference with v1's own aerosol: the same vertical profile and phase function, the one thing Prague cannot offer; libRadtran, run offline (GPL, never vendored or imported), is the natural one. (3) Equal aerosol load: Prague set to v1's vertical depth, which the dataset only allows for an `aod_550` of 0.037 and above.

## Decisions, each with its rejected alternative

- **The full dataset for every number, the ground-level one as a cross-check.** The issue names the ground-level dataset and an observer at 340 m. The ground-level dataset has one altitude, 0, and the model clamps an altitude outside the dataset's range without a word, so a 340 m query would have been evaluated at 0 m. The driver prints the altitude the model uses and the tool refuses any that is not the one asked for. Rejected: the ground-level file alone at "0.34 km" (the clamp makes the stated altitude false); an altitude of 0 for both (then the stated observer altitude is not what was compared). The ground-level file at altitude 0 gives, for every query of the fixture, exactly the full dataset's numbers (recorded in the fixture), and `xyz_altitude_0` shows the altitude changes v1's contrast deviation by at most 0.009.
- **A driver of ours against the unmodified source, not the repository's CLI.** The CLI renders fisheye images in sRGB through its own 5 nm colour table; the control needs spectral radiance at chosen angles and the repository's own colour-matching functions. Rejected: the CLI and its EXR output; the GPL `skylight` port.
- **Linear interpolation between channel centres, held flat beyond the ends.** Prague's 11 channels are 40 nm wide; its fit samples a smooth sky spectrum, and the authors' own example renderer treats the value as the radiance at the bin's centre. Rejected as primary: the model's own piecewise-constant lookup (a step at every bin edge, large where the spectrum is steep, as in the blue); it is the diagnostic `xyz_binned` and moves the worst u'v' from 0.0052 to 0.0095. The ends (below 340 and above 740 nm) weigh under 0.1 percent of Y.
- **The model's own interpolation in visibility (linear in V) as primary.** Extinction goes as 1 / V, so linear in V between 40 and 59.4 km is not linear in extinction. Rejected as primary: re-interpolating in 1 / V ourselves, so that the comparison is of the reference implementation as published; kept as `xyz_inverse_visibility` (it moves the aod 0.1 ratio by 0.014 on average).
- **The Rayleigh and aerosol terms of the mapping are v1's analytic ground values.** Rejected: the mean over v1's lowest 50 m shell (0.3 percent lower in Rayleigh, 1.6 percent in aerosol), because the spec words the aerosol term as `aod_550` over the scale height.
- **Matching Prague's aerosol depth is a diagnostic, never the primary.** The spec says the mapping is computed, not tuned. Rejected: replacing the Koschmieder visibility by the depth-matched one, even though it changes the aod 0.1 reading, because it relies on subtracting v1's Rayleigh and an assumed ozone column from Prague's transmittance.
- **A rule for the aod 0.1 question written before the numbers.** Rejected: reading the answer off the mean of d alone (+0.118, which would look like "the Monte Carlo is right" for the wrong reason: the aureole).
- **The camera-geometry guard.** The fixture's six nominal elevations must equal `build_views()["nominal"]` to 1e-9, so a refit of WHTM1 turns the control red until the fixture is regenerated. Rejected: reading the elevations from the camera at test time, which would compare v1 and Prague at elevations Prague was never evaluated at.
- **The regeneration check is a network-marked test.** It needs the built model and takes minutes. Rejected: committing the model's output only; a check that the fixture still comes from the tool is worth three minutes on demand.

## What broke, and what surprised

- **The ground-level dataset cannot serve the stated observer** (above). Found before any number was taken, from the model's header ("values outside range of the used dataset are clamped to the nearest supported value") and the dataset's own axes (one altitude).
- **The model lifts every viewpoint by 50 m** (`SAFETY_ALTITUDE`), so a viewpoint at 340 m is evaluated at 390 m. The driver now gives the model a viewpoint 50 m lower, and at altitude 0 that is 50 m underground, which the model's own assertion refuses; the driver is built with `-DNDEBUG`, and the tool checks the altitude the model reports.
- **The dataset's visibility axis has six nodes** (20, 27.6, 40, 59.4, 90 and 131.8 km), not a continuum: `aod_550` 0.1 is interpolated across 40 to 59.4 km in V, which is where the 1 / V diagnostic comes from.
- **Prague cannot be as clear as `aod_550` 0.03 under any reading**: at 131.8 km its own aerosol depth is still 0.037.
- **A first draft of a positive control was wrong about colour science**: it assumed scaling only Z leaves `log(R/G)` unchanged; the XYZ to sRGB matrix mixes Z into R, so the test now asserts only that the scaled band's deviation exceeds the tolerance. The test was fixed before it ran against a fixture.
- **An ad hoc `sed -i` on this Mac** takes its first argument as the backup suffix and did nothing; the rename was redone in Python. No effect on the repository.
- The host's load average was near 200 throughout; the model builds in 7 seconds and a fixture evaluation in about 3.5 minutes, nothing timed out.

## Known limits

- Prague's atmosphere is not v1's: OPAC aerosol profiles with an inversion layer instead of a 1.5 km exponential; wavelength-dependent Henyey-Greenstein phase functions with lower asymmetry instead of 0.7; the U.S. Standard ozone profile (about 345 DU) instead of 280 DU; a 6378 km planet instead of 6371; a ground at sea-level pressure under a 340 m observer instead of v1's observer on the ground at 970 hPa; 11 spectral channels instead of 41 samples. The control shows these differences do not move the daytime colour of the lowest band by more than 0.056 in log contrast, not that either model is right.
- One sun elevation (+10), one albedo (0.2), one ozone (280 DU), two aerosol loads. A twilight sun is where lastlight forecasts, and Prague is valid there (to -4.2 degrees); this PR does not compare there.
- The datasets carry no licence statement of their own that I could find; nothing of them is committed.

## Next

The conductor decides what the pass means for v1's adoption (#35's rule is a separate gate). For the aod 0.1 gap, the natural follow-up is the sun 0 comparison at the gap's own geometry, and a libRadtran run with v1's aerosol.
