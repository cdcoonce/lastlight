# Sixth observed evening (2026-10-15): the second clear evening in a row

The fourth evening in a row that ran with no hands on it. Frames, the archive and observe's output stay private. This entry records only derived numbers, against sun elevation. The seal is in the fifth evening's entry (`docs/log/2026-10-14-fifth-evening.md`): a fresh `ledger` clone verifies 2026-10-15 `PASS (strip verified)`.

**The collection.** The timer's run saved 6 new frames from WHTM1 and 7 from ESMO1. 12 and 11 more polls found the newest frame unchanged; there were no stale frames and no errors. WHTM1's feed skipped one 15-minute slot inside the window, the one that would have fallen at about sun +1.

**What the sky did:** clear on the ridge again. Before sunset a pale blue sky over a hazy yellow band; after it a yellow-to-orange band along the whole ridge, grading up through peach to blue. A few thin streaks of high cloud crossed the upper left of the frame late in the window, lit white and pink, well above the bands `observe` reads; the last frame they show in falls below the window and was not used.

**What `lastlight observe` read** (run on the capture host; WHTM1, frozen provisional fit, default window +6 to -6, the registry's bands). 3 of 6 frames were used, with 2 dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline, the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +3.97 | +0.248 | -0.620 | +0.050 | -0.135 | no |
| -2.06 | +0.450 | -0.921 | +0.065 | -0.162 | no |
| -5.17 | +0.299 | -0.969 | +0.045 | -0.200 | yes |

The lowest band lost 3,787, 4,929 and 8,790 of its 16,000 pixels to saturation, frame by frame. The last frame also lost 5,400 to the light-dome sector, as designed below sun -4, so its lowest band rests on 1,810 pixels.

**Reading it:**

- The gradient is monotone up the five bands at every frame, as on the fifth evening: log(B/G) -0.62, -0.47, -0.34, -0.24, -0.14 at sun +4; -0.92, -0.67, -0.46, -0.31, -0.16 at sun -2; -0.97, -0.74, -0.54, -0.36, -0.20 at sun -5.
- The two clear evenings agree closely where they overlap in sun: the lowest band's log(B/G) was -0.61 at sun +4.3 and -0.96 at sun -5.0 on the fifth, and -0.62 at sun +4.0 and -0.97 at sun -5.2 here. The top band agrees within 0.05 (0.041 and 0.024). This is what G1 (#31) is built to measure over 10 or more pairs. One pair is not a measurement, and nothing is computed for G1 here.
- The frame at sun -2 shows that most of the change from before sunset to deep twilight happens by sun -2 in the lowest band (-0.62 to -0.92), and little after (-0.97). The top band keeps reddening slowly throughout.
- **The v1 check.** With 3 used frames this evening is considered. Whether it counts is decided by HRRR's analysis of total cloud along the G0 transect, never by these frames. HRRR read a mean total cloud of 0.0 percent along the sightline, and `aod_550` 0.0168, the cleanest air of the six evenings, so **this is the second counted evening**: number 2 of the 5 the verdict needs. The report reads `NOT YET DECIDABLE (2 of 5 counted evenings)` (`docs/gates/renderer-v1/report.md`).
- **The second counted evening's ratio.** v1's r_e is 0.432, under the rule's window of 0.5 to 2.0; v0's is 1.61, inside it. This time the signs agree at every frame: v1 paints the lowest band's log(B/G) at -0.37, -0.40 and -0.64 where the camera read -0.62, -0.92 and -0.97, so v1's contrast is 0.60, 0.43 and 0.66 of the camera's, frame by frame (the ratio of the medians is 0.43). v0 paints -0.53, -1.92 and -1.49. On this clear evening with clean air, v1's contrast is weaker than the camera's at every frame, and v0's, without the multiple scattering v1 adds, is stronger at two of three. The first counted evening went the other way for v1 (1.22, with a cloud base in the band). At the pixel-scale ends of the fit, v1's ratio is 0.670 and 0.425.
- The verdict is the median of r_e over the first 5 counted evenings, taken once. Two are in: 1.22 and 0.43. Nothing is computed from two, and nothing in the check changes because the second fell outside the window.

**Where the outputs live now:** observe ran on the capture host, against its own archive, writing to a private directory there. Its output was copied to the build machine's private observed directory, which now holds 2026-10-10 to 2026-10-15. The frames were looked at once on the build machine to describe the sky, then deleted there.
