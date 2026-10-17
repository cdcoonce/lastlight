# Seventh observed evening (2026-10-16): a hazy evening after two clear ones

This is the fifth evening in a row that ran with no hands on it. Frames, the archive and observe's output stay private, and this entry records only derived numbers, against sun elevation.

**The seal.** The scheduled seal for 2026-10-16 sealed on its first attempt and Pages redeployed after it. A fresh `ledger` clone verifies all six public evenings, 2026-10-11 to 2026-10-16, `PASS (strip verified)`, each with exit 0.

**The collection.** The timer's run saved 7 new frames from each of WHTM1 and ESMO1. Another 11 polls of each found the newest frame unchanged; there were no stale frames and no errors. WHTM1's frames came every 15 minutes with no slot skipped.

**What the sky did.** It was hazy. A milky beige layer lay over the ridge from the first frame used to the last. The ridge stayed visible but soft-edged, and no cloud edge showed anywhere in the frame. Before sunset the sun's glow spread wide and pale across the left of the frame instead of the clear evenings' blue sky over a yellow band. After sunset the haze turned a dull beige-yellow. By sun -5, a yellow band sat on top of the haze, grading up to pale blue. Whether the haze was dust, smoke or neither is not something the frames can say.

**What `lastlight observe` read.** It ran on the capture host for WHTM1, with the frozen provisional fit, the default window of +6 to -6 and the registry's bands. 4 of 7 frames were used; 2 were dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline and the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +3.82 | +0.267 | -0.534 | +0.051 | -0.120 | no |
| +0.76 | +0.308 | -0.362 | +0.060 | -0.125 | no |
| -2.41 | +0.237 | -0.292 | +0.054 | -0.121 | no |
| -5.42 | +0.209 | -0.664 | +0.029 | -0.142 | yes |

No pixel of the lowest band saturated in any frame. On the two clear evenings, 1,722 to 8,790 of its 16,000 pixels saturated in every frame. The last frame lost 5,400 pixels to the light-dome sector, as designed below sun -4, so its lowest band rests on 10,600 pixels.

**Reading it:**

- **The gradient is weak and, at two frames, not monotone.** log(B/G) up the five bands is -0.53, -0.43, -0.32, -0.22, -0.12 at sun +4. At sun +1 it is -0.36, -0.36, -0.30, -0.21, -0.13: the lowest two bands are level (-0.362 and -0.363). At sun -2 it is -0.29, -0.35, -0.31, -0.22, -0.12: the lowest band is bluer than the one above it. At sun -5 it is -0.66, -0.64, -0.44, -0.28, -0.14. On the fifth and sixth evenings it was monotone at every frame.
- **Against the clear evenings, the haze weakened the lowest band most.** Its log(B/G) was -0.61 at sun +4.3 (fifth evening) and -0.62 at sun +4.0 (sixth); here it is -0.53 at sun +3.8. After sunset it was -0.92 at sun -2.1 and -0.96 to -0.97 at sun -5.0 to -5.2; here it is -0.29 at sun -2.4 and -0.66 at sun -5.4. The top band moved much less: -0.094 and -0.135 at sun +4 on the clear evenings against -0.120 here, and -0.176 and -0.200 at sun -5 against -0.142 here.
- Most of the clear evenings' change in the lowest band happened by sun -2. Here the lowest band stayed pale through sun -2 and deepened only by sun -5, when the yellow band showed above the haze.
- **G1.** Whether this evening and the sixth make a pair for G1 (#31) is for G1's own clear-evening rule to say, once it runs on 10 or more pairs; nothing is computed for G1 here. Haze is not cloud, so a clear evening by cloud can still look like this one.
- **The v1 check.** With 4 used frames this evening is considered. Whether it counts is decided by HRRR's analysis of total cloud along the G0 transect, never by these frames. HRRR read a mean total cloud of 4.53 percent along the sightline, under the 10 percent line, and the check's `aod_550` was 0.0177. So **this is the third counted evening**: number 3 of the 5 the verdict needs. The report reads `NOT YET DECIDABLE (3 of 5 counted evenings)` (`docs/gates/renderer-v1/report.md`).
- **The third counted evening's ratio.** v1's r_e is 1.025, inside the rule's window of 0.5 to 2.0; v0's is 3.53, outside it. v1 paints the lowest band's log(B/G) at -0.38, -0.52, -0.40 and -0.65 where the camera read -0.53, -0.36, -0.29 and -0.66. So v1's contrast is 0.71, 1.44, 1.36 and 0.99 of the camera's, frame by frame, and the ratio of the medians is 1.02. v0 paints -0.55, -1.70, -1.87 and -1.47. At the pixel-scale ends of the fit, v1's ratio is 1.483 and 1.010.
- **What moved the ratio was the sky, not the painting.** v1's inputs were nearly those of the sixth evening (`aod_550` 0.0177 against 0.0168, HRRR cloud under 5 percent both times), and it painted nearly the same contrasts: -0.378 and -0.373 at sun +4, -0.654 and -0.642 at sun -5. The camera's contrast fell under the haze, so nearly the same painting read 0.43 of the camera's last night and 1.02 tonight. The check's aerosol input shows nothing like the haze the camera saw. The counting rule is about cloud, and haze is not cloud, so the evening counts as the rule says.
- The verdict is the median of r_e over the first 5 counted evenings, taken once. Three are in: 1.22, 0.43 and 1.02. Nothing is computed from three, and nothing in the check changes.

**Where the outputs live now.** observe ran on the capture host, against its own archive, writing to a private directory there. Its output was copied to the build machine's private observed directory, which now holds 2026-10-10 to 2026-10-16. The frames were looked at once on the build machine to describe the sky, then deleted there.
