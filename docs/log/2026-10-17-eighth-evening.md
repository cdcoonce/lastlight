# Eighth observed evening (2026-10-17): broken mid and high cloud over a clear band at the ridge

This is the sixth evening in a row that ran with no hands on it. Frames, the archive and observe's output stay private, and this entry records only derived numbers, against sun elevation.

**The seal.** The scheduled seal for 2026-10-17 sealed on its first attempt and Pages redeployed after it. A fresh `ledger` clone verifies all seven public evenings, 2026-10-11 to 2026-10-17, `PASS (strip verified)`, each with exit 0.

**The collection.** The timer's run saved 7 new frames from WHTM1 and 6 from ESMO1. Another 11 polls of WHTM1 and 12 of ESMO1 found the newest frame unchanged; there were no stale frames and no errors.

**What the sky did.** A broken layer of mid and high cloud lay across the upper frame from the first frame used to the last, in streaks and patches with blue between them. Below it, a clear band sat over the ridge. Before sunset the cloud was lit white and grey, and the band above the ridge was pale yellow. After sunset the band turned a deep yellow, the cloud above it went grey and purple with a lit underside, and by the last frame used a dark deck covered most of the frame above a bright strip at the horizon.

**What `lastlight observe` read.** It ran on the capture host for WHTM1, with the frozen provisional fit, the default window of +6 to -6 and the registry's bands. 4 of 7 frames were used; 2 were dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline and the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +3.56 | +0.256 | -0.505 | +0.023 | +0.010 | no |
| +0.52 | +0.243 | -0.510 | -0.042 | -0.100 | no |
| -2.62 | +0.309 | -0.849 | +0.035 | +0.035 | no |
| -5.66 | +0.414 | -0.771 | +0.068 | -0.108 | yes |

626 of the lowest band's 16,000 pixels saturated in the first frame used. On the two clear evenings, 1,722 to 8,790 saturated in every frame; on the hazy evening, none did.

**Reading it:**

- **The gradient is monotone at three frames.** log(B/G) up the five bands is -0.51, -0.39, -0.24, -0.10, +0.01 at sun +3.6; -0.51, -0.47, -0.40, -0.30, -0.10 at sun +0.5; and -0.85, -0.66, -0.34, -0.06, +0.04 at sun -2.6. At sun -5.7 it is -0.77, -0.79, -0.62, -0.31, -0.11: the lowest two bands are nearly level, and the lowest is the less blue.
- **The lowest band behaved like a clear evening's after sunset.** Its log(B/G) was -0.85 at sun -2.6 and -0.77 at sun -5.7, against -0.92 at sun -2.1 and -0.96 to -0.97 at sun -5.0 to -5.2 on the clear evenings, and -0.29 and -0.66 under the haze. Before sunset it was paler, at -0.51 against -0.61 to -0.62 on the clear evenings.
- **The top band saw the cloud.** Its log(B/G) was +0.01 at sun +3.6 and +0.04 at sun -2.6, near neutral, against -0.09 to -0.14 at sun +4 and -0.18 to -0.20 at sun -5 on the clear evenings. Grey cloud in the top rows takes the blue out of the band's contrast with the reference.
- **G1.** Whether this evening pairs with any other for G1 (#31) is for G1's own clear-evening rule to say, once it runs on 10 or more pairs; nothing is computed for G1 here.
- **The v1 check.** With 4 used frames this evening is considered. Whether it counts is decided by HRRR's analysis of total cloud along the G0 transect, never by these frames. HRRR read a mean total cloud of 39.71 percent along the sightline, over the 10 percent line, so **this evening does not count**. The report still reads `NOT YET DECIDABLE (3 of 5 counted evenings)` (`docs/gates/renderer-v1/report.md`), with 1.22, 0.43 and 1.02 in. The check's `aod_550` was 0.0313.
- **Its ratios, reported and not counted.** v1's r_e is 0.614 and v0's is 2.95; at the pixel-scale ends of the fit, v1's ratio is 0.895 and 0.605. They are in the report beside the verdict, as for the other evenings not counted, and do not enter it. The check scores the lowest band, below most of the cloud, but the ratio is still not a clear-sky reading, which is why the counting rule leaves the evening out.
- The verdict is the median of r_e over the first 5 counted evenings, taken once. Nothing in the check changes.

**Where the outputs live now.** observe ran on the capture host, against its own archive, writing to a private directory there. Its output was copied to the build machine's private observed directory, which now holds 2026-10-10 to 2026-10-17. The frames were looked at once on the build machine to describe the sky, then deleted there.
