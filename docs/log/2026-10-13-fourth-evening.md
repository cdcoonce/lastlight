# Fourth observed evening (2026-10-13): the first painted with the adaptive quadrature

The second evening in a row that ran with no hands on it: the scheduled workflow sealed it, the capture host's timer collected it, and `lastlight observe` read it there. Frames, the archive and observe's output stay private. This entry records only derived numbers, against sun elevation.

**The seal.** The scheduled run sealed 2026-10-13 on its first attempt, with no retry. The forecast records `code_sha` `6c4b5b9`, so it is the first painting made with the adaptive quadrature from #40 (PR #42). That quadrature is identical to the old one up to `aod_550` 0.06, so the change shows only on hazy evenings. A fresh clone of the `ledger` branch verifies all three public evenings with `lastlight verify <date>`: 2026-10-11, 2026-10-12 and 2026-10-13 each print `PASS (strip verified)`. The pages workflow redeployed after the seal, and the almanac shows the new painting.

**The collection.** The timer's run saved 6 new frames from WHTM1 and 7 from ESMO1. 12 and 11 more polls found the newest frame unchanged, and there were no stale frames and no errors. WHTM1's feed skipped one of its 15-minute slots inside the window, which is why it has one frame fewer than ESMO1.

**What the sky did:** clear above, with a low band of broken cloud lying along the ridge. The thin strip of sky between the ridge and the band glowed yellow, then orange. At sun +4.5 the band's underside was lit gold. After the sun set the band went lavender-grey over an orange horizon strip, while the clear sky above graded from peach to pale blue. This was the clearest of the four evenings above the band, but the band sat where the camera's lowest band looks.

**What `lastlight observe` read** (run on the capture host; WHTM1, frozen provisional fit, default window +6 to -6, the registry's bands). 3 of 6 frames were used, with 2 dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline, the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +4.45 | +0.372 | -0.519 | +0.049 | -0.128 | no |
| -1.59 | +0.318 | +0.203 | +0.063 | -0.129 | no |
| -4.69 | +0.038 | +0.157 | +0.028 | -0.152 | yes |

The lowest band lost 783, 0 and 752 of its 16,000 pixels to saturation, frame by frame. The last frame also lost 5,400 to the light-dome sector, as designed below sun -4.

**Reading it:**

- Before sunset the lowest band read redder than the reference (log(B/G) -0.52), as on the third evening. It held the gold-lit underside of the band and the strip below it.
- After sunset the sign flipped: +0.20 and +0.16. The lowest band was then mostly the band's dark lavender-grey base, against a peach-to-blue clear reference. A grey cloud base in shadow is nearer neutral than the peach sky above it, so against that reference it reads bluer. This is cloud against sky again, not the clear-sky gradient the renderer paints.
- The top band stayed between -0.13 and -0.15 in log(B/G) at every frame. That is a mild reddening toward the band under clear air, and it is the steadiest number of the evening.
- Whether this evening counts toward renderer v1's magnitude check is decided by the check (#35) from HRRR's analysis of total cloud along the G0 transect, never from these frames. The check read a mean total cloud of 5.5 percent along the sightline (and `aod_550` 0.045), under the 10 percent bar, so **this is the first counted evening**: number 1 of the 5 the verdict needs (`docs/gates/renderer-v1/report.md`, still `NOT YET DECIDABLE (1 of 5 counted evenings)`). HRRR's analysis and the frames disagree about the band along the ridge: the transect saw almost no cloud where the camera saw a band sitting on the skyline. The criterion counts by HRRR, deliberately, so that neither the renderer nor the camera decides which evenings count, and nothing in it changes because of this.
- **The counted evening's ratio, and what it hides.** v1's per-evening ratio r_e is 1.22, inside the rule's 0.5 to 2.0; v0's is 9.95. But r_e compares magnitudes, |rendered| against |observed|, and at two of the three frames the signs disagree: v1 paints the lowest band redder than the reference (log(B/G) -0.09 and -0.25), while the camera read it bluer (+0.20 and +0.16) because the band held the cloud base. Only the frame before sunset agrees in sign (-0.31 painted, -0.52 observed). The criterion asks the coarse magnitude question and was fixed before any evening was seen; this is reported, not acted on. If more counted evenings carry a cloud band on the skyline that HRRR misses, the sign disagreement belongs in G1's and the vividness score's design, not in a change to this check.

**Where the outputs live now:** observe ran on the capture host, against its own archive, writing to a private directory there. Its output (derived numbers, no frames) was copied to the build machine's private observed directory, which now holds 2026-10-10 to 2026-10-13.
