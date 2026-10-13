# Third observed evening (2026-10-12): the first one the timer collected

This is the first evening collected by the nightly timer on the capture host (installed by the owner on 2026-10-12, runbook `docs/runbooks/collector.md`), not by a session. It is also the first evening sealed by the scheduled workflow, not by hand. Frames, the archive and observe's output stay private. This entry records only derived numbers, against sun elevation.

**The seal.** The scheduled run sealed 2026-10-12 on its first attempt; the retry added in #26 was not needed. A fresh clone of the `ledger` branch verifies both public evenings with the signature chain check from #25: `lastlight verify 2026-10-12` and `lastlight verify 2026-10-11` each print `PASS (strip verified)`. The pages workflow redeployed after the seal, and the almanac shows the new painting.

**The collection.** The timer's run saved 7 new frames per camera (WHTM1 and ESMO1). 11 more polls per camera found the newest frame unchanged, and there were no stale frames and no errors. The cadence was again exactly 15 minutes. Frames now live only on the capture host.

**What the sky did:** a grey deck aloft with a clear slot between its base and the ridge. A bright yellow band ran along the horizon in every used frame. Just after the sun set, the underside of the deck lit pink. By sun -4.5 it was grey again over a narrow glowing band. For the vividness scale this evening sits between the two dull overcast evenings and a clear one: the colour was real, but confined to a slot under cloud.

**What `lastlight observe` read** (run on the capture host; WHTM1, frozen provisional fit, default window +6 to -6, the registry's bands). 4 of 7 frames were used, with 2 dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline, the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +4.82 | +0.259 | -0.730 | +0.052 | -0.077 | no |
| +1.76 | +0.194 | -0.697 | +0.144 | -0.158 | no |
| -1.37 | -0.195 | -0.699 | -0.057 | -0.087 | no |
| -4.48 | +0.249 | -0.998 | +0.054 | -0.090 | yes |

The lowest band lost 3,440, 2,442, 4,024 and 1,614 of its 16,000 pixels to saturation, frame by frame. The last frame also lost the light-dome sector, as designed below sun -4.

**Reading it:**

- For the first time, the lowest band is **redder** than the reference. Its log(B/G) contrast runs from -0.70 to -1.00 at every used frame, the sign of a twilight sky reddening toward the horizon. On the two overcast evenings it was +0.04 to +0.37.
- The frames say why, and it is not the clear-sky gradient. The lowest band holds the bright slot of clear sky under the deck. The reference band holds the deck's base. A contrast between clear air and cloud is not what the renderer paints, which is clear sky at both elevations. The forecast sealed for this evening recorded 41 percent low cloud along the sightline (in the ledger's forecast notes).
- Saturation is not neutral here. Observe drops saturated pixels, which are the brightest part of the slot. What remains is the slot's dimmer edge, so a band this bright reads less red than it is. Up to a quarter of the band was dropped. Tier A is a contrast measure, not an absolute one, and evenings like this are where that shows.
- Whether this evening counts toward renderer v1's magnitude check is decided by the check (#35) from HRRR's analysis of total cloud along the G0 transect, never from these frames. These observed numbers were read before that check existed. The criterion (`docs/gates/renderer-v1/criterion.md`) was fixed before either, and nothing in it changes because of them. Once built, the check read a mean total cloud of 49.5 percent along the sightline, so the evening is not counted (`docs/gates/renderer-v1/report.md`).

**Where the outputs live now:** observe ran on the capture host, against its own archive, writing to a private directory there. Its output (derived numbers, no frames) was copied to the build machine's private observed directory, so the magnitude check can read all three evenings in one place.
