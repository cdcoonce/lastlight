# Fifth observed evening (2026-10-14): the clearest yet, and not counted

The third evening in a row that ran with no hands on it. Frames, the archive and observe's output stay private. This entry records only derived numbers, against sun elevation.

**The seals.** The scheduled runs sealed 2026-10-14 and 2026-10-15, each on its first attempt. A fresh clone of the `ledger` branch verifies all five public evenings with `lastlight verify <date>`: 2026-10-11 to 2026-10-15 each print `PASS (strip verified)`, exit 0. The pages workflow redeployed after each seal.

**The collection.** The timer's run saved 5 new frames from WHTM1 and 7 from ESMO1. 13 and 11 more polls found the newest frame unchanged; there were no stale frames and no errors. Inside the window WHTM1's feed skipped two of its 15-minute slots in a row, the two that would have fallen at about sun +1.5 and -1.7, so the sun crossed the horizon between two frames 9 degrees apart.

**What the sky did:** cloudless, apart from a faint wisp high in the frame early on. Before sunset the sky above the ridge was a pale, even blue over a hazy yellow band. After sunset a clean band of yellow, then orange, lay along the ridge and graded up through peach to blue. There was no cloud on the skyline at any frame. This is the clear-sky gradient the renderer paints, and the first evening of the five with nothing else in the lowest band.

**What `lastlight observe` read** (run on the capture host; WHTM1, frozen provisional fit, default window +6 to -6, the registry's bands). 2 of 5 frames were used, with 2 dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline, the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +4.34 | +0.235 | -0.614 | +0.029 | -0.094 | no |
| -4.96 | +0.306 | -0.959 | +0.033 | -0.176 | yes |

The lowest band lost 1,722 and 8,215 of its 16,000 pixels to saturation, frame by frame: the bright band on the ridge clips. The second frame also lost 5,400 to the light-dome sector, as designed below sun -4, so its lowest band rests on 2,385 pixels.

**Reading it:**

- Both frames read the lowest band redder than the reference, and the gradient is monotone up the five bands at both: log(B/G) -0.61, -0.46, -0.33, -0.20, -0.09 before sunset and -0.96, -0.72, -0.50, -0.34, -0.18 after. No earlier evening was monotone at every frame; the third was at three of four.
- The sizes match the third evening's, whose frames looked through a clear slot under a deck: -0.73 and -1.00 there in the lowest band before and after sunset, against -0.61 and -0.96 here. Two evenings with clear air on the ridge now agree on the order of the lowest band's contrast, near -1 by sun -5.
- The top band moved from -0.09 to -0.18 as the sun went down: the reddening climbs the sky, as it should.
- **Not considered by the v1 check.** The check's definition 3 considers an evening only if `observe` used at least 3 frames, and the feed's two skipped slots left 2. `lastlight renderer-check` therefore lists 2026-10-14 as `not considered: 2 used frames`, and it adds nothing to the count (the next evening, 2026-10-15, became the second counted evening). The rule was fixed before any evening was seen and is not changed because the clearest evening fell under it. HRRR's cloud for the evening was not fetched, because the check fetches nothing for an evening it does not consider.
- G1 (#31) needs pairs of consecutive clear evenings. This is the first evening clear from the ridge up, so it is the first that can be half of a pair.

**Where the outputs live now:** observe ran on the capture host, against its own archive, writing to a private directory there. Its output (derived numbers, no frames) was copied to the build machine's private observed directory. The frames were looked at once on the build machine to describe the sky, then deleted there.
