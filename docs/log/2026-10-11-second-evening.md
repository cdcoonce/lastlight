# Second observed evening (2026-10-11)

There is still no nightly timer on the capture host, so the conductor ran the collector by hand from the main clone, as on the first evening. The previous session started it as the sun crossed +8 degrees. That session then ended, and the collector stopped with it. This session found it gone and restarted it within the same frame interval: the first poll after the restart re-fetched the newest frame, recognised it as unchanged, and no frame was missed. It then ran until the sun was 10 degrees down. Each camera ended with 6 frames, 2 from the first run and 4 from the second. The cadence was again exactly 15 minutes, and there were no errors or stale frames. Frames, the archive and observe's output stay private. This entry records only derived numbers, against sun elevation.

**What the sky did:** overcast again. A low grey deck sat over the ridge with lower ragged cloud along it, and a warm band showed only at the sunward (left) edge of the frame, close to the horizon. The rest of the western sky was grey through civil twilight. On the vividness scale this was another dull evening.

**What `lastlight observe` read** (WHTM1, frozen provisional fit, default window +6 to -6, the registry's bands). 4 of the 6 frames were used, 1 dropped above the window and 1 below it. The lowest band is rows 8 to 28 above the frozen skyline, the top band rows 88 to 108, each against the reference band of rows 118 to 138.

| Sun elevation | Lowest band log(R/G) | Lowest band log(B/G) | Top band log(R/G) | Top band log(B/G) | Light-dome columns excluded |
| --- | --- | --- | --- | --- | --- |
| +4.98 | -0.153 | +0.328 | -0.061 | +0.063 | no |
| +2.02 | +0.024 | +0.042 | +0.009 | -0.006 | no |
| -1.07 | -0.006 | +0.371 | +0.002 | +0.093 | no |
| -4.19 | -0.072 | +0.171 | -0.066 | +0.052 | yes |

Re-run on the first evening with today's code, observe reads 4 of 8 frames (3 dropped above the window, 1 below it). Its lowest-band values are those of `docs/log/2026-10-11-observed-contrasts.md`, between -0.21 and +0.14 in either channel.

**Reading it:**

- On this evening the lowest band is **bluer** than the reference: log(B/G) contrast is +0.04 to +0.37 at every used frame, the opposite sign to a clear twilight sky reddening toward the horizon. The renderer's clear-sky value at sun 0 is about -2.6 in v0 units. The frames explain the difference. The rows just above the ridge hold the grey base of a low cloud deck, and the reference rows hold cloud lit a little warmer from above the deck. The camera saw clouds, not the clear-sky gradient the renderer paints.
- So both observed evenings are overcast. Neither tests the clear-sky renderer, in either direction. The renderer v1 magnitude check pre-registered in `docs/gates/renderer-v1/criterion.md` (PR #33) counts an evening only when HRRR's total cloud cover along the G0 transect is at most 10 percent, so cloud decks like these two cannot decide it. G1 (issue #31) also waits for consecutive clear evenings.
- Cloud is the common case at this camera so far, and the design commitment holds: clear-sky curves and cloud-regime vividness are scored separately. An overcast evening is data for the vividness side and for the cloud regime, not a renderer failure.

**What it costs to keep doing this by hand:** an evening is collected only when a session happens to span it, and a session that ends mid-window takes its collector with it. The nightly timer on the capture host (runbook `docs/runbooks/collector.md`) is still the owner's install.
