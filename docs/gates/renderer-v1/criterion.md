# Renderer v1 adoption: the camera magnitude check (pre-registered)

Registered in the commit that adds this file, on 2026-10-11. At that commit renderer v1 does not exist, no v1 contrast has been computed, and no comparison of any renderer with an observed evening has been run beyond the one in `docs/log/2026-10-11-g0-gate.md` (v0 against the overcast evening of 2026-10-10, which this check counts only if that evening qualifies under the rules below). Issue #32. Nothing in this file may be changed after a result has been seen. A correction is a dated amendment at the bottom, made in a later commit and repeated in the report.

## Why this check exists

G0 passed in renderer-v0 units, and its margin rests on log(B/G). At a sun elevation of 0, the single-scattering renderer puts WHTM1's lowest band at a log(B/G) contrast of about -2.6 against the reference band. The first observed evening read between -0.21 and +0.14. Renderer v1 adds the multiply scattered light that v0 leaves out. This check asks one coarse question before v1 paints any sealed forecast: is v1's within-frame contrast within a factor of two of what the camera sees? A pass means the magnitude is plausible. It says nothing about skill.

## The rule

> **Renderer v1 is adopted for the daily forecast if and only if, over the first 5 counted evenings, the median of the per-evening ratio r_e lies between 0.5 and 2.0 inclusive.** With fewer than 5 counted evenings the verdict is "not yet decidable" and v0 keeps painting. If the rule fails, v0 keeps painting and the report says so.

For a counted evening e, the per-evening ratio is

r_e = median over the evening's used frames of |rendered contrast| / median over the same frames of |observed contrast|,

where the contrast is the lowest band's log(B/G) minus the reference band's. If the denominator is 0, r_e is +infinity, and it still takes part in the median.

## Definitions fixed before any result

1. **Camera and bands.** WHTM1 only. The lowest band is `bands[0]` of the camera's `[camera.contrast]` rows in `registry.toml` (rows 8 to 28 above the frozen skyline), and the reference band is `reference_rows` (rows 118 to 138). This is the definition `lastlight observe` uses, unchanged.
2. **Observed contrast.** The `log_bg` of `bands[0]` in the evening's `observed.json` from `lastlight observe` (schema version 1), run with its defaults: the sun window +6 to -6 degrees, the frozen fit, and the registry's contrast definition. A frame is used when `bands[0].log_bg` is not null. Frames, archive and observe output stay private; the report publishes only dates, sun elevations and derived numbers.
3. **Evenings considered.** Every evening from 2026-10-10 on whose `observed.json` has at least 3 used frames.
4. **Counted evenings.** An evening considered is counted when it is clear by data that neither the renderer nor the camera supplies: the mean, over the points of the sightline transect that the G0 study uses (from the public reference point along the solar azimuth), of HRRR's total cloud cover (`TCDC`, entire atmosphere) at forecast hour 0, for the whole UTC hour nearest the instant the sun reaches the horizon at the reference point, is at most 10 percent. Evenings considered and not counted are reported with their cloud cover, and they do not enter the verdict.
5. **The first 5.** The verdict is computed once, on the 5 earliest counted evenings in date order. Later evenings are reported beside it and never change it. This prevents running the check again and again until it passes.
6. **Rendered contrast.** For each used frame, `lastlight.render` with the v1 engine, through the same seam G0 uses (`sky_xyz`, then G0's `log_chromaticity` with its channel floor). It is evaluated at the frame's recorded `sun_elevation_deg`, without snapping to the sweep. The view elevations are WHTM1's band and reference elevations at the frozen fit's nominal pixel scale (G0's `build_views()["nominal"]`). The relative azimuth is the fit's centre azimuth minus the sun's azimuth when the sun stands at that elevation on that date at the reference point.
7. **The evening's atmosphere** is built exactly as G0 builds it from archived analyses: `aod_550` is GEFS-Aerosols' total aerosol optical depth at 550 nm at forecast hour 0 of the latest cycle at or before the sun reaching the horizon, `ozone_du` is the monthly climatology, and every other field keeps its default. An evening whose inputs are missing, or fall outside v1's table, is dropped with its reason and is not counted.
8. **Which v1.** The v1 under test is the engine and lookup table that land with issue #32. The SHA-256 of that table is recorded here as a dated amendment when it lands, before the check is built or run. A later change to v1's physics or table is a different renderer. It needs a new registration and starts again from its own first 5 counted evenings.
9. **Inclusive bounds.** A median of exactly 0.5 or exactly 2.0 passes.

## Reported, not part of the verdict

- The same ratio for v0, so the size of v1's change is visible.
- The same ratio for log(R/G).
- The ratio at both ends of the fit's `uncertainty.px_per_deg` range (the band elevations move, and the rendered contrast moves with them).
- Every evening considered, counted or not, with its cloud cover, its frame count, and its r_e for v0 and v1.
- The per-frame pairs (sun elevation, rendered, observed) for every evening considered.

## What this check cannot say

The webcam's tone curve is assumed to be sRGB when frames are linearised. A camera with a different curve compresses or stretches every log ratio, and the factor-of-two window partly absorbs that. Five evenings in one season at one camera do not establish skill. They only tell whether v1's contrasts are of the camera's order. G1 (issue #31) measures the camera's repeatability, and scoring (M3) measures skill.

## Amendments

### 2026-10-12: which v1 is under test (definition 8)

Recorded when v1 landed (PR #39, merge `3373137`), before the check exists (#35) and before any v1 contrast has been compared with an observed evening. The renderer under test is engine `v1` with the table `src/lastlight/render/data/bruneton_ms_xyz.bin`, 1,873,606 bytes, SHA-256 `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368` (also pinned as `lastlight.render.v1.TABLE_SHA256`). The rule, the definitions and the bounds above are unchanged.
