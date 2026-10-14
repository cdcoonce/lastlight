# Renderer v1: the camera magnitude check

Verdict: NOT YET DECIDABLE (1 of 5 counted evenings)

The pre-registered check of [`criterion.md`](criterion.md), run by `lastlight renderer-check` over WHTM1's observed evenings. It asks one coarse question: is renderer v1's within-frame contrast within a factor of two of what the camera sees? A pass says the magnitude is plausible. It says nothing about skill.

## Verdict

1 of the 5 counted evenings the verdict needs so far: 2026-10-13. With fewer than 5 the criterion's verdict is not yet decidable and v0 keeps painting.

## The renderer under test

Engine `v1` with the table `src/lastlight/render/data/bruneton_ms_xyz.bin`, whose SHA-256 is `c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368` (the criterion's amendment of 2026-10-12). The check hashes the bytes of the table file the renderer loads and compares them with that constant, and compares three probes of the engine with the values registered with it; it refuses to give a verdict, and writes nothing, if either differs. A report exists only because both matched.

## Evenings

| Date | Status | Frames used | Cloud mean (%) | aod_550 | r_e v1 | r_e v0 | r_e v1, scale low | r_e v1, scale high | r_e v0, log(R/G) | r_e v1, log(R/G) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-10 | not counted, cloud 99.12% | 4 | 99.12 | 0.0319 | 5.9142 | 25.4116 | 9.8098 | 5.8063 | 16.8820 | 12.7853 |
| 2026-10-11 | not counted, cloud 31.41% | 4 | 31.41 | 0.0686 | 0.6882 | 10.7200 | 1.2393 | 0.6902 | 11.5571 | 5.7688 |
| 2026-10-12 | not counted, cloud 49.53% | 4 | 49.53 | 0.0760 | 0.2491 | 4.9109 | 0.3951 | 0.2480 | 2.6777 | 1.1598 |
| 2026-10-13 | counted (first 5, number 1) | 3 | 5.53 | 0.0451 | 1.2229 | 9.9524 | 2.2588 | 1.1813 | 2.0689 | 1.1493 |

## Reading the table

- **r_e** for an evening is the median over its used frames of the absolute rendered log(B/G) contrast of the lowest band (minus the reference's) divided by the median over the same frames of the absolute observed one: a ratio of two medians, `inf` when the observed median is 0. The verdict column is `r_e v1`.
- **Reported beside the verdict, not part of it:** `r_e v0` (the same ratio for the single-scattering renderer), `r_e v1` at both ends of the fit's pixel-scale range (`scale low` and `scale high`), and the ratios for log(R/G).
- **Counted** means the mean total cloud cover along the sightline, from HRRR's analysis, is at most 10 percent. Evenings not counted are shown with their cloud and their ratios, and do not enter the verdict. Dropped evenings have no ratios.
- The per-frame pairs (sun elevation, observed, rendered with v0 and v1) are in `frames.csv`; the per-evening numbers in `evenings.csv`.

## What this check cannot say

See [What this check cannot say](criterion.md#what-this-check-cannot-say) in the criterion. It is not repeated here.
