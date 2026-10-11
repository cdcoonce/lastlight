# G0 pre-registered criterion

Registered in the commit that adds this file, before any code that produces a G0 result and before any result exists. Issue #10 carried an earlier sentence (a 20 degree reference band, `late` counted); the conductor's comment of 2026-10-11 replaced it, because WHTM1's frame never reaches 20 degrees of elevation. This file holds the replacement, verbatim, and nothing here may be changed after a result has been seen. A correction is a dated amendment at the bottom of this file, made in a later commit and repeated in the report.

## The criterion

> For each date d and each counted sun-elevation band s, let D(d, s) be the mean, over the sweep elevations inside s, over WHTM1's contrast bands k, and over both channels c in {log(R/G), log(B/G)}, of |C_d(s, k, c) - median over dates of C(s, k, c)|. Here C is the rendered within-frame contrast in linear RGB: band k's median minus the reference band's, computed for WHTM1's frozen geometry. Bands are mapped from rows to view elevation with the frozen fit's nominal pixel scale, and taken over the fit's sky columns. **G0 passes if D(d, s) >= 0.05 in at least 2 of the 3 counted bands on at least 25 percent of dates.**

Counted bands, from `lastlight.geometry.ELEVATION_BANDS`: `golden` (+6 to 0), `early` (0 to -2) and `mid` (-2 to -5). The band `late` (-5 to -8) is reported and not counted: tier A scores the sun from +6 to -6 by default, because city light contaminates WHTM1's low sky below about -6.

| Parameter | Value |
| --- | --- |
| Threshold on D(d, s) | 0.05 |
| Counted bands a date must reach the threshold in | at least 2 of 3 |
| Share of dates that must be such dates | at least 25 percent |
| Period | evenings 2025-10-01 to 2026-09-30 |

Reported without changing the verdict: the same statistic recomputed at both ends of the fit's `uncertainty.px_per_deg` range, and delta-E 2000 of the absolute curve for tier B.

## Definitions fixed before the first result

The criterion leaves these open. They are fixed here, from the design contract and the code that exists, so that none of them can be chosen after a result.

1. **Which sweep elevations are inside a band.** The sweep is `geometry.ELEVATION_SWEEP` (+6.0 to -8.0 in half degrees, 29 values). A band `(upper, lower)` holds the elevations `e` with `lower < e <= upper`; the last band, `late`, also holds -8.0. Every sweep elevation is in exactly one band: `golden` 12 values, `early` 4, `mid` 6, `late` 7.
2. **The median over dates** is taken separately at each sweep elevation, band k and channel c, over the dates that have a rendered curve (the dates used). D is then the mean of the absolute differences over the elevations inside s, the 5 bands k and the 2 channels.
3. **"25 percent of dates"** is a share of the dates used (those with a rendered curve), not of the dates attempted. Dropped dates are listed with their reason; the share of all dates attempted is reported next to it.
4. **The bands and the reference** are the rows of `[camera.contrast]` for WHTM1 in `registry.toml` (`bands_rows` and `reference_rows`, pixel rows above the frozen skyline). The sky columns are the columns in which every band and the reference lie inside the fit's sky mask (`lastlight.observe.bands.columns_inside`).
5. **Row to view elevation.** The skyline's elevation is the median, over the sky columns, of the elevation of the fitted skyline row (the fit's own camera model). A band's elevation is that plus its centre row offset, `(lo + hi - 1) / 2` rows, divided by the nominal pixel scale (`pixels_per_degree` of the fit's `hfov_deg`). The sensitivity runs replace the nominal scale by each end of `uncertainty.px_per_deg`, keeping the skyline where it is.
6. **View direction.** Each band is rendered toward the fit's centre azimuth, at the relative azimuth to the sun at that sweep elevation on that date (the fit's centre azimuth minus the sun's azimuth, computed internally and never published).
7. **Linear RGB** is linear sRGB (D65, the matrix from CIE XYZ), the webcam proxy, taken from the renderer's XYZ with no gamut clipping and no gain. A channel is floored at the smallest fraction of the largest channel that `lastlight.observe` can still read in an 8-bit frame (code 6 against code 249, linearised), because single scattering near the horizon is outside the sRGB gamut and a logarithm needs a positive number. The report counts how often the floor was used.
8. **A date's contrast** is log(R/G) and log(B/G) of a band minus the same of the reference band, for each band and each sweep elevation.
9. **Inputs.** HRRR analyses (forecast hour 0) at the whole UTC hour nearest the instant the sun reaches the horizon at the public reference point, for smoke aerosol (AOTK) and water vapour (PWAT); an aerosol archive that needs no account for total and dust aerosol optical depth at 550 nm; the monthly ozone climatology. The report states the sources actually used and every date dropped, with its reason.
10. **The five most extreme dates** are the five with the largest mean of D(d, s) over the three counted bands.
11. **Also reported, and not part of the verdict:** (a) delta-E 2000 of each date's absolute curve (the forecast's own curve, `render_curve`) against the period's median curve, by band `early`, `mid` and `late`; (b) a seasonal-control check, the same rule with the median taken over the dates within 15 days of d instead of over the whole period, because the sun's azimuth and the ozone column change with the season and a year-wide median counts that as variation.
