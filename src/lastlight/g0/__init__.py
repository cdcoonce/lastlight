"""The G0 gate: does the rendered twilight curve vary enough from day to day to be measured?

`docs/gates/G0/criterion.md` holds the pre-registered criterion; `docs/gates/G0/report.md` the
result. The modules, in the order the data flows through them:

- `view`: WHTM1's bands as view elevations (rows above the frozen skyline to degrees), the
  sun's azimuth relative to the camera's, and the pixel scales the sensitivity runs use.
- `contrast`: the rendered within-frame contrast, `log(R/G)` and `log(B/G)` of each band minus
  the reference band, in linear sRGB, the same quantity `lastlight.observe` measures.
- `record`: one date's result (or its reason for being dropped) and the cache that makes the
  run resumable.
- `transport`: the polite wrapper around the HTTP transport for a year of requests.
- `collect`: one evening from archived inputs to a record, and the resumable run over a range.
- `stats`: D(d, s), the 2 of 3 bands / 25 percent of dates rule, the seasonal control and
  delta-E 2000, all on arrays.
- `analysis`: records to a `Study`.
- `report`, `figures`: the report, the results file and the figures.
"""
