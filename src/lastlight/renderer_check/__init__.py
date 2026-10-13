"""The camera magnitude check that decides whether renderer v1 may paint the daily forecast.

`docs/gates/renderer-v1/criterion.md` holds the pre-registered rule and its definitions;
`docs/gates/renderer-v1/report.md` the result. This package implements that file and nothing
else, in the order the data flows:

- `rules`: the numbers the criterion fixes (the ratio, the cloud filter, the first 5, the
  verdict) on plain values.
- `observed`: the private `observed.json` files from `lastlight observe`, read as untrusted input.
- `store`, `fetch`: each considered evening's inputs (G0's atmosphere, HRRR's total cloud cover)
  collected once and kept in the check's own cache.
- `render`: a used frame rendered with v0 and with v1 through G0's seam.
- `guards`: the two refusals, the table digest and the v1 physics fingerprint.
- `evaluate`: observed files and inputs to a verdict, evening by evening.
- `report`: the public outputs (dates, sun elevations and derived numbers only).
- `run`: the stages of `lastlight renderer-check`.
"""
