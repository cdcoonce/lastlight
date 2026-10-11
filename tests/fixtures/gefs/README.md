# GEFS-Aerosols test fixtures

Recorded from the public NOAA bucket (`noaa-gefs-pds`), one GEFS-Aerosols 2-D file: the control
run of 2025-10-01, first cycle of the day, forecast hour 0
(`gefs.chem.t00z.a2d_0p25.f000.grib2`).

| File | What it is |
| --- | --- |
| `gefs.chem.t00z.a2d_0p25.f000.grib2.idx` | The real `.idx` of that file, verbatim (32 messages). Tests check byte-range selection against NOAA's own offsets. |
| `gefs.chem.t00z.a2d_0p25.f000.crop.grib2` | The two messages G0 reads (aerosol optical depth at 550 nm, total aerosol and dust), each cropped to a 21 x 21 sub-grid of the 0.25 degree grid around Phoenix and concatenated into one GRIB2 file (about 3 kB). Decoded values equal the original messages' to float precision; only the grid is smaller. |
| `gefs.chem.t00z.a2d_0p25.f000.crop.grib2.idx` | An `.idx` for the crop, in the same format, so the whole fetch path runs offline against it. Its offsets belong to the crop, not to the real file. |

Regenerate them with `uv run python tools/make_gefs_fixture.py` (needs the network, writes only
into this directory). The script prints the value of both fields at the reference cell, which
`tests/gefs_support.py` asserts, and fails if a cropped value differs from the original.
