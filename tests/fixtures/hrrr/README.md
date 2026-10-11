# HRRR test fixtures

Recorded from the public NOAA bucket (`noaa-hrrr-bdp-pds`), one HRRR surface file: the run of
2026-10-10 hour 22, forecast hour 0 (`hrrr.t22z.wrfsfcf00.grib2`).

| File | What it is |
| --- | --- |
| `hrrr.t22z.wrfsfcf00.grib2.idx` | The real `.idx` of that file, verbatim (170 messages). Tests check byte-range selection against NOAA's own offsets. |
| `hrrr.t22z.wrfsfcf00.crop.grib2` | The nine default fields of that file, each cropped to a 141 x 151 sub-grid of the 3 km grid around Phoenix and concatenated into one GRIB2 file (about 360 kB). Decoded values equal the original messages' (to float precision); only the grid is smaller. |
| `hrrr.t22z.wrfsfcf00.crop.grib2.idx` | An `.idx` for the crop, in the same format, so the whole fetch path runs offline against it. Its offsets belong to the crop, not to the real file. |

Regenerate them with `uv run python tools/make_hrrr_fixture.py` (needs the network, writes only
into this directory). The script prints the value of every field at the two anchor cells the
tests assert, and fails if a cropped value differs from the original.

The sub-grid reaches 150 km south, 270 km north and 420 km west of the reference point (151 by
141 cells), so that the default 400 km transect toward any sunset azimuth of the year (southwest
in winter, northwest in summer) stays inside it. Nothing here locates anything beyond the
public reference point's metro area.
