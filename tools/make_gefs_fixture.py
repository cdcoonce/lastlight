"""Rebuild `tests/fixtures/gefs/` from the public GEFS-Aerosols bucket (needs the network).

Usage: `uv run python tools/make_gefs_fixture.py`

It reads one real GEFS-Aerosols file (the control run's 0.25 degree 2-D aerosol product, forecast
hour 0) by byte range, with only the standard library and ecCodes, and writes into
`tests/fixtures/gefs/`:

* the file's real `.idx`, verbatim (32 messages);
* the two messages G0 uses (total aerosol and dust optical depth at 550 nm) cropped to a
  21 x 21 sub-grid around the public reference point, concatenated into one GRIB2 file; and
* an `.idx` for that crop, in the same format, so the whole fetch path runs offline against it.

The crop is re-encoded with ecCodes at 24 bits per value, then decoded again; the script fails
unless every cropped value equals the original at the same cell, and prints both anchor values
(`tests/gefs_support.py` asserts them).

The bucket keeps old runs, so the same run can be fetched again; a different run needs the
expected values in `tests/gefs_support.py` updated to match.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import eccodes
import numpy as np

BUCKET_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
CYCLE = "20251001"
STEM = "gefs.chem.t00z.a2d_0p25.f000"
URL = f"{BUCKET_URL}/gefs.{CYCLE}/00/chem/pgrb2ap25/{STEM}.grib2"
USER_AGENT = "lastlight/0.1.0 (public-data client; github.com/cdcoonce/lastlight)"

# The two messages G0 reads, as the idx labels them (everything after the level and forecast).
WANTED = (
    "aerosol=Total aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07",
    "aerosol=Dust dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07",
)
REFERENCE_POINT = (33.45, -112.07)  # the public reference point (docs/design.md section 3)
# Cells kept either side of the reference cell of the 0.25 degree grid: 10 each way.
HALF = 10
MISSING = 1.0e20
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "gefs"


def get(url: str, byte_range: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as reply:
        return reply.read()


def crop_message(data: bytes, rows: slice, cols: slice) -> bytes:
    """Re-encode one message restricted to `rows` and `cols` of its regular lat/lon grid."""
    handle = eccodes.codes_new_from_message(data)
    try:
        ni = eccodes.codes_get_long(handle, "Ni")
        nj = eccodes.codes_get_long(handle, "Nj")
        eccodes.codes_set(handle, "missingValue", MISSING)
        values = eccodes.codes_get_values(handle).reshape(nj, ni)[rows, cols]
        lats = eccodes.codes_get_array(handle, "latitudes").reshape(nj, ni)
        lons = eccodes.codes_get_array(handle, "longitudes").reshape(nj, ni)
        out = eccodes.codes_clone(handle)
    finally:
        eccodes.codes_release(handle)
    try:
        eccodes.codes_set(out, "missingValue", MISSING)
        eccodes.codes_set(out, "Ni", values.shape[1])
        eccodes.codes_set(out, "Nj", values.shape[0])
        eccodes.codes_set(out, "latitudeOfFirstGridPointInDegrees", float(lats[rows, cols][0, 0]))
        eccodes.codes_set(out, "longitudeOfFirstGridPointInDegrees", float(lons[rows, cols][0, 0]))
        eccodes.codes_set(out, "latitudeOfLastGridPointInDegrees", float(lats[rows, cols][-1, -1]))
        eccodes.codes_set(out, "longitudeOfLastGridPointInDegrees", float(lons[rows, cols][-1, -1]))
        if eccodes.codes_get_long(out, "bitsPerValue"):  # not a constant field
            eccodes.codes_set(out, "bitsPerValue", 24)
        eccodes.codes_set_values(out, values.ravel())
        return bytes(eccodes.codes_get_message(out))
    finally:
        eccodes.codes_release(out)


def decode(data: bytes) -> np.ndarray:
    handle = eccodes.codes_new_from_message(data)
    try:
        eccodes.codes_set(handle, "missingValue", MISSING)
        ni = eccodes.codes_get_long(handle, "Ni")
        nj = eccodes.codes_get_long(handle, "Nj")
        return eccodes.codes_get_values(handle).reshape(nj, ni)
    finally:
        eccodes.codes_release(handle)


def main() -> None:
    idx_text = get(URL + ".idx").decode()
    lines = [line for line in idx_text.splitlines() if line]
    offsets = [int(line.split(":")[1]) for line in lines]
    picks = []
    for suffix in WANTED:
        hits = [
            i
            for i, line in enumerate(lines)
            if line.split(":", 6)[3] == "AOTK" and line.split(":", 6)[6] == suffix
        ]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one idx line for {suffix!r}, found {len(hits)}")
        picks.append(hits[0])
    picks.sort()  # file order

    lat, lon = REFERENCE_POINT
    row = round((90.0 - lat) / 0.25)  # row 0 is 90 N, running south
    col = round(((lon % 360.0) - 0.0) / 0.25)  # column 0 is 0 E, running east
    rows, cols = slice(row - HALF, row + HALF + 1), slice(col - HALF, col + HALF + 1)
    shape_crop = (2 * HALF + 1, 2 * HALF + 1)

    blob = b""
    crop_lines = []
    for number, index in enumerate(picks, start=1):
        end = offsets[index + 1] - 1
        original = get(URL, f"{offsets[index]}-{end}")
        cropped = crop_message(original, rows, cols)
        before = decode(original)[rows, cols]
        after = decode(cropped)
        if after.shape != shape_crop or not np.allclose(before, after, rtol=0, atol=1e-6):
            raise SystemExit(f"line {index + 1}: the crop differs from the original")
        fields = lines[index].split(":", 6)
        crop_lines.append(f"{number}:{len(blob)}:{':'.join(fields[2:])}")
        blob += cropped
        print(
            f"idx line {index + 1} ({fields[6].split(':')[0]}): {len(cropped)} bytes, value at the "
            f"reference cell {float(before[HALF, HALF])!r}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{STEM}.grib2.idx").write_text(idx_text)
    (OUT_DIR / f"{STEM}.crop.grib2").write_bytes(blob)
    (OUT_DIR / f"{STEM}.crop.grib2.idx").write_text("\n".join(crop_lines) + "\n")
    print(f"wrote {len(blob)} bytes of GRIB2 and two idx files to {OUT_DIR}")
    print(f"reference cell in the original grid: {(row, col)}; in the crop: {(HALF, HALF)}")


if __name__ == "__main__":
    main()
