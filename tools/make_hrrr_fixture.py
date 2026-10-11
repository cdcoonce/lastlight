"""Rebuild `tests/fixtures/hrrr/` from the public HRRR bucket (needs the network).

Usage: `uv run python tools/make_hrrr_fixture.py`

It reads one real HRRR surface file by byte range, through the same `lastlight.inputs.hrrr`
code the forecast uses, and writes into `tests/fixtures/hrrr/`:

* the file's real `.idx`, verbatim;
* the nine default fields cropped to a 141 x 151 sub-grid around the public reference point,
  concatenated into one GRIB2 file; and
* an `.idx` for that crop, in the same format.

The crop is re-encoded with ecCodes (the same packing as the original, 24 bits per value so the
decoded values are the original's to float precision). The script then decodes the crop and
fails unless every cropped value equals the original at the same cell, and prints each field at
the two anchor cells the tests assert (`tests/hrrr_support.py`).

The bucket keeps old runs, so the same run can be fetched again; a different run needs the
expected values in `tests/hrrr_support.py` and `tests/test_hrrr_idx.py` updated to match.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import eccodes
import numpy as np

from lastlight.inputs import hrrr
from lastlight.inputs.http import UrllibTransport

CYCLE_DATE = dt.date(2026, 10, 10)
CYCLE_HOUR = 22
FORECAST_HOUR = 0

REFERENCE_POINT = (33.45, -112.07)  # the public reference point (docs/design.md section 3)
# Cells kept around the reference point's cell: 140 west, 10 east, 50 south, 90 north. That is
# enough for a 400 km transect toward any sunset azimuth of the year to stay inside the crop.
WEST, EAST, SOUTH, NORTH = 140, 10, 50, 90
WEST_ANCHOR_OFFSET = 66  # the second anchor cell is this many cells west of the reference cell

MISSING = 1.0e20  # stands in for the file's own missing-value code while decoding
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "hrrr"


def crop_message(data: bytes, rows: slice, cols: slice) -> bytes:
    """Re-encode one message restricted to `rows` and `cols` of its grid."""
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
        if (values == MISSING).any():
            eccodes.codes_set(out, "bitmapPresent", 1)
        if eccodes.codes_get_long(out, "bitsPerValue"):  # not a constant field
            eccodes.codes_set(out, "bitsPerValue", 24)
        eccodes.codes_set_values(out, values.ravel())
        return bytes(eccodes.codes_get_message(out))
    finally:
        eccodes.codes_release(out)


def decode_flat(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    handle = eccodes.codes_new_from_message(data)
    try:
        eccodes.codes_set(handle, "missingValue", MISSING)
        return eccodes.codes_get_values(handle).reshape(shape)
    finally:
        eccodes.codes_release(handle)


def main() -> None:
    transport = UrllibTransport()
    idx_text = transport.get(hrrr.idx_url(CYCLE_DATE, CYCLE_HOUR, FORECAST_HOUR)).decode()
    entries = hrrr.parse_idx(idx_text)
    ranges = hrrr.select_ranges(entries, hrrr.DEFAULT_FIELDS)
    order = sorted(hrrr.DEFAULT_FIELDS, key=lambda spec: ranges[spec.name].start)

    with tempfile.TemporaryDirectory() as cache:
        # The grid and the reference cell come from the product code, not from this script.
        fields = hrrr.fetch_fields(
            CYCLE_DATE, CYCLE_HOUR, FORECAST_HOUR, hrrr.DEFAULT_FIELDS, cache, transport=transport
        )
        row, col = hrrr.sample_transect(fields, [REFERENCE_POINT])[0].cell
        messages = hrrr.fetch_messages(
            CYCLE_DATE, CYCLE_HOUR, FORECAST_HOUR, hrrr.DEFAULT_FIELDS, cache, transport=transport
        )
    rows, cols = slice(row - SOUTH, row + NORTH + 1), slice(col - WEST, col + EAST + 1)
    shape = fields.grid.lats.shape
    crop_shape = (SOUTH + NORTH + 1, WEST + EAST + 1)
    anchors = {"reference cell": (SOUTH, WEST), "west cell": (SOUTH, WEST - WEST_ANCHOR_OFFSET)}

    blob = b""
    lines = []
    for number, spec in enumerate(order, start=1):
        original = messages[spec.name]
        cropped = crop_message(original, rows, cols)
        before = decode_flat(original, shape)[rows, cols]
        after = decode_flat(cropped, crop_shape)
        masked = before == MISSING
        if not (np.array_equal(masked, after == MISSING)):
            raise SystemExit(f"{spec.name}: the crop masks different cells from the original")
        if not np.allclose(before[~masked], after[~masked], rtol=0, atol=1e-3):
            raise SystemExit(f"{spec.name}: a cropped value differs from the original")
        entry = next(e for e in entries if (e.variable, e.level) == (spec.variable, spec.level))
        lines.append(
            f"{number}:{len(blob)}:d={entry.date_code}:{entry.variable}:{entry.level}:{entry.forecast}:"
        )
        blob += cropped
        shown = {
            label: ("-" if masked[cell] else f"{float(before[cell])!r}")
            for label, cell in anchors.items()
        }
        print(f"{spec.name:15s} {len(cropped):7d} bytes, masked {int(masked.sum()):5d}, {shown}")

    stem = f"hrrr.t{CYCLE_HOUR:02d}z.wrfsfcf{FORECAST_HOUR:02d}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}.grib2.idx").write_text(idx_text)
    (OUT_DIR / f"{stem}.crop.grib2").write_bytes(blob)
    (OUT_DIR / f"{stem}.crop.grib2.idx").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(blob)} bytes of GRIB2 and two idx files to {OUT_DIR}")
    print(f"reference cell in the crop: {anchors['reference cell']}, original cell: {(row, col)}")


if __name__ == "__main__":
    main()
