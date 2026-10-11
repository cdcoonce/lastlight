"""Shared support for the GEFS-Aerosols tests: the recorded fixtures and a transport serving them.

`tests/fixtures/gefs/` holds three recorded files (see its README for how they were made):

* the real `.idx` of one GEFS-Aerosols 2-D file (control run, 0.25 degrees, forecast hour 0),
  verbatim, for byte-range selection against the offsets NOAA published;
* a crop of its two G0 fields (total and dust aerosol optical depth at 550 nm) to a 21 x 21
  sub-grid around the public reference point, concatenated into one GRIB2 file; and
* the `.idx` of the crop, in the same format, so the whole fetch path runs offline against it.

The crop is rows 216 to 236 (36.0 to 31.0 N) and columns 982 to 1002 (245.5 to 250.5 E) of the
0.25 degree grid; its cell (10, 10) is the one nearest the public reference point.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import eccodes
from hrrr_support import FakeTransport

from lastlight.inputs import gefs
from lastlight.inputs.http import ByteRange

FIXTURES = Path(__file__).parent / "fixtures" / "gefs"
STEM = "gefs.chem.t00z.a2d_0p25.f000"
FIXTURE_CYCLE_DATE = dt.date(2025, 10, 1)

REAL_IDX_TEXT = (FIXTURES / f"{STEM}.grib2.idx").read_text()
CROP_IDX_TEXT = (FIXTURES / f"{STEM}.crop.grib2.idx").read_text()
CROP_GRIB = (FIXTURES / f"{STEM}.crop.grib2").read_bytes()

CROP_SHAPE = (21, 21)
REFERENCE_CELL = (10, 10)

# What the real file holds at the reference cell (33.5 N, 248.0 E), decoded from the full
# 1440 x 721 grid before cropping.
TOTAL_AT_REFERENCE = 0.024992790312500006
DUST_AT_REFERENCE = 0.0021700384130859383

# The crop's messages in file order, as byte ranges inside CROP_GRIB (the last one is open-ended,
# as the idx makes the last message of a file).
CROP_MESSAGES: dict[str, ByteRange] = {
    "TOTAL_AOD_550": ByteRange(0, 1481),
    "DUST_AOD_550": ByteRange(1482, None),
}


def crop_message(name: str) -> bytes:
    byte_range = CROP_MESSAGES[name]
    stop = len(CROP_GRIB) if byte_range.end is None else byte_range.end + 1
    return CROP_GRIB[byte_range.start : stop]


def edited(data: bytes, *, cycle: tuple[dt.date, int] | None = None, **keys: float) -> bytes:
    """One message with its run changed and/or ecCodes keys set; the length stays the same."""
    handle = eccodes.codes_new_from_message(data)
    try:
        if cycle is not None:
            date, hour = cycle
            eccodes.codes_set(handle, "dataDate", date.year * 10000 + date.month * 100 + date.day)
            eccodes.codes_set(handle, "dataTime", hour * 100)
        for key, value in keys.items():
            eccodes.codes_set(handle, key, value)
        return bytes(eccodes.codes_get_message(handle))
    finally:
        eccodes.codes_release(handle)


def redated_crop(cycle_date: dt.date, cycle_hour: int, forecast_hour: int = 0) -> bytes:
    """The crop with each message re-dated to another run, so any evening can be served."""
    parts = [edited(crop_message(name), cycle=(cycle_date, cycle_hour)) for name in CROP_MESSAGES]
    assert [len(part) for part in parts] == [len(crop_message(name)) for name in CROP_MESSAGES]
    return b"".join(parts)


def crop_transport(
    cycle_date: dt.date = FIXTURE_CYCLE_DATE, cycle_hour: int = 0, forecast_hour: int = 0
) -> FakeTransport:
    """A transport that serves the (re-dated) crop and its idx at the real URLs for one run."""
    return FakeTransport(
        {
            gefs.chem_url(cycle_date, cycle_hour, forecast_hour): redated_crop(
                cycle_date, cycle_hour, forecast_hour
            ),
            gefs.idx_url(cycle_date, cycle_hour, forecast_hour): CROP_IDX_TEXT.encode(),
        }
    )
