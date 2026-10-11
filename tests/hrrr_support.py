"""Shared support for the HRRR tests: the recorded fixtures and a transport that serves them.

`tests/fixtures/hrrr/` holds three recorded files (see its README for how they were made):

* the real `.idx` of one HRRR surface file, verbatim, for byte-range selection against the
  offsets NOAA published;
* a crop of that file's nine default fields to a small sub-grid around Phoenix, concatenated
  into one GRIB2 file; and
* the `.idx` of the crop, in the same format, so the whole fetch path runs against it offline.

The crop is 141 rows by 151 columns of the 3 km grid: 420 km north to south, wide enough that a
400 km transect toward any sunset azimuth of the year stays inside it. Its cell (70, 140) is the one
nearest the public reference point, and cell (70, 74) lies about 200 km to the west.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lastlight.inputs import hrrr
from lastlight.inputs.http import ByteRange, FetchError

FIXTURES = Path(__file__).parent / "fixtures" / "hrrr"

CYCLE_DATE = dt.date(2026, 10, 10)
CYCLE_HOUR = 22

REAL_IDX_TEXT = (FIXTURES / "hrrr.t22z.wrfsfcf00.grib2.idx").read_text()
CROP_IDX_TEXT = (FIXTURES / "hrrr.t22z.wrfsfcf00.crop.grib2.idx").read_text()
CROP_GRIB = (FIXTURES / "hrrr.t22z.wrfsfcf00.crop.grib2").read_bytes()

CROP_SHAPE = (141, 151)
REFERENCE_CELL = (70, 140)
WEST_CELL = (70, 74)
WEST_EDGE_CELL = (70, 0)
CORNER_CELL = (0, 0)  # cloud top has no value here: the model has no cloud there
CLOUD_TOP_MASKED_CELLS = 7693

# What the real HRRR file holds at those two cells, decoded from the full 1799 x 1059 grid
# before cropping (and cross-checked with a second, independent GRIB decoder). Both cells carry
# every field. Cloud base and top are geopotential metres.
EXPECTED_AT_REFERENCE_CELL: dict[str, float] = {
    "LCDC": 99.0,
    "MCDC": 100.0,
    "HCDC": 19.0,
    "TCDC": 100.0,
    "AOTK": 0.0,
    "PWAT": 44.39161813259125,
    "VIS": 22700.0,
    "HGT_CLOUD_BASE": 3400.7301864624023,
    "HGT_CLOUD_TOP": 6119.95442199707,
}
EXPECTED_AT_WEST_CELL: dict[str, float] = {
    "LCDC": 6.0,
    "MCDC": 100.0,
    "HCDC": 99.0,
    "TCDC": 100.0,
    "AOTK": 0.0,
    "PWAT": 40.14161813259125,
    "VIS": 23800.0,
    "HGT_CLOUD_BASE": 3532.1051864624023,
    "HGT_CLOUD_TOP": 8451.45442199707,
}

# The crop's idx lines, in file order: variable, level, and the byte range inside CROP_GRIB.
CROP_MESSAGES: dict[str, ByteRange] = {}


def _index_crop() -> None:
    rows = [line.split(":") for line in CROP_IDX_TEXT.splitlines() if line]
    offsets = [int(row[1]) for row in rows] + [len(CROP_GRIB)]
    names = ["VIS", "PWAT", "AOTK", "LCDC", "MCDC", "HCDC", "TCDC", "HGT_CLOUD_BASE", "HGT_CLOUD_TOP"]
    for name, start, stop in zip(names, offsets, offsets[1:], strict=True):
        CROP_MESSAGES[name] = ByteRange(start, stop - 1)


_index_crop()


def crop_message(name: str) -> bytes:
    """The bytes of one cropped message, by `FieldSpec.name`."""
    byte_range = CROP_MESSAGES[name]
    assert byte_range.end is not None
    return CROP_GRIB[byte_range.start : byte_range.end + 1]


@dataclass(frozen=True)
class Call:
    method: str  # "HEAD" or "GET"
    url: str
    byte_range: ByteRange | None


class FakeTransport:
    """An in-memory `Transport`: a dict of url -> bytes, honouring byte ranges, recording calls."""

    def __init__(self, files: Mapping[str, bytes]) -> None:
        self.files = dict(files)
        self.calls: list[Call] = []

    def exists(self, url: str) -> bool:
        self.calls.append(Call("HEAD", url, None))
        return url in self.files

    def get(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        self.calls.append(Call("GET", url, byte_range))
        if url not in self.files:
            raise FetchError(f"{url}: HTTP 404")
        body = self.files[url]
        if byte_range is None:
            return body
        stop = len(body) if byte_range.end is None else byte_range.end + 1
        return body[byte_range.start : stop]

    def ranged_gets(self) -> list[Call]:
        return [call for call in self.calls if call.method == "GET" and call.byte_range]

    def bytes_served_by_ranges(self) -> int:
        total = 0
        for call in self.ranged_gets():
            assert call.byte_range is not None
            stop = len(self.files[call.url]) if call.byte_range.end is None else call.byte_range.end + 1
            total += stop - call.byte_range.start
        return total


def crop_transport(
    cycle_date: dt.date = CYCLE_DATE, cycle_hour: int = CYCLE_HOUR, forecast_hour: int = 0
) -> FakeTransport:
    """A transport that serves the crop (and its idx) at the real URLs for one cycle."""
    return FakeTransport(
        {
            hrrr.grib_url(cycle_date, cycle_hour, forecast_hour): CROP_GRIB,
            hrrr.idx_url(cycle_date, cycle_hour, forecast_hour): CROP_IDX_TEXT.encode(),
        }
    )
