"""Decoding one GRIB2 message into a masked array and its grid, with ecCodes.

This module is the only place that imports the GRIB library, so swapping the decoder means
rewriting this file and nothing else. ecCodes was chosen because its binary wheel (`eccodeslib`)
installs with `uv` on macOS and ubuntu-latest without system packages, it reads every packing
HRRR uses (simple, and complex with spatial differencing) and computes the Lambert-conformal
latitudes and longitudes itself, and it can also write GRIB2, which the fixture builder uses to
crop a message to a small sub-grid.

A message is decoded in isolation: `decode_message` takes the bytes of exactly one message.
Cells the file marks as missing (a bitmap, as for cloud base where there is no cloud) come back
masked. The file's own missing-value code (9999 in HRRR) is replaced by a sentinel far outside
any physical value before decoding, so it can never be mistaken for data.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import eccodes
import numpy as np

# Far outside every field in a numerical weather file; used only to find the masked cells.
_MISSING = 1.0e20


class GribError(ValueError):
    """The bytes are not exactly one GRIB2 message this module can decode."""


@dataclass(frozen=True)
class GridSpec:
    """The Lambert-conformal grid a message lives on. Two messages share a grid iff these match."""

    ni: int  # points along a row, west to east
    nj: int  # rows, south to north
    first_lat: float  # degrees north of the first grid point
    first_lon: float  # degrees east of the first grid point
    lov: float  # orientation longitude, degrees east
    latin1: float
    latin2: float
    dx_m: float
    dy_m: float


@dataclass(frozen=True)
class DecodedMessage:
    """One decoded message: which parameter it is, when it is valid, and its values."""

    discipline: int
    category: int
    number: int
    surface_type: int  # GRIB2 code table 4.5: type of the first fixed surface
    reference_date: dt.date  # the model run (cycle) date
    reference_hour: int  # the model run (cycle) hour
    step_hours: int  # forecast hour
    grid: GridSpec
    values: np.ma.MaskedArray  # shape (nj, ni), float64, masked where the file has no value


def _open(data: bytes) -> int:
    """An ecCodes handle for the single message in `data`; the caller must release it."""
    if data[:4] != b"GRIB":
        raise GribError("not a GRIB message: it does not start with the GRIB marker")
    try:
        handle = eccodes.codes_new_from_message(data)
    except eccodes.CodesInternalError as error:
        raise GribError(f"not a decodable GRIB2 message: {error}") from error
    try:
        if eccodes.codes_get_long(handle, "edition") != 2:
            raise GribError("only GRIB edition 2 is supported")
        if eccodes.codes_get_string(handle, "gridType") != "lambert":
            raise GribError("only Lambert-conformal grids are supported")
        size = eccodes.codes_get_message_size(handle)
        if size != len(data):
            raise GribError(
                f"expected exactly one GRIB message of {len(data)} bytes, found one of {size}"
            )
    except eccodes.CodesInternalError as error:
        eccodes.codes_release(handle)
        raise GribError(f"not a decodable GRIB2 message: {error}") from error
    except BaseException:
        eccodes.codes_release(handle)
        raise
    return handle


def _long(handle: int, key: str) -> int:
    return int(eccodes.codes_get_long(handle, key))


def _double(handle: int, key: str) -> float:
    return float(eccodes.codes_get_double(handle, key))


def _grid_spec(handle: int) -> GridSpec:
    return GridSpec(
        ni=_long(handle, "Ni"),
        nj=_long(handle, "Nj"),
        first_lat=_double(handle, "latitudeOfFirstGridPointInDegrees"),
        first_lon=_double(handle, "longitudeOfFirstGridPointInDegrees"),
        lov=_double(handle, "LoVInDegrees"),
        latin1=_double(handle, "Latin1InDegrees"),
        latin2=_double(handle, "Latin2InDegrees"),
        dx_m=_double(handle, "DxInMetres"),
        dy_m=_double(handle, "DyInMetres"),
    )


def decode_message(data: bytes) -> DecodedMessage:
    """Decode the single GRIB2 message in `data`."""
    handle = _open(data)
    try:
        grid = _grid_spec(handle)
        if _long(handle, "scanningMode") != 64:
            raise GribError("only west-to-east, south-to-north scanning is supported")
        eccodes.codes_set(handle, "missingValue", _MISSING)
        flat = np.asarray(eccodes.codes_get_values(handle), dtype=np.float64)
        if flat.size != grid.ni * grid.nj:
            raise GribError(f"{flat.size} values for a {grid.ni} x {grid.nj} grid")
        values = np.ma.array(
            flat.reshape(grid.nj, grid.ni), mask=(flat == _MISSING).reshape(grid.nj, grid.ni)
        )
        date = _long(handle, "dataDate")
        return DecodedMessage(
            discipline=_long(handle, "discipline"),
            category=_long(handle, "parameterCategory"),
            number=_long(handle, "parameterNumber"),
            surface_type=_long(handle, "typeOfFirstFixedSurface"),
            reference_date=dt.date(date // 10000, date // 100 % 100, date % 100),
            reference_hour=_long(handle, "dataTime") // 100,
            step_hours=_long(handle, "step"),
            grid=grid,
            values=values,
        )
    except eccodes.CodesInternalError as error:
        raise GribError(f"cannot decode the message: {error}") from error
    finally:
        eccodes.codes_release(handle)


def grid_coordinates(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of every grid point of the message, each shaped (nj, ni).

    Latitudes are degrees north; longitudes are degrees east in [-180, 180] (the file stores
    them in [0, 360)).
    """
    handle = _open(data)
    try:
        grid = _grid_spec(handle)
        shape = (grid.nj, grid.ni)
        lats = np.asarray(eccodes.codes_get_array(handle, "latitudes"), dtype=np.float64)
        lons = np.asarray(eccodes.codes_get_array(handle, "longitudes"), dtype=np.float64)
        return lats.reshape(shape), ((lons + 180.0) % 360.0 - 180.0).reshape(shape)
    except eccodes.CodesInternalError as error:
        raise GribError(f"cannot read the grid: {error}") from error
    finally:
        eccodes.codes_release(handle)
