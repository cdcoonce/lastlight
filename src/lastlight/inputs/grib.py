"""Decoding one GRIB2 message into a masked array and its grid, with ecCodes.

This module is the only place that imports the GRIB library, so swapping the decoder means
rewriting this file and nothing else. ecCodes was chosen because its binary wheel (`eccodeslib`)
installs with `uv` on macOS and ubuntu-latest without system packages, it reads every packing
HRRR uses (simple, and complex with spatial differencing) and computes the Lambert-conformal
latitudes and longitudes itself, and it can also write GRIB2, which the fixture builder uses to
crop a message to a small sub-grid.

A message is decoded in isolation: `decode_message` (HRRR's Lambert-conformal grid) and
`decode_latlon_message` (GEFS-Aerosols' regular latitude/longitude grid) take the bytes of exactly
one message. Cells the file marks as missing (a bitmap, as for cloud base where there is no cloud)
come back masked. The file's own missing-value code (9999 in HRRR) is replaced by a sentinel far
outside any physical value before decoding, so it can never be mistaken for data.
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


def _open(data: bytes, grid_type: str = "lambert", grid_name: str = "Lambert-conformal") -> int:
    """An ecCodes handle for the single message in `data`; the caller must release it.

    The message must be on a grid of `grid_type` (ecCodes' name), called `grid_name` in errors.
    """
    if data[:4] != b"GRIB":
        raise GribError("not a GRIB message: it does not start with the GRIB marker")
    try:
        handle = eccodes.codes_new_from_message(data)
    except eccodes.CodesInternalError as error:
        raise GribError(f"not a decodable GRIB2 message: {error}") from error
    try:
        if eccodes.codes_get_long(handle, "edition") != 2:
            raise GribError("only GRIB edition 2 is supported")
        if eccodes.codes_get_string(handle, "gridType") != grid_type:
            raise GribError(f"only {grid_name} grids are supported")
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


# --- a regular latitude/longitude grid (GEFS-Aerosols) ---------------------------------------


@dataclass(frozen=True)
class LatLonMessage:
    """One decoded message on a regular latitude/longitude grid, with its aerosol labels.

    `aerosol_type` (GRIB2 code table 4.233: 62000 total aerosol, 62001 dust) and `wavelength_nm`
    (the band the optical depth is for, low and high) are None for a message with no aerosol
    product definition. `lats` (degrees north) and `lons` (degrees east, in [-180, 180)) are the
    grid's axes in the file's order, and `values` has shape (len(lats), len(lons)).
    """

    discipline: int
    category: int
    number: int
    surface_type: int
    reference_date: dt.date
    reference_hour: int
    step_hours: int
    aerosol_type: int | None
    wavelength_nm: tuple[float, float] | None
    lats: np.ndarray
    lons: np.ndarray
    values: np.ma.MaskedArray


def _optional_long(handle: int, key: str) -> int | None:
    try:
        return _long(handle, key)
    except eccodes.CodesInternalError:
        return None


def _wavelength_nm(handle: int) -> tuple[float, float] | None:
    """The aerosol wavelength band in nm, from the scaled values of the product definition."""
    bounds = []
    for ordinal in ("First", "Second"):
        value = _optional_long(handle, f"scaledValueOf{ordinal}Wavelength")
        factor = _optional_long(handle, f"scaleFactorOf{ordinal}Wavelength")
        if value is None or factor is None:
            return None
        bounds.append(value * 10.0 ** (9 - factor))  # metres are value * 10^-factor
    return bounds[0], bounds[1]


def decode_latlon_message(data: bytes) -> LatLonMessage:
    """Decode the single GRIB2 message in `data`, which must be on a regular lat/lon grid."""
    handle = _open(data, "regular_ll", "latitude/longitude")
    try:
        ni, nj = _long(handle, "Ni"), _long(handle, "Nj")
        if _long(handle, "scanningMode") != 0:
            raise GribError("only west-to-east, north-to-south scanning is supported")
        eccodes.codes_set(handle, "missingValue", _MISSING)
        flat = np.asarray(eccodes.codes_get_values(handle), dtype=np.float64)
        if flat.size != ni * nj:
            raise GribError(f"{flat.size} values for a {ni} x {nj} grid")
        lats = np.asarray(eccodes.codes_get_array(handle, "latitudes"), dtype=np.float64)
        lons = np.asarray(eccodes.codes_get_array(handle, "longitudes"), dtype=np.float64)
        values = np.ma.array(flat.reshape(nj, ni), mask=(flat == _MISSING).reshape(nj, ni))
        date = _long(handle, "dataDate")
        return LatLonMessage(
            discipline=_long(handle, "discipline"),
            category=_long(handle, "parameterCategory"),
            number=_long(handle, "parameterNumber"),
            surface_type=_long(handle, "typeOfFirstFixedSurface"),
            reference_date=dt.date(date // 10000, date // 100 % 100, date % 100),
            reference_hour=_long(handle, "dataTime") // 100,
            step_hours=_long(handle, "step"),
            aerosol_type=_optional_long(handle, "aerosolType"),
            wavelength_nm=_wavelength_nm(handle),
            lats=lats.reshape(nj, ni)[:, 0],
            lons=((lons.reshape(nj, ni)[0, :] + 180.0) % 360.0) - 180.0,
            values=values,
        )
    except eccodes.CodesInternalError as error:
        raise GribError(f"cannot decode the message: {error}") from error
    finally:
        eccodes.codes_release(handle)
