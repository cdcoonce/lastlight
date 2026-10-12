"""GEFS-Aerosols ingest: total and dust aerosol optical depth for an evening (issue #10).

HRRR carries wildfire smoke only (`AOTK`), so the G0 spread study (#10) needs another source for
dust and the rest of the aerosol column. NOAA's GEFS-Aerosols (the GEFS control run with the
GOCART aerosol model) publishes 0.25 degree 2-D aerosol products on the public AWS bucket
`noaa-gefs-pds`, no account needed, kept for years. The flow mirrors `lastlight.inputs.hrrr`: the
`.idx` beside each file lists its messages with byte offsets, so only the two wanted messages
(optical depth at 550 nm for total aerosol and for dust) are downloaded, each cached by URL plus
range, decoded with ecCodes (`lastlight.inputs.grib`), and checked to be the aerosol, wavelength
and run that was asked for before the value of the nearest cell is read.

Which run: the **analysis** (forecast hour 0) of the latest six-hourly cycle at or before the
instant the sun reaches the horizon at the reference point. The analysis, not a forecast, for the
same reason G0 reads HRRR analyses: the question is how much the real atmosphere varies, and a
forecast would mix its error into the answer. The instant stays internal and never leaves
`aerosol_run`.

Why not NASA GEOS-FP, tried first (`docs/gates/G0/report.md`): the NCCS OPeNDAP server's
`tavg3_2d_aer_Nx` assimilation collection starts on 27 February 2026, rolling forward, so it does
not cover a full year back from the autumn of 2026; the file-based archive would mean downloading
whole global files for every date.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from lastlight import geometry
from lastlight.atmosphere import AerosolColumn
from lastlight.inputs import grib
from lastlight.inputs.http import ByteRange, Transport, UrllibTransport

BUCKET_URL: Final = "https://noaa-gefs-pds.s3.amazonaws.com"
SOURCE_LABEL: Final = "GEFS-Aerosols analysis"
"""How the aerosol column names its source in the notes."""

CYCLE_HOURS: Final = (0, 6, 12, 18)
FORECAST_STEP_HOURS: Final = 3
MAX_FORECAST_HOUR: Final = 384
"""The 2-D aerosol product is written every three hours, out to sixteen days."""

SAMPLING_METHOD: Final = "nearest-cell"
_CELL_TOLERANCE: Final = 1.01  # slack on the half-diagonal, for float noise at a cell corner
_PACKING_NOISE: Final = 1e-4  # a tiny negative optical depth is the packing's rounding, not data
_WAVELENGTH_TOLERANCE_NM: Final = 0.5
_EXPECTED_PARAMETER: Final = (0, 20, 102)  # aerosol optical thickness
_WHOLE_COLUMN: Final = 10  # GRIB2 code table 4.5: entire atmosphere


class AerosolArchiveError(RuntimeError):
    """Archive data that is missing, inconsistent or not what was asked for."""


# --- URLs and runs --------------------------------------------------------------------------------


def _check_run(cycle_hour: int, forecast_hour: int) -> None:
    if cycle_hour not in CYCLE_HOURS:
        raise ValueError(f"cycle hour must be one of {CYCLE_HOURS}, got {cycle_hour}")
    if forecast_hour % FORECAST_STEP_HOURS or not 0 <= forecast_hour <= MAX_FORECAST_HOUR:
        raise ValueError(
            f"forecast hour must be a multiple of {FORECAST_STEP_HOURS} from 0 to "
            f"{MAX_FORECAST_HOUR}, got {forecast_hour}"
        )


def chem_url(cycle_date: dt.date, cycle_hour: int, forecast_hour: int = 0) -> str:
    """The 0.25 degree 2-D aerosol GRIB2 file of one cycle and forecast hour in the bucket."""
    _check_run(cycle_hour, forecast_hour)
    return (
        f"{BUCKET_URL}/gefs.{cycle_date:%Y%m%d}/{cycle_hour:02d}/chem/pgrb2ap25/"
        f"gefs.chem.t{cycle_hour:02d}z.a2d_0p25.f{forecast_hour:03d}.grib2"
    )


def idx_url(cycle_date: dt.date, cycle_hour: int, forecast_hour: int = 0) -> str:
    return chem_url(cycle_date, cycle_hour, forecast_hour) + ".idx"


@dataclass(frozen=True)
class RunPlan:
    cycle_date: dt.date
    cycle_hour: int
    forecast_hour: int


def aerosol_run(evening: dt.date, lat: float, lon: float) -> RunPlan:
    """The analysis to read for the evening of `evening` at (`lat`, `lon`).

    The latest cycle (every six hours) at or before the instant the sun reaches geometric
    elevation 0, at forecast hour 0. The instant is internal to this function.
    """
    instant = geometry.elevation_crossings(evening, lat, lon)[0.0]
    cycle = instant.replace(hour=instant.hour // 6 * 6, minute=0, second=0, microsecond=0)
    return RunPlan(cycle.date(), cycle.hour, 0)


# --- the .idx -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdxEntry:
    """One line of an `.idx`: `number:offset:d=YYYYMMDDHH:variable:level:forecast:label...`.

    `label` is everything after the forecast, rejoined: GEFS-Aerosols names the aerosol, its size
    bound and its wavelength band there (`aerosol=Dust dry:aerosol_size <2e-05:aerosol_wavelength
    >=5.45e-07,<=5.65e-07`), and the same variable repeats with different labels.
    """

    number: int
    offset: int
    date_code: str
    variable: str
    level: str
    forecast: str
    label: str


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse an `.idx`. Offsets must strictly increase: a message ends where the next begins."""
    entries: list[IdxEntry] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6 or not parts[2].startswith("d="):
            raise AerosolArchiveError(f"malformed idx line {line_number}: {raw!r}")
        try:
            number, offset = int(parts[0]), int(parts[1])
        except ValueError:
            raise AerosolArchiveError(f"malformed idx line {line_number}: {raw!r}") from None
        if entries and offset <= entries[-1].offset:
            raise AerosolArchiveError(
                f"idx line {line_number}: offset {offset} does not follow "
                f"offset {entries[-1].offset}"
            )
        entries.append(
            IdxEntry(
                number, offset, parts[2][2:], parts[3], parts[4], parts[5], ":".join(parts[6:])
            )
        )
    return entries


@dataclass(frozen=True)
class AodSpec:
    """One optical-depth field: its idx label and what the message itself must carry.

    `aerosol_type` is GRIB2 code table 4.233 (62000 total aerosol, 62001 dust) and
    `wavelength_nm` the band the message is for; both are checked on the decoded message, which
    catches an idx that has gone out of step with its file.
    """

    name: str
    label: str
    aerosol_type: int
    wavelength_nm: tuple[float, float]


_AOD_550 = "aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07"
TOTAL_AOD_550: Final = AodSpec(
    "TOTAL_AOD_550", f"aerosol=Total aerosol:{_AOD_550}", 62000, (545.0, 565.0)
)
DUST_AOD_550: Final = AodSpec("DUST_AOD_550", f"aerosol=Dust dry:{_AOD_550}", 62001, (545.0, 565.0))
AOD_SPECS: Final[tuple[AodSpec, ...]] = (TOTAL_AOD_550, DUST_AOD_550)
"""Aerosol optical depth at 550 nm, for the total column and for dust, in the order read."""


def select_ranges(entries: Sequence[IdxEntry], specs: Sequence[AodSpec]) -> dict[str, ByteRange]:
    """The byte range of each field's message, keyed by field name, in `specs` order.

    A message runs from its own offset to one byte before the next message's offset; the last
    message of a file has an open end. Each field must match exactly one idx line.
    """
    ranges: dict[str, ByteRange] = {}
    for spec in specs:
        hits = [
            position
            for position, entry in enumerate(entries)
            if entry.variable == "AOTK"
            and entry.level == "entire atmosphere"
            and entry.label == spec.label
        ]
        if len(hits) != 1:
            raise AerosolArchiveError(
                f"{spec.name}: expected exactly one idx message for AOTK with {spec.label!r}, "
                f"found {len(hits)}"
            )
        position = hits[0]
        following = entries[position + 1].offset - 1 if position + 1 < len(entries) else None
        ranges[spec.name] = ByteRange(entries[position].offset, following)
    return ranges


# --- downloading and caching ----------------------------------------------------------------------


def _cache_path(cache_dir: Path, url: str, byte_range: ByteRange) -> Path:
    key = f"{url}\n{byte_range.header()}"
    return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.grib2"


def _is_whole_message(data: bytes, byte_range: ByteRange) -> bool:
    if byte_range.length is not None and len(data) != byte_range.length:
        return False
    return data[:4] == b"GRIB" and data[-4:] == b"7777"


def _read_range(url: str, byte_range: ByteRange, cache_dir: Path, transport: Transport) -> bytes:
    path = _cache_path(cache_dir, url, byte_range)
    if path.is_file():
        cached = path.read_bytes()
        if _is_whole_message(cached, byte_range):
            return cached
        path.unlink()  # a download that did not finish; fetch it again
    data = transport.get(url, byte_range)
    if not _is_whole_message(data, byte_range):
        raise AerosolArchiveError(
            f"{url} {byte_range.header()}: the bytes are not one whole GRIB message "
            "(is the idx out of step with its file?)"
        )
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    partial.write_bytes(data)
    os.replace(partial, path)
    return data


# --- decoding and sampling ------------------------------------------------------------------------


def check_optical_depth(name: str, value: float) -> float:
    """The optical depth if it is a physical one; a tiny negative (packing noise) reads as 0."""
    if not math.isfinite(value) or value < -_PACKING_NOISE:
        raise AerosolArchiveError(f"{name}: {value!r} is not an aerosol optical depth")
    return max(value, 0.0)


def cell_value(name: str, values: np.ma.MaskedArray, cell: tuple[int, int]) -> float:
    """The value at `cell`; a cell the file leaves empty is an error, never a zero."""
    if np.ma.getmaskarray(values)[cell]:
        raise AerosolArchiveError(f"{name}: the file has no value in cell {cell}")
    return float(values.data[cell])


def _check_message(spec: AodSpec, message: grib.LatLonMessage, plan: RunPlan) -> None:
    if (message.discipline, message.category, message.number) != _EXPECTED_PARAMETER:
        raise AerosolArchiveError(
            f"{spec.name}: the message is GRIB parameter "
            f"{(message.discipline, message.category, message.number)}, expected "
            f"{_EXPECTED_PARAMETER} (aerosol optical thickness)"
        )
    if message.surface_type != _WHOLE_COLUMN:
        raise AerosolArchiveError(f"{spec.name}: the message is not for the entire atmosphere")
    if message.aerosol_type != spec.aerosol_type:
        raise AerosolArchiveError(
            f"{spec.name}: the idx promised aerosol type {spec.aerosol_type} but the message "
            f"is aerosol type {message.aerosol_type}"
        )
    band = message.wavelength_nm
    if band is None or any(
        abs(found - wanted) > _WAVELENGTH_TOLERANCE_NM
        for found, wanted in zip(band, spec.wavelength_nm, strict=True)
    ):
        raise AerosolArchiveError(
            f"{spec.name}: the message's wavelength band is {band}, expected {spec.wavelength_nm}"
        )
    if (message.reference_date, message.reference_hour) != (plan.cycle_date, plan.cycle_hour):
        raise AerosolArchiveError(
            f"{spec.name}: the message is from cycle {message.reference_date} hour "
            f"{message.reference_hour}, not the requested cycle {plan.cycle_date} hour "
            f"{plan.cycle_hour}"
        )
    if message.step_hours != plan.forecast_hour:
        raise AerosolArchiveError(
            f"{spec.name}: the message is forecast hour {message.step_hours}, "
            f"not the requested forecast hour {plan.forecast_hour}"
        )


def _nearest_cell(
    message: grib.LatLonMessage, lat: float, lon: float
) -> tuple[tuple[int, int], float]:
    """(row, column) of the cell whose centre is nearest, and its distance in km.

    A point farther from its nearest centre than half a cell diagonal is off the grid.
    """
    row = int(np.argmin(np.abs(message.lats - lat)))
    column = int(np.argmin(np.abs((message.lons - lon + 180.0) % 360.0 - 180.0)))
    cell_lat, cell_lon = float(message.lats[row]), float(message.lons[column])
    p_lat, c_lat = math.radians(lat), math.radians(cell_lat)
    haversine = (
        math.sin((c_lat - p_lat) / 2.0) ** 2
        + math.cos(p_lat) * math.cos(c_lat) * math.sin(math.radians(cell_lon - lon) / 2.0) ** 2
    )
    distance_km = 2.0 * geometry.EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))
    d_lat = abs(float(message.lats[1] - message.lats[0])) if message.lats.size > 1 else 0.25
    d_lon = abs(float(message.lons[1] - message.lons[0])) if message.lons.size > 1 else 0.25
    half_diagonal_km = (
        0.5
        * math.hypot(
            math.radians(d_lat) * geometry.EARTH_RADIUS_KM,
            math.radians(d_lon) * geometry.EARTH_RADIUS_KM * math.cos(p_lat),
        )
        * _CELL_TOLERANCE
    )
    if distance_km > half_diagonal_km:
        raise AerosolArchiveError(
            f"the point is outside the aerosol grid: the nearest cell is {distance_km:.1f} km away"
        )
    return (row, column), distance_km


@dataclass(frozen=True)
class AerosolSample:
    """What the archive says at the reference point for one evening.

    `digests` maps each field name to the SHA-256 of the raw GRIB message it was decoded from.
    """

    cycle_date: dt.date
    cycle_hour: int
    forecast_hour: int
    total_aod_550: float
    dust_aod_550: float
    cell: tuple[int, int]
    cell_distance_km: float
    digests: dict[str, str]

    def column(self) -> AerosolColumn:
        """The pair as the reduction takes it."""
        return AerosolColumn(
            total_aod_550=self.total_aod_550,
            dust_aod_550=self.dust_aod_550,
            source=SOURCE_LABEL,
        )


def fetch_aerosol(
    evening: dt.date,
    lat: float,
    lon: float,
    cache_dir: Path | str,
    *,
    transport: Transport | None = None,
    run: RunPlan | None = None,
) -> AerosolSample:
    """Total and dust aerosol optical depth at 550 nm at (`lat`, `lon`) for the evening.

    Downloads (by byte range, cached) and decodes the two messages of the run `aerosol_run`
    chooses (or `run`), checks each is the aerosol, wavelength and run it should be, and reads
    the nearest cell. A `FetchError` (the network, or a 404 for a run not in the archive) is
    raised as it is; anything inconsistent is an `AerosolArchiveError`.
    """
    transport = transport or UrllibTransport()
    plan = run or aerosol_run(evening, lat, lon)
    cache = Path(cache_dir)
    url = chem_url(plan.cycle_date, plan.cycle_hour, plan.forecast_hour)
    entries = parse_idx(
        transport.get(idx_url(plan.cycle_date, plan.cycle_hour, plan.forecast_hour)).decode()
    )
    if not entries:
        raise AerosolArchiveError(f"{url}.idx: the idx lists no messages")
    ranges = select_ranges(entries, AOD_SPECS)
    cache.mkdir(parents=True, exist_ok=True)

    values: dict[str, float] = {}
    digests: dict[str, str] = {}
    located: tuple[tuple[int, int], float] | None = None
    first_axes: tuple[np.ndarray, np.ndarray] | None = None
    for spec in AOD_SPECS:
        data = _read_range(url, ranges[spec.name], cache, transport)
        try:
            message = grib.decode_latlon_message(data)
        except grib.GribError as error:
            raise AerosolArchiveError(f"{spec.name}: {error}") from error
        _check_message(spec, message, plan)
        if first_axes is None:
            first_axes = (message.lats, message.lons)
            located = _nearest_cell(message, lat, lon)
        elif not (
            np.array_equal(message.lats, first_axes[0])
            and np.array_equal(message.lons, first_axes[1])
        ):
            raise AerosolArchiveError(f"{spec.name}: the message is on a different grid")
        assert located is not None
        values[spec.name] = check_optical_depth(
            spec.name, cell_value(spec.name, message.values, located[0])
        )
        digests[spec.name] = hashlib.sha256(data).hexdigest()
    assert located is not None
    sample = AerosolSample(
        cycle_date=plan.cycle_date,
        cycle_hour=plan.cycle_hour,
        forecast_hour=plan.forecast_hour,
        total_aod_550=values[TOTAL_AOD_550.name],
        dust_aod_550=values[DUST_AOD_550.name],
        cell=located[0],
        cell_distance_km=located[1],
        digests=digests,
    )
    try:
        sample.column()  # the pair must be consistent (dust within the total)
    except ValueError as error:
        raise AerosolArchiveError(str(error)) from error
    return sample
