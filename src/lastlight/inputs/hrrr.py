"""HRRR ingest along the sunlight's path (`docs/design.md` section 4, `inputs.hrrr`).

Tonight's twilight colour depends on cloud, smoke and water vapour along the path the setting
sun's light travels, westward for a few hundred kilometres. HRRR, NOAA's hourly 3 km model, is
free on a public AWS bucket with no account. A surface file there is about 140 MB, and each
`.idx` beside it lists every message with its byte offset, so this module reads the `.idx`,
picks the few messages it wants (`DEFAULT_FIELDS`) and downloads only those byte ranges, caching
each range by URL plus range.

The flow is `latest_run` or `plan_run` (which run), `fetch_fields` (download, cache, decode) and
`sample_transect` (the value of every field at every point of a transect, nearest cell).
Every inconsistency is an error rather than a guess: an idx that does not match its file, a
message that is not the parameter, run or hour it was asked for, fields on different grids, a
point off the grid, a non-finite value. A cell the model genuinely leaves empty (cloud base where
there is no cloud) is `None`, never NaN.

The GRIB library sits behind `lastlight.inputs.grib`; HTTP behind `lastlight.inputs.http`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Final

import numpy as np

from lastlight import geometry
from lastlight.inputs.grib import (
    DecodedMessage,
    GribError,
    GridSpec,
    decode_message,
    grid_coordinates,
)
from lastlight.inputs.http import ByteRange, Transport, UrllibTransport

BUCKET_URL: Final = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

SAMPLING_METHOD: Final = "nearest-cell"
"""How `sample_transect` turns a point into a value: the grid cell whose centre is closest."""

TRANSECT_DISTANCES_KM: Final[tuple[int, ...]] = tuple(range(0, 401, 25))
"""Distances along the sunset azimuth, from the reference point, that the CLI samples."""

LATEST_RUN_LOOKBACK_HOURS: Final = 6
_COMPLETE_AFTER: Final = dt.timedelta(hours=2)
_COMPLETENESS_FORECAST_HOUR: Final = 18  # the last file every cycle writes; see latest_run
_CELL_TOLERANCE: Final = 1.01  # slack on the half-diagonal, for float noise at a cell corner


class HrrrError(RuntimeError):
    """HRRR data that is missing, inconsistent or not what was asked for."""


class OutsideGridError(HrrrError):
    """A requested point lies beyond the edge of the HRRR grid."""


# --- the fields -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One HRRR field: how its idx line is labelled and which GRIB parameter it must decode to.

    `variable` and `level` are the wgrib2 labels in the `.idx` (both must match exactly, because
    some variables repeat across levels: total cloud cover is listed for the boundary layer and
    for the entire atmosphere). `grib_id` is the parameter the message itself must carry, as
    (discipline, category, number, type of first fixed surface); it catches an idx that has gone
    out of step with its file.
    """

    name: str
    variable: str
    level: str
    units: str
    grib_id: tuple[int, int, int, int]


_WHOLE_COLUMN = "entire atmosphere (considered as a single layer)"

DEFAULT_FIELDS: Final[tuple[FieldSpec, ...]] = (
    FieldSpec("LCDC", "LCDC", "low cloud layer", "%", (0, 6, 3, 214)),
    FieldSpec("MCDC", "MCDC", "middle cloud layer", "%", (0, 6, 4, 224)),
    FieldSpec("HCDC", "HCDC", "high cloud layer", "%", (0, 6, 5, 234)),
    FieldSpec("TCDC", "TCDC", "entire atmosphere", "%", (0, 6, 1, 10)),
    FieldSpec("AOTK", "AOTK", _WHOLE_COLUMN, "1", (0, 20, 102, 200)),
    FieldSpec("PWAT", "PWAT", _WHOLE_COLUMN, "kg m-2", (0, 1, 3, 200)),
    FieldSpec("VIS", "VIS", "surface", "m", (0, 19, 0, 1)),
    FieldSpec("HGT_CLOUD_BASE", "HGT", "cloud base", "gpm", (0, 3, 5, 2)),
    FieldSpec("HGT_CLOUD_TOP", "HGT", "cloud top", "gpm", (0, 3, 5, 3)),
)
"""Low, middle and high cloud cover, total cloud cover, aerosol optical thickness, precipitable
water, visibility, and cloud base and top height. Units are as the file labels them; `gpm` is
geopotential metres (the file does not say what surface cloud height is measured from)."""


def field_spec(name: str) -> FieldSpec:
    """The default field called `name`."""
    for spec in DEFAULT_FIELDS:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown default field {name!r}; known: {[s.name for s in DEFAULT_FIELDS]}")


# --- URLs and runs --------------------------------------------------------------------------------


def max_forecast_hour(cycle_hour: int) -> int:
    """HRRR runs to forecast hour 48 at 00, 06, 12 and 18 and to hour 18 at every other hour."""
    return 48 if cycle_hour % 6 == 0 else 18


def grib_url(cycle_date: dt.date, cycle_hour: int, forecast_hour: int) -> str:
    """The surface GRIB2 file of one run and forecast hour in the public bucket."""
    if not 0 <= cycle_hour <= 23:
        raise ValueError(f"cycle hour must be 0..23, got {cycle_hour}")
    limit = max_forecast_hour(cycle_hour)
    if not 0 <= forecast_hour <= limit:
        raise ValueError(
            f"forecast hour must be 0..{limit} for cycle hour {cycle_hour}, got {forecast_hour}"
        )
    return (
        f"{BUCKET_URL}/hrrr.{cycle_date:%Y%m%d}/conus/"
        f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    )


def idx_url(cycle_date: dt.date, cycle_hour: int, forecast_hour: int) -> str:
    return grib_url(cycle_date, cycle_hour, forecast_hour) + ".idx"


def _utc(when: dt.datetime) -> dt.datetime:
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("the time must be timezone-aware; a naive datetime is ambiguous")
    return when.astimezone(dt.UTC)


def latest_run(now_utc: dt.datetime, *, transport: Transport | None = None) -> tuple[dt.date, int]:
    """(cycle date, cycle hour) of the newest run expected to be complete.

    A run is expected to be complete two hours after its cycle hour. The newest such cycle is
    checked and, if its files are not there yet, earlier ones in turn, up to
    `LATEST_RUN_LOOKBACK_HOURS`. "There" means the `.idx` of the run's forecast-hour-18 file
    exists: that is the last file every cycle writes (about an hour and a half after the
    cycle), so a run whose hour-18 index exists has all its earlier hours too.
    """
    transport = transport or UrllibTransport()
    newest = (_utc(now_utc) - _COMPLETE_AFTER).replace(minute=0, second=0, microsecond=0)
    for back in range(LATEST_RUN_LOOKBACK_HOURS):
        cycle = newest - dt.timedelta(hours=back)
        if transport.exists(idx_url(cycle.date(), cycle.hour, _COMPLETENESS_FORECAST_HOUR)):
            return cycle.date(), cycle.hour
    raise HrrrError(
        f"no complete HRRR run found among the {LATEST_RUN_LOOKBACK_HOURS} newest cycles "
        "(more than two hours old)"
    )


@dataclass(frozen=True)
class RunPlan:
    cycle_date: dt.date
    cycle_hour: int
    forecast_hour: int


def plan_run(
    evening: dt.date,
    lat: float,
    lon: float,
    *,
    cycle_hour: int | None,
    forecast_hour: int | None,
    now_utc: dt.datetime,
    transport: Transport | None = None,
) -> RunPlan:
    """Which run and forecast hour to read for the evening of `evening` at (`lat`, `lon`).

    The aim is the whole UTC hour nearest the instant the sun reaches geometric elevation 0
    (the "target hour"; the instant itself is internal and never leaves this function).

    * `cycle_hour` given: the latest cycle with that hour at or before the target hour.
    * only `forecast_hour` given: the cycle that makes that forecast valid at the target hour.
    * neither: the analysis (hour 0) of the target-hour cycle once it is two hours old, else the
      newest complete run (`latest_run`) and the forecast hour that reaches the target.
    * the forecast hour, when not given, is the whole hours from the cycle to the target hour.

    Raises HrrrError if the forecast hour is more than the cycle's run provides.
    """
    now = _utc(now_utc)
    instant = geometry.elevation_crossings(evening, lat, lon)[0.0]
    target = (instant + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    if cycle_hour is not None:
        if not 0 <= cycle_hour <= 23:
            raise ValueError(f"cycle hour must be 0..23, got {cycle_hour}")
        cycle = target.replace(hour=cycle_hour)
        if cycle > target:
            cycle -= dt.timedelta(days=1)
    elif forecast_hour is not None:
        cycle = target - dt.timedelta(hours=forecast_hour)
    elif target + _COMPLETE_AFTER <= now:
        cycle = target
    else:
        cycle_date, newest_hour = latest_run(now, transport=transport)
        cycle = dt.datetime.combine(cycle_date, dt.time(newest_hour), tzinfo=dt.UTC)
    if forecast_hour is None:
        forecast_hour = int((target - cycle).total_seconds() // 3600)
    limit = max_forecast_hour(cycle.hour)
    if not 0 <= forecast_hour <= limit:
        raise HrrrError(
            f"forecast hour {forecast_hour} is outside the 0..{limit} that cycle hour "
            f"{cycle.hour} provides"
        )
    return RunPlan(cycle.date(), cycle.hour, forecast_hour)


# --- the .idx -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdxEntry:
    """One line of a wgrib2 `.idx`: `number:offset:d=YYYYMMDDHH:variable:level:forecast:`."""

    number: int
    offset: int
    date_code: str
    variable: str
    level: str
    forecast: str


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse an `.idx`. Offsets must strictly increase: a message ends where the next begins."""
    entries: list[IdxEntry] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6 or not parts[2].startswith("d="):
            raise HrrrError(f"malformed idx line {line_number}: {raw!r}")
        try:
            number, offset = int(parts[0]), int(parts[1])
        except ValueError:
            raise HrrrError(f"malformed idx line {line_number}: {raw!r}") from None
        if entries and offset <= entries[-1].offset:
            raise HrrrError(
                f"idx line {line_number}: offset {offset} does not follow "
                f"offset {entries[-1].offset}"
            )
        entries.append(IdxEntry(number, offset, parts[2][2:], parts[3], parts[4], parts[5]))
    return entries


def select_ranges(entries: Sequence[IdxEntry], fields: Sequence[FieldSpec]) -> dict[str, ByteRange]:
    """The byte range of each field's message, keyed by field name, in `fields` order.

    A message runs from its own offset to one byte before the next message's offset; the last
    message of a file has an open end. Each field must match exactly one idx line.
    """
    ranges: dict[str, ByteRange] = {}
    for spec in fields:
        hits = [
            position
            for position, entry in enumerate(entries)
            if entry.variable == spec.variable and entry.level == spec.level
        ]
        if len(hits) != 1:
            raise HrrrError(
                f"{spec.name}: expected exactly one idx message for "
                f"{spec.variable}:{spec.level}, found {len(hits)}"
            )
        position = hits[0]
        following = entries[position + 1].offset - 1 if position + 1 < len(entries) else None
        ranges[spec.name] = ByteRange(entries[position].offset, following)
    return ranges


def fetch_idx(
    cycle_date: dt.date,
    cycle_hour: int,
    forecast_hour: int,
    *,
    transport: Transport | None = None,
) -> list[IdxEntry]:
    """Download and parse the `.idx` of one file."""
    transport = transport or UrllibTransport()
    url = idx_url(cycle_date, cycle_hour, forecast_hour)
    entries = parse_idx(transport.get(url).decode("utf-8"))
    if not entries:
        raise HrrrError(f"{url}: the idx lists no messages")
    return entries


# --- downloading and caching ----------------------------------------------------------------------


def _checked_specs(fields: Sequence[FieldSpec]) -> list[FieldSpec]:
    specs = list(fields)
    if not specs:
        raise ValueError("at least one field is required")
    names = [spec.name for spec in specs]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"field {name} is listed twice")
    return specs


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
        raise HrrrError(
            f"{url} {byte_range.header()}: the bytes are not one whole GRIB message "
            "(is the idx out of step with its file?)"
        )
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    partial.write_bytes(data)
    os.replace(partial, path)
    return data


def fetch_messages(
    cycle_date: dt.date,
    cycle_hour: int,
    forecast_hour: int,
    fields: Sequence[FieldSpec],
    cache_dir: Path | str,
    *,
    transport: Transport | None = None,
) -> dict[str, bytes]:
    """The raw GRIB2 bytes of each field's message, keyed by field name.

    Reads the `.idx`, then downloads one byte range per field (never the whole file), each
    cached in `cache_dir` by URL plus range. A cached range is used without any request; the
    small `.idx` is always read again, because it names the ranges.
    """
    specs = _checked_specs(fields)
    transport = transport or UrllibTransport()
    cache = Path(cache_dir)
    ranges = select_ranges(
        fetch_idx(cycle_date, cycle_hour, forecast_hour, transport=transport), specs
    )
    url = grib_url(cycle_date, cycle_hour, forecast_hour)
    cache.mkdir(parents=True, exist_ok=True)
    return {spec.name: _read_range(url, ranges[spec.name], cache, transport) for spec in specs}


# --- decoding -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Where the cells are: the grid definition and the latitude and longitude of every cell."""

    spec: GridSpec
    lats: np.ndarray  # (nj, ni), degrees north
    lons: np.ndarray  # (nj, ni), degrees east in [-180, 180]

    @property
    def cell_radius_km(self) -> float:
        """The farthest a point inside the grid can be from its nearest cell centre."""
        return 0.5 * math.hypot(self.spec.dx_m, self.spec.dy_m) / 1000.0 * _CELL_TOLERANCE


@dataclass(frozen=True)
class FieldData:
    spec: FieldSpec
    values: np.ma.MaskedArray  # (nj, ni), masked where the model has no value


@dataclass(frozen=True)
class HrrrFields:
    """Decoded fields of one run and forecast hour, all on one grid.

    `digests` maps each field name to the SHA-256 (lowercase hex) of the raw GRIB message it was
    decoded from, so a forecast can record exactly which bytes it read.
    """

    cycle_date: dt.date
    cycle_hour: int
    forecast_hour: int
    grid: Grid
    fields: dict[str, FieldData]
    digests: dict[str, str] = dataclass_field(default_factory=dict)


def _check_message(
    spec: FieldSpec,
    message: DecodedMessage,
    cycle_date: dt.date,
    cycle_hour: int,
    forecast_hour: int,
) -> None:
    observed = (message.discipline, message.category, message.number, message.surface_type)
    if observed != spec.grib_id:
        raise HrrrError(
            f"{spec.name}: the idx promised {spec.variable}:{spec.level} but the message is "
            f"GRIB parameter {observed}, expected {spec.grib_id}"
        )
    if (message.reference_date, message.reference_hour) != (cycle_date, cycle_hour):
        raise HrrrError(
            f"{spec.name}: the message is from cycle {message.reference_date} hour "
            f"{message.reference_hour}, not the requested cycle {cycle_date} hour {cycle_hour}"
        )
    if message.step_hours != forecast_hour:
        raise HrrrError(
            f"{spec.name}: the message is forecast hour {message.step_hours}, "
            f"not the requested forecast hour {forecast_hour}"
        )
    if not np.isfinite(message.values.compressed()).all():
        raise HrrrError(f"{spec.name}: the message holds non-finite values")


def fetch_fields(
    cycle_date: dt.date,
    cycle_hour: int,
    forecast_hour: int,
    fields: Sequence[FieldSpec],
    cache_dir: Path | str,
    *,
    transport: Transport | None = None,
) -> HrrrFields:
    """Download (by byte range, cached) and decode `fields` of one run and forecast hour.

    Each message must be the parameter, run and forecast hour it was asked for, and all must
    share one grid; otherwise HrrrError.
    """
    specs = _checked_specs(fields)
    messages = fetch_messages(
        cycle_date, cycle_hour, forecast_hour, specs, cache_dir, transport=transport
    )
    decoded: dict[str, FieldData] = {}
    grid_spec: GridSpec | None = None
    for spec in specs:
        try:
            message = decode_message(messages[spec.name])
        except GribError as error:
            raise HrrrError(f"{spec.name}: {error}") from error
        _check_message(spec, message, cycle_date, cycle_hour, forecast_hour)
        if grid_spec is None:
            grid_spec = message.grid
        elif message.grid != grid_spec:
            raise HrrrError(f"{spec.name}: the message is on a different grid from {specs[0].name}")
        decoded[spec.name] = FieldData(spec, message.values)
    assert grid_spec is not None  # specs is not empty
    try:
        lats, lons = grid_coordinates(messages[specs[0].name])
    except GribError as error:
        raise HrrrError(f"cannot read the grid: {error}") from error
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in messages.items()}
    return HrrrFields(
        cycle_date, cycle_hour, forecast_hour, Grid(grid_spec, lats, lons), decoded, digests
    )


# --- sampling along the transect ------------------------------------------------------------------


@dataclass(frozen=True)
class TransectSample:
    """One field's value at one transect point.

    `value` is None where the model leaves the cell empty (cloud base and top where there is no
    cloud); it is never NaN. `cell` is the (row, column) of the cell used and `cell_distance_km`
    how far its centre is from the point; `method` says how the cell was chosen.
    """

    point_index: int
    field: str
    lat: float
    lon: float
    value: float | None
    units: str
    method: str
    cell: tuple[int, int]
    cell_distance_km: float


def _check_point(index: int, lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise ValueError(
            f"point {index}: latitude must be a finite number in [-90, 90], got {lat!r}"
        )
    if not (math.isfinite(lon) and -180.0 <= lon <= 180.0):
        raise ValueError(
            f"point {index}: longitude must be a finite number of degrees east in "
            f"[-180, 180], got {lon!r}"
        )


def sample_transect(
    fields: HrrrFields, points: Sequence[tuple[float, float]]
) -> list[TransectSample]:
    """The value of every field at every (lat, lon) point: one row per (point, field).

    Rows come point by point, in the order the fields were fetched. The value is that of the
    grid cell whose centre is nearest the point (`SAMPLING_METHOD`); no interpolation. A point
    farther from its nearest cell than half a cell diagonal is off the grid, and the whole call
    raises OutsideGridError instead of returning a partial table. Longitudes are degrees east
    in [-180, 180] (Phoenix is negative).
    """
    grid = fields.grid
    lat_rad = np.radians(grid.lats)
    lon_rad = np.radians(grid.lons)
    cos_lat = np.cos(lat_rad)
    located: list[tuple[tuple[int, int], float]] = []
    for index, (lat, lon) in enumerate(points):
        _check_point(index, lat, lon)
        p_lat, p_lon = math.radians(lat), math.radians(lon)
        haversine = (
            np.sin((lat_rad - p_lat) / 2.0) ** 2
            + math.cos(p_lat) * cos_lat * np.sin((lon_rad - p_lon) / 2.0) ** 2
        )
        row, column = np.unravel_index(np.argmin(haversine), haversine.shape)
        distance_km = 2.0 * geometry.EARTH_RADIUS_KM * math.asin(math.sqrt(haversine[row, column]))
        if distance_km > grid.cell_radius_km:
            raise OutsideGridError(
                f"point {index} is outside the HRRR grid: the nearest cell is "
                f"{distance_km:.1f} km away"
            )
        located.append(((int(row), int(column)), distance_km))

    samples: list[TransectSample] = []
    for index, (lat, lon) in enumerate(points):
        cell, distance_km = located[index]
        for name, data in fields.fields.items():
            samples.append(
                TransectSample(
                    point_index=index,
                    field=name,
                    lat=float(lat),
                    lon=float(lon),
                    value=_cell_value(name, data.values, cell),
                    units=data.spec.units,
                    method=SAMPLING_METHOD,
                    cell=cell,
                    cell_distance_km=distance_km,
                )
            )
    return samples


def _cell_value(name: str, values: np.ma.MaskedArray, cell: tuple[int, int]) -> float | None:
    if np.ma.getmaskarray(values)[cell]:
        return None
    value = float(values.data[cell])
    if not math.isfinite(value):
        raise HrrrError(f"{name}: non-finite value in cell {cell}")
    return value


# --- the table ------------------------------------------------------------------------------------

_VALUE_FORMATS: Final = {
    "%": "{:.0f}",
    "1": "{:.3f}",
    "kg m-2": "{:.1f}",
    "m": "{:.0f}",
    "gpm": "{:.0f}",
}


def _format_value(value: float | None, units: str) -> str:
    return "-" if value is None else _VALUE_FORMATS.get(units, "{:g}").format(value)


def format_transect_table(
    samples: Sequence[TransectSample],
    specs: Sequence[FieldSpec],
    distances_km: Sequence[float],
) -> str:
    """A text table: one row per distance, one column per field, units on the second line.

    `samples` is the result of `sample_transect` for points at `distances_km`, in order. A
    missing value prints as a dash.
    """
    if len(samples) != len(distances_km) * len(specs):
        raise ValueError(
            f"{len(samples)} samples do not fit {len(distances_km)} distances "
            f"and {len(specs)} fields"
        )
    by_key = {(sample.point_index, sample.field): sample for sample in samples}
    header = ["km", *(spec.name for spec in specs)]
    units = ["", *(spec.units for spec in specs)]
    body = [
        [f"{distance:g}"]
        + [_format_value(by_key[(index, spec.name)].value, spec.units) for spec in specs]
        for index, distance in enumerate(distances_km)
    ]
    widths = [
        max(len(cell) for cell in column) for column in zip(header, units, *body, strict=True)
    ]
    lines = [header, units, *body]
    return "\n".join(
        "  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True))
        for line in lines
    )
