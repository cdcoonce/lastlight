"""Renderer v1: v0's single scattering plus multiple scattering read from a Bruneton table.

v1's XYZ is v0's exact single-scattering XYZ (`lastlight.render.transfer`, unchanged) plus the
XYZ of light scattered two or more times, including light the ground reflected once or more.
That extra light comes from a table that `tools/make_bruneton_table.py` precomputes with the
numpy port of Bruneton's 2017 method (`lastlight.render.bruneton`) and that is committed next to
this module. XYZ is linear in radiance, so the sum is exact. The table holds no spectra, so v1 has
no `sky_spectrum`; callers read XYZ through `lastlight.render.curve.sky_xyz(..., engine="v1")`.

Only `aod_550` and `ozone_du` vary from evening to evening; every other `Atmosphere` field is
baked into the table at its default, with a ground albedo of 0.2. v1 refuses, with a
`ValueError` naming the field or the bound, an atmosphere whose fixed fields differ from the baked
values and any point outside the table's domain: `aod_550` 0 to 1.0, `ozone_du` 250 to 330, sun
elevation -12 to +12, view elevation 0 to 90, and any relative azimuth (the sky is symmetric about
the sun's vertical, so it is folded onto 0 to 180). It never clamps.

The table is interpolated multilinearly in the log of each channel over its five axes, so the
exponential darkening of twilight with sun elevation is followed between nodes. The file is
untrusted until checked: its SHA-256 must be the pinned `TABLE_SHA256`, and its header, shape,
dtype, axes and values are validated before use.

Container format (little-endian): 8 bytes of magic `LLMSXYZ\\0`; the header length as a uint32;
the header, UTF-8 JSON with `format`, `version`, `dtype` ("<f4"), `shape`, `axis_order`, `axes`,
`ground_albedo`, `baked_atmosphere`, `orders`, `order_check` and `resolution`; then the values as
float32 in C order with that shape (aod_550, ozone_du, sun elevation, view elevation, relative
azimuth, XYZ). No timestamps, so the same inputs give the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from lastlight.render.atmosphere import Atmosphere
from lastlight.render.bruneton import GROUND_ALBEDO

TABLE_NAME = "bruneton_ms_xyz.bin"
TABLE_SHA256 = "c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368"
"""SHA-256 of the committed table; the loader refuses any other file."""

FORMAT = "lastlight-multiple-scattering-xyz"
VERSION = 1
MAGIC = b"LLMSXYZ\x00"
DTYPE = "<f4"

AXES = ("aod_550", "ozone_du", "sun_elevation_deg", "view_elevation_deg", "rel_azimuth_deg")
DOMAIN: Mapping[str, tuple[float, float]] = {
    "aod_550": (0.0, 1.0),
    "ozone_du": (250.0, 330.0),
    "sun_elevation_deg": (-12.0, 12.0),
    "view_elevation_deg": (0.0, 90.0),
    "rel_azimuth_deg": (0.0, 180.0),
}
"""Inclusive bounds of each axis: the table's first and last nodes."""

BAKED_FIELDS = (
    "angstrom",
    "aerosol_scale_height_km",
    "pressure_hpa",
    "asymmetry",
    "single_scatter_albedo",
)
"""The `Atmosphere` fields the table fixes at their defaults."""

_MAX_HEADER = 1 << 20


class TableError(ValueError):
    """The multiple-scattering table file is not one v1 can trust."""


def _defaults() -> dict[str, float]:
    return {f.name: float(f.default) for f in fields(Atmosphere) if f.name in BAKED_FIELDS}


@dataclass(frozen=True, eq=False)
class MultipleScatteringTable:
    """A validated table: its axes, its float32 XYZ values and its header."""

    axes: Mapping[str, np.ndarray]
    values: np.ndarray
    metadata: Mapping[str, Any]
    _logs: np.ndarray

    def xyz(
        self,
        aod_550: float,
        ozone_du: float,
        sun_elevation: float,
        view_elevations: Sequence[float],
        rel_azimuth_deg: float,
    ) -> np.ndarray:
        """Multiple-scattering XYZ at each view elevation, shape (views, 3).

        The point must lie in the domain (see `multiple_scattering_xyz`, which checks it); the
        relative azimuth must already be folded onto 0 to 180.
        """
        weights = [
            _bracket(self.axes[name], value)
            for name, value in zip(
                ("aod_550", "ozone_du", "sun_elevation_deg"),
                (aod_550, ozone_du, sun_elevation),
                strict=True,
            )
        ]
        (ia, wa), (io, wo), (isun, ws) = weights
        iz, wz = _bracket(self.axes["rel_azimuth_deg"], rel_azimuth_deg)
        block = self._logs[ia : ia + 2, io : io + 2, isun : isun + 2, :, iz : iz + 2, :]
        logs = np.einsum("a,o,s,aosvzc,z->vc", wa, wo, ws, block, wz)
        views = np.asarray(view_elevations, dtype=float)
        nodes = self.axes["view_elevation_deg"]
        lower = np.clip(np.searchsorted(nodes, views, side="right") - 1, 0, nodes.size - 2)
        frac = (views - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
        return np.exp(logs[lower] * (1.0 - frac)[:, None] + logs[lower + 1] * frac[:, None])


def _bracket(nodes: np.ndarray, value: float) -> tuple[int, np.ndarray]:
    """Lower node index and the two linear weights for `value` inside `nodes`."""
    lower = int(np.clip(np.searchsorted(nodes, value, side="right") - 1, 0, nodes.size - 2))
    frac = (value - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
    return lower, np.array([1.0 - frac, frac])


# --- the container ---------------------------------------------------------------------------


def encode_table(
    axes: Mapping[str, Sequence[float]], values: np.ndarray, metadata: Mapping[str, Any]
) -> bytes:
    """The container bytes for a table: deterministic for the same arguments."""
    array = np.ascontiguousarray(values, dtype=DTYPE)
    header = dict(metadata)
    header.update(
        {
            "format": FORMAT,
            "version": VERSION,
            "dtype": DTYPE,
            "shape": list(array.shape),
            "axis_order": [*AXES, "xyz"],
            "axes": {name: [float(x) for x in axes[name]] for name in AXES},
        }
    )
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    encoded = text.encode("ascii")
    return MAGIC + struct.pack("<I", len(encoded)) + encoded + array.tobytes(order="C")


def decode_table(data: bytes) -> MultipleScatteringTable:
    """Parse and validate container bytes (no digest check: see `load_table`)."""
    if len(data) < 12 or data[:8] != MAGIC:
        raise TableError("not a multiple-scattering table: the magic bytes are wrong")
    (length,) = struct.unpack("<I", data[8:12])
    if length > _MAX_HEADER or 12 + length > len(data):
        raise TableError(f"the header length {length} does not fit the file")
    try:
        header = json.loads(data[12 : 12 + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TableError(f"the header is not JSON: {error}") from error
    if not isinstance(header, dict):
        raise TableError("the header is not a JSON object")
    _check_header(header)
    shape = tuple(header["shape"])
    body = data[12 + length :]
    expected = 4 * math.prod(shape)
    if len(body) != expected:
        raise TableError(f"the values take {len(body)} bytes, the shape needs {expected}")
    values = np.frombuffer(body, dtype=DTYPE).reshape(shape)
    if not np.isfinite(values).all():
        raise TableError("the table holds a value that is not finite")
    if not (values > 0.0).all():
        raise TableError("the table holds a value that is not positive")
    axes = {name: np.array(header["axes"][name], dtype=float) for name in AXES}
    for array in axes.values():
        array.flags.writeable = False
    logs = np.log(values.astype(float))
    logs.flags.writeable = False
    return MultipleScatteringTable(axes=axes, values=values, metadata=header, _logs=logs)


def _check_header(header: Mapping[str, Any]) -> None:
    if header.get("format") != FORMAT or header.get("version") != VERSION:
        raise TableError(
            f"the header names format {header.get('format')!r} version "
            f"{header.get('version')!r}; v1 reads {FORMAT!r} version {VERSION}"
        )
    if header.get("dtype") != DTYPE:
        raise TableError(f"the values must be {DTYPE!r}, the header says {header.get('dtype')!r}")
    if header.get("axis_order") != [*AXES, "xyz"]:
        raise TableError(f"the axes must be in the order {[*AXES, 'xyz']}")
    axes = header.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(AXES):
        raise TableError(f"the header must give exactly the axes {list(AXES)}")
    sizes = []
    for name in AXES:
        nodes = axes[name]
        if not isinstance(nodes, list) or len(nodes) < 2:
            raise TableError(f"the {name} axis needs at least two nodes")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in nodes):
            raise TableError(f"the {name} axis holds a node that is not a finite number")
        if not all(b > a for a, b in zip(nodes, nodes[1:], strict=False)):
            raise TableError(f"the {name} axis must increase strictly")
        low, high = DOMAIN[name]
        if (nodes[0], nodes[-1]) != (low, high):
            raise TableError(
                f"the {name} axis runs from {nodes[0]} to {nodes[-1]}, not the domain "
                f"{low} to {high}"
            )
        sizes.append(len(nodes))
    if header.get("shape") != [*sizes, 3]:
        raise TableError(f"the shape {header.get('shape')} is not the axes' {[*sizes, 3]}")
    if header.get("ground_albedo") != GROUND_ALBEDO:
        raise TableError(
            f"the table's ground albedo is {header.get('ground_albedo')!r}; v1 is built for "
            f"{GROUND_ALBEDO}"
        )
    if header.get("baked_atmosphere") != _defaults():
        raise TableError(
            f"the table bakes {header.get('baked_atmosphere')!r}, not the Atmosphere defaults "
            f"{_defaults()!r}"
        )


def load_table(
    path: Path | str | None = None, *, expected_sha256: str = TABLE_SHA256
) -> MultipleScatteringTable:
    """Read, check and parse a table: the committed one by default.

    Raises `TableError` when the file's SHA-256 is not `expected_sha256` or its contents fail
    any check of `decode_table`.
    """
    if path is None:
        data = (resources.files("lastlight.render") / "data" / TABLE_NAME).read_bytes()
    else:
        data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise TableError(
            f"the table's SHA-256 is {digest}, not the pinned {expected_sha256}: refusing it"
        )
    return decode_table(data)


@cache
def default_table() -> MultipleScatteringTable:
    """The committed table, loaded and checked once per process."""
    return load_table()


# --- the engine ------------------------------------------------------------------------------


def fold_azimuth(rel_azimuth_deg: float) -> float:
    """Any relative azimuth onto 0 to 180: the sky is the same on either side of the sun."""
    if not math.isfinite(rel_azimuth_deg):
        raise ValueError(f"rel_azimuth_deg must be finite, got {rel_azimuth_deg!r}")
    return abs((rel_azimuth_deg + 180.0) % 360.0 - 180.0)


def check_domain(atm: Atmosphere, sun_elevation: float, view_elevations: Sequence[float]) -> None:
    """Raise `ValueError` naming the field or bound if v1 cannot render this point."""
    baked = _defaults()
    for name in BAKED_FIELDS:
        value = getattr(atm, name)
        if value != baked[name]:
            raise ValueError(
                f"renderer v1's table bakes {name} = {baked[name]}; this atmosphere has "
                f"{name} = {value!r}"
            )
    _within("aod_550", atm.aod_550, DOMAIN["aod_550"])
    _within("ozone_du", atm.ozone_du, DOMAIN["ozone_du"])
    _within("sun elevation", sun_elevation, DOMAIN["sun_elevation_deg"])
    for view in view_elevations:
        _within("view elevation", view, DOMAIN["view_elevation_deg"])


def _within(name: str, value: float, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not (math.isfinite(value) and low <= value <= high):
        raise ValueError(
            f"{name} {value!r} is outside renderer v1's table: it must lie between {low} and "
            f"{high} inclusive"
        )


def multiple_scattering_xyz(
    atm: Atmosphere,
    sun_elevation: float,
    view_elevations: Sequence[float],
    rel_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """XYZ of the light scattered more than once, at each view elevation: shape (views, 3).

    On the same scale as `lastlight.render.curve.sky_xyz` (a white card under the
    extraterrestrial sun has Y = 1), so v1's sky is `sky_xyz(..., engine="v0")` plus this.
    """
    check_domain(atm, sun_elevation, view_elevations)
    azimuth = fold_azimuth(float(rel_azimuth_deg))
    return default_table().xyz(
        atm.aod_550, atm.ozone_du, float(sun_elevation), view_elevations, azimuth
    )
