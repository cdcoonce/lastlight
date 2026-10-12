"""Renderer v1, the committed table: size, digest, contents, licence and interpolation (#32).

The multiple-scattering table is generated offline by `tools/make_bruneton_table.py` and committed
under `src/lastlight/render/data/`. These tests pin what the conductor's spec fixes:

* it is at most 2 MB, and its SHA-256 is pinned in the code (`v1.TABLE_SHA256`);
* the loader treats the file as untrusted: it refuses a digest that does not match, and, with
  the digest check satisfied, a wrong magic, header, dtype, shape, axis or value;
* its axes cover the declared domain and the baked atmosphere is `Atmosphere`'s defaults with a
  ground albedo of 0.2;
* off its nodes, its value agrees with a direct computation at the same resolution within 3
  percent (the direct computation is made by the generator, `--reference`, and committed).

The container format (also in `lastlight.render.v1`): 8 bytes of magic `LLMSXYZ\\0`, the header
length as a little-endian uint32, the header as UTF-8 JSON, then the values as little-endian
float32 in C order with the header's `shape`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from lastlight.render import Atmosphere, bruneton, v1

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "src" / "lastlight" / "render" / "data" / v1.TABLE_NAME
README = ROOT / "src" / "lastlight" / "render" / "data" / "README.md"
REFERENCE = ROOT / "tests" / "fixtures" / "bruneton_offnode.json"

MAGIC = b"LLMSXYZ\x00"
TWO_MEGABYTES = 2_000_000

DOMAIN = {
    "aod_550": (0.0, 1.0),
    "ozone_du": (250.0, 330.0),
    "sun_elevation_deg": (-12.0, 12.0),
    "view_elevation_deg": (0.0, 90.0),
    "rel_azimuth_deg": (0.0, 180.0),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split(data: bytes) -> tuple[dict, np.ndarray]:
    """The header and the values of a container, read by the format above (not by `v1`)."""
    assert data[:8] == MAGIC
    (length,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + length].decode("utf-8"))
    values = np.frombuffer(data[12 + length :], dtype="<f4").reshape(header["shape"])
    return header, values


def join(header: dict, values: np.ndarray) -> bytes:
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(text)) + text + np.asarray(values, "<f4").tobytes()


def load_bytes(tmp_path: Path, data: bytes) -> v1.MultipleScatteringTable:
    """Load `data` with its own digest pinned, so only the other checks can refuse it."""
    path = tmp_path / "table.bin"
    path.write_bytes(data)
    return v1.load_table(path, expected_sha256=sha256(data))


# --- the committed file -------------------------------------------------------------------


def test_the_committed_table_is_at_most_two_megabytes() -> None:
    assert TABLE.stat().st_size <= TWO_MEGABYTES


def test_the_committed_tables_sha256_is_the_pinned_one() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", v1.TABLE_SHA256)
    assert sha256(TABLE.read_bytes()) == v1.TABLE_SHA256


def test_the_default_table_loads_with_the_pinned_digest() -> None:
    table = v1.default_table()
    assert table.values.shape[-1] == 3
    assert table.values.dtype == np.float32


def test_the_axes_cover_exactly_the_declared_domain() -> None:
    header, values = split(TABLE.read_bytes())
    assert header["axis_order"] == [*DOMAIN, "xyz"]
    for name, (low, high) in DOMAIN.items():
        axis = header["axes"][name]
        assert (axis[0], axis[-1]) == (low, high), name
        assert all(b > a for a, b in zip(axis, axis[1:], strict=False)), name
    assert list(values.shape) == [len(header["axes"][name]) for name in DOMAIN] + [3]


def test_the_extra_light_is_positive_and_finite_everywhere_in_the_table() -> None:
    _, values = split(TABLE.read_bytes())
    assert np.isfinite(values).all()
    assert (values > 0.0).all()


def test_the_table_records_its_ground_albedo_and_the_atmosphere_it_bakes() -> None:
    header, _ = split(TABLE.read_bytes())
    assert header["ground_albedo"] == bruneton.GROUND_ALBEDO == 0.2
    defaults = Atmosphere(aod_550=0.0)
    baked = header["baked_atmosphere"]
    assert set(baked) == {
        "angstrom",
        "aerosol_scale_height_km",
        "pressure_hpa",
        "asymmetry",
        "single_scatter_albedo",
    }
    for name, value in baked.items():
        assert value == getattr(defaults, name), name


def test_the_table_records_its_scattering_orders_and_that_the_next_adds_under_1_percent() -> None:
    """Orders 2 to N, with N chosen so that order N+1 adds under 1 percent (sun 0, WHTM1)."""
    header, _ = split(TABLE.read_bytes())
    assert header["orders"] == [2, bruneton.RESOLUTION.orders]
    check = header["order_check"]
    assert check["sun_elevation_deg"] == 0.0
    assert check["max_share_of_next_order"] < 0.01


def test_the_header_holds_no_timestamp() -> None:
    header, _ = split(TABLE.read_bytes())
    text = json.dumps(header)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert not re.search(r"\d{1,2}:\d{2}", text)


# --- the loader refuses what it must not trust ----------------------------------------------


def test_the_loader_refuses_a_table_whose_digest_is_not_the_pinned_one(tmp_path: Path) -> None:
    data = bytearray(TABLE.read_bytes())
    data[-1] ^= 0x01
    path = tmp_path / "tampered.bin"
    path.write_bytes(bytes(data))
    with pytest.raises(v1.TableError, match="SHA-256"):
        v1.load_table(path)


def test_the_loader_accepts_an_untouched_copy_under_its_own_digest(tmp_path: Path) -> None:
    table = load_bytes(tmp_path, TABLE.read_bytes())
    np.testing.assert_array_equal(table.values, v1.default_table().values)


def _edited(edit) -> bytes:
    header, values = split(TABLE.read_bytes())
    header = json.loads(json.dumps(header))
    values = np.array(values)
    result = edit(header, values)
    return result if isinstance(result, bytes) else join(header, values)


def _set(header: dict, path: tuple, value) -> None:
    target = header
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _nan(header: dict, values: np.ndarray) -> None:
    values[0, 0, 0, 0, 0, 1] = np.nan


def _negative(header: dict, values: np.ndarray) -> None:
    values[1, 1, 1, 1, 1, 1] = -values[1, 1, 1, 1, 1, 1]


def _zero(header: dict, values: np.ndarray) -> None:
    values[2, 0, 3, 0, 0, 2] = 0.0


def _infinite(header: dict, values: np.ndarray) -> None:
    values[0, 1, 0, 2, 0, 0] = np.inf


def _bad_magic(header: dict, values: np.ndarray) -> bytes:
    return b"NOTATABL" + join(header, values)[8:]


def _truncated(header: dict, values: np.ndarray) -> bytes:
    return join(header, values)[:-4]


def _trailing(header: dict, values: np.ndarray) -> bytes:
    return join(header, values) + b"\x00\x00\x00\x00"


def _not_json(header: dict, values: np.ndarray) -> bytes:
    text = b"{not json"
    return MAGIC + struct.pack("<I", len(text)) + text + np.asarray(values, "<f4").tobytes()


def _long_header(header: dict, values: np.ndarray) -> bytes:
    return MAGIC + struct.pack("<I", 2**31) + b"{}"


BAD_TABLES = {
    "a NaN value": _nan,
    "an infinite value": _infinite,
    "a negative value": _negative,
    "a zero value": _zero,
    "the wrong magic": _bad_magic,
    "a truncated body": _truncated,
    "trailing bytes": _trailing,
    "a header that is not JSON": _not_json,
    "a header longer than the file": _long_header,
    "a float64 dtype": lambda h, v: _set(h, ("dtype",), "<f8"),
    "a big-endian dtype": lambda h, v: _set(h, ("dtype",), ">f4"),
    "a shape that is not the axes'": lambda h, v: _set(h, ("shape",), [*h["shape"][:-1], 4]),
    "a missing axis": lambda h, v: h["axes"].pop("ozone_du"),
    "axes in another order": lambda h, v: _set(
        h, ("axis_order",), ["ozone_du", "aod_550", *h["axis_order"][2:]]
    ),
    "a decreasing axis": lambda h, v: _set(
        h, ("axes", "sun_elevation_deg"), h["axes"]["sun_elevation_deg"][::-1]
    ),
    "a repeated node": lambda h, v: _set(
        h,
        ("axes", "view_elevation_deg"),
        [0.0, 0.0, *h["axes"]["view_elevation_deg"][2:]],
    ),
    "an axis that stops short of the domain": lambda h, v: _set(
        h, ("axes", "aod_550"), [*h["axes"]["aod_550"][:-1], 0.9]
    ),
    "an axis that starts past the domain": lambda h, v: _set(
        h, ("axes", "sun_elevation_deg"), [-11.0, *h["axes"]["sun_elevation_deg"][1:]]
    ),
    "a non-finite node": lambda h, v: _set(
        h, ("axes", "rel_azimuth_deg"), [*h["axes"]["rel_azimuth_deg"][:-1], math.inf]
    ),
    "another ground albedo": lambda h, v: _set(h, ("ground_albedo",), 0.3),
    "another baked asymmetry": lambda h, v: _set(h, ("baked_atmosphere", "asymmetry"), 0.6),
    "another baked pressure": lambda h, v: _set(h, ("baked_atmosphere", "pressure_hpa"), 1013.25),
    "another format": lambda h, v: _set(h, ("format",), "something-else"),
    "another version": lambda h, v: _set(h, ("version",), 2),
}


@pytest.mark.parametrize("edit", BAD_TABLES.values(), ids=BAD_TABLES.keys())
def test_the_loader_refuses_a_table_it_cannot_trust_even_with_a_matching_digest(
    tmp_path: Path, edit
) -> None:
    with pytest.raises(v1.TableError):
        load_bytes(tmp_path, _edited(edit))


def test_the_container_is_binary_so_the_privacy_scan_reads_it_as_such() -> None:
    assert b"\x00" in TABLE.read_bytes()[:8192]


# --- licence ------------------------------------------------------------------------------

BSD_CLAUSES = (
    "Copyright (c) 2017 Eric Bruneton",
    "Redistributions of source code must retain the above copyright",
    "Redistributions in binary form must reproduce the above copyright",
    "Neither the name of the copyright",
    'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
)


def _flat(text: str) -> str:
    return " ".join(text.replace("#", " ").split())


@pytest.mark.parametrize(
    "path",
    [README, ROOT / "src" / "lastlight" / "render" / "bruneton.py"],
    ids=["data-readme", "port-module"],
)
def test_bruneton_s_bsd_3_clause_notice_is_kept(path: Path) -> None:
    text = _flat(path.read_text())
    for clause in BSD_CLAUSES:
        assert _flat(clause) in text, clause


# --- interpolation ------------------------------------------------------------------------


def test_off_its_nodes_the_table_agrees_with_a_direct_computation_within_3_percent() -> None:
    """At aod 0.035, ozone 300, sun -3.25, view 1.7, relative azimuth 27 (off every node).

    The direct computation is the generator's (`--reference`): the same precomputation at that
    atmosphere, then the ground observer's multiple scattering at exactly that geometry, with no
    table in between. It ran at the table's own resolution and orders.
    """
    reference = json.loads(REFERENCE.read_text())
    point = reference["point"]
    assert point == {
        "aod_550": 0.035,
        "ozone_du": 300.0,
        "sun_elevation_deg": -3.25,
        "view_elevation_deg": 1.7,
        "rel_azimuth_deg": 27.0,
    }
    table = v1.default_table()
    header, _ = split(TABLE.read_bytes())
    assert reference["resolution"] == header["resolution"]
    assert reference["orders"] == header["orders"]
    assert reference["ground_albedo"] == header["ground_albedo"]
    for name, value in point.items():
        assert value not in header["axes"][name], f"{name} {value} is a node"

    got = table.xyz(
        point["aod_550"],
        point["ozone_du"],
        point["sun_elevation_deg"],
        [point["view_elevation_deg"]],
        point["rel_azimuth_deg"],
    )[0]
    want = np.array(reference["xyz"])
    assert (want > 0.0).all()
    np.testing.assert_allclose(got, want, rtol=0.03)


def test_at_a_node_the_table_returns_the_stored_value() -> None:
    table = v1.default_table()
    axes = table.axes
    i, j, k, m, n = 2, 1, 7, 3, 4
    got = table.xyz(
        axes["aod_550"][i],
        axes["ozone_du"][j],
        axes["sun_elevation_deg"][k],
        [axes["view_elevation_deg"][m]],
        axes["rel_azimuth_deg"][n],
    )[0]
    np.testing.assert_allclose(got, table.values[i, j, k, m, n], rtol=1e-6)
