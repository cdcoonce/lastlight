"""The rendered curve and the strip: grid, exposure convention, speed, and a metadata-free PNG."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lastlight.render import (
    SUN_ELEVATIONS,
    VIEW_ELEVATIONS,
    Atmosphere,
    render_curve,
    render_row,
    render_strip,
    save_strip,
)
from lastlight.render.colorimetry import lab_to_srgb

ATMOSPHERE = Atmosphere(aod_550=0.1)
KEYS = {"sun_elevation", "view_elevation", "L", "a", "b"}


@pytest.fixture(scope="module")
def curve() -> list[dict[str, float]]:
    return render_curve(ATMOSPHERE)


# --- the grid -------------------------------------------------------------------------------


def test_sun_elevations_run_from_plus_6_to_minus_8_in_half_degree_steps() -> None:
    assert SUN_ELEVATIONS[0] == 6.0
    assert SUN_ELEVATIONS[-1] == -8.0
    assert len(SUN_ELEVATIONS) == 29
    assert all(a - b == 0.5 for a, b in zip(SUN_ELEVATIONS, SUN_ELEVATIONS[1:], strict=False))


def test_sweep_matches_the_geometry_module_once_it_exists() -> None:
    """The renderer's sweep must be geometry.ELEVATION_SWEEP; geometry lands with issue #4."""
    if importlib.util.find_spec("lastlight.geometry") is None:
        pytest.skip("lastlight.geometry has not landed yet (issue #4)")
    from lastlight import geometry

    assert tuple(SUN_ELEVATIONS) == tuple(geometry.ELEVATION_SWEEP)


def test_view_elevations_are_the_seven_camera_bands() -> None:
    assert tuple(VIEW_ELEVATIONS) == (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0)


def test_curve_has_one_row_per_sun_and_view_elevation(curve: list[dict[str, float]]) -> None:
    assert len(curve) == len(SUN_ELEVATIONS) * len(VIEW_ELEVATIONS) == 203
    assert all(set(row) == KEYS for row in curve)
    assert [(r["sun_elevation"], r["view_elevation"]) for r in curve] == [
        (s, v) for s in SUN_ELEVATIONS for v in VIEW_ELEVATIONS
    ]


def test_curve_is_plain_json_ready_python_floats(curve: list[dict[str, float]]) -> None:
    """The seal slice hashes this as canonical JSON: no numpy scalars, no NaN."""
    assert all(type(value) is float for row in curve for value in row.values())
    json.dumps(curve, allow_nan=False)


def test_curve_is_deterministic(curve: list[dict[str, float]]) -> None:
    assert render_curve(ATMOSPHERE) == curve


# --- the exposure convention ----------------------------------------------------------------


def test_exposure_is_set_per_sun_elevation_so_the_brightest_view_is_L70(
    curve: list[dict[str, float]],
) -> None:
    """Recorded choice: each sun elevation is exposed so its brightest view reads L* = 70.

    The camera will be normalised the same way, so the comparison is of colour along the sky at
    one moment, not of how fast the evening darkens.
    """
    for sun in SUN_ELEVATIONS:
        lightness = [r["L"] for r in curve if r["sun_elevation"] == sun]
        assert max(lightness) == pytest.approx(70.0, abs=1e-9), sun
        assert min(lightness) >= 0.0


def test_render_row_is_one_column_of_the_curve(curve: list[dict[str, float]]) -> None:
    assert render_row(ATMOSPHERE, -3.0) == [r for r in curve if r["sun_elevation"] == -3.0]


def test_a_fully_shadowed_sky_renders_as_black_not_as_an_error() -> None:
    row = render_row(ATMOSPHERE, -30.0)
    assert len(row) == len(VIEW_ELEVATIONS)
    assert all(r["L"] == r["a"] == r["b"] == 0.0 for r in row)


def test_the_full_curve_renders_in_under_ten_seconds() -> None:
    start = time.perf_counter()
    render_curve(Atmosphere(aod_550=0.2, ozone_du=300.0))
    assert time.perf_counter() - start < 10.0


# --- the strip ------------------------------------------------------------------------------

CELL_W, CELL_H = 8, 12


def test_strip_is_sun_elevation_across_and_view_elevation_up(curve: list[dict[str, float]]) -> None:
    image = render_strip(curve, cell_width=CELL_W, cell_height=CELL_H)
    assert image.mode == "RGB"
    assert image.size == (len(SUN_ELEVATIONS) * CELL_W, len(VIEW_ELEVATIONS) * CELL_H)
    pixels = np.asarray(image)

    def painted(sun: float, view: float) -> tuple[int, ...]:
        column = SUN_ELEVATIONS.index(sun)
        row = len(VIEW_ELEVATIONS) - 1 - VIEW_ELEVATIONS.index(view)  # highest view at the top
        return tuple(pixels[row * CELL_H + CELL_H // 2, column * CELL_W + CELL_W // 2])

    for sun, view in ((6.0, 2.0), (6.0, 45.0), (-8.0, 2.0), (-1.5, 20.0)):
        entry = next(r for r in curve if r["sun_elevation"] == sun and r["view_elevation"] == view)
        expected = tuple(round(255 * c) for c in lab_to_srgb(entry["L"], entry["a"], entry["b"]))
        assert painted(sun, view) == expected


def test_strip_accepts_the_curve_in_any_order(curve: list[dict[str, float]]) -> None:
    shuffled = list(reversed(curve))
    assert render_strip(shuffled).tobytes() == render_strip(curve).tobytes()


def test_strip_rejects_an_incomplete_curve(curve: list[dict[str, float]]) -> None:
    with pytest.raises(ValueError, match="rectangular"):
        render_strip(curve[:-1])
    with pytest.raises(ValueError, match="empty"):
        render_strip([])


def _png_chunks(data: bytes) -> list[str]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, offset = [], 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunks.append(data[offset + 4 : offset + 8].decode("ascii"))
        offset += 12 + length
    return chunks


def test_saved_strip_carries_no_metadata(curve: list[dict[str, float]], tmp_path: Path) -> None:
    """Privacy rule 3: sky only, every ancillary chunk stripped (EXIF, XMP, ICC, text, time)."""
    image = render_strip(curve)
    assert image.info == {}
    path = tmp_path / "strip.png"
    save_strip(image, path)
    chunks = _png_chunks(path.read_bytes())
    assert chunks[0] == "IHDR"
    assert chunks[-1] == "IEND"
    assert set(chunks) <= {"IHDR", "IDAT", "IEND"}
    with Image.open(io.BytesIO(path.read_bytes())) as reopened:
        assert reopened.info == {}
        assert reopened.getexif() == {}


def test_save_strip_removes_metadata_the_caller_attached(
    curve: list[dict[str, float]], tmp_path: Path
) -> None:
    image = render_strip(curve)
    image.info["dpi"] = (300, 300)
    image.info["icc_profile"] = b"not a profile"
    image.info["Comment"] = "locating text"
    path = tmp_path / "strip.png"
    save_strip(image, path)
    assert set(_png_chunks(path.read_bytes())) <= {"IHDR", "IDAT", "IEND"}
