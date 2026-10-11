"""The frozen fit file, the sky mask drawn from it, and the committed WHTM1 fit."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, REF_ELEV_M, REF_LAT, REF_LON, WIDTH
from PIL import Image

from lastlight.camgeom.fitfile import (
    GEOMETRY_DIR,
    CameraFit,
    FitFileError,
    FrameRecord,
    load_fit,
    write_fit,
)
from lastlight.camgeom.mask import sky_mask, write_mask_png
from lastlight.camgeom.model import CameraModel
from lastlight.camgeom.orientation import Orientation, hfov_is_pinned
from lastlight.camgeom.skyline import default_overlay_boxes
from lastlight.privacy.rules import text_hits

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "src" / "lastlight" / "cameras" / "geometry" / "WHTM1.json"

COLUMNS = tuple(float(x) for x in range(0, WIDTH, 10)) + (float(WIDTH - 1),)


def assert_nothing_private(text: str) -> None:
    """The repository's own text rules (P001 to P003) find nothing in `text`."""
    found = [text[hit.start : hit.end] for hit in text_hits(text)]
    assert not found, f"the privacy rules flag {found}"


def example_fit(**changes) -> CameraFit:
    fit = CameraFit(
        camera_id="SYN1",
        width=WIDTH,
        height=HEIGHT,
        orientation=Orientation(274.0, 2.0, 0.4, 66.0),
        skyline_rms_px=0.8,
        skyline_rms_deg=0.075,
        skyline_median_abs_px=0.6,
        n_columns=790,
        skyline_x_px=COLUMNS,
        skyline_y_px=tuple(150.0 + 0.01 * x for x in COLUMNS),
        overlay_boxes=default_overlay_boxes(WIDTH, HEIGHT),
        frames_used=(FrameRecord("2026-10-10", 14.65, "ab" * 32),),
        sun_check={"status": "skipped", "reason": "sun hidden by cloud", "tolerance_deg": 1.0},
        radial_term={"k1": 0.1, "skyline_rms_px": 0.7, "adopted": False},
        position_search={"box_km": 8.0, "positions": 60, "adopted": True},
        frozen_on="2026-10-10",
        position_basis="operator description, supplied at fit time and not stored",
    )
    return dataclasses.replace(fit, **changes)


# --- writing and reading --------------------------------------------------------------------------


def test_a_fit_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    fit = example_fit()
    path = write_fit(fit, tmp_path / "geometry" / "SYN1.json")
    loaded = load_fit("SYN1", directory=tmp_path / "geometry")
    assert path.read_text().endswith("\n")
    assert loaded.to_json() == fit.to_json()
    assert loaded.orientation == fit.orientation
    assert loaded.frames_used == fit.frames_used
    assert loaded.overlay_boxes == fit.overlay_boxes


def test_the_file_is_versioned_and_says_what_it_fitted() -> None:
    data = example_fit().to_json()
    assert data["schema_version"] == 1
    assert data["camera_id"] == "SYN1"
    assert data["image"] == {"width": WIDTH, "height": HEIGHT}
    assert set(data["orientation"]) == {"azimuth_deg", "elevation_deg", "roll_deg", "hfov_deg"}
    assert set(data["skyline_fit"]) == {"rms_px", "rms_deg", "median_abs_px", "n_columns"}
    assert data["fit_frozen_on"] == "2026-10-10"
    assert data["frames_used"] == [
        {"evening": "2026-10-10", "sun_elevation_deg": 14.65, "sha256": "ab" * 32}
    ]


def test_a_fit_without_a_position_search_says_so_with_null() -> None:
    data = example_fit(position_search=None).to_json()
    assert data["position_search"] is None
    assert CameraFit.from_json(data).position_search is None


def test_the_file_never_carries_the_cameras_coordinates_or_a_clock_time(tmp_path: Path) -> None:
    text = write_fit(example_fit(), tmp_path / "SYN1.json").read_text()
    assert_nothing_private(text)
    keys = set(re.findall(r'"([A-Za-z_0-9]+)":', text))
    assert not keys & {"lat", "lon", "latitude", "longitude", "elev_m", "position"}


def test_the_privacy_probe_itself_catches_what_it_is_meant_to() -> None:
    # The leaks are assembled at run time so that this file holds none of them itself.
    clock = ":".join(["17", "42"])
    iso = "2026-10-10" + "T" + ":".join(["00", "24", "00"]) + "Z"
    pair = f"[{REF_LAT + 0.1:.1f}, {REF_LON - 0.2:.1f}]"
    for leak in [f'"t": "{clock}"', f'"t": "{iso}"', f"sunset at {clock}", pair]:
        with pytest.raises(AssertionError):
            assert_nothing_private(leak)
    assert_nothing_private("[14.65, 274.0, 66.0]")


def test_loading_a_fit_that_is_not_there_names_the_camera(tmp_path: Path) -> None:
    with pytest.raises(FitFileError, match="NOPE1"):
        load_fit("NOPE1", directory=tmp_path)


def _tampered(tmp_path: Path, edit) -> Path:
    data = example_fit().to_json()
    edit(data)
    path = tmp_path / "BAD1.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(schema_version=99), "schema_version"),
        (lambda d: d.pop("orientation"), "orientation"),
        (lambda d: d["skyline"].update(y_px=[1.0]), "skyline"),
        (lambda d: d.update(camera_id="OTHER"), "camera_id"),
    ],
)
def test_a_malformed_fit_file_is_refused_with_the_reason(tmp_path: Path, edit, message) -> None:
    _tampered(tmp_path, edit)
    with pytest.raises(FitFileError, match=message):
        load_fit("BAD1", directory=tmp_path)


def test_a_model_is_built_from_the_fit_at_a_position_the_caller_supplies() -> None:
    model = example_fit().camera_model(REF_LAT, REF_LON, REF_ELEV_M)
    assert isinstance(model, CameraModel)
    assert (model.azimuth_deg, model.elevation_deg, model.roll_deg, model.hfov_deg) == (
        274.0,
        2.0,
        0.4,
        66.0,
    )
    assert (model.width, model.height, model.camera_id) == (WIDTH, HEIGHT, "SYN1")
    assert (model.lat, model.lon, model.elev_m) == (REF_LAT, REF_LON, REF_ELEV_M)


def test_the_fitted_skyline_is_interpolated_between_its_stored_columns() -> None:
    fit = example_fit(skyline_x_px=(0.0, 100.0, 799.0), skyline_y_px=(100.0, 120.0, None))
    y = fit.skyline_y(np.array([0.0, 50.0, 100.0, 500.0]))
    assert y[:3] == pytest.approx([100.0, 110.0, 120.0])
    assert math.isnan(y[3]), "a column next to a missing knot has no skyline"


# --- the sky mask  --------------------------------------------------------------------------------


def flat_skyline_fit(row: float, **changes) -> CameraFit:
    return example_fit(skyline_y_px=tuple(row for _ in COLUMNS), **changes)


def test_the_mask_is_sky_above_the_skyline_minus_a_margin() -> None:
    fit = flat_skyline_fit(150.0)
    margin_px = (
        fit.camera_model(REF_LAT, REF_LON, REF_ELEV_M).focal_px * math.pi / 180.0
    )  # 1 degree
    mask = sky_mask(fit, margin_deg=1.0)
    assert mask.shape == (HEIGHT, WIDTH) and mask.dtype == bool
    boundary = int(150.0 - margin_px)
    assert mask[: boundary - 1, :500].all()
    assert not mask[boundary + 2 :, :500].any()


def test_a_wider_margin_removes_more_sky() -> None:
    fit = flat_skyline_fit(150.0)
    assert sky_mask(fit, margin_deg=3.0).sum() < sky_mask(fit, margin_deg=1.0).sum()
    assert sky_mask(fit, margin_deg=0.0)[:148, :500].all()


def test_the_timestamp_region_is_never_sky_even_when_it_is_above_the_skyline() -> None:
    fit = flat_skyline_fit(292.0)  # a horizon so low that the stamp box sits in the "sky"
    mask = sky_mask(fit, margin_deg=0.0)
    (box,) = fit.overlay_boxes
    assert not mask[box.y0 : box.y1, box.x0 : box.x1].any()
    assert mask[box.y0 - 1, box.x0 + 10], "just above the stamp is still sky"
    assert mask[box.y0 : box.y1, : box.x0 - 1].any(), "and the rest of those rows is sky"


def test_a_column_with_no_fitted_skyline_is_not_called_sky() -> None:
    ys = [150.0 for _ in COLUMNS]
    ys[10] = None
    ys[11] = None
    fit = example_fit(skyline_y_px=tuple(ys))
    mask = sky_mask(fit, margin_deg=1.0)
    gap = int(COLUMNS[10]) + 5
    assert not mask[:, gap].any()
    assert mask[:100, 300].all()


def test_a_negative_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="margin"):
        sky_mask(example_fit(), margin_deg=-1.0)


def _png_chunk_types(path: Path) -> list[str]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    types, offset = [], 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        types.append(data[offset + 4 : offset + 8].decode("ascii"))
        offset += 12 + length
    return types


def test_the_mask_png_round_trips_and_carries_no_metadata_chunks(tmp_path: Path) -> None:
    mask = sky_mask(flat_skyline_fit(150.0), margin_deg=1.0)
    path = write_mask_png(mask, tmp_path / "masks" / "SYN1.png")
    with Image.open(path) as image:
        loaded = np.asarray(image) > 0
    assert np.array_equal(loaded, mask)
    forbidden = {"tEXt", "zTXt", "iTXt", "eXIf", "tIME", "pHYs", "iCCP"}
    assert not forbidden & set(_png_chunk_types(path))


# --- the committed fit for the real camera  -------------------------------------------------------


def test_the_committed_whtm1_fit_is_a_valid_frozen_fit() -> None:
    fit = load_fit("WHTM1")
    assert COMMITTED.parent == GEOMETRY_DIR
    assert (fit.width, fit.height) == (800, 300)
    assert abs(fit.orientation.azimuth_deg - 274.0) < 20.0, "within the operator's stated view"
    assert not hfov_is_pinned(fit.orientation.hfov_deg), "a fit on a limit is not a fit"
    assert abs(fit.orientation.elevation_deg) < 10.0 and abs(fit.orientation.roll_deg) < 5.0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", fit.frozen_on)
    assert 0.0 < fit.skyline_rms_px < 2.0, "the issue's target for the skyline residual"
    assert fit.skyline_rms_deg == pytest.approx(
        fit.skyline_rms_px / (fit.camera_model(REF_LAT, REF_LON, 0.0).focal_px * math.pi / 180),
        rel=0.02,
    )
    assert fit.n_columns > 500
    assert len(fit.skyline_x_px) == len(fit.skyline_y_px) > 20
    assert fit.radial_term is not None and fit.radial_term["adopted"] is False
    search = fit.position_search
    assert search is not None, "the committed fit must record the position search that was run"
    assert search["adopted"] is True and search["positions"] >= 25 and search["box_km"] >= 6.0
    assert search["rms_px_best"] == pytest.approx(fit.skyline_rms_px, abs=0.01)
    assert search["near_best_positions"] >= 1


def test_the_committed_fit_names_its_frames_without_pixels_or_clock_times() -> None:
    fit = load_fit("WHTM1")
    assert fit.frames_used, "a fit must say which frames it was frozen on"
    for record in fit.frames_used:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", record.evening)
        assert re.fullmatch(r"[0-9a-f]{64}", record.sha256)
        assert -10.0 < record.sun_elevation_deg < 90.0


def test_the_committed_sun_check_passed_or_says_why_it_was_skipped() -> None:
    check = load_fit("WHTM1").sun_check
    assert check["status"] in {"passed", "skipped"}
    assert check["reason"]
    assert check["tolerance_deg"] > 0


def test_the_committed_fit_holds_no_coordinates_clock_times_or_image_data() -> None:
    text = COMMITTED.read_text()
    assert_nothing_private(text)
    keys = set(re.findall(r'"([A-Za-z_0-9]+)":', text))
    assert not keys & {"lat", "lon", "latitude", "longitude", "elev_m", "position", "pixels"}
    assert len(text) < 20_000


def test_the_committed_fit_makes_a_sensible_sky_mask() -> None:
    fit = load_fit("WHTM1")
    mask = sky_mask(fit, margin_deg=1.0)
    share = mask.mean()
    assert 0.3 < share < 0.7, "about half of the frame is sky, as the camera's page says"
    assert not mask[-30:, -200:].any(), "the timestamp corner is never sky"
