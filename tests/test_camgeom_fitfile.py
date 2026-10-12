"""The frozen fit file, the sky mask drawn from it, and the committed WHTM1 fit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import (
    HEIGHT,
    REF_ELEV_M,
    REF_LAT,
    REF_LON,
    SKYLINE_COLUMNS,
    WIDTH,
    example_fit,
)
from PIL import Image

from lastlight.camgeom.fitfile import (
    GEOMETRY_DIR,
    SCHEMA_VERSION,
    CameraFit,
    FitFileError,
    azimuth_difference_deg,
    load_fit,
    uncertainty_from_ranges,
    write_fit,
)
from lastlight.camgeom.mask import sky_mask, write_mask_png
from lastlight.camgeom.model import CameraModel, pixels_per_degree
from lastlight.camgeom.orientation import LeastSquaresGuard, Orientation, hfov_is_pinned
from lastlight.privacy.rules import text_hits

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "src" / "lastlight" / "cameras" / "geometry" / "WHTM1.json"

COLUMNS = SKYLINE_COLUMNS


def assert_nothing_private(text: str) -> None:
    """The repository's own text rules (P001 to P003) find nothing in `text`."""
    found = [text[hit.start : hit.end] for hit in text_hits(text)]
    assert not found, f"the privacy rules flag {found}"


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
    assert data["schema_version"] == 2
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


# --- provisional status, uncertainty and the stated direction -----------------------------------


def test_the_file_says_it_is_provisional_and_how_far_the_pointing_could_be_off() -> None:
    data = example_fit().to_json()
    assert SCHEMA_VERSION == 2 and data["schema_version"] == 2
    assert data["provisional"] is True
    assert data["uncertainty"] == {
        "azimuth_deg": [272.0, 276.0],
        "hfov_deg": [60.0, 70.0],
        "px_per_deg": [10.3, 12.4],
        "basis": "positions within 10 percent of the best residual inside the box",
        "lower_bounds": False,
    }
    assert data["stated_azimuth_deg"] == 272.0
    assert data["azimuth_minus_stated_deg"] == 2.0


def test_a_fit_object_exposes_the_same_fields_and_they_survive_a_round_trip(
    tmp_path: Path,
) -> None:
    fit = example_fit()
    assert fit.provisional is True
    assert fit.uncertainty is not None and fit.uncertainty.hfov_deg == (60.0, 70.0)
    assert (fit.stated_azimuth_deg, fit.azimuth_minus_stated_deg) == (272.0, 2.0)
    write_fit(fit, tmp_path / "SYN1.json")
    loaded = load_fit("SYN1", directory=tmp_path)
    assert loaded.provisional is True
    assert loaded.uncertainty == fit.uncertainty
    assert (loaded.stated_azimuth_deg, loaded.azimuth_minus_stated_deg) == (272.0, 2.0)


def test_a_fit_with_no_search_and_no_registry_entry_says_so_with_null() -> None:
    fit = example_fit(uncertainty=None, stated_azimuth_deg=None, azimuth_minus_stated_deg=None)
    data = fit.to_json()
    assert data["uncertainty"] is None and data["stated_azimuth_deg"] is None
    again = CameraFit.from_json(data)
    assert again.uncertainty is None and again.azimuth_minus_stated_deg is None


def test_a_fit_whose_sun_check_passed_is_not_provisional() -> None:
    passed = example_fit(
        sun_check={"status": "passed", "reason": "1 of 1 frames", "tolerance_deg": 1.0},
        provisional=False,
    )
    assert CameraFit.from_json(passed.to_json()).provisional is False


def test_the_provisional_flag_must_agree_with_the_sun_check() -> None:
    # A hand edit that clears the flag without a passing sun check, or sets it after one, is
    # refused: the flag is derived from the check, never an independent claim.
    for status, claimed in (("skipped", False), ("failed", False), ("passed", True)):
        data = example_fit().to_json()
        data["sun_check"]["status"] = status
        data["provisional"] = claimed
        with pytest.raises(FitFileError, match="provisional"):
            CameraFit.from_json(data)


def test_the_difference_from_the_stated_azimuth_wraps_through_north() -> None:
    assert azimuth_difference_deg(288.7, 274.0) == pytest.approx(14.7)
    assert azimuth_difference_deg(5.0, 355.0) == pytest.approx(10.0)
    assert azimuth_difference_deg(355.0, 5.0) == pytest.approx(-10.0)


def test_the_pixel_scale_range_follows_from_the_field_of_view_range() -> None:
    found = uncertainty_from_ranges((283.3, 291.8), (17.7, 28.5), 800, "basis", True)
    assert found.hfov_deg == (17.7, 28.5)
    assert found.px_per_deg[1] > found.px_per_deg[0], "a narrower lens is a finer scale"
    focal = lambda h: 400.0 / math.tan(math.radians(h) / 2.0)  # noqa: E731
    assert found.px_per_deg[0] == pytest.approx(focal(28.5) * math.pi / 180.0)
    assert found.px_per_deg[1] == pytest.approx(focal(17.7) * math.pi / 180.0)
    assert found.lower_bounds is True


def test_the_documentation_of_the_model_and_the_mask_says_the_mapping_is_provisional() -> None:
    assert "provisional" in (CameraFit.camera_model.__doc__ or "")
    assert "provisional" in (sky_mask.__doc__ or "")
    assert "0.024" in (sky_mask.__doc__ or ""), "the example scale is that of the real fit"


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


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["orientation"].update(hfov_deg=0.0), "hfov_deg"),
        (lambda d: d["orientation"].update(hfov_deg=-20.0), "hfov_deg"),
        (lambda d: d["orientation"].update(hfov_deg=180.0), "hfov_deg"),
        (lambda d: d["orientation"].update(hfov_deg=math.nan), "hfov_deg"),
        (lambda d: d["orientation"].update(azimuth_deg=math.nan), "azimuth_deg"),
        (lambda d: d["orientation"].update(elevation_deg=95.0), "elevation_deg"),
        (lambda d: d["orientation"].update(roll_deg="level"), "roll_deg"),
        (lambda d: d["image"].update(width=0), "width"),
        (lambda d: d["image"].update(height=-300), "height"),
        (lambda d: d["image"].update(width=800.5), "width"),
        (lambda d: d["skyline_fit"].update(rms_px=math.inf), "rms_px"),
        (lambda d: d["skyline_fit"].update(n_columns=-1), "n_columns"),
        (lambda d: d["skyline"]["x_px"].__setitem__(5, d["skyline"]["x_px"][4]), "increasing"),
        (lambda d: d["skyline"]["x_px"].reverse(), "increasing"),
        (lambda d: d["skyline"]["y_px"].__setitem__(3, math.nan), "y_px"),
        (lambda d: d["uncertainty"]["hfov_deg"].reverse(), "uncertainty"),
        (lambda d: d["uncertainty"].update(lower_bounds="no"), "lower_bounds"),
        (lambda d: d.update(azimuth_minus_stated_deg=9.0), "azimuth_minus_stated"),
        (lambda d: d.update(azimuth_minus_stated_deg=None), "stated"),
        (lambda d: d.update(provisional="yes"), "provisional"),
    ],
)
def test_a_fit_file_that_camera_model_or_the_mask_could_not_use_is_refused(
    tmp_path: Path, edit, message
) -> None:
    _tampered(tmp_path, edit)
    with pytest.raises(FitFileError, match=message):
        load_fit("BAD1", directory=tmp_path)


def test_a_fit_file_from_the_first_schema_is_refused_with_the_version() -> None:
    data = example_fit().to_json()
    data["schema_version"] = 1
    for key in ("provisional", "uncertainty", "stated_azimuth_deg", "azimuth_minus_stated_deg"):
        del data[key]
    with pytest.raises(FitFileError, match="schema_version"):
        CameraFit.from_json(data)


@pytest.mark.parametrize("content", ["[]", "42", "null", '"text"', "{not json", ""])
def test_a_file_that_is_not_a_json_object_is_a_fit_file_error(tmp_path: Path, content: str) -> None:
    (tmp_path / "BAD1.json").write_text(content)
    with pytest.raises(FitFileError, match="BAD1.json"):
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


@pytest.mark.parametrize("hfov", [66.0, 19.0])
def test_the_margin_is_converted_to_rows_at_the_pixel_scale_of_the_fitted_lens(
    hfov: float,
) -> None:
    fit = flat_skyline_fit(150.0, orientation=Orientation(274.0, 2.0, 0.4, hfov))
    pixels_per_degree_here = (WIDTH / 2) / math.tan(math.radians(hfov / 2)) * math.pi / 180.0
    for margin in (0.5, 1.0, 2.0):
        limit = 150.0 - margin * pixels_per_degree_here
        assert limit > 1, "the margin must leave some sky in the test frame"
        assert sky_mask(fit, margin_deg=margin)[:, 100].sum() == math.ceil(limit)


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
    assert fit.stated_azimuth_deg == 274.0, "the registry's direction for WHTM1, recorded"
    assert fit.azimuth_minus_stated_deg == pytest.approx(
        fit.orientation.azimuth_deg - 274.0, abs=0.01
    ), "and how far the fit landed from it, so the 15 degrees is on the record"
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
    assert search["step_km"] <= 1.0 and search["positions"] >= 17 * 17, "a 1 km grid over the box"
    assert isinstance(search["best_on_box_edge"], bool)
    assert isinstance(search["residual_falls_beyond_box"], bool)
    assert search["ring_positions"] >= 40, "the probes just outside the box were run"


def test_the_committed_fit_is_marked_provisional_with_the_ranges_it_could_have_landed_in() -> None:
    fit = load_fit("WHTM1")
    assert fit.sun_check["status"] != "passed"
    assert fit.provisional is True
    u = fit.uncertainty
    assert u is not None, "a fit from a position search carries its uncertainty"
    o = fit.orientation
    assert u.azimuth_deg[0] <= o.azimuth_deg <= u.azimuth_deg[1]
    assert u.hfov_deg[0] <= o.hfov_deg <= u.hfov_deg[1]
    scale = pixels_per_degree(o.hfov_deg, fit.width)
    assert u.px_per_deg[0] <= scale <= u.px_per_deg[1]
    assert u.hfov_deg[1] > 1.3 * u.hfov_deg[0], "the field of view is known only to a wide range"
    search = fit.position_search
    assert search is not None
    assert u.lower_bounds == (search["best_on_box_edge"] or search["residual_falls_beyond_box"])


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


# --- the least-squares guard record (issue 28) ----------------------------------------------------
#
# A fit made after the guard existed says how the guard judged the least-squares stage. A fit
# made before it (the committed WHTM1, whose digest every observation cites) has no such object,
# is still accepted, and is not given one.

WHTM1_SHA256 = "a2f462fea27f99552e8262c742166bbd92b6f3c4e815acc7f0e302f26220cfc5"
REFIT_RECORD = {
    "climbed_energy": 17.8033,
    "fitted_energy": 10.0647,
    "refit_energy": 17.8031,
    "margin": 0.02,
    "frames": 3,
    "frames_lowered": 3,
    "refused": True,
    "columns_dropped": 187,
    "outcome": "refit",
}
ACCEPTED_RECORD = {
    **REFIT_RECORD,
    "fitted_energy": 17.8031,
    "refit_energy": None,
    "frames_lowered": 0,
    "refused": False,
    "columns_dropped": 0,
    "outcome": "accepted",
}
KEPT_RECORD = {
    **REFIT_RECORD,
    "refit_energy": 10.0647,
    "columns_dropped": 0,
    "outcome": "kept_climb",
}


def a_guard(record: dict | None = None) -> LeastSquaresGuard:
    return LeastSquaresGuard(**(REFIT_RECORD if record is None else record))


def test_write_fit_writes_the_guard_record_and_it_survives_a_round_trip(tmp_path: Path) -> None:
    fit = example_fit(least_squares_guard=a_guard())
    path = write_fit(fit, tmp_path / "SYN1.json")
    assert json.loads(path.read_text())["least_squares_guard"] == REFIT_RECORD
    loaded = load_fit("SYN1", directory=tmp_path)
    assert loaded.least_squares_guard == a_guard()
    assert loaded.to_json() == fit.to_json()


@pytest.mark.parametrize("record", [ACCEPTED_RECORD, REFIT_RECORD, KEPT_RECORD])
def test_each_outcome_of_the_guard_is_a_valid_record(record: dict) -> None:
    data = example_fit(least_squares_guard=a_guard(record)).to_json()
    assert data["least_squares_guard"] == record
    assert CameraFit.from_json(data).least_squares_guard == a_guard(record)


def test_a_fit_without_a_guard_record_is_accepted_and_is_not_given_one(tmp_path: Path) -> None:
    fit = example_fit()
    assert fit.least_squares_guard is None
    data = fit.to_json()
    assert "least_squares_guard" not in data, "a record nobody made is not invented"
    again = CameraFit.from_json(data)
    assert again.least_squares_guard is None and again.to_json() == data
    assert json.loads(write_fit(fit, tmp_path / "SYN1.json").read_text()) == data


def test_a_guard_record_that_is_null_is_refused_rather_than_read_as_absent() -> None:
    data = example_fit().to_json()
    data["least_squares_guard"] = None
    with pytest.raises(FitFileError, match="least_squares_guard"):
        CameraFit.from_json(data)


def test_the_committed_whtm1_fit_predates_the_guard_and_is_byte_for_byte_unchanged(
    tmp_path: Path,
) -> None:
    assert hashlib.sha256(COMMITTED.read_bytes()).hexdigest() == WHTM1_SHA256, (
        "WHTM1.json is frozen: every observation cites its digest"
    )
    fit = load_fit("WHTM1")
    assert fit.least_squares_guard is None
    assert "least_squares_guard" not in COMMITTED.read_text()
    assert write_fit(fit, tmp_path / "WHTM1.json").read_bytes() == COMMITTED.read_bytes()


def test_the_guard_record_holds_no_coordinates_or_clock_times(tmp_path: Path) -> None:
    fit = example_fit(least_squares_guard=a_guard())
    assert_nothing_private(write_fit(fit, tmp_path / "SYN1.json").read_text())


def _with_record(record: dict, **changes):
    def edit(data: dict) -> None:
        data["least_squares_guard"] = {**record, **changes}

    return edit


def _without(record: dict, key: str):
    def edit(data: dict) -> None:
        data["least_squares_guard"] = {k: v for k, v in record.items() if k != key}

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(least_squares_guard=[]), "least_squares_guard"),
        (_with_record(REFIT_RECORD, outcome="rejected"), "outcome"),
        (_with_record(REFIT_RECORD, outcome=None), "outcome"),
        (_with_record(REFIT_RECORD, refused=False), "refused"),
        (_with_record(ACCEPTED_RECORD, refused=True), "refused"),
        (_with_record(REFIT_RECORD, refused="yes"), "refused"),
        (_with_record(REFIT_RECORD, climbed_energy=math.nan), "climbed_energy"),
        (_with_record(REFIT_RECORD, climbed_energy="high"), "climbed_energy"),
        (_with_record(REFIT_RECORD, climbed_energy=-1.0), "climbed_energy"),
        (_with_record(REFIT_RECORD, fitted_energy=True), "fitted_energy"),
        (_with_record(REFIT_RECORD, fitted_energy=math.inf), "fitted_energy"),
        (_with_record(REFIT_RECORD, refit_energy=None), "refit_energy"),
        (_with_record(REFIT_RECORD, refit_energy=math.nan), "refit_energy"),
        (_with_record(ACCEPTED_RECORD, refit_energy=3.0), "refit_energy"),
        (_with_record(REFIT_RECORD, margin=0.0), "margin"),
        (_with_record(REFIT_RECORD, margin=1.0), "margin"),
        (_with_record(REFIT_RECORD, margin=math.nan), "margin"),
        (_with_record(REFIT_RECORD, frames=0, frames_lowered=0), "frames"),
        (_with_record(REFIT_RECORD, frames=3.0), "frames"),
        (_with_record(REFIT_RECORD, frames_lowered=4), "frames_lowered"),
        (_with_record(REFIT_RECORD, frames_lowered=-1), "frames_lowered"),
        (_with_record(REFIT_RECORD, frames_lowered=1), "refused"),
        (_with_record(ACCEPTED_RECORD, frames_lowered=2), "refused"),
        (_with_record(REFIT_RECORD, columns_dropped=-1), "columns_dropped"),
        (_with_record(REFIT_RECORD, columns_dropped=12.5), "columns_dropped"),
        (_with_record(ACCEPTED_RECORD, columns_dropped=5), "columns_dropped"),
        (_without(REFIT_RECORD, "outcome"), "least_squares_guard.outcome"),
        (_without(REFIT_RECORD, "columns_dropped"), "least_squares_guard.columns_dropped"),
        (_without(REFIT_RECORD, "refit_energy"), "least_squares_guard.refit_energy"),
    ],
)
def test_a_guard_record_that_contradicts_itself_is_refused_with_the_field(
    tmp_path: Path, edit, message
) -> None:
    data = example_fit().to_json()
    edit(data)
    (tmp_path / "BAD1.json").write_text(json.dumps(data))
    with pytest.raises(FitFileError, match=message):
        load_fit("BAD1", directory=tmp_path)
