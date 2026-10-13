"""Reading `observed.json` files for the check: which frames are used, and what is refused.

An observed file is private input from another process, so every field the check reads is
checked for its type, and a file that is not WHTM1's, not schema version 1, or not named for its
own date is refused with a line that names the file and the field.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from test_renderer_check_support import frame_json, observed_document, write_observed, write_raw

from lastlight.renderer_check import observed

DAY = dt.date(2026, 10, 10)


def read_one(tmp_path, document) -> observed.ObservedEvening:
    path = write_raw(tmp_path, f"{DAY.isoformat()}.json", json.dumps(document))
    return observed.read_observed_file(path)


def refusal(tmp_path, document, name: str | None = None) -> observed.ObservedError:
    path = write_raw(tmp_path, name or f"{DAY.isoformat()}.json", json.dumps(document))
    with pytest.raises(observed.ObservedError) as caught:
        observed.read_observed_file(path)
    return caught.value


# --- which frames are used -----------------------------------------------------------------------


def test_a_frame_is_used_when_the_lowest_bands_log_bg_is_not_null(tmp_path) -> None:
    frames = [
        frame_json(3.0, -0.10, 0.02),
        frame_json(1.5, None, 0.03),
        frame_json(0.0, 0.05, None),
    ]
    evening = read_one(tmp_path, observed_document(DAY, frames))
    assert [f.sun_elevation_deg for f in evening.frames] == [3.0, 0.0]
    assert [f.log_bg for f in evening.frames] == [-0.10, 0.05]


def test_the_light_dome_flag_and_other_exclusions_do_not_exclude_a_frame(tmp_path) -> None:
    frames = [
        frame_json(3.0, -0.10, light_dome_excluded=True),
        frame_json(1.5, 0.20),
        frame_json(0.0, 0.30),
    ]
    frames[1]["reference"]["n_excluded"] = 9000
    frames[2]["bands"][1]["missing"] = "too_few_pixels"  # another band missing: still used
    evening = read_one(tmp_path, observed_document(DAY, frames))
    assert len(evening.frames) == 3


def test_only_band_zero_decides_whether_a_frame_is_used(tmp_path) -> None:
    frame = frame_json(3.0, None)
    frame["bands"][1]["log_bg"] = 0.4  # a higher band has a value; the lowest does not
    evening = read_one(tmp_path, observed_document(DAY, [frame]))
    assert evening.frames == ()


def test_the_log_rg_of_a_used_frame_may_be_null(tmp_path) -> None:
    evening = read_one(tmp_path, observed_document(DAY, [frame_json(3.0, 0.1, None)]))
    assert evening.frames[0].log_rg is None and evening.frames[0].log_bg == 0.1


def test_frames_found_and_used_counts_in_the_file_are_not_trusted(tmp_path) -> None:
    document = observed_document(DAY, [frame_json(3.0, 0.1)], frames_used=99, frames_found=100)
    assert len(read_one(tmp_path, document).frames) == 1


def test_integers_are_accepted_as_numbers(tmp_path) -> None:
    evening = read_one(tmp_path, observed_document(DAY, [frame_json(2, 0)]))
    assert evening.frames[0].sun_elevation_deg == 2.0 and evening.frames[0].log_bg == 0.0


def test_the_evening_carries_its_date_from_the_file(tmp_path) -> None:
    assert read_one(tmp_path, observed_document(DAY, [])).date == DAY


# --- the directory -------------------------------------------------------------------------------


def test_a_directory_is_read_in_date_order_and_other_names_are_ignored(tmp_path) -> None:
    write_observed(tmp_path, dt.date(2026, 10, 12), [(1.0, 0.1)])
    write_observed(tmp_path, dt.date(2026, 10, 10), [(1.0, 0.1)])
    write_observed(tmp_path, dt.date(2026, 10, 11), [(1.0, 0.1)])
    write_raw(tmp_path, "README.txt", "not an observed file")
    write_raw(tmp_path, "2026-10-13.json.bak", "{}")
    evenings = observed.read_observed_dir(tmp_path)
    assert [e.date for e in evenings] == [
        dt.date(2026, 10, 10),
        dt.date(2026, 10, 11),
        dt.date(2026, 10, 12),
    ]


def test_a_missing_directory_is_an_error(tmp_path) -> None:
    with pytest.raises(observed.ObservedError, match="not a directory"):
        observed.read_observed_dir(tmp_path / "nowhere")


def test_a_directory_with_no_observed_files_is_an_error(tmp_path) -> None:
    write_raw(tmp_path, "notes.txt", "x")
    with pytest.raises(observed.ObservedError, match="no YYYY-MM-DD.json"):
        observed.read_observed_dir(tmp_path)


# --- refusals: what is not WHTM1's, schema 1, or its own date ------------------------------------


def test_another_camera_is_refused_naming_the_file_and_the_field(tmp_path) -> None:
    error = refusal(tmp_path, observed_document(DAY, [], camera_id="ELSE1"))
    message = str(error)
    assert "2026-10-10.json" in message and "camera_id" in message
    assert "\n" not in message


def test_another_schema_version_is_refused(tmp_path) -> None:
    for version in (2, 0, "1", 1.0, None, True):
        error = refusal(tmp_path, observed_document(DAY, [], schema_version=version))
        assert "2026-10-10.json" in str(error) and "schema_version" in str(error)


def test_a_date_that_disagrees_with_the_file_name_is_refused(tmp_path) -> None:
    document = observed_document(dt.date(2026, 10, 11), [frame_json(1.0, 0.1)])
    error = refusal(tmp_path, document, name="2026-10-10.json")
    assert "2026-10-10.json" in str(error) and "date" in str(error)


@pytest.mark.parametrize("date", [None, 20261010, "2026-10-1", "10 October 2026", ""])
def test_a_date_field_that_is_not_an_iso_date_is_refused(tmp_path, date) -> None:
    error = refusal(tmp_path, observed_document(DAY, [], date=date))
    assert "date" in str(error)


def test_a_file_that_is_not_json_is_refused_naming_the_file(tmp_path) -> None:
    path = write_raw(tmp_path, "2026-10-10.json", "{not json")
    with pytest.raises(observed.ObservedError, match="2026-10-10.json"):
        observed.read_observed_file(path)


def test_bytes_that_are_not_utf8_are_decoded_with_replacement_not_crashed(tmp_path) -> None:
    path = tmp_path / "2026-10-10.json"
    document = json.dumps(observed_document(DAY, [frame_json(1.0, 0.1)]))
    path.write_bytes(document.encode().replace(b"WHTM1", b"WH\xff\xfeTM1"))
    with pytest.raises(observed.ObservedError, match="camera_id"):
        observed.read_observed_file(path)
    path.write_bytes(document.encode() + b"\n")
    assert len(observed.read_observed_file(path).frames) == 1


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ([], "(file)"),
        ({"schema_version": 1}, "camera_id"),
        (observed_document(DAY, []) | {"frames": "no"}, "frames"),
        (observed_document(DAY, []) | {"frames": None}, "frames"),
        (observed_document(DAY, [3]), "frames[0]"),
        (observed_document(DAY, [{"sun_elevation_deg": 1.0}]), "frames[0].bands"),
        (observed_document(DAY, [{"sun_elevation_deg": 1.0, "bands": []}]), "frames[0].bands"),
        (observed_document(DAY, [{"sun_elevation_deg": 1.0, "bands": [4]}]), "bands[0]"),
        (observed_document(DAY, [{"sun_elevation_deg": 1.0, "bands": [{}]}]), "log_bg"),
    ],
)
def test_a_document_of_the_wrong_shape_is_refused_naming_the_field(
    tmp_path, document, field: str
) -> None:
    assert field in str(refusal(tmp_path, document))


@pytest.mark.parametrize("bad", ["0.1", True, [0.1], {"v": 1}, float("nan"), float("inf")])
def test_a_log_bg_that_is_not_a_finite_number_is_refused(tmp_path, bad) -> None:
    frame = frame_json(1.0, 0.1)
    frame["bands"][0]["log_bg"] = bad
    text = json.dumps(observed_document(DAY, [frame]), allow_nan=True)
    path = write_raw(tmp_path, "2026-10-10.json", text)
    with pytest.raises(observed.ObservedError) as caught:
        observed.read_observed_file(path)
    assert "log_bg" in str(caught.value) and "2026-10-10.json" in str(caught.value)


@pytest.mark.parametrize("bad", ["1.0", True, None, [1.0], float("nan"), float("-inf")])
def test_a_sun_elevation_that_is_not_a_finite_number_is_refused(tmp_path, bad) -> None:
    frame = frame_json(1.0, 0.1)
    frame["sun_elevation_deg"] = bad
    path = write_raw(tmp_path, "2026-10-10.json", json.dumps(observed_document(DAY, [frame])))
    with pytest.raises(observed.ObservedError, match="sun_elevation_deg"):
        observed.read_observed_file(path)


def test_a_huge_number_is_refused_rather_than_overflowing(tmp_path) -> None:
    text = json.dumps(observed_document(DAY, [frame_json(1.0, 0.1)]))
    text = text.replace('"log_bg": 0.1', '"log_bg": 1' + "0" * 400)
    path = write_raw(tmp_path, "2026-10-10.json", text)
    with pytest.raises(observed.ObservedError, match="log_bg"):
        observed.read_observed_file(path)


def test_a_literal_beyond_the_float_range_is_refused(tmp_path) -> None:
    text = json.dumps(observed_document(DAY, [frame_json(1.0, 0.1)]))
    path = write_raw(tmp_path, "2026-10-10.json", text.replace('"log_bg": 0.1', '"log_bg": 1e999'))
    with pytest.raises(observed.ObservedError, match="log_bg"):
        observed.read_observed_file(path)


def test_an_unused_frames_other_fields_are_not_trusted_or_required(tmp_path) -> None:
    frame = {"bands": [{"log_bg": None}], "sun_elevation_deg": "garbage"}
    evening = read_one(tmp_path, observed_document(DAY, [frame, frame_json(1.0, 0.1)]))
    assert len(evening.frames) == 1


def test_a_refused_file_in_a_directory_refuses_the_directory(tmp_path) -> None:
    write_observed(tmp_path, dt.date(2026, 10, 10), [(1.0, 0.1)])
    write_observed(tmp_path, dt.date(2026, 10, 11), [(1.0, 0.1)], camera_id="OTHER")
    with pytest.raises(observed.ObservedError, match="2026-10-11.json"):
        observed.read_observed_dir(tmp_path)
