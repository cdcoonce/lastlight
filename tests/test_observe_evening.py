"""An evening's frames to `observed.json`: the window, the drops, and what the file may hold.

The frames are synthetic JPEGs written into a throwaway archive under the names the collector
gives (`frame_stem`). `observe_evening` must read each frame's instant from that name and compute
the sun's elevation itself at the camera's registry position; a sidecar is never consulted.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pytest
from observe_archive import EVENING, fill_archive, instant_at_sun, whtm1, write_jpeg
from observe_support import grey_frame, to_uint8

from lastlight.cameras import ContrastBands, load_registry
from lastlight.cameras.archive import EveningArchive, frame_stem
from lastlight.camgeom.fitfile import GEOMETRY_DIR
from lastlight.camgeom.frames import parse_frame_time
from lastlight.geometry import solar_position
from lastlight.observe.evening import (
    DROP_REASONS,
    SCHEMA_VERSION,
    ObserveError,
    observe_evening,
    write_observed,
)
from lastlight.privacy import scan

SWEEP = [8.0, 5.0, 0.0, -2.0, -5.0, -7.0]


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    fill_archive(tmp_path / "frames", SWEEP)
    return tmp_path / "frames"


def observe(archive: Path, **kwargs):
    return observe_evening(whtm1(), archive=archive, date=EVENING, **kwargs)


def reasons(observation) -> dict[str, int]:
    return {reason: n for reason, n in observation.dropped.items() if n}


# --- which frames are kept ------------------------------------------------------------------------


def test_the_default_window_keeps_sun_plus_6_to_minus_6_and_counts_the_rest_by_reason(
    archive: Path,
) -> None:
    result = observe(archive)
    assert result.frames_found == 6
    assert [round(f.sun_elevation_deg) for f in result.frames] == [5, 0, -2, -5]
    assert reasons(result) == {"sun_above_window": 1, "sun_below_window": 1}


def test_min_and_max_sun_elevation_override_the_window(archive: Path) -> None:
    result = observe(archive, min_sun_elev=-8.0, max_sun_elev=1.0)
    assert [round(f.sun_elevation_deg) for f in result.frames] == [0, -2, -5, -7]
    assert reasons(result) == {"sun_above_window": 2}


def test_the_window_edges_are_inclusive(archive: Path) -> None:
    sun = solar_position(instant_at_sun(0.0), whtm1().position.lat, whtm1().position.lon)
    result = observe(archive, min_sun_elev=sun.elevation_deg, max_sun_elev=sun.elevation_deg)
    assert len(result.frames) == 1


def test_an_inverted_window_is_refused(archive: Path) -> None:
    with pytest.raises(ObserveError, match="window"):
        observe(archive, min_sun_elev=3.0, max_sun_elev=-3.0)


def test_every_drop_reason_is_always_present_so_a_zero_is_a_recorded_zero(archive: Path) -> None:
    assert set(observe(archive).dropped) == set(DROP_REASONS)


# --- the sun elevation is computed, from the name, at the registry position ----------------------


def test_sun_elevation_is_computed_from_the_time_in_the_name_at_the_registry_position(
    archive: Path,
) -> None:
    camera = whtm1()
    names = (archive / "WHTM1" / EVENING.isoformat()).glob("*.jpg")
    expected = sorted(
        (
            solar_position(
                parse_frame_time(p), camera.position.lat, camera.position.lon
            ).elevation_deg
            for p in names
        ),
        reverse=True,
    )
    result = observe(archive, min_sun_elev=-10.0, max_sun_elev=10.0)
    assert [f.sun_elevation_deg for f in result.frames] == pytest.approx(expected, abs=1e-9)
    assert len(result.frames) == 6


def test_the_collectors_own_naming_code_and_the_reader_agree() -> None:
    instant = dt.datetime(2026, 10, 11, 1, 2, 3, tzinfo=dt.UTC)
    assert parse_frame_time(f"{frame_stem(instant)}.jpg") == instant


def test_a_sidecar_is_never_trusted_for_the_sun_elevation(archive: Path) -> None:
    folder = archive / "WHTM1" / EVENING.isoformat()
    for jpg in folder.glob("*.jpg"):
        jpg.with_suffix(".json").write_text(json.dumps({"sun_elevation_deg": 40.0}))
    result = observe(archive)
    assert [round(f.sun_elevation_deg) for f in result.frames] == [5, 0, -2, -5]


def test_frames_come_out_in_the_order_they_were_taken(archive: Path) -> None:
    elevations = [f.sun_elevation_deg for f in observe(archive).frames]
    assert elevations == sorted(elevations, reverse=True)  # the sun is setting


# --- frames that cannot be used ------------------------------------------------------------------


def test_a_frame_with_no_collector_time_in_its_name_is_dropped_and_counted(archive: Path) -> None:
    folder = archive / "WHTM1" / EVENING.isoformat()
    write_jpeg(folder / "snapshot.jpg", to_uint8(grey_frame()))
    result = observe(archive)
    assert result.dropped["no_time_in_name"] == 1
    assert result.frames_found == 7
    assert len(result.frames) == 4


def test_a_frame_that_will_not_decode_is_dropped_and_counted(archive: Path) -> None:
    folder = archive / "WHTM1" / EVENING.isoformat()
    stem = frame_stem(instant_at_sun(-2.0))
    (folder / f"{stem}.jpg").write_bytes(b"not a jpeg")  # overwrite the frame taken at -2
    result = observe(archive)
    assert result.dropped["unreadable"] == 1
    assert [round(f.sun_elevation_deg) for f in result.frames] == [5, 0, -5]


def test_a_frame_of_the_wrong_size_is_dropped_and_counted_not_resized(archive: Path) -> None:
    folder = archive / "WHTM1" / EVENING.isoformat()
    stem = frame_stem(instant_at_sun(-2.0))
    write_jpeg(folder / f"{stem}.jpg", to_uint8(grey_frame())[:150, :400])
    result = observe(archive)
    assert result.dropped["wrong_size"] == 1
    assert len(result.frames) == 3


def test_an_evening_folder_with_no_frame_is_an_observation_with_none(tmp_path: Path) -> None:
    EveningArchive(tmp_path, "WHTM1", EVENING).directory.mkdir(parents=True)
    result = observe(tmp_path)
    assert result.frames == () and result.frames_found == 0


# --- refusals: bad input and bad configuration ---------------------------------------------------


def test_a_missing_evening_folder_is_refused_and_says_which(tmp_path: Path) -> None:
    with pytest.raises(ObserveError, match=r"WHTM1.*2026-10-10"):
        observe(tmp_path)


def test_a_camera_without_a_contrast_table_is_refused_with_the_reason(archive: Path) -> None:
    esmo1 = load_registry()["ESMO1"]
    with pytest.raises(ObserveError, match=r"ESMO1.*contrast table"):
        observe_evening(esmo1, archive=archive, date=EVENING)


def test_a_camera_with_no_frozen_fit_is_refused(archive: Path, tmp_path: Path) -> None:
    with pytest.raises(ObserveError, match=r"WHTM1.*geometry"):
        observe(archive, geometry_dir=tmp_path / "no-fits-here")


def test_a_contrast_table_that_does_not_fit_the_frozen_fit_is_refused_before_any_frame_is_read(
    archive: Path,
) -> None:
    bad = ContrastBands(8, ((8, 28),), (448, 468))
    camera = dataclasses.replace(whtm1(), contrast=bad)
    with pytest.raises(ObserveError, match=r"reference.*\(448, 468\)"):
        observe_evening(camera, archive=archive, date=EVENING)


# --- what the document says ----------------------------------------------------------------------


def test_the_document_names_the_geometry_it_used(archive: Path) -> None:
    document = observe(archive).to_json()
    fit_bytes = (GEOMETRY_DIR / "WHTM1.json").read_bytes()
    assert document["schema_version"] == SCHEMA_VERSION == 1
    assert document["camera_id"] == "WHTM1"
    assert document["date"] == "2026-10-10"
    assert document["geometry"] == {
        "fit_sha256": hashlib.sha256(fit_bytes).hexdigest(),
        "provisional": json.loads(fit_bytes)["provisional"],
    }
    assert document["geometry"]["provisional"] is True  # the sun has not confirmed the scale


def test_each_frame_is_identified_by_a_sha256_prefix_and_its_sun_elevation(archive: Path) -> None:
    result = observe(archive)
    frames = sorted((archive / "WHTM1" / EVENING.isoformat()).glob("*.jpg"))
    digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in frames}
    for frame in result.frames:
        assert len(frame.sha256_prefix) == 12
        assert any(d.startswith(frame.sha256_prefix) for d in digests)
    assert len({f.sha256_prefix for f in result.frames}) == len(result.frames)


def test_each_frame_has_a_five_band_profile_with_the_expected_contrast(archive: Path) -> None:
    for frame in observe(archive).frames:
        first = frame.profile.bands[0]
        assert len(frame.profile.bands) == 5
        assert first.log_rg == pytest.approx(0.22, abs=0.03)  # the synthetic gradient's low band
        assert first.log_bg == pytest.approx(-0.165, abs=0.03)


def test_exposure_and_white_balance_changes_between_frames_do_not_move_the_contrasts(
    tmp_path: Path,
) -> None:
    gains = [(1.0, 1.0, 1.0), (1.8, 1.0, 0.6), (0.7, 1.2, 1.9), (1.2, 0.5, 1.0)]
    fill_archive(tmp_path, [5.0, 0.0, -2.0, -5.0], gains=gains)
    result = observe(tmp_path)
    reference = result.frames[0].profile.bands
    for frame in result.frames[1:]:
        for want, got in zip(reference, frame.profile.bands, strict=True):
            # Whole 8-bit codes and JPEG's own rounding, at the low codes a gain of 0.6 gives.
            assert got.log_rg == pytest.approx(want.log_rg, abs=0.05)
            assert got.log_bg == pytest.approx(want.log_bg, abs=0.05)


def test_the_light_dome_is_excluded_only_in_frames_below_minus_4(archive: Path) -> None:
    result = observe(archive)
    dome = {round(f.sun_elevation_deg): f.profile.light_dome_excluded for f in result.frames}
    assert dome == {5: False, 0: False, -2: False, -5: True}
    last = result.frames[-1].profile
    assert all(band.excluded.light_dome == 270 * 20 for band in last.bands)  # columns 530 to 799


def test_excluded_pixels_are_totalled_per_band_across_the_frames_used(archive: Path) -> None:
    document = observe(archive).to_json()
    by_band = document["excluded_pixels"]["bands"]
    assert [entry["rows"] for entry in by_band] == [
        [8, 28],
        [28, 48],
        [48, 68],
        [68, 88],
        [88, 108],
    ]
    for entry in by_band:
        assert entry["excluded"]["light_dome"] == 270 * 20  # one frame of the four is below -4
        assert entry["n_excluded"] == sum(entry["excluded"].values())
    assert document["excluded_pixels"]["reference"]["excluded"]["light_dome"] == 270 * 20


def test_the_definition_of_the_bands_is_echoed_so_the_file_describes_itself(archive: Path) -> None:
    definition = observe(archive).to_json()["contrast_definition"]
    assert definition["bands_rows"] == [[8, 28], [28, 48], [48, 68], [68, 88], [88, 108]]
    assert definition["reference_rows"] == [118, 138]
    assert definition["skyline_margin_rows"] == 8
    assert definition["light_dome_columns"] == [[530, 800]]
    assert definition["light_dome_below_sun_elevation_deg"] == -4.0
    assert "pixel rows above" in definition["unit"]


def test_the_window_and_the_counts_are_recorded(archive: Path) -> None:
    document = observe(archive).to_json()
    assert document["sun_window_deg"] == {"max": 6.0, "min": -6.0}
    assert document["frames_found"] == 6
    assert document["frames_used"] == 4
    assert document["frames_dropped"]["sun_above_window"] == 1
    assert document["frames_dropped"]["sun_below_window"] == 1


def test_the_view_elevation_is_labelled_nominal_in_the_document(archive: Path) -> None:
    document = observe(archive).to_json()
    band = document["frames"][0]["bands"][0]
    assert set(band["view_elevation_deg"]) == {"nominal", "range"}
    assert "nominal" in document["view_elevation_note"]


# --- the file is derived numbers only ------------------------------------------------------------


def test_the_written_file_passes_the_privacy_scanner(archive: Path, tmp_path: Path) -> None:
    out = write_observed(observe(archive), tmp_path / "out" / "observed.json")
    result = scan([out], root=tmp_path)
    assert [finding.format() for finding in result.findings] == []
    assert result.files_scanned == 1


def test_the_written_file_has_no_file_name_time_or_pixel_array(
    archive: Path, tmp_path: Path
) -> None:
    out = write_observed(observe(archive), tmp_path / "observed.json")
    text = out.read_text()
    document = json.loads(text)

    folder = archive / "WHTM1" / EVENING.isoformat()
    for jpg in folder.glob("*.jpg"):
        assert jpg.stem not in text and jpg.name not in text
    assert ".jpg" not in text
    assert not re.search(r"\d{1,2}:\d{2}", text)  # no clock time
    assert not re.search(r"\d{8}T\d{4}", text)  # no collector stamp
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d", text)  # no ISO instant, only the date

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(item, f"{path}/{key}")
        elif isinstance(value, list):
            assert len(value) <= 12, f"{path} is a list of {len(value)}: a pixel array?"
            for index, item in enumerate(value):
                yield from walk(item, f"{path}[{index}]")
        else:
            yield path, value

    leaves = dict(walk(document))
    assert leaves, "the walk must reach the leaves"
    prose = {"/view_elevation_note", "/contrast_definition/unit"}  # the two sentences in the file
    long_strings = [k for k, v in leaves.items() if isinstance(v, str) and len(v) > 80]
    assert set(long_strings) <= prose, long_strings


def test_the_file_is_deterministic_and_ends_with_a_newline(archive: Path, tmp_path: Path) -> None:
    first = write_observed(observe(archive), tmp_path / "a.json").read_text()
    second = write_observed(observe(archive), tmp_path / "b.json").read_text()
    assert first == second and first.endswith("\n")
    assert json.loads(first)["camera_id"] == "WHTM1"


def test_write_observed_makes_the_parent_directories(archive: Path, tmp_path: Path) -> None:
    out = write_observed(observe(archive), tmp_path / "deep" / "er" / "observed.json")
    assert out.is_file()
