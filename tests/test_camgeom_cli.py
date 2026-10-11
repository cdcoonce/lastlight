"""`lastlight camgeom fit` and `lastlight camgeom mask`, end to end and offline."""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import (
    REF_LAT,
    REF_LON,
    TRUE_MODEL,
    paint_disk,
    render_frame,
    ridge_dem,
    write_geotiff,
)
from PIL import Image

from lastlight.camgeom.dem import horizon_profile, tile_name
from lastlight.camgeom.fitfile import load_fit
from lastlight.camgeom.frames import frame_sha256
from lastlight.camgeom.skyline import default_overlay_boxes
from lastlight.camgeom.suncheck import predict_sun_pixel
from lastlight.cli import main
from lastlight.commands.camgeom import COMMAND

STAMP = "%Y%m%dT%H%M%SZ"
EVENING = dt.datetime(2026, 10, 11, 0, 30, tzinfo=dt.UTC)
MOUNT_HEIGHT_M = 10.0
GROUND_M = 300.0


@pytest.fixture(scope="module")
def dem_tif(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_geotiff(tmp_path_factory.mktemp("dem") / "ridge.tif", ridge_dem())


@pytest.fixture(scope="module")
def horizon_fn():
    """The skyline the DEM predicts for the synthetic camera, as the pipeline will compute it."""
    azimuths = np.arange(180.0, 360.0, 0.05)
    elevation = horizon_profile(ridge_dem(), REF_LAT, REF_LON, GROUND_M + MOUNT_HEIGHT_M, azimuths)
    return lambda az: np.interp(az, azimuths, elevation)


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory, horizon_fn) -> Path:
    root = tmp_path_factory.mktemp("archive") / "SYN1"
    sun = predict_sun_pixel(TRUE_MODEL, EVENING)
    assert sun is not None
    times = [EVENING + dt.timedelta(minutes=5 * k) for k in range(3)]
    times.append(EVENING + dt.timedelta(hours=3, minutes=30))  # after dark: must be left out
    for index, when in enumerate(times):
        image = render_frame(
            TRUE_MODEL, horizon_fn, seed=index, stamp=default_overlay_boxes(800, 300)[0]
        )
        if index == 0:
            image = paint_disk(image, sun.x + 2.0, sun.y)  # the sun, a fifth of a degree off
        path = root / "2026-10-10" / f"{when.strftime(STAMP)}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path, quality=92)
    Image.fromarray(np.zeros((300, 800, 3), dtype=np.uint8)).save(root / "unlabelled.jpg")
    return root


@pytest.fixture(scope="module")
def site_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("site") / "sites.toml"
    for camera in ("SYN1", "WHTM1"):
        with path.open("a") as out:
            out.write(
                f"[{camera}]\nlat = {REF_LAT}\nlon = {REF_LON}\n"
                f"mount_height_m = {MOUNT_HEIGHT_M}\nview_azimuth_deg = 272.0\n"
            )
    return path


def small_archive(root: Path, horizon_fn, *, disk_offset_px: float | None) -> Path:
    """Two daytime frames; the first holds a sun disk `disk_offset_px` from where it belongs."""
    sun = predict_sun_pixel(TRUE_MODEL, EVENING)
    assert sun is not None
    for index in range(2):
        when = EVENING + dt.timedelta(minutes=5 * index)
        image = render_frame(
            TRUE_MODEL, horizon_fn, seed=20 + index, stamp=default_overlay_boxes(800, 300)[0]
        )
        if index == 0 and disk_offset_px is not None:
            image = paint_disk(image, sun.x + disk_offset_px, sun.y)
        path = root / "2026-10-10" / f"{when.strftime(STAMP)}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path, quality=92)
    return root


def run(*argv: str) -> int:
    return main(["camgeom", *argv], commands=[COMMAND])


# --- fit  -----------------------------------------------------------------------------------------


class Fitted:
    """One real run of `camgeom fit`, shared by the tests that read its results."""

    def __init__(self, out_dir: Path, code: int, stdout: str, stderr: str) -> None:
        self.out_dir, self.code, self.stdout, self.stderr = out_dir, code, stdout, stderr

    @property
    def path(self) -> Path:
        return self.out_dir / "SYN1.json"


@pytest.fixture(scope="module")
def fitted(
    tmp_path_factory: pytest.TempPathFactory,
    archive: Path,
    dem_tif: Path,
    site_file: Path,
) -> Fitted:
    out_dir = tmp_path_factory.mktemp("geometry")
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = run(
            "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
            "--dem", str(dem_tif), "--out", str(out_dir / "SYN1.json"), "--frozen-on", "2026-10-10",
        )  # fmt: skip
    return Fitted(out_dir, code, stdout.getvalue(), stderr.getvalue())


def test_fit_writes_a_frozen_fit_that_recovers_the_cameras_pointing(fitted: Fitted) -> None:
    assert fitted.code == 0, fitted.stderr
    fit = load_fit("SYN1", directory=fitted.out_dir)
    o = fit.orientation
    assert o.azimuth_deg == pytest.approx(TRUE_MODEL.azimuth_deg, abs=0.3)
    assert o.elevation_deg == pytest.approx(TRUE_MODEL.elevation_deg, abs=0.3)
    assert o.hfov_deg == pytest.approx(TRUE_MODEL.hfov_deg, rel=0.015)
    assert fit.frozen_on == "2026-10-10"
    assert fit.skyline_rms_px < 1.5
    assert (fit.width, fit.height) == (800, 300)
    assert "SYN1" in fitted.stdout and "rms" in fitted.stdout and "px" in fitted.stdout


def test_the_frames_used_are_the_daytime_ones_and_are_identified_by_their_hash(
    fitted: Fitted, archive: Path
) -> None:
    cutoff = EVENING + dt.timedelta(hours=1)
    daytime = sorted(
        frame_sha256(path)
        for path in (archive / "2026-10-10").glob("*.jpg")
        if dt.datetime.strptime(path.stem, STAMP).replace(tzinfo=dt.UTC) < cutoff
    )
    fit = load_fit("SYN1", directory=fitted.out_dir)
    assert sorted(record.sha256 for record in fit.frames_used) == daytime
    assert len(daytime) == 3
    assert all(record.evening == "2026-10-10" for record in fit.frames_used)
    assert all(record.sun_elevation_deg > 0.0 for record in fit.frames_used)
    assert "1 frame skipped" in fitted.stdout and "1 without a time" in fitted.stdout


def test_the_sun_check_runs_on_the_frames_and_passes_when_the_disk_is_where_it_should_be(
    fitted: Fitted,
) -> None:
    check = load_fit("SYN1", directory=fitted.out_dir).sun_check
    assert check["status"] == "passed"
    assert check["detected"] == 1 and check["frames_examined"] == 3
    assert check["max_error_deg"] < 1.0


def test_the_written_file_carries_no_coordinates_or_clock_times(fitted: Fitted) -> None:
    text = fitted.path.read_text()
    data = json.loads(text)
    assert "lat" not in data and "position" not in data
    assert str(REF_LAT) not in text and str(REF_LON) not in text
    assert not re.search(r"(?<![\d.])\d{1,2}:\d{2}(?!\d)", text)
    assert data["radial_term"]["adopted"] is False
    assert data["radial_term"]["at_bound"] is False
    assert "position_basis" in data


def test_fit_reads_the_dem_tile_from_the_cache_and_never_the_network(
    tmp_path: Path, archive: Path, dem_tif: Path, site_file: Path
) -> None:
    cache = tmp_path / "dem-cache"
    cache.mkdir()
    (cache / f"{tile_name(33, -113)}.tif").write_bytes(dem_tif.read_bytes())
    out = tmp_path / "SYN1.json"
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem-cache", str(cache), "--out", str(out), "--frozen-on", "2026-10-10",
    )  # fmt: skip
    assert code == 0
    assert load_fit("SYN1", directory=tmp_path).orientation.azimuth_deg == pytest.approx(
        TRUE_MODEL.azimuth_deg, abs=0.3
    )


def test_a_position_search_that_cannot_move_still_records_itself(
    tmp_path: Path, archive: Path, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture
) -> None:
    # A box smaller than the step searches only the site itself: the whole path is exercised
    # (flags, search, summary, file) for the price of one extra horizon.
    out = tmp_path / "SYN1.json"
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(out), "--frozen-on", "2026-10-10",
        "--position-box-km", "1", "--position-step-km", "2",
    )  # fmt: skip
    assert code == 0
    search = load_fit("SYN1", directory=tmp_path).position_search
    assert search is not None
    assert search["positions"] == 1 and search["adopted"] is True
    assert (search["offset_east_km"], search["offset_north_km"]) == (0.0, 0.0)
    assert search["rms_px_at_site"] == search["rms_px_best"]
    assert "position search" in capsys.readouterr().out


def test_without_the_search_flags_no_search_is_recorded(fitted: Fitted) -> None:
    assert load_fit("SYN1", directory=fitted.out_dir).position_search is None


def test_a_fit_whose_sun_check_passed_is_not_provisional(fitted: Fitted) -> None:
    fit = load_fit("SYN1", directory=fitted.out_dir)
    assert fit.sun_check["status"] == "passed" and fit.provisional is False
    assert "PROVISIONAL" not in fitted.stdout


def test_a_skipped_sun_check_exits_0_and_marks_the_fit_provisional(
    tmp_path: Path, horizon_fn, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture
) -> None:
    frames = small_archive(tmp_path / "frames", horizon_fn, disk_offset_px=None)
    out = tmp_path / "SYN1.json"
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(frames), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(out), "--frozen-on", "2026-10-10",
    )  # fmt: skip
    assert code == 0
    fit = load_fit("SYN1", directory=tmp_path)
    assert fit.sun_check["status"] == "skipped"
    assert fit.provisional is True
    assert "PROVISIONAL" in capsys.readouterr().out


def test_a_failed_sun_check_exits_1_and_still_writes_the_fit_for_inspection(
    tmp_path: Path, horizon_fn, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture
) -> None:
    # The sun's disk lies 35 pixels (about 3 degrees) from where the fit says the sun is.
    frames = small_archive(tmp_path / "frames", horizon_fn, disk_offset_px=35.0)
    out = tmp_path / "SYN1.json"
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(frames), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(out), "--frozen-on", "2026-10-10",
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 1
    assert out.exists(), "the file is written so that it can be inspected"
    fit = load_fit("SYN1", directory=tmp_path)
    assert fit.sun_check["status"] == "failed" and fit.provisional is True
    assert "sun check FAILED" in captured.err and str(out) in captured.err
    assert "failed" in captured.out


def test_the_fit_records_how_far_it_landed_from_the_direction_the_operator_states(
    tmp_path: Path, archive: Path, dem_tif: Path, site_file: Path
) -> None:
    out = tmp_path / "WHTM1.json"
    code = run(
        "fit", "--camera", "WHTM1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(out), "--frozen-on", "2026-10-10",
    )  # fmt: skip
    assert code == 0
    fit = load_fit("WHTM1", directory=tmp_path)
    assert fit.stated_azimuth_deg == 274.0, "the registry's direction for WHTM1"
    assert fit.azimuth_minus_stated_deg == pytest.approx(
        fit.orientation.azimuth_deg - 274.0, abs=0.01
    )


def test_a_camera_the_registry_does_not_list_has_no_stated_azimuth(fitted: Fitted) -> None:
    fit = load_fit("SYN1", directory=fitted.out_dir)
    assert fit.stated_azimuth_deg is None and fit.azimuth_minus_stated_deg is None


def test_fit_refuses_when_no_frame_is_bright_enough_and_names_the_threshold(
    tmp_path: Path, archive: Path, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture
) -> None:
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(tmp_path / "x.json"), "--min-sun-elevation", "40",
    )  # fmt: skip
    assert code == 2
    assert "40" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--camera", "NOPE1"], "NOPE1"),
        (["--frames", "{tmp}/missing"], "frames"),
        (["--site-file", "{tmp}/missing.toml"], "missing.toml"),
        (["--dem", "{tmp}/missing.tif"], "missing.tif"),
    ],
)
def test_fit_exits_2_with_a_reason_when_an_input_is_missing(
    tmp_path: Path,
    archive: Path,
    dem_tif: Path,
    site_file: Path,
    capsys: pytest.CaptureFixture,
    argv: list[str],
    message: str,
) -> None:
    base = {
        "--camera": "SYN1",
        "--frames": str(archive),
        "--site-file": str(site_file),
        "--dem": str(dem_tif),
        "--out": str(tmp_path / "x.json"),
    }
    for flag, value in zip(argv[::2], argv[1::2], strict=True):
        base[flag] = value.replace("{tmp}", str(tmp_path))
    code = run("fit", *[part for pair in base.items() for part in pair])
    assert code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def _bad_frames(tmp_path: Path, kind: str) -> Path:
    frames = tmp_path / "frames"
    good = frames / "2026-10-10"
    good.mkdir(parents=True)
    when = EVENING + dt.timedelta(minutes=1)
    path = good / f"{when.strftime(STAMP)}.jpg"
    if kind == "truncated":
        noise = np.random.default_rng(0).integers(0, 256, (300, 800, 3), dtype=np.uint8)
        Image.fromarray(noise).save(path, quality=95)
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 3])
    elif kind == "garbage":
        path.write_bytes(b"this is not a picture, whatever the name says" * 50)
    else:
        path.write_bytes(b"")
    return frames


@pytest.mark.parametrize("kind", ["truncated", "garbage", "empty"])
def test_a_frame_that_cannot_be_decoded_exits_2_with_a_one_line_reason(
    tmp_path: Path, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture, kind: str
) -> None:
    frames = _bad_frames(tmp_path, kind)
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(frames), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(tmp_path / "x.json"),
    )  # fmt: skip
    err = capsys.readouterr().err
    assert code == 2
    assert "not a readable JPEG" in err
    assert err.strip().count("\n") == 0 and "Traceback" not in err
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("kind", ["not_a_tiff", "tiff_header_only"])
def test_a_corrupt_dem_tile_exits_2_with_a_one_line_reason(
    tmp_path: Path, archive: Path, site_file: Path, capsys: pytest.CaptureFixture, kind: str
) -> None:
    tile = tmp_path / "corrupt.tif"
    tile.write_bytes((b"II*\x00" if kind == "tiff_header_only" else b"junk") + b"\x07" * 6000)
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem", str(tile), "--out", str(tmp_path / "x.json"),
    )  # fmt: skip
    err = capsys.readouterr().err
    assert code == 2
    assert "corrupt.tif" in err
    assert err.strip().count("\n") == 0 and "Traceback" not in err
    assert not (tmp_path / "x.json").exists()


def test_a_fit_that_cannot_write_its_file_exits_2_with_a_one_line_reason(
    tmp_path: Path, archive: Path, dem_tif: Path, site_file: Path, capsys: pytest.CaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    code = run(
        "fit", "--camera", "SYN1", "--frames", str(archive), "--site-file", str(site_file),
        "--dem", str(dem_tif), "--out", str(blocker / "SYN1.json"),
    )  # fmt: skip
    err = capsys.readouterr().err
    assert code == 2 and "blocker" in err
    assert err.strip().count("\n") == 0 and "Traceback" not in err


def test_fit_without_a_site_file_says_which_environment_variable_would_do(
    tmp_path: Path, archive: Path, dem_tif: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:  # fmt: skip
    monkeypatch.delenv("LASTLIGHT_CAMERA_SITES", raising=False)
    code = run("fit", "--camera", "SYN1", "--frames", str(archive), "--dem", str(dem_tif))
    assert code == 2
    assert "LASTLIGHT_CAMERA_SITES" in capsys.readouterr().err


def test_camgeom_without_an_action_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    assert run() == 2
    assert "fit" in capsys.readouterr().err


# --- holdout  ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def other_frames(tmp_path_factory: pytest.TempPathFactory, horizon_fn, archive: Path) -> Path:
    """Two frames the fit never saw, and a copy of one it was frozen on."""
    root = tmp_path_factory.mktemp("held-out")
    for index, minutes in enumerate((12, 17)):
        when = EVENING + dt.timedelta(minutes=minutes)
        image = render_frame(
            TRUE_MODEL, horizon_fn, seed=60 + index, stamp=default_overlay_boxes(800, 300)[0]
        )
        Image.fromarray(image).save(root / f"{when.strftime(STAMP)}.jpg", quality=92)
    used = sorted((archive / "2026-10-10").glob("*.jpg"))[0]
    (root / "copy-of-a-fitted-frame.jpg").write_bytes(used.read_bytes())
    return root


def test_holdout_prints_one_line_per_unused_frame_and_the_band(
    fitted: Fitted, other_frames: Path, capsys: pytest.CaptureFixture
) -> None:
    code = run(
        "holdout", "--camera", "SYN1", "--fit", str(fitted.path), "--frames", str(other_frames),
        "--band", "12",
    )  # fmt: skip
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    frame_lines = [line for line in lines if "held-out frame" in line and "sha256" in line]
    assert len(frame_lines) == 2, lines
    for line in frame_lines:
        assert re.search(r"sha256 [0-9a-f]{8}\)", line)
        rms = float(re.search(r"rms ([\d.]+) px", line).group(1))
        assert rms < 1.5, "the frozen skyline follows the ridge in frames it was not fitted on"
    assert "+-12 rows" in lines[0] and "1 frame used in the fit left out" in lines[0]
    assert "band +-12 rows" in lines[-1]


def test_holdout_names_no_file_and_no_time(
    fitted: Fitted, other_frames: Path, capsys: pytest.CaptureFixture
) -> None:
    run("holdout", "--camera", "SYN1", "--fit", str(fitted.path), "--frames", str(other_frames))
    out = capsys.readouterr().out
    assert ".jpg" not in out and "2026" not in out and "T00" not in out


def test_holdout_with_only_frames_the_fit_used_says_so_and_exits_2(
    fitted: Fitted, archive: Path, capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    used_only = tmp_path / "used"
    used_only.mkdir()
    fit = load_fit("SYN1", directory=fitted.out_dir)
    for path in (archive / "2026-10-10").glob("*.jpg"):
        if frame_sha256(path) in {r.sha256 for r in fit.frames_used}:
            (used_only / path.name).write_bytes(path.read_bytes())
    code = run("holdout", "--camera", "SYN1", "--fit", str(fitted.path), "--frames", str(used_only))
    assert code == 2
    assert "all were used in the fit" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--fit", "{tmp}/missing.json"], "missing.json"),
        (["--camera", "OTHER1"], "OTHER1"),
        (["--frames", "{tmp}/nowhere"], "nowhere"),
        (["--band", "0"], "band"),
    ],
)
def test_holdout_exits_2_with_a_reason_when_an_input_is_unusable(
    tmp_path: Path,
    fitted: Fitted,
    other_frames: Path,
    capsys: pytest.CaptureFixture,
    argv: list[str],
    message: str,
) -> None:
    base = {
        "--camera": "SYN1",
        "--fit": str(fitted.path),
        "--frames": str(other_frames),
        "--band": "12",
    }
    for flag, value in zip(argv[::2], argv[1::2], strict=True):
        base[flag] = value.replace("{tmp}", str(tmp_path))
    code = run("holdout", *[part for pair in base.items() for part in pair])
    err = capsys.readouterr().err
    assert code == 2 and message in err
    assert err.strip().count("\n") == 0 and "Traceback" not in err


def test_holdout_on_an_unreadable_frame_exits_2_with_a_one_line_reason(
    tmp_path: Path, fitted: Fitted, capsys: pytest.CaptureFixture
) -> None:
    frames = _bad_frames(tmp_path, "garbage")
    code = run("holdout", "--camera", "SYN1", "--fit", str(fitted.path), "--frames", str(frames))
    err = capsys.readouterr().err
    assert code == 2 and "not a readable JPEG" in err
    assert err.strip().count("\n") == 0


# --- mask  ----------------------------------------------------------------------------------------


def test_mask_writes_a_png_of_the_sky_from_the_frozen_fit(
    tmp_path: Path, fitted: Fitted, capsys: pytest.CaptureFixture
) -> None:
    geometry = fitted.out_dir
    png = tmp_path / "out" / "SYN1.png"

    code = run(
        "mask", "--camera", "SYN1", "--geometry-dir", str(geometry), "--out", str(png),
        "--margin-deg", "1.0",
    )  # fmt: skip

    assert code == 0
    with Image.open(png) as image:
        mask = np.asarray(image) > 0
    assert mask.shape == (300, 800)
    assert 0.3 < mask.mean() < 0.7
    assert not mask[-30:, -200:].any()
    assert "SYN1" in capsys.readouterr().out


def test_mask_of_an_unfitted_camera_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = run(
        "mask",
        "--camera",
        "NOPE1",
        "--geometry-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "m.png"),
    )
    assert code == 2
    assert "NOPE1" in capsys.readouterr().err
    assert not (tmp_path / "m.png").exists()


def test_mask_refuses_a_negative_margin(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = run("mask", "--camera", "SYN1", "--geometry-dir", str(tmp_path), "--margin-deg", "-1")
    assert code == 2
    assert "margin" in capsys.readouterr().err
