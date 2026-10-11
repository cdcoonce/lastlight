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
from lastlight.camgeom.dem import horizon_profile, tile_name
from lastlight.camgeom.fitfile import load_fit
from lastlight.camgeom.frames import frame_sha256
from lastlight.camgeom.skyline import default_overlay_boxes
from lastlight.camgeom.suncheck import predict_sun_pixel
from lastlight.commands.camgeom import COMMAND
from PIL import Image

from lastlight.cli import main

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
    path.write_text(
        f"[SYN1]\nlat = {REF_LAT}\nlon = {REF_LON}\n"
        f"mount_height_m = {MOUNT_HEIGHT_M}\nview_azimuth_deg = 272.0\n"
    )
    return path


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
