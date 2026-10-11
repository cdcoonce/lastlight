"""Where a fit gets its frames and the camera's site from."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import REF_LAT, REF_LON
from PIL import Image

from lastlight.camgeom.frames import (
    discover_frames,
    evening_of,
    frame_sha256,
    load_rgb,
    parse_frame_time,
)
from lastlight.camgeom.sites import SITES_ENV, SiteFileError, load_camera_site, resolve_site_file

STAMP = "%Y%m%dT%H%M%SZ"
WHEN = dt.datetime(2026, 10, 11, 0, 30, 9, tzinfo=dt.UTC)


def save_jpeg(path: Path, value: int = 120, size: tuple[int, int] = (80, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path, quality=90)
    return path


# --- frame names and the archive layout ----------------------------------------------------------


def test_a_frame_name_is_the_last_modified_time_in_utc() -> None:
    assert parse_frame_time(f"{WHEN.strftime(STAMP)}.jpg") == WHEN
    assert parse_frame_time(Path("a") / f"{WHEN.strftime(STAMP)}.jpeg") == WHEN


@pytest.mark.parametrize("name", ["snapshot.jpg", "frame-1.jpg", "not-a-time.jpg", ".jpg"])
def test_a_name_that_is_not_a_timestamp_has_no_time(name: str) -> None:
    assert parse_frame_time(name) is None


def test_an_impossible_timestamp_has_no_time() -> None:
    assert parse_frame_time("20261345" + "T256199Z.jpg") is None


def test_discovery_walks_the_archive_and_orders_frames_by_time(tmp_path: Path) -> None:
    early, late = WHEN, WHEN + dt.timedelta(minutes=20)
    save_jpeg(tmp_path / "2026-10-11" / f"{late.strftime(STAMP)}.jpg")
    save_jpeg(tmp_path / "2026-10-10" / f"{early.strftime(STAMP)}.jpg")
    save_jpeg(tmp_path / "2026-10-10" / "mystery.jpg")
    (tmp_path / "2026-10-10" / f"{early.strftime(STAMP)}.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a frame")

    found = discover_frames(tmp_path)

    assert [frame.when_utc for frame in found] == [early, late, None]
    assert all(frame.path.suffix == ".jpg" for frame in found)


def test_discovery_of_a_single_flat_directory_works_too(tmp_path: Path) -> None:
    save_jpeg(tmp_path / f"{WHEN.strftime(STAMP)}.jpg")
    assert len(discover_frames(tmp_path)) == 1


def test_discovery_of_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="frames"):
        discover_frames(tmp_path / "nowhere")


def test_a_frame_decodes_to_rgb_rows_by_columns_by_three(tmp_path: Path) -> None:
    path = save_jpeg(tmp_path / "f.jpg", value=100, size=(80, 30))
    image = load_rgb(path)
    assert image.shape == (30, 80, 3) and image.dtype == np.uint8
    assert abs(int(image.mean()) - 100) <= 2


def test_a_frame_is_identified_by_the_sha256_of_its_bytes(tmp_path: Path) -> None:
    path = save_jpeg(tmp_path / "f.jpg")
    assert frame_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_evening_is_the_local_mean_solar_date_so_a_late_frame_keeps_its_day() -> None:
    # Just after midnight UTC on the 11th is still the late afternoon of the 10th here.
    assert evening_of(WHEN, REF_LON) == dt.date(2026, 10, 10)
    assert evening_of(dt.datetime(2026, 10, 10, 21, 0, tzinfo=dt.UTC), REF_LON) == dt.date(
        2026, 10, 10
    )
    assert evening_of(dt.datetime(2026, 10, 11, 9, 0, tzinfo=dt.UTC), REF_LON) == dt.date(
        2026, 10, 11
    )


# --- the private site file  -----------------------------------------------------------------------


def write_sites(path: Path, body: str | None = None) -> Path:
    path.write_text(
        body
        or f"[SYN1]\nlat = {REF_LAT}\nlon = {REF_LON}\n"
        "mount_height_m = 12.5\nview_azimuth_deg = 272.0\n"
    )
    return path


def test_a_site_is_read_by_camera_id(tmp_path: Path) -> None:
    site = load_camera_site("SYN1", write_sites(tmp_path / "sites.toml"))
    assert (site.camera_id, site.lat, site.lon) == ("SYN1", REF_LAT, REF_LON)
    assert (site.mount_height_m, site.view_azimuth_deg) == (12.5, 272.0)


def test_the_mount_height_defaults_to_ten_metres(tmp_path: Path) -> None:
    body = f"[SYN1]\nlat = {REF_LAT}\nlon = {REF_LON}\nview_azimuth_deg = 272.0\n"
    assert load_camera_site("SYN1", write_sites(tmp_path / "s.toml", body)).mount_height_m == 10.0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[OTHER]\nlat = 1.0\nlon = 2.0\nview_azimuth_deg = 3.0\n", "SYN1"),
        ("[SYN1]\nlon = 2.0\nview_azimuth_deg = 3.0\n", "lat"),
        ("[SYN1]\nlat = 1.0\nlon = 2.0\n", "view_azimuth_deg"),
        ("[SYN1]\nlat = 91.0\nlon = 2.0\nview_azimuth_deg = 3.0\n", "lat"),
        ("[SYN1]\nlat = 1.0\nlon = 200.0\nview_azimuth_deg = 3.0\n", "lon"),
        ("[SYN1]\nlat = 1.0\nlon = 2.0\nview_azimuth_deg = 3.0\nmount_height_m = -5\n", "mount"),
        ("this is not toml = = =", "SYN1|site file"),
    ],
)
def test_a_bad_site_file_is_refused_without_echoing_its_contents(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(SiteFileError, match=message) as error:
        load_camera_site("SYN1", write_sites(tmp_path / "s.toml", body))
    assert "200.0" not in str(error.value)


def test_the_site_file_comes_from_the_argument_or_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = write_sites(tmp_path / "a.toml")
    other = write_sites(tmp_path / "b.toml")
    monkeypatch.setenv(SITES_ENV, str(other))
    assert resolve_site_file(explicit) == explicit
    assert resolve_site_file(None) == other
    monkeypatch.delenv(SITES_ENV)
    with pytest.raises(SiteFileError, match=SITES_ENV):
        resolve_site_file(None)


def test_a_missing_site_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(SiteFileError, match="nowhere.toml"):
        load_camera_site("SYN1", tmp_path / "nowhere.toml")
