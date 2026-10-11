"""The DEM horizon: what skyline a terrain model predicts from a camera position."""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import (
    EARTH_RADIUS_M,
    REF_ELEV_M,
    REF_LAT,
    REF_LON,
    analytic_elevation_deg,
    cone_dem,
    destination,
    flat_dem,
    grid_around,
    haversine_m,
    write_geotiff,
)

from lastlight.camgeom.dem import (
    MIN_TILE_BYTES,
    Dem,
    DemError,
    fetch_tiles,
    horizon_profile,
    read_dem,
    tile_name,
    tile_url,
    tiles_for_bounds,
    view_bounds,
)

# --- the acceptance criterion: a cone against its analytic elevation -----------------------------


def test_horizon_of_a_cone_shaped_mountain_matches_the_analytic_elevation() -> None:
    bearing, distance_m, apex_m = 280.0, 28_000.0, 1500.0
    apex_lat, apex_lon = destination(REF_LAT, REF_LON, bearing, distance_m)
    dem = cone_dem(apex_lat, apex_lon, apex_m, slope=0.12)
    true_distance = float(haversine_m(REF_LAT, REF_LON, apex_lat, apex_lon))

    (found,) = horizon_profile(dem, REF_LAT, REF_LON, REF_ELEV_M, [bearing])

    expected = analytic_elevation_deg(REF_ELEV_M, apex_m, true_distance)
    assert 2.0 < expected < 2.6, "the scene must be a real mountain, not a bump"
    assert found == pytest.approx(expected, abs=0.05)


def test_the_cone_test_would_fail_without_earth_curvature() -> None:
    # Guard on the test itself: a flat-Earth answer is more than 0.05 degrees away, so the
    # comparison above can only pass with the curvature term.
    distance_m, apex_m = 28_000.0, 1500.0
    flat_earth = math.degrees(math.atan2(apex_m - REF_ELEV_M, distance_m))
    curved = analytic_elevation_deg(REF_ELEV_M, apex_m, distance_m)
    assert abs(flat_earth - curved) > 0.1


def test_horizon_of_a_level_plain_is_the_geometric_dip_of_the_horizon() -> None:
    height_m = 100.0
    dip = math.degrees(math.acos(EARTH_RADIUS_M / (EARTH_RADIUS_M + height_m)))
    dem = flat_dem(0.0)
    found = horizon_profile(
        dem, REF_LAT, REF_LON, height_m, [0.0, 90.0, 215.0], max_range_km=100.0, min_range_km=0.5
    )
    assert found == pytest.approx([-dip] * 3, abs=0.01)


def test_the_horizon_falls_away_on_either_side_of_the_mountain() -> None:
    bearing, distance_m = 280.0, 28_000.0
    apex_lat, apex_lon = destination(REF_LAT, REF_LON, bearing, distance_m)
    dem = cone_dem(apex_lat, apex_lon, 1500.0, slope=0.12)
    azimuths = bearing + np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    left, mid_left, centre, mid_right, right = horizon_profile(
        dem, REF_LAT, REF_LON, REF_ELEV_M, azimuths
    )
    assert centre > mid_left > left
    assert centre > mid_right > right
    assert mid_left == pytest.approx(mid_right, abs=0.02)


def test_a_ray_that_never_meets_the_dem_has_no_horizon() -> None:
    apex_lat, apex_lon = destination(REF_LAT, REF_LON, 280.0, 28_000.0)
    dem = cone_dem(apex_lat, apex_lon, 1500.0, slope=0.12)
    found = horizon_profile(dem, REF_LAT, REF_LON, REF_ELEV_M, [100.0, 280.0])
    assert math.isnan(found[0])
    assert not math.isnan(found[1])


def test_terrain_nearer_than_the_minimum_range_is_ignored() -> None:
    # A 60 m pillar a kilometre away: local clutter that a mast-mounted camera sees over.
    lats, lons, step, lat_top, lon_left = grid_around(REF_LAT, REF_LON, 6_000.0, 3.0)
    height = np.zeros((lats.size, lons.size), dtype=np.float32)
    pillar_lat, pillar_lon = destination(REF_LAT, REF_LON, 90.0, 1_000.0)
    near = (np.abs(lats[:, None] - pillar_lat) < 0.0004) & (
        np.abs(lons[None, :] - pillar_lon) < 0.0004
    )
    height[near] = 60.0
    dem = Dem(height, lat_top, lon_left, step, step)

    kept = horizon_profile(dem, REF_LAT, REF_LON, 10.0, [90.0], min_range_km=0.5)[0]
    skipped = horizon_profile(dem, REF_LAT, REF_LON, 10.0, [90.0], min_range_km=2.0)[0]
    assert kept > 1.5, "with a short minimum range the pillar must be the horizon"
    assert skipped < 0.5, "with the default clearance it must not be"


def test_the_horizon_is_computed_for_every_azimuth_requested_in_order() -> None:
    apex_lat, apex_lon = destination(REF_LAT, REF_LON, 280.0, 28_000.0)
    dem = cone_dem(apex_lat, apex_lon, 1500.0, slope=0.12)
    azimuths = np.array([281.0, 279.0, 280.0, 640.0])  # 640 is 280 once wrapped
    found = horizon_profile(dem, REF_LAT, REF_LON, REF_ELEV_M, azimuths)
    assert found.shape == (4,)
    assert found[3] == pytest.approx(found[2], abs=1e-6)
    assert found[2] > found[0] and found[2] > found[1]


# Grid positions below are offsets from the corner of the one-degree tile that holds the
# reference point, so that no coordinate pair is spelled out in this file.
TILE_LAT, TILE_LON = math.floor(REF_LAT), math.floor(REF_LON)
TOP, LEFT = TILE_LAT + 1.0, TILE_LON + 0.0  # the tile's north-west corner


# --- reading elevations ---------------------------------------------------------------------------


def small_dem() -> Dem:
    values = np.array([[0.0, 10.0, 20.0], [100.0, 110.0, 120.0], [200.0, 210.0, 220.0]])
    return Dem(
        values.astype(np.float32), lat_top=TOP, lon_left=LEFT, step_lat_deg=0.5, step_lon_deg=0.5
    )


def test_sampling_a_node_returns_its_value_and_a_cell_centre_the_average() -> None:
    dem = small_dem()
    assert dem.sample(TOP, LEFT) == pytest.approx(0.0)
    assert dem.sample(TOP - 0.5, LEFT + 0.5) == pytest.approx(110.0)
    assert dem.sample(TOP - 0.25, LEFT + 0.25) == pytest.approx(55.0)  # mean of 0, 10, 100, 110


def test_sampling_outside_the_grid_is_nan() -> None:
    dem = small_dem()
    assert math.isnan(float(dem.sample(TOP + 0.2, LEFT + 0.5)))
    assert math.isnan(float(dem.sample(TOP - 0.5, LEFT - 1.0)))
    assert math.isnan(float(dem.sample(TOP - 1.1, LEFT + 0.5)))


def test_sampling_takes_arrays_of_points() -> None:
    dem = small_dem()
    values = dem.sample(np.array([TOP, TOP - 1.0]), np.array([LEFT, LEFT + 1.0]))
    assert values == pytest.approx([0.0, 220.0])


def test_a_dem_with_a_nodata_hole_gives_nan_there_and_not_a_wrong_number() -> None:
    values = np.array([[1.0, 2.0], [3.0, math.nan]], dtype=np.float32)
    dem = Dem(values, TOP, LEFT, 0.5, 0.5)
    assert math.isnan(float(dem.sample(TOP - 0.25, LEFT + 0.25)))
    assert dem.sample(TOP, LEFT) == pytest.approx(1.0)


def test_a_geotiff_reads_back_with_its_elevations_and_georeferencing(tmp_path: Path) -> None:
    source = cone_dem(
        *destination(REF_LAT, REF_LON, 280.0, 5_000.0), 900.0, 0.15, half_extent_m=4_000.0
    )
    path = write_geotiff(tmp_path / "tile.tif", source)
    rows, cols = source.elevation_m.shape
    south = source.lat_top - (rows - 1) * source.step_lat_deg
    east = source.lon_left + (cols - 1) * source.step_lon_deg

    got = read_dem(
        [path],
        bounds=(
            source.lon_left - source.step_lon_deg / 2,
            south - source.step_lat_deg / 2,
            east + source.step_lon_deg / 2,
            source.lat_top + source.step_lat_deg / 2,
        ),
    )

    assert got.elevation_m.shape == source.elevation_m.shape
    assert got.lat_top == pytest.approx(source.lat_top, abs=1e-9)
    assert got.lon_left == pytest.approx(source.lon_left, abs=1e-9)
    assert got.step_lat_deg == pytest.approx(source.step_lat_deg, abs=1e-12)
    assert np.allclose(got.elevation_m, source.elevation_m)


def test_reading_a_window_of_a_geotiff_returns_only_that_window(tmp_path: Path) -> None:
    source = cone_dem(
        *destination(REF_LAT, REF_LON, 280.0, 5_000.0), 900.0, 0.15, half_extent_m=4_000.0
    )
    path = write_geotiff(tmp_path / "tile.tif", source)
    half_step = source.step_lat_deg / 2
    got = read_dem(
        [path],
        bounds=(
            source.lon_left + 10 * source.step_lon_deg - half_step,
            source.lat_top - 30 * source.step_lat_deg - half_step,
            source.lon_left + 25 * source.step_lon_deg + half_step,
            source.lat_top - 20 * source.step_lat_deg + half_step,
        ),
    )
    assert got.elevation_m.shape == (11, 16)
    assert np.allclose(got.elevation_m, source.elevation_m[20:31, 10:26])


def test_two_adjacent_tiles_read_as_one_continuous_grid(tmp_path: Path) -> None:
    step = 0.1
    west = Dem(np.arange(100, dtype=np.float32).reshape(10, 10), TOP, LEFT, step, step)
    east = Dem(
        1000.0 + np.arange(100, dtype=np.float32).reshape(10, 10), TOP, LEFT + 1.0, step, step
    )
    paths = [write_geotiff(tmp_path / "w.tif", west), write_geotiff(tmp_path / "e.tif", east)]
    got = read_dem(paths, bounds=(LEFT - 0.05, TOP - 0.95, LEFT + 1.95, TOP + 0.05))
    assert got.elevation_m.shape == (10, 20)
    assert np.allclose(got.elevation_m[:, :10], west.elevation_m)
    assert np.allclose(got.elevation_m[:, 10:], east.elevation_m)


def test_ground_outside_every_tile_reads_as_no_data(tmp_path: Path) -> None:
    tile = Dem(np.ones((10, 10), dtype=np.float32), TOP, LEFT, 0.1, 0.1)
    path = write_geotiff(tmp_path / "t.tif", tile)
    got = read_dem([path], bounds=(LEFT - 0.05, TOP - 0.95, LEFT + 1.45, TOP + 0.05))
    assert np.isnan(got.elevation_m[:, -1]).all()
    assert np.isfinite(got.elevation_m[:, 0]).all()


# --- finding and fetching Copernicus GLO-30 tiles  ------------------------------------------------


@pytest.mark.parametrize(
    ("lat_floor", "lon_floor", "expected"),
    [
        (33, -113, "Copernicus_DSM_COG_10_N33_00_W113_00_DEM"),
        (-1, 7, "Copernicus_DSM_COG_10_S01_00_E007_00_DEM"),
        (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
    ],
)
def test_tile_names_follow_the_copernicus_scheme(
    lat_floor: int, lon_floor: int, expected: str
) -> None:
    assert tile_name(lat_floor, lon_floor) == expected


def test_a_tile_url_points_at_the_public_bucket() -> None:
    name = tile_name(33, -113)
    assert tile_url(name) == f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def test_tiles_for_bounds_lists_every_tile_the_box_touches() -> None:
    one = tiles_for_bounds(TILE_LAT + 0.2, TILE_LAT + 0.9, TILE_LON + 0.1, TILE_LON + 0.9)
    assert one == [tile_name(TILE_LAT, TILE_LON)]
    four = tiles_for_bounds(TILE_LAT - 0.2, TILE_LAT + 0.2, TILE_LON + 0.8, TILE_LON + 1.2)
    assert sorted(four) == sorted(
        [
            tile_name(TILE_LAT - 1, TILE_LON),
            tile_name(TILE_LAT - 1, TILE_LON + 1),
            tile_name(TILE_LAT, TILE_LON),
            tile_name(TILE_LAT, TILE_LON + 1),
        ]
    )


def test_a_box_that_ends_exactly_on_a_tile_edge_does_not_pull_in_the_next_tile() -> None:
    edges = tiles_for_bounds(TILE_LAT + 0.0, TILE_LAT + 1.0, TILE_LON + 0.0, TILE_LON + 1.0)
    assert edges == [tile_name(TILE_LAT, TILE_LON)]


def fake_tile(size: int = MIN_TILE_BYTES + 100) -> bytes:
    """Bytes that pass for a tile: they start like a TIFF and are not a stub."""
    return b"II*\x00" + b"\x01" * (size - 4)


def test_fetch_tiles_downloads_once_and_reuses_the_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        return fake_tile()

    names = [tile_name(33, -113)]
    first = fetch_tiles(names, tmp_path, fetch=fake_fetch)
    second = fetch_tiles(names, tmp_path, fetch=fake_fetch)
    assert calls == [tile_url(names[0])]
    assert first == second == [tmp_path / f"{names[0]}.tif"]
    assert first[0].read_bytes() == fake_tile()


def test_a_failed_download_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    def broken(url: str) -> bytes:
        raise OSError("connection reset")

    with pytest.raises(DemError, match="connection reset"):
        fetch_tiles([tile_name(33, -113)], tmp_path, fetch=broken)
    assert list(tmp_path.iterdir()) == []


BAD_TILES = {
    "wrong_magic": b"<?xml version='1.0'?>" + b" " * MIN_TILE_BYTES,
    "html_error_page": b"<html><body>Access Denied</body></html>",
    "tiff_header_only": b"II*\x00" + b"\x00" * 12,
    "empty": b"",
}


@pytest.mark.parametrize("bad", BAD_TILES.values(), ids=BAD_TILES.keys())
def test_a_cached_tile_that_cannot_be_a_tile_is_discarded_and_fetched_again(
    tmp_path: Path, bad: bytes
) -> None:
    name = tile_name(TILE_LAT, TILE_LON)
    (tmp_path / f"{name}.tif").write_bytes(bad)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return fake_tile()

    (path,) = fetch_tiles([name], tmp_path, fetch=fetch)
    assert calls == [tile_url(name)]
    assert path.read_bytes() == fake_tile()


@pytest.mark.parametrize("bad", BAD_TILES.values(), ids=BAD_TILES.keys())
def test_a_download_that_is_not_a_tile_is_refused_and_never_cached(
    tmp_path: Path, bad: bytes
) -> None:
    # S3 answers a missing or forbidden object with a small XML or HTML page and an HTTP error,
    # but a proxy can hand back such a page with a success status; it must not become the tile.
    with pytest.raises(DemError, match="not a usable tile"):
        fetch_tiles([tile_name(TILE_LAT, TILE_LON)], tmp_path, fetch=lambda url: bad)
    assert list(tmp_path.iterdir()) == []


def test_a_good_cached_tile_is_used_without_a_fetch(tmp_path: Path) -> None:
    name = tile_name(TILE_LAT, TILE_LON)
    (tmp_path / f"{name}.tif").write_bytes(fake_tile())

    def never(url: str) -> bytes:
        raise AssertionError("a good tile in the cache must not be downloaded again")

    assert fetch_tiles([name], tmp_path, fetch=never) == [tmp_path / f"{name}.tif"]


def test_a_tile_that_is_not_a_readable_raster_is_a_dem_error_naming_it(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tif"
    broken.write_bytes(fake_tile(5000))  # starts like a TIFF, holds nothing
    with pytest.raises(DemError, match="broken.tif"):
        read_dem([broken], bounds=(LEFT, TOP - 1, LEFT + 1, TOP))
    garbage = tmp_path / "garbage.tif"
    garbage.write_bytes(b"not a raster at all" * 100)
    with pytest.raises(DemError, match="garbage.tif"):
        read_dem([garbage], bounds=(LEFT, TOP - 1, LEFT + 1, TOP))
    with pytest.raises(DemError, match="missing.tif"):
        read_dem([tmp_path / "missing.tif"], bounds=(LEFT, TOP - 1, LEFT + 1, TOP))


def test_an_interrupted_write_leaves_neither_the_tile_nor_a_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = Path.write_bytes

    def half_then_fail(self: Path, data: bytes) -> int:
        real_write(self, data[: len(data) // 2])
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", half_then_fail)
    name = tile_name(TILE_LAT, TILE_LON)
    with pytest.raises(DemError, match="disk full"):
        fetch_tiles([name], tmp_path, fetch=lambda url: fake_tile())
    assert list(tmp_path.iterdir()) == [], "a truncated tile would be trusted by the next run"

    monkeypatch.undo()
    (path,) = fetch_tiles([name], tmp_path, fetch=lambda url: fake_tile())
    assert path.read_bytes() == fake_tile()


@pytest.mark.network
def test_the_real_tile_for_the_phoenix_area_is_a_tiff_on_the_public_bucket() -> None:
    url = tile_url(tile_name(33, -113))
    request = urllib.request.Request(url, headers={"Range": "bytes=0-3"})
    with urllib.request.urlopen(request, timeout=60) as response:
        head = response.read()
    assert head in {b"II*\x00", b"II+\x00"}


# --- the ground a view can reach ---------------------------------------------------------------


def test_the_view_bounds_reach_the_far_end_of_every_ray_and_include_the_observer() -> None:
    west, south, east, north = view_bounds(REF_LAT, REF_LON, [180.0, 270.0, 360.0], 50.0)
    one_degree_of_latitude_km = 111.2
    assert REF_LAT - south == pytest.approx(50.0 / one_degree_of_latitude_km, rel=0.01)
    assert north - REF_LAT == pytest.approx(50.0 / one_degree_of_latitude_km, rel=0.01)
    assert REF_LON - west == pytest.approx(
        50.0 / (one_degree_of_latitude_km * math.cos(math.radians(REF_LAT))), rel=0.01
    )
    assert east == pytest.approx(REF_LON, abs=1e-6), "no ray goes east, so the observer is the edge"


def test_the_view_bounds_can_be_padded_on_every_side() -> None:
    plain = view_bounds(REF_LAT, REF_LON, [250.0, 290.0], 30.0)
    padded = view_bounds(REF_LAT, REF_LON, [250.0, 290.0], 30.0, margin_deg=0.02)
    assert padded == pytest.approx(
        (plain[0] - 0.02, plain[1] - 0.02, plain[2] + 0.02, plain[3] + 0.02)
    )
