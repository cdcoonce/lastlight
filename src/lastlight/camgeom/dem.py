"""Terrain elevations, and the horizon a terrain model predicts from a camera position.

The terrain is the Copernicus GLO-30 digital surface model: 1 arc-second posts (about 30 m),
cloud-optimised GeoTIFF tiles of one degree, on a public AWS bucket that needs no account.
A tile is downloaded once into a local cache and read from there.

`horizon_profile` is the theoretical skyline: for each azimuth, the largest elevation angle at
which any terrain along that bearing is seen, on a spherical Earth of radius 6371 km (the same
sphere as `lastlight.geometry`). Curvature is part of the angle, not a correction applied
afterwards: a point at ground distance s and height H is seen at

    atan2((R + H) cos(s / R) - (R + h), (R + H) sin(s / R))

above the horizontal at an observer of height h. Atmospheric refraction is not modelled: over
50 km it moves a ridge by under 0.02 degrees, below the other errors of the fit.
"""

from __future__ import annotations

import math
import os
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

from lastlight import __version__

EARTH_RADIUS_M = 6_371_000.0
BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
USER_AGENT = f"lastlight/{__version__} (+https://github.com/cdcoonce/lastlight)"
DEFAULT_CACHE_DIR = Path("~/.cache/lastlight/dem").expanduser()
CACHE_ENV = "LASTLIGHT_DEM_CACHE"

Fetch = Callable[[str], bytes]


@dataclass(frozen=True, slots=True)
class Dem:
    """Elevations in metres on a regular latitude and longitude grid, north at the top.

    `elevation_m[row, col]` is the height at latitude `lat_top - row * step_lat_deg` and
    longitude `lon_left + col * step_lon_deg`: the posts are points, as in GLO-30. NaN means no
    data.
    """

    elevation_m: np.ndarray
    lat_top: float
    lon_left: float
    step_lat_deg: float
    step_lon_deg: float

    def __post_init__(self) -> None:
        if self.elevation_m.ndim != 2 or min(self.elevation_m.shape) < 2:
            raise ValueError("a DEM needs at least two rows and two columns")
        if not (self.step_lat_deg > 0.0 and self.step_lon_deg > 0.0):
            raise ValueError("the grid steps must be positive")

    def sample(self, lat, lon):
        """Bilinear elevation at a point or points; NaN outside the grid or beside a hole."""
        lat_a = np.asarray(lat, dtype=float)
        lon_a = np.asarray(lon, dtype=float)
        rows, cols = self.elevation_m.shape
        row = (self.lat_top - lat_a) / self.step_lat_deg
        col = (lon_a - self.lon_left) / self.step_lon_deg
        slack = 1e-6
        inside = (
            (row >= -slack)
            & (row <= rows - 1 + slack)
            & (col >= -slack)
            & (col <= cols - 1 + slack)
        )
        row0 = np.clip(np.floor(np.where(inside, row, 0.0)).astype(np.int64), 0, rows - 2)
        col0 = np.clip(np.floor(np.where(inside, col, 0.0)).astype(np.int64), 0, cols - 2)
        frac_row = np.clip(np.where(inside, row, 0.0) - row0, 0.0, 1.0)
        frac_col = np.clip(np.where(inside, col, 0.0) - col0, 0.0, 1.0)
        grid = self.elevation_m
        corners = (
            (row0, col0, (1 - frac_row) * (1 - frac_col)),
            (row0, col0 + 1, (1 - frac_row) * frac_col),
            (row0 + 1, col0, frac_row * (1 - frac_col)),
            (row0 + 1, col0 + 1, frac_row * frac_col),
        )
        # A corner with no weight must not poison the sum, so a post beside a hole still reads.
        value = sum(np.where(weight > 0.0, grid[r, c] * weight, 0.0) for r, c, weight in corners)
        value = np.where(inside, value, np.nan)
        return float(value) if value.ndim == 0 else value


def _ground_points(lat: float, lon: float, bearing_rad: np.ndarray, theta: np.ndarray):
    """Latitude and longitude after an angular distance `theta` along great circles."""
    phi = math.radians(lat)
    sin_phi2 = np.sin(phi) * np.cos(theta) + np.cos(phi) * np.sin(theta) * np.cos(bearing_rad)
    dlon = np.arctan2(
        np.sin(bearing_rad) * np.sin(theta) * np.cos(phi),
        np.cos(theta) - np.sin(phi) * sin_phi2,
    )
    return np.degrees(np.arcsin(np.clip(sin_phi2, -1.0, 1.0))), lon + np.degrees(dlon)


def horizon_profile(
    dem: Dem,
    lat: float,
    lon: float,
    elev_m: float,
    azimuths_deg: Iterable[float],
    *,
    max_range_km: float = 50.0,
    step_m: float = 30.0,
    min_range_km: float = 2.0,
    chunk: int = 200,
) -> np.ndarray:
    """Elevation angle of the horizon in degrees at each azimuth, NaN where there is no terrain.

    The terrain is sampled every `step_m` along each great circle from `min_range_km` to
    `max_range_km`. Terrain nearer than `min_range_km` is skipped: buildings and trees beside a
    mast-mounted camera are in the surface model but are not what the skyline is made of. An
    azimuth whose ray never meets DEM data has no horizon (NaN), never a made-up one.
    """
    azimuths = np.asarray(list(azimuths_deg), dtype=float) % 360.0
    distance = np.arange(min_range_km * 1000.0, max_range_km * 1000.0 + step_m / 2, step_m)
    theta = distance / EARTH_RADIUS_M
    profile = np.full(azimuths.shape, np.nan)
    for start in range(0, azimuths.size, chunk):
        bearing = np.radians(azimuths[start : start + chunk])[:, None]
        ground_lat, ground_lon = _ground_points(lat, lon, bearing, theta[None, :])
        height = dem.sample(ground_lat, ground_lon)
        along = (EARTH_RADIUS_M + height) * np.sin(theta)
        up = (EARTH_RADIUS_M + height) * np.cos(theta) - (EARTH_RADIUS_M + elev_m)
        angle = np.degrees(np.arctan2(up, along))
        best = np.where(np.isnan(angle), -np.inf, angle).max(axis=1)
        profile[start : start + chunk] = np.where(np.isfinite(best), best, np.nan)
    return profile


# --- the bounding box a view needs, and the tiles that cover it ---------------------------------


def view_bounds(
    lat: float,
    lon: float,
    azimuths_deg: Sequence[float],
    range_km: float,
    *,
    margin_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """(west, south, east, north) in degrees of the ground a set of rays can touch.

    The box is padded by `margin_deg`. Name the tiles from the exact box, so that a ray's last
    kilometre does not pull in a whole neighbouring tile, and read the DEM with a small margin
    so that the camera's own position falls inside the posts that are read, not on their edge.
    """
    azimuths = np.asarray(azimuths_deg, dtype=float)
    ends_lat, ends_lon = _ground_points(
        lat, lon, np.radians(azimuths), np.array([range_km * 1000.0 / EARTH_RADIUS_M])
    )
    lats = np.append(np.ravel(ends_lat), lat)
    lons = np.append(np.ravel(ends_lon), lon)
    return (
        float(lons.min()) - margin_deg,
        float(lats.min()) - margin_deg,
        float(lons.max()) + margin_deg,
        float(lats.max()) + margin_deg,
    )


def tile_name(lat_floor: int, lon_floor: int) -> str:
    """Copernicus GLO-30 tile name from the south-west corner of the one-degree tile."""
    north_south = "N" if lat_floor >= 0 else "S"
    east_west = "E" if lon_floor >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{north_south}{abs(lat_floor):02d}_00_"
        f"{east_west}{abs(lon_floor):03d}_00_DEM"
    )


def tile_url(name: str) -> str:
    return f"{BUCKET_URL}/{name}/{name}.tif"


def tiles_for_bounds(south: float, north: float, west: float, east: float) -> list[str]:
    """Names of every one-degree tile the box touches; a box ending on a tile edge stops there."""
    lat_range = range(math.floor(south), math.ceil(north))
    lon_range = range(math.floor(west), math.ceil(east))
    return [tile_name(lat_floor, lon_floor) for lat_floor in lat_range for lon_floor in lon_range]


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override).expanduser() if override else DEFAULT_CACHE_DIR


def fetch_tiles(names: Iterable[str], cache_dir: Path, *, fetch: Fetch = _http_get) -> list[Path]:
    """Local paths of the named tiles, downloading any that the cache does not hold yet.

    A download is written to a temporary name and renamed when complete, so an interrupted
    transfer never leaves a truncated tile that a later run would trust.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = cache_dir / f"{name}.tif"
        if not path.exists():
            partial = path.with_suffix(".tif.part")
            try:
                partial.write_bytes(fetch(tile_url(name)))
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        paths.append(path)
    return paths


def read_dem(paths: Sequence[Path], bounds: tuple[float, float, float, float]) -> Dem:
    """Mosaic the tiles that cover `bounds` (west, south, east, north), NaN where none does."""
    datasets = [rasterio.open(path) for path in paths]
    try:
        mosaic, transform = merge(
            datasets, bounds=bounds, nodata=np.nan, dtype="float32", method="first"
        )
    finally:
        for dataset in datasets:
            dataset.close()
    step_lon, step_lat = transform.a, -transform.e
    return Dem(
        elevation_m=mosaic[0],
        lat_top=transform.f - step_lat / 2.0,
        lon_left=transform.c + step_lon / 2.0,
        step_lat_deg=step_lat,
        step_lon_deg=step_lon,
    )
