"""Synthetic scenes for the camgeom tests.

A scene is a known `CameraModel` looking at a known skyline (elevation against azimuth).
Frames are rendered from it pixel by pixel, so what the code under test recovers can be
compared with the truth. No real frame, DEM tile or camera position is used: the observer
sits at the public reference point, and every other place is derived from it.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

from lastlight.camgeom.dem import Dem
from lastlight.camgeom.fitfile import CameraFit, FrameRecord, Uncertainty
from lastlight.camgeom.model import CameraModel
from lastlight.camgeom.orientation import Orientation
from lastlight.camgeom.skyline import Box, default_overlay_boxes

REF_LAT, REF_LON = 33.45, -112.07
REF_ELEV_M = 340.0
EARTH_RADIUS_M = 6_371_000.0
WIDTH, HEIGHT = 800, 300

TRUE_MODEL = CameraModel(
    camera_id="SYN1",
    lat=REF_LAT,
    lon=REF_LON,
    elev_m=REF_ELEV_M,
    azimuth_deg=274.0,
    elevation_deg=2.0,
    roll_deg=0.4,
    hfov_deg=66.0,
    width=WIDTH,
    height=HEIGHT,
)


def ridge_elevation_deg(azimuth_deg: np.ndarray | float) -> np.ndarray:
    """A synthetic skyline with a few peaks and some texture, in degrees above the horizontal."""
    az = np.asarray(azimuth_deg, dtype=float)
    return (
        1.6
        + 1.1 * np.exp(-(((az - 266.0) / 2.5) ** 2))
        + 1.8 * np.exp(-(((az - 281.0) / 3.5) ** 2))
        + 0.9 * np.exp(-(((az - 289.0) / 1.5) ** 2))
        + 0.35 * np.sin(np.radians(az * 37.0))
    )


COMB_PERIOD_DEG = 5.5
COMB_AMPLITUDES = (
    0.93, 0.94, 1.06, 1.07, 1.29, 1.29, 1.11, 0.91, 1.05, 1.10,
    1.17, 0.90, 1.24, 1.00, 0.96, 1.10, 1.14, 0.90, 1.19, 0.95,
)  # fmt: skip


def comb_elevation_deg(azimuth_deg: np.ndarray | float) -> np.ndarray:
    """A near-periodic skyline: a peak every 5.5 degrees of azimuth, each a little different.

    A fit that is shown this ridge through a 20 degree lens sees three or four peaks, and the
    same view shifted by a whole number of periods looks almost the same. Only the small
    differences in height tell the right alignment from its neighbours.
    """
    az = np.asarray(azimuth_deg, dtype=float)
    out = np.full(az.shape, 1.37)
    for k, amplitude in enumerate(COMB_AMPLITUDES):
        out = out + 1.57 * amplitude * np.exp(-(((az - (203.2 + k * COMB_PERIOD_DEG)) / 1.75) ** 2))
    return out


def mismatched_ridge_elevation_deg(azimuth_deg: np.ndarray | float) -> np.ndarray:
    """`ridge_elevation_deg` plus a smooth error of about 0.1 degree, as a DEM that is slightly off.

    Frames rendered from this ridge and fitted against the plain one carry a residual no
    orientation can remove, which is what real frames and a real terrain model do.
    """
    az = np.asarray(azimuth_deg, dtype=float)
    return (
        ridge_elevation_deg(az)
        + 0.08 * np.sin(np.radians(az * 41.0))
        + 0.05 * np.sin(np.radians(az * 13.0 + 1.0))
    )


def haze(image: np.ndarray, amount: float, level: float = 140.0) -> np.ndarray:
    """Wash a frame toward a flat grey: `amount` of 0.7 leaves 30 percent of its contrast."""
    washed = image.astype(float) * (1.0 - amount) + level * amount
    return np.clip(np.rint(washed), 0, 255).astype(np.uint8)


def paint_cloud_base(
    image: np.ndarray,
    skyline_rows: np.ndarray,
    *,
    offset_start: float,
    offset_end: float,
    x0: int,
    x1: int,
    thickness: int = 14,
    shade_below: float = 40.0,
    shade_rows: int = 8,
) -> np.ndarray:
    """A bright cloud whose dark underside runs parallel to the skyline, some rows above it.

    From column `x0` to `x1` the cloud's base lies `offset_start` rows above the skyline,
    drifting to `offset_end` rows above it. The sky just under the base is `shade_below` dark
    (a cloud's shadow), so the base is a strong bright-over-dark edge: the kind that can
    out-shout a faint ridge.
    """
    out = image.copy()
    for column in range(x0, x1):
        offset = offset_start + (offset_end - offset_start) * (column - x0) / max(1, x1 - x0)
        base = int(round(skyline_rows[column] - offset))
        out[max(base - thickness, 0) : base, column] = 250
        out[base : base + shade_rows, column] = int(shade_below)
    return out


def render_frame(
    model: CameraModel,
    horizon_deg=ridge_elevation_deg,
    *,
    seed: int = 0,
    noise: float = 5.0,
    blur: float = 0.8,
    stamp: Box | None = None,
    supersample: int = 4,
) -> np.ndarray:
    """An RGB uint8 frame of a sky above a dark skyline, with sensor noise and an optional stamp.

    The sky is brighter near the horizon than overhead, as in the real frames, and the terrain
    is a dark brown. The skyline is antialiased by vertical supersampling so a sub-pixel
    position can be recovered. `stamp` paints a white rectangle, like the burned-in timestamp.
    """
    rows = np.arange(model.height, dtype=float)[:, None, None]
    sub = (np.arange(supersample, dtype=float) + 0.5) / supersample - 0.5
    cols = np.arange(model.width, dtype=float)[None, None, :]
    az, el = model.pixel_to_sky(cols + 0 * rows, rows + sub[None, :, None] + 0 * cols)
    terrain = (el < horizon_deg(az)).mean(axis=1)  # coverage of each pixel by terrain
    _, el_centre = model.pixel_to_sky(
        np.arange(model.width, dtype=float)[None, :] + np.zeros((model.height, 1)),
        np.arange(model.height, dtype=float)[:, None] + np.zeros((1, model.width)),
    )
    sky_level = np.clip(175.0 - 4.0 * el_centre, 60.0, 200.0)
    sky = np.stack([sky_level * 0.85, sky_level * 0.92, sky_level * 1.0 + 20.0], axis=-1)
    ground = np.array([70.0, 62.0, 55.0])
    image = sky * (1.0 - terrain[..., None]) + ground * terrain[..., None]
    if blur > 0:
        image = ndimage.gaussian_filter(image, sigma=(blur, blur, 0))
    if stamp is not None:
        image[stamp.y0 : stamp.y1, stamp.x0 : stamp.x1] = 255.0
    image = image + np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


# --- places: everything is derived from the reference point -------------------------------------


def destination(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """The point reached from (lat, lon) after `distance_m` along a great circle."""
    phi, bearing, angular = (
        math.radians(lat),
        math.radians(bearing_deg),
        distance_m / EARTH_RADIUS_M,
    )
    sin_phi2 = math.sin(phi) * math.cos(angular) + math.cos(phi) * math.sin(angular) * math.cos(
        bearing
    )
    phi2 = math.asin(sin_phi2)
    dlon = math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(phi),
        math.cos(angular) - math.sin(phi) * sin_phi2,
    )
    return math.degrees(phi2), math.degrees(math.radians(lon) + dlon)


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres on the same sphere as the code under test."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def analytic_elevation_deg(
    observer_height_m: float, target_height_m: float, distance_m: float
) -> float:
    """Elevation of a point above the observer's local horizontal on a sphere, exactly."""
    theta = distance_m / EARTH_RADIUS_M
    up = (EARTH_RADIUS_M + target_height_m) * math.cos(theta) - (EARTH_RADIUS_M + observer_height_m)
    out = (EARTH_RADIUS_M + target_height_m) * math.sin(theta)
    return math.degrees(math.atan2(up, out))


# --- synthetic terrain ----------------------------------------------------------------------------


def grid_around(
    lat: float, lon: float, half_extent_m: float, step_arcsec: float
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Latitude and longitude of every node of a grid with a node exactly at (lat, lon)."""
    step = step_arcsec / 3600.0
    n_lat = int(half_extent_m / (EARTH_RADIUS_M * math.radians(step)))
    n_lon = int(half_extent_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)) * math.radians(step)))
    lats = lat + (n_lat - np.arange(2 * n_lat + 1)) * step  # row 0 is the northern edge
    lons = lon + (np.arange(2 * n_lon + 1) - n_lon) * step
    return lats, lons, step, lats[0], lons[0]


def cone_dem(
    apex_lat: float,
    apex_lon: float,
    apex_height_m: float,
    slope: float,
    *,
    ground_m: float = 300.0,
    half_extent_m: float = 14_000.0,
    step_arcsec: float = 1.0,
) -> Dem:
    """A cone-shaped mountain with its apex exactly on a grid node, on flat ground."""
    lats, lons, step, lat_top, lon_left = grid_around(
        apex_lat, apex_lon, half_extent_m, step_arcsec
    )
    distance = haversine_m(apex_lat, apex_lon, lats[:, None], lons[None, :])
    height = np.maximum(ground_m, apex_height_m - slope * distance)
    return Dem(height.astype(np.float32), lat_top, lon_left, step, step)


def flat_dem(
    height_m: float, *, lat: float = REF_LAT, lon: float = REF_LON, half_extent_m: float = 120_000.0
) -> Dem:
    """A level plain on a coarse grid, wide enough for any ray out to the geometric horizon."""
    _, _, step, lat_top, lon_left = grid_around(lat, lon, half_extent_m, 30.0)
    size = int(2 * half_extent_m / (EARTH_RADIUS_M * math.radians(step))) + 1
    return Dem(np.full((size, size), height_m, dtype=np.float32), lat_top, lon_left, step, step)


def ridge_dem(*, step_arcsec: float = 3.0) -> Dem:
    """Cones of different heights and distances to the west of the reference point.

    Seen from the reference point they make a rugged skyline between about 255 and 305
    degrees, the kind of profile the orientation fit needs to lock onto.
    """
    cones = [  # (bearing, distance m, height m, slope)
        (258.0, 14_000.0, 1050.0, 0.20),
        (265.0, 22_000.0, 1500.0, 0.16),
        (272.0, 12_000.0, 800.0, 0.22),
        (279.0, 27_000.0, 1700.0, 0.14),
        (287.0, 17_000.0, 1250.0, 0.18),
        (295.0, 24_000.0, 1450.0, 0.15),
        (302.0, 13_000.0, 900.0, 0.2),
    ]
    lats, lons, step, lat_top, lon_left = grid_around(REF_LAT, REF_LON, 38_000.0, step_arcsec)
    height = np.full((lats.size, lons.size), 300.0)
    for bearing, distance, apex, slope in cones:
        apex_lat, apex_lon = destination(REF_LAT, REF_LON, bearing, distance)
        d = haversine_m(apex_lat, apex_lon, lats[:, None], lons[None, :])
        height = np.maximum(height, apex - slope * d)
    return Dem(height.astype(np.float32), lat_top, lon_left, step, step)


def write_geotiff(path: Path, dem: Dem) -> Path:
    """Write a DEM as a GeoTIFF laid out like the Copernicus tiles (pixel centres on nodes)."""
    transform = from_origin(
        dem.lon_left - dem.step_lon_deg / 2,
        dem.lat_top + dem.step_lat_deg / 2,
        dem.step_lon_deg,
        dem.step_lat_deg,
    )
    rows, cols = dem.elevation_m.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
    ) as out:
        out.write(dem.elevation_m.astype("float32"), 1)
    return path


def true_skyline_y(model: CameraModel, horizon_deg=ridge_elevation_deg) -> np.ndarray:
    """The exact row of the skyline in every column, by bisection on the camera model."""
    x = np.arange(model.width, dtype=float)
    low = np.full(model.width, -50.0)
    high = np.full(model.width, model.height + 50.0)
    for _ in range(50):
        middle = (low + high) / 2
        az, el = model.pixel_to_sky(x, middle)
        is_sky = el > horizon_deg(az)
        low = np.where(is_sky, middle, low)
        high = np.where(is_sky, high, middle)
    return (low + high) / 2


def paint_disk(image: np.ndarray, x: float, y: float, radius: float = 3.5) -> np.ndarray:
    """A saturated sun disk with a soft glow around it."""
    image = image.copy().astype(float)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    distance = np.hypot(cols - x, rows - y)
    image += 40.0 * np.exp(-((distance / 9.0) ** 2))[..., None]
    image[distance <= radius] = (255.0, 255.0, 250.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def paint_glow(image: np.ndarray, x: float, y: float, peak: float = 215.0) -> np.ndarray:
    """The sun behind thin cloud: a broad brightening with no saturated disk in it."""
    image = image.copy().astype(float)
    rows, cols = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    distance = np.hypot(cols - x, rows - y)
    image = np.maximum(image, peak * np.exp(-((distance / 14.0) ** 2))[..., None])
    return np.clip(image, 0, 255).astype(np.uint8)


# --- a fit file --------------------------------------------------------------------------------

SKYLINE_COLUMNS = tuple(float(x) for x in range(0, WIDTH, 10)) + (float(WIDTH - 1),)


def example_fit(**changes) -> CameraFit:
    """A valid, provisional `CameraFit` for the synthetic camera; `changes` replace fields."""
    fit = CameraFit(
        camera_id="SYN1",
        width=WIDTH,
        height=HEIGHT,
        orientation=Orientation(274.0, 2.0, 0.4, 66.0),
        skyline_rms_px=0.8,
        skyline_rms_deg=0.075,
        skyline_median_abs_px=0.6,
        n_columns=790,
        skyline_x_px=SKYLINE_COLUMNS,
        skyline_y_px=tuple(150.0 + 0.01 * x for x in SKYLINE_COLUMNS),
        overlay_boxes=default_overlay_boxes(WIDTH, HEIGHT),
        frames_used=(FrameRecord("2026-10-10", 14.65, "ab" * 32),),
        sun_check={"status": "skipped", "reason": "sun hidden by cloud", "tolerance_deg": 1.0},
        radial_term={"k1": 0.1, "skyline_rms_px": 0.7, "adopted": False},
        position_search={"box_km": 8.0, "positions": 60, "adopted": True},
        frozen_on="2026-10-10",
        position_basis="operator description, supplied at fit time and not stored",
        provisional=True,
        uncertainty=Uncertainty(
            azimuth_deg=(272.0, 276.0),
            hfov_deg=(60.0, 70.0),
            px_per_deg=(10.3, 12.4),
            basis="positions within 10 percent of the best residual inside the box",
            lower_bounds=False,
        ),
        stated_azimuth_deg=272.0,
        azimuth_minus_stated_deg=2.0,
    )
    return dataclasses.replace(fit, **changes)
