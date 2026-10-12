"""Build renderer v1's multiple-scattering table, one atmosphere node per run.

Usage (from the repository root):

    uv run python tools/make_bruneton_table.py --list
    uv run python tools/make_bruneton_table.py --node 0          # ... up to --node 23
    uv run python tools/make_bruneton_table.py --assemble        # prints the SHA-256 to pin
    uv run python tools/make_bruneton_table.py --reference
    uv run python tools/make_bruneton_table.py --convergence baseline --atmosphere aod-0.03
    uv run python tools/make_bruneton_table.py --stored stored-geometry
    uv run python tools/make_bruneton_table.py --monte-carlo
    uv run python tools/make_bruneton_table.py --convergence-report

Each `--node I` runs the Bruneton precomputation (`lastlight.render.bruneton.precompute`) for one
(`aod_550`, `ozone_du`) pair of the grid, gathers the ground observer's multiple-scattering XYZ on
the table's sun, view and azimuth nodes, and writes it to a cache outside the repository
(`~/.cache/lastlight/bruneton/<grid>-<key>/`, where the key digests the grid, the resolution and
the port's source, so nodes made by different code are never mixed). A node takes about four
minutes on the build machine, so a run never needs ten; several can run at once. `--assemble`
reads every node, in index order whatever order they were computed in, and writes the table
container (`lastlight.render.v1.encode_table`): no timestamps, so the same nodes give the same
bytes. It prints the SHA-256 to pin in `lastlight.render.v1.TABLE_SHA256`.

The checks, each one run per invocation and cached beside the nodes:

* `--reference`: the direct computation at an off-node point, which the interpolation test
  compares the table with (`tests/fixtures/bruneton_offnode.json`);
* `--convergence VARIANT --atmosphere aod-0.03|aod-1.0`: the precomputation with one resolution
  doubled (`CONVERGENCE_VARIANTS`; `baseline` is the table's own, `orders` adds one order), at
  WHTM1's band geometries for sun +6 to -6;
* `--stored stored-geometry|stored-aod|stored-ozone`: direct values where the committed table
  must interpolate (halfway between its nodes);
* `--monte-carlo`: an independent backward Monte Carlo against the port;
* `--convergence-report`: `docs/renderer/bruneton-convergence.md` from all of the above.

The `tiny` grid (`--grid tiny`) is the tests' own: the same code at a resolution that runs in
seconds.

The port and its licence: `src/lastlight/render/bruneton.py` (BSD-3-Clause, Copyright (c) 2017
Eric Bruneton) and `src/lastlight/render/data/README.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lastlight.render import bruneton, v1
from lastlight.render.atmosphere import Atmosphere
from lastlight.render.curve import sky_xyz

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lastlight" / "bruneton"
TABLE_PATH = ROOT / "src" / "lastlight" / "render" / "data" / v1.TABLE_NAME
REFERENCE_PATH = ROOT / "tests" / "fixtures" / "bruneton_offnode.json"
REPORT_PATH = ROOT / "docs" / "renderer" / "bruneton-convergence.md"
PORT_SOURCE = ROOT / "src" / "lastlight" / "render" / "bruneton.py"

WHTM1_VIEWS = (1.51, 1.96, 2.41, 2.86, 3.32, 3.99)
"""WHTM1's five contrast bands and its reference band as view elevations, in degrees: G0's
nominal view (`lastlight.g0.collect.build_views()["nominal"]`), to 0.01 degree."""
WHTM1_AZIMUTH = 27.64
"""WHTM1's centre azimuth minus the sun's when the sun stands at 0 on 2026-10-10, in degrees."""
ORDER_CHECK_AZIMUTHS = (0.0, 15.0, 30.0, 45.0)
"""Relative azimuths of the order check: WHTM1's view against the sun over the year, folded."""

OFF_NODE_POINT = {
    "aod_550": 0.035,
    "ozone_du": 300.0,
    "sun_elevation_deg": -3.25,
    "view_elevation_deg": 1.7,
    "rel_azimuth_deg": 27.0,
}


class MissingNodeError(RuntimeError):
    """`--assemble` was asked for a table while some node has not been computed."""


@dataclass(frozen=True)
class Grid:
    """The table's nodes on every axis, and the precomputation's resolution."""

    aod: tuple[float, ...]
    ozone: tuple[float, ...]
    sun: tuple[float, ...]
    view: tuple[float, ...]
    azimuth: tuple[float, ...]
    resolution: bruneton.Resolution


GRIDS: dict[str, Grid] = {
    "full": Grid(
        aod=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25, 0.5, 1.0),
        ozone=(250.0, 330.0),
        sun=tuple(float(e) for e in range(-12, 13)),
        view=(
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            13.0,
            17.0,
            22.0,
            30.0,
            40.0,
            55.0,
            70.0,
            90.0,
        ),  # fmt: skip
        azimuth=(0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 55.0, 70.0, 90.0, 120.0, 150.0, 180.0),
        resolution=bruneton.RESOLUTION,
    ),
    "tiny": Grid(
        aod=(0.0, 1.0),
        ozone=(250.0, 330.0),
        sun=(-12.0, 0.0, 12.0),
        view=(0.0, 10.0, 90.0),
        azimuth=(0.0, 90.0, 180.0),
        resolution=bruneton.TINY_RESOLUTION,
    ),
}


def node_count(grid_name: str) -> int:
    grid = GRIDS[grid_name]
    return len(grid.aod) * len(grid.ozone)


def node_atmosphere(grid_name: str, index: int) -> Atmosphere:
    """The atmosphere of node `index`: aod major, ozone minor."""
    grid = GRIDS[grid_name]
    if not 0 <= index < node_count(grid_name):
        raise ValueError(
            f"node {index} is not in the {grid_name} grid (0 to {node_count(grid_name) - 1})"
        )
    aod = grid.aod[index // len(grid.ozone)]
    ozone = grid.ozone[index % len(grid.ozone)]
    return Atmosphere(aod_550=aod, ozone_du=ozone)


def _grid_key(grid_name: str) -> str:
    grid = GRIDS[grid_name]
    spec = {
        "grid": grid_name,
        "axes": [grid.aod, grid.ozone, grid.sun, grid.view, grid.azimuth],
        "resolution": grid.resolution.as_dict(),
        "ground_albedo": bruneton.GROUND_ALBEDO,
        "order_check": [WHTM1_VIEWS, ORDER_CHECK_AZIMUTHS],
        "port": hashlib.sha256(PORT_SOURCE.read_bytes()).hexdigest(),
    }
    text = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def node_directory(grid_name: str, cache_dir: Path) -> Path:
    """Where the nodes of a grid live: one directory per grid, resolution and port source."""
    return Path(cache_dir) / f"{grid_name}-{_grid_key(grid_name)}"


def _node_paths(grid_name: str, cache_dir: Path, index: int) -> tuple[Path, Path]:
    directory = node_directory(grid_name, cache_dir)
    return directory / f"node-{index:03d}.npy", directory / f"node-{index:03d}.json"


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def compute_node(grid_name: str, index: int, cache_dir: Path | str) -> Path:
    """Precompute node `index` and cache its XYZ block (suns, views, azimuths, 3)."""
    grid = GRIDS[grid_name]
    atm = node_atmosphere(grid_name, index)
    start = time.perf_counter()
    field = bruneton.precompute(atm, resolution=grid.resolution, next_order=True)
    block = field.xyz_block(grid.sun, grid.view, grid.azimuth)
    share = 0.0
    for azimuth in ORDER_CHECK_AZIMUTHS:
        total = field.multiple_xyz(0.0, WHTM1_VIEWS, azimuth)
        extra = field.next_order_xyz(0.0, WHTM1_VIEWS, azimuth)
        share = max(share, float((extra / total).max()))
    elapsed = time.perf_counter() - start
    values_path, info_path = _node_paths(grid_name, Path(cache_dir), index)
    buffer = _npy_bytes(np.ascontiguousarray(block, dtype=float))
    _write_atomically(values_path, buffer)
    info = {
        "index": index,
        "aod_550": atm.aod_550,
        "ozone_du": atm.ozone_du,
        "share_of_next_order": share,
        "seconds": round(elapsed, 1),
    }
    _write_atomically(info_path, json.dumps(info, sort_keys=True, indent=1).encode("utf-8"))
    return values_path


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _load_node(grid_name: str, cache_dir: Path, index: int) -> tuple[np.ndarray, dict]:
    values_path, info_path = _node_paths(grid_name, cache_dir, index)
    if not values_path.exists() or not info_path.exists():
        raise MissingNodeError(
            f"node {index} of the {grid_name} grid is not in {node_directory(grid_name, cache_dir)}"
            f"; run --node {index} first"
        )
    grid = GRIDS[grid_name]
    block = np.load(values_path, allow_pickle=False)
    shape = (len(grid.sun), len(grid.view), len(grid.azimuth), 3)
    if block.shape != shape or not np.isfinite(block).all():
        raise MissingNodeError(f"node {index} in the cache is damaged: shape {block.shape}")
    return block, json.loads(info_path.read_text())


def table_metadata(grid_name: str, max_share: float) -> dict:
    grid = GRIDS[grid_name]
    defaults = {name: float(getattr(Atmosphere(aod_550=0.0), name)) for name in v1.BAKED_FIELDS}
    return {
        "baked_atmosphere": defaults,
        "ground_albedo": bruneton.GROUND_ALBEDO,
        "orders": [2, grid.resolution.orders],
        "order_check": {
            "sun_elevation_deg": 0.0,
            "view_elevations_deg": list(WHTM1_VIEWS),
            "rel_azimuths_deg": list(ORDER_CHECK_AZIMUTHS),
            "max_share_of_next_order": max_share,
        },
        "resolution": grid.resolution.as_dict(),
        "quantity": (
            "CIE 1931 XYZ of the light reaching a ground observer after two or more bounces "
            "(scattering or ground reflection), relative to a white card under the "
            "extraterrestrial sun (Y = 1)"
        ),
        "source": (
            "tools/make_bruneton_table.py: numpy port of E. Bruneton's precomputed atmospheric "
            "scattering (2017, BSD-3-Clause) on lastlight's v0 atmosphere"
        ),
    }


def assemble(grid_name: str, cache_dir: Path | str, out: Path | str) -> str:
    """Write the table from every cached node; return its SHA-256."""
    grid = GRIDS[grid_name]
    cache = Path(cache_dir)
    blocks, shares = [], []
    for index in range(node_count(grid_name)):
        block, info = _load_node(grid_name, cache, index)
        blocks.append(block)
        shares.append(float(info["share_of_next_order"]))
    values = np.stack(blocks).reshape(len(grid.aod), len(grid.ozone), *blocks[0].shape)
    axes = {
        "aod_550": grid.aod,
        "ozone_du": grid.ozone,
        "sun_elevation_deg": grid.sun,
        "view_elevation_deg": grid.view,
        "rel_azimuth_deg": grid.azimuth,
    }
    data = v1.encode_table(axes, values, table_metadata(grid_name, max(shares)))
    digest = hashlib.sha256(data).hexdigest()
    v1.decode_table(data)  # the loader's own checks, before anything is written
    _write_atomically(Path(out), data)
    return digest


# --- the off-node reference --------------------------------------------------------------------


def reference(grid_name: str = "full") -> dict:
    """The direct computation at `OFF_NODE_POINT`, at the grid's resolution."""
    grid = GRIDS[grid_name]
    point = OFF_NODE_POINT
    atm = Atmosphere(aod_550=point["aod_550"], ozone_du=point["ozone_du"])
    field = bruneton.precompute(atm, resolution=grid.resolution)
    xyz = field.multiple_xyz(
        point["sun_elevation_deg"], [point["view_elevation_deg"]], point["rel_azimuth_deg"]
    )[0]
    return {
        "point": point,
        "resolution": grid.resolution.as_dict(),
        "orders": [2, grid.resolution.orders],
        "ground_albedo": bruneton.GROUND_ALBEDO,
        "xyz": [float(x) for x in xyz],
        "made_by": "tools/make_bruneton_table.py --reference",
    }


# --- convergence -------------------------------------------------------------------------------

CONVERGENCE_SUNS = tuple(float(e) for e in range(6, -7, -1))
CONVERGENCE_ATMOSPHERES = {"aod-0.03": 0.03, "aod-1.0": 1.0}


def _doubled(resolution: bruneton.Resolution, name: str) -> bruneton.Resolution:
    r = resolution
    variants = {
        "altitudes": {"altitudes": 2 * r.altitudes - 1},
        "views": {"views_sky": 2 * r.views_sky - 1, "views_ground": 2 * r.views_ground - 1},
        "sun_elevations": {"sun_elevations": 2 * r.sun_elevations - 1},
        "azimuths": {"azimuths": 2 * r.azimuths - 1},
        "single_samples": {"single_samples": 2 * r.single_samples},
        "march_samples": {"march_samples": 2 * r.march_samples},
        "sphere_zeniths": {"sphere_sky": 2 * r.sphere_sky, "sphere_ground": 2 * r.sphere_ground},
        "sphere_azimuths": {"sphere_azimuths": 2 * r.sphere_azimuths},
        "transmittance": {
            "transmittance_views": 2 * r.transmittance_views,
            "transmittance_altitudes": 2 * r.transmittance_altitudes,
        },
        "orders": {"orders": r.orders + 1},
        "baseline": {},
    }
    return replace(r, **variants[name])


CONVERGENCE_VARIANTS = (
    "baseline",
    "altitudes",
    "views",
    "sun_elevations",
    "azimuths",
    "single_samples",
    "march_samples",
    "sphere_zeniths",
    "sphere_azimuths",
    "transmittance",
    "orders",
)


def convergence_run(name: str, atmosphere: str, cache_dir: Path | str) -> Path:
    """Multiple-scattering XYZ at WHTM1's bands, sun +6 to -6, for one variant: cached JSON."""
    if name not in CONVERGENCE_VARIANTS:
        raise ValueError(f"unknown convergence variant {name!r}: {CONVERGENCE_VARIANTS}")
    resolution = _doubled(GRIDS["full"].resolution, name)
    atm = Atmosphere(aod_550=CONVERGENCE_ATMOSPHERES[atmosphere], ozone_du=290.0)
    start = time.perf_counter()
    field = bruneton.precompute(atm, resolution=resolution)
    xyz = field.xyz_block(CONVERGENCE_SUNS, WHTM1_VIEWS, [WHTM1_AZIMUTH])[:, :, 0, :]
    elapsed = time.perf_counter() - start
    path = Path(cache_dir) / "convergence" / _grid_key("full") / f"{atmosphere}-{name}.json"
    record = {
        "variant": name,
        "atmosphere": atmosphere,
        "resolution": resolution.as_dict(),
        "xyz": xyz.tolist(),
        "seconds": round(elapsed, 1),
    }
    _write_atomically(path, json.dumps(record, sort_keys=True).encode("utf-8"))
    return path


STORED_RUNS = {
    "stored-geometry": (0.03, 250.0),
    "stored-aod": (0.035, 250.0),
    "stored-ozone": (0.03, 290.0),
}
"""Direct computations that measure the committed table's own interpolation: at a node
atmosphere, halfway between its sun, view and azimuth nodes; and halfway between two aod nodes
and between the two ozone nodes, on the table's geometry nodes."""


def _midpoints(nodes: Sequence[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(nodes, nodes[1:], strict=False)]


def stored_run(name: str, cache_dir: Path | str) -> Path:
    """Direct values where the committed table must interpolate: cached JSON."""
    grid = GRIDS["full"]
    aod, ozone = STORED_RUNS[name]
    field = bruneton.precompute(Atmosphere(aod_550=aod, ozone_du=ozone), resolution=grid.resolution)
    record: dict = {"name": name, "aod_550": aod, "ozone_du": ozone}
    if name == "stored-geometry":
        record["sun"] = field.xyz_block(_midpoints(grid.sun), grid.view, grid.azimuth).tolist()
        record["view"] = field.xyz_block(grid.sun, _midpoints(grid.view), grid.azimuth).tolist()
        record["azimuth"] = field.xyz_block(grid.sun, grid.view, _midpoints(grid.azimuth)).tolist()
    else:
        record["nodes"] = field.xyz_block(grid.sun, grid.view, grid.azimuth).tolist()
    path = Path(cache_dir) / "convergence" / _grid_key("full") / f"{name}.json"
    _write_atomically(path, json.dumps(record, sort_keys=True).encode("utf-8"))
    return path


# --- an independent check: backward Monte Carlo -------------------------------------------------

MONTE_CARLO_GEOMETRIES = (
    (6.0, 1.51, WHTM1_AZIMUTH),
    (0.0, 1.51, WHTM1_AZIMUTH),
    (-2.0, 1.51, WHTM1_AZIMUTH),
    (-4.0, 1.51, WHTM1_AZIMUTH),
    (-6.0, 1.51, WHTM1_AZIMUTH),
    (6.0, 90.0, 0.0),
    (0.0, 90.0, 0.0),
    (-6.0, 90.0, 0.0),
    (6.0, 45.0, 180.0),
    (0.0, 45.0, 180.0),
    (3.0, 10.0, 90.0),
)
MONTE_CARLO_WAVELENGTH = 550.0


def _rotate(axis: np.ndarray, cosine: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unit vectors at angle acos(cosine) from each `axis`, at a uniformly random azimuth."""
    phi = 2.0 * math.pi * rng.random(axis.shape[0])
    sine = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
    helper = np.where(np.abs(axis[:, [2]]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(axis, first)
    return (
        axis * cosine[:, None]
        + first * (sine * np.cos(phi))[:, None]
        + second * (sine * np.sin(phi))[:, None]
    )


def monte_carlo(
    atm: Atmosphere,
    sun_deg: float,
    view_deg: float,
    azimuth_deg: float,
    *,
    photons: int,
    orders: int = 16,
    seed: int = 7,
) -> np.ndarray:
    """Radiance by bounce order at one wavelength, by backward Monte Carlo: shape (orders + 1,).

    Paths start at the ground observer along the sightline. Free paths are drawn by delta
    tracking against v0's point densities, the sun's light is added at every scattering or
    ground reflection (shadow test, transmittance by direct quadrature, no table), and a new
    direction is drawn from the phase function (or the Lambertian ground). The estimate added at
    the k-th bounce is order k. It shares only the atmosphere's density functions with the port.
    """
    rng = np.random.default_rng(seed)
    index = int(np.flatnonzero(bruneton.WAVELENGTHS_NM == MONTE_CARLO_WAVELENGTH)[0])
    optics = bruneton._optics(atm)
    extinction, scattering = optics.extinction[:, index], optics.scattering[:, index]
    scale, g, flux = optics.aerosol_scale_height_km, optics.asymmetry, optics.sun[index]
    radius, top = bruneton._R, bruneton._R_TOP
    sun = np.array([math.cos(math.radians(sun_deg)), 0.0, math.sin(math.radians(sun_deg))])
    e, a = math.radians(view_deg), math.radians(azimuth_deg)
    view = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    peak = bruneton._profiles(np.array([0.0, bruneton.OZONE_LAYER_CENTRE_KM]), scale)
    majorant = 1.0001 * (extinction[:2] @ peak[:2, 0] + extinction[2] * peak[2, 1])

    def sunward_transmittance(points: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(points, axis=1)
        p = points @ sun
        shadow = (p < 0.0) & (p * p - (r * r - radius * radius) > 0.0)
        length = -p + np.sqrt(np.maximum(p * p - (r * r - top * top), 0.0))
        steps = (np.arange(400) + 0.5) / 400.0
        s = length[:, None] * steps[None, :]
        h = np.sqrt(np.maximum(r[:, None] ** 2 + 2 * p[:, None] * s + s * s, radius**2)) - radius
        depth = np.einsum("knm,k->n", bruneton._profiles(h, scale), extinction) * length / 400
        return np.where(shadow, 0.0, np.exp(-depth))

    position = np.tile([0.0, 0.0, radius + 1e-6], (photons, 1))
    direction = np.tile(view, (photons, 1))
    weight = np.ones(photons)
    alive = np.ones(photons, bool)
    result = np.zeros(orders + 1)
    for order in range(1, orders + 1):
        live = np.flatnonzero(alive)
        here, going = position[live], direction[live]
        event = np.zeros(live.size, int)  # 0 flying, 1 scattered, 2 ground, 3 escaped
        while (event == 0).any():
            flying = np.flatnonzero(event == 0)
            p0, d0 = here[flying], going[flying]
            step = -np.log(rng.random(flying.size)) / majorant
            r0 = np.linalg.norm(p0, axis=1)
            b0 = np.einsum("ij,ij->i", p0, d0)
            disc = b0 * b0 - (r0 * r0 - radius * radius)
            to_ground = np.where(
                (b0 < 0) & (disc > 0), -b0 - np.sqrt(np.maximum(disc, 0.0)), np.inf
            )
            to_top = -b0 + np.sqrt(np.maximum(b0 * b0 - (r0 * r0 - top * top), 0.0))
            ground = to_ground <= np.minimum(step, to_top)
            escape = ~ground & (step >= to_top)
            travel = np.where(ground, to_ground, np.where(escape, to_top, step))
            moved = p0 + travel[:, None] * d0
            dens = bruneton._profiles(np.linalg.norm(moved, axis=1) - radius, scale)
            real = rng.random(flying.size) < (extinction @ dens) / majorant
            outcome = np.where(ground, 2, np.where(escape, 3, np.where(real, 1, 0)))
            here[flying] = moved
            event[flying] = outcome
        position[live] = here
        alive[live[event == 3]] = False
        hit = live[event == 1]
        if hit.size:
            dens = bruneton._profiles(np.linalg.norm(position[hit], axis=1) - radius, scale)
            sigma_r, sigma_a = scattering[0] * dens[0], scattering[1] * dens[1]
            total = extinction @ dens
            cosine = direction[hit] @ sun
            phase_r = 3 / (16 * math.pi) * (1 + cosine**2)
            phase_a = (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cosine) ** 1.5)
            light = flux * sunward_transmittance(position[hit])
            result[order] += np.sum(
                weight[hit] * (sigma_r * phase_r + sigma_a * phase_a) / total * light
            )
            weight[hit] *= (sigma_r + sigma_a) / total
            rayleigh = rng.random(hit.size) < sigma_r / (sigma_r + sigma_a)
            cosines = np.empty(hit.size)
            pending = np.ones(hit.size, bool)
            while pending.any():  # Rayleigh by rejection
                trial = rng.uniform(-1.0, 1.0, pending.sum())
                keep = rng.random(pending.sum()) < (1.0 + trial * trial) / 2.0
                chosen = np.flatnonzero(pending)[keep]
                cosines[chosen] = trial[keep]
                pending[chosen] = False
            xi = rng.random(hit.size)
            henyey = (1 + g * g - ((1 - g * g) / (1 - g + 2 * g * xi)) ** 2) / (2 * g)
            direction[hit] = _rotate(direction[hit], np.where(rayleigh, cosines, henyey), rng)
        floor = live[event == 2]
        if floor.size:
            normal = position[floor] / np.linalg.norm(position[floor], axis=1)[:, None]
            light = flux * sunward_transmittance(normal * (radius + 1e-6))
            albedo = bruneton.GROUND_ALBEDO
            result[order] += np.sum(
                weight[floor] * albedo / math.pi * light * np.maximum(normal @ sun, 0)
            )
            weight[floor] *= albedo
            direction[floor] = _rotate(normal, np.sqrt(rng.random(floor.size)), rng)
            position[floor] = normal * (radius + 1e-6)
    return result / photons


def monte_carlo_run(cache_dir: Path | str, photons: int = 300_000) -> Path:
    """The Monte Carlo against the port, at `MONTE_CARLO_GEOMETRIES`: cached JSON."""
    from lastlight.render import sky_spectrum

    atm = Atmosphere(aod_550=0.03, ozone_du=290.0)
    field = bruneton.precompute(atm, resolution=GRIDS["full"].resolution)
    index = int(np.flatnonzero(bruneton.WAVELENGTHS_NM == MONTE_CARLO_WAVELENGTH)[0])
    rows = []
    for sun, view, azimuth in MONTE_CARLO_GEOMETRIES:
        orders = monte_carlo(atm, sun, view, azimuth, photons=photons)
        rows.append(
            {
                "sun": sun,
                "view": view,
                "azimuth": azimuth,
                "mc_single": float(orders[1]),
                "mc_multiple": float(orders[2:].sum()),
                "v0_single": float(sky_spectrum(atm, sun, view, azimuth)[index]),
                "port_multiple": float(field.multiple_spectrum(sun, [view], azimuth)[0, index]),
            }
        )
    path = Path(cache_dir) / "convergence" / _grid_key("full") / "monte-carlo.json"
    record = {"photons": photons, "wavelength_nm": MONTE_CARLO_WAVELENGTH, "rows": rows}
    _write_atomically(path, json.dumps(record, sort_keys=True).encode("utf-8"))
    return path


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a / b - 1.0)))


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _interpolation_errors(
    table: v1.MultipleScatteringTable, directory: Path
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, relative errors, low-view mask) of the table against direct values off its nodes."""
    grid = GRIDS["full"]
    sun, view, azimuth = (np.array(axis) for axis in (grid.sun, grid.view, grid.azimuth))
    results = []
    path = directory / "stored-geometry.json"
    if path.exists():
        record = json.loads(path.read_text())
        aod, ozone = record["aod_550"], record["ozone_du"]
        cases = (
            ("sun elevation", _midpoints(grid.sun), grid.view, grid.azimuth, "sun"),
            ("view elevation", grid.sun, _midpoints(grid.view), grid.azimuth, "view"),
            ("relative azimuth", grid.sun, grid.view, _midpoints(grid.azimuth), "azimuth"),
        )
        for label, suns, views, azimuths, key in cases:
            direct = np.array(record[key])
            table_values = np.array(
                [[table.xyz(aod, ozone, s, views, a) for a in azimuths] for s in suns]
            ).transpose(0, 2, 1, 3)
            low = np.broadcast_to((np.array(views) <= 5.0)[None, :, None, None], direct.shape)
            results.append((f"halfway between {label} nodes", table_values / direct - 1, low))
    for name, label in (("stored-aod", "aod_550"), ("stored-ozone", "ozone_du")):
        path = directory / f"{name}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        direct = np.array(record["nodes"])
        table_values = np.array(
            [
                [table.xyz(record["aod_550"], record["ozone_du"], s, view, a) for a in azimuth]
                for s in sun
            ]
        ).transpose(0, 2, 1, 3)
        low = np.broadcast_to((view <= 5.0)[None, :, None, None], direct.shape)
        value = record[label]
        results.append((f"between {label} nodes ({label} {value})", table_values / direct - 1, low))
    return results


def _whtm1_share(atm: Atmosphere, sun: float) -> float:
    extra = v1.multiple_scattering_xyz(atm, sun, WHTM1_VIEWS[:1], WHTM1_AZIMUTH)[0, 1]
    total = sky_xyz(atm, sun, WHTM1_VIEWS[:1], WHTM1_AZIMUTH, engine="v1")[0, 1]
    return float(extra / total)


def _dome_share(atm: Atmosphere, sun: float) -> float:
    views = tuple(float(v) for v in range(2, 90, 4))
    weight = np.cos(np.radians(views))
    extra = total = 0.0
    for azimuth in range(0, 181, 15):
        extra += v1.multiple_scattering_xyz(atm, sun, views, azimuth)[:, 1] @ weight
        total += sky_xyz(atm, sun, views, azimuth, engine="v1")[:, 1] @ weight
    return float(extra / total)


def convergence_report(cache_dir: Path | str, table: v1.MultipleScatteringTable) -> str:
    """The Markdown report from the cached runs (convergence, table, Monte Carlo) and the table."""
    from lastlight.g0.contrast import log_chromaticity

    directory = Path(cache_dir) / "convergence" / _grid_key("full")
    nodes = sorted(node_directory("full", Path(cache_dir)).glob("node-*.json"))
    seconds = [json.loads(p.read_text())["seconds"] for p in nodes]
    shares = [json.loads(p.read_text())["share_of_next_order"] for p in nodes]
    resolution = GRIDS["full"].resolution
    size = TABLE_PATH.stat().st_size
    lines = [
        "# Renderer v1: convergence and checks of the Bruneton table",
        "",
        "Generated by `uv run python tools/make_bruneton_table.py --convergence-report` (issue",
        "#32) from the runs listed in the tool's docstring. Multiple scattering means the light",
        "reaching a ground observer after two or more bounces (scattering, or reflection by the",
        "ground with albedo 0.2), as CIE XYZ on v0's scale (Y = 1 for a white card under the",
        "extraterrestrial sun).",
        "",
        f"- Table: `src/lastlight/render/data/{v1.TABLE_NAME}`, {size} bytes,",
        f"  SHA-256 `{v1.TABLE_SHA256}`.",
        f"- Generation: {len(nodes)} nodes, {sum(seconds) / 60:.0f} minutes of node time in all,",
        f"  the slowest node {max(seconds):.0f} s (one `--node` per run, several runs at once).",
        "- WHTM1's geometry below: its five contrast bands and its reference band as view",
        f"  elevations {', '.join(f'{v:.2f}' for v in WHTM1_VIEWS)} degrees, relative azimuth",
        f"  {WHTM1_AZIMUTH} degrees (the sun at 0 on 2026-10-10), every sun elevation from +6 to",
        "  -6 in 1-degree steps, ozone 290 DU and the default atmosphere otherwise.",
        "",
        "## Precomputation resolution",
        "",
        "The table's resolution (`lastlight.render.bruneton.RESOLUTION`); each row of the next",
        "sections doubles one of these (or adds one order):",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    lines += [f"| `{key}` | {value} |" for key, value in resolution.as_dict().items()]
    for atmosphere, aod in CONVERGENCE_ATMOSPHERES.items():
        base_path = directory / f"{atmosphere}-baseline.json"
        if not base_path.exists():
            continue
        baseline = np.array(json.loads(base_path.read_text())["xyz"])
        atm = Atmosphere(aod_550=aod, ozone_du=290.0)
        single = np.array([sky_xyz(atm, s, WHTM1_VIEWS, WHTM1_AZIMUTH) for s in CONVERGENCE_SUNS])
        lines += [
            "",
            f"## Doubling each resolution, aod_550 {aod}",
            "",
            "Largest relative change, over the 13 sun elevations and WHTM1's 6 views, of the",
            "multiple-scattering X, Y and Z, and of v1's total Y (v0's single scattering plus the",
            "multiple scattering), when one setting is doubled or one order is added.",
            "",
            "| Doubled | X | Y | Z | total Y |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in CONVERGENCE_VARIANTS[1:]:
            path = directory / f"{atmosphere}-{name}.json"
            if not path.exists():
                lines.append(f"| `{name}` | not run | | | |")
                continue
            other = np.array(json.loads(path.read_text())["xyz"])
            change = [_relative(other[..., c], baseline[..., c]) for c in range(3)]
            total = _relative(single[..., 1] + other[..., 1], single[..., 1] + baseline[..., 1])
            lines.append(
                f"| `{name}` | " + " | ".join(_percent(c) for c in (*change, total)) + " |"
            )
        interpolated = np.array(
            [table.xyz(aod, 290.0, s, WHTM1_VIEWS, WHTM1_AZIMUTH) for s in CONVERGENCE_SUNS]
        )
        error = [_relative(interpolated[..., c], baseline[..., c]) for c in range(3)]
        lines += [
            "",
            "The committed table at the same points (ozone 290 between its two nodes, the views",
            "and the azimuth between theirs, and aod_550 0.03 between its nodes), against this",
            "direct",
            f"computation: largest relative difference X {_percent(error[0])}, Y",
            f"{_percent(error[1])}, Z {_percent(error[2])}.",
            "",
            "WHTM1's lowest band, log(B/G) contrast against the reference band (linear sRGB",
            "with G0's channel floor, `lastlight.g0.contrast.log_chromaticity`):",
            "",
            "| Sun | v0 | v1, direct | v1, table |",
            "| --- | --- | --- | --- |",
        ]
        for k, sun in enumerate(CONVERGENCE_SUNS):
            cells = []
            for xyz in (single[k], single[k] + baseline[k], single[k] + interpolated[k]):
                logs, _ = log_chromaticity(xyz)
                cells.append(f"{logs[0, 1] - logs[-1, 1]:.3f}")
            lines.append(f"| {sun:+.0f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Scattering orders",
        "",
        f"Orders 2 to {resolution.orders} are summed. Every node also computes order",
        f"{resolution.orders + 1} at sun 0, at WHTM1's band elevations and relative azimuths",
        f"{', '.join(f'{a:g}' for a in ORDER_CHECK_AZIMUTHS)} degrees. The largest share it",
        f"would add to any channel is {_percent(max(shares))} (at aod_550 1.0, where the haze",
        "makes the orders converge slowest), against the criterion of 1 percent; at aod_550",
        "0.1 and below it is under 0.005 percent.",
        "",
        "## The table's own interpolation",
        "",
        "Relative error of the table, interpolated linearly in the log of each channel, against",
        "direct computations where it must interpolate: halfway between nodes on each geometric",
        "axis at the node atmosphere aod_550 0.03 and ozone 250, and on its geometry nodes at",
        "an aod_550 and an ozone column between its nodes. Over the whole table, and over the",
        "views up to 5 degrees, where WHTM1's bands lie:",
        "",
        "| Where | largest | 99th percentile | largest, views 0 to 5 |",
        "| --- | --- | --- | --- |",
    ]
    for label, errors, low in _interpolation_errors(table, directory):
        magnitude = np.abs(errors)
        lines.append(
            f"| {label} | {_percent(magnitude.max())} | "
            f"{_percent(float(np.percentile(magnitude, 99)))} | {_percent(magnitude[low].max())} |"
        )
    reference = json.loads(REFERENCE_PATH.read_text())
    point = reference["point"]
    got = table.xyz(
        point["aod_550"],
        point["ozone_du"],
        point["sun_elevation_deg"],
        [point["view_elevation_deg"]],
        point["rel_azimuth_deg"],
    )[0]
    error = got / np.array(reference["xyz"]) - 1
    lines += [
        "",
        "At the off-node check point of the tests (aod_550 0.035, ozone 300, sun -3.25, view",
        "1.7, relative azimuth 27, off every node), the table against the direct computation",
        f"(`tests/fixtures/bruneton_offnode.json`): X {error[0]:+.2%}, Y {error[1]:+.2%}, Z",
        f"{error[2]:+.2%}; the criterion is 3 percent.",
    ]
    path = directory / "monte-carlo.json"
    if path.exists():
        record = json.loads(path.read_text())
        lines += [
            "",
            "## An independent check: backward Monte Carlo",
            "",
            f"`monte_carlo` in the tool traces {record['photons']} paths per geometry back from",
            "the observer through v0's point densities (delta tracking, the sun added at every",
            "bounce with a direct-quadrature transmittance and the exact shadow test, Rayleigh",
            "and Henyey-Greenstein phase sampling, the same Lambertian ground). It shares no",
            "table, quadrature or interpolation with the port. At",
            f"{record['wavelength_nm']:.0f} nm, aod_550 0.03 and ozone 290. Its sampling noise",
            "is about 1 to 2 percent where the low sky is lit, and larger at sun -6, where few",
            "paths reach the thin lit air high up (the single-scattering column compares the",
            "Monte Carlo with v0):",
            "",
            "| Sun | View | Azimuth | single, MC / v0 | multiple, port / MC | share, MC | share, "
            "port |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in record["rows"]:
            mc_share = row["mc_multiple"] / (row["mc_single"] + row["mc_multiple"])
            port_share = row["port_multiple"] / (row["v0_single"] + row["port_multiple"])
            lines.append(
                f"| {row['sun']:+.0f} | {row['view']:g} | {row['azimuth']:g} | "
                f"{row['mc_single'] / row['v0_single']:.3f} | "
                f"{row['port_multiple'] / row['mc_multiple']:.3f} | {mc_share:.3f} | "
                f"{port_share:.3f} |"
            )
    atm = Atmosphere(aod_550=0.03)
    suns = [float(e) for e in range(6, -13, -1)]
    lines += [
        "",
        "## The multiply scattered share as the sun sets",
        "",
        "The conductor's sign check said the multiply scattered share of the light rises as the",
        "sun sets from +6 to -6. Over the sky as a whole it does (the share of Y over the dome,",
        "weighted by solid angle, from the table and v0, by aod_550):",
        "",
        "| aod_550 | sun +6 | sun 0 | sun -6 | sun -9 | sun -12 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for aod in (0.0, 0.03, 0.3, 1.0):
        dome = Atmosphere(aod_550=aod)
        cells = " | ".join(f"{_dome_share(dome, s):.3f}" for s in (6.0, 0.0, -6.0, -9.0, -12.0))
        lines.append(f"| {aod} | {cells} |")
    whtm1 = [_whtm1_share(atm, s) for s in suns]
    lines += [
        "",
        "Direction by direction it does not. Toward the sun near the horizon the share climbs",
        "to a peak near sun -2 and then falls, as single-scattered light from distant air that",
        "is still in sunlight brightens along the low sightline (the twilight glow); high in the",
        "sky and at the zenith it dips slightly from +6 to about -3, as the ground's reflection",
        "of the direct sun fades, before multiple scattering takes over below -6. The Monte",
        "Carlo above shows the same peak. WHTM1's lowest band, aod_550 0.03, ozone 280, sun +6",
        "to -12:",
        "",
        "| " + " | ".join(f"{s:+.0f}" for s in suns) + " |",
        "| " + " | ".join("---" for _ in suns) + " |",
        "| " + " | ".join(f"{x:.2f}" for x in whtm1) + " |",
        "",
    ]
    return "\n".join(lines)


# --- command line ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", choices=sorted(GRIDS), default="full")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--node", type=int, help="compute one node of the grid")
    action.add_argument("--assemble", action="store_true", help="write the table from the nodes")
    action.add_argument("--list", action="store_true", help="list the nodes and which are cached")
    action.add_argument("--reference", action="store_true", help="the off-node direct value")
    action.add_argument("--convergence", metavar="VARIANT", help="one convergence run")
    action.add_argument("--stored", choices=sorted(STORED_RUNS), help="one table-accuracy run")
    action.add_argument("--monte-carlo", action="store_true", help="the Monte Carlo cross-check")
    action.add_argument("--convergence-report", action="store_true", help="write the report")
    parser.add_argument("--atmosphere", choices=sorted(CONVERGENCE_ATMOSPHERES), default="aod-0.03")
    parser.add_argument(
        "--out", type=Path, help="where --assemble, --reference or the report write"
    )
    args = parser.parse_args(argv)

    if args.node is not None:
        path = compute_node(args.grid, args.node, args.cache_dir)
        info = json.loads(path.with_suffix(".json").read_text())
        print(
            f"node {args.node}: aod_550 {info['aod_550']} ozone_du {info['ozone_du']}, "
            f"next order adds {100 * info['share_of_next_order']:.3f} percent, "
            f"{info['seconds']} s -> {path}"
        )
    elif args.assemble:
        out = args.out or (TABLE_PATH if args.grid == "full" else None)
        if out is None:
            parser.error("--assemble on the tiny grid needs --out")
        digest = assemble(args.grid, args.cache_dir, out)
        print(f"{out}: {Path(out).stat().st_size} bytes, sha256 {digest}")
    elif args.list:
        for index in range(node_count(args.grid)):
            atm = node_atmosphere(args.grid, index)
            values_path, info_path = _node_paths(args.grid, args.cache_dir, index)
            state = json.loads(info_path.read_text()) if info_path.exists() else None
            done = f"cached, {state['seconds']} s" if state else "missing"
            print(f"node {index}: aod_550 {atm.aod_550} ozone_du {atm.ozone_du}: {done}")
    elif args.reference:
        out = args.out or REFERENCE_PATH
        record = reference(args.grid)
        _write_atomically(Path(out), (json.dumps(record, indent=2) + "\n").encode("utf-8"))
        print(f"{out}: xyz {record['xyz']}")
    elif args.convergence:
        path = convergence_run(args.convergence, args.atmosphere, args.cache_dir)
        print(f"{path}: {json.loads(path.read_text())['seconds']} s")
    elif args.stored:
        print(f"{stored_run(args.stored, args.cache_dir)}: written")
    elif args.monte_carlo:
        print(f"{monte_carlo_run(args.cache_dir)}: written")
    elif args.convergence_report:
        out = args.out or REPORT_PATH
        text = convergence_report(args.cache_dir, v1.default_table())
        _write_atomically(Path(out), text.encode("utf-8"))
        print(f"{out}: written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
