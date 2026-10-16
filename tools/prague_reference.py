"""Fetch, build and evaluate the Prague Sky Model for renderer v1's daylight control (#34).

Usage (fetch needs the network; build needs a C++17 compiler, `c++` on the path):

    uv run python tools/prague_reference.py fetch            # model source and datasets
    uv run python tools/prague_reference.py build            # compile the evaluation driver
    uv run python tools/prague_reference.py fixture          # write the sun10.json fixture
    uv run python tools/prague_reference.py fixture --check  # re-evaluate, compare, write nothing
    uv run python tools/prague_reference.py report           # the results tables, from the fixture
    uv run python tools/prague_reference.py fixture --sun 0  # the sun0.json fixture (#45)
    uv run python tools/prague_reference.py report --sun 0   # rules A and B at sun 0, from fixtures
    uv run python tools/prague_reference.py fixture --sun -1 # the sun-1.json fixture (#49)
    uv run python tools/prague_reference.py fixture --sun 1  # the sun1.json fixture (#49)
    uv run python tools/prague_reference.py report --offset  # the offset rules, from all four
    uv run python tools/prague_reference.py report --depth   # the aerosol amount's evidence (#55)

Everything the model needs lives outside the repository, in `~/.cache/lastlight/prague/`
(`--cache DIR` overrides it): the reference implementation of Wilkie et al. (2021) and Vevoda
et al. (2022), Apache-2.0, Charles University, cloned at a pinned commit from its authors'
repository; the authors' datasets from the links in that repository's README (Google Drive,
public, no account, a "can't scan for viruses" page to click through for large files), checked
against pinned sizes and SHA-256 digests; and a small driver of ours (the C++ below, which only
calls the model's public API) compiled against the model's two source files. Nothing of the
model or its datasets is committed to this repository, and the GPL `skylight` port is neither
used nor imported.

The datasets. The "ground-level" file (103 MB) holds one observer altitude, zero, and the model
clamps any altitude to the dataset's range without a word, so a 340 m observer would silently be
evaluated at 0 m. The control needs 340 m, the reference point's height, so the primary
evaluation uses the full dataset (2.2 GB, 22 altitudes, the same other axes). The ground-level
file is fetched too and used as a cross-check: at altitude 0 it must give exactly the numbers the
full dataset gives.

What the fixture holds. Prague's spectral sky radiance (11 channels of 40 nm, 320 to 760 nm,
the channel value standing for the radiance at its centre, as the authors' own example renderer
treats it) becomes CIE XYZ by sampling it on the renderer's 10 nm grid, linearly between channel
centres and held flat beyond the first and last centre (the end bands, below 340 and above 740
nm, weigh under 0.1 percent of Y), then integrating with the repository's own colour-matching
functions and white-card normalisation (`lastlight.render.colorimetry.spectrum_to_xyz`). The
diagnostics record the other readings of the same model: piecewise-constant channels (the
model's own wavelength lookup), visibility interpolated linearly in 1 / V between the two
bracketing dataset nodes instead of linearly in V, and an observer at altitude 0.

The visibility. Prague takes the ground-level horizontal visibility V in km (its valid range is
20 to 131.8). v1's `aod_550` maps to it by Koschmieder's law at 550 nm, V = 3.912 / beta, with
beta the extinction at the ground: v1's Rayleigh extinction there (the vertical optical depth at
its pressure over its 8 km scale height) plus `aod_550` over its aerosol scale height. Nothing is
tuned: it is computed from `lastlight.render.atmosphere`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity
from lastlight.inputs.http import USER_AGENT
from lastlight.render import Atmosphere
from lastlight.render.atmosphere import (
    RAYLEIGH_SCALE_HEIGHT_KM,
    REFERENCE_WAVELENGTH_NM,
    STANDARD_PRESSURE_HPA,
    WAVELENGTHS_NM,
    ozone_absorption_per_atm_cm,
    rayleigh_optical_depth,
)
from lastlight.render.colorimetry import spectrum_to_xyz
from lastlight.render.curve import sky_xyz
from lastlight.render.v1 import DOMAIN, default_table, multiple_scattering_xyz

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "prague"
FIXTURE = FIXTURE_DIR / "sun10.json"
FIXTURE_SUN0 = FIXTURE_DIR / "sun0.json"
FIXTURE_SUN_MINUS1 = FIXTURE_DIR / "sun-1.json"
FIXTURE_SUN_PLUS1 = FIXTURE_DIR / "sun1.json"
DEFAULT_CACHE = Path.home() / ".cache" / "lastlight" / "prague"

MODEL_URL = "https://github.com/PetrVevoda/pragueskymodel.git"
MODEL_COMMIT = "2385c912e9051c1258013ff8c3ce2e19e10fb917"
"""The repository's last commit (2023-08-27, "a link to the Python version added to the readme")."""
SAFETY_ALTITUDE_M = 50.0
"""`PragueSkyModel.cpp` lifts every viewpoint by this much; the driver checks the result."""


@dataclass(frozen=True)
class Dataset:
    name: str
    drive_id: str
    size: int
    sha256: str


DATASETS = {
    "full": Dataset(
        "SkyModelDataset.dat",
        "1IShL7T3umxGOEFvyYGQpHKMTneXEvyTM",
        2397370196,
        "33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727",
    ),
    "ground": Dataset(
        "SkyModelDatasetGround.dat",
        "1IflyFZTJxC_N298yXq_2GK4ycIsVJZk6",
        107632816,
        "76bd619dc6dfcbc900c2996436dd6cf68197c03e3982fe5580f2b109ce1c71c2",
    ),
}
"""The authors' datasets, from the links in the model repository's README, as downloaded."""

# --- the control's configuration (fixed by issue #34) -------------------------------------------

SUN_ELEVATION_DEG = 10.0
OBSERVER_ALTITUDE_KM = 0.34
GROUND_ALBEDO = 0.2
REL_AZIMUTHS_DEG = (0.0, 90.0, 180.0)
DIAGNOSTIC_ELEVATIONS_DEG = (30.0, 60.0)
AEROSOL_OPTICAL_DEPTHS = (0.03, 0.1)
KOSCHMIEDER = 3.912
US_STANDARD_OZONE_DU = 345.0
"""Prague's atmosphere carries the U.S. Standard ozone profile; its column is about 345 DU."""
NOMINAL_VIEWS = 6


@dataclass(frozen=True)
class FixtureSpec:
    """One fixture: the sun, the relative azimuths, where it is written, how to remake it."""

    sun_elevation_deg: float
    rel_azimuths_deg: tuple[float, ...]
    path: Path
    command: str
    sun_must_be_node: bool = False
    """The sun must be a node of Prague's sun axis and of v1's table (the offset fixtures, #49)."""


SUN10 = FixtureSpec(
    SUN_ELEVATION_DEG,
    REL_AZIMUTHS_DEG,
    FIXTURE,
    "uv run python tools/prague_reference.py fixture",
)
"""#34's fixture. Its spec is what the tool did before there were specs: nothing here may change
a byte of `sun10.json`."""
SUN0 = FixtureSpec(
    0.0,
    (0.0, 30.0, 90.0, 180.0),
    FIXTURE_SUN0,
    "uv run python tools/prague_reference.py fixture --sun 0",
)
"""#45's fixture: the same eight views and two loads at sun 0, and relative azimuth 30 as well."""
SUN_MINUS1 = FixtureSpec(
    -1.0,
    (0.0, 30.0, 90.0, 180.0),
    FIXTURE_SUN_MINUS1,
    "uv run python tools/prague_reference.py fixture --sun -1",
    sun_must_be_node=True,
)
SUN_PLUS1 = FixtureSpec(
    1.0,
    (0.0, 30.0, 90.0, 180.0),
    FIXTURE_SUN_PLUS1,
    "uv run python tools/prague_reference.py fixture --sun 1",
    sun_must_be_node=True,
)
"""#49's fixtures: sun 0 with its neighbours one degree either side. Same views, azimuths and loads
as `sun0.json`. Both suns are nodes of Prague's sun axis and of v1's table, so neither model
interpolates in the sun there, as at sun 0; the builder refuses to write either otherwise."""
FIXTURE_SPECS = {spec.sun_elevation_deg: spec for spec in (SUN10, SUN0, SUN_MINUS1, SUN_PLUS1)}

DRIVER = r"""// lastlight's evaluation driver for the Prague Sky Model reference implementation.
// This file is lastlight's own code; it only calls the model's public API.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>

#include "PragueSkyModel.h"

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: %s dataset visibility_km albedo viewpoint_z_m sun_elevation_deg "
                     "single_visibility(0|1) mode(sky|transmittance)\n",
                     argv[0]);
        return 2;
    }
    const std::string dataset = argv[1];
    const double visibility = std::atof(argv[2]);
    const double albedo = std::atof(argv[3]);
    const double viewpointZ = std::atof(argv[4]);
    const double sunElevation = std::atof(argv[5]) * M_PI / 180.0;
    const bool single = std::atoi(argv[6]) != 0;
    const std::string mode = argv[7];

    PragueSkyModel model;
    try {
        model.initialize(dataset, single ? visibility : 0.0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "initialize failed: %s\n", error.what());
        return 1;
    }
    const PragueSkyModel::AvailableData data = model.getAvailableData();
    std::printf("# available channels=%d start=%.17g width=%.17g altitude=[%.17g,%.17g] "
                "elevation=[%.17g,%.17g] visibility=[%.17g,%.17g] albedo=[%.17g,%.17g]\n",
                data.channels, data.channelStart, data.channelWidth, data.altitudeMin,
                data.altitudeMax, data.elevationMin, data.elevationMax, data.visibilityMin,
                data.visibilityMax, data.albedoMin, data.albedoMax);

    double relAzimuth = 0.0;
    double viewElevation = 0.0;
    while (std::scanf("%lf %lf", &relAzimuth, &viewElevation) == 2) {
        const double azimuth = relAzimuth * M_PI / 180.0;
        const double elevation = viewElevation * M_PI / 180.0;
        // Z is up, the sun is at azimuth 0, the view at the relative azimuth.
        const PragueSkyModel::Vector3 direction(
            std::cos(azimuth) * std::cos(elevation), std::sin(azimuth) * std::cos(elevation),
            std::sin(elevation));
        const PragueSkyModel::Vector3 viewPoint(0.0, 0.0, viewpointZ);
        const PragueSkyModel::Parameters params =
            model.computeParameters(viewPoint, direction, sunElevation, 0.0, visibility, albedo);
        std::printf("%.17g %.17g %.17g %.17g %.17g", relAzimuth, viewElevation, params.altitude,
                    params.elevation * 180.0 / M_PI, params.gamma * 180.0 / M_PI);
        for (int channel = 0; channel < data.channels; ++channel) {
            const double wavelength = data.channelStart + (channel + 0.5) * data.channelWidth;
            const double value =
                mode == "transmittance"
                    ? model.transmittance(params, wavelength, std::numeric_limits<double>::max())
                    : model.skyRadiance(params, wavelength);
            std::printf(" %.9g", value);
        }
        std::printf("\n");
    }
    return 0;
}
"""


class PragueError(RuntimeError):
    """The reference model could not be fetched, built or evaluated as documented."""


# --- the mapping from v1's atmosphere to Prague's visibility --------------------------------------


def ground_extinction_per_km(atm: Atmosphere) -> float:
    """Extinction at the ground at 550 nm in v1's atmosphere: Rayleigh plus aerosol, per km.

    The Rayleigh coefficient is the vertical optical depth at the atmosphere's pressure over the
    8 km scale height of its exponential density; the aerosol's is `aod_550` over the aerosol's
    scale height. Both come from `lastlight.render.atmosphere`.
    """
    rayleigh = float(rayleigh_optical_depth(REFERENCE_WAVELENGTH_NM, atm.pressure_hpa))
    return rayleigh / RAYLEIGH_SCALE_HEIGHT_KM + atm.aod_550 / atm.aerosol_scale_height_km


def visibility_km(atm: Atmosphere) -> float:
    """Koschmieder's visual range at 550 nm for `ground_extinction_per_km`, in km."""
    return KOSCHMIEDER / ground_extinction_per_km(atm)


# --- from Prague's channels to XYZ ----------------------------------------------------------------


def channel_spectrum(
    values: Sequence[float], start_nm: float, width_nm: float, *, binned: bool = False
) -> np.ndarray:
    """Prague's channel values on the renderer's 10 nm grid, `WAVELENGTHS_NM`.

    By default each value stands for the radiance at its channel's centre (`start + (i + 1/2)
    width`) and the grid is read off linearly between centres, held flat beyond the first and last
    centre. With `binned`, a wavelength takes the value of the channel whose bin holds it and
    zero outside the dataset's range: the lookup `PragueSkyModel::skyRadiance` itself uses.
    """
    channels = np.asarray(values, dtype=float)
    if binned:
        index = np.floor((WAVELENGTHS_NM - start_nm) / width_nm).astype(int)
        inside = (index >= 0) & (index < channels.size)
        spectrum = np.zeros_like(WAVELENGTHS_NM)
        spectrum[inside] = channels[index[inside]]
        return spectrum
    centres = start_nm + (np.arange(channels.size) + 0.5) * width_nm
    return np.interp(WAVELENGTHS_NM, centres, channels)


def channels_to_xyz(
    values: Sequence[float], start_nm: float, width_nm: float, *, binned: bool = False
) -> np.ndarray:
    return spectrum_to_xyz(channel_spectrum(values, start_nm, width_nm, binned=binned))


# --- the dataset header ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axes:
    """The parameter nodes of a Prague dataset, read from the start of the file."""

    visibilities_km: tuple[float, ...]
    albedos: tuple[float, ...]
    altitudes_m: tuple[float, ...]
    elevations_deg: tuple[float, ...]
    channels: int
    channel_start_nm: float
    channel_width_nm: float


def read_axes(path: Path) -> Axes:
    """The axes of a dataset: visibilities, albedos, altitudes, elevations, spectral channels."""
    with path.open("rb") as handle:

        def read(fmt: str) -> tuple:
            size = struct.calcsize(fmt)
            data = handle.read(size)
            if len(data) != size:
                raise PragueError(f"{path.name}: the header ends early")
            return struct.unpack(fmt, data)

        def axis() -> tuple[float, ...]:
            (count,) = read("<i")
            return read(f"<{count}d")

        visibilities, albedos, altitudes, elevations = axis(), axis(), axis(), axis()
        (channels,) = read("<i")
        start, width = read("<2d")
    return Axes(visibilities, albedos, altitudes, elevations, channels, start, width)


# --- fetching -------------------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def fetch_model(cache: Path) -> Path:
    """Clone the model repository into `cache/src` and check out the pinned commit."""
    source = cache / "src"
    if not (source / ".git").exists():
        cache.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", MODEL_URL, str(source)])
    head = run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if head != MODEL_COMMIT:
        run(["git", "-C", str(source), "fetch", "--quiet", "origin"])
        run(["git", "-C", str(source), "checkout", "--quiet", "--detach", MODEL_COMMIT])
    head = run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if head != MODEL_COMMIT:
        raise PragueError(f"the model checkout is at {head}, not the pinned {MODEL_COMMIT}")
    return source


def _open(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=120)


def _confirmation_url(page: str) -> str:
    """The "download anyway" link of Google Drive's page for files too large to scan.

    The page is a form with hidden fields; submitting it as a GET is what the button does.
    """
    action = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page)
    if action is None:
        raise PragueError(
            "Google Drive did not offer the file: its page has no download form "
            "(a quota or permission notice?). Nothing was substituted."
        )
    fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
    return f"{action.group(1)}?{urllib.parse.urlencode(fields)}"


def fetch_dataset(cache: Path, dataset: Dataset) -> Path:
    """Download one dataset into `cache/dl` unless it is already there and matches its digest."""
    target = cache / "dl" / dataset.name
    if target.exists() and target.stat().st_size == dataset.size:
        if sha256_of(target) == dataset.sha256:
            return target
        raise PragueError(f"{target} has the right size and a different SHA-256: remove it")
    target.parent.mkdir(parents=True, exist_ok=True)
    first = f"https://drive.google.com/uc?export=download&id={dataset.drive_id}"
    with _open(first) as reply:
        head = reply.read(1 << 16)
        content_type = reply.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            url = _confirmation_url((head + reply.read()).decode("utf-8", "replace"))
        else:
            url = None
    part = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with _open(url or first) as reply, part.open("wb") as out:
        if reply.headers.get("Content-Type", "").startswith("text/html"):
            raise PragueError(f"{dataset.name}: Google Drive answered with a page, not a file")
        for block in iter(lambda: reply.read(1 << 22), b""):
            out.write(block)
            digest.update(block)
            total += len(block)
    if total != dataset.size or digest.hexdigest() != dataset.sha256:
        part.unlink()
        raise PragueError(
            f"{dataset.name}: got {total} bytes with SHA-256 {digest.hexdigest()}; "
            f"expected {dataset.size} and {dataset.sha256}"
        )
    part.rename(target)
    return target


def cmd_fetch(cache: Path) -> None:
    fetch_model(cache)
    print(f"model: {MODEL_URL} at {MODEL_COMMIT}")
    for key, dataset in DATASETS.items():
        path = fetch_dataset(cache, dataset)
        print(f"dataset {key}: {path.name} {dataset.size} bytes sha256 {dataset.sha256}")


# --- building and running the model ---------------------------------------------------------------


def binary_path(cache: Path) -> Path:
    return cache / "build" / "prague_eval"


def cmd_build(cache: Path) -> None:
    source = fetch_model(cache)
    build = cache / "build"
    build.mkdir(parents=True, exist_ok=True)
    driver = build / "driver.cpp"
    driver.write_text(DRIVER)
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise PragueError("no C++ compiler (c++, clang++ or g++) on the path")
    model_dir = source / "src"
    run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-DNDEBUG",
            "-I",
            str(model_dir),
            str(driver),
            str(model_dir / "PragueSkyModel.cpp"),
            "-o",
            str(binary_path(cache)),
        ]
    )
    print(f"built {binary_path(cache)} with {compiler}")


@dataclass(frozen=True)
class Evaluation:
    """Prague's output for a list of (relative azimuth, view elevation) queries."""

    channels: np.ndarray  # (queries, channel count): radiance or transmittance per channel
    start_nm: float
    width_nm: float


def evaluate(
    cache: Path,
    dataset: Dataset,
    *,
    visibility: float,
    altitude_m: float,
    queries: Sequence[tuple[float, float]],
    sun_elevation_deg: float,
    mode: str = "sky",
) -> Evaluation:
    """Run the driver at `sun_elevation_deg`, ground albedo 0.2, the model's own interpolation.

    Only the one or two dataset visibilities around `visibility` are loaded, which does not change
    the result. The model raises the viewpoint by `SAFETY_ALTITUDE_M`; the driver reports the
    altitude it ended up evaluating and this refuses any that is not the one asked for.
    """
    binary = binary_path(cache)
    if not binary.exists():
        raise PragueError(f"{binary} is not built: run `prague_reference.py build`")
    text = "".join(f"{azimuth!r} {elevation!r}\n" for azimuth, elevation in queries)
    command = [
        str(binary),
        str(cache / "dl" / dataset.name),
        repr(visibility),
        repr(GROUND_ALBEDO),
        repr(altitude_m - SAFETY_ALTITUDE_M),
        repr(sun_elevation_deg),
        "1",
        mode,
    ]
    result = subprocess.run(command, input=text, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise PragueError(f"the Prague driver failed: {result.stderr.strip()}")
    header = None
    rows = []
    for line in result.stdout.splitlines():
        if line.startswith("#"):
            header = line
            continue
        rows.append([float(part) for part in line.split()])
    if header is None or len(rows) != len(queries):
        raise PragueError("the Prague driver gave an unexpected number of lines")
    start = float(re.search(r"start=(\S+)", header).group(1))  # type: ignore[union-attr]
    width = float(re.search(r"width=(\S+)", header).group(1))  # type: ignore[union-attr]
    table = np.array(rows)
    if not np.allclose(table[:, 2], altitude_m, atol=1e-6):
        raise PragueError(f"the model evaluated at altitude {table[:, 2]}, not {altitude_m} m")
    if not np.allclose(table[:, 3], sun_elevation_deg, atol=1e-9):
        raise PragueError(f"the model's sun elevation is {table[:, 3]}, not {sun_elevation_deg}")
    return Evaluation(table[:, 5:], start, width)


# --- the fixture ----------------------------------------------------------------------------------


def significant(value: float, digits: int = 9) -> float:
    return float(f"{value:.{digits}g}")


def view_elevations() -> list[float]:
    nominal = [float(x) for x in build_views()["nominal"].elevations_deg]
    return [*nominal, *DIAGNOSTIC_ELEVATIONS_DEG]


def _queries(views: Sequence[float], azimuths: Sequence[float]) -> list[tuple[float, float]]:
    return [(azimuth, view) for azimuth in azimuths for view in views]


def check_sun_elevation(sun_elevation_deg: float, axes: Axes) -> None:
    """Refuse a sun elevation outside the dataset's sun axis (the model would clamp it silently)."""
    low, high = axes.elevations_deg[0], axes.elevations_deg[-1]
    if not (math.isfinite(sun_elevation_deg) and low <= sun_elevation_deg <= high):
        raise PragueError(
            f"sun elevation {sun_elevation_deg!r} is outside the dataset's sun axis, {low:g} to "
            f"{high:g} degrees: the model would clamp it without a word, so no fixture is made"
        )


SUN_NODE_TOLERANCE = 1e-9
"""A sun is a node of an axis when it is within this many degrees of one of its nodes."""


def check_sun_node(sun_elevation_deg: float, nodes: Sequence[float], axis: str) -> None:
    """Refuse a sun that is not a node of `axis` (to `SUN_NODE_TOLERANCE`), naming the axis.

    At a node a model does not interpolate in the sun, which is what the offset fixtures need: the
    curvature of the offset across sun -1, 0 and +1 is then not read through an interpolation.
    """
    on_a_node = math.isfinite(sun_elevation_deg) and any(
        abs(float(node) - sun_elevation_deg) <= SUN_NODE_TOLERANCE for node in nodes
    )
    if not on_a_node:
        raise PragueError(
            f"sun elevation {sun_elevation_deg!r} is not a node of {axis}: the fixture would "
            "interpolate in the sun there, so it is not made"
        )


def xyz_block(
    channels: np.ndarray,
    start: float,
    width: float,
    views: Sequence[float],
    azimuths: Sequence[float],
    *,
    binned: bool = False,
    where: str = "",
) -> dict[str, list[list[float]]]:
    """Rows ordered azimuth-major, as `_queries`, become {azimuth: [[X, Y, Z] per view]}.

    Prague's output is untrusted: a luminance (Y) that is not finite and above 0 is refused with
    an error that names the point (`where` says which reading and load it belongs to).
    """
    count = len(views)
    block: dict[str, list[list[float]]] = {}
    for i, azimuth in enumerate(azimuths):
        rows = channels[i * count : (i + 1) * count]
        xyz = [channels_to_xyz(row, start, width, binned=binned) for row in rows]
        for view, triple in zip(views, xyz, strict=True):
            if not (math.isfinite(triple[1]) and triple[1] > 0.0):
                raise PragueError(
                    f"Prague's luminance {float(triple[1])!r} is not finite and above 0 at "
                    f"{where}, relative azimuth {azimuth:g}, view {view:g} degrees"
                )
        block[f"{azimuth:g}"] = [[significant(v) for v in triple] for triple in xyz]
    return block


def _bracket(nodes: Sequence[float], value: float) -> tuple[float, float]:
    below = max(node for node in nodes if node <= value)
    above = min(node for node in nodes if node >= value)
    return below, above


def v1_gas_depth_at_550(pressure_hpa: float) -> tuple[float, float]:
    """v1's gas depth at 550 nm: its Rayleigh depth at `pressure_hpa` and its ozone depth.

    The ozone depth is for the 345 DU column of the U.S. Standard profile Prague uses, from v1's
    own absorption table. Returned as the pair (Rayleigh, ozone), never summed, so the callers
    keep their own order of subtraction.
    """
    rayleigh = float(rayleigh_optical_depth(REFERENCE_WAVELENGTH_NM, pressure_hpa))
    ozone = float(ozone_absorption_per_atm_cm(REFERENCE_WAVELENGTH_NM)) * US_STANDARD_OZONE_DU / 1e3
    return rayleigh, ozone


def aerosol_depth_from_transmittance(
    transmittance: Sequence[float], centres: Sequence[float]
) -> float:
    """Prague's vertical aerosol optical depth at 550 nm, from its zenith transmittance.

    The optical depth of all that the zenith ray meets, interpolated to 550 nm between the two
    channels around it, less the Rayleigh depth at v1's pressure and the ozone depth for a
    345 DU column (the U.S. Standard profile Prague uses), both from v1's own tables. The two
    subtractions are v1's numbers, not Prague's, so the result carries about +-0.01.
    """
    tau = -np.log(np.asarray(transmittance, dtype=float))
    total = float(np.interp(REFERENCE_WAVELENGTH_NM, centres, tau))
    rayleigh, ozone = v1_gas_depth_at_550(Atmosphere(0.0).pressure_hpa)
    return total - rayleigh - ozone


def matched_visibility_km(
    cache: Path,
    dataset: Dataset,
    aod: float,
    low: float,
    high: float,
    centres: Sequence[float],
    sun_elevation_deg: float,
) -> tuple[float, float, bool]:
    """The visibility at which Prague's own aerosol optical depth equals `aod`, by bisection.

    Returns (visibility, the depth Prague has there, clamped). A diagnostic for how much of the
    comparison is the Koschmieder mapping: where Prague cannot be that clear (its largest
    visibility still has more aerosol than `aod`) the result is that largest visibility with
    `clamped` set.
    """

    def depth(visibility: float) -> float:
        zenith = evaluate(
            cache,
            dataset,
            visibility=visibility,
            altitude_m=OBSERVER_ALTITUDE_KM * 1000.0,
            queries=[(0.0, 90.0)],
            sun_elevation_deg=sun_elevation_deg,
            mode="transmittance",
        )
        return aerosol_depth_from_transmittance(zenith.channels[0], centres)

    if depth(high) >= aod:
        return high, depth(high), True
    if depth(low) <= aod:
        raise PragueError(f"Prague at {low} km is clearer than aod_550 {aod}: unexpected")
    for _ in range(16):
        middle = 0.5 * (low + high)
        if depth(middle) > aod:
            low = middle
        else:
            high = middle
    visibility = 0.5 * (low + high)
    return visibility, depth(visibility), False


def build_fixture(cache: Path, spec: FixtureSpec = SUN10) -> dict:
    """Evaluate Prague for both atmospheres and every variant: the whole fixture, as a dict.

    The sun elevation and the relative azimuths come from `spec`; the default is #34's fixture.
    """
    fetch_model(cache)
    paths = {key: cache / "dl" / dataset.name for key, dataset in DATASETS.items()}
    for key, dataset in DATASETS.items():
        if not paths[key].exists() or paths[key].stat().st_size != dataset.size:
            raise PragueError(f"{paths[key]} is missing: run `prague_reference.py fetch`")
        if sha256_of(paths[key]) != dataset.sha256:
            raise PragueError(f"{paths[key]} does not have its pinned SHA-256")
    axes = read_axes(paths["full"])
    ground_axes = read_axes(paths["ground"])
    sun = spec.sun_elevation_deg
    azimuths = spec.rel_azimuths_deg
    check_sun_elevation(sun, axes)
    if spec.sun_must_be_node:
        check_sun_node(sun, axes.elevations_deg, "Prague's sun axis")
        check_sun_node(
            sun, default_table().axes["sun_elevation_deg"], "renderer v1's table sun axis"
        )
    views = view_elevations()
    queries = _queries(views, azimuths)
    low, high = axes.visibilities_km[0], axes.visibilities_km[-1]
    full, ground = DATASETS["full"], DATASETS["ground"]
    altitude_m = OBSERVER_ALTITUDE_KM * 1000.0

    atmospheres = {}
    ground_check = True
    for aod in AEROSOL_OPTICAL_DEPTHS:
        atm = Atmosphere(aod_550=aod)
        visibility = visibility_km(atm)
        if not low <= visibility <= high:
            raise PragueError(
                f"aod_550 {aod} maps to visibility {visibility:.2f} km, outside Prague's valid "
                f"range {low} to {high} km: no comparison is made for it"
            )
        primary = evaluate(
            cache,
            full,
            visibility=visibility,
            altitude_m=altitude_m,
            queries=queries,
            sun_elevation_deg=sun,
        )

        # Visibility interpolated in 1 / V (extinction) between the bracketing dataset nodes.
        below, above = _bracket(axes.visibilities_km, visibility)
        if below == above:
            blended = primary.channels
        else:
            weight = (1.0 / below - 1.0 / visibility) / (1.0 / below - 1.0 / above)
            lower = evaluate(
                cache,
                full,
                visibility=below,
                altitude_m=altitude_m,
                queries=queries,
                sun_elevation_deg=sun,
            )
            upper = evaluate(
                cache,
                full,
                visibility=above,
                altitude_m=altitude_m,
                queries=queries,
                sun_elevation_deg=sun,
            )
            blended = (1.0 - weight) * lower.channels + weight * upper.channels

        # Observer at altitude 0, from the full dataset and from the ground-level one.
        at_zero = evaluate(
            cache,
            full,
            visibility=visibility,
            altitude_m=0.0,
            queries=queries,
            sun_elevation_deg=sun,
        )
        at_zero_ground = evaluate(
            cache,
            ground,
            visibility=visibility,
            altitude_m=0.0,
            queries=queries,
            sun_elevation_deg=sun,
        )
        ground_check = ground_check and np.array_equal(at_zero.channels, at_zero_ground.channels)

        zenith = evaluate(
            cache,
            full,
            visibility=visibility,
            altitude_m=altitude_m,
            queries=[(0.0, 90.0)],
            sun_elevation_deg=sun,
            mode="transmittance",
        )
        centres = [
            axes.channel_start_nm + (i + 0.5) * axes.channel_width_nm for i in range(axes.channels)
        ]
        matched_v, matched_depth, clamped = matched_visibility_km(
            cache, full, aod, low, high, centres, sun
        )
        matched = evaluate(
            cache,
            full,
            visibility=matched_v,
            altitude_m=altitude_m,
            queries=queries,
            sun_elevation_deg=sun,
        )

        def block(
            reading: str,
            channels: np.ndarray,
            start: float,
            width: float,
            *,
            binned: bool = False,
            _aod: float = aod,
        ) -> dict[str, list[list[float]]]:
            return xyz_block(
                channels,
                start,
                width,
                views,
                azimuths,
                binned=binned,
                where=f"aod_550 {_aod:g}, reading {reading}",
            )

        atmospheres[f"{aod:g}"] = {
            "aod_550": aod,
            "ground_extinction_per_km": significant(ground_extinction_per_km(atm), 12),
            "visibility_km": significant(visibility, 12),
            "bracketing_visibility_nodes_km": [below, above],
            "xyz": block("xyz", primary.channels, primary.start_nm, primary.width_nm),
            "diagnostics": {
                "xyz_binned": block(
                    "xyz_binned",
                    primary.channels,
                    primary.start_nm,
                    primary.width_nm,
                    binned=True,
                ),
                "xyz_inverse_visibility": block(
                    "xyz_inverse_visibility", blended, primary.start_nm, primary.width_nm
                ),
                "xyz_altitude_0": block(
                    "xyz_altitude_0", at_zero.channels, at_zero.start_nm, at_zero.width_nm
                ),
                "zenith_transmittance": [significant(t) for t in zenith.channels[0]],
                "xyz_matched_aerosol_depth": block(
                    "xyz_matched_aerosol_depth",
                    matched.channels,
                    matched.start_nm,
                    matched.width_nm,
                ),
                "matched_aerosol_depth": {
                    "visibility_km": significant(matched_v, 9),
                    "prague_aerosol_depth": significant(matched_depth, 6),
                    "clamped_at_largest_visibility": clamped,
                },
            },
        }
    start, width = axes.channel_start_nm, axes.channel_width_nm
    return {
        "format": "lastlight-prague-control",
        "version": 1,
        "sun_elevation_deg": sun,
        "observer_altitude_km": OBSERVER_ALTITUDE_KM,
        "ground_albedo": GROUND_ALBEDO,
        "view_elevations_deg": views,
        "nominal_view_count": NOMINAL_VIEWS,
        "rel_azimuths_deg": list(azimuths),
        "prague_visibility_range_km": [low, high],
        "xyz_convention": (
            "CIE 1931 XYZ on the scale of lastlight.render.colorimetry: a white card lit by the "
            "extraterrestrial sun has Y = 1. Prague's channel values stand for the radiance at "
            "their centres and are read off linearly between centres, held flat beyond the first "
            "and last centre, on the renderer's 10 nm grid."
        ),
        "channel_centres_nm": [start + (i + 0.5) * width for i in range(axes.channels)],
        "atmospheres": atmospheres,
        "provenance": {
            "model": {
                "repository": MODEL_URL,
                "commit": MODEL_COMMIT,
                "licence": "Apache-2.0",
                "copyright": "Copyright 2022 Charles University",
            },
            "datasets": {
                key: {
                    "file": dataset.name,
                    "bytes": dataset.size,
                    "sha256": dataset.sha256,
                    "drive_id": dataset.drive_id,
                }
                for key, dataset in DATASETS.items()
            },
            "ground_level_dataset_matches_full_at_altitude_0": bool(ground_check),
            "dataset_axes": {
                "visibilities_km": list(axes.visibilities_km),
                "albedos": [significant(a, 6) for a in axes.albedos],
                "altitudes_m": [significant(a, 8) for a in axes.altitudes_m],
                "elevations_deg": list(axes.elevations_deg),
                "ground_level_altitudes_m": list(ground_axes.altitudes_m),
            },
            "command": spec.command,
        },
    }


def dump(data: dict) -> str:
    """The fixture as JSON text: indented, keys in the order written, one trailing newline."""
    return json.dumps(data, indent=1) + "\n"


def _compare(left: object, right: object, path: str, problems: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            problems.append(f"{path}: keys differ")
            return
        for key in left:
            _compare(left[key], right[key], f"{path}/{key}", problems)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            problems.append(f"{path}: lengths differ")
            return
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, f"{path}[{i}]", problems)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-12):
            problems.append(f"{path}: {left!r} against {right!r}")
    elif left != right:
        problems.append(f"{path}: {left!r} against {right!r}")


def fixture_spec(sun_elevation_deg: float, output: Path | None = None) -> FixtureSpec:
    """The registered fixture for a sun elevation; `output` moves where it is written or read."""
    base = FIXTURE_SPECS.get(sun_elevation_deg)
    if base is None:
        known = ", ".join(f"{sun:g}" for sun in FIXTURE_SPECS)
        raise PragueError(
            f"no fixture is registered for sun elevation {sun_elevation_deg!r} "
            f"(registered: {known})"
        )
    if output is None:
        return base
    return replace(base, path=output)


def _shown(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def cmd_fixture(cache: Path, spec: FixtureSpec = SUN10, *, check: bool) -> int:
    if check and not spec.path.exists():
        raise PragueError(f"{spec.path} does not exist: nothing to check")
    fresh = build_fixture(cache, spec)
    if not check:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        spec.path.write_text(dump(fresh))
        print(f"wrote {_shown(spec.path)} ({spec.path.stat().st_size} bytes)")
        return 0
    committed = json.loads(spec.path.read_text())
    problems: list[str] = []
    _compare(committed, fresh, "", problems)
    for problem in problems[:20]:
        print(problem)
    print(f"{len(problems)} difference(s) between the committed fixture and a fresh evaluation")
    return 1 if problems else 0


# --- the report -----------------------------------------------------------------------------------

# The aod_550 0.1 rule (decided before any comparison was looked at). v1's orders 2 and up sit
# 6.6 percent above the tool's own Monte Carlo at aod_550 0.1 and agree at 0.03. If Prague sides
# with the Monte Carlo, v1 at 0.1 is high against Prague by about 6.6 percent of its multiply
# scattered share, relative to how it stands at 0.03; if with the port, it is not. The two
# predictions are told apart only if the predicted gap exceeds both the spread of the observed
# ratio across the 24 views and azimuths and the shift that reading Prague's visibility in 1 / V
# instead of V makes to it; otherwise the answer is "cannot tell".
MONTE_CARLO_EXCESS = 0.066


# --- the sun-0 decision rules (#45) ---------------------------------------------------------------
#
# Registered in `docs/renderer/prague-sun0-criterion.md`, in the commit that landed that file and
# before any Prague number at sun 0 existed. The functions below take arrays and numbers and return
# the verdict and the branch that produced it; they read no fixture and run no model. A
# disagreement between that file and this code is a defect in this code.

UNCERTAINTY_BAR_FACTOR = 2.0
"""Rule A: the two predictions are separated only if E exceeds this many U (the 2U bar)."""
STD_DDOF = 1
"""Rule A: s is the sample standard deviation (ddof = 1)."""
RULE_A_AZIMUTHS_DEG = (90.0, 180.0)
"""Rule A's point set P: these relative azimuths, all eight views: 16 points."""
RULE_A_ALTERNATIVE_READINGS = ("xyz_inverse_visibility", "xyz_matched_aerosol_depth")
"""The two ways of setting Prague's aerosol that make up m. `xyz_binned` (a spectral reading) and
`xyz_altitude_0` (an observer altitude) are deliberately not among them."""
RULE_A_REPORTED_READINGS = ("xyz_binned", "xyz_altitude_0")
"""Reported beside rule A's verdict, never part of m."""

V1_ASYMMETRY_K = 70
"""v1's Henyey-Greenstein asymmetry, 0.7, in integer hundredths: the k that g = 0.7 means."""
ASYMMETRY_GRID_K = tuple(range(30, 91))
"""Rule B's grid: g = k / 100 for the 61 integers k from 30 to 90. Bounds are compared on k."""
LOBE_FACTOR = 0.5
"""Rule B: the fitted asymmetry must cut the excess to at most this share of v1's own."""
EXCESS_FLOOR = 0.05
"""Rule B: a sun-0 condition counts only if v1's own r there is at least this (5 percent rms)."""

CANNOT_TELL = "cannot tell"
PORT = "port"
MONTE_CARLO = "Monte Carlo"
NO_EXCESS = "no excess at sun 0"
AEROSOL_LOBE = "aerosol lobe"
NEITHER = "neither"


def asymmetry_of(k: int) -> float:
    """The Henyey-Greenstein asymmetry of grid index `k` (integer hundredths)."""
    return k / 100


def rule_a_verdict(D: float, E: float, U: float) -> tuple[str, int]:
    """Rule A's verdict and the branch (1 to 4) that produced it, in the registered order.

    1. E <= 2U: cannot tell (the predictions are not separated by twice the uncertainty).
    2. else |D| <= U: port.
    3. else |D - E| <= U: Monte Carlo.
    4. else: cannot tell (neither prediction lies within U of D).

    Every bound is inclusive on the side that is the less decisive for the rule's claim: E exactly
    2U is "cannot tell", D exactly U from either prediction is a call. `2.0 * U` is exact.
    """
    if E <= UNCERTAINTY_BAR_FACTOR * U:
        return CANNOT_TELL, 1
    if abs(D) <= U:
        return PORT, 2
    if abs(D - E) <= U:
        return MONTE_CARLO, 3
    return CANNOT_TELL, 4


@dataclass(frozen=True)
class RuleA:
    """Rule A's statistics and verdict. `D_alternatives` is D under each alternative reading."""

    D: float
    E: float
    s: float
    m: float
    U: float
    verdict: str
    branch: int
    D_alternatives: dict[str, float]


def _vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def rule_a(
    d: Sequence[float], e: Sequence[float], d_alternatives: dict[str, Sequence[float]]
) -> RuleA:
    """Rule A on the rule set's points: D, E, s, m, U, the verdict and the branch.

    `d` and `e` are d_i and e_i at the points of P (primary reading); `d_alternatives` maps each
    alternative reading to its own d_i at the same points. D and E are the means of d and e;
    s the sample standard deviation of d (ddof = 1); m the largest |D_r - D| over the
    alternative readings; U = max(s, m). All comparisons use these unrounded values.
    """
    d_all = _vector("d", d)
    e_all = _vector("e", e)
    others = {name: _vector(f"d under {name}", values) for name, values in d_alternatives.items()}
    if not others:
        raise ValueError("rule A needs at least one alternative reading to set m")
    if e_all.size != d_all.size or any(v.size != d_all.size for v in others.values()):
        raise ValueError("d, e and every alternative reading need the same number of points")
    for name, values in (("d", d_all), ("e", e_all), *others.items()):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite at every point")
    if d_all.size < 2:
        raise ValueError("rule A needs at least two points (s has ddof = 1)")
    D = float(np.mean(d_all))
    E = float(np.mean(e_all))
    s = float(np.std(d_all, ddof=STD_DDOF))
    shifts = {name: float(np.mean(values)) for name, values in others.items()}
    m = max(abs(shift - D) for shift in shifts.values())
    U = max(s, m)
    verdict, branch = rule_a_verdict(D, E, U)
    return RuleA(D, E, s, m, U, verdict, branch, shifts)


def best_asymmetry_k(r_by_k: dict[int, float]) -> int:
    """g*'s grid index: the k that minimises r on the grid.

    If two grid values give exactly the same r, the one nearer k = 70 wins, and at equal distance
    the lower one. `r_by_k` must hold r for every k of the grid and nothing else.
    """
    if set(r_by_k) != set(ASYMMETRY_GRID_K):
        raise ValueError("r must be given for exactly the grid k = 30 to 90, nothing else")
    if not all(math.isfinite(r) for r in r_by_k.values()):
        raise ValueError("r must be finite at every grid value")
    return min(ASYMMETRY_GRID_K, key=lambda k: (r_by_k[k], abs(k - V1_ASYMMETRY_K), k))


def excess(model_y: Sequence[float], prague_y: Sequence[float]) -> tuple[float, float]:
    """(r, signed mean) of ln(model / Prague) over the given views, in luminance.

    r is the root mean square of the logarithms: unsigned. The signed mean is their arithmetic
    mean. Both refuse a luminance that is not finite and above 0.
    """
    model = _vector("model luminance", model_y)
    prague = _vector("Prague luminance", prague_y)
    if model.size != prague.size or model.size == 0:
        raise ValueError("model and Prague luminance need the same, non-zero number of views")
    for name, values in (("model", model), ("Prague", prague)):
        if not (np.isfinite(values).all() and (values > 0.0).all()):
            raise ValueError(f"{name} luminance must be finite and above 0 at every view")
    logs = np.log(model / prague)
    return float(np.sqrt(np.mean(logs**2))), float(np.mean(logs))


@dataclass(frozen=True)
class RuleB:
    """Rule B's verdict and branch (1 to 3), with the per-condition tests behind it."""

    verdict: str
    branch: int
    counted: tuple[bool, ...]
    halved: tuple[bool, ...]


def rule_b(r_base: Sequence[float], r_fit: Sequence[float]) -> RuleB:
    """Rule B on the sun-0 conditions: `r_base` is v1's own r (g = 0.7), `r_fit` r at g*(a).

    A condition counts only if r_base >= 0.05. Then, in order:

    1. no condition counts: no excess at sun 0.
    2. aerosol lobe, if for every counted condition r_fit <= 0.5 * r_base.
    3. else neither (not the port either: see `port_bound`).

    With one counted condition the lobe test is made on that one alone, deliberately.
    """
    base = _vector("r_base", r_base)
    fit = _vector("r_fit", r_fit)
    if base.size != fit.size:
        raise ValueError("r_base and r_fit need the same number of conditions")
    for name, values in (("r_base", base), ("r_fit", fit)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        if (values < 0.0).any():
            raise ValueError(f"{name} is a root mean square and cannot be negative")
    counted = tuple(bool(r >= EXCESS_FLOOR) for r in base)
    halved = tuple(bool(f <= LOBE_FACTOR * r) for r, f in zip(base, fit, strict=True))
    if not any(counted):
        return RuleB(NO_EXCESS, 1, counted, halved)
    if all(h for c, h in zip(counted, halved, strict=True) if c):
        return RuleB(AEROSOL_LOBE, 2, counted, halved)
    return RuleB(NEITHER, 3, counted, halved)


def port_bound(total_y: Sequence[float], multiple_y: Sequence[float]) -> np.ndarray:
    """b = ln(T / (T - M * 0.066 / 1.066)) at each view: the most the port adds to ln(v1 / Prague).

    T and M are v1's total and multiple-scattering luminance: 6.6 percent of the multiple
    scattering, the gap at `aod_550` 0.1, charged at every load. It is never negative: the port can
    only make v1 too bright.
    """
    total = _vector("total luminance", total_y)
    multiple = _vector("multiple-scattering luminance", multiple_y)
    if total.size != multiple.size:
        raise ValueError("total and multiple-scattering luminance need the same number of views")
    if not (np.isfinite(total).all() and np.isfinite(multiple).all()):
        raise ValueError("luminance must be finite")
    if (multiple < 0.0).any() or (multiple >= total).any():
        raise ValueError("multiple-scattering luminance must be in [0, total)")
    port = multiple * MONTE_CARLO_EXCESS / (1.0 + MONTE_CARLO_EXCESS)
    return np.log(total / (total - port))


def port_bound_rms(total_y: Sequence[float], multiple_y: Sequence[float]) -> float:
    """B: the root mean square of `port_bound` over the views."""
    return float(np.sqrt(np.mean(port_bound(total_y, multiple_y) ** 2)))


# --- the offset decision rules (#49) --------------------------------------------------------------
#
# Registered in `docs/renderer/prague-offset-criterion.md`, in the commit that landed that file and
# before Prague had been evaluated at sun -1 or +1. The functions below take numbers and return the
# verdict and the branch that produced it; they read no fixture and run no model. A disagreement
# between that file and this code is a defect in this code.

OFFSET_AZIMUTHS_DEG = (90.0, 180.0)
"""The offset's point set P: these relative azimuths, all eight views: 16 points."""
OFFSET_RULED_SUNS_DEG = (-1.0, 1.0)
"""The suns at which Prague was first evaluated for rules 1 and 3; sun 0 and +10 are the pilot."""
OFFSET_BAR_FACTOR = 2.0
"""The noise bar is this many times the reading noise m (rule 1: 2m; rule 3: 2 m_F)."""
CARRIES_SHARE = 0.5
"""A candidate carries a condition when the part it removes is at least this share of the offset."""
FIT_FLOOR = 0.05
"""Rule 3's floor under its noise bar: smooth causes bend the offset by up to about 0.042."""
DEPTH_ERROR = 0.01
"""Rule 4's reported range: the aerosol depth estimate off by this much either way (no rule)."""
SLAB_FACTOR = math.exp(OBSERVER_ALTITUDE_KM / Atmosphere(0.0).aerosol_scale_height_km)
"""Rule 2's stand-in for the 340 m slab of air and aerosol under Prague's observer (no rule)."""
DEEPER_SCALE_HEIGHT_KM = 2.0
"""Rule 4's reported growth: v0's aerosol layer deepened from v1's 1.5 km to this (no rule)."""

CARRIES = "carries"
CONTRIBUTES = "contributes"
NOT_THIS = "not this"
CONDITION_VERDICTS = (CARRIES, CONTRIBUTES, NOT_THIS)
"""What one condition can say. A rule adds `CANNOT_TELL` (above) when its conditions disagree."""


def condition_verdict(removed: float, offset: float, bar: float) -> str:
    """One condition's verdict for a removed part, an offset and a bar, in the registered order.

    1. carries, if removed >= offset / 2 and removed > bar;
    2. else contributes, if removed > bar;
    3. else not this: removed <= bar, which includes every removed <= 0 (the candidate moves the
       models apart, or does not move them by more than the noise).

    The bar is strict (a part exactly at the bar is not over it) and the half is inclusive (a
    part exactly half the offset carries). `0.5 * offset` is exact.
    """
    if removed > bar:
        return CARRIES if removed >= CARRIES_SHARE * offset else CONTRIBUTES
    return NOT_THIS


def rule_verdict(verdicts: Iterable[str]) -> tuple[str, int]:
    """A rule's verdict over its conditions and the branch (1 to 4) that produced it, in order.

    1. every condition carries: carries.
    2. else every condition carries or contributes: contributes.
    3. else every condition is not this: not this.
    4. else: cannot tell, because the conditions disagree.

    It does not depend on the order of the conditions, and refuses an empty sequence.
    """
    seen = list(verdicts)
    if not seen:
        raise ValueError("a rule's verdict needs at least one condition: no conditions were given")
    for verdict in seen:
        if verdict not in CONDITION_VERDICTS:
            allowed = ", ".join(CONDITION_VERDICTS)
            raise ValueError(f"{verdict!r} is not a condition's verdict (one of: {allowed})")
    if all(verdict == CARRIES for verdict in seen):
        return CARRIES, 1
    if all(verdict in (CARRIES, CONTRIBUTES) for verdict in seen):
        return CONTRIBUTES, 2
    if all(verdict == NOT_THIS for verdict in seen):
        return NOT_THIS, 3
    return CANNOT_TELL, 4


def bend(at_minus: float, at_zero: float, at_plus: float) -> float:
    """How far the centre sits above the straight line through its neighbours: kappa.

    kappa = at_zero - (at_minus + at_plus) / 2, the neighbours 1 degree of sun either side.
    """
    return at_zero - (at_minus + at_plus) / 2.0


def reading_noise(offset: float, offset_inv: float) -> float:
    """m: how far the offset moves when Prague's aerosol is interpolated in 1 / V instead of V."""
    return abs(offset_inv - offset)


def fit_bar(m_minus: float, m_zero: float, m_plus: float) -> float:
    """Rule 3's bar: max(2 m_F, 0.05), m_F the largest reading noise at sun -1, 0 and +1."""
    return max(OFFSET_BAR_FACTOR * max(m_minus, m_zero, m_plus), FIT_FLOOR)


def offset_of(model_y: Sequence[float], prague_y: Sequence[float]) -> float:
    """The offset L over the given points: the mean of ln(model / Prague), in luminance.

    Signed, and a mean (logarithms of opposite sign cancel). Refuses a luminance that is not
    finite and above 0, and a different number of points on the two sides.
    """
    model = _vector("model luminance", model_y)
    prague = _vector("Prague luminance", prague_y)
    if model.size != prague.size or model.size == 0:
        raise ValueError("model and Prague luminance need the same, non-zero number of points")
    for name, values in (("model", model), ("Prague", prague)):
        if not (np.isfinite(values).all() and (values > 0.0).all()):
            raise ValueError(f"{name} luminance must be finite and above 0 at every point")
    return float(np.mean(np.log(model / prague)))


def refuse_no_excess(offset: float, condition: str, reading: str) -> None:
    """Refuse an offset that is not above 0: there is no excess to attribute.

    No condition is ever dropped, so the report is not written at all: the error names the
    condition and the reading.
    """
    if not (math.isfinite(offset) and offset > 0.0):
        raise PragueError(
            f"the offset L under reading {reading} at {condition} is {offset!r}, not above 0: "
            "there is no excess to attribute, so the report is not written"
        )


# --- the engines: v1 as it is, and the lobe diagnostic --------------------------------------------
#
# The sun elevation is always an argument. The fixture being evaluated says what it is; the module's
# `SUN_ELEVATION_DEG` is only the default of the sun +10 fixture and nothing here reads it.

GAP_AOD_550 = 0.1
"""The load at which the port's multiple scattering sits 6.6 percent above the Monte Carlo."""
TOWARD_SUN_AZIMUTH_DEG = 0.0
"""Rule B's views are the nominal ones at this relative azimuth (the solar vertical, sunward)."""


def engine_xyz(
    aod: float, sun_elevation_deg: float, views: Sequence[float], azimuth: float, engine: str
) -> np.ndarray:
    """`sky_xyz` for `Atmosphere(aod_550=aod)`, every other field at its default."""
    return sky_xyz(Atmosphere(aod_550=aod), sun_elevation_deg, views, azimuth, engine=engine)


def lobe_xyz(
    aod_550: float,
    asymmetry: float,
    sun_elevation_deg: float,
    views: Sequence[float],
    rel_azimuth_deg: float,
) -> np.ndarray:
    """XYZ_g, the lobe diagnostic: v0's single scattering at `asymmetry`, plus v1's table.

    v0's single scattering is the registered quadrature at `Atmosphere(aod_550, asymmetry)`; the
    multiple scattering is the table's, whose asymmetry is baked at 0.7, so it does not change
    with g. v1 itself refuses an asymmetry other than 0.7 and is never called with one. At
    g = 0.7 this equals v1's XYZ exactly (the same two arrays, added in the same order).
    """
    single = sky_xyz(
        Atmosphere(aod_550=aod_550, asymmetry=asymmetry),
        sun_elevation_deg,
        views,
        rel_azimuth_deg,
        engine="v0",
    )
    multiple = multiple_scattering_xyz(
        Atmosphere(aod_550=aod_550), sun_elevation_deg, views, rel_azimuth_deg
    )
    return single + multiple


V1_OZONE_DU = float(Atmosphere(0.0).ozone_du)
"""v1's ozone column, 280 DU: the table interpolates between 250 and 330 and v1 is run at this."""


def ozone_xyz(
    aod_550: float,
    ozone_du: float,
    sun_elevation_deg: float,
    views: Sequence[float],
    rel_azimuth_deg: float,
) -> np.ndarray:
    """XYZ_oz(O), the ozone diagnostic: v0's single scattering at O, plus v1's table continued.

    v0's single scattering is the registered quadrature at `Atmosphere(aod_550, ozone_du)`, which
    takes any ozone. The multiple scattering continues the table's own rule past its two ozone
    nodes (the domain's bounds, 250 and 330 DU): per channel, exp((1 - t) ln MS_lo + t ln MS_hi)
    with t = (O - lo) / (hi - lo), MS_x the table at ozone x. v1 interpolates log-linearly in
    every axis and the ozone axis has exactly those two nodes, so at O = 280 this is v1's XYZ; at
    O = 345 it carries v1's own line 15 DU past the last node. v1 itself is never asked for an
    ozone outside its table.
    """
    low, high = DOMAIN["ozone_du"]
    t = (ozone_du - low) / (high - low)
    single = sky_xyz(
        Atmosphere(aod_550=aod_550, ozone_du=ozone_du),
        sun_elevation_deg,
        views,
        rel_azimuth_deg,
        engine="v0",
    )
    at_low = multiple_scattering_xyz(
        Atmosphere(aod_550=aod_550, ozone_du=low), sun_elevation_deg, views, rel_azimuth_deg
    )
    at_high = multiple_scattering_xyz(
        Atmosphere(aod_550=aod_550, ozone_du=high), sun_elevation_deg, views, rel_azimuth_deg
    )
    return single + np.exp((1.0 - t) * np.log(at_low) + t * np.log(at_high))


def _over_points(luminance, azimuths: Sequence[float]) -> np.ndarray:
    """`luminance(azimuth)` (one value per view) at each relative azimuth, joined: the points.

    The points are the given azimuths, all the views: chosen by value, never by position.
    """
    chosen = [float(azimuth) for azimuth in azimuths]
    if not chosen:
        raise ValueError("at least one relative azimuth is needed to take the points")
    return np.concatenate([luminance(azimuth) for azimuth in chosen])


def ozone_part(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """Delta_O: the mean over the points of ln(Y_v1 / Y_oz(345)): what goes with Prague's ozone."""
    v1 = _over_points(
        lambda az: engine_xyz(aod_550, sun_elevation_deg, views, az, "v1")[:, 1], azimuths
    )
    prague_ozone = _over_points(
        lambda az: ozone_xyz(aod_550, US_STANDARD_OZONE_DU, sun_elevation_deg, views, az)[:, 1],
        azimuths,
    )
    return offset_of(v1, prague_ozone)


def _single_scattering_y(
    aod_550: float,
    ozone_du: float,
    sun_elevation_deg: float,
    views: Sequence[float],
    azimuths: Sequence[float],
) -> np.ndarray:
    return _over_points(
        lambda az: sky_xyz(
            Atmosphere(aod_550=aod_550, ozone_du=ozone_du),
            sun_elevation_deg,
            views,
            az,
            engine="v0",
        )[:, 1],
        azimuths,
    )


def ozone_part_single_scattering(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """Delta_O from v0's single scattering alone: mean ln(Y_v0(280) / Y_v0(345)) over the points."""
    at_v1 = _single_scattering_y(aod_550, V1_OZONE_DU, sun_elevation_deg, views, azimuths)
    at_prague = _single_scattering_y(
        aod_550, US_STANDARD_OZONE_DU, sun_elevation_deg, views, azimuths
    )
    return offset_of(at_v1, at_prague)


def ozone_line_ratio(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """The check on the ozone line: v0's ln(Y(280) / Y(345)) over ln(Y(280) / Y(330)), as means.

    v0 takes any ozone exactly. A straight line in ozone, which is what the continuation assumes
    of the multiple scattering, gives 65 / 50 = 1.3. A check, not a rule.
    """
    at_v1 = _single_scattering_y(aod_550, V1_OZONE_DU, sun_elevation_deg, views, azimuths)
    at_prague = _single_scattering_y(
        aod_550, US_STANDARD_OZONE_DU, sun_elevation_deg, views, azimuths
    )
    at_node = _single_scattering_y(
        aod_550, DOMAIN["ozone_du"][1], sun_elevation_deg, views, azimuths
    )
    return offset_of(at_v1, at_prague) / offset_of(at_v1, at_node)


def v1_level(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """v1's own sky: the mean over the points of ln Y_v1 (no Prague)."""
    y = _over_points(
        lambda az: engine_xyz(aod_550, sun_elevation_deg, views, az, "v1")[:, 1], azimuths
    )
    return float(np.mean(np.log(y)))


def slab_change(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """Rule 2's stand-in for the slab under Prague's observer: v1 with `aod_550` times SLAB_FACTOR.

    The change it makes to the mean over the points of ln Y_v1. Reported, never ruled.
    """
    raised = v1_level(aod_550 * SLAB_FACTOR, sun_elevation_deg, views, azimuths)
    return raised - v1_level(aod_550, sun_elevation_deg, views, azimuths)


def deeper_layer_change(
    aod_550: float, sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> float:
    """Rule 4's deeper aerosol layer: v0's single scattering at DEEPER_SCALE_HEIGHT_KM against v1's.

    The change it makes to the mean over the points of ln Y (v0 on both sides, scale height 2.0
    against 1.5 km). Reported, never ruled.
    """

    def level(scale_height_km: float) -> float:
        y = _over_points(
            lambda az: sky_xyz(
                Atmosphere(aod_550=aod_550, aerosol_scale_height_km=scale_height_km),
                sun_elevation_deg,
                views,
                az,
                engine="v0",
            )[:, 1],
            azimuths,
        )
        return float(np.mean(np.log(y)))

    return level(DEEPER_SCALE_HEIGHT_KM) - level(Atmosphere(0.0).aerosol_scale_height_km)


def predicted_if_monte_carlo(
    sun_elevation_deg: float, views: Sequence[float], azimuths: Sequence[float]
) -> np.ndarray:
    """e_i: what v1 / Prague at 0.1 would rise by, relative to 0.03, if the port's MS is 6.6% high.

    e_i = T_i / (T_i - M_i * 0.066 / 1.066) - 1, with T_i v1's total Y and M_i its multiple
    scattering Y at `aod_550` 0.1 and the given sun. Flat, azimuth-major, as the point table.
    """
    out = []
    for azimuth in azimuths:
        total = engine_xyz(GAP_AOD_550, sun_elevation_deg, views, azimuth, "v1")[:, 1]
        multiple = multiple_scattering_xyz(
            Atmosphere(aod_550=GAP_AOD_550), sun_elevation_deg, views, azimuth
        )[:, 1]
        excess_y = multiple * MONTE_CARLO_EXCESS / (1.0 + MONTE_CARLO_EXCESS)
        out.append(total / (total - excess_y) - 1.0)
    return np.concatenate(out)


# --- selecting points: by value, never by position ------------------------------------------------


def point_table(azimuths: Sequence[float], views: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """The (relative azimuth, view elevation) of every point, flat and azimuth-major."""
    azimuth = np.repeat(np.asarray(azimuths, dtype=float), len(views))
    view = np.tile(np.asarray(views, dtype=float), len(azimuths))
    return azimuth, view


def mask_points(
    azimuth: np.ndarray,
    view: np.ndarray,
    *,
    azimuths: Sequence[float] | None = None,
    views: Sequence[float] | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """The points whose azimuth is one of `azimuths` and whose view is one of `views`, by value.

    Either filter may be left out (every azimuth, every view). A value matches within `tol`.
    """
    mask = np.ones(azimuth.shape, dtype=bool)
    if azimuths is not None:
        mask &= np.any(np.abs(azimuth[:, None] - np.asarray(azimuths, dtype=float)) <= tol, axis=1)
    if views is not None:
        mask &= np.any(np.abs(view[:, None] - np.asarray(views, dtype=float)) <= tol, axis=1)
    return mask


def nominal_indices(views: Sequence[float], tol: float = 1e-9) -> list[int]:
    """Where WHTM1's six nominal views are among `views`, found by value, in the camera's order.

    Each must appear exactly once (to `tol`): a fixture made for another camera fit is refused.
    """
    indices = []
    for wanted in (float(x) for x in build_views()["nominal"].elevations_deg):
        hits = [i for i, value in enumerate(views) if abs(value - wanted) <= tol]
        if not hits:
            raise ValueError(f"the nominal view {wanted:.4f} is not in the fixture's views")
        if len(hits) > 1:
            raise ValueError(f"the nominal view {wanted:.4f} appears more than once")
        indices.append(hits[0])
    return indices


# --- the excess curve: r and its signed mean against the asymmetry --------------------------------


@dataclass(frozen=True)
class LobeCurve:
    """r and the signed mean of ln(Y_g / Y_Prague) over the nominal views at azimuth 0, per k."""

    sun_elevation_deg: float
    aod_550: float
    ks: tuple[int, ...]
    r: tuple[float, ...]
    signed_mean: tuple[float, ...]

    def r_at(self, k: int) -> float:
        return self.r[self.ks.index(k)]

    def signed_mean_at(self, k: int) -> float:
        return self.signed_mean[self.ks.index(k)]

    def r_by_k(self) -> dict[int, float]:
        return dict(zip(self.ks, self.r, strict=True))


def lobe_curve(
    aod_550: float,
    sun_elevation_deg: float,
    views: Sequence[float],
    prague_y: Sequence[float],
    ks: Sequence[int] = ASYMMETRY_GRID_K,
) -> LobeCurve:
    """The excess of XYZ_g over Prague's luminance at the given views (azimuth 0), for each k.

    `views` are the six nominal elevations and `prague_y` Prague's primary-reading luminance at
    them, in the same order. The default `ks` is the whole grid.
    """
    rs, means = [], []
    for k in ks:
        y = lobe_xyz(aod_550, asymmetry_of(k), sun_elevation_deg, views, TOWARD_SUN_AZIMUTH_DEG)[
            :, 1
        ]
        r, mean = excess(y, prague_y)
        rs.append(r)
        means.append(mean)
    return LobeCurve(sun_elevation_deg, aod_550, tuple(ks), tuple(rs), tuple(means))


def _fixture_xyz(entry: dict, azimuth: float, variant: str = "xyz") -> np.ndarray:
    block = entry["xyz"] if variant == "xyz" else entry["diagnostics"][variant]
    return np.array(block[f"{azimuth:g}"], dtype=float)


def _contrast(xyz: np.ndarray) -> np.ndarray:
    logs, _ = log_chromaticity(xyz[:NOMINAL_VIEWS])
    return (logs[:-1] - logs[-1])[0]


def _uv(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz
    denominator = x + 15.0 * y + 3.0 * z
    return np.array([4.0 * x / denominator, 9.0 * y / denominator])


def _ratio_of_ratios(data: dict, variant: str = "xyz") -> np.ndarray:
    """v1 over Prague in luminance at aod_550 0.1, divided by the same at 0.03: one per point."""
    views = data["view_elevations_deg"]
    sun = data["sun_elevation_deg"]
    out = []
    for azimuth in data["rel_azimuths_deg"]:
        ratios = []
        for aod in AEROSOL_OPTICAL_DEPTHS:
            entry = data["atmospheres"][f"{aod:g}"]
            v1 = engine_xyz(aod, sun, views, azimuth, "v1")[:, 1]
            ratios.append(v1 / _fixture_xyz(entry, azimuth, variant)[:, 1])
        out.append(ratios[1] / ratios[0])
    return np.concatenate(out)


def _predicted_if_monte_carlo(data: dict) -> np.ndarray:
    """What v1 / Prague at 0.1 would rise by, relative to 0.03, if the port's MS is 6.6% high."""
    return predicted_if_monte_carlo(
        data["sun_elevation_deg"], data["view_elevations_deg"], data["rel_azimuths_deg"]
    )


def report(data: dict) -> str:
    """The #34 tables (sun +10) and its aod 0.1 rule. Refuses any other sun: see `report_sun0`."""
    if data["sun_elevation_deg"] != SUN_ELEVATION_DEG:
        raise PragueError(
            f"this is #34's report for the sun +{SUN_ELEVATION_DEG:g} fixture, not for sun "
            f"{data['sun_elevation_deg']:g}: its aod 0.1 rule is not applied there (use --sun)"
        )
    views = data["view_elevations_deg"]
    sun = data["sun_elevation_deg"]
    azimuths = data["rel_azimuths_deg"]
    lines: list[str] = []
    out = lines.append
    for aod in AEROSOL_OPTICAL_DEPTHS:
        what = "the control" if aod == AEROSOL_OPTICAL_DEPTHS[0] else "a diagnostic, no tolerance"
        out(f"### Sun +10, `aod_550` {aod:g}, lowest band against the reference band ({what})")
        out("")
        out(
            "| rel. azimuth | Prague log(R/G), log(B/G) | v1 minus Prague | v0 minus Prague "
            "| v1 u'v' distance | v0 u'v' distance |"
        )
        out("| --- | --- | --- | --- | --- | --- |")
        entry = data["atmospheres"][f"{aod:g}"]
        for azimuth in azimuths:
            prague = _fixture_xyz(entry, azimuth)
            v1 = engine_xyz(aod, sun, views, azimuth, "v1")
            v0 = engine_xyz(aod, sun, views, azimuth, "v0")
            p = _contrast(prague)
            d1, d0 = _contrast(v1) - p, _contrast(v0) - p
            u1 = np.linalg.norm(_uv(v1[NOMINAL_VIEWS - 1]) - _uv(prague[NOMINAL_VIEWS - 1]))
            u0 = np.linalg.norm(_uv(v0[NOMINAL_VIEWS - 1]) - _uv(prague[NOMINAL_VIEWS - 1]))
            out(
                f"| {azimuth:g} | {p[0]:+.3f}, {p[1]:+.3f} | {d1[0]:+.3f}, {d1[1]:+.3f} "
                f"| {d0[0]:+.3f}, {d0[1]:+.3f} | {u1:.4f} | {u0:.4f} |"
            )
        out("")
    out("Tolerances of the control: 0.10 per log ratio, 0.02 in u'v'.")
    out("")

    for aod in AEROSOL_OPTICAL_DEPTHS:
        entry = data["atmospheres"][f"{aod:g}"]
        out(
            f"### `aod_550` {aod:g} (visibility {entry['visibility_km']:.2f} km): "
            "totals against Prague, and v1's multiply scattered share"
        )
        out("")
        out(
            "| rel. azimuth | view elevation | Prague Y | v1 / Prague | v0 / Prague "
            "| MS share of v1 |"
        )
        out("| --- | --- | --- | --- | --- | --- |")
        for azimuth in azimuths:
            prague = _fixture_xyz(entry, azimuth)
            v1 = engine_xyz(aod, sun, views, azimuth, "v1")
            v0 = engine_xyz(aod, sun, views, azimuth, "v0")
            multiple = multiple_scattering_xyz(Atmosphere(aod_550=aod), sun, views, azimuth)
            for i, view in enumerate(views):
                out(
                    f"| {azimuth:g} | {view:.2f} | {prague[i, 1]:.5f} "
                    f"| {v1[i, 1] / prague[i, 1]:.3f} | {v0[i, 1] / prague[i, 1]:.3f} "
                    f"| {multiple[i, 1] / v1[i, 1]:.3f} |"
                )
        out("")

    out("### Which model does Prague side with at `aod_550` 0.1?")
    out("")
    out(
        "d = (v1 / Prague in luminance at 0.1) divided by (v1 / Prague at 0.03), minus 1, at each "
        "of the 24 views and azimuths. If Prague sides with the port, d is 0; if with the Monte "
        "Carlo, d is the expected column (6.6 percent of v1's multiply scattered share)."
    )
    out("")
    expected = _predicted_if_monte_carlo(data)
    away = mask_points(
        *point_table(azimuths, views), azimuths=RULE_A_AZIMUTHS_DEG
    )  # off the aureole
    readings = {
        "Koschmieder mapping (the rule's reading)": _ratio_of_ratios(data) - 1.0,
        "visibility read in 1 / V": _ratio_of_ratios(data, "xyz_inverse_visibility") - 1.0,
        "aerosol depth matched": _ratio_of_ratios(data, "xyz_matched_aerosol_depth") - 1.0,
    }
    out("| reading of Prague | points | mean d | sd of d | min to max | expected if Monte Carlo |")
    out("| --- | --- | --- | --- | --- | --- |")
    for name, d in readings.items():
        for label, mask in (("all 24", np.ones_like(away)), ("azimuth 90, 180 (16)", away)):
            out(
                f"| {name} | {label} | {d[mask].mean():+.4f} | {d[mask].std(ddof=1):.4f} "
                f"| {d[mask].min():+.4f} to {d[mask].max():+.4f} | {expected[mask].mean():+.4f} |"
            )
    d = readings["Koschmieder mapping (the rule's reading)"]
    spread = float(np.std(d, ddof=1))
    reading_shift = float(np.mean(np.abs(d - readings["visibility read in 1 / V"])))
    predicted = float(np.mean(expected))
    if predicted <= max(spread, reading_shift):
        verdict = "cannot tell"
    elif abs(float(np.mean(d))) < abs(float(np.mean(d)) - predicted):
        verdict = "sides with the port"
    else:
        verdict = "sides with the Monte Carlo"
    out("")
    out(
        f"Rule (written before the numbers were seen): the two are told apart only if the "
        f"predicted gap ({predicted:+.4f}) exceeds both the spread of d across the views and "
        f"azimuths ({spread:.4f}) and the shift from reading Prague's visibility in 1 / V "
        f"({reading_shift:.4f}); result: **{verdict}**. The other rows and the azimuth 90, 180 "
        f"subset are exploratory context, chosen after the first look."
    )
    out("")

    out("### The other readings of Prague (`aod_550` 0.03, v1 minus Prague, lowest band)")
    out("")
    out(
        "Each cell: log(R/G) and log(B/G) deviation, then the u'v' distance at the reference "
        "elevation."
    )
    out("")
    out("| reading | az 0 | az 90 | az 180 |")
    out("| --- | --- | --- | --- |")
    entry = data["atmospheres"]["0.03"]
    for variant in (
        "xyz",
        "xyz_binned",
        "xyz_inverse_visibility",
        "xyz_altitude_0",
        "xyz_matched_aerosol_depth",
    ):
        cells = []
        for azimuth in azimuths:
            prague = _fixture_xyz(entry, azimuth, variant)
            v1 = engine_xyz(0.03, sun, views, azimuth, "v1")
            delta = _contrast(v1) - _contrast(prague)
            dist = np.linalg.norm(_uv(v1[NOMINAL_VIEWS - 1]) - _uv(prague[NOMINAL_VIEWS - 1]))
            cells.append(f"{delta[0]:+.3f}, {delta[1]:+.3f}, {dist:.4f}")
        out(f"| {variant} | " + " | ".join(cells) + " |")
    out("")

    out("### Prague's own vertical aerosol optical depth, and the matched reading")
    out("")
    out(
        "| v1 `aod_550` | mapped visibility (km) | Prague's depth there | matched visibility (km) "
        "| Prague's depth there | clamped at 131.8 km |"
    )
    out("| --- | --- | --- | --- | --- | --- |")
    for aod in AEROSOL_OPTICAL_DEPTHS:
        entry = data["atmospheres"][f"{aod:g}"]
        depth = aerosol_depth_from_transmittance(
            entry["diagnostics"]["zenith_transmittance"], data["channel_centres_nm"]
        )
        matched = entry["diagnostics"]["matched_aerosol_depth"]
        out(
            f"| {aod:g} | {entry['visibility_km']:.2f} | {depth:.3f} "
            f"| {matched['visibility_km']:.2f} | {matched['prague_aerosol_depth']:.3f} "
            f"| {matched['clamped_at_largest_visibility']} |"
        )
    return "\n".join(lines) + "\n"


# --- the sun-0 rules, run on the fixtures (#45) ---------------------------------------------------

CONTRAST_TOLERANCE = 0.10
"""#34's tolerance on the lowest band's log ratio against the reference band (reported only)."""
CHROMATICITY_TOLERANCE = 0.02
"""#34's tolerance on the reference band's u'v' distance (reported only)."""
GAP_AZIMUTH_DEG = 30.0
GAP_VIEW_DEG = 1.51
"""Where the port's 6.6 percent excess over the Monte Carlo was measured (sun 0, aod 0.1)."""
GAP_VIEW_TOL_DEG = 0.005
"""A fixture view is the gap's when it rounds to 1.51 degrees (the lowest band is 1.5066)."""


@dataclass(frozen=True)
class RuleAReport:
    """Rule A run on one fixture: the ruling over P, and every point for the reported items."""

    result: RuleA
    azimuth: np.ndarray  # relative azimuth of every point of the fixture, flat, azimuth-major
    view: np.ndarray
    in_p: np.ndarray  # the rule set P, selected by value
    d: np.ndarray  # d_i under the primary reading, every point
    e: np.ndarray  # e_i, every point
    d_readings: dict[str, np.ndarray]  # d_i under each other reading, every point
    D_reported: dict[str, float]  # D under the readings that are not part of m


def evaluate_rule_a(data: dict) -> RuleAReport:
    """Rule A on a fixture: v1 at the fixture's sun and views, Prague's readings from the file."""
    views = data["view_elevations_deg"]
    azimuths = data["rel_azimuths_deg"]
    azimuth, view = point_table(azimuths, views)
    in_p = mask_points(azimuth, view, azimuths=RULE_A_AZIMUTHS_DEG)
    if int(in_p.sum()) != len(RULE_A_AZIMUTHS_DEG) * len(views):
        raise PragueError("the fixture lacks rule A's relative azimuths 90 and 180")
    d = _ratio_of_ratios(data) - 1.0
    e = predicted_if_monte_carlo(data["sun_elevation_deg"], views, azimuths)
    names = (*RULE_A_ALTERNATIVE_READINGS, *RULE_A_REPORTED_READINGS)
    readings = {name: _ratio_of_ratios(data, name) - 1.0 for name in names}
    result = rule_a(
        d[in_p], e[in_p], {name: readings[name][in_p] for name in RULE_A_ALTERNATIVE_READINGS}
    )
    reported = {name: float(np.mean(readings[name][in_p])) for name in RULE_A_REPORTED_READINGS}
    return RuleAReport(result, azimuth, view, in_p, d, e, readings, reported)


def _toward_sun(data: dict, aod: float) -> tuple[list[float], np.ndarray]:
    """The six nominal views at relative azimuth 0 (by value) and Prague's primary Y at them."""
    views = data["view_elevations_deg"]
    index = nominal_indices(views)
    prague = _fixture_xyz(data["atmospheres"][f"{aod:g}"], TOWARD_SUN_AZIMUTH_DEG)
    return [views[i] for i in index], prague[index, 1]


def condition_curve(data: dict, aod: float) -> LobeCurve:
    """r and the signed mean over the whole grid for one condition (the fixture's sun, `aod`)."""
    views, prague_y = _toward_sun(data, aod)
    return lobe_curve(aod, data["sun_elevation_deg"], views, prague_y)


def condition_port_bound(data: dict, aod: float) -> float:
    """B for one condition: the port's most, rms over the six nominal views at azimuth 0."""
    views, _ = _toward_sun(data, aod)
    sun = data["sun_elevation_deg"]
    total = engine_xyz(aod, sun, views, TOWARD_SUN_AZIMUTH_DEG, "v1")[:, 1]
    multiple = multiple_scattering_xyz(Atmosphere(aod_550=aod), sun, views, TOWARD_SUN_AZIMUTH_DEG)[
        :, 1
    ]
    return port_bound_rms(total, multiple)


def side_and_back_r(data: dict, aod: float, k: int) -> float:
    """rms of ln(Y_g / Y_Prague) over every view at azimuths 90 and 180 together (not ruled)."""
    sun = data["sun_elevation_deg"]
    views = data["view_elevations_deg"]
    entry = data["atmospheres"][f"{aod:g}"]
    model, prague = [], []
    for azimuth in RULE_A_AZIMUTHS_DEG:
        model.append(lobe_xyz(aod, asymmetry_of(k), sun, views, azimuth)[:, 1])
        prague.append(_fixture_xyz(entry, azimuth)[:, 1])
    return excess(np.concatenate(model), np.concatenate(prague))[0]


@dataclass(frozen=True)
class ControlRow:
    """#34's control at one azimuth: v1 and XYZ_g against Prague (contrast deviations, u'v')."""

    azimuth: float
    v1_contrast: tuple[float, float]
    lobe_contrast: tuple[float, float]
    v1_distance: float
    lobe_distance: float


def control_with_lobe(pilot: dict, k: int) -> list[ControlRow]:
    """#34's control at the pilot fixture's `aod_550` 0.03 with XYZ_g in place of v1 (reported)."""
    aod = AEROSOL_OPTICAL_DEPTHS[0]
    views = pilot["view_elevations_deg"]
    sun = pilot["sun_elevation_deg"]
    index = nominal_indices(views)
    nominal = [views[i] for i in index]
    rows = []
    for azimuth in pilot["rel_azimuths_deg"]:
        prague = _fixture_xyz(pilot["atmospheres"][f"{aod:g}"], azimuth)[index]
        v1 = engine_xyz(aod, sun, nominal, azimuth, "v1")
        lobe = lobe_xyz(aod, asymmetry_of(k), sun, nominal, azimuth)
        p = _contrast(prague)
        d1, dg = _contrast(v1) - p, _contrast(lobe) - p
        reference = NOMINAL_VIEWS - 1
        u1 = float(np.linalg.norm(_uv(v1[reference]) - _uv(prague[reference])))
        ug = float(np.linalg.norm(_uv(lobe[reference]) - _uv(prague[reference])))
        rows.append(ControlRow(float(azimuth), (d1[0], d1[1]), (dg[0], dg[1]), u1, ug))
    return rows


@dataclass(frozen=True)
class RuleBReport:
    """Rule B run on the pilot fixture (sun +10, the fit) and the sun-0 fixture (the ruling)."""

    pilot: dict[float, LobeCurve]  # per load, at sun +10
    sun0: dict[float, LobeCurve]  # per load, at sun 0
    k_star: dict[float, int]  # g*(a) as k, fitted at sun +10
    k_best_sun0: dict[float, int]  # the grid value that minimises r at each sun-0 condition alone
    port_bound: dict[tuple[float, float], float]  # B per (sun, load), all four conditions
    result: RuleB  # over the two sun-0 conditions, in the order of AEROSOL_OPTICAL_DEPTHS


def evaluate_rule_b(pilot: dict, sun0: dict) -> RuleBReport:
    """Fit g*(a) at the pilot fixture (sun +10), then rule on the sun-0 fixture alone."""
    if pilot["sun_elevation_deg"] != SUN_ELEVATION_DEG:
        raise PragueError("rule B fits g*(a) at sun +10: the pilot fixture is not at that sun")
    if sun0["sun_elevation_deg"] != 0.0:
        raise PragueError("rule B rules at sun 0: the second fixture is not at that sun")
    loads = AEROSOL_OPTICAL_DEPTHS
    pilot_curves = {a: condition_curve(pilot, a) for a in loads}
    sun0_curves = {a: condition_curve(sun0, a) for a in loads}
    k_star = {a: best_asymmetry_k(pilot_curves[a].r_by_k()) for a in loads}
    k_best = {a: best_asymmetry_k(sun0_curves[a].r_by_k()) for a in loads}
    bounds = {
        (data["sun_elevation_deg"], a): condition_port_bound(data, a)
        for data in (pilot, sun0)
        for a in loads
    }
    result = rule_b(
        [sun0_curves[a].r_at(V1_ASYMMETRY_K) for a in loads],
        [sun0_curves[a].r_at(k_star[a]) for a in loads],
    )
    return RuleBReport(pilot_curves, sun0_curves, k_star, k_best, bounds, result)


RULE_B_OTHER_READINGS = (
    "xyz_inverse_visibility",
    "xyz_matched_aerosol_depth",
    "xyz_binned",
    "xyz_altitude_0",
)
"""The fixture's other readings of Prague, over which rule B is run and reported, never ruled.

Rule A has a term for the reading (m, over two of these); rule B, as registered, has none, and its
verdict is defined on the primary reading alone."""


@dataclass(frozen=True)
class RuleBReading:
    """Rule B run end to end under one reading of Prague: a report, not a ruling."""

    reading: str
    k_star: dict[float, int]  # g*(a), refitted at sun +10 under this reading
    r_base: dict[float, float]  # at sun 0, g = 0.7, under this reading
    r_fit: dict[float, float]  # at sun 0, at this reading's own g*(a)
    result: RuleB  # rule B's logic on those two, over the loads in order


def _toward_sun_reading_y(data: dict, aod: float, reading: str) -> np.ndarray:
    """Prague's luminance under `reading` at the six nominal views (relative azimuth 0)."""
    index = nominal_indices(data["view_elevations_deg"])
    entry = data["atmospheres"][f"{aod:g}"]
    return _fixture_xyz(entry, TOWARD_SUN_AZIMUTH_DEG, reading)[index, 1]


def _curve_against(
    aod: float, sun_elevation_deg: float, model_y: Sequence[np.ndarray], prague_y: np.ndarray
) -> LobeCurve:
    """`lobe_curve`, from the lobe diagnostic's luminance already computed for every k."""
    pairs = [excess(y, prague_y) for y in model_y]
    r = tuple(r for r, _ in pairs)
    return LobeCurve(sun_elevation_deg, aod, ASYMMETRY_GRID_K, r, tuple(m for _, m in pairs))


def evaluate_rule_b_readings(
    pilot: dict, sun0: dict, readings: Sequence[str] = ("xyz", *RULE_B_OTHER_READINGS)
) -> dict[str, RuleBReading]:
    """Rule B under each given reading of Prague (by default: the registered one, then the others).

    For each reading, g*(a) is refitted at the pilot fixture (sun +10) from Prague's luminance
    under that reading; r at g = 0.7 and at g*(a) is taken at sun 0 under the same reading; and
    `rule_b` is applied unchanged. Only Prague's reading varies. The lobe diagnostic does not
    depend on it, so it is computed once per condition and shared. Under "xyz" this is
    `evaluate_rule_b`.
    """
    if pilot["sun_elevation_deg"] != SUN_ELEVATION_DEG:
        raise PragueError("rule B fits g*(a) at sun +10: the pilot fixture is not at that sun")
    if sun0["sun_elevation_deg"] != 0.0:
        raise PragueError("rule B rules at sun 0: the second fixture is not at that sun")
    loads = AEROSOL_OPTICAL_DEPTHS
    views = {}
    model = {}
    for name, data in (("pilot", pilot), ("sun0", sun0)):
        views[name] = _toward_sun(data, loads[0])[0]
        for aod in loads:
            model[name, aod] = [
                lobe_xyz(
                    aod,
                    asymmetry_of(k),
                    data["sun_elevation_deg"],
                    views[name],
                    TOWARD_SUN_AZIMUTH_DEG,
                )[:, 1]
                for k in ASYMMETRY_GRID_K
            ]
    out = {}
    for reading in readings:
        pilot_curves = {
            a: _curve_against(
                a,
                pilot["sun_elevation_deg"],
                model["pilot", a],
                _toward_sun_reading_y(pilot, a, reading),
            )
            for a in loads
        }
        sun0_curves = {
            a: _curve_against(
                a,
                sun0["sun_elevation_deg"],
                model["sun0", a],
                _toward_sun_reading_y(sun0, a, reading),
            )
            for a in loads
        }
        fitted = {a: best_asymmetry_k(curve.r_by_k()) for a, curve in pilot_curves.items()}
        base = [sun0_curves[a].r_at(V1_ASYMMETRY_K) for a in loads]
        fit = [sun0_curves[a].r_at(fitted[a]) for a in loads]
        out[reading] = RuleBReading(
            reading,
            fitted,
            dict(zip(loads, base, strict=True)),
            dict(zip(loads, fit, strict=True)),
            rule_b(base, fit),
        )
    return out


def _stats_row(label: str, d: np.ndarray, e: np.ndarray) -> str:
    sd = f"{d.std(ddof=STD_DDOF):.4f}" if d.size > 1 else "n/a"
    return (
        f"| {label} | {d.size} | {d.mean():+.4f} | {sd} "
        f"| {d.min():+.4f} to {d.max():+.4f} | {e.mean():+.4f} |"
    )


def _gap_mask(a: RuleAReport) -> np.ndarray:
    """The gap's own point: azimuth 30 and the view that rounds to 1.51, by value."""
    mask = mask_points(a.azimuth, a.view, azimuths=(GAP_AZIMUTH_DEG,))
    mask &= mask_points(a.azimuth, a.view, views=(GAP_VIEW_DEG,), tol=GAP_VIEW_TOL_DEG)
    if int(mask.sum()) != 1:
        raise PragueError(
            f"expected one point at azimuth {GAP_AZIMUTH_DEG:g}, view {GAP_VIEW_DEG:g}, "
            f"found {int(mask.sum())}"
        )
    return mask


def _exploratory_sun0(
    sun0: dict, a: RuleAReport, b: RuleBReport, by_reading: dict[str, RuleBReading]
) -> list[str]:
    """What the numbers say beyond the two rules: chosen after the first look, so not evidence.

    Every figure the build log quotes beyond the rulings is printed here, so that it can be
    reproduced from the fixtures.
    """
    views = sun0["view_elevations_deg"]
    loads = AEROSOL_OPTICAL_DEPTHS
    lines: list[str] = []
    out = lines.append
    out("")
    out(
        "### Exploratory: chosen after the first look at the numbers above; not part of either rule"
    )
    out("")
    out("v1 over Prague in luminance at sun 0, all eight views at each azimuth:")
    out("")
    out(
        "| `aod_550` | rel. azimuth | v1 / Prague, min to max | mean | MS share of v1, min to max |"
    )
    out("| --- | --- | --- | --- | --- |")
    for aod in loads:
        entry = sun0["atmospheres"][f"{aod:g}"]
        for azimuth in sun0["rel_azimuths_deg"]:
            prague = _fixture_xyz(entry, azimuth)[:, 1]
            v1 = engine_xyz(aod, 0.0, views, azimuth, "v1")[:, 1]
            ms = multiple_scattering_xyz(Atmosphere(aod_550=aod), 0.0, views, azimuth)[:, 1]
            ratio = v1 / prague
            out(
                f"| {aod:g} | {azimuth:g} | {ratio.min():.3f} to {ratio.max():.3f} "
                f"| {ratio.mean():.3f} | {(ms / v1).min():.2f} to {(ms / v1).max():.2f} |"
            )
    out("")
    out(
        "The excess at relative azimuth 30 (six nominal views; the rule rules on azimuth 0 only), "
        "with v1's asymmetry and the fitted one:"
    )
    out("")
    out("| `aod_550` | r at 0.7 | r at g*(a) | signed mean at 0.7 | at g*(a) |")
    out("| --- | --- | --- | --- | --- |")
    index = nominal_indices(views)
    nominal = [views[i] for i in index]
    for aod in loads:
        prague = _fixture_xyz(sun0["atmospheres"][f"{aod:g}"], GAP_AZIMUTH_DEG)[index, 1]
        cells = []
        for k in (V1_ASYMMETRY_K, b.k_star[aod]):
            y = lobe_xyz(aod, asymmetry_of(k), 0.0, nominal, GAP_AZIMUTH_DEG)[:, 1]
            cells.append(excess(y, prague))
        (r70, m70), (rk, mk) = cells
        out(f"| {aod:g} | {r70:.4f} | {rk:.4f} | {m70:+.4f} | {mk:+.4f} |")
    out("")
    out("ln(Y_g / Y_Prague) at each of the six nominal views at relative azimuth 0, lowest first:")
    out("")
    out("| condition | g | per view |")
    out("| --- | --- | --- |")
    for aod in loads:
        prague = _fixture_xyz(sun0["atmospheres"][f"{aod:g}"], TOWARD_SUN_AZIMUTH_DEG)[index, 1]
        picks = [(V1_ASYMMETRY_K, "0.7"), (b.k_star[aod], f"g*({aod:g})")]
        if b.k_best_sun0[aod] != b.k_star[aod]:
            picks.append((b.k_best_sun0[aod], "the sun-0 minimiser"))
        for k, label in picks:
            y = lobe_xyz(aod, asymmetry_of(k), 0.0, nominal, TOWARD_SUN_AZIMUTH_DEG)[:, 1]
            logs = " ".join(f"{x:+.3f}" for x in np.log(y / prague))
            out(f"| (sun 0, {aod:g}) | {asymmetry_of(k):.2f} ({label}) | {logs} |")
    out("")
    out(
        "Rule B under the other readings of Prague (not the registered reading; reported, not "
        "ruled). The registered verdict above is defined on the primary reading, and rule B, "
        "unlike rule A, has no term for the reading. Each row runs the same rule end to end under "
        "one reading of the same model: g*(a) refitted at sun +10 under that reading, then r at "
        "0.7, r at g*(a), their ratio and the count at sun 0 under the same reading. The last "
        "column applies rule B's branches to those numbers; it is not a ruling."
    )
    out("")
    out(
        "| reading of Prague | `aod_550` | g*(a), refitted at sun +10 | r at 0.7 | r at g*(a) "
        "| r(g*) / r(0.7) | counts (r at 0.7 >= 0.05) | rule B's logic would give (not a ruling) |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for reading, row in by_reading.items():
        name = "`xyz` (the registered reading)" if reading == "xyz" else f"`{reading}`"
        gives = f"{row.result.verdict} (branch {row.result.branch})"
        for i, aod in enumerate(loads):
            out(
                f"| {name} | {aod:g} | {asymmetry_of(row.k_star[aod]):.2f} | {row.r_base[aod]:.4f} "
                f"| {row.r_fit[aod]:.4f} | {row.r_fit[aod] / row.r_base[aod]:.3f} "
                f"| {'yes' if row.result.counted[i] else 'no'} | {gives} |"
            )
    out("")
    r = a.result
    bar = r.E / UNCERTAINTY_BAR_FACTOR
    verdict, branch = rule_a_verdict(r.D, r.E, math.nextafter(bar, 0.0))
    out(
        f"Rule A's counterfactual (not a ruling): branch 1 passes only if U < E / 2 = {bar:.4f}. "
        f"At D = {r.D:+.4f} the rule would fall through to branch {branch} ({verdict}) at the "
        f"largest such U: |D| = {abs(r.D):.4f} and |D - E| = {abs(r.D - r.E):.4f}."
    )
    return lines


def report_sun0(pilot: dict, sun0: dict) -> str:
    """Rules A and B at sun 0 from the two committed fixtures: every quantity the criterion lists.

    Prints no verdict but the two rules' (#34's aod 0.1 rule is not applied here). Everything
    after each ruling is reported, not ruled.
    """
    if sun0["sun_elevation_deg"] != 0.0:
        raise PragueError("report_sun0 needs the sun-0 fixture")
    a = evaluate_rule_a(sun0)
    b = evaluate_rule_b(pilot, sun0)
    a_pilot = evaluate_rule_a(pilot)
    loads = AEROSOL_OPTICAL_DEPTHS
    lines: list[str] = []
    out = lines.append
    r = a.result

    # --- rule A -------------------------------------------------------------------------------
    out("### Rule A at sun 0: which side of the `aod_550` 0.1 gap is Prague on?")
    out("")
    out(
        "d = (v1 / Prague in luminance at 0.1) divided by (v1 / Prague at 0.03), minus 1. If "
        "Prague sides with the port, d is 0; with the Monte Carlo, d is e (6.6 percent of v1's "
        "multiply scattered share, at sun 0). P is relative azimuths 90 and 180, all eight views: "
        f"{int(a.in_p.sum())} points. Every value is unrounded in the comparison."
    )
    out("")
    out("| quantity | value |")
    out("| --- | --- |")
    out(f"| D, mean of d over P | {r.D:+.4f} |")
    out(f"| E, mean of e over P | {r.E:+.4f} |")
    out(f"| s, sd of d over P (ddof 1) | {r.s:.4f} |")
    for name in RULE_A_ALTERNATIVE_READINGS:
        out(f"| D under `{name}` (part of m) | {r.D_alternatives[name]:+.4f} |")
    out(f"| m, largest shift of D over those two readings | {r.m:.4f} |")
    out(f"| U, the larger of s and m | {r.U:.4f} |")
    out(f"| 2U, the bar E must exceed | {UNCERTAINTY_BAR_FACTOR * r.U:.4f} |")
    for name in RULE_A_REPORTED_READINGS:
        out(f"| D under `{name}` (reported, not part of m) | {a.D_reported[name]:+.4f} |")
    out("")
    out(f"Verdict: **{r.verdict}** (branch {r.branch}).")
    if r.verdict == CANNOT_TELL:
        out("")
        if r.branch == 1:
            out(
                f"Branch 1 fired: E ({r.E:.4f}) is not above 2U "
                f"({UNCERTAINTY_BAR_FACTOR * r.U:.4f}). This says the comparison lacks power, "
                "not anything about either model."
            )
        else:
            out(
                f"Branch 4 fired: E ({r.E:.4f}) is above 2U, but D ({r.D:+.4f}) is farther than U "
                f"({r.U:.4f}) from both 0 and E: neither prediction fits."
            )
        out(
            "What would tell: a reference with v1's own aerosol (the same vertical profile and "
            "phase function, the one thing Prague cannot offer), from libRadtran, run offline "
            "(GPL, never vendored or imported)."
        )
    out("")
    out("Reported, not ruled: d and e at every point of the fixture, by group.")
    out("")
    out("| points | n | mean d | sd of d | min to max of d | mean e |")
    out("| --- | --- | --- | --- | --- | --- |")
    everything = np.ones_like(a.in_p)
    out(_stats_row(f"all {a.d.size}", a.d[everything], a.e[everything]))
    for azimuth in sun0["rel_azimuths_deg"]:
        mask = mask_points(a.azimuth, a.view, azimuths=(azimuth,))
        out(_stats_row(f"azimuth {azimuth:g} alone", a.d[mask], a.e[mask]))
    gap = _gap_mask(a)
    out(
        f"| the gap's own point (azimuth {GAP_AZIMUTH_DEG:g}, view {GAP_VIEW_DEG:g}) | 1 "
        f"| {a.d[gap][0]:+.4f} | n/a | n/a | {a.e[gap][0]:+.4f} |"
    )
    out("")
    p = a_pilot.result
    out(
        "The sun +10 pilot over the same P (the numbers the criterion quotes; not a ruling): "
        f"D {p.D:+.4f}, E {p.E:+.4f}, s {p.s:.4f}, m {p.m:.4f}, U {p.U:.4f}."
    )
    out("")

    # --- rule B -------------------------------------------------------------------------------
    out("### Rule B: what is the toward-sun luminance excess made of?")
    out("")
    out(
        "r is the root mean square of ln(Y_g / Y_Prague) over the six nominal views at relative "
        "azimuth 0 (unsigned); the signed mean of the same logarithms is beside it. g*(a) is "
        "fitted per load at sun +10 on the grid k = 30 to 90 (g = k / 100) and ruled on at sun 0."
    )
    out("")
    out("#### The fit at sun +10 (the pilot)")
    out("")
    out("| `aod_550` | g*(a) | r at g = 0.7 | r at g*(a) | signed mean at 0.7 | at g*(a) |")
    out("| --- | --- | --- | --- | --- | --- |")
    for aod in loads:
        c = b.pilot[aod]
        k = b.k_star[aod]
        out(
            f"| {aod:g} | {asymmetry_of(k):.2f} | {c.r_at(V1_ASYMMETRY_K):.4f} | {c.r_at(k):.4f} "
            f"| {c.signed_mean_at(V1_ASYMMETRY_K):+.4f} | {c.signed_mean_at(k):+.4f} |"
        )
    out("")
    low, high = loads
    if b.k_star[low] != b.k_star[high]:
        out(
            f"g*({low:g}) = {asymmetry_of(b.k_star[low]):.2f} and g*({high:g}) = "
            f"{asymmetry_of(b.k_star[high]):.2f} differ: the asymmetry that fits depends on the "
            "aerosol load, which is itself evidence against a pure phase-function error "
            "(whatever the sun-0 verdict below)."
        )
    else:
        out(
            f"g*({low:g}) and g*({high:g}) are both {asymmetry_of(b.k_star[low]):.2f}: the fitted "
            "asymmetry does not depend on the load here."
        )
    out("")
    out("#### The ruled conditions: sun 0")
    out("")
    out(
        "| `aod_550` | r at 0.7 | r at g*(a) | r(g*) / r(0.7) | signed mean at 0.7 | at g*(a) "
        "| B | 2B | counts (r at 0.7 >= 0.05) |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, aod in enumerate(loads):
        c = b.sun0[aod]
        k = b.k_star[aod]
        bound = b.port_bound[(0.0, aod)]
        out(
            f"| {aod:g} | {c.r_at(V1_ASYMMETRY_K):.4f} | {c.r_at(k):.4f} "
            f"| {c.r_at(k) / c.r_at(V1_ASYMMETRY_K):.3f} "
            f"| {c.signed_mean_at(V1_ASYMMETRY_K):+.4f} | {c.signed_mean_at(k):+.4f} "
            f"| {bound:.4f} | {2 * bound:.4f} | {'yes' if b.result.counted[i] else 'no'} |"
        )
    out("")
    out(f"Verdict: **{b.result.verdict}** (branch {b.result.branch}).")
    if b.result.verdict == AEROSOL_LOBE:
        out("")
        out(
            "Renderer v1 is unchanged by this. A physics change to its aerosol lobe would be a new "
            "renderer with its own registration, which the conductor files."
        )
    out("")
    out("Reported, not ruled:")
    out("")
    out("- The sun +10 cross-load pilot (r at sun +10 with the other load's g*):")
    for fitted, applied in ((high, low), (low, high)):
        k = b.k_star[fitted]
        c = b.pilot[applied]
        out(
            f"  - g*({fitted:g}) = {asymmetry_of(k):.2f} applied at (sun +10, {applied:g}): r "
            f"{c.r_at(k):.4f} against {c.r_at(V1_ASYMMETRY_K):.4f} at 0.7 and "
            f"{c.r_at(b.k_star[applied]):.4f} at its own g*({applied:g})."
        )
    out("- The grid value that minimises r at each sun-0 condition alone:")
    for aod in loads:
        k = b.k_best_sun0[aod]
        c = b.sun0[aod]
        out(
            f"  - (sun 0, {aod:g}): g = {asymmetry_of(k):.2f}, r {c.r_at(k):.4f} "
            f"(r at 0.7 {c.r_at(V1_ASYMMETRY_K):.4f}, at g*({aod:g}) {c.r_at(b.k_star[aod]):.4f})."
        )
    out(
        "- The port bound at sun +10 (B, 2B): "
        + "; ".join(
            f"{aod:g}: {b.port_bound[(10.0, aod)]:.4f}, {2 * b.port_bound[(10.0, aod)]:.4f}"
            for aod in loads
        )
        + "."
    )
    out("")
    out("Whether fixing the lobe costs the side and back sky: rms of ln(Y_g / Y_Prague) over the")
    out("eight views at azimuths 90 and 180 together (16 points).")
    out("")
    out("| condition | r at g = 0.7 | r at g*(a) |")
    out("| --- | --- | --- |")
    for label, data in (("sun +10", pilot), ("sun 0", sun0)):
        for aod in loads:
            out(
                f"| ({label}, {aod:g}) | {side_and_back_r(data, aod, V1_ASYMMETRY_K):.4f} "
                f"| {side_and_back_r(data, aod, b.k_star[aod]):.4f} |"
            )
    out("")
    k = b.k_star[low]
    out(
        f"#34's control at sun +10, `aod_550` {low:g}, recomputed with XYZ at g*({low:g}) = "
        f"{asymmetry_of(k):.2f} in place of v1 (lowest band against the reference band; tolerances "
        f"{CONTRAST_TOLERANCE:g} per log ratio, {CHROMATICITY_TOLERANCE:g} in u'v'):"
    )
    out("")
    out(
        "| rel. azimuth | v1 minus Prague | XYZ_g* minus Prague | v1 u'v' distance "
        "| XYZ_g* u'v' distance | XYZ_g* within both |"
    )
    out("| --- | --- | --- | --- | --- | --- |")
    for row in control_with_lobe(pilot, k):
        within = (
            max(abs(row.lobe_contrast[0]), abs(row.lobe_contrast[1])) <= CONTRAST_TOLERANCE
            and row.lobe_distance <= CHROMATICITY_TOLERANCE
        )
        out(
            f"| {row.azimuth:g} | {row.v1_contrast[0]:+.3f}, {row.v1_contrast[1]:+.3f} "
            f"| {row.lobe_contrast[0]:+.3f}, {row.lobe_contrast[1]:+.3f} | {row.v1_distance:.4f} "
            f"| {row.lobe_distance:.4f} | {'yes' if within else 'no'} |"
        )
    lines.extend(_exploratory_sun0(sun0, a, b, evaluate_rule_b_readings(pilot, sun0)))
    return "\n".join(lines) + "\n"


# --- the offset rules, run on the fixtures (#49) --------------------------------------------------
#
# `evaluate_offset` takes the four fixtures keyed by their sun elevation (+10, 0, -1, +1) and runs
# the registered rules on them; `report_offset` prints every quantity the criterion lists. Rules 2
# and 4 are registered as "no test": they have evidence printed, and no verdict.

OFFSET_LOADS = AEROSOL_OPTICAL_DEPTHS
OFFSET_FIXTURE_SUNS_DEG = (10.0, 0.0, -1.0, 1.0)
"""The four fixtures in the order the report lists them: the pilot suns, then the ruled ones."""
OFFSET_PILOT_SUNS_DEG = (10.0, 0.0)
OFFSET_PRIMARY_READING = "xyz"
OFFSET_ALTITUDE_READING = "xyz_altitude_0"
OFFSET_INVERSE_READING = "xyz_inverse_visibility"
OFFSET_BINNED_READING = "xyz_binned"
OFFSET_MATCHED_READING = "xyz_matched_aerosol_depth"
OFFSET_READINGS = (
    OFFSET_PRIMARY_READING,
    OFFSET_ALTITUDE_READING,
    OFFSET_INVERSE_READING,
    OFFSET_BINNED_READING,
    OFFSET_MATCHED_READING,
)
OFFSET_POINT_TOLERANCE = 1e-9
"""An azimuth of the fixture is the rule set's when it is within this many degrees of it."""
OZONE_COLUMN_TOLERANCE_DU = 1e-9


def _sun_label(sun: float) -> str:
    return "0" if sun == 0 else f"{sun:+g}"


def _condition_label(sun: float, aod: float) -> str:
    return f"(sun {_sun_label(sun)}, {aod:g})"


def _refusal_label(sun: float, aod: float) -> str:
    return f"sun {_sun_label(sun)}, aod_550 {aod:g}"


def _ratio(numerator: float, denominator: float, spec: str) -> str:
    """`numerator / denominator` formatted by `spec`, or "n/a" where the denominator is 0."""
    return "n/a" if denominator == 0 else format(numerator / denominator, spec)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition: bool, where: str, what: str) -> None:
    if not condition:
        raise PragueError(f"the fixture {where} is malformed: {what}")


def _check_block(block: object, views: int, where: str) -> None:
    _require(isinstance(block, list) and len(block) == views, where, f"not a list of {views} rows")
    for row in block:  # type: ignore[union-attr]
        _require(
            isinstance(row, list) and len(row) == 3 and all(_is_number(x) for x in row),
            where,
            "a row is not three finite numbers",
        )
        _require(row[1] > 0.0, where, "a luminance is not above 0")


def check_offset_fixture(data: object, sun: float) -> None:
    """Refuse a fixture whose fields are not what the rules read: it is another tool's output.

    Types, finiteness, shapes and the sun are checked before any value is used. A bool is not a
    number; a luminance must be finite and above 0.
    """
    where = f"for sun {_sun_label(sun)}"
    _require(isinstance(data, dict), where, "not a JSON object")
    assert isinstance(data, dict)
    _require(data.get("format") == "lastlight-prague-control", where, "wrong format")
    recorded = data.get("sun_elevation_deg")
    _require(_is_number(recorded), where, "sun_elevation_deg is not a finite number")
    _require(
        abs(recorded - sun) <= OFFSET_POINT_TOLERANCE,  # type: ignore[operator]
        where,
        f"it records sun {recorded!r}, not {sun:g}",
    )
    views = data.get("view_elevations_deg")
    _require(
        isinstance(views, list) and bool(views) and all(_is_number(v) for v in views),
        where,
        "view_elevations_deg is not a list of finite numbers",
    )
    azimuths = data.get("rel_azimuths_deg")
    _require(
        isinstance(azimuths, list) and bool(azimuths) and all(_is_number(a) for a in azimuths),
        where,
        "rel_azimuths_deg is not a list of finite numbers",
    )
    keys = [f"{a:g}" for a in azimuths]  # type: ignore[union-attr]
    _require(len(set(keys)) == len(keys), where, "a relative azimuth is listed twice")
    for wanted in OFFSET_AZIMUTHS_DEG:
        _require(
            any(abs(a - wanted) <= OFFSET_POINT_TOLERANCE for a in azimuths),  # type: ignore[union-attr]
            where,
            f"it lacks relative azimuth {wanted:g}",
        )
    centres = data.get("channel_centres_nm")
    _require(
        isinstance(centres, list) and len(centres) >= 2 and all(_is_number(c) for c in centres),
        where,
        "channel_centres_nm is not a list of finite numbers",
    )
    atmospheres = data.get("atmospheres")
    _require(isinstance(atmospheres, dict), where, "atmospheres is not an object")
    for aod in OFFSET_LOADS:
        name = f"{where}, aod_550 {aod:g}"
        entry = atmospheres.get(f"{aod:g}")  # type: ignore[union-attr]
        _require(isinstance(entry, dict), name, "that atmosphere is missing")
        _require(entry.get("aod_550") == aod, name, "aod_550 does not match its key")
        diagnostics = entry.get("diagnostics")
        _require(isinstance(diagnostics, dict), name, "diagnostics is not an object")
        for reading in OFFSET_READINGS:
            block = (
                entry.get("xyz") if reading == OFFSET_PRIMARY_READING else diagnostics.get(reading)  # type: ignore[union-attr]
            )
            _require(isinstance(block, dict), f"{name}, reading {reading}", "not an object")
            for key in keys:
                _check_block(
                    block.get(key), len(views), f"{name}, reading {reading}, azimuth {key}"
                )  # type: ignore[arg-type]
        transmittance = diagnostics.get("zenith_transmittance")
        _require(
            isinstance(transmittance, list)
            and len(transmittance) == len(centres)  # type: ignore[arg-type]
            and all(_is_number(t) and t > 0.0 for t in transmittance),
            name,
            "zenith_transmittance is not a list of finite numbers above 0, one per channel",
        )
        matched = diagnostics.get("matched_aerosol_depth")
        _require(
            isinstance(matched, dict)
            and _is_number(matched.get("prague_aerosol_depth"))
            and isinstance(matched.get("clamped_at_largest_visibility"), bool),
            name,
            "matched_aerosol_depth lacks a finite depth or a clamp flag",
        )


@dataclass(frozen=True)
class OffsetPoint:
    """The offset's quantities at one condition over one set of points (all views of each azimuth).

    L is the offset under the primary reading; L_A under `xyz_altitude_0`, L_inv under
    `xyz_inverse_visibility`, L_binned under `xyz_binned`, L_matched under the matched depth.
    `delta_o` is the part that goes with Prague's ozone, `delta_o_v0` the same from v0's single
    scattering alone, `line` the ozone line check.
    """

    points: int
    L: float
    L_A: float
    L_inv: float
    L_binned: float
    L_matched: float
    delta_o: float
    delta_o_v0: float
    line: float

    @property
    def m(self) -> float:
        """The reading noise at this condition."""
        return reading_noise(self.L, self.L_inv)

    @property
    def delta_h(self) -> float:
        """Delta_H = L - L_A: the mean of ln(Y_Prague,alt0 / Y_Prague)."""
        return self.L - self.L_A


def _chosen_azimuths(data: dict, wanted: Sequence[float]) -> list[float]:
    """The fixture's own azimuth values that are in `wanted`, chosen by value, in fixture order."""
    chosen = [
        float(a)
        for a in data["rel_azimuths_deg"]
        if any(abs(a - w) <= OFFSET_POINT_TOLERANCE for w in wanted)
    ]
    if len(chosen) != len(wanted):
        raise PragueError(f"the fixture lacks some of the relative azimuths {list(wanted)}")
    return chosen


def offset_point(data: dict, aod: float, azimuths: Sequence[float]) -> OffsetPoint:
    """The offset's quantities over the given azimuths of a fixture, all views of each.

    v1, the ozone engine and v0 are evaluated at the fixture's sun and views; Prague's luminance
    comes from the fixture. The points are chosen by azimuth value, never by position.
    """
    sun = float(data["sun_elevation_deg"])
    views = [float(v) for v in data["view_elevations_deg"]]
    entry = data["atmospheres"][f"{aod:g}"]
    chosen = _chosen_azimuths(data, azimuths)

    def model(luminance) -> np.ndarray:
        return np.concatenate([luminance(az) for az in chosen])

    def prague(reading: str) -> np.ndarray:
        return np.concatenate([_fixture_xyz(entry, az, reading)[:, 1] for az in chosen])

    def single(ozone_du: float) -> np.ndarray:
        return model(
            lambda az: sky_xyz(
                Atmosphere(aod_550=aod, ozone_du=ozone_du), sun, views, az, engine="v0"
            )[:, 1]
        )

    v1 = model(lambda az: engine_xyz(aod, sun, views, az, "v1")[:, 1])
    ozone = model(lambda az: ozone_xyz(aod, US_STANDARD_OZONE_DU, sun, views, az)[:, 1])
    at_v1, at_prague, at_node = (
        single(V1_OZONE_DU),
        single(US_STANDARD_OZONE_DU),
        single(DOMAIN["ozone_du"][1]),
    )
    return OffsetPoint(
        points=v1.size,
        L=offset_of(v1, prague(OFFSET_PRIMARY_READING)),
        L_A=offset_of(v1, prague(OFFSET_ALTITUDE_READING)),
        L_inv=offset_of(v1, prague(OFFSET_INVERSE_READING)),
        L_binned=offset_of(v1, prague(OFFSET_BINNED_READING)),
        L_matched=offset_of(v1, prague(OFFSET_MATCHED_READING)),
        delta_o=offset_of(v1, ozone),
        delta_o_v0=offset_of(at_v1, at_prague),
        line=offset_of(at_v1, at_prague) / offset_of(at_v1, at_node),
    )


def depth_range(delta_m: float, gap: float, error: float = DEPTH_ERROR) -> tuple[float, float]:
    """Rule 4's reported range of Delta_M if the depth estimate is `error` off either way.

    Delta_M x (1 - error / g) to Delta_M x (1 + error / g), g being Prague's aerosol depth at the
    primary reading minus its depth at the match. Printed as computed, even where the lower end is
    negative; the match is linear in the depth, which is the criterion's scaling.
    """
    return delta_m * (1.0 - error / gap), delta_m * (1.0 + error / gap)


@dataclass(frozen=True)
class OffsetCondition:
    """Everything the report prints for one condition (sun, load) of one fixture."""

    sun: float
    aod: float
    p: OffsetPoint  # over the rule set P
    by_azimuth: dict[float, OffsetPoint]  # each fixture azimuth alone, in the fixture's order
    slab: float  # rule 2's stand-in: the change it makes to the mean over P of ln Y_v1
    deeper: float  # rule 4's deeper layer: the change it makes to the mean over P of ln Y
    depth: float  # Prague's aerosol depth at the primary reading
    depth_matched: float  # and at the match
    clamped: bool  # the match is clamped at Prague's clearest visibility

    @property
    def label(self) -> str:
        return _condition_label(self.sun, self.aod)

    @property
    def delta_m(self) -> float | None:
        """L - L_matched, or None where the match is clamped (no range then)."""
        return None if self.clamped else self.p.L - self.p.L_matched


def offset_condition(data: dict, aod: float) -> OffsetCondition:
    sun = float(data["sun_elevation_deg"])
    views = [float(v) for v in data["view_elevations_deg"]]
    entry = data["atmospheres"][f"{aod:g}"]
    p_azimuths = _chosen_azimuths(data, OFFSET_AZIMUTHS_DEG)
    matched = entry["diagnostics"]["matched_aerosol_depth"]
    return OffsetCondition(
        sun=sun,
        aod=aod,
        p=offset_point(data, aod, OFFSET_AZIMUTHS_DEG),
        by_azimuth={
            float(az): offset_point(data, aod, (float(az),)) for az in data["rel_azimuths_deg"]
        },
        slab=slab_change(aod, sun, views, p_azimuths),
        deeper=deeper_layer_change(aod, sun, views, p_azimuths),
        depth=aerosol_depth_from_transmittance(
            entry["diagnostics"]["zenith_transmittance"], data["channel_centres_nm"]
        ),
        depth_matched=float(matched["prague_aerosol_depth"]),
        clamped=bool(matched["clamped_at_largest_visibility"]),
    )


@dataclass(frozen=True)
class RuleCondition:
    """One ruled condition: the removed part, the offset and the bar, and the verdict they give."""

    label: str
    removed: float
    offset: float
    bar: float
    verdict: str

    @property
    def share(self) -> float:
        """The removed part as a share of the offset (not a number where the offset is 0)."""
        return self.removed / self.offset if self.offset else math.nan


def _rule_condition(label: str, removed: float, offset: float, bar: float) -> RuleCondition:
    return RuleCondition(label, removed, offset, bar, condition_verdict(removed, offset, bar))


@dataclass(frozen=True)
class Rule1:
    """Rule 1 (ozone): its four ruled conditions, its verdict and branch, and the pilot's beside."""

    conditions: dict[tuple[float, float], RuleCondition]  # keyed (sun, aod_550): -1 then +1
    verdict: str
    branch: int
    pilot: dict[tuple[float, float], RuleCondition]  # the pilot's four: sun +10 then sun 0
    pilot_verdict: str  # rule 1's rule-level verdict over the two sun-0 conditions (not a ruling)
    pilot_branch: int


@dataclass(frozen=True)
class FitCondition:
    """Rule 3 at one load: the three offsets, kappa under both readings, and the verdict."""

    aod: float
    L: tuple[float, float, float]  # primary reading, at sun -1, 0, +1
    L_A: tuple[float, float, float]  # altitude 0
    m: tuple[float, float, float]
    kappa: float
    kappa_a: float
    condition: RuleCondition  # removed = min(kappa, kappa_a), offset = L(0), bar = fit_bar(m)


@dataclass(frozen=True)
class Rule3:
    """Rule 3 (Prague's fit near the horizon sun): its two loads, its verdict and branch."""

    conditions: tuple[FitCondition, ...]
    verdict: str
    branch: int


@dataclass(frozen=True)
class Kappas:
    """The bends reported beside rule 3 at one load (never ruled)."""

    delta_o: float  # kappa for Delta_O
    L: float
    L_A: float
    left: float  # kappa for L - Delta_O
    left_a: float  # kappa_A for L_A - Delta_O


@dataclass(frozen=True)
class OffsetResult:
    """The four fixtures run under the registered rules."""

    conditions: dict[tuple[float, float], OffsetCondition]  # keyed (sun, aod_550)
    rule1: Rule1
    rule3: Rule3
    kappas: dict[float, Kappas]  # keyed by aod_550
    v1_bend: dict[float, float]  # v1's own sky: sun 0 above the line through -1 and +1
    ozone_columns: dict[tuple[float, float], float | None]  # pilot: Prague's column at the bar
    left_after_ozone: dict[tuple[float, float], float]  # rule 1's ruled conditions


def _ozone_column_at_bar(data: dict, aod: float, bar: float) -> float | None:
    """The Prague ozone column at which Delta_O falls to `bar`, if one lies in v1's 280 to 345.

    Delta_O grows with the column from 0 at v1's own. Below the returned column the condition
    would read "not this". None where the bar is not over 0 or is already above Delta_O(345).
    """
    from scipy.optimize import brentq

    sun = float(data["sun_elevation_deg"])
    views = [float(v) for v in data["view_elevations_deg"]]
    p_azimuths = _chosen_azimuths(data, OFFSET_AZIMUTHS_DEG)

    def delta(ozone_du: float) -> float:
        v1 = np.concatenate([engine_xyz(aod, sun, views, az, "v1")[:, 1] for az in p_azimuths])
        engine = np.concatenate(
            [ozone_xyz(aod, ozone_du, sun, views, az)[:, 1] for az in p_azimuths]
        )
        return offset_of(v1, engine)

    top = delta(US_STANDARD_OZONE_DU)
    if not 0.0 < bar < top:
        return None
    return float(
        brentq(
            lambda column: delta(column) - bar,
            V1_OZONE_DU,
            US_STANDARD_OZONE_DU,
            xtol=OZONE_COLUMN_TOLERANCE_DU,
        )
    )


def evaluate_offset(fixtures: dict[float, dict]) -> OffsetResult:
    """The four rules' machinery on the four fixtures: rule 1, rule 3 and what is reported.

    `fixtures` maps the sun elevation (+10, 0, -1, +1) to the fixture's contents. Every field is
    checked before it is read. If an offset a rule uses is not above 0 the rules are not run:
    `PragueError` names the condition and reading, and nothing is reported.
    """
    if set(fixtures) != set(OFFSET_FIXTURE_SUNS_DEG):
        raise PragueError(f"the offset rules need the fixtures at suns {OFFSET_FIXTURE_SUNS_DEG}")
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        check_offset_fixture(fixtures[sun], sun)
    conditions = {
        (sun, aod): offset_condition(fixtures[sun], aod)
        for sun in OFFSET_FIXTURE_SUNS_DEG
        for aod in OFFSET_LOADS
    }

    # No excess to attribute: refuse before any verdict. Rule 1 uses L at the four ruled
    # conditions; rule 3 uses L and L_A at sun -1, 0 and +1, under both loads.
    minus, plus = OFFSET_RULED_SUNS_DEG
    for aod in OFFSET_LOADS:
        for sun in (minus, plus):
            refuse_no_excess(
                conditions[sun, aod].p.L, _refusal_label(sun, aod), OFFSET_PRIMARY_READING
            )
        for sun in (minus, 0.0, plus):
            c = conditions[sun, aod]
            label = _refusal_label(sun, aod)
            refuse_no_excess(c.p.L, label, OFFSET_PRIMARY_READING)
            refuse_no_excess(c.p.L_A, label, OFFSET_ALTITUDE_READING)

    def ozone_condition(sun: float, aod: float) -> RuleCondition:
        p = conditions[sun, aod].p
        return _rule_condition(_condition_label(sun, aod), p.delta_o, p.L, OFFSET_BAR_FACTOR * p.m)

    ruled = {
        (sun, aod): ozone_condition(sun, aod)
        for sun in OFFSET_RULED_SUNS_DEG
        for aod in OFFSET_LOADS
    }
    verdict, branch = rule_verdict(c.verdict for c in ruled.values())
    pilot = {
        (sun, aod): ozone_condition(sun, aod)
        for sun in OFFSET_PILOT_SUNS_DEG
        for aod in OFFSET_LOADS
    }
    pilot_verdict, pilot_branch = rule_verdict(
        c.verdict for (sun, _), c in pilot.items() if sun == 0.0
    )
    rule1 = Rule1(ruled, verdict, branch, pilot, pilot_verdict, pilot_branch)

    fits = []
    for aod in OFFSET_LOADS:
        at = [conditions[sun, aod].p for sun in (minus, 0.0, plus)]
        kappa = bend(*(p.L for p in at))
        kappa_a = bend(*(p.L_A for p in at))
        m = (at[0].m, at[1].m, at[2].m)
        fits.append(
            FitCondition(
                aod=aod,
                L=(at[0].L, at[1].L, at[2].L),
                L_A=(at[0].L_A, at[1].L_A, at[2].L_A),
                m=m,
                kappa=kappa,
                kappa_a=kappa_a,
                condition=_rule_condition(
                    f"a = {aod:g}", min(kappa, kappa_a), at[1].L, fit_bar(*m)
                ),
            )
        )
    verdict3, branch3 = rule_verdict(f.condition.verdict for f in fits)
    rule3 = Rule3(tuple(fits), verdict3, branch3)

    kappas = {}
    v1_bend = {}
    for aod in OFFSET_LOADS:
        at = [conditions[sun, aod].p for sun in (minus, 0.0, plus)]
        kappas[aod] = Kappas(
            delta_o=bend(*(p.delta_o for p in at)),
            L=bend(*(p.L for p in at)),
            L_A=bend(*(p.L_A for p in at)),
            left=bend(*(p.L - p.delta_o for p in at)),
            left_a=bend(*(p.L_A - p.delta_o for p in at)),
        )
        levels = [
            v1_level(
                aod,
                sun,
                [float(v) for v in fixtures[sun]["view_elevations_deg"]],
                _chosen_azimuths(fixtures[sun], OFFSET_AZIMUTHS_DEG),
            )
            for sun in (minus, 0.0, plus)
        ]
        v1_bend[aod] = bend(*levels)

    columns = {
        key: _ozone_column_at_bar(fixtures[key[0]], key[1], c.bar) for key, c in pilot.items()
    }
    left = {
        key: c.offset - c.removed if c.verdict in (CARRIES, CONTRIBUTES) else c.offset
        for key, c in ruled.items()
    }
    return OffsetResult(conditions, rule1, rule3, kappas, v1_bend, columns, left)


def _fixture_file(sun: float) -> str:
    return FIXTURE_SPECS[sun].path.name


def _pilot_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    c = result.conditions
    pilot = result.rule1.pilot
    biggest_m = max(c[key].p.m for key in pilot)
    out(
        "### The pilot (not a test): the numbers the criterion printed before it landed, recomputed"
    )
    out("")
    out(
        "The criterion computed these from the two earlier fixtures (sun +10 and sun 0), v0 and "
        "v1 before any ruled number existed. The rules rule only on Prague at sun -1 and +1. "
        "Over the 16 points P (relative azimuths 90 and 180, all eight views): L is the mean of "
        "ln(Y_v1 / Y_Prague), m = |L_inv - L| the reading noise, 2m the bar, Delta_O the part "
        f"that goes when v1 is given Prague's ozone ({US_STANDARD_OZONE_DU:g} DU instead of "
        f"{V1_OZONE_DU:g}), Delta_H = L - L_A the part that goes with Prague's observer on its "
        "ground."
    )
    out("")
    out("| condition | L | m | 2m | Delta_O, ozone | Delta_H, altitude 0 |")
    out("| --- | --- | --- | --- | --- | --- |")
    for key in pilot:
        p = c[key].p
        out(
            f"| {c[key].label} | {p.L:.4f} | {p.m:.4f} | {OFFSET_BAR_FACTOR * p.m:.4f} "
            f"| {p.delta_o:.4f} | {p.delta_h:+.4f} |"
        )
    out("")
    out("Ozone at the pilot conditions, under the per-condition verdict (not a ruling):")
    out("")
    out(
        "| condition | share Delta_O / L | margin over the bar, Delta_O / 2m | verdict "
        '| a Prague column under (DU) turns it into "not this" | left after ozone, L - Delta_O |'
    )
    out("| --- | --- | --- | --- | --- | --- |")
    for key, rc in pilot.items():
        p = c[key].p
        column = result.ozone_columns[key]
        shown_column = "none in 280 to 345" if column is None else f"{column:.2f}"
        out(
            f"| {rc.label} | {_ratio(rc.removed, rc.offset, '.3f')} "
            f"| {_ratio(rc.removed, rc.bar, '.2f')} | {rc.verdict} "
            f"| {shown_column} | {p.L - p.delta_o:.4f} |"
        )
    out("")
    out("Rule 1's bars at the ruled conditions, from v0 and v1 alone (no Prague):")
    out("")
    out(
        "| condition | Delta_O | carries only if L is at most 2 Delta_O "
        '| "not this" only if m is at least Delta_O / 2 '
        f"| that m in multiples of the pilot's largest m ({biggest_m:.4f}) |"
    )
    out("| --- | --- | --- | --- | --- |")
    for rc in result.rule1.conditions.values():
        out(
            f"| {rc.label} | {rc.removed:.4f} | {2 * rc.removed:.4f} | {rc.removed / 2:.4f} "
            f"| {_ratio(rc.removed / 2, biggest_m, '.1f')} |"
        )
    out("")
    bends = "; ".join(f"{aod:g}: {result.v1_bend[aod]:+.4f}" for aod in OFFSET_LOADS)
    out(
        "v1's own sky across sun -1, 0 and +1: the mean over P of ln Y_v1 at sun 0, minus the "
        f"mean of its values at -1 and +1 ({bends})."
    )
    out("")
    return lines


def _rule1_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    rule = result.rule1
    out("### Rule 1: ozone (v1 280 DU, Prague about 345 DU)")
    out("")
    out(
        "Removed part Delta_O, offset L, bar 2m, at the four ruled conditions (sun -1 and +1, "
        "both loads). The share is Delta_O / L."
    )
    out("")
    out("| condition | Delta_O | L | bar 2m | share Delta_O / L | verdict |")
    out("| --- | --- | --- | --- | --- | --- |")
    for rc in rule.conditions.values():
        out(
            f"| {rc.label} | {rc.removed:.4f} | {rc.offset:.4f} | {rc.bar:.4f} "
            f"| {_ratio(rc.removed, rc.offset, '.3f')} | {rc.verdict} |"
        )
    out("")
    out(f"Verdict: **{rule.verdict}** (branch {rule.branch}).")
    out("")
    out(
        "Beside it, the pilot's two sun-0 conditions under the same rule-level verdict (not a "
        f"ruling): {rule.pilot_verdict} (branch {rule.pilot_branch})."
    )
    out("")
    return lines


def _rule2_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    out("### Rule 2: no test")
    out("")
    out(
        "The criterion registers no rule for the observer's altitude and the ground, and says "
        "why: the only reading that moves Prague's observer, `xyz_altitude_0`, also puts it under "
        "340 m more air and aerosol, so Delta_H = L - L_A measures that slab, not the ground. The "
        f"slab's stand-in is v1 with its `aod_550` raised by exp(0.34 / 1.5) = {SLAB_FACTOR:.4f}. "
        "At every condition of all four fixtures, over P:"
    )
    out("")
    out(
        "| condition | Delta_H | slab stand-in, change in the mean of ln Y_v1 "
        "| Delta_H minus stand-in |"
    )
    out("| --- | --- | --- | --- |")
    for c in result.conditions.values():
        out(f"| {c.label} | {c.p.delta_h:+.4f} | {c.slab:+.4f} | {c.p.delta_h - c.slab:+.4f} |")
    out("")
    out(
        "No verdict is printed. What would tell: Prague with its ground raised to the observer, "
        "which the model cannot do; or a reference with v1's own geometry (libRadtran, run "
        "offline; GPL, never vendored or imported)."
    )
    out("")
    return lines


def _rule3_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    rule = result.rule3
    out("### Rule 3: Prague's fit near the horizon sun")
    out("")
    out(
        "kappa(a) = L(0, a) - (L(-1, a) + L(+1, a)) / 2, and kappa_A(a) the same from L_A: how "
        "far sun 0's offset sits above the straight line through its two neighbours. Removed "
        "part min(kappa, kappa_A), offset L(0, a), bar max(2 m_F, 0.05) with m_F the largest of "
        'm(-1, a), m(0, a) and m(+1, a). A negative kappa reads "not this".'
    )
    out("")
    out("| `aod_550` | L(-1) | L(0) | L(+1) | L_A(-1) | L_A(0) | L_A(+1) | m(-1) | m(0) | m(+1) |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in rule.conditions:
        out(f"| {f.aod:g} | " + " | ".join(f"{x:.4f}" for x in (*f.L, *f.L_A, *f.m)) + " |")
    out("")
    out(
        "| `aod_550` | kappa | kappa_A | removed, min(kappa, kappa_A) | offset L(0) "
        "| bar max(2 m_F, 0.05) | share removed / L(0) | verdict |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in rule.conditions:
        rc = f.condition
        out(
            f"| {f.aod:g} | {f.kappa:+.4f} | {f.kappa_a:+.4f} | {rc.removed:+.4f} "
            f"| {rc.offset:.4f} | {rc.bar:.4f} | {_ratio(rc.removed, rc.offset, '+.3f')} "
            f"| {rc.verdict} |"
        )
    out("")
    out(f"Verdict: **{rule.verdict}** (branch {rule.branch}).")
    if rule.verdict in (CARRIES, CONTRIBUTES):
        out("")
        out(
            "This says one of the two models has a kink at its sun-0 node, most likely Prague's "
            "(v1's single scattering is exact in the sun), but kappa cannot tell which: v1's "
            "table has a node at 0 too."
        )
    out("")
    return lines


def _rule4_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    out("### Rule 4: no test")
    out("")
    out(
        "The criterion registers no rule for the aerosol's amount and profile, and says why: the "
        "depth estimate's error is nearly the whole gap, and the load response does not separate "
        "the aerosol from the air. At every condition of all four fixtures, over P:"
    )
    out("")
    out(
        "| condition | Delta_M = L - L_matched | Prague's depth, primary | Prague's depth at the "
        f"match | Delta_M if the depth estimate is {DEPTH_ERROR:g} off either way |"
    )
    out("| --- | --- | --- | --- | --- |")
    for c in result.conditions.values():
        if c.clamped:
            out(
                f"| {c.label} | clamped | {c.depth:.4f} | {c.depth_matched:.4f} "
                "| clamped (no range) |"
            )
            continue
        gap = c.depth - c.depth_matched
        delta_m = c.delta_m
        assert delta_m is not None
        if gap > 0.0:
            low, high = depth_range(delta_m, gap)
            span = f"{low:.4f} to {high:.4f}"
        else:
            span = "n/a (Prague's depth is not above the match)"
        out(f"| {c.label} | {delta_m:+.4f} | {c.depth:.4f} | {c.depth_matched:.4f} | {span} |")
    out("")
    out(
        f"The load step L(s, 0.1) - L(s, 0.03) at each sun, and the change a deeper aerosol layer "
        f"(scale height {Atmosphere(0.0).aerosol_scale_height_km:g} to "
        f"{DEEPER_SCALE_HEIGHT_KM:g} km, v0's single scattering) makes to the mean over P of ln Y "
        "at both loads, with its ratio between them:"
    )
    out("")
    low, high = OFFSET_LOADS
    out(
        f"| sun | load step L({high:g}) - L({low:g}) | deeper layer at {low:g} "
        f"| deeper layer at {high:g} | ratio {high:g} to {low:g} |"
    )
    out("| --- | --- | --- | --- | --- |")
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        a, b = result.conditions[sun, low], result.conditions[sun, high]
        ratio = f"{b.deeper / a.deeper:.2f}" if a.deeper != 0.0 else "n/a"
        out(
            f"| {_sun_label(sun)} | {b.p.L - a.p.L:+.4f} | {a.deeper:+.4f} | {b.deeper:+.4f} "
            f"| {ratio} |"
        )
    out("")
    out(
        "No verdict is printed. What would tell: the depth estimate's error, derived from "
        "Prague's own gas depths (fitted from its spectral transmittance), which may make the "
        "amount testable; and a reference with v1's own aerosol (libRadtran, run offline)."
    )
    out("")
    return lines


def _reported_section(result: OffsetResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    out("### Reported, never ruled")
    out("")
    out(
        "For every condition of the four fixtures, over P and at each relative azimuth the "
        "fixture has (the azimuths toward the sun are in the same table as those away from it). "
        "m is |L_inv - L| over the same points; L_binned is L under `xyz_binned`, a spectral "
        "reading, never part of m; Delta_O (v0) is Delta_O from v0's single scattering alone; the "
        "line check is Delta_O (v0) over the same quantity at 330 DU, the table's last node (65 / "
        "50 = 1.3 on a straight line in ozone)."
    )
    out("")
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        out(f"#### Sun {_sun_label(sun)} (`{_fixture_file(sun)}`)")
        out("")
        out(
            "| `aod_550` | points | L | L_A | m | Delta_O | Delta_H | Delta_O / L | Delta_H / L "
            "| Delta_O (v0) | line check | L_binned |"
        )
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for aod in OFFSET_LOADS:
            c = result.conditions[sun, aod]
            rows = [("P (azimuths 90, 180)", c.p)]
            rows += [(f"azimuth {az:g} alone", p) for az, p in c.by_azimuth.items()]
            for name, p in rows:
                out(
                    f"| {aod:g} | {name} ({p.points}) | {p.L:.4f} | {p.L_A:.4f} | {p.m:.4f} "
                    f"| {p.delta_o:.4f} | {p.delta_h:+.4f} | {_ratio(p.delta_o, p.L, '.3f')} "
                    f"| {_ratio(p.delta_h, p.L, '+.3f')} | {p.delta_o_v0:.4f} | {p.line:.4f} "
                    f"| {p.L_binned:.4f} |"
                )
        out("")
    out(
        "What is left of L after ozone at each ruled condition of rule 1 (L - Delta_O where that "
        "condition's verdict is carries or contributes, otherwise L):"
    )
    out("")
    out("| condition | left |")
    out("| --- | --- |")
    for key, rc in result.rule1.conditions.items():
        out(f"| {rc.label} | {result.left_after_ozone[key]:.4f} |")
    out("")
    out("The bends across sun -1, 0 and +1 (kappa; kappa_A under `xyz_altitude_0`):")
    out("")
    out(
        "| `aod_550` | kappa for Delta_O | kappa for L | kappa_A for L_A "
        "| kappa for L - Delta_O | kappa_A for L_A - Delta_O |"
    )
    out("| --- | --- | --- | --- | --- | --- |")
    for aod in OFFSET_LOADS:
        k = result.kappas[aod]
        out(
            f"| {aod:g} | {k.delta_o:+.4f} | {k.L:+.4f} | {k.L_A:+.4f} | {k.left:+.4f} "
            f"| {k.left_a:+.4f} |"
        )
    out("")
    return lines


def format_offset(result: OffsetResult) -> str:
    """Every quantity the criterion lists, in its order: the pilot, rules 1 to 4, the reported."""
    lines: list[str] = []
    for section in (
        _pilot_section,
        _rule1_section,
        _rule2_section,
        _rule3_section,
        _rule4_section,
        _reported_section,
    ):
        lines.extend(section(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def report_offset(fixtures: dict[float, dict]) -> str:
    """The offset rules on the four fixtures, as the report text; refuses before any output."""
    return format_offset(evaluate_offset(fixtures))


def read_offset_fixtures() -> dict[float, dict]:
    """The four committed fixtures, keyed by sun elevation; unreadable or malformed ones refused."""
    fixtures = {}
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        path = FIXTURE_SPECS[sun].path
        try:
            fixtures[sun] = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise PragueError(f"{_shown(path)} cannot be read as a fixture: {error}") from error
    return fixtures


def cmd_report_offset() -> None:
    """Print the offset report; the text is complete before anything is written to stdout."""
    sys.stdout.write(report_offset(read_offset_fixtures()))


# --- #55: Prague's aerosol amount from Prague's own gas depths (registered as "no test") ---------
#
# `docs/renderer/prague-aerosol-depth-criterion.md` registers the six-node route as "no test" and
# says what this tool prints instead: the pilot, eight gas estimates fitted to the two spectra the
# fixtures hold, the crossings of #49's amount part, the clearest node, and the two invariance
# checks of the criterion's problem 1. No six-node fit is run. The report reads only the four
# committed fixtures and v1, evaluates no Prague model and writes no file. Where this code and the
# criterion disagree, the code is the defect.

PAPER_ALTITUDE_M = 340.0
"""The observer's height in the fixtures, 0.34 km, in metres."""
PAPER_PRESSURE_HPA = STANDARD_PRESSURE_HPA * (1 - 0.0065 * PAPER_ALTITUDE_M / 288.15) ** 5.25588
"""The U.S. Standard pressure at 0.34 km, unrounded (973.0664 hPa as the criterion prints it)."""
GAS_CHANNEL_CENTRES_NM = tuple(340.0 + 40.0 * i for i in range(11))
"""The eleven channel centres every fixture holds, ascending: 340 to 740 nm by 40."""
GAS_CENTRE_TOLERANCE_NM = 1e-9
"""Two centres this close are the same centre."""
GAS_FIT_FIRST_NM = 380.0
"""The fit uses the channels centred from here up; SPCTRL2's ozone table starts at 380 nm."""
GAS_FIT_CENTRES_NM = tuple(c for c in GAS_CHANNEL_CENTRES_NM if c >= GAS_FIT_FIRST_NM)
"""The fitted channels' centres: 380, 420, ..., 740 nm."""
GAS_OZONE_START = US_STANDARD_OZONE_DU / 1e3
"""The paper's ozone column in atm cm (0.345): the fixed value and the start of a free fit."""
GAS_START_ALPHA = 1.3
GAS_START_GAMMA = 0.0
GAS_START_FLOOR = 0.001
"""A start aerosol depth under this is raised to it before its logarithm is taken."""
GAS_FIT_SETTINGS = {"method": "trf", "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15, "max_nfev": 20000}
"""`scipy.optimize.least_squares`, every other argument at its default."""
DEPTH_LOADS = AEROSOL_OPTICAL_DEPTHS
"""The two loads the fixtures hold, the clearest first: the pilot's two spectra, in this order."""
DEPTH_VISIBILITIES_KM = (123.8187372, 49.9864309567)
"""The loads' visibilities as the criterion registers them: V_0.03, then V_p."""
DEPTH_VISIBILITY_TOLERANCE_KM = 1e-6
DEPTH_SUNS_DEG = (-1.0, 0.0, 1.0, 10.0)
"""The suns the crossings are printed at, in the criterion's order."""
CLEAREST_VISIBILITY_KM = 131.8
INVARIANCE_FACTORS = (0.5, 2.0)
ALPHA_CONTROL_STEP = 0.5
INVARIANCE_COPIES = 3
SE_MULTIPLE = 2.0


@dataclass(frozen=True)
class GasEstimate:
    """One of the eight estimates: the aerosol's form, and which gas parameters are fitted."""

    number: int
    curved: bool
    ozone_free: bool
    rayleigh_free: bool

    @property
    def gas_parameters(self) -> int:
        return int(self.rayleigh_free) + int(self.ozone_free)

    @property
    def aerosol_width(self) -> int:
        """Parameters per visibility: u and alpha, and gamma where the form is curved."""
        return 3 if self.curved else 2

    @property
    def form(self) -> str:
        return "curved" if self.curved else "power law"

    @property
    def label(self) -> str:
        ozone = "free" if self.ozone_free else "paper"
        rayleigh = "free" if self.rayleigh_free else "paper"
        return f"estimate {self.number} ({self.form}, ozone {ozone}, Rayleigh {rayleigh})"


GAS_ESTIMATES = tuple(
    GasEstimate(number, curved, ozone_free, rayleigh_free)
    for number, (curved, ozone_free, rayleigh_free) in enumerate(
        itertools.product((False, True), (True, False), (True, False)), start=1
    )
)
"""The eight estimates; the aerosol form varies slowest, then the ozone, then the Rayleigh."""


@dataclass(frozen=True)
class GasData:
    """What every fit needs: the fitted channels' shapes and the spectra as -ln T."""

    centres: np.ndarray  # the ten fitted channel centres
    x: np.ndarray  # ln(lambda / 550 nm) at them
    rho: np.ndarray  # v1's Rayleigh depth over its depth at 550 nm
    kappa: np.ndarray  # v1's ozone coefficient per atm cm
    rho_bar: float  # rho and kappa interpolated to 550 nm
    kappa_bar: float
    r_paper: float  # the paper's Rayleigh depth at 550 nm
    y: np.ndarray  # -ln T over the fitted channels, one row per visibility
    y_bar: np.ndarray  # -ln T interpolated to 550 nm over all the channels, per visibility

    @property
    def g_paper(self) -> float:
        """The paper's gas depth at 550 nm: its Rayleigh and its U.S. Standard ozone."""
        return self.r_paper * self.rho_bar + GAS_OZONE_START * self.kappa_bar


def gas_data(centres: Sequence[float], transmittance: Sequence[Sequence[float]]) -> GasData:
    """The fit's inputs from the channel centres and one row of zenith transmittance per visibility.

    The fixtures' channel centres must be the registered eleven, 340 to 740 nm by 40 and ascending
    (each within 1e-9 nm). The fit takes the ten whose centres are 380, 420, ..., 740 nm, chosen by
    those exact centres (the 340 nm channel is left out). ȳ(V), the start's total, interpolates
    all the channels at 550 nm.
    """
    all_centres = np.asarray(centres, dtype=float)
    tau = -np.log(np.asarray(transmittance, dtype=float))
    if tau.ndim != 2 or tau.shape[1] != all_centres.size or tau.shape[0] == 0:
        raise PragueError("the spectra are not one row of transmittance per visibility and channel")
    registered = np.asarray(GAS_CHANNEL_CENTRES_NM)
    if all_centres.shape != registered.shape or not np.all(
        np.abs(all_centres - registered) <= GAS_CENTRE_TOLERANCE_NM
    ):
        raise PragueError(
            f"the fixtures' channel centres are {all_centres.tolist()} nm, not the registered "
            "eleven, 340 to 740 nm by 40 and ascending: the fit takes the channels centred at "
            f"{GAS_FIT_CENTRES_NM[0]:g}, {GAS_FIT_CENTRES_NM[1]:g}, ..., "
            f"{GAS_FIT_CENTRES_NM[-1]:g} nm and no others"
        )
    fit_centres = np.asarray(GAS_FIT_CENTRES_NM)
    fitted = (
        np.abs(all_centres[:, None] - fit_centres[None, :]).min(axis=1) <= GAS_CENTRE_TOLERANCE_NM
    )
    chosen = all_centres[fitted]
    rho = rayleigh_optical_depth(chosen) / rayleigh_optical_depth(REFERENCE_WAVELENGTH_NM)
    kappa = ozone_absorption_per_atm_cm(chosen)
    return GasData(
        centres=chosen,
        x=np.log(chosen / REFERENCE_WAVELENGTH_NM),
        rho=rho,
        kappa=kappa,
        rho_bar=float(np.interp(REFERENCE_WAVELENGTH_NM, chosen, rho)),
        kappa_bar=float(np.interp(REFERENCE_WAVELENGTH_NM, chosen, kappa)),
        r_paper=float(rayleigh_optical_depth(REFERENCE_WAVELENGTH_NM, PAPER_PRESSURE_HPA)),
        y=tau[:, fitted],
        y_bar=np.array([np.interp(REFERENCE_WAVELENGTH_NM, all_centres, row) for row in tau]),
    )


def _unpack(
    estimate: GasEstimate, data: GasData, theta: np.ndarray, visibilities: int
) -> tuple[float, float, np.ndarray]:
    """(a_R, a_O, the aerosol parameters one row per visibility) from the parameter vector.

    The vector is the fitted gas parameters (Rayleigh, then ozone) and then each visibility's
    u, alpha (and gamma): a gas parameter that is not fitted is the paper's.
    """
    theta = np.asarray(theta, dtype=float)
    width = estimate.aerosol_width
    if theta.size != estimate.gas_parameters + visibilities * width:
        raise ValueError(
            f"{estimate.label}: {theta.size} parameters for {visibilities} visibilities"
        )
    index = 0
    a_r, a_o = data.r_paper, GAS_OZONE_START
    if estimate.rayleigh_free:
        a_r, index = float(theta[index]), index + 1
    if estimate.ozone_free:
        a_o, index = float(theta[index]), index + 1
    return a_r, a_o, theta[index:].reshape(visibilities, width)


def _aerosol(blocks: np.ndarray, x: np.ndarray) -> np.ndarray:
    """exp(u - alpha x + gamma x^2) by channel, one row per visibility."""
    exponent = blocks[:, 0:1] - blocks[:, 1:2] * x[None, :]
    if blocks.shape[1] == 3:
        exponent = exponent + blocks[:, 2:3] * x[None, :] ** 2
    return np.exp(exponent)


def gas_residuals(
    estimate: GasEstimate, data: GasData, theta: np.ndarray, visibilities: int
) -> np.ndarray:
    """The model minus -ln T over the fitted channels, one visibility after another."""
    a_r, a_o, blocks = _unpack(estimate, data, theta, visibilities)
    model = a_r * data.rho[None, :] + a_o * data.kappa[None, :] + _aerosol(blocks, data.x)
    return (model - data.y[:visibilities]).ravel()


def gas_jacobian(
    estimate: GasEstimate, data: GasData, theta: np.ndarray, visibilities: int
) -> np.ndarray:
    """The analytic Jacobian of the residuals: the gas columns are shared, the aerosol's are not.

    A visibility's columns are A, -x A and x^2 A (A its aerosol depth by channel) in its own
    rows and zero in every other visibility's. `visibilities` may exceed the data's rows: the
    Jacobian does not read the data, which the six-node invariance check relies on.
    """
    _, _, blocks = _unpack(estimate, data, theta, visibilities)
    channels = data.x.size
    gas = estimate.gas_parameters
    aerosol = _aerosol(blocks, data.x)
    jac = np.zeros((visibilities * channels, gas + visibilities * estimate.aerosol_width))
    index = 0
    if estimate.rayleigh_free:
        jac[:, index] = np.tile(data.rho, visibilities)
        index += 1
    if estimate.ozone_free:
        jac[:, index] = np.tile(data.kappa, visibilities)
    for i in range(visibilities):
        rows = slice(i * channels, (i + 1) * channels)
        base = gas + i * estimate.aerosol_width
        jac[rows, base] = aerosol[i]
        jac[rows, base + 1] = -data.x * aerosol[i]
        if estimate.curved:
            jac[rows, base + 2] = data.x**2 * aerosol[i]
    return jac


def gas_start(estimate: GasEstimate, data: GasData) -> np.ndarray:
    """The criterion's start: the paper's Rayleigh and ozone, and at each visibility an aerosol
    depth of ȳ(V) less the paper's gas (at least 0.001), alpha 1.3 and gamma 0."""
    start: list[float] = []
    if estimate.rayleigh_free:
        start.append(data.r_paper)
    if estimate.ozone_free:
        start.append(GAS_OZONE_START)
    for y_bar in data.y_bar:
        start.append(math.log(max(float(y_bar) - data.g_paper, GAS_START_FLOOR)))
        start.append(GAS_START_ALPHA)
        if estimate.curved:
            start.append(GAS_START_GAMMA)
    return np.array(start)


def gas_weights(estimate: GasEstimate, data: GasData, parameters: int) -> np.ndarray:
    """q, the derivative of G with respect to the parameters: rho-bar for a_R, kappa-bar for a_O
    and 0 for every aerosol parameter."""
    q = np.zeros(parameters)
    index = 0
    if estimate.rayleigh_free:
        q[index] = data.rho_bar
        index += 1
    if estimate.ozone_free:
        q[index] = data.kappa_bar
    return q


def residual_variance(ssr: float, values: int, parameters: int) -> float:
    """s2: the sum of squared residuals over the values less the fitted parameters.

    Infinite where nothing is left over (no degrees of freedom), so the error is then infinite.
    """
    free = values - parameters
    return ssr / free if free > 0 else math.inf


def standard_error(jac: np.ndarray, q: np.ndarray, s2: float) -> float:
    """sqrt(q' C q) with C = s2 (J'J)^-1; infinite if `numpy.linalg.inv` raises or if q' C q is
    not a positive finite number."""
    try:
        inverse = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return math.inf
    with np.errstate(invalid="ignore", over="ignore"):
        variance = float(q @ (s2 * inverse) @ q)
    return math.sqrt(variance) if math.isfinite(variance) and variance > 0.0 else math.inf


@dataclass(frozen=True)
class GasFit:
    """One estimate fitted: the solution, its gas, its error and the conditioning at it."""

    estimate: GasEstimate
    theta: np.ndarray
    jac: np.ndarray  # the analytic Jacobian at the solution, unscaled
    ssr: float
    values: int
    parameters: int
    status: int
    a_r: float
    a_o: float
    gas: float  # G = a_R rho-bar + a_O kappa-bar
    se: float  # 0 where no gas parameter is fitted, infinite where it is not defined
    rms: float  # sqrt(SSR / values)
    condition: float  # the 2-norm condition number of `jac`

    @property
    def s2(self) -> float:
        return residual_variance(self.ssr, self.values, self.parameters)

    @property
    def ozone_du(self) -> float:
        return self.a_o * 1e3

    def excess_over(self, y_bar_p: float) -> float:
        """g = ȳ(V_p) - G - 0.1: Prague's aerosol depth at V_p minus v1's, under this estimate."""
        return y_bar_p - self.gas - DEPTH_LOADS[-1]


def fit_gas_estimate(estimate: GasEstimate, data: GasData) -> GasFit:
    """Fit one estimate to every visibility of `data`; a fit with status <= 0 is refused."""
    visibilities = data.y.shape[0]

    def residuals(theta: np.ndarray) -> np.ndarray:
        return gas_residuals(estimate, data, theta, visibilities)

    def jacobian(theta: np.ndarray) -> np.ndarray:
        return gas_jacobian(estimate, data, theta, visibilities)

    result = least_squares(residuals, gas_start(estimate, data), jac=jacobian, **GAS_FIT_SETTINGS)
    if result.status <= 0:
        raise PragueError(
            f"{estimate.label} did not converge (status {result.status}, "
            f"{getattr(result, 'message', 'no message')}): the report is not written"
        )
    theta = np.asarray(result.x, dtype=float)
    final = residuals(theta)
    ssr = float(final @ final)
    jac = jacobian(theta)
    a_r, a_o, _ = _unpack(estimate, data, theta, visibilities)
    s2 = residual_variance(ssr, final.size, theta.size)
    se = (
        standard_error(jac, gas_weights(estimate, data, theta.size), s2)
        if estimate.gas_parameters
        else 0.0
    )
    return GasFit(
        estimate=estimate,
        theta=theta,
        jac=jac,
        ssr=ssr,
        values=final.size,
        parameters=theta.size,
        status=int(result.status),
        a_r=a_r,
        a_o=a_o,
        gas=a_r * data.rho_bar + a_o * data.kappa_bar,
        se=se,
        rms=math.sqrt(ssr / final.size),
        condition=float(np.linalg.cond(jac)),
    )


@dataclass(frozen=True)
class Invariance:
    """Problem 1's checks, for the estimates that fit a gas parameter (keyed by their number).

    Each ratio is a standard error over the pilot's, at the pilot's s2: the aerosol columns
    scaled by each factor; the first visibility's alpha raised (the control, which must move);
    each spectrum's fitted aerosol taken three times (six nodes).
    """

    estimates: tuple[int, ...]
    scale_ratios: dict[int, dict[float, float]]
    alpha_ratios: dict[int, float]
    triple_ratios: dict[int, float]

    @property
    def scale_largest(self) -> float:
        return max(abs(r - 1.0) for ratios in self.scale_ratios.values() for r in ratios.values())

    @property
    def alpha_smallest(self) -> float:
        return min(abs(r - 1.0) for r in self.alpha_ratios.values())

    @property
    def triple_largest(self) -> float:
        return max(abs(r - 1 / math.sqrt(INVARIANCE_COPIES)) for r in self.triple_ratios.values())


def invariance_checks(fits: Sequence[GasFit], data: GasData) -> Invariance:
    """The invariance of the standard error under the nodes' amounts and under repeated nodes."""
    visibilities = data.y.shape[0]
    numbers: list[int] = []
    scale: dict[int, dict[float, float]] = {}
    alpha: dict[int, float] = {}
    triple: dict[int, float] = {}
    for fit in fits:
        estimate = fit.estimate
        if not estimate.gas_parameters:
            continue
        gas = estimate.gas_parameters
        q = gas_weights(estimate, data, fit.theta.size)
        numbers.append(estimate.number)
        ratios = {}
        for factor in INVARIANCE_FACTORS:
            scaled = fit.jac.copy()
            scaled[:, gas:] *= factor
            ratios[factor] = standard_error(scaled, q, fit.s2) / fit.se
        scale[estimate.number] = ratios
        raised = fit.theta.copy()
        raised[gas + 1] += ALPHA_CONTROL_STEP  # the first visibility's alpha
        moved = gas_jacobian(estimate, data, raised, visibilities)
        alpha[estimate.number] = standard_error(moved, q, fit.s2) / fit.se
        blocks = fit.theta[gas:].reshape(visibilities, estimate.aerosol_width)
        repeated = np.concatenate(
            [fit.theta[:gas], np.repeat(blocks, INVARIANCE_COPIES, axis=0).ravel()]
        )
        nodes = visibilities * INVARIANCE_COPIES
        wide = gas_jacobian(estimate, data, repeated, nodes)
        q_wide = gas_weights(estimate, data, repeated.size)
        triple[estimate.number] = standard_error(wide, q_wide, fit.s2) / fit.se
    return Invariance(tuple(numbers), scale, alpha, triple)


@dataclass(frozen=True)
class DepthSun:
    """#49's amount part at one sun, and where it crosses the bar (the 0.1 load's condition)."""

    sun: float
    L: float  # #49's offset at the condition
    m: float  # the reading noise
    delta_m: float  # the amount's part, L - L_matched
    g_old: float  # the old excess: Prague's depth at V_p less its depth at the match
    g_paper: float  # the paper's excess, estimate 4's, for the ratio at the paper's excess

    @property
    def bar(self) -> float:
        return OFFSET_BAR_FACTOR * self.m

    @property
    def g_star(self) -> float:
        """The excess at which the amount's part equals the bar, scaled linearly in g."""
        return self.g_old * self.bar / self.delta_m

    @property
    def carrying(self) -> float:
        """The excess the amount's part would need to reach half the offset."""
        return self.g_old * (self.L * CARRIES_SHARE) / self.delta_m

    @property
    def delta_m_over_bar(self) -> float:
        return self.delta_m / self.bar

    @property
    def at_paper(self) -> float:
        """The amount's part at the paper's excess, scaled linearly, in multiples of the bar."""
        return self.delta_m * self.g_paper / (self.g_old * self.bar)


@dataclass(frozen=True)
class DepthResult:
    """The pilot, and everything the report prints, from the four fixtures."""

    data: GasData
    visibilities_km: tuple[float, float]  # V_0.03 and V_p
    fits: tuple[GasFit, ...]
    gas_old: float  # v1's gas at 550 nm at its own pressure, what the old estimate subtracts
    gas_paper: float
    gas_interpolation: float  # of G_paper - G_old: v1's gas interpolated like G, less G_old
    gas_pressure: float  # and the rest, the pressure
    suns: tuple[DepthSun, ...]
    y_bar_clearest: float  # ȳ(131.8 km), from the recorded clamped match and the old gas
    invariance: Invariance

    @property
    def y_bar_p(self) -> float:
        return float(self.data.y_bar[-1])

    @property
    def g_max(self) -> float:
        """The largest excess Prague's clearest node leaves: ȳ(V_p) - ȳ(131.8)."""
        return self.y_bar_p - self.y_bar_clearest

    @property
    def threshold_003(self) -> float:
        """The gas a match at the 0.03 load would need."""
        return self.y_bar_clearest - DEPTH_LOADS[0]

    def excess(self, fit: GasFit) -> float:
        return fit.excess_over(self.y_bar_p)

    def low(self, fit: GasFit) -> float:
        return self.excess(fit) - SE_MULTIPLE * fit.se

    def high(self, fit: GasFit) -> float:
        return self.excess(fit) + SE_MULTIPLE * fit.se


def _depth_entry(data: dict, aod: float) -> dict:
    return data["atmospheres"][f"{aod:g}"]


def check_depth_fixtures_agree(fixtures: dict[float, dict]) -> None:
    """Refuse fixtures that do not hold identical spectra and the same clamped match.

    The four fixtures hold the same Prague transmittance (it does not depend on the sun), and
    the pilot fits one pair of spectra: a disagreement means another tool wrote them.
    """
    first = OFFSET_FIXTURE_SUNS_DEG[0]
    base = fixtures[first]
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        _require(
            _is_number(_depth_entry(fixtures[sun], DEPTH_LOADS[0]).get("visibility_km"))
            and _is_number(_depth_entry(fixtures[sun], DEPTH_LOADS[1]).get("visibility_km")),
            f"for sun {_sun_label(sun)}",
            "visibility_km is not a finite number",
        )
    for sun in OFFSET_FIXTURE_SUNS_DEG[1:]:
        data = fixtures[sun]
        if data["channel_centres_nm"] != base["channel_centres_nm"]:
            raise PragueError(
                f"the channel centres of the sun {_sun_label(sun)} fixture are not identical to "
                f"the sun {_sun_label(first)} fixture's: the four fixtures must hold identical "
                "spectra"
            )
        for aod in DEPTH_LOADS:
            ours, theirs = _depth_entry(data, aod), _depth_entry(base, aod)
            same = (
                ours["diagnostics"]["zenith_transmittance"]
                == theirs["diagnostics"]["zenith_transmittance"]
                and ours["visibility_km"] == theirs["visibility_km"]
            )
            if not same:
                raise PragueError(
                    f"the {aod:g} spectrum of the sun {_sun_label(sun)} fixture is not identical "
                    f"to the sun {_sun_label(first)} fixture's: the pilot fits one pair of "
                    "spectra, so the four fixtures must hold identical spectra"
                )
        clear = DEPTH_LOADS[0]
        if (
            _depth_entry(data, clear)["diagnostics"]["matched_aerosol_depth"]
            != _depth_entry(base, clear)["diagnostics"]["matched_aerosol_depth"]
        ):
            raise PragueError(
                f"the {clear:g} load's clamped match of the sun {_sun_label(sun)} fixture is not "
                f"the sun {_sun_label(first)} fixture's: the four fixtures must agree"
            )


def check_depth_visibilities(fixtures: dict[float, dict]) -> None:
    """Refuse loads whose visibility is not the one the criterion registers (to 1e-6 km).

    The fixtures have already been found to agree, so the first one speaks for all four.
    """
    base = fixtures[OFFSET_FIXTURE_SUNS_DEG[0]]
    for aod, registered in zip(DEPTH_LOADS, DEPTH_VISIBILITIES_KM, strict=True):
        recorded = _depth_entry(base, aod)["visibility_km"]
        if abs(recorded - registered) > DEPTH_VISIBILITY_TOLERANCE_KM:
            raise PragueError(
                f"the {aod:g} load's visibility in the fixtures is {recorded!r} km, not the "
                f"criterion's {registered!r} km: the pilot's two spectra are Prague's at those two "
                "visibilities"
            )


def clearest_match_depth(fixtures: dict[float, dict]) -> float:
    """The 0.03 load's recorded depth at Prague's largest visibility: refused unless it is clamped.

    A match that is not clamped sits at some other visibility, and its depth is no value of
    ȳ at 131.8 km.
    """
    matched = _depth_entry(fixtures[OFFSET_FIXTURE_SUNS_DEG[0]], DEPTH_LOADS[0])["diagnostics"][
        "matched_aerosol_depth"
    ]
    if not matched["clamped_at_largest_visibility"]:
        raise PragueError(
            f"the {DEPTH_LOADS[0]:g} load's match is not clamped at Prague's largest visibility, "
            f"{CLEAREST_VISIBILITY_KM:g} km, so its recorded depth is not the depth there"
        )
    return float(matched["prague_aerosol_depth"])


def depth_crossing(sun: float, condition: OffsetCondition, g_paper: float) -> DepthSun:
    """#49's amount part at one sun, refused where it is not a quantity that can be scaled."""
    where = f"sun {_sun_label(sun)}"
    delta_m = condition.delta_m
    if condition.clamped or delta_m is None:
        raise PragueError(
            f"{where}: the {condition.aod:g} load's match is clamped at Prague's clearest "
            "visibility, so there is no amount's part to scale"
        )
    if not (math.isfinite(delta_m) and delta_m > 0.0):
        raise PragueError(
            f"{where}: the amount's part Delta_M is {delta_m!r}, not above 0: there is no "
            "crossing to find"
        )
    m = condition.p.m
    if not (math.isfinite(m) and m > 0.0):
        raise PragueError(f"{where}: the reading noise m is {m!r}, not above 0: the bar 2m is 0")
    g_old = condition.depth - condition.depth_matched
    if not (math.isfinite(g_old) and g_old > 0.0):
        raise PragueError(
            f"{where}: Prague's depth is not above its depth at the match (g_old {g_old!r}): "
            "there is no old excess to scale"
        )
    return DepthSun(sun, condition.p.L, m, delta_m, g_old, g_paper)


def depth_pilot(fixtures: dict[float, dict]) -> DepthResult:
    """The eight estimates fitted to the fixtures' two spectra, and what is printed beside them.

    `fixtures` maps the sun elevation (+10, 0, -1, +1) to the fixture's contents, which are
    checked, and neither edited nor written. Nothing here evaluates Prague.
    """
    if set(fixtures) != set(OFFSET_FIXTURE_SUNS_DEG):
        raise PragueError(f"the depth report needs the fixtures at suns {OFFSET_FIXTURE_SUNS_DEG}")
    for sun in OFFSET_FIXTURE_SUNS_DEG:
        check_offset_fixture(fixtures[sun], sun)
    check_depth_fixtures_agree(fixtures)
    check_depth_visibilities(fixtures)
    clearest_depth = clearest_match_depth(fixtures)

    base = fixtures[OFFSET_FIXTURE_SUNS_DEG[0]]
    data = gas_data(
        base["channel_centres_nm"],
        [_depth_entry(base, aod)["diagnostics"]["zenith_transmittance"] for aod in DEPTH_LOADS],
    )
    fits = tuple(fit_gas_estimate(estimate, data) for estimate in GAS_ESTIMATES)
    paper = next(fit for fit in fits if fit.estimate.number == 4)  # the paper's gas, both fixed

    rayleigh_old, ozone_old = v1_gas_depth_at_550(Atmosphere(0.0).pressure_hpa)
    gas_old = rayleigh_old + ozone_old
    interpolated = rayleigh_old * data.rho_bar + GAS_OZONE_START * data.kappa_bar
    y_bar_p = float(data.y_bar[-1])
    g_paper = paper.excess_over(y_bar_p)
    suns = tuple(
        depth_crossing(sun, offset_condition(fixtures[sun], DEPTH_LOADS[-1]), g_paper)
        for sun in DEPTH_SUNS_DEG
    )
    return DepthResult(
        data=data,
        visibilities_km=tuple(  # type: ignore[arg-type]
            float(_depth_entry(base, aod)["visibility_km"]) for aod in DEPTH_LOADS
        ),
        fits=fits,
        gas_old=gas_old,
        gas_paper=paper.gas,
        gas_interpolation=interpolated - gas_old,
        gas_pressure=paper.gas - interpolated,
        suns=suns,
        y_bar_clearest=clearest_depth + gas_old,
        invariance=invariance_checks(fits, data),
    )


def format_two_se(se: float) -> str:
    """Twice the standard error as printed: 4 decimals, 0 where no gas parameter is fitted."""
    if se == 0.0:
        return "0"
    return f"{SE_MULTIPLE * se:.4f}"


def format_condition(value: float) -> str:
    """A condition number as the criterion prints it: three significant digits under 1000, and
    four in scientific form from there (882, 14.1, 1.470e4). A condition number that is not finite
    (a singular J) prints as inf or nan."""
    if not math.isfinite(value):
        return str(float(value))
    if value < 999.5:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _depth_pilot_section(result: DepthResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    low_km, p_km = result.visibilities_km
    out("### #55: Prague's aerosol amount from its own gas depths, no test (pre-registered)")
    out("")
    out(
        "The criterion (`docs/renderer/prague-aerosol-depth-criterion.md`) registers the six-node "
        "route as no test, with its evidence. This report prints the evidence that data in hand "
        "can give: eight gas estimates fitted to the two spectra the fixtures hold, the crossings "
        "of #49's amount part, the clearest node, and the invariance checks of the criterion's "
        "problem 1. No six-node fit is run, no verdict is printed, and nothing is written: it "
        "reads the four committed fixtures and v1, and runs no Prague evaluation."
    )
    out("")
    out("#### The pilot (not a test): eight gas estimates fitted to the two committed spectra")
    out("")
    out(
        f"Prague's zenith transmittance at V = {low_km:.4f} km (the {DEPTH_LOADS[0]:g} load's "
        f"visibility) and V_p = {p_km:.4f} km (the {DEPTH_LOADS[1]:g} load's), the same eleven "
        "numbers each in all four fixtures. y = -ln T over the ten channels centred 380 to 740 nm "
        "(340 nm left out), 20 values. Model: a_R rho_c + a_O kappa_c + exp(u_i - alpha_i x_c + "
        "gamma_i x_c^2), x_c = ln(lambda_c / 550 nm); a_R and a_O shared, u_i, alpha_i, gamma_i "
        "one set per visibility. Unweighted least squares, `scipy.optimize.least_squares` "
        "(method trf, analytic Jacobian, xtol = ftol = gtol = 1e-15, at most 20000 evaluations). "
        f"The paper's Rayleigh is R_paper = {result.data.r_paper:.5f} at {PAPER_PRESSURE_HPA:.4f} "
        f"hPa (the U.S. Standard pressure at {PAPER_ALTITUDE_M:g} m, unrounded); its ozone is "
        f"{GAS_OZONE_START:g} atm cm."
    )
    out("")
    out("| estimate | aerosol form | ozone | Rayleigh |")
    out("| --- | --- | --- | --- |")
    for fit in result.fits:
        e = fit.estimate
        out(
            f"| {e.number} | {e.form} | {'free' if e.ozone_free else 'paper'} "
            f"| {'free' if e.rayleigh_free else 'paper'} |"
        )
    out("")
    out(
        "G is the gas depth at 550 nm, a_R rho-bar + a_O kappa-bar (rho-bar "
        f"{result.data.rho_bar:.5f}, kappa-bar {result.data.kappa_bar:.6f} per atm cm); 2SE twice "
        "its standard error, sqrt(q' s2 (J'J)^-1 q) with s2 the sum of squared residuals over the "
        "values less the fitted parameters (0 where no gas parameter is fitted); g = ybar(V_p) - "
        "G - 0.1 is Prague's aerosol depth at V_p minus v1's under that estimate; the residual "
        "rms is over all 20 values; the condition number is that of the unscaled J at the "
        "solution."
    )
    out("")
    out(
        "| estimate | G | 2SE | g | g - 2SE | g + 2SE | residual rms | a_R | ozone (DU) "
        "| condition number |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for fit in result.fits:
        out(
            f"| {fit.estimate.number} | {fit.gas:.4f} | {format_two_se(fit.se)} "
            f"| {result.excess(fit):.4f} | {result.low(fit):.4f} | {result.high(fit):.4f} "
            f"| {fit.rms:.5f} | {fit.a_r:.4f} | {fit.ozone_du:.1f} "
            f"| {format_condition(fit.condition)} |"
        )
    out("")
    return lines


def _depth_printed_section(result: DepthResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    out("#### The pilot, the bullets marked printed: the old estimate, the crossings, the node")
    out("")
    out(
        "The old estimate subtracts v1's gas at 550 nm (its Rayleigh depth at its own "
        f"{Atmosphere(0.0).pressure_hpa:g} hPa and its ozone for {US_STANDARD_OZONE_DU:g} DU) "
        "from a total interpolated between the 540 and 580 nm channel centres. The same gas "
        "interpolated the same way is G_paper, estimate 4's gas. The difference splits into the "
        "interpolation (v1's gas interpolated like G, less G_old) and the pressure (G_paper "
        "less that):"
    )
    out("")
    out("| gas depth at 550 nm | value |")
    out("| --- | --- |")
    out(f"| G_paper | {result.gas_paper:.5f} |")
    out(f"| G_old | {result.gas_old:.5f} |")
    out(f"| G_paper - G_old | {result.gas_paper - result.gas_old:.5f} |")
    out(f"| of which the interpolation | {result.gas_interpolation:.5f} |")
    out(f"| of which the pressure | {result.gas_pressure:.5f} |")
    out("")
    paper = next(f for f in result.fits if f.estimate.number == 4)
    out(
        f"The paper's gas puts the excess at g_4 = {result.excess(paper):.4f}, not the old "
        f"g_old = {result.suns[0].g_old:.4f} (Prague's depth at V_p less its depth at the "
        "match, both under the old estimate)."
    )
    out("")
    out(
        "The crossings, at the 0.1 load of each fixture. L is #49's offset, m its reading noise, "
        "b = 2m the bar, Delta_M #49's amount part; scaled linearly in g from the match at the "
        "old excess, the amount's part equals the bar at g* = g_old b / Delta_M and reaches half "
        "the offset at the carrying excess g_old (L / 2) / Delta_M; at the paper's excess it is "
        "Delta_M g_4 / (g_old b) times the bar:"
    )
    out("")
    out(
        "| sun | L | m | bar b = 2m | Delta_M | g_old | g* | carrying excess | Delta_M / b "
        "| Delta_M g_4 / (g_old b) |"
    )
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in result.suns:
        out(
            f"| {_sun_label(row.sun)} | {row.L:.4f} | {row.m:.4f} | {row.bar:.4f} "
            f"| {row.delta_m:.4f} | {row.g_old:.5f} | {row.g_star:.4f} | {row.carrying:.4f} "
            f"| {row.delta_m_over_bar:.3f} | {row.at_paper:.3f} |"
        )
    out("")
    out(
        f"The clearest node. ybar({CLEAREST_VISIBILITY_KM:g}) is the {DEPTH_LOADS[0]:g} load's "
        "recorded clamped match depth plus G_old; g_max = ybar(V_p) - ybar("
        f"{CLEAREST_VISIBILITY_KM:g}) is the most excess Prague's clearest node leaves; a match "
        f"at the {DEPTH_LOADS[0]:g} load would need a gas of ybar({CLEAREST_VISIBILITY_KM:g}) - "
        f"{DEPTH_LOADS[0]:g}:"
    )
    out("")
    out("| the clearest node | value |")
    out("| --- | --- |")
    out(f"| ybar({CLEAREST_VISIBILITY_KM:g}) | {result.y_bar_clearest:.5f} |")
    out(f"| g_max | {result.g_max:.4f} |")
    out(f"| {DEPTH_LOADS[0]:g} load's G threshold | {result.threshold_003:.4f} |")
    out("")
    largest = max(fit.gas for fit in result.fits)
    out(f"The largest of the eight G is {largest:.4f}.")
    out("")
    return lines


def _depth_invariance_section(result: DepthResult) -> list[str]:
    lines: list[str] = []
    out = lines.append
    inv = result.invariance
    out("#### The pilot, invariance of the standard error under the amounts (problem 1)")
    out("")
    out(
        "For the six estimates that fit a gas parameter, at the pilot's s2 throughout. Each "
        "node's aerosol columns of J scaled by 0.5 and by 2, as |ratio - 1| of the standard "
        "error to the pilot's; the control that must move, J with the first visibility's alpha "
        f"raised by {ALPHA_CONTROL_STEP:g}, as the ratio; and the six-node ratio, each pilot "
        "spectrum's fitted aerosol taken three times (each copy with its own aerosol "
        "parameters, the gas columns shared):"
    )
    out("")
    out(
        "| estimate | aerosol x 0.5, abs(ratio - 1) | aerosol x 2, abs(ratio - 1) "
        "| control, alpha_1 + 0.5, ratio | six nodes, ratio |"
    )
    out("| --- | --- | --- | --- | --- |")
    for number in inv.estimates:
        half, double = (abs(inv.scale_ratios[number][f] - 1.0) for f in INVARIANCE_FACTORS)
        out(
            f"| {number} | {half:.1e} | {double:.1e} | {inv.alpha_ratios[number]:.4f} "
            f"| {inv.triple_ratios[number]:.4f} |"
        )
    out("")
    out(
        f"Largest |ratio - 1| over the six estimates and both factors: {inv.scale_largest:.1e} "
        "(the criterion: under 1e-9)"
    )
    out("")
    out(
        f"Smallest |ratio - 1| of the control over the six estimates: {inv.alpha_smallest:.4f} "
        "(the criterion: 0.0180)"
    )
    out("")
    out(
        f"Largest |ratio - 1/sqrt(3)| over the six estimates: {inv.triple_largest:.1e} "
        "(the criterion: under 1e-9)"
    )
    out("")
    out(
        "No verdict is printed. The criterion registers the route as no test: with each node's "
        "shape free the amounts add no information about the gas, which is what these ratios "
        "show. What would tell is in its section on #57."
    )
    out("")
    return lines


def format_depth(result: DepthResult) -> str:
    """The whole report: the pilot table, the printed bullets, the invariance checks."""
    lines: list[str] = []
    for section in (_depth_pilot_section, _depth_printed_section, _depth_invariance_section):
        lines.extend(section(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def report_depth(fixtures: dict[float, dict]) -> str:
    """The #55 evidence on the four fixtures, as the report text; refuses before any output."""
    return format_depth(depth_pilot(fixtures))


def cmd_report_depth() -> None:
    """Print the depth report; the text is complete before anything is written to stdout."""
    sys.stdout.write(report_depth(read_offset_fixtures()))


def cmd_report(spec: FixtureSpec = SUN10) -> None:
    if spec.sun_elevation_deg not in (SUN_ELEVATION_DEG, 0.0):
        raise PragueError(
            f"there is no report for the sun {spec.sun_elevation_deg:g} fixture alone: "
            "`report --offset` reads all four fixtures"
        )
    data = json.loads(spec.path.read_text())
    if spec.sun_elevation_deg == SUN_ELEVATION_DEG:
        sys.stdout.write(report(data))
    else:
        # Rules A and B at sun 0: the pilot fit is the sun +10 fixture's, the ruling this one's.
        sys.stdout.write(report_sun0(json.loads(SUN10.path.read_text()), data))


# --- command line ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fetch", "build", "fixture", "report"], nargs="+")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--check", action="store_true", help="fixture: compare, write nothing")
    parser.add_argument(
        "--sun",
        type=float,
        default=SUN_ELEVATION_DEG,
        help="fixture, report: the sun elevation in degrees of a registered fixture (10, 0, -1, 1)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="fixture: write or check this file instead"
    )
    parser.add_argument(
        "--offset",
        action="store_true",
        help="report: the offset rules of #49, from all four fixtures (--sun is not used)",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="report: the evidence for #55 (the aerosol amount, no test), from all four fixtures "
        "(--sun is not used)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offset and "report" not in args.command:
        parser.error("--offset is an option of `report`")
    if args.depth and "report" not in args.command:
        parser.error("--depth is an option of `report`")
    if args.depth and args.offset:
        parser.error("--depth and --offset are different reports: give one")
    status = 0
    for command in args.command:
        try:
            if command == "fetch":
                cmd_fetch(args.cache)
            elif command == "build":
                cmd_build(args.cache)
            elif command == "fixture":
                spec = fixture_spec(args.sun, args.output)
                status = max(status, cmd_fixture(args.cache, spec, check=args.check))
            elif args.depth:
                cmd_report_depth()
            elif args.offset:
                cmd_report_offset()
            else:
                cmd_report(fixture_spec(args.sun))
        except (PragueError, urllib.error.URLError, subprocess.CalledProcessError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
