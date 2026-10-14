"""Fetch, build and evaluate the Prague Sky Model for renderer v1's daylight control (#34).

Usage (fetch needs the network; build needs a C++17 compiler, `c++` on the path):

    uv run python tools/prague_reference.py fetch            # model source and datasets
    uv run python tools/prague_reference.py build            # compile the evaluation driver
    uv run python tools/prague_reference.py fixture          # write the sun10.json fixture
    uv run python tools/prague_reference.py fixture --check  # re-evaluate, compare, write nothing
    uv run python tools/prague_reference.py report           # the results tables, from the fixture

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity
from lastlight.inputs.http import USER_AGENT
from lastlight.render import Atmosphere
from lastlight.render.atmosphere import (
    RAYLEIGH_SCALE_HEIGHT_KM,
    REFERENCE_WAVELENGTH_NM,
    WAVELENGTHS_NM,
    ozone_absorption_per_atm_cm,
    rayleigh_optical_depth,
)
from lastlight.render.colorimetry import spectrum_to_xyz
from lastlight.render.curve import sky_xyz
from lastlight.render.v1 import multiple_scattering_xyz

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "prague" / "sun10.json"
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
    mode: str = "sky",
) -> Evaluation:
    """Run the driver: sun +10, ground albedo 0.2, the model's own interpolation everywhere.

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
        repr(SUN_ELEVATION_DEG),
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
    if not np.allclose(table[:, 3], SUN_ELEVATION_DEG, atol=1e-9):
        raise PragueError(f"the model's sun elevation is {table[:, 3]}")
    return Evaluation(table[:, 5:], start, width)


# --- the fixture ----------------------------------------------------------------------------------


def significant(value: float, digits: int = 9) -> float:
    return float(f"{value:.{digits}g}")


def view_elevations() -> list[float]:
    nominal = [float(x) for x in build_views()["nominal"].elevations_deg]
    return [*nominal, *DIAGNOSTIC_ELEVATIONS_DEG]


def _queries(views: Sequence[float]) -> list[tuple[float, float]]:
    return [(azimuth, view) for azimuth in REL_AZIMUTHS_DEG for view in views]


def _xyz_block(
    channels: np.ndarray, start: float, width: float, views: int, *, binned: bool = False
) -> dict[str, list[list[float]]]:
    """Rows ordered azimuth-major, as `_queries`, become {azimuth: [[X, Y, Z] per view]}."""
    block: dict[str, list[list[float]]] = {}
    for i, azimuth in enumerate(REL_AZIMUTHS_DEG):
        rows = channels[i * views : (i + 1) * views]
        block[f"{azimuth:g}"] = [
            [significant(v) for v in channels_to_xyz(row, start, width, binned=binned)]
            for row in rows
        ]
    return block


def _bracket(nodes: Sequence[float], value: float) -> tuple[float, float]:
    below = max(node for node in nodes if node <= value)
    above = min(node for node in nodes if node >= value)
    return below, above


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
    rayleigh = float(rayleigh_optical_depth(REFERENCE_WAVELENGTH_NM, Atmosphere(0.0).pressure_hpa))
    ozone = float(ozone_absorption_per_atm_cm(REFERENCE_WAVELENGTH_NM)) * US_STANDARD_OZONE_DU / 1e3
    return total - rayleigh - ozone


def matched_visibility_km(
    cache: Path, dataset: Dataset, aod: float, low: float, high: float, centres: Sequence[float]
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


def build_fixture(cache: Path) -> dict:
    """Evaluate Prague for both atmospheres and every variant: the whole fixture, as a dict."""
    fetch_model(cache)
    paths = {key: cache / "dl" / dataset.name for key, dataset in DATASETS.items()}
    for key, dataset in DATASETS.items():
        if not paths[key].exists() or paths[key].stat().st_size != dataset.size:
            raise PragueError(f"{paths[key]} is missing: run `prague_reference.py fetch`")
        if sha256_of(paths[key]) != dataset.sha256:
            raise PragueError(f"{paths[key]} does not have its pinned SHA-256")
    axes = read_axes(paths["full"])
    ground_axes = read_axes(paths["ground"])
    views = view_elevations()
    queries = _queries(views)
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
            cache, full, visibility=visibility, altitude_m=altitude_m, queries=queries
        )
        n = len(views)

        # Visibility interpolated in 1 / V (extinction) between the bracketing dataset nodes.
        below, above = _bracket(axes.visibilities_km, visibility)
        if below == above:
            blended = primary.channels
        else:
            weight = (1.0 / below - 1.0 / visibility) / (1.0 / below - 1.0 / above)
            lower = evaluate(cache, full, visibility=below, altitude_m=altitude_m, queries=queries)
            upper = evaluate(cache, full, visibility=above, altitude_m=altitude_m, queries=queries)
            blended = (1.0 - weight) * lower.channels + weight * upper.channels

        # Observer at altitude 0, from the full dataset and from the ground-level one.
        at_zero = evaluate(cache, full, visibility=visibility, altitude_m=0.0, queries=queries)
        at_zero_ground = evaluate(
            cache, ground, visibility=visibility, altitude_m=0.0, queries=queries
        )
        ground_check = ground_check and np.array_equal(at_zero.channels, at_zero_ground.channels)

        zenith = evaluate(
            cache,
            full,
            visibility=visibility,
            altitude_m=altitude_m,
            queries=[(0.0, 90.0)],
            mode="transmittance",
        )
        centres = [
            axes.channel_start_nm + (i + 0.5) * axes.channel_width_nm for i in range(axes.channels)
        ]
        matched_v, matched_depth, clamped = matched_visibility_km(
            cache, full, aod, low, high, centres
        )
        matched = evaluate(
            cache, full, visibility=matched_v, altitude_m=altitude_m, queries=queries
        )
        atmospheres[f"{aod:g}"] = {
            "aod_550": aod,
            "ground_extinction_per_km": significant(ground_extinction_per_km(atm), 12),
            "visibility_km": significant(visibility, 12),
            "bracketing_visibility_nodes_km": [below, above],
            "xyz": _xyz_block(primary.channels, primary.start_nm, primary.width_nm, n),
            "diagnostics": {
                "xyz_binned": _xyz_block(
                    primary.channels, primary.start_nm, primary.width_nm, n, binned=True
                ),
                "xyz_inverse_visibility": _xyz_block(
                    blended, primary.start_nm, primary.width_nm, n
                ),
                "xyz_altitude_0": _xyz_block(
                    at_zero.channels, at_zero.start_nm, at_zero.width_nm, n
                ),
                "zenith_transmittance": [significant(t) for t in zenith.channels[0]],
                "xyz_matched_aerosol_depth": _xyz_block(
                    matched.channels, matched.start_nm, matched.width_nm, n
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
        "sun_elevation_deg": SUN_ELEVATION_DEG,
        "observer_altitude_km": OBSERVER_ALTITUDE_KM,
        "ground_albedo": GROUND_ALBEDO,
        "view_elevations_deg": views,
        "nominal_view_count": NOMINAL_VIEWS,
        "rel_azimuths_deg": list(REL_AZIMUTHS_DEG),
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
            "command": "uv run python tools/prague_reference.py fixture",
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


def cmd_fixture(cache: Path, *, check: bool) -> int:
    fresh = build_fixture(cache)
    if not check:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(dump(fresh))
        print(f"wrote {FIXTURE.relative_to(REPO_ROOT)} ({FIXTURE.stat().st_size} bytes)")
        return 0
    committed = json.loads(FIXTURE.read_text())
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


def _engine_xyz(aod: float, views: Sequence[float], azimuth: float, engine: str) -> np.ndarray:
    return sky_xyz(Atmosphere(aod_550=aod), SUN_ELEVATION_DEG, views, azimuth, engine=engine)


def _ratio_of_ratios(data: dict, variant: str = "xyz") -> np.ndarray:
    """v1 over Prague in luminance at aod_550 0.1, divided by the same at 0.03: one per point."""
    views = data["view_elevations_deg"]
    out = []
    for azimuth in REL_AZIMUTHS_DEG:
        ratios = []
        for aod in AEROSOL_OPTICAL_DEPTHS:
            entry = data["atmospheres"][f"{aod:g}"]
            v1 = _engine_xyz(aod, views, azimuth, "v1")[:, 1]
            ratios.append(v1 / _fixture_xyz(entry, azimuth, variant)[:, 1])
        out.append(ratios[1] / ratios[0])
    return np.concatenate(out)


def _predicted_if_monte_carlo(data: dict) -> np.ndarray:
    """What v1 / Prague at 0.1 would rise by, relative to 0.03, if the port's MS is 6.6% high."""
    views = data["view_elevations_deg"]
    out = []
    for azimuth in REL_AZIMUTHS_DEG:
        total = _engine_xyz(0.1, views, azimuth, "v1")[:, 1]
        multiple = multiple_scattering_xyz(
            Atmosphere(aod_550=0.1), SUN_ELEVATION_DEG, views, azimuth
        )[:, 1]
        excess = multiple * MONTE_CARLO_EXCESS / (1.0 + MONTE_CARLO_EXCESS)
        out.append(total / (total - excess) - 1.0)
    return np.concatenate(out)


def report(data: dict) -> str:
    views = data["view_elevations_deg"]
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
        for azimuth in REL_AZIMUTHS_DEG:
            prague = _fixture_xyz(entry, azimuth)
            v1 = _engine_xyz(aod, views, azimuth, "v1")
            v0 = _engine_xyz(aod, views, azimuth, "v0")
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
        for azimuth in REL_AZIMUTHS_DEG:
            prague = _fixture_xyz(entry, azimuth)
            v1 = _engine_xyz(aod, views, azimuth, "v1")
            v0 = _engine_xyz(aod, views, azimuth, "v0")
            multiple = multiple_scattering_xyz(
                Atmosphere(aod_550=aod), SUN_ELEVATION_DEG, views, azimuth
            )
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
    away = np.repeat([False, True, True], len(views))  # azimuths 90 and 180: off the aureole
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
        for azimuth in REL_AZIMUTHS_DEG:
            prague = _fixture_xyz(entry, azimuth, variant)
            v1 = _engine_xyz(0.03, views, azimuth, "v1")
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


def cmd_report() -> None:
    sys.stdout.write(report(json.loads(FIXTURE.read_text())))


# --- command line ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["fetch", "build", "fixture", "report"], nargs="+")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--check", action="store_true", help="fixture: compare, write nothing")
    args = parser.parse_args(argv)
    status = 0
    for command in args.command:
        try:
            if command == "fetch":
                cmd_fetch(args.cache)
            elif command == "build":
                cmd_build(args.cache)
            elif command == "fixture":
                status = max(status, cmd_fixture(args.cache, check=args.check))
            else:
                cmd_report()
        except (PragueError, urllib.error.URLError, subprocess.CalledProcessError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
