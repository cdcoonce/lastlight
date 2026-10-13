"""The Prague Sky Model daylight control for renderer v1 (#34).

Renderer v1 has only been checked against its own Monte Carlo and the camera's first evenings.
The Prague Sky Model (Wilkie et al. 2021, Vevoda et al. 2022; Apache-2.0, Charles University) is
an independent daylight model fitted to brute-force path tracing, so v1 should agree with it in
the daytime, where multiple scattering is a smaller share than at twilight. The model and its
dataset are fetched and built outside the repository by `tools/prague_reference.py`; what this
file reads is the small fixture that tool writes, `tests/fixtures/prague/sun10.json`, so it runs
anywhere with no network and no `~/.cache/lastlight/prague/`.

The control, fixed by the issue before any comparison and never changed afterwards:

* at sun +10 degrees, in the low-aerosol atmosphere (`aod_550` 0.03), at each of the relative
  azimuths 0, 90 and 180 degrees, the lowest band's `log(R/G)` and `log(B/G)` contrast against
  the reference band (the quantity G0 and `lastlight observe` define, from
  `lastlight.g0.contrast.log_chromaticity`) is within 0.10 log units of Prague's; and
* the reference elevation's chromaticity is within 0.02 of Prague's in CIE 1976 u'v' (Euclidean).

If v1 misses a tolerance at some azimuth, that one comparison is marked
`xfail(strict=True)` naming the measured deviation; the renderer and the tolerance are not
changed to pass it. The same comparison for v0, the hazier atmosphere (`aod_550` 0.1, issue #34's
conductor note) and the other ways the fixture can be read are diagnostics: they only have to
be finite. The numbers are in `docs/log/2026-10-12-prague-control.md`.

The fixture records, per atmosphere and relative azimuth, CIE XYZ at the camera's band and
reference elevations (`build_views()["nominal"]`) followed by two diagnostic elevations, on the
scale of `lastlight.render.colorimetry` (a white card lit by the extraterrestrial sun has Y = 1).
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity, rendered_contrasts
from lastlight.render import Atmosphere
from lastlight.render.atmosphere import (
    RAYLEIGH_SCALE_HEIGHT_KM,
    WAVELENGTHS_NM,
    rayleigh_optical_depth,
)
from lastlight.render.curve import sky_xyz
from lastlight.render.v1 import multiple_scattering_xyz

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "prague" / "sun10.json"
NOTICE = FIXTURE.parent / "NOTICE"
PROVENANCE = FIXTURE.parent / "PROVENANCE.md"
TOOL = ROOT / "tools" / "prague_reference.py"
CACHE = Path.home() / ".cache" / "lastlight" / "prague"

SUN_ELEVATION = 10.0
OBSERVER_ALTITUDE_KM = 0.34
GROUND_ALBEDO = 0.2
AZIMUTHS = (0.0, 90.0, 180.0)
LOW_AEROSOL = 0.03
HAZIER = 0.1
NOMINAL_VIEWS = 6  # WHTM1's five bands, then its reference elevation
DIAGNOSTIC_ELEVATIONS = (30.0, 60.0)
KOSCHMIEDER = 3.912
"""Visual range in units of 1 / extinction: ln(50), the 2 percent contrast threshold."""

CONTRAST_TOLERANCE = 0.10
"""Log units, per channel ratio: the issue's tolerance on the lowest band's contrast."""
CHROMATICITY_TOLERANCE = 0.02
"""Euclidean distance in CIE 1976 u'v' at the reference elevation: the issue's tolerance."""

VARIANTS = (
    "xyz_binned",
    "xyz_inverse_visibility",
    "xyz_altitude_0",
    "xyz_matched_aerosol_depth",
)
"""The fixture's diagnostic readings of Prague, each laid out like the primary `xyz`."""


@cache
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def atmosphere_entry(aod: float) -> dict:
    return fixture()["atmospheres"][f"{aod:g}"]


def prague_xyz(aod: float, azimuth: float, variant: str = "xyz") -> np.ndarray:
    """Prague's XYZ at the fixture's view elevations, shape (8, 3)."""
    entry = atmosphere_entry(aod)
    block = entry["xyz"] if variant == "xyz" else entry["diagnostics"][variant]
    return np.array(block[f"{azimuth:g}"], dtype=float)


def view_elevations() -> list[float]:
    return fixture()["view_elevations_deg"]


def v1_xyz(aod: float, azimuth: float, engine: str = "v1") -> np.ndarray:
    return sky_xyz(
        Atmosphere(aod_550=aod), SUN_ELEVATION, view_elevations(), azimuth, engine=engine
    )


def lowest_band_contrast(xyz: np.ndarray) -> np.ndarray:
    """`[log(R/G), log(B/G)]` of the lowest band minus the reference, as G0 and observe define it.

    `xyz` holds the nominal views (bands, then the reference) first. This is
    `lastlight.g0.contrast.rendered_contrasts` applied to XYZ already in hand.
    """
    logs, _ = log_chromaticity(xyz[:NOMINAL_VIEWS])
    return (logs[:-1] - logs[-1])[0]


def u_prime_v_prime(xyz: np.ndarray) -> np.ndarray:
    """CIE 1976 (u', v') of XYZ values on the last axis."""
    x, y, z = np.moveaxis(np.asarray(xyz, dtype=float), -1, 0)
    denominator = x + 15.0 * y + 3.0 * z
    return np.stack([4.0 * x / denominator, 9.0 * y / denominator], axis=-1)


def contrast_deviation(xyz: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`xyz`'s lowest-band contrast minus `reference`'s: the two log-ratio differences."""
    return lowest_band_contrast(xyz) - lowest_band_contrast(reference)


def chromaticity_distance(xyz: np.ndarray, reference: np.ndarray) -> float:
    """Euclidean distance in u'v' between the two reference-elevation colours."""
    here = u_prime_v_prime(xyz[NOMINAL_VIEWS - 1])
    there = u_prime_v_prime(reference[NOMINAL_VIEWS - 1])
    return float(np.linalg.norm(here - there))


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prague_reference", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("prague_reference", module)
    spec.loader.exec_module(module)
    return module


# --- the fixture ------------------------------------------------------------------------------


def test_the_fixture_has_the_documented_layout() -> None:
    data = fixture()
    assert data["format"] == "lastlight-prague-control"
    assert data["version"] == 1
    assert data["sun_elevation_deg"] == SUN_ELEVATION
    assert data["observer_altitude_km"] == OBSERVER_ALTITUDE_KM
    assert data["ground_albedo"] == GROUND_ALBEDO
    assert [float(azimuth) for azimuth in data["rel_azimuths_deg"]] == list(AZIMUTHS)
    views = data["view_elevations_deg"]
    assert len(views) == NOMINAL_VIEWS + len(DIAGNOSTIC_ELEVATIONS)
    assert views[NOMINAL_VIEWS:] == list(DIAGNOSTIC_ELEVATIONS)
    assert set(data["atmospheres"]) == {f"{LOW_AEROSOL:g}", f"{HAZIER:g}"}
    for aod in (LOW_AEROSOL, HAZIER):
        assert atmosphere_entry(aod)["aod_550"] == aod
        for azimuth in AZIMUTHS:
            for variant in ("xyz", *VARIANTS):
                xyz = prague_xyz(aod, azimuth, variant)
                assert xyz.shape == (len(views), 3), (aod, azimuth, variant)
                assert np.isfinite(xyz).all() and (xyz > 0.0).all(), (aod, azimuth, variant)


def test_the_fixture_views_are_the_cameras_nominal_views() -> None:
    """A refit of WHTM1 moves its bands: the control must be regenerated, not left stale."""
    nominal = build_views()["nominal"].elevations_deg
    recorded = view_elevations()[:NOMINAL_VIEWS]
    assert len(nominal) == NOMINAL_VIEWS
    assert recorded == pytest.approx(list(nominal), abs=1e-9), (
        "the camera's nominal band and reference elevations changed since the fixture was made: "
        "rerun tools/prague_reference.py fixture"
    )


def test_the_fixture_visibility_is_the_documented_mapping() -> None:
    """Koschmieder at 550 nm with v1's own ground extinction, inside Prague's valid range."""
    low, high = fixture()["prague_visibility_range_km"]
    for aod in (LOW_AEROSOL, HAZIER):
        atm = Atmosphere(aod_550=aod)
        rayleigh = float(rayleigh_optical_depth(550.0, atm.pressure_hpa)) / RAYLEIGH_SCALE_HEIGHT_KM
        beta = rayleigh + aod / atm.aerosol_scale_height_km
        visibility = KOSCHMIEDER / beta
        assert atmosphere_entry(aod)["visibility_km"] == pytest.approx(visibility, rel=1e-9)
        assert atmosphere_entry(aod)["ground_extinction_per_km"] == pytest.approx(beta, rel=1e-9)
        assert low <= visibility <= high


def test_the_tool_computes_the_same_mapping() -> None:
    tool = load_tool()
    for aod in (LOW_AEROSOL, HAZIER):
        recorded = atmosphere_entry(aod)["visibility_km"]
        assert tool.visibility_km(Atmosphere(aod_550=aod)) == pytest.approx(recorded, rel=1e-9)


def test_the_notice_and_provenance_name_the_model() -> None:
    """The licence asks for its notice wherever the model's numbers go; PROVENANCE pins the run."""
    provenance = fixture()["provenance"]
    model = provenance["model"]
    notice = NOTICE.read_text(encoding="utf-8")
    assert "Prague Sky Model" in notice
    assert model["copyright"] == "Copyright 2022 Charles University"
    assert model["copyright"] in notice
    assert f"SPDX-License-Identifier: {model['licence']}" in notice
    assert "Licensed under the Apache License, Version 2.0" in notice
    assert "http://www.apache.org/licenses/LICENSE-2.0" in notice
    assert model["repository"].removesuffix(".git") in notice
    commit = model["commit"]
    digest = provenance["datasets"]["full"]["sha256"]
    assert len(commit) == 40 and len(digest) == 64  # a blank value would be "in" any text
    text = PROVENANCE.read_text(encoding="utf-8")
    assert commit in text
    assert digest in text


# --- the control ------------------------------------------------------------------------------


@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_v1_lowest_band_contrast_matches_prague(azimuth: float) -> None:
    delta = contrast_deviation(v1_xyz(LOW_AEROSOL, azimuth), prague_xyz(LOW_AEROSOL, azimuth))
    assert np.abs(delta).max() <= CONTRAST_TOLERANCE, (
        f"v1 minus Prague at relative azimuth {azimuth:g}: log(R/G) {delta[0]:+.3f}, "
        f"log(B/G) {delta[1]:+.3f} (tolerance {CONTRAST_TOLERANCE})"
    )


@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_v1_reference_chromaticity_matches_prague(azimuth: float) -> None:
    distance = chromaticity_distance(v1_xyz(LOW_AEROSOL, azimuth), prague_xyz(LOW_AEROSOL, azimuth))
    assert distance <= CHROMATICITY_TOLERANCE, (
        f"v1 against Prague at relative azimuth {azimuth:g}: u'v' distance {distance:.4f} "
        f"(tolerance {CHROMATICITY_TOLERANCE})"
    )


# --- the checks can fail ----------------------------------------------------------------------


def test_the_tolerances_are_the_issues() -> None:
    """Issue #34 fixed 0.10 log units and 0.02 in u'v' before any comparison."""
    assert CONTRAST_TOLERANCE == 0.10
    assert CHROMATICITY_TOLERANCE == 0.02


@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_the_contrast_helper_is_g0s_definition(azimuth: float) -> None:
    """The control's contrast is `rendered_contrasts`': a band minus the reference band."""
    atm = Atmosphere(aod_550=LOW_AEROSOL)
    nominal = build_views()["nominal"]
    xyz = sky_xyz(atm, SUN_ELEVATION, nominal.elevations_deg, azimuth)
    contrasts, _ = rendered_contrasts(atm, SUN_ELEVATION, nominal, azimuth)
    assert contrasts.shape == (NOMINAL_VIEWS - 1, 2)
    assert lowest_band_contrast(xyz) == pytest.approx(contrasts[0], abs=1e-12)


def xyz_at_chromaticity(u: float, v: float, luminance: float = 1.0) -> np.ndarray:
    """XYZ with the given CIE 1976 (u', v') and Y: the inverse of `u_prime_v_prime`."""
    denominator = 9.0 * luminance / v
    x = u * denominator / 4.0
    z = (denominator - x - 15.0 * luminance) / 3.0
    return np.array([x, luminance, z])


def test_the_chromaticity_distance_is_euclidean() -> None:
    """A 3-4-5 pair: the Euclidean distance is 0.005, the largest axis difference only 0.004."""
    here = xyz_at_chromaticity(0.200, 0.460)
    there = xyz_at_chromaticity(0.203, 0.464)
    assert u_prime_v_prime(here) == pytest.approx([0.200, 0.460], abs=1e-12)
    assert u_prime_v_prime(there) == pytest.approx([0.203, 0.464], abs=1e-12)

    def views(reference: np.ndarray) -> np.ndarray:
        """Nominal views whose reference elevation alone has the colour `reference`."""
        rows = np.tile(xyz_at_chromaticity(0.300, 0.300), (NOMINAL_VIEWS, 1))
        rows[NOMINAL_VIEWS - 1] = reference
        return rows

    assert chromaticity_distance(views(here), views(there)) == pytest.approx(0.005, abs=1e-9)
    assert chromaticity_distance(views(there), views(here)) == pytest.approx(0.005, abs=1e-9)


def test_the_chromaticity_helper_reproduces_a_known_white() -> None:
    """D65 (X, Y, Z) = (0.95047, 1, 1.08883) has u' = 0.19784 and v' = 0.46834."""
    u, v = u_prime_v_prime(np.array([0.95047, 1.0, 1.08883]))
    assert u == pytest.approx(0.19784, abs=2e-5)
    assert v == pytest.approx(0.46834, abs=2e-5)


def test_the_contrast_check_sees_a_lowest_band_that_is_too_blue() -> None:
    truth = prague_xyz(LOW_AEROSOL, 0.0)
    assert np.abs(contrast_deviation(truth, truth)).max() == 0.0
    bluer = truth.copy()
    bluer[0, 2] *= 1.3  # the lowest band alone, Z only: a bluer band against the same reference
    delta = contrast_deviation(bluer, truth)
    assert delta[1] > CONTRAST_TOLERANCE  # log(B/G) rises by well over the tolerance
    assert np.abs(delta).max() > CONTRAST_TOLERANCE


def test_the_chromaticity_check_sees_a_reference_colour_that_is_off() -> None:
    truth = prague_xyz(LOW_AEROSOL, 0.0)
    assert chromaticity_distance(truth, truth) == 0.0
    off = truth.copy()
    off[NOMINAL_VIEWS - 1, 2] *= 1.5
    assert chromaticity_distance(off, truth) > CHROMATICITY_TOLERANCE


# --- the tool's reading of Prague's channels ---------------------------------------------------


def test_prague_channels_are_interpolated_linearly_between_their_centres() -> None:
    tool = load_tool()
    start, width, count = 320.0, 40.0, 11
    centres = start + (np.arange(count) + 0.5) * width
    values = 2.0 + 0.01 * centres  # linear in wavelength: interpolation must reproduce it
    spectrum = tool.channel_spectrum(values, start, width)
    assert spectrum.shape == WAVELENGTHS_NM.shape
    grid = np.array(WAVELENGTHS_NM)
    inside = (grid >= centres[0]) & (grid <= centres[-1])
    assert spectrum[inside] == pytest.approx(2.0 + 0.01 * grid[inside], rel=1e-12)
    assert spectrum[grid > centres[-1]] == pytest.approx(values[-1])  # held flat


def test_the_binned_reading_is_the_models_own_piecewise_constant_lookup() -> None:
    tool = load_tool()
    start, width, count = 320.0, 40.0, 11
    values = np.arange(1.0, count + 1.0)
    spectrum = tool.channel_spectrum(values, start, width, binned=True)
    for wavelength, value in zip(WAVELENGTHS_NM, spectrum, strict=True):
        index = math.floor((wavelength - start) / width)
        expected = values[index] if 0 <= index < count else 0.0
        assert value == expected, wavelength


# --- diagnostics: reported in the build log, only required to be finite ------------------------


@pytest.mark.parametrize("aod", [LOW_AEROSOL, HAZIER])
@pytest.mark.parametrize("azimuth", AZIMUTHS)
def test_v0_and_the_multiple_scattering_share_are_reported_as_finite_numbers(
    aod: float, azimuth: float
) -> None:
    prague = prague_xyz(aod, azimuth)
    for engine in ("v0", "v1"):
        theirs = v1_xyz(aod, azimuth, engine)
        assert np.isfinite(contrast_deviation(theirs, prague)).all()
        assert math.isfinite(chromaticity_distance(theirs, prague))
        ratio = theirs[:, 1] / prague[:, 1]  # total luminance against Prague, every view
        assert np.isfinite(ratio).all() and (ratio > 0.0).all()
    total = v1_xyz(aod, azimuth)
    multiple = multiple_scattering_xyz(
        Atmosphere(aod_550=aod), SUN_ELEVATION, view_elevations(), azimuth
    )
    share = multiple[:, 1] / total[:, 1]
    assert np.isfinite(share).all() and ((share > 0.0) & (share < 1.0)).all()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("aod", [LOW_AEROSOL, HAZIER])
def test_the_other_readings_of_prague_give_finite_deviations(variant: str, aod: float) -> None:
    for azimuth in AZIMUTHS:
        theirs = prague_xyz(aod, azimuth, variant)
        ours = v1_xyz(aod, azimuth)
        assert np.isfinite(contrast_deviation(ours, theirs)).all()
        assert math.isfinite(chromaticity_distance(ours, theirs))


# --- regeneration (needs the built model, so deselected by default) -----------------------------


@pytest.mark.network
def test_the_fixture_matches_a_fresh_evaluation_of_the_built_model() -> None:
    """`tools/prague_reference.py fixture --check` re-evaluates Prague and compares to the file."""
    if not (CACHE / "build" / "prague_eval").exists():
        pytest.skip("the Prague model is not built: run tools/prague_reference.py fetch build")
    result = subprocess.run(
        [sys.executable, str(TOOL), "fixture", "--check"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
