"""The quadrature keyword of renderer v0's single scattering (issue #40).

`"registered"` is the integration renderer v1's pre-registered table was built beside and the
camera magnitude check pins; it must never move. `"adaptive"` cuts the optically thick pieces of a
low sightline into thinner ones, and only the daily forecast's v0 painting asks for it. These tests
check, in order: the registered integration by value (the numbers below were computed on `main` at
807ed34, before the keyword existed), the keyword's contract, that adaptive is the registered
result exactly where nothing needs refining, that it differs where something does, and which
callers ask for it. The accuracy of adaptive against the brute-force sum is in
`test_render_physics.py`, beside the comparison it extends.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import daily_support as support
import numpy as np
import pytest

import lastlight.painting
import lastlight.render
from lastlight.daily import seal_daily
from lastlight.painting import paint
from lastlight.render import (
    VIEW_ELEVATIONS,
    WAVELENGTHS_NM,
    Atmosphere,
    render_curve,
    render_row,
    sky_spectrum,
    sky_xyz,
)
from lastlight.render import __all__ as render_exports

# WHTM1's lowest band and its reference band, at the frozen fit's nominal pixel scale.
LOWEST_BAND = 1.5066301585401483
REFERENCE_BAND = 3.9943811580741357
PINNED_VIEWS = [LOWEST_BAND, REFERENCE_BAND]

# sky_xyz(Atmosphere(aod_550=aod, ozone_du=272.0), sun, PINNED_VIEWS, azimuth, engine=engine),
# computed on main at 807ed34. Never regenerate these to make a test pass.
REGISTERED_PINS = [
    pytest.param(
        "v0", 0.06, 0.0, 20.0,
        [[0.022784643320134683, 0.01757923820655384, 0.001470827124095022],
         [0.030757163094225295, 0.028056610958648812, 0.007946208579905665]],
        id="v0-aod0.06-sun0",
    ),
    pytest.param(
        "v0", 0.03, -3.0, 25.0,
        [[0.0037726643085672605, 0.003316645575444221, 0.0005688199397762706],
         [0.005210648253853109, 0.005354349399999691, 0.002715008199005599]],
        id="v0-aod0.03-sun-3",
    ),
    pytest.param(
        "v0", 0.2, 3.0, 15.0,
        [[0.04340643459434342, 0.03389462084318697, 0.0033383335171692954],
         [0.08763549519869064, 0.07519843350596886, 0.012676410961999372]],
        id="v0-aod0.2-sun3",
    ),
    pytest.param(
        "v1", 0.06, 0.0, 20.0,
        [[0.036618204379745126, 0.031229995875955967, 0.012895763795915597],
         [0.042844952295487154, 0.0404837916060278, 0.020678019486736227]],
        id="v1-aod0.06-sun0",
    ),
    pytest.param(
        "v1", 0.03, -3.0, 25.0,
        [[0.005803768231628109, 0.005443700408744306, 0.0028857776736254457],
         [0.006884026755780439, 0.007155401230767449, 0.0052584096899803276]],
        id="v1-aod0.03-sun-3",
    ),
    pytest.param(
        "v1", 0.2, 3.0, 15.0,
        [[0.09255670618269227, 0.08138806553723171, 0.032392593837250565],
         [0.14232634249246007, 0.12913720925432798, 0.046845946329301724]],
        id="v1-aod0.2-sun3",
    ),
]  # fmt: skip
PIN_FIELDS = "engine, aod, sun, azimuth, expected"


def _atmosphere(aod: float) -> Atmosphere:
    return Atmosphere(aod_550=aod, ozone_du=272.0)


# --- the registered integration does not move ---------------------------------------------


@pytest.mark.parametrize(PIN_FIELDS, REGISTERED_PINS)
def test_the_default_integration_is_the_registered_one_by_value(
    engine: str, aod: float, sun: float, azimuth: float, expected: list[list[float]]
) -> None:
    actual = sky_xyz(_atmosphere(aod), sun, PINNED_VIEWS, azimuth, engine=engine)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize(PIN_FIELDS, REGISTERED_PINS)
def test_registered_passed_explicitly_gives_the_same_values(
    engine: str, aod: float, sun: float, azimuth: float, expected: list[list[float]]
) -> None:
    actual = sky_xyz(
        _atmosphere(aod), sun, PINNED_VIEWS, azimuth, engine=engine, quadrature="registered"
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-15)


def test_the_default_is_exactly_the_registered_quadrature() -> None:
    atm = Atmosphere(aod_550=1.0)
    np.testing.assert_array_equal(
        sky_spectrum(atm, 3.0, 2.0, 30.0),
        sky_spectrum(atm, 3.0, 2.0, 30.0, quadrature="registered"),
    )
    assert render_curve(atm) == render_curve(atm, quadrature="registered")
    assert render_row(atm, 1.0, 30.0) == render_row(atm, 1.0, 30.0, quadrature="registered")


# --- the keyword ----------------------------------------------------------------------------


def test_the_quadratures_are_registered_and_adaptive_and_exported() -> None:
    assert lastlight.render.QUADRATURES == ("registered", "adaptive")
    assert "QUADRATURES" in render_exports


def _every_entry_point(atm: Atmosphere, quadrature: str) -> list[Any]:
    return [
        lambda: sky_spectrum(atm, 3.0, 5.0, quadrature=quadrature),
        lambda: sky_xyz(atm, 3.0, [2.0, 5.0], quadrature=quadrature),
        lambda: sky_xyz(atm, 3.0, [2.0, 5.0], quadrature=quadrature, engine="v1"),
        lambda: sky_xyz(atm, 3.0, [], quadrature=quadrature),  # no view: still checked
        lambda: render_row(atm, 3.0, quadrature=quadrature),
        lambda: render_row(atm, 3.0, quadrature=quadrature, engine="v1"),
        lambda: render_curve(atm, quadrature=quadrature),
        lambda: render_curve(atm, quadrature=quadrature, engine="v1"),
    ]


@pytest.mark.parametrize("bad", ["Adaptive", "", "exact", "registered ", "fine"])
def test_an_unknown_quadrature_is_a_value_error_naming_the_accepted_ones(bad: str) -> None:
    for call in _every_entry_point(Atmosphere(aod_550=0.1), bad):
        with pytest.raises(ValueError) as caught:
            call()
        message = str(caught.value)
        assert "registered" in message and "adaptive" in message, message


def test_the_quadrature_is_a_keyword_only_argument_defaulting_to_registered() -> None:
    for function in (sky_spectrum, sky_xyz, render_row, render_curve):
        parameter = inspect.signature(function).parameters["quadrature"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function.__name__
        assert parameter.default == "registered", function.__name__


def test_both_quadratures_are_accepted_everywhere() -> None:
    for quadrature in lastlight.render.QUADRATURES:
        for call in _every_entry_point(Atmosphere(aod_550=0.1), quadrature):
            call()


# --- adaptive is the registered result where nothing needs refining ----------------------

LOW_LOAD_GEOMETRIES = [
    (sun, view, azimuth)
    for sun in (6.0, 0.0, -3.0, -8.0)
    for view in (LOWEST_BAND, 2.0, 3.3, 5.0, 20.0, 45.0, 90.0)
    for azimuth in (0.0, 30.0, 135.0)
]


@pytest.mark.parametrize("aod", [0.0, 0.03, 0.06])
def test_adaptive_is_the_registered_result_exactly_at_ordinary_aerosol_loads(aod: float) -> None:
    atm = _atmosphere(aod)
    for sun, view, azimuth in LOW_LOAD_GEOMETRIES:
        np.testing.assert_array_equal(
            sky_spectrum(atm, sun, view, azimuth, quadrature="adaptive"),
            sky_spectrum(atm, sun, view, azimuth),
            err_msg=f"sun {sun}, view {view}, azimuth {azimuth}",
        )


def test_an_ordinary_load_curve_is_the_same_curve_under_adaptive() -> None:
    atm = Atmosphere(aod_550=0.06, ozone_du=272.0)
    assert render_curve(atm, quadrature="adaptive") == render_curve(atm)


def test_the_atmospheres_beside_the_default_are_also_left_alone_when_thin() -> None:
    """A clean, dry, low-pressure atmosphere and one with a thin aerosol layer."""
    for atm in (
        Atmosphere(aod_550=0.0, ozone_du=0.0, pressure_hpa=500.0),
        Atmosphere(aod_550=0.03, aerosol_scale_height_km=0.3),
    ):
        np.testing.assert_array_equal(
            sky_spectrum(atm, 2.0, LOWEST_BAND, 30.0, quadrature="adaptive"),
            sky_spectrum(atm, 2.0, LOWEST_BAND, 30.0),
        )


def test_adaptive_stays_within_one_percent_of_registered_up_to_aod_0_1() -> None:
    """Where adaptive does engage at light loads, it moves the answer a little, never a lot."""
    worst = 0.0
    for aod in (0.07, 0.085, 0.1):
        atm = _atmosphere(aod)
        for sun, view, azimuth in LOW_LOAD_GEOMETRIES:
            registered = sky_spectrum(atm, sun, view, azimuth)
            adaptive = sky_spectrum(atm, sun, view, azimuth, quadrature="adaptive")
            lit = registered > 0.0
            worst = max(worst, float(np.abs(adaptive[lit] / registered[lit] - 1.0).max()))
    assert worst < 0.01, worst


# --- adaptive is a different answer where the sightline is optically thick --------------


def test_adaptive_raises_the_low_sky_under_heavy_aerosol_where_registered_reads_low() -> None:
    """The case of issue #40: aod 1.0, sun +6, 30 degrees round, WHTM1's lowest band, 450 nm.

    Against the brute-force sum registered is 5.8 percent low and adaptive within 0.5 percent
    (`test_render_physics.py`); here only the direction and size of the move are pinned.
    """
    atm = Atmosphere(aod_550=1.0)
    registered = sky_spectrum(atm, 6.0, LOWEST_BAND, 30.0)
    adaptive = sky_spectrum(atm, 6.0, LOWEST_BAND, 30.0, quadrature="adaptive")
    blue = int(np.flatnonzero(WAVELENGTHS_NM == 450.0)[0])
    assert 1.04 < adaptive[blue] / registered[blue] < 1.08, adaptive[blue] / registered[blue]


def test_adaptive_is_deterministic_and_finite() -> None:
    atm = Atmosphere(aod_550=1.0)
    first = sky_spectrum(atm, 0.0, 2.0, 0.0, quadrature="adaptive")
    np.testing.assert_array_equal(first, sky_spectrum(atm, 0.0, 2.0, 0.0, quadrature="adaptive"))
    assert np.isfinite(first).all() and (first >= 0.0).all()


def test_a_sky_in_earths_shadow_is_still_black_under_adaptive() -> None:
    for aod in (0.1, 1.0):
        for view in (2.0, 20.0, 45.0):
            sky = sky_spectrum(Atmosphere(aod_550=aod), -30.0, view, quadrature="adaptive")
            assert not sky.any()


def test_adaptive_keeps_the_mirror_symmetry_in_relative_azimuth() -> None:
    atm = Atmosphere(aod_550=0.5)
    np.testing.assert_allclose(
        sky_spectrum(atm, 3.0, 2.0, 40.0, quadrature="adaptive"),
        sky_spectrum(atm, 3.0, 2.0, -40.0, quadrature="adaptive"),
        rtol=1e-12,
    )


def test_adaptive_copes_with_a_level_view_through_a_single_thick_shell() -> None:
    """A view at 0 degrees runs 25 km inside the lowest 50 m shell: finite, and not wild."""
    atm = Atmosphere(aod_550=1.0)
    level = sky_spectrum(atm, 6.0, 0.0, quadrature="adaptive")
    registered = sky_spectrum(atm, 6.0, 0.0)
    assert np.isfinite(level).all() and (level >= 0.0).all()
    np.testing.assert_allclose(level, registered, rtol=0.5)


# --- the keyword reaches every renderer entry point -------------------------------------


def test_sky_xyz_asks_the_spectrum_for_the_quadrature_it_was_given() -> None:
    atm = Atmosphere(aod_550=1.0)
    views = [LOWEST_BAND, 5.0]
    plain = sky_xyz(atm, 6.0, views, 30.0)
    adaptive = sky_xyz(atm, 6.0, views, 30.0, quadrature="adaptive")
    assert (adaptive[0] > plain[0]).all()  # the thick, low view rises
    one_by_one = np.vstack(
        [sky_xyz(atm, 6.0, [view], 30.0, quadrature="adaptive") for view in views]
    )
    np.testing.assert_array_equal(adaptive, one_by_one)


def test_v1_adds_the_same_multiple_scattering_whatever_the_single_scattering_quadrature() -> None:
    atm = Atmosphere(aod_550=0.5)
    views = [LOWEST_BAND, 2.0, 3.3]
    extra = {}
    for quadrature in lastlight.render.QUADRATURES:
        v1 = sky_xyz(atm, 3.0, views, 30.0, engine="v1", quadrature=quadrature)
        v0 = sky_xyz(atm, 3.0, views, 30.0, engine="v0", quadrature=quadrature)
        extra[quadrature] = v1 - v0
    np.testing.assert_allclose(extra["adaptive"], extra["registered"], rtol=1e-9, atol=1e-15)
    v1_adaptive = sky_xyz(atm, 3.0, views, 30.0, engine="v1", quadrature="adaptive")
    v1_registered = sky_xyz(atm, 3.0, views, 30.0, engine="v1")
    assert (v1_adaptive[0] > v1_registered[0]).all()  # v1's single-scattering part moved


def test_render_row_and_render_curve_pass_the_quadrature_down() -> None:
    atm = Atmosphere(aod_550=1.0)
    row = render_row(atm, 5.0, 30.0, quadrature="adaptive")
    assert row != render_row(atm, 5.0, 30.0)
    curve = render_curve(atm, quadrature="adaptive")
    assert curve != render_curve(atm)
    assert [r for r in curve if r["sun_elevation"] == 5.0] == render_row(
        atm, 5.0, quadrature="adaptive"
    )
    assert len(curve) == 29 * len(VIEW_ELEVATIONS)


def test_adaptive_curve_for_a_heavy_aerosol_day_renders_in_under_ten_seconds() -> None:
    """Issue #40's budget: `render_curve` at aod 1.0 under adaptive stays under 10 s."""
    atm = Atmosphere(aod_550=1.0, ozone_du=300.0)
    start = time.perf_counter()
    render_curve(atm, quadrature="adaptive")
    assert time.perf_counter() - start < 10.0


# --- who asks for it --------------------------------------------------------------------


def _spy_on_the_painting_seam(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the keywords `paint` hands to `render_curve`, and still render."""
    calls: list[dict[str, Any]] = []
    real = lastlight.painting.render_curve

    def spy(atm: Atmosphere, **keywords: Any) -> Any:
        calls.append(dict(keywords))
        return real(atm, **keywords)

    monkeypatch.setattr(lastlight.painting, "render_curve", spy)
    return calls


@pytest.mark.usefixtures("trust_test_tsa")
def test_the_sealed_daily_painting_asks_for_the_adaptive_quadrature_with_v0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    support.install_tsa(monkeypatch, support.before_sunset())
    calls = _spy_on_the_painting_seam(monkeypatch)

    result = seal_daily(
        support.EVENING,
        tmp_path / "ledger",
        transport=support.transport(),
        now_utc=support.before_sunset(),
        cache_dir=tmp_path / "hrrr",
        cycle_hour=support.CYCLE_HOUR,
        forecast_hour=support.FORECAST_HOUR,
        tsa_url=support.TSA_URL,
        code_sha=support.CODE_SHA,
    )

    assert result.status == "sealed"
    assert calls == [{"engine": "v0", "quadrature": "adaptive"}]


def test_paint_asks_for_adaptive_by_default_and_the_registered_one_for_v1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only v0's painting opts in; v1's pre-registered output must not move."""
    calls = _spy_on_the_painting_seam(monkeypatch)
    for engine in ("v0", "v1"):
        paint(
            support.EVENING,
            transport=support.transport(),
            now_utc=support.before_sunset(),
            cache_dir=tmp_path / engine,
            cycle_hour=support.CYCLE_HOUR,
            forecast_hour=support.FORECAST_HOUR,
            code_sha=support.CODE_SHA,
            with_strip=False,
            engine=engine,
        )
    assert calls == [
        {"engine": "v0", "quadrature": "adaptive"},
        {"engine": "v1", "quadrature": "registered"},
    ]
