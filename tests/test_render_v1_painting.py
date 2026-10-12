"""Renderer v1 and the daily painting: the forecast keeps v0 until the adoption rule passes (#32).

`painting.paint` gains `engine="v0"`, and the model name follows the engine (`lastlight-v0` or
`lastlight-v1`); `model.version` stays the package version. With the default engine the painting
of the recorded HRRR crop must be what it was before this slice. That is checked against
`tests/fixtures/painting_v0_before_v1.json`, the forecast and strip that `paint` made on the
commit this slice branched from, with the same crop, clock and code SHA as `tests/test_painting.py`.
The comparison is by value and not by hash: a SHA-256 of v0's output moves with the last digits
of a float, and those differ between macOS and Linux.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import daily_support as support
import hrrr_support
import numpy as np
import pytest
from PIL import Image

from lastlight import geometry
from lastlight.atmosphere import reduce_transect
from lastlight.inputs import hrrr
from lastlight.painting import Painting, paint
from lastlight.render import Atmosphere, render_curve
from lastlight.site import REFERENCE_POINT

BEFORE_THIS_SLICE = Path(__file__).parent / "fixtures" / "painting_v0_before_v1.json"
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-12  # for values at or near zero, where a relative bound means nothing
STRIP_TOLERANCE = 1  # code values, on each of the three channels of every pixel
NOT_PORTABLE = "forecast.strip_sha256"  # a SHA-256 of PNG bytes: the strip is compared by pixels


def _paint(cache_dir: Path, **options) -> Painting:
    return paint(
        support.EVENING,
        transport=support.transport(),
        now_utc=support.before_sunset(),
        cache_dir=cache_dir,
        cycle_hour=support.CYCLE_HOUR,
        forecast_hour=support.FORECAST_HOUR,
        code_sha=support.CODE_SHA,
        **options,
    )


@pytest.fixture(scope="module")
def default_painting(tmp_path_factory: pytest.TempPathFactory) -> Painting:
    return _paint(tmp_path_factory.mktemp("hrrr-default"))


@pytest.fixture(scope="module")
def v1_painting(tmp_path_factory: pytest.TempPathFactory) -> Painting:
    return _paint(tmp_path_factory.mktemp("hrrr-v1"), engine="v1")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def differences(expected: Any, got: Any, path: str = "forecast") -> list[str]:
    """Where `got` departs from `expected`: same keys and lengths, strings and integers exact.

    Floats must agree within `RELATIVE_TOLERANCE` (and `ABSOLUTE_TOLERANCE` near zero). The one
    string left out is `NOT_PORTABLE`: PNG bytes can differ between zlib builds while the pixels
    are the same, so the strip is compared by its pixels instead.
    """
    if isinstance(expected, dict):
        if not isinstance(got, dict) or got.keys() != expected.keys():
            return [f"{path}: keys {sorted(got) if isinstance(got, dict) else got!r}"]
        return [
            line
            for key in expected
            if f"{path}.{key}" != NOT_PORTABLE
            for line in differences(expected[key], got[key], f"{path}.{key}")
        ]
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return [f"{path}: length {len(got) if isinstance(got, list) else got!r}"]
        return [
            line
            for index, (want, have) in enumerate(zip(expected, got, strict=True))
            for line in differences(want, have, f"{path}[{index}]")
        ]
    if isinstance(expected, float):
        close = isinstance(got, float) and math.isclose(
            expected, got, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE
        )
        return [] if close else [f"{path}: {got!r} is not {expected!r}"]
    same = type(got) is type(expected) and got == expected
    return [] if same else [f"{path}: {got!r} is not {expected!r}"]


def pixels(png: bytes) -> np.ndarray:
    """The decoded RGB values of a PNG, as signed integers so a difference cannot wrap."""
    image = Image.open(io.BytesIO(png))
    assert image.mode == "RGB"
    return np.asarray(image, dtype=np.int16)


def recorded_atmosphere(cache_dir: Path) -> Atmosphere:
    """The reduction of the recorded crop, done step by step here, not through `paint`."""
    lat, lon, _elevation = REFERENCE_POINT
    fields = hrrr.fetch_fields(
        hrrr_support.CYCLE_DATE,
        support.CYCLE_HOUR,
        support.FORECAST_HOUR,
        hrrr.DEFAULT_FIELDS,
        cache_dir,
        transport=support.transport(),
    )
    points = geometry.transect(
        support.EVENING, lat, lon, [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
    )
    return reduce_transect(hrrr.sample_transect(fields, points), support.EVENING).atmosphere


def test_the_default_engine_paints_the_same_painting_as_before_this_slice(
    default_painting: Painting,
) -> None:
    """The default painting equals the one `paint` made before engine v1 existed.

    `tests/fixtures/painting_v0_before_v1.json` was made on 95f3ecd5ddef (the commit this slice
    branched from, the head of main after #38) in a clean `git worktree` after `uv sync --frozen`,
    with `tests/` on the path:

        painting = paint(support.EVENING, transport=support.transport(),
                         now_utc=support.before_sunset(), cache_dir=<an empty directory>,
                         cycle_hour=support.CYCLE_HOUR, forecast_hour=support.FORECAST_HOUR,
                         code_sha=support.CODE_SHA)
        document = {"forecast": painting.forecast.model_dump(mode="json"),
                    "strip_png_base64": base64.b64encode(painting.strip_png).decode("ascii")}
        Path(...).write_text(json.dumps(document, indent=1) + "\\n")

    On macOS arm64 that checkout's canonical forecast digest was f76d6917d2fd and its strip
    digest e7901571c72f, the two SHA-256 pins this test replaced. They were dropped because a
    byte hash is not portable: v0's floats differ in their last digits between macOS and Linux.

    Every string, integer and key is compared exactly. Every float agrees within a relative
    1e-9 (absolute 1e-12 near zero), written unrounded so the six-digit canonical rounding
    cannot turn float noise into a difference. `strip_sha256` is the one string not compared
    with the fixture: it hashes PNG bytes, so it is checked against this run's own strip, and
    the strip is compared by decoded pixels, each channel within one code value.
    """
    fixture = json.loads(BEFORE_THIS_SLICE.read_text(encoding="utf-8"))
    before_png = base64.b64decode(fixture["strip_png_base64"])
    assert sha256(before_png) == fixture["forecast"]["strip_sha256"], "the fixture disagrees"

    problems = differences(fixture["forecast"], default_painting.forecast.model_dump(mode="json"))
    assert not problems, f"{len(problems)} differences, the first: " + "; ".join(problems[:5])

    assert default_painting.strip_png is not None
    assert default_painting.forecast.strip_sha256 == sha256(default_painting.strip_png)
    now, before = pixels(default_painting.strip_png), pixels(before_png)
    assert now.shape == before.shape
    assert int(np.abs(now - before).max()) <= STRIP_TOLERANCE


def test_the_comparison_allows_float_noise_and_nothing_else() -> None:
    expected = {
        "name": "lastlight-v0",
        "count": 17,
        "curve": [{"L": 70.0, "a": 0.0, "b": 41.345883035285524}],
    }

    def with_curve(**changes: Any) -> dict[str, Any]:
        return {**expected, "curve": [{**expected["curve"][0], **changes}]}

    noisy = with_curve(L=70.0 * (1 + 1e-12), a=1e-15, b=41.345883035285524 * (1 - 1e-12))
    assert differences(expected, noisy, "x") == []
    for broken in (
        with_curve(L=70.0 * (1 + 1e-6)),
        with_curve(a=1e-6),
        with_curve(L=70),
        with_curve(extra=1.0),
        {**expected, "name": "lastlight-v1"},
        {**expected, "count": 17.0},
        {**expected, "curve": expected["curve"] * 2},
        {k: v for k, v in expected.items() if k != "count"},
    ):
        assert differences(expected, broken, "x") != []
    assert differences({"strip_sha256": "a" * 64}, {"strip_sha256": "b" * 64}) == []


def test_the_default_model_is_named_lastlight_v0(default_painting: Painting) -> None:
    assert default_painting.forecast.model.name == "lastlight-v0"
    assert default_painting.forecast.model.version == "0.1.0"


def test_painting_with_v1_names_the_model_lastlight_v1_and_keeps_the_version(
    v1_painting: Painting,
) -> None:
    assert v1_painting.forecast.model.name == "lastlight-v1"
    assert v1_painting.forecast.model.version == "0.1.0"


def test_painting_with_v1_renders_the_v1_curve_of_the_same_atmosphere(
    v1_painting: Painting, default_painting: Painting, tmp_path: Path
) -> None:
    expected = render_curve(recorded_atmosphere(tmp_path), engine="v1")
    got = [point.model_dump() for point in v1_painting.forecast.curve]
    assert len(got) == len(expected)
    for point, row in zip(got, expected, strict=True):
        assert (point["sun_elevation"], point["view_elevation"]) == (
            row["sun_elevation"],
            row["view_elevation"],
        )
        assert (point["L"], point["a"], point["b"]) == pytest.approx((row["L"], row["a"], row["b"]))
    assert v1_painting.forecast.curve != default_painting.forecast.curve
    assert v1_painting.forecast.strip_sha256 != default_painting.forecast.strip_sha256
    assert v1_painting.forecast.inputs == default_painting.forecast.inputs


@pytest.mark.parametrize("engine", ["v2", "", "lastlight-v1"])
def test_an_unknown_engine_is_refused_before_anything_is_fetched(
    engine: str, tmp_path: Path
) -> None:
    transport = support.transport()
    with pytest.raises(ValueError, match="engine"):
        paint(
            support.EVENING,
            transport=transport,
            now_utc=support.before_sunset(),
            cache_dir=tmp_path,
            cycle_hour=support.CYCLE_HOUR,
            forecast_hour=support.FORECAST_HOUR,
            code_sha=support.CODE_SHA,
            engine=engine,
        )
    assert transport.calls == []
