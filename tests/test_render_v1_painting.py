"""Renderer v1 and the daily painting: the forecast keeps v0 until the adoption rule passes (#32).

`painting.paint` gains `engine="v0"`, and the model name follows the engine (`lastlight-v0` or
`lastlight-v1`); `model.version` stays the package version. With the default engine the painting
of the recorded HRRR crop must be byte for byte what it was before this slice: the two SHA-256
digests below were computed from `paint` on the commit this slice branched from (d40b2b4), with
the same fixture, clock and code SHA as `tests/test_painting.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import daily_support as support
import hrrr_support
import pytest

from lastlight import geometry
from lastlight.atmosphere import reduce_transect
from lastlight.inputs import hrrr
from lastlight.painting import Painting, paint
from lastlight.render import Atmosphere, render_curve
from lastlight.seal.canonical import canonical_bytes
from lastlight.site import REFERENCE_POINT

CANONICAL_SHA256_BEFORE = "f76d6917d2fd4e39c92d61731b946d47c9f2c79879b8101bbf29702b6abe6c62"
STRIP_SHA256_BEFORE = "e7901571c72fa01862be84edc62106e5ce488871f6abe4c56cf1c713c3bfd5c7"


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


def test_the_default_engine_paints_the_same_bytes_as_before_this_slice(
    default_painting: Painting,
) -> None:
    assert sha256(canonical_bytes(default_painting.forecast)) == CANONICAL_SHA256_BEFORE
    assert default_painting.strip_png is not None
    assert sha256(default_painting.strip_png) == STRIP_SHA256_BEFORE
    assert default_painting.forecast.strip_sha256 == STRIP_SHA256_BEFORE


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
