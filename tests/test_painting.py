"""The daily painting: HRRR fixture in, sealed-ready `Forecast` and strip PNG out (issue #8)."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import daily_support as support
import hrrr_support
import image_factory
import pytest
from PIL import Image

from lastlight import geometry
from lastlight.atmosphere import reduce_transect
from lastlight.forecast import Forecast
from lastlight.inputs import hrrr
from lastlight.painting import Painting, current_code_sha, paint
from lastlight.privacy import scan
from lastlight.privacy.images import image_issues
from lastlight.privacy.rules import text_hits
from lastlight.render import SUN_ELEVATIONS, VIEW_ELEVATIONS, render_curve, render_strip
from lastlight.seal.canonical import canonical_bytes
from lastlight.site import REFERENCE_POINT
from lastlight.vividness import VIVIDNESS_SD


@pytest.fixture(scope="module")
def painting(tmp_path_factory: pytest.TempPathFactory) -> Painting:
    return paint(
        support.EVENING,
        transport=support.transport(),
        now_utc=support.before_sunset(),
        cache_dir=tmp_path_factory.mktemp("hrrr"),
        cycle_hour=support.CYCLE_HOUR,
        forecast_hour=support.FORECAST_HOUR,
        code_sha=support.CODE_SHA,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the forecast -------------------------------------------------------------------------


def test_the_painting_holds_a_valid_forecast_for_the_evening(painting: Painting) -> None:
    forecast = painting.forecast
    assert isinstance(forecast, Forecast)
    assert forecast.date == support.EVENING
    assert forecast.schema_version == 1
    lat, lon, elevation = REFERENCE_POINT
    assert (forecast.reference_point.lat, forecast.reference_point.lon) == (lat, lon)
    assert forecast.reference_point.elev_m == elevation


def test_the_model_block_names_the_v0_renderer_the_version_and_the_code(painting: Painting) -> None:
    model = painting.forecast.model
    assert model.name == "lastlight-v0"
    assert model.version == "0.1.0"
    assert model.code_sha == support.CODE_SHA


def test_every_input_is_recorded_with_its_source_cycle_and_digest(painting: Painting) -> None:
    inputs = painting.forecast.inputs
    assert [item.source for item in inputs] == [f"hrrr:{spec.name}" for spec in hrrr.DEFAULT_FIELDS]
    for item, spec in zip(inputs, hrrr.DEFAULT_FIELDS, strict=True):
        assert item.cycle_date == hrrr_support.CYCLE_DATE
        assert item.cycle_hour_utc == support.CYCLE_HOUR
        assert item.forecast_hour == support.FORECAST_HOUR
        assert item.digest == sha256(hrrr_support.crop_message(spec.name))


def test_the_curve_is_the_renderers_curve_for_the_reduced_atmosphere(
    painting: Painting, tmp_path: Path
) -> None:
    lat, lon, _elevation = REFERENCE_POINT
    fields = hrrr.fetch_fields(
        hrrr_support.CYCLE_DATE, support.CYCLE_HOUR, 0, hrrr.DEFAULT_FIELDS, tmp_path,
        transport=support.transport(),
    )  # fmt: skip
    points = geometry.transect(
        support.EVENING, lat, lon, [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
    )
    atmosphere = reduce_transect(hrrr.sample_transect(fields, points), support.EVENING).atmosphere
    expected = render_curve(atmosphere)

    curve = painting.forecast.curve
    assert len(curve) == len(SUN_ELEVATIONS) * len(VIEW_ELEVATIONS) == 203
    for point, row in zip(curve, expected, strict=True):
        assert (point.sun_elevation, point.view_elevation) == (
            row["sun_elevation"],
            row["view_elevation"],
        )
        assert (point.L, point.a, point.b) == pytest.approx((row["L"], row["a"], row["b"]))


def test_the_vividness_is_the_v0_placeholder_with_its_wide_spread(painting: Painting) -> None:
    vividness = painting.forecast.vividness
    assert vividness.sd == VIVIDNESS_SD
    assert 0.0 <= vividness.mean <= 1.0


def test_the_notes_carry_the_cloud_summary_and_name_the_v0_vividness_rule(
    painting: Painting,
) -> None:
    notes = painting.forecast.notes
    assert "not rendered" in notes
    assert "v0-placeholder" in notes
    assert "SBUV" in notes
    assert text_hits(notes) == []


# --- the strip ----------------------------------------------------------------------------


def test_the_forecast_names_the_strip_by_its_sha256(painting: Painting) -> None:
    assert painting.forecast.strip_sha256 == sha256(painting.strip_png)


def test_the_strip_is_the_curve_painted_as_a_png(painting: Painting) -> None:
    expected = render_strip([point.model_dump() for point in painting.forecast.curve])
    with Image.open(io.BytesIO(painting.strip_png)) as image:
        assert image.format == "PNG" and image.mode == "RGB"
        assert image.size == expected.size == (464, 168)
        assert list(image.getdata()) == list(expected.getdata())


def test_the_strip_carries_no_metadata_at_all(painting: Painting) -> None:
    assert image_issues(painting.strip_png, ".png") == []
    assert set(image_factory.png_chunk_types(painting.strip_png)) <= {b"IHDR", b"IDAT", b"IEND"}
    with Image.open(io.BytesIO(painting.strip_png)) as image:
        assert image.info == {}
        assert image.getexif() == {}


# --- privacy ------------------------------------------------------------------------------


def test_the_forecast_json_and_the_strip_pass_the_privacy_scanner(
    painting: Painting, tmp_path: Path
) -> None:
    (tmp_path / "forecast.json").write_text(painting.forecast.model_dump_json(indent=2))
    (tmp_path / "forecast.canonical.json").write_bytes(canonical_bytes(painting.forecast))
    (tmp_path / "strip.png").write_bytes(painting.strip_png)
    result = scan([tmp_path], root=tmp_path)
    assert [finding.format() for finding in result.findings] == []
    assert result.files_scanned == 3


def test_the_forecast_json_has_no_time_of_day(painting: Painting) -> None:
    text = canonical_bytes(painting.forecast).decode()
    assert text_hits(text) == []
    assert re.search(r"\d{1,2}:\d{2}", text) is None
    assert re.search(r"\d{4}-\d{2}-\d{2}T", text) is None  # no ISO datetime either


# --- determinism and reuse ----------------------------------------------------------------


def test_painting_the_same_evening_twice_gives_the_same_bytes(
    painting: Painting, tmp_path: Path
) -> None:
    again = paint(
        support.EVENING,
        transport=support.transport(),
        now_utc=support.before_sunset(),
        cache_dir=tmp_path,
        cycle_hour=support.CYCLE_HOUR,
        forecast_hour=support.FORECAST_HOUR,
        code_sha=support.CODE_SHA,
    )
    assert canonical_bytes(again.forecast) == canonical_bytes(painting.forecast)
    assert again.strip_png == painting.strip_png


def test_a_cached_download_is_not_fetched_again(tmp_path: Path) -> None:
    transport = support.transport()
    for _ in range(2):
        paint(
            support.EVENING,
            transport=transport,
            now_utc=support.before_sunset(),
            cache_dir=tmp_path,
            cycle_hour=support.CYCLE_HOUR,
            forecast_hour=support.FORECAST_HOUR,
            code_sha=support.CODE_SHA,
        )
    assert len(transport.ranged_gets()) == len(hrrr.DEFAULT_FIELDS)


def test_missing_weather_data_fails_loudly_rather_than_painting_from_nothing(
    tmp_path: Path,
) -> None:
    from lastlight.inputs.http import FetchError

    empty = hrrr_support.FakeTransport({})
    with pytest.raises((hrrr.HrrrError, FetchError)):
        paint(
            support.EVENING,
            transport=empty,
            now_utc=support.before_sunset(),
            cache_dir=tmp_path,
            cycle_hour=support.CYCLE_HOUR,
            forecast_hour=support.FORECAST_HOUR,
            code_sha=support.CODE_SHA,
        )


# --- which code made it -------------------------------------------------------------------

SHA = "ab" * 20


def test_the_code_sha_comes_from_the_environment_in_ci() -> None:
    assert current_code_sha(environ={"GITHUB_SHA": SHA}) == SHA


def test_an_explicit_override_beats_the_ci_variable() -> None:
    assert current_code_sha(environ={"LASTLIGHT_CODE_SHA": "c" * 40, "GITHUB_SHA": SHA}) == "c" * 40


def test_the_code_sha_falls_back_to_the_checkout_and_then_to_unknown(tmp_path: Path) -> None:
    sha = current_code_sha(environ={})
    assert sha == "unknown" or (len(sha) == 40 and set(sha) <= set("0123456789abcdef"))
    assert current_code_sha(environ={}, cwd=tmp_path) == "unknown"
