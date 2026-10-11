"""The forecast schema (issue #5): shape, strictness, and no time of day anywhere."""

from __future__ import annotations

import copy
import datetime as dt
import json
from typing import Any

import pytest
import seal_helpers as helpers
from pydantic import ValidationError

from lastlight.forecast import Forecast

TOP_LEVEL_FIELDS = {
    "schema_version",
    "date",
    "model",
    "reference_point",
    "inputs",
    "curve",
    "vividness",
    "notes",
    "strip_sha256",
}


def make(**changes: Any) -> Forecast:
    return Forecast.model_validate({**helpers.stub_dict(), **changes})


def with_path(path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(helpers.stub_dict())
    target: Any = data
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return data


def error_locations(error: ValidationError) -> list[tuple[str | int, ...]]:
    return [issue["loc"] for issue in error.errors(include_input=False)]


# --- shape --------------------------------------------------------------------------------


def test_the_stub_forecast_validates_with_typed_fields() -> None:
    forecast = helpers.stub_forecast()
    assert forecast.schema_version == 1
    assert forecast.date == dt.date(2026, 10, 11)
    assert forecast.model.name == "lastlight-stub"
    assert forecast.reference_point.elev_m == 340.0
    assert forecast.inputs[0].cycle_date == dt.date(2026, 10, 10)
    assert forecast.inputs[0].cycle_hour_utc == 18
    assert pytest.approx(78.1234567) == forecast.curve[0].L
    assert forecast.vividness.sd == 0.07


def test_the_json_schema_pins_the_field_names() -> None:
    schema = Forecast.model_json_schema()
    assert set(schema["properties"]) == TOP_LEVEL_FIELDS
    assert set(schema["required"]) == TOP_LEVEL_FIELDS
    defs = schema["$defs"]
    assert set(defs["CurvePoint"]["properties"]) == {
        "sun_elevation",
        "view_elevation",
        "L",
        "a",
        "b",
    }
    assert set(defs["Input"]["properties"]) == {
        "source",
        "cycle_date",
        "cycle_hour_utc",
        "forecast_hour",
        "digest",
    }
    assert set(defs["Model"]["properties"]) == {"name", "version", "code_sha"}
    assert set(defs["ReferencePoint"]["properties"]) == {"lat", "lon", "elev_m"}
    assert set(defs["Vividness"]["properties"]) == {"mean", "sd"}


def test_no_field_can_hold_a_time_of_day() -> None:
    schema_text = json.dumps(Forecast.model_json_schema())
    assert "date-time" not in schema_text
    assert '"format": "time"' not in schema_text


def test_a_forecast_is_immutable() -> None:
    forecast = helpers.stub_forecast()
    with pytest.raises(ValidationError):
        forecast.notes = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "path",
    [
        ("surprise",),
        ("model", "surprise"),
        ("reference_point", "surprise"),
        ("inputs", 0, "surprise"),
        ("curve", 0, "surprise"),
        ("vividness", "surprise"),
    ],
)
def test_unknown_fields_are_rejected_at_every_level(path: tuple[str | int, ...]) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(path, 1))
    assert error_locations(caught.value) == [path]


@pytest.mark.parametrize("field", sorted(TOP_LEVEL_FIELDS))
def test_every_top_level_field_is_required(field: str) -> None:
    data = helpers.stub_dict()
    del data[field]
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(data)
    assert error_locations(caught.value) == [(field,)]


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_only_schema_version_one_is_accepted(version: object) -> None:
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(("schema_version",), version))


# --- the painting's digest ----------------------------------------------------------------

STRIP_DIGEST = "ab" * 32


def test_the_stub_declares_no_strip() -> None:
    assert helpers.stub_forecast().strip_sha256 is None


def test_a_forecast_can_declare_the_sha256_of_its_strip() -> None:
    forecast = make(strip_sha256=STRIP_DIGEST)
    assert forecast.strip_sha256 == STRIP_DIGEST


def test_strip_sha256_is_required_but_may_be_null() -> None:
    # Required: a code path that forgets the painting must fail, not seal a forecast without one.
    data = helpers.stub_dict()
    del data["strip_sha256"]
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(data)
    assert error_locations(caught.value) == [("strip_sha256",)]
    assert make(strip_sha256=None).strip_sha256 is None


@pytest.mark.parametrize(
    "bad",
    [
        STRIP_DIGEST.upper(),
        STRIP_DIGEST[:-1],
        STRIP_DIGEST + "0",
        STRIP_DIGEST + "\n",
        " " + STRIP_DIGEST,
        "g" * 64,
        "",
        0,
        True,
        b"ab" * 32,
        ["ab" * 32],
    ],
)
def test_strip_sha256_is_64_lowercase_hex_characters_or_null(bad: object) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(("strip_sha256",), bad))
    assert error_locations(caught.value) == [("strip_sha256",)]


def test_the_json_schema_marks_strip_sha256_as_a_nullable_string() -> None:
    schema = Forecast.model_json_schema()["properties"]["strip_sha256"]
    kinds = {option.get("type") for option in schema["anyOf"]}
    assert kinds == {"string", "null"}


# --- dates --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"2026-10-11T{helpers.MIDNIGHT}",
        f"2026-10-11T{helpers.MIDNIGHT}Z",
        f"2026-10-11 {helpers.clock('00', '00')}",
        "20261011",
        "10/11/2026",
        "2026-13-01",
        "2026-02-30",
        "",
    ],
)
def test_the_date_must_be_a_plain_iso_calendar_date(text: str) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(("date",), text))
    assert error_locations(caught.value) == [("date",)]


@pytest.mark.parametrize("path", [("date",), ("inputs", 0, "cycle_date")])
def test_a_datetime_value_is_not_a_date(path: tuple[str | int, ...]) -> None:
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(path, dt.datetime(2026, 10, 11, tzinfo=dt.UTC)))


@pytest.mark.parametrize("hour", [-1, 24, 1.5, True, "18"])
def test_cycle_hour_must_be_a_whole_hour_of_the_day(hour: object) -> None:
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(("inputs", 0, "cycle_hour_utc"), hour))


@pytest.mark.parametrize("hour", [-1, 1.5, True])
def test_forecast_hour_must_be_a_non_negative_whole_number(hour: object) -> None:
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(("inputs", 0, "forecast_hour"), hour))


# --- numbers ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ("curve", 0, "L"),
        ("curve", 0, "a"),
        ("curve", 0, "b"),
        ("curve", 0, "sun_elevation"),
        ("curve", 0, "view_elevation"),
        ("vividness", "mean"),
        ("vividness", "sd"),
    ],
)
@pytest.mark.parametrize("bad", ["1.5", True, None])
def test_numbers_must_be_real_numbers_not_strings_booleans_or_null(
    path: tuple[str | int, ...], bad: object
) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(path, bad))
    assert error_locations(caught.value) == [path]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ['"mean": 0.4200000001', '"sd": 0.07', '"b": -2.5e-07'])
def test_non_finite_json_tokens_are_rejected(token: str, field: str) -> None:
    name = field.split(":")[0]
    text = helpers.stub_text().replace(field, f"{name}: {token}")
    assert token in text
    with pytest.raises(ValidationError):
        Forecast.model_validate_json(text)


def test_integers_are_accepted_where_a_float_is_expected() -> None:
    forecast = Forecast.model_validate(with_path(("reference_point", "elev_m"), 340))
    assert forecast.reference_point.elev_m == 340.0


def test_the_spread_of_a_vividness_distribution_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(("vividness", "sd"), -0.01))


# --- the reference point ------------------------------------------------------------------


@pytest.mark.parametrize(
    "point",
    [
        {"lat": helpers.LAT + 0.05, "lon": helpers.LON, "elev_m": helpers.ELEVATION},
        {"lat": helpers.LAT, "lon": helpers.LON + 0.07, "elev_m": helpers.ELEVATION},
        {"lat": helpers.LAT, "lon": helpers.LON, "elev_m": helpers.ELEVATION + 1.0},
        {"lat": helpers.LAT, "lon": -helpers.LON, "elev_m": helpers.ELEVATION},
    ],
)
def test_only_the_public_reference_point_may_be_sealed(point: dict[str, float]) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(("reference_point",), point))
    assert error_locations(caught.value) == [("reference_point",)]


# --- no time of day -----------------------------------------------------------------------

CLOCK_TIMES = [
    helpers.clock(),
    f"{helpers.clock('5')} PM",
    f"2026-10-11T{helpers.clock(second='00')}Z",
    f"2026-10-11 {helpers.clock()}",
    f"{helpers.HOUR}h{helpers.MINUTE}",
]
STRING_FIELDS = [
    ("notes",),
    ("model", "name"),
    ("model", "version"),
    ("model", "code_sha"),
    ("inputs", 0, "source"),
    ("inputs", 0, "digest"),
]


@pytest.mark.parametrize("clock_time", CLOCK_TIMES)
def test_a_time_of_day_in_notes_is_rejected_without_echoing_it(clock_time: str) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(("notes",), f"dust arrives about {clock_time} today"))
    assert error_locations(caught.value) == [("notes",)]
    assert clock_time not in caught.value.errors(include_input=False)[0]["msg"]


@pytest.mark.parametrize("path", STRING_FIELDS)
def test_a_time_of_day_is_rejected_in_every_string_field(path: tuple[str | int, ...]) -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(path, f"run at {helpers.clock()}"))
    assert error_locations(caught.value) == [path]


@pytest.mark.parametrize(
    "text",
    [
        f"{helpers.SUN_EVENT} around 5 PM",  # P002: a sun event pinned to a loose time
        f"camera near {helpers.NEAR_REFERENCE_PAIR}",  # P003: not the reference point itself
    ],
)
def test_text_the_privacy_scanner_would_flag_is_rejected_too(text: str) -> None:
    # The ledger is append-only and `privacy-scan` runs over it, so a sealed leak could never
    # be removed again. Refuse at seal time everything the scanner's text rules would refuse.
    with pytest.raises(ValidationError):
        Forecast.model_validate(with_path(("notes",), text))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "dust plume 2 to 3 km deep over the west; ratio 3:2",
        "sun elevation -4.5 degrees, reference point 33.45, -112.07",
        "café ☃",
    ],
)
def test_ordinary_notes_are_accepted(text: str) -> None:
    assert Forecast.model_validate(with_path(("notes",), text)).notes == text


def test_a_string_that_cannot_be_encoded_as_utf8_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        Forecast.model_validate(with_path(("notes",), "lone \ud800 surrogate"))
    assert error_locations(caught.value) == [("notes",)]
    assert "encoded as UTF-8" in caught.value.errors(include_input=False)[0]["msg"]


def test_json_text_with_a_lone_surrogate_escape_is_rejected() -> None:
    text = helpers.stub_text().replace("stub forecast", "lone \\ud800 surrogate")
    assert "\\ud800" in text
    with pytest.raises(ValidationError):
        Forecast.model_validate_json(text)
