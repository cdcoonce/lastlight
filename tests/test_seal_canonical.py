"""Canonical forecast JSON and its SHA-256 digest (issue #5)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

import pytest
import seal_helpers as helpers

from lastlight.forecast import Forecast
from lastlight.seal.canonical import canonical_bytes, digest_hex


def canonical_of(data: dict[str, Any]) -> bytes:
    return canonical_bytes(Forecast.model_validate(data))


def reordered(value: Any) -> Any:
    """The same JSON value with every object's keys in reverse order."""
    if isinstance(value, dict):
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reordered(item) for item in value]
    return value


def with_value(path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(helpers.stub_dict())
    target: Any = data
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return data


# --- the golden form ----------------------------------------------------------------------


def test_the_stub_forecast_has_exactly_the_hand_written_canonical_bytes() -> None:
    assert canonical_of(helpers.stub_dict()) == helpers.STUB_CANONICAL


def test_the_digest_is_the_lowercase_hex_sha256_of_the_canonical_bytes() -> None:
    expected = hashlib.sha256(helpers.STUB_CANONICAL).hexdigest()
    assert digest_hex(helpers.STUB_CANONICAL) == expected == helpers.STUB_DIGEST
    assert re.fullmatch(r"[0-9a-f]{64}", expected)


def test_canonical_json_is_compact_sorted_utf8() -> None:
    raw = helpers.STUB_CANONICAL.decode("utf-8")
    assert " " not in raw.replace("stub forecast for the seal tests", "")
    assert "\n" not in raw
    parsed = json.loads(raw)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == raw


def test_non_ascii_text_is_written_as_utf8_not_escaped() -> None:
    data = with_value(("notes",), "café")
    assert b"caf\xc3\xa9" in canonical_of(data)
    assert b"\\u" not in canonical_of(data)


# --- stability ----------------------------------------------------------------------------


def test_key_order_does_not_change_the_digest() -> None:
    shuffled = reordered(helpers.stub_dict())
    assert list(shuffled) != list(helpers.stub_dict())
    assert digest_hex(canonical_of(shuffled)) == helpers.STUB_DIGEST


def test_the_list_order_of_curve_points_does_change_the_digest() -> None:
    data = helpers.stub_dict()
    data["curve"].reverse()
    assert digest_hex(canonical_of(data)) != helpers.STUB_DIGEST


def test_float_noise_beyond_six_significant_digits_does_not_change_the_digest() -> None:
    data = helpers.stub_dict()
    for point in data["curve"]:
        for key in ("L", "a", "b", "sun_elevation", "view_elevation"):
            point[key] = point[key] * (1 + 3e-10)
    data["vividness"]["mean"] *= 1 - 2e-9
    assert digest_hex(canonical_of(data)) == helpers.STUB_DIGEST


def test_a_change_in_the_sixth_significant_digit_changes_the_digest() -> None:
    data = with_value(("curve", 0, "L"), 78.1236)
    assert digest_hex(canonical_of(data)) != helpers.STUB_DIGEST


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (123456.789, b"123457.0"),
        (0.000123456789, b"0.000123457"),
        (1234567.0, b"1234570.0"),
        (-9.87654321, b"-9.87654"),
        (0.1 + 0.2, b"0.3"),
        (1e-9 * 1.23456789, b"1.23457e-09"),
    ],
)
def test_floats_are_rounded_to_six_significant_digits(value: float, text: bytes) -> None:
    out = canonical_of(with_value(("curve", 0, "a"), value))
    assert b'"a":' + text + b"," in out


def test_negative_zero_is_written_as_zero() -> None:
    out = canonical_of(with_value(("curve", 0, "a"), -0.0))
    assert b'"a":0.0,' in out


def test_integers_stay_integers_and_floats_stay_floats() -> None:
    out = canonical_of(with_value(("reference_point", "elev_m"), 340))
    assert b'"elev_m":340.0' in out
    assert b'"cycle_hour_utc":18,' in out
    assert b'"schema_version":1,' in out


def test_a_forecast_without_a_strip_says_so_with_an_explicit_null() -> None:
    # Absent and null are different statements in a document nobody can edit afterwards, and a
    # forecast always says which one it makes.
    assert b'"strip_sha256":null,' in helpers.STUB_CANONICAL
    assert b'"strip_sha256":null,' in canonical_of(helpers.stub_dict())


def test_the_strip_digest_is_part_of_the_sealed_bytes() -> None:
    digest = "cd" * 32
    with_strip = canonical_of(with_value(("strip_sha256",), digest))
    assert b'"strip_sha256":"' + digest.encode() + b'",' in with_strip
    assert digest_hex(with_strip) != helpers.STUB_DIGEST
    other = canonical_of(with_value(("strip_sha256",), "cd" * 31 + "ce"))
    assert digest_hex(other) != digest_hex(with_strip)


def test_canonical_bytes_are_a_fixed_point_of_parse_and_canonicalise() -> None:
    again = canonical_bytes(Forecast.model_validate_json(helpers.STUB_CANONICAL))
    assert again == helpers.STUB_CANONICAL


def test_digest_hex_of_nothing_in_particular_is_the_sha256_of_those_bytes() -> None:
    assert digest_hex(b"") == hashlib.sha256(b"").hexdigest()
