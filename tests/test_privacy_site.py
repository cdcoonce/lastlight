"""`lastlight.site`: the public reference point and the private-site sentinel."""

from __future__ import annotations

from lastlight import site


def test_reference_point_is_downtown_phoenix_in_degrees_and_metres() -> None:
    assert site.REFERENCE_POINT == (33.45, -112.07, 340.0)


def test_reference_point_docstring_says_it_is_downtown_phoenix_and_public() -> None:
    # A constant has no __doc__ of its own; the module docstring carries the explanation.
    doc = (site.__doc__ or "").lower()
    assert "downtown phoenix" in doc
    assert "public" in doc
    assert "reference_point" in doc


def test_private_site_sentinel_is_the_documented_key() -> None:
    assert site.PRIVATE_SITE_SENTINEL == "LASTLIGHT_PRIVATE_SITE"
