"""The per-camera `contrast` table: tier A's bands, defined in pixel rows above the frozen skyline.

The shipped numbers are pinned (they are a frozen observation definition, and a refit must not
move them silently); the loader's refusals are tried one at a time against a valid entry.
"""

from __future__ import annotations

import pytest

from lastlight.cameras import ContrastBands, RegistryError, load_registry, parse_registry

CONTRAST = """
[camera.contrast]
skyline_margin_rows = 8
bands_rows = [[8, 28], [28, 48], [48, 68]]
reference_rows = [78, 98]
light_dome_columns = [[500, 800]]
"""

BASE = """
[[camera]]
id = "TEST1"
operator = "Test Operator"
page_url = "https://example.test/site"
image_url = "https://example.test/image.jpg"
view_azimuth_deg = 270
attribution = "Image source: Test Operator"

[camera.position]
lat = 33.45
lon = -112.07
elev_m = 340
"""

VALID = BASE + CONTRAST


def _contrast(**changes: str) -> str:
    """`VALID` with the named lines of the contrast table replaced, for example
    `_contrast(bands_rows="[[8, 28], [20, 40]]")`."""
    text = VALID
    for key, value in changes.items():
        lines = [line for line in CONTRAST.splitlines() if line.startswith(f"{key} = ")]
        assert lines, f"no such line in the fixture: {key}"
        text = text.replace(lines[0], f"{key} = {value}")
    return text


# --- the shipped table ---------------------------------------------------------------------------


def test_whtm1_ships_the_conductors_row_bands() -> None:
    contrast = load_registry()["WHTM1"].contrast
    assert isinstance(contrast, ContrastBands)
    assert contrast.skyline_margin_rows == 8
    assert contrast.bands_rows == ((8, 28), (28, 48), (48, 68), (68, 88), (88, 108))
    assert contrast.reference_rows == (118, 138)


def test_whtm1_light_dome_is_a_column_range_on_the_right_hand_edge() -> None:
    contrast = load_registry()["WHTM1"].contrast
    assert contrast is not None
    assert contrast.light_dome_columns, "the orange north-west edge must be seeded"
    for low, high in contrast.light_dome_columns:
        assert 400 <= low < high <= 800  # right half of the 800 column frame, never past it
    assert contrast.light_dome_columns[-1][1] == 800  # it runs to the right-hand edge


def test_esmo1_has_no_contrast_table_until_it_has_a_frozen_fit() -> None:
    assert load_registry()["ESMO1"].contrast is None


def test_a_camera_without_the_table_has_none() -> None:
    assert parse_registry(BASE)["TEST1"].contrast is None


# --- a valid table -------------------------------------------------------------------------------


def test_a_valid_table_parses_into_tuples() -> None:
    contrast = parse_registry(VALID)["TEST1"].contrast
    assert contrast == ContrastBands(
        skyline_margin_rows=8,
        bands_rows=((8, 28), (28, 48), (48, 68)),
        reference_rows=(78, 98),
        light_dome_columns=((500, 800),),
    )


def test_the_light_dome_may_be_omitted_and_then_excludes_nothing() -> None:
    text = "\n".join(
        line for line in VALID.splitlines() if not line.startswith("light_dome_columns")
    )
    assert parse_registry(text)["TEST1"].contrast.light_dome_columns == ()


def test_bands_may_leave_a_gap_above_the_margin() -> None:
    contrast = parse_registry(_contrast(bands_rows="[[10, 20], [30, 40]]"))["TEST1"].contrast
    assert contrast.bands_rows == ((10, 20), (30, 40))


# --- the refusals --------------------------------------------------------------------------------


def test_the_table_must_be_a_table() -> None:
    text = BASE.replace("view_azimuth_deg = 270", "view_azimuth_deg = 270\ncontrast = 3")
    with pytest.raises(RegistryError, match=r"TEST1.*contrast.*table"):
        parse_registry(text)


@pytest.mark.parametrize("key", ["skyline_margin_rows", "bands_rows", "reference_rows"])
def test_a_missing_required_contrast_field_is_rejected_by_name(key: str) -> None:
    text = "\n".join(line for line in VALID.splitlines() if not line.startswith(f"{key} = "))
    with pytest.raises(RegistryError, match=rf"TEST1.*contrast.*'{key}'"):
        parse_registry(text)


def test_an_unknown_contrast_key_is_rejected_so_a_typo_cannot_hide() -> None:
    text = VALID.replace("reference_rows", "reference_row")
    with pytest.raises(RegistryError, match=r"reference_row"):
        parse_registry(text)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"skyline_margin_rows": "-1"}, "skyline_margin_rows"),
        ({"skyline_margin_rows": "8.5"}, "skyline_margin_rows"),
        ({"skyline_margin_rows": "true"}, "skyline_margin_rows"),
        ({"skyline_margin_rows": '"8"'}, "skyline_margin_rows"),
        ({"bands_rows": "[]"}, "bands_rows"),
        ({"bands_rows": "[8, 28]"}, "bands_rows"),  # not a list of pairs
        ({"bands_rows": "[[8, 28, 48]]"}, "bands_rows"),
        ({"bands_rows": "[[8.0, 28]]"}, "bands_rows"),
        ({"bands_rows": "[[28, 8]]"}, "bands_rows"),  # empty: low is not below high
        ({"bands_rows": "[[8, 8]]"}, "bands_rows"),
        ({"bands_rows": "[[-8, 28]]"}, "bands_rows"),
        ({"bands_rows": "[[8, 28], [20, 40]]"}, "overlap"),
        ({"bands_rows": "[[28, 48], [8, 28]]"}, "ascending"),
        ({"bands_rows": "[[4, 28]]"}, "margin"),  # the lowest band starts inside the margin
        ({"reference_rows": "[78]"}, "reference_rows"),
        ({"reference_rows": "[98, 78]"}, "reference_rows"),
        ({"reference_rows": "[20, 40]"}, "overlap"),  # on top of a band
        ({"reference_rows": "[4, 20]"}, "margin"),
        ({"light_dome_columns": "[500, 800]"}, "light_dome_columns"),
        ({"light_dome_columns": "[[800, 500]]"}, "light_dome_columns"),
        ({"light_dome_columns": "[[-1, 100]]"}, "light_dome_columns"),
        ({"light_dome_columns": "[[100, 300], [200, 400]]"}, "light_dome_columns"),
    ],
)
def test_a_malformed_contrast_table_is_rejected_and_names_the_camera_and_field(
    changes: dict[str, str], message: str
) -> None:
    with pytest.raises(RegistryError, match=rf"TEST1.*{message}"):
        parse_registry(_contrast(**changes))
