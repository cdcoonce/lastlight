"""The per-camera `contrast` table: tier A's bands, defined in pixel rows above the frozen skyline.

The shipped numbers are pinned (they are a frozen observation definition, and a refit must not
move them silently); the loader's refusals are tried one at a time against a valid entry.
"""

from __future__ import annotations

from importlib import resources

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


def test_whtm1_light_dome_is_marked_provisional_and_the_registry_says_why() -> None:
    contrast = load_registry()["WHTM1"].contrast
    assert contrast is not None
    assert contrast.light_dome_provisional is True  # one overcast frame below -6 degrees

    text = (
        resources.files("lastlight.cameras").joinpath("registry.toml").read_text(encoding="utf-8")
    )
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("light_dome_columns = "))
    block = []
    for line in reversed(lines[:index]):
        if not line.startswith("#"):
            break
        block.append(line)
    comment = " ".join(reversed(block)).lower()
    assert "provisional" in comment and "overcast" in comment and "one" in comment, comment


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


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_the_light_dome_provisional_flag_is_a_boolean_in_the_table(
    value: str, expected: bool
) -> None:
    contrast = parse_registry(VALID + f"light_dome_provisional = {value}\n")["TEST1"].contrast
    assert contrast.light_dome_provisional is expected


def test_the_light_dome_is_not_provisional_unless_the_table_says_so() -> None:
    assert parse_registry(VALID)["TEST1"].contrast.light_dome_provisional is False


@pytest.mark.parametrize("value", ["1", "0", '"true"', '"yes"', "1.0", "[true]"])
def test_a_light_dome_provisional_flag_that_is_not_a_boolean_is_rejected_by_name(
    value: str,
) -> None:
    with pytest.raises(RegistryError, match=r"TEST1.*light_dome_provisional.*true or false"):
        parse_registry(VALID + f"light_dome_provisional = {value}\n")


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


def test_a_misspelt_required_contrast_key_is_refused_for_the_field_it_leaves_missing() -> None:
    # Renaming a required key trips the missing-field check first; this is not the unknown-key
    # check (the next tests), and its message names the field that is now missing.
    text = VALID.replace("reference_rows", "reference_row")
    with pytest.raises(
        RegistryError, match=r"TEST1.*contrast is missing required field 'reference_rows'"
    ):
        parse_registry(text)


@pytest.mark.parametrize(
    ("text", "unknown"),
    [
        # Every required key intact, plus one the loader does not know.
        (VALID + "bogus_key = 3\n", "bogus_key"),
        (VALID + "bands_row = [[8, 28]]\n", "bands_row"),
        # The optional key misspelt: nothing required is missing, so only the unknown-key check
        # can stop it, and without it the light dome would silently be switched off.
        (VALID.replace("light_dome_columns", "light_dome_column"), "light_dome_column"),
        (VALID.replace("light_dome_columns", "light_dome_cols"), "light_dome_cols"),
    ],
    ids=["extra-key", "extra-bands-row", "typo-light-dome-column", "typo-light-dome-cols"],
)
def test_an_unknown_contrast_key_is_rejected_by_name_so_a_typo_cannot_hide(
    text: str, unknown: str
) -> None:
    # The whole list is matched, quoted, so that a longer or shorter sibling name (the real
    # `light_dome_columns` contains `light_dome_column`) cannot satisfy the match.
    with pytest.raises(
        RegistryError, match=rf"TEST1.*contrast has unknown key\(s\) \['{unknown}'\]"
    ):
        parse_registry(text)


def test_the_valid_fixture_has_every_key_the_unknown_key_tests_rely_on() -> None:
    # Guard for the tests above: with nothing added or renamed the table parses, so the refusals
    # there are caused by the added or misspelt key alone.
    assert parse_registry(VALID)["TEST1"].contrast is not None
    assert "light_dome_columns = [[500, 800]]" in VALID


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
