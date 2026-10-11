"""The tier A camera registry: the shipped entries and the loader's refusals."""

from __future__ import annotations

import re
import tomllib
from importlib import resources

import pytest

from lastlight.cameras import Camera, RegistryError, load_registry, parse_registry

ADEQ_ATTRIBUTION = "Image source: Arizona Department of Environmental Quality, PhoenixVis (phoenixvis.net)"
ADEQ_BASIS = (
    "Public page; robots.txt allows all; ADEQ terms silent on automated use; "
    "owner decision 2026-10-10; removed on request"
)

VALID = """
[[camera]]
id = "TEST1"
operator = "Test Operator"
page_url = "https://example.test/site"
image_url = "https://example.test/image.jpg"
view_azimuth_deg = 270
attribution = "Image source: Test Operator"
publish_thumbnails = false

[camera.position]
lat = 33.45
lon = -112.07
elev_m = 340
"""


def _without(text: str, key: str) -> str:
    return re.sub(rf"^{key} = .*\n", "", text, flags=re.MULTILINE)


# --- the shipped registry ----------------------------------------------------------------------


def test_the_registry_ships_inside_the_package() -> None:
    assert resources.files("lastlight.cameras").joinpath("registry.toml").is_file()


def test_the_shipped_registry_holds_the_two_adeq_cameras_in_order() -> None:
    assert list(load_registry()) == ["WHTM1", "ESMO1"]


def test_whtm1_matches_the_design_contract() -> None:
    camera = load_registry()["WHTM1"]
    assert camera.image_url == "https://phoenixvis.net/images/photos-main/WHTM1.jpg"
    assert camera.page_url == "https://phoenixvis.net/site.aspx?site=whtm1"
    assert camera.view_azimuth_deg == 274.0
    assert "ADEQ" in camera.operator


def test_esmo1_matches_the_design_contract() -> None:
    camera = load_registry()["ESMO1"]
    assert camera.image_url == "https://phoenixvis.net/images/photos-main/ESMO1.jpg"
    assert camera.page_url == "https://phoenixvis.net/site.aspx?site=esmo1"
    assert 202.5 <= camera.view_azimuth_deg <= 247.5  # southwest
    assert "ADEQ" in camera.operator


@pytest.mark.parametrize("camera_id", ["WHTM1", "ESMO1"])
def test_adeq_cameras_carry_the_attribution_and_the_recorded_basis(camera_id: str) -> None:
    camera = load_registry()[camera_id]
    assert camera.attribution == ADEQ_ATTRIBUTION
    assert camera.publish_thumbnails is True
    assert camera.basis == ADEQ_BASIS


@pytest.mark.parametrize("camera_id", ["WHTM1", "ESMO1"])
def test_every_shipped_entry_has_every_required_field(camera_id: str) -> None:
    raw = tomllib.loads(
        resources.files("lastlight.cameras").joinpath("registry.toml").read_text(encoding="utf-8")
    )
    entry = next(item for item in raw["camera"] if item["id"] == camera_id)
    required = {
        "id",
        "operator",
        "page_url",
        "image_url",
        "view_azimuth_deg",
        "attribution",
        "position",
        "publish_thumbnails",
        "basis",
    }
    assert required <= set(entry)
    assert set(entry["position"]) == {"lat", "lon", "elev_m"}


@pytest.mark.parametrize("camera_id", ["WHTM1", "ESMO1"])
def test_shipped_positions_are_inside_the_phoenix_metro(camera_id: str) -> None:
    position = load_registry()[camera_id].position
    assert 33.2 <= position.lat <= 33.8
    assert -112.6 <= position.lon <= -111.8
    assert 0.0 < position.elev_m < 1500.0


def test_the_loader_reads_a_registry_file_from_a_path(tmp_path) -> None:
    path = tmp_path / "registry.toml"
    path.write_text(VALID)
    assert list(load_registry(path)) == ["TEST1"]


# --- the loader's refusals -------------------------------------------------------------------


def test_a_valid_entry_parses_into_a_camera() -> None:
    camera = parse_registry(VALID)["TEST1"]
    assert isinstance(camera, Camera)
    assert camera.operator == "Test Operator"
    assert camera.view_azimuth_deg == 270.0
    assert (camera.position.lat, camera.position.lon, camera.position.elev_m) == (
        33.45,
        -112.07,
        340.0,
    )
    assert camera.publish_thumbnails is False
    assert camera.basis == ""


@pytest.mark.parametrize("basis_line", ['basis = ""', 'basis = "   "', None])
def test_publishing_thumbnails_with_an_empty_basis_is_rejected(basis_line: str | None) -> None:
    text = VALID.replace("publish_thumbnails = false", "publish_thumbnails = true")
    if basis_line is not None:
        text = text.replace("publish_thumbnails = true", f"publish_thumbnails = true\n{basis_line}")
    with pytest.raises(RegistryError, match=r"TEST1.*basis"):
        parse_registry(text)


def test_publishing_thumbnails_with_a_basis_is_accepted() -> None:
    text = VALID.replace(
        "publish_thumbnails = false", 'publish_thumbnails = true\nbasis = "Public page"'
    )
    camera = parse_registry(text)["TEST1"]
    assert camera.publish_thumbnails is True
    assert camera.basis == "Public page"


def test_a_camera_without_a_basis_may_not_publish() -> None:
    camera = parse_registry(_without(VALID, "publish_thumbnails"))["TEST1"]
    assert camera.publish_thumbnails is False


@pytest.mark.parametrize(
    "key", ["id", "operator", "page_url", "image_url", "view_azimuth_deg", "attribution"]
)
def test_a_missing_required_field_is_rejected_by_name(key: str) -> None:
    with pytest.raises(RegistryError, match=rf"missing.*'{key}'"):
        parse_registry(_without(VALID, key))


@pytest.mark.parametrize("key", ["lat", "lon", "elev_m"])
def test_a_missing_position_field_is_rejected_by_name(key: str) -> None:
    with pytest.raises(RegistryError, match=rf"missing.*'{key}'"):
        parse_registry(_without(VALID, key))


def test_a_missing_position_table_is_rejected() -> None:
    text = VALID.split("[camera.position]")[0]
    with pytest.raises(RegistryError, match="position"):
        parse_registry(text)


@pytest.mark.parametrize("key", ["operator", "page_url", "image_url", "attribution"])
def test_an_empty_required_string_is_rejected(key: str) -> None:
    text = re.sub(rf'^{key} = ".*"$', f'{key} = "  "', VALID, flags=re.MULTILINE)
    with pytest.raises(RegistryError, match=key):
        parse_registry(text)


def test_an_unknown_key_is_rejected_so_a_typo_cannot_hide() -> None:
    text = VALID.replace("publish_thumbnails = false", "publish_thumbnail = true")
    with pytest.raises(RegistryError, match="publish_thumbnail"):
        parse_registry(text)


def test_publish_thumbnails_must_be_a_real_boolean() -> None:
    text = VALID.replace("publish_thumbnails = false", 'publish_thumbnails = "true"')
    with pytest.raises(RegistryError, match="publish_thumbnails"):
        parse_registry(text)


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(RegistryError, match="TEST1"):
        parse_registry(VALID + VALID)


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "with space", "", ".hidden"])
def test_an_id_that_could_not_be_a_directory_name_is_rejected(bad_id: str) -> None:
    with pytest.raises(RegistryError, match="id"):
        parse_registry(VALID.replace('id = "TEST1"', f'id = "{bad_id}"'))


@pytest.mark.parametrize("azimuth", ["-1", "360", "400", '"west"'])
def test_a_view_azimuth_outside_zero_to_360_is_rejected(azimuth: str) -> None:
    with pytest.raises(RegistryError, match="view_azimuth_deg"):
        parse_registry(VALID.replace("view_azimuth_deg = 270", f"view_azimuth_deg = {azimuth}"))


@pytest.mark.parametrize(
    ("old", "new"),
    [("lat = 33.45", "lat = 91"), ("lon = -112.07", "lon = -181"), ("lat = 33.45", 'lat = "x"')],
)
def test_a_position_outside_the_globe_is_rejected(old: str, new: str) -> None:
    with pytest.raises(RegistryError, match="position"):
        parse_registry(VALID.replace(old, new))


@pytest.mark.parametrize("url_key", ["page_url", "image_url"])
def test_urls_must_be_http_or_https(url_key: str) -> None:
    text = re.sub(rf'^{url_key} = ".*"$', f'{url_key} = "file:///etc/passwd"', VALID, flags=re.M)
    with pytest.raises(RegistryError, match=url_key):
        parse_registry(text)


def test_an_empty_registry_is_rejected() -> None:
    with pytest.raises(RegistryError, match="no cameras"):
        parse_registry("")


def test_broken_toml_is_a_registry_error() -> None:
    with pytest.raises(RegistryError, match="TOML"):
        parse_registry("[[camera\n")
