"""Rule P004: image metadata. Every planted leak has a clean twin from a real encoder."""

from __future__ import annotations

from pathlib import Path

import image_factory as images
import pytest

from lastlight.privacy import Finding, scan


def scan_image(tmp_path: Path, data: bytes, name: str) -> list[Finding]:
    path = tmp_path / name
    path.write_bytes(data)
    return scan([path], root=tmp_path).findings


def rule_ids(findings: list[Finding]) -> list[str]:
    return sorted(finding.rule for finding in findings)


# --- the clean twins really are clean --------------------------------------------------------


def test_the_plain_pillow_png_has_only_the_critical_chunks() -> None:
    assert images.png_chunk_types(images.plain_png()) == [b"IHDR", b"IDAT", b"IEND"]


# --- PNG -------------------------------------------------------------------------------------


def test_a_pillow_png_with_a_text_chunk_fails_and_the_same_image_without_metadata_passes(
    tmp_path: Path,
) -> None:
    leaky = scan_image(tmp_path, images.png_with_text(), "leaky.png")
    clean = scan_image(tmp_path, images.plain_png(), "clean.png")
    assert rule_ids(leaky) == ["P004"]
    assert "tEXt" in leaky[0].excerpt
    assert clean == []


@pytest.mark.parametrize("kind", ["tEXt", "zTXt", "iTXt"])
def test_png_textual_chunks_written_by_pillow_are_flagged(tmp_path: Path, kind: str) -> None:
    png = images.png_with_text(kind=kind)
    assert kind.encode() in images.png_chunk_types(png)
    findings = scan_image(tmp_path, png, "leaky.png")
    assert rule_ids(findings) == ["P004"]
    assert kind in findings[0].excerpt


@pytest.mark.parametrize("chunk", ["tEXt", "zTXt", "iTXt", "eXIf", "tIME", "iCCP"])
def test_each_metadata_chunk_type_is_flagged(tmp_path: Path, chunk: str) -> None:
    findings = scan_image(tmp_path, images.png_with_chunk(chunk.encode()), "leaky.png")
    assert rule_ids(findings) == ["P004"]
    assert findings[0].excerpt.startswith(chunk)


@pytest.mark.parametrize("chunk", ["gAMA", "sRGB", "pHYs", "bKGD"])
def test_harmless_ancillary_chunks_pass(tmp_path: Path, chunk: str) -> None:
    assert scan_image(tmp_path, images.png_with_chunk(chunk.encode()), "ok.png") == []


def test_png_exif_with_gps_tags_is_flagged_and_says_so(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.png_with_exif(gps=True), "gps.png")
    assert rule_ids(findings) == ["P004"]
    assert "GPS" in findings[0].excerpt


def test_png_exif_without_gps_is_still_flagged_but_does_not_claim_gps(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.png_with_exif(gps=False), "exif.png")
    assert rule_ids(findings) == ["P004"]
    assert "GPS" not in findings[0].excerpt


def test_every_metadata_chunk_is_its_own_finding(tmp_path: Path) -> None:
    data = images.png_with_chunk(b"tEXt")
    iend = data.rindex(b"\x00\x00\x00\x00IEND")
    data = data[:iend] + images.png_chunk(b"tIME", b"\x00" * 7) + data[iend:]
    assert rule_ids(scan_image(tmp_path, data, "two.png")) == ["P004", "P004"]


def test_a_png_with_data_after_iend_is_flagged(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.plain_png() + b"hidden payload", "tail.png")
    assert rule_ids(findings) == ["P004"]
    assert "after IEND" in findings[0].excerpt


def test_a_truncated_png_fails_closed(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.plain_png()[:-6], "cut.png")
    assert rule_ids(findings) == ["P004"]
    assert "malformed" in findings[0].excerpt


def test_a_png_suffix_on_something_else_fails_closed(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, b"not an image at all", "fake.png")
    assert rule_ids(findings) == ["P004"]


def test_png_is_recognised_by_content_not_by_suffix(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.png_with_text(), "renamed.dat")
    assert rule_ids(findings) == ["P004"]


# --- JPEG ------------------------------------------------------------------------------------


def test_a_pillow_jpeg_without_metadata_passes(tmp_path: Path) -> None:
    assert scan_image(tmp_path, images.plain_jpeg(), "clean.jpg") == []


def test_jpeg_exif_app1_segment_is_flagged(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.jpeg_with_exif(gps=False), "exif.jpg")
    assert rule_ids(findings) == ["P004"]
    assert "APP1" in findings[0].excerpt
    assert "GPS" not in findings[0].excerpt


def test_jpeg_exif_with_gps_tags_is_flagged_and_says_so(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.jpeg_with_exif(gps=True), "gps.jpg")
    assert rule_ids(findings) == ["P004"]
    assert "GPS" in findings[0].excerpt


@pytest.mark.parametrize("gps", [False, True])
def test_jpeg_xmp_app1_segment_is_flagged(tmp_path: Path, gps: bool) -> None:
    findings = scan_image(tmp_path, images.jpeg_with_xmp(gps=gps), "xmp.jpeg")
    assert rule_ids(findings) == ["P004"]
    assert "APP1" in findings[0].excerpt
    assert ("GPS" in findings[0].excerpt) is gps


@pytest.mark.parametrize(
    ("marker", "label"),
    [(0xFE, "COM"), (0xED, "APP13")],
)
def test_jpeg_comment_and_iptc_segments_are_flagged(
    tmp_path: Path, marker: int, label: str
) -> None:
    findings = scan_image(tmp_path, images.jpeg_with_segment(marker, b"free text"), "m.jpg")
    assert rule_ids(findings) == ["P004"]
    assert label in findings[0].excerpt


def test_jpeg_icc_profile_segment_is_flagged(tmp_path: Path) -> None:
    payload = b"ICC_PROFILE\x00\x01\x01" + b"\x00" * 20
    findings = scan_image(tmp_path, images.jpeg_with_segment(0xE2, payload), "icc.jpg")
    assert rule_ids(findings) == ["P004"]


def test_jpeg_app2_that_is_not_an_icc_profile_passes(tmp_path: Path) -> None:
    findings = scan_image(
        tmp_path, images.jpeg_with_segment(0xE2, b"MPF\x00" + b"\x00" * 8), "m.jpg"
    )
    assert findings == []


def test_a_jpeg_with_data_after_eoi_is_flagged(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.plain_jpeg() + b"hidden payload", "tail.jpg")
    assert rule_ids(findings) == ["P004"]
    assert "after EOI" in findings[0].excerpt


def test_a_truncated_jpeg_fails_closed(tmp_path: Path) -> None:
    findings = scan_image(tmp_path, images.plain_jpeg()[:30], "cut.jpg")
    assert rule_ids(findings) == ["P004"]
    assert "malformed" in findings[0].excerpt


def test_metadata_after_the_scan_data_is_still_found(tmp_path: Path) -> None:
    jpeg = images.plain_jpeg()
    eoi = jpeg.rindex(b"\xff\xd9")
    # A comment segment placed after the entropy-coded data but before EOI.
    comment = b"\xff\xfe\x00\x0ahello!!!"
    findings = scan_image(tmp_path, jpeg[:eoi] + comment + jpeg[eoi:], "late.jpg")
    assert rule_ids(findings) == ["P004"]


# --- WebP and TIFF ---------------------------------------------------------------------------


def test_a_pillow_webp_without_metadata_passes(tmp_path: Path) -> None:
    assert scan_image(tmp_path, images.plain_webp(), "clean.webp") == []


@pytest.mark.parametrize("chunk", [b"EXIF", b"XMP ", b"ICCP"])
def test_webp_metadata_chunks_are_flagged(tmp_path: Path, chunk: bytes) -> None:
    findings = scan_image(tmp_path, images.webp_with_chunk(chunk), "leaky.webp")
    assert rule_ids(findings) == ["P004"]
    assert chunk.decode().strip() in findings[0].excerpt


def test_webp_exif_with_gps_says_so(tmp_path: Path) -> None:
    data = images.webp_with_chunk(b"EXIF", images.tiff_bytes(gps=True))
    findings = scan_image(tmp_path, data, "gps.webp")
    assert rule_ids(findings) == ["P004"]
    assert "GPS" in findings[0].excerpt


@pytest.mark.parametrize("byte_order", [b"II", b"MM"])
def test_a_tiff_with_a_gps_tag_is_flagged_in_either_byte_order(
    tmp_path: Path, byte_order: bytes
) -> None:
    findings = scan_image(tmp_path, images.tiff_bytes(gps=True, byte_order=byte_order), "g.tif")
    assert rule_ids(findings) == ["P004"]
    assert "GPS" in findings[0].excerpt


def test_a_tiff_without_gps_tags_passes(tmp_path: Path) -> None:
    assert scan_image(tmp_path, images.tiff_bytes(gps=False), "plain.tiff") == []
    assert scan_image(tmp_path, images.plain_tiff(), "pillow.tif") == []


# --- formats the scanner cannot read ---------------------------------------------------------


@pytest.mark.parametrize("suffix", [".gif", ".heic", ".avif", ".bmp"])
def test_image_formats_without_a_walker_fail_closed(tmp_path: Path, suffix: str) -> None:
    findings = scan_image(tmp_path, b"\x01\x02 any bytes", "photo" + suffix)
    assert rule_ids(findings) == ["P004"]
    assert "cannot be checked" in findings[0].excerpt


def test_svg_is_text_and_goes_through_the_text_rules_only(tmp_path: Path) -> None:
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><title>sky</title></svg>\n'
    assert scan_image(tmp_path, svg.encode(), "sky.svg") == []


def test_the_finding_names_the_file_and_uses_line_one(tmp_path: Path) -> None:
    (finding,) = scan_image(tmp_path, images.png_with_text(), "leaky.png")
    assert finding.format().startswith("leaky.png:1: P004: ")
