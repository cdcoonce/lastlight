"""Tiny image builders for the privacy-scanner tests.

Clean images come from Pillow, so the "clean twin" of every planted leak is a real encoder's
output. Leaks are spliced in at the byte level (PNG chunks, JPEG segments, RIFF chunks, TIFF
tags) so one test file can plant exactly one named leak without depending on which metadata
a given Pillow build is able to write.
"""

from __future__ import annotations

import io
import struct
import zlib

from PIL import Image, PngImagePlugin

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), (10, 20, 30))


def _encode(fmt: str, **options: object) -> bytes:
    buffer = io.BytesIO()
    _image().save(buffer, format=fmt, **options)
    return buffer.getvalue()


def plain_png() -> bytes:
    """A PNG with exactly IHDR, IDAT and IEND."""
    return _encode("PNG")


def plain_jpeg() -> bytes:
    """A JPEG with only the JFIF APP0 header; no EXIF, XMP, comment or profile."""
    return _encode("JPEG")


def plain_webp() -> bytes:
    return _encode("WEBP")


def plain_tiff() -> bytes:
    return _encode("TIFF")


def png_with_text(key: str = "Comment", value: str = "hello", *, kind: str = "tEXt") -> bytes:
    """A PNG written by Pillow with one textual chunk of the given kind (tEXt, zTXt, iTXt)."""
    info = PngImagePlugin.PngInfo()
    if kind == "tEXt":
        info.add_text(key, value)
    elif kind == "zTXt":
        info.add_text(key, value, zip=True)
    elif kind == "iTXt":
        info.add_itxt(key, value, lang="en", tkey=key)
    else:
        raise ValueError(kind)
    return _encode("PNG", pnginfo=info)


def png_chunk_types(png: bytes) -> list[bytes]:
    """The chunk types of a well-formed PNG, in order."""
    types: list[bytes] = []
    position = len(PNG_SIGNATURE)
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        types.append(png[position + 4 : position + 8])
        position += 12 + length
    return types


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload)
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def png_with_chunk(chunk_type: bytes, payload: bytes = b"\x00" * 7) -> bytes:
    """The plain PNG with one extra chunk spliced in just before IEND."""
    png = plain_png()
    iend = png.rindex(b"\x00\x00\x00\x00IEND")
    return png[:iend] + png_chunk(chunk_type, payload) + png[iend:]


IHDR_END = len(PNG_SIGNATURE) + 4 + 4 + 13 + 4
"""Where the first chunk after IHDR starts: signature, then IHDR (length, type, 13 bytes, CRC)."""

COLOUR_AND_DENSITY_CHUNKS = {
    b"iCCP": b"lastlight test profile\x00\x00" + zlib.compress(bytes(128)),
    b"sRGB": b"\x00",
    b"gAMA": struct.pack(">I", 45455),
    b"pHYs": struct.pack(">IIB", 11811, 11811, 1),
    b"cHRM": struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000),
}
"""A valid payload for each colour-profile, gamma, chromaticity and density chunk of a PNG.

These are what a plain Pillow save can carry over from an input image, unlike the textual and
time chunks, which only a deliberate writer adds.
"""


def png_with_chunks_after_ihdr(png: bytes, chunks: dict[bytes, bytes]) -> bytes:
    """`png` with the given chunks spliced in right after IHDR, before any PLTE or IDAT.

    That is where the colour-space and density chunks belong, so the result is a valid PNG
    whether or not `png` is paletted.
    """
    assert png.startswith(PNG_SIGNATURE)
    spliced = b"".join(png_chunk(chunk_type, payload) for chunk_type, payload in chunks.items())
    return png[:IHDR_END] + spliced + png[IHDR_END:]


def png_colour_type(png: bytes) -> int:
    """The colour type byte of IHDR: 2 is RGB, 3 is paletted."""
    return png[len(PNG_SIGNATURE) + 4 + 4 + 9]


def tiff_bytes(*, gps: bool, byte_order: bytes = b"II") -> bytes:
    """A minimal TIFF structure. With `gps`, IFD0 carries the GPSInfo pointer (tag 0x8825)."""
    e = "<" if byte_order == b"II" else ">"
    header = byte_order + struct.pack(e + "HI", 42, 8)
    tag = 0x8825 if gps else 0x0100  # GPSInfo, or ImageWidth
    gps_ifd_offset = 8 + 2 + 12 + 4
    if gps:
        entry = struct.pack(e + "HHII", tag, 4, 1, gps_ifd_offset)
    else:
        entry = struct.pack(e + "HHI", tag, 3, 1) + struct.pack(e + "H", 8) + b"\x00\x00"
    ifd0 = struct.pack(e + "H", 1) + entry + struct.pack(e + "I", 0)
    gps_ifd = struct.pack(e + "H", 0) + struct.pack(e + "I", 0)
    return header + ifd0 + (gps_ifd if gps else b"")


def tiff_with_tags(tags: list[int], *, byte_order: bytes = b"II") -> bytes:
    """A minimal TIFF whose IFD0 holds one entry per tag, in the order given.

    GPSInfo (0x8825) points at an empty GPS IFD; every other tag is a SHORT with value 8.
    """
    e = "<" if byte_order == b"II" else ">"
    header = byte_order + struct.pack(e + "HI", 42, 8)
    gps_ifd_offset = 8 + 2 + 12 * len(tags) + 4
    entries = b""
    for tag in tags:
        if tag == 0x8825:
            entries += struct.pack(e + "HHII", tag, 4, 1, gps_ifd_offset)
        else:
            entries += struct.pack(e + "HHI", tag, 3, 1) + struct.pack(e + "H", 8) + b"\x00\x00"
    ifd0 = struct.pack(e + "H", len(tags)) + entries + struct.pack(e + "I", 0)
    gps_ifd = struct.pack(e + "H", 0) + struct.pack(e + "I", 0)
    return header + ifd0 + (gps_ifd if 0x8825 in tags else b"")


def bigtiff_bytes(*, byte_order: bytes = b"II") -> bytes:
    """A minimal BigTIFF (version 43, 8-byte offsets) with one empty IFD."""
    e = "<" if byte_order == b"II" else ">"
    header = byte_order + struct.pack(e + "HHHQ", 43, 8, 0, 16)
    return header + struct.pack(e + "QQ", 0, 0)


def png_with_exif(*, gps: bool) -> bytes:
    return png_with_chunk(b"eXIf", tiff_bytes(gps=gps))


def jpeg_with_segment(marker: int, payload: bytes) -> bytes:
    """The plain JPEG with one extra segment (marker byte after 0xFF) right after SOI."""
    jpeg = plain_jpeg()
    segment = b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:2] + segment + jpeg[2:]


def jpeg_with_exif(*, gps: bool) -> bytes:
    return jpeg_with_segment(0xE1, b"Exif\x00\x00" + tiff_bytes(gps=gps))


def jpeg_with_xmp(*, gps: bool) -> bytes:
    body = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"'
    body += b" exif:GPSLatitude='1'/>" if gps else b"/>"
    return jpeg_with_segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + body)


def webp_with_chunks(*chunks: tuple[bytes, bytes]) -> bytes:
    """The plain WebP with extra RIFF chunks appended in order (RIFF size and padding fixed up)."""
    extra = b""
    for chunk_type, payload in chunks:
        extra += chunk_type + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2:
            extra += b"\x00"
    body = plain_webp()[8:] + extra
    return b"RIFF" + struct.pack("<I", len(body)) + body


def webp_with_chunk(chunk_type: bytes, payload: bytes = b"\x00" * 6) -> bytes:
    """The plain WebP with one extra RIFF chunk appended (and the RIFF size fixed up)."""
    return webp_with_chunks((chunk_type, payload))
