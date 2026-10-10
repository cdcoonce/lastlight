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


def webp_with_chunk(chunk_type: bytes, payload: bytes = b"\x00" * 6) -> bytes:
    """The plain WebP with one extra RIFF chunk appended (and the RIFF size fixed up)."""
    webp = plain_webp()
    extra = chunk_type + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        extra += b"\x00"
    body = webp[8:] + extra
    return b"RIFF" + struct.pack("<I", len(body)) + body
