"""Rule P004: metadata in image files.

Published images must be sky-only and carry no metadata (docs/design.md section 3). A walker
reads each container's structure, not its pixels, and returns one short description per
leak. Anything that cannot be walked, or looks malformed, is reported too: a file the scanner
cannot read is a file it cannot vouch for, so it fails closed.

Walked containers: PNG, JPEG, WebP, and TIFF (for GPS tags only). Other image suffixes are
reported as unreadable. SVG is text and goes through the text rules instead.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PNG_METADATA_CHUNKS = {
    b"tEXt": "tEXt chunk",
    b"zTXt": "zTXt chunk",
    b"iTXt": "iTXt chunk",
    b"eXIf": "eXIf chunk",
    b"tIME": "tIME chunk",
    b"iCCP": "iCCP chunk (embedded ICC profile)",
}

WEBP_METADATA_CHUNKS = {b"EXIF": "EXIF chunk", b"XMP ": "XMP chunk", b"ICCP": "ICCP chunk"}

_GPS_INFO_TAG = 0x8825
_JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".jfif"}
_TIFF_SUFFIXES = {".tif", ".tiff"}
_UNWALKED_SUFFIXES = {
    ".gif": "GIF",
    ".bmp": "BMP",
    ".heic": "HEIC",
    ".heif": "HEIF",
    ".avif": "AVIF",
    ".jxl": "JPEG XL",
    ".ico": "ICO",
    ".psd": "PSD",
    ".jp2": "JPEG 2000",
}


def image_issues(data: bytes, suffix: str) -> list[str] | None:
    """Metadata findings in an image, or None when `data` is not an image at all.

    The container is recognised by its leading bytes, so a renamed file is still walked.
    A file with an image suffix that is not what the suffix says fails closed.
    """
    walkers: dict[str, Callable[[bytes], list[str]]] = {
        "png": _png_issues,
        "jpeg": _jpeg_issues,
        "webp": _webp_issues,
        "tiff": _tiff_issues,
    }
    kind = _sniff(data)
    if kind is not None:
        return walkers[kind](data)
    suffix = suffix.lower()
    if suffix == ".png":
        return ["malformed PNG: bad signature"]
    if suffix in _JPEG_SUFFIXES:
        return ["malformed JPEG: bad signature"]
    if suffix == ".webp":
        return ["malformed WebP: bad signature"]
    if suffix in _TIFF_SUFFIXES:
        return ["malformed TIFF: bad header"]
    if suffix in _UNWALKED_SUFFIXES:
        return [f"{_UNWALKED_SUFFIXES[suffix]} image cannot be checked for metadata"]
    return None


def _sniff(data: bytes) -> str | None:
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    return None


# --- EXIF / TIFF structure ------------------------------------------------------------------


def tiff_has_gps(tiff: bytes) -> bool | None:
    """Whether the TIFF structure's first IFD carries the GPSInfo tag; None if unreadable.

    EXIF is a TIFF structure, so this also reads the payload of a JPEG APP1, a PNG eXIf
    chunk and a WebP EXIF chunk.
    """
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return None
    order = "<" if tiff[:2] == b"II" else ">"
    try:
        magic, ifd_offset = struct.unpack_from(order + "HI", tiff, 2)
        if magic != 42:
            return None
        (count,) = struct.unpack_from(order + "H", tiff, ifd_offset)
        for index in range(count):
            (tag,) = struct.unpack_from(order + "H", tiff, ifd_offset + 2 + 12 * index)
            if tag == _GPS_INFO_TAG:
                return True
    except struct.error:
        return None
    return False


def _exif_note(payload: bytes) -> str:
    return " with GPS tags" if tiff_has_gps(payload) else ""


def _xmp_has_gps(payload: bytes) -> bool:
    return b"GPSLatitude" in payload or b"GPSLongitude" in payload


# --- PNG -----------------------------------------------------------------------------------


def _png_issues(data: bytes) -> list[str]:
    issues: list[str] = []
    position = len(PNG_SIGNATURE)
    while True:
        if position + 8 > len(data):
            issues.append("malformed PNG: ends before IEND")
            return issues
        (length,) = struct.unpack_from(">I", data, position)
        chunk_type = data[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(data):
            issues.append("malformed PNG: truncated chunk")
            return issues
        if chunk_type in PNG_METADATA_CHUNKS:
            note = _exif_note(data[position + 8 : end - 4]) if chunk_type == b"eXIf" else ""
            issues.append(PNG_METADATA_CHUNKS[chunk_type] + note)
        position = end
        if chunk_type == b"IEND":
            if position < len(data):
                issues.append("data after IEND")
            return issues


# --- JPEG ----------------------------------------------------------------------------------

# One or more 0xFF, then a marker byte that is not stuffing (00), a restart (D0-D7) or fill (FF).
_NEXT_MARKER = re.compile(rb"\xff+[\x01-\xcf\xd8-\xfe]")


def _jpeg_segment_issue(marker: int, payload: bytes) -> str | None:
    if marker == 0xE1:
        if payload.startswith(b"Exif\x00\x00"):
            return "APP1 segment (EXIF" + _exif_note(payload[6:]) + ")"
        if payload.startswith(b"http://ns.adobe.com/xap/"):
            return "APP1 segment (XMP" + (" with GPS tags" if _xmp_has_gps(payload) else "") + ")"
        return "APP1 segment (unrecognised)"
    if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
        return "APP2 segment (embedded ICC profile)"
    if marker == 0xED:
        return "APP13 segment (IPTC/Photoshop)"
    if marker == 0xFE:
        return "COM segment (comment)"
    return None


def _jpeg_issues(data: bytes) -> list[str]:
    issues: list[str] = []
    position = 2
    while True:
        while position + 1 < len(data) and data[position] == 0xFF and data[position + 1] == 0xFF:
            position += 1  # fill bytes before a marker
        if position + 2 > len(data) or data[position] != 0xFF:
            issues.append("malformed JPEG: ends before EOI")
            return issues
        marker = data[position + 1]
        position += 2
        if marker == 0xD9:  # EOI
            if position < len(data):
                issues.append("data after EOI")
            return issues
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # markers without a length
            continue
        if position + 2 > len(data):
            issues.append("malformed JPEG: truncated segment")
            return issues
        (length,) = struct.unpack_from(">H", data, position)
        if length < 2 or position + length > len(data):
            issues.append("malformed JPEG: truncated segment")
            return issues
        issue = _jpeg_segment_issue(marker, data[position + 2 : position + length])
        if issue is not None:
            issues.append(issue)
        position += length
        if marker == 0xDA:  # SOS: entropy-coded data follows, up to the next real marker
            following = _NEXT_MARKER.search(data, position)
            if following is None:
                issues.append("malformed JPEG: ends before EOI")
                return issues
            position = following.start()


# --- WebP ----------------------------------------------------------------------------------


def _webp_issues(data: bytes) -> list[str]:
    issues: list[str] = []
    (riff_size,) = struct.unpack_from("<I", data, 4)
    declared_end = 8 + riff_size
    if declared_end > len(data) or declared_end < 12:
        return ["malformed WebP: RIFF size does not fit the file"]
    position = 12
    while position + 8 <= declared_end:
        chunk_id = data[position : position + 4]
        (size,) = struct.unpack_from("<I", data, position + 4)
        payload_end = position + 8 + size
        if payload_end > declared_end:
            issues.append("malformed WebP: truncated chunk")
            return issues
        if chunk_id in WEBP_METADATA_CHUNKS:
            payload = data[position + 8 : payload_end]
            if chunk_id == b"EXIF":
                note = _exif_note(payload.removeprefix(b"Exif\x00\x00"))
            elif chunk_id == b"XMP ":
                note = " with GPS tags" if _xmp_has_gps(payload) else ""
            else:
                note = ""
            issues.append(WEBP_METADATA_CHUNKS[chunk_id] + note)
        position = payload_end + (size % 2)
    if position < declared_end:  # too short for a chunk header: bytes with no name
        issues.append("malformed WebP: bytes after the last chunk")
    if declared_end < len(data):
        issues.append("data after RIFF")
    return issues


# --- TIFF ----------------------------------------------------------------------------------


def _tiff_issues(data: bytes) -> list[str]:
    if data[2:3] == b"+" or data[3:4] == b"+":
        return ["BigTIFF image cannot be checked for metadata"]
    gps = tiff_has_gps(data)
    if gps is None:
        return ["malformed TIFF: unreadable first IFD"]
    return ["TIFF GPS tags"] if gps else []
