"""Canonical forecast JSON and its digest.

The canonical form is the one byte string every honest party can reproduce from the same
forecast: UTF-8, object keys sorted, no whitespace between tokens, and every float rounded to
six significant digits before it is written. The rounding makes the digest insensitive to
float noise beyond what the forecast means (a value computed as 0.1 + 0.2 and one computed as
0.3 are the same number here). Floats are written the way Python's `repr` writes them, and
negative zero is written as zero. Integers stay integers, so `18` is never `18.0`. The order of
list items is part of the forecast and is kept.

The digest is the lowercase hex SHA-256 of those bytes. It is what the ledger index records
and what the RFC 3161 timestamp covers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from lastlight.forecast import Forecast

SIGNIFICANT_DIGITS = 6


def _round(value: Any) -> Any:
    if isinstance(value, float):
        rounded = float(f"{value:.{SIGNIFICANT_DIGITS}g}")
        return 0.0 if rounded == 0.0 else rounded  # folds -0.0 into 0.0
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def canonical_bytes(forecast: Forecast) -> bytes:
    """The canonical JSON of `forecast` as UTF-8 bytes."""
    document = _round(forecast.model_dump(mode="json"))
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def digest_hex(data: bytes) -> str:
    """Lowercase hex SHA-256 of `data`."""
    return hashlib.sha256(data).hexdigest()
