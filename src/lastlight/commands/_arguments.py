"""Argument types shared by the commands of the daily forecast (underscore: not a command)."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections.abc import Callable

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def iso_date(text: str) -> dt.date:
    """A calendar date written exactly YYYY-MM-DD (not the compact or ISO-week spellings)."""
    if not _ISO_DATE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a date written YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a real calendar date") from None


def bounded_int(low: int, high: int, what: str) -> Callable[[str], int]:
    """An argparse type for a whole number from `low` to `high`, named `what` in errors."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{what} must be a whole number, got {text!r}"
            ) from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{what} must be {low}..{high}, got {value}")
        return value

    return parse
