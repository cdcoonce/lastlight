"""The collector's only source of time, behind a two-method interface so tests can fake it."""

from __future__ import annotations

import datetime as dt
import time
from typing import Protocol


class Clock(Protocol):
    """`now` returns a timezone-aware instant; `sleep` blocks (a fake clock just advances)."""

    def now(self) -> dt.datetime: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real wall clock."""

    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
