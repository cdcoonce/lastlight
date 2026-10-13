"""The check's own store of per-evening inputs: one JSON file per evening under `renderer-v1/`.

For each considered evening the report needs three numbers from the archives (`aod_550`,
`ozone_du` and the mean total cloud cover along the sightline) or the reason the evening was
dropped. They are collected once, politely, by `lastlight.renderer_check.fetch`, and kept here so
the report stage reads no archive and no network. The store is cache, outside git, and it is not
G0's: `<cache-dir>/g0/records/` is never written by this command.

`inputs` holds which archive runs were read (private provenance), as in G0's records; nothing in
a record is published as it is.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from lastlight.g0.record import DROP_REASONS

SCHEMA_VERSION: Final = 1

INPUT_DROP_REASONS: Final[dict[str, str]] = {
    **{code: text for code, text in DROP_REASONS.items() if code != "not_collected"},
    "tcdc_missing": "A point of the sightline has no total cloud cover value in the analysis.",
}
"""Why an evening may be dropped for want of inputs: G0's reasons (the archives it reads) and
`tcdc_missing`. Codes are stable; the report prints the sentence."""

_FILE_NAME: Final = re.compile(r"(\d{4}-\d{2}-\d{2})\.json")


class InputRecordError(ValueError):
    """A record is malformed, inconsistent or from another schema. The message says how."""


def _finite(day: dt.date, name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputRecordError(f"{day}: {name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise InputRecordError(f"{day}: {name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class EveningInputs:
    """An evening's inputs, or its reason for being dropped (`drop_reason` None means inputs).

    `cloud_mean_pct` is the arithmetic mean of HRRR's total cloud cover over `cloud_points`
    points of the sightline; whether the evening is clear enough to count is decided at report
    time, from this number.
    """

    date: dt.date
    drop_reason: str | None
    drop_detail: str
    aod_550: float | None
    ozone_du: float | None
    cloud_mean_pct: float | None
    cloud_points: int | None
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.aod_550, self.ozone_du, self.cloud_mean_pct, self.cloud_points)
        if self.drop_reason is None:
            if any(value is None for value in values):
                raise InputRecordError(f"{self.date}: an evening with inputs needs all of them")
            aod = _finite(self.date, "aod_550", self.aod_550)
            ozone = _finite(self.date, "ozone_du", self.ozone_du)
            cloud = _finite(self.date, "cloud_mean_pct", self.cloud_mean_pct)
            if aod < 0.0 or ozone <= 0.0:
                raise InputRecordError(f"{self.date}: aod_550 and ozone_du must be positive")
            if not 0.0 <= cloud <= 100.0:
                raise InputRecordError(f"{self.date}: cloud_mean_pct {cloud} is not a percentage")
            if isinstance(self.cloud_points, bool) or not isinstance(self.cloud_points, int):
                raise InputRecordError(f"{self.date}: cloud_points must be a whole number")
        else:
            if self.drop_reason not in INPUT_DROP_REASONS:
                raise InputRecordError(
                    f"{self.date}: unknown drop reason {self.drop_reason!r}; "
                    f"known: {list(INPUT_DROP_REASONS)}"
                )
            if any(value is not None for value in values):
                raise InputRecordError(f"{self.date}: a dropped evening must not hold inputs")

    @classmethod
    def rendered(
        cls,
        date: dt.date,
        *,
        aod_550: float,
        ozone_du: float,
        cloud_mean_pct: float,
        cloud_points: int = 17,
        inputs: dict[str, Any] | None = None,
    ) -> EveningInputs:
        """An evening whose inputs were collected."""
        return cls(
            date, None, "", aod_550, ozone_du, cloud_mean_pct, cloud_points, dict(inputs or {})
        )

    @classmethod
    def dropped(
        cls, date: dt.date, reason: str, detail: str = "", inputs: dict[str, Any] | None = None
    ) -> EveningInputs:
        """An evening that cannot be built, with the reason from `INPUT_DROP_REASONS`."""
        return cls(date, reason, detail, None, None, None, None, dict(inputs or {}))

    @property
    def ok(self) -> bool:
        return self.drop_reason is None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "date": self.date.isoformat(),
            "drop_reason": self.drop_reason,
            "drop_detail": self.drop_detail,
            "aod_550": self.aod_550,
            "ozone_du": self.ozone_du,
            "cloud_mean_pct": self.cloud_mean_pct,
            "cloud_points": self.cloud_points,
            "inputs": self.inputs,
        }

    @classmethod
    def from_json(cls, document: Any) -> EveningInputs:
        if not isinstance(document, dict):
            raise InputRecordError("the record is not a JSON object")
        if document.get("schema_version") != SCHEMA_VERSION:
            raise InputRecordError(
                f"unsupported record schema {document.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        try:
            return cls(
                date=dt.date.fromisoformat(document["date"]),
                drop_reason=document["drop_reason"],
                drop_detail=document["drop_detail"],
                aod_550=document["aod_550"],
                ozone_du=document["ozone_du"],
                cloud_mean_pct=document["cloud_mean_pct"],
                cloud_points=document["cloud_points"],
                inputs=dict(document["inputs"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, InputRecordError):
                raise
            raise InputRecordError(f"the record does not follow the schema: {error!r}") from error


class InputStore:
    """A directory of records, one `YYYY-MM-DD.json` per evening, written atomically."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path(self, day: dt.date) -> Path:
        return self.directory / f"{day.isoformat()}.json"

    def dates(self) -> set[dt.date]:
        """The dates that have a record file (names that are not dates are ignored)."""
        found: set[dt.date] = set()
        if not self.directory.is_dir():
            return found
        for path in self.directory.iterdir():
            match = _FILE_NAME.fullmatch(path.name)
            if match is None:
                continue
            try:
                found.add(dt.date.fromisoformat(match.group(1)))
            except ValueError:
                continue
        return found

    def load(self, day: dt.date) -> EveningInputs | None:
        """The record of `day`, or None if there is none. A damaged file is an error."""
        path = self.path(day)
        if not path.is_file():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            record = EveningInputs.from_json(document)
        except (json.JSONDecodeError, InputRecordError) as error:
            raise InputRecordError(f"{path.name}: {error}") from error
        if record.date != day:
            raise InputRecordError(
                f"{path.name} holds the record of {record.date}, not of {day}; was a file renamed?"
            )
        return record

    def save(self, record: EveningInputs) -> None:
        """Write the record, replacing any earlier one for the date, all or nothing."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(record.date)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(
            json.dumps(record.to_json(), allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, path)
