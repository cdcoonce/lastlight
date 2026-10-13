"""One date's G0 result, and the cache of them that makes the run resumable.

A `DateRecord` is either rendered (the contrasts of the date at three pixel scales, its absolute
curve, the atmosphere it was rendered from) or dropped (a reason from `DROP_REASONS`, never a
silent gap: `docs/design.md` section 2, rule 5). Records live as one JSON file per date in a cache
directory outside git; the run writes each as soon as it exists, and the final report pass reads
only these files. Nothing in a record is published as it is: the report and results file carry
derived numbers only. `inputs` (which archive runs were read, with their digests) stays in the
cache.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

SCHEMA_VERSION: Final = 1
RENDER_SCALES: Final = ("nominal", "low", "high")
"""The pixel scales a date is rendered at: the fit's nominal one and each end of its range."""

CONTRAST_SHAPE: Final = (29, 5, 2)
CURVE_SHAPE: Final = (29, 7, 3)

DROP_REASONS: Final[dict[str, str]] = {
    "hrrr_run_missing": "The HRRR run for this evening is not in the archive.",
    "hrrr_field_missing": "The HRRR file for this evening lacks a field the study needs.",
    "hrrr_field_empty": "A field the study needs has no value anywhere along the sightline.",
    "hrrr_unreadable": "The HRRR messages for this evening could not be read as asked for.",
    "aerosol_run_missing": "The GEFS-Aerosols run for this evening is not in the archive.",
    "aerosol_unreadable": "The GEFS-Aerosols messages for this evening could not be read.",
    "not_collected": "No record exists for this date: it has not been collected.",
    "v1_outside_table": "The date's atmosphere is outside the table renderer v1 reads.",
}
"""Why a date may be left out, with the sentence the report prints. Codes are stable."""

_FILE_NAME: Final = re.compile(r"(\d{4}-\d{2}-\d{2})\.json")


class RecordError(ValueError):
    """A record is malformed, inconsistent or from another schema. The message says how."""


@dataclass(frozen=True)
class AtmosphereRecord:
    """The atmosphere a date was rendered from, for the results file.

    `aod_total` is the aerosol optical depth at 550 nm that was rendered (the archive's total),
    `aod_dust` its dust share, `aod_smoke` HRRR's smoke mean along the sightline (recorded, not
    added), `pwat_kg_m2` precipitable water (recorded, not rendered) and `lcdc_pct` low cloud
    cover along the sightline (recorded, not rendered).
    """

    aod_total: float
    aod_dust: float
    aod_smoke: float
    ozone_du: float
    pwat_kg_m2: float | None
    lcdc_pct: float


@dataclass(frozen=True, eq=False)
class DateRecord:
    """A date's result. `drop_reason` is None exactly when the date was rendered.

    `contrast[scale]` has shape (29 sweep elevations, 5 bands, 2 channels); `floor_channels[scale]`
    counts the channels that were raised to the floor while rendering it; `curve_lab` is the
    forecast's own absolute curve, shape (29, 7 views, 3) in CIELAB.
    """

    date: dt.date
    drop_reason: str | None
    drop_detail: str
    atmosphere: AtmosphereRecord | None
    contrast: dict[str, np.ndarray]
    floor_channels: dict[str, int]
    curve_lab: np.ndarray | None
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.drop_reason is None:
            self._check_rendered()
        else:
            self._check_dropped()

    @property
    def ok(self) -> bool:
        return self.drop_reason is None

    def _check_dropped(self) -> None:
        if self.drop_reason not in DROP_REASONS:
            raise RecordError(
                f"{self.date}: unknown drop reason {self.drop_reason!r}; "
                f"known: {list(DROP_REASONS)}"
            )
        if self.atmosphere is not None or self.contrast or self.curve_lab is not None:
            raise RecordError(
                f"{self.date}: a record with a drop reason must not hold a rendered curve"
            )

    def _check_rendered(self) -> None:
        if self.atmosphere is None or self.curve_lab is None:
            raise RecordError(f"{self.date}: a rendered date needs its atmosphere and curve")
        if set(self.contrast) != set(RENDER_SCALES) or set(self.floor_channels) != set(
            RENDER_SCALES
        ):
            raise RecordError(
                f"{self.date}: a rendered date needs all the scales {RENDER_SCALES}, "
                f"got {tuple(self.contrast)}"
            )
        for scale, array in self.contrast.items():
            _check_array(self.date, f"contrast[{scale}]", array, CONTRAST_SHAPE)
        _check_array(self.date, "curve_lab", self.curve_lab, CURVE_SHAPE)

    def to_json(self) -> dict[str, Any]:
        atmosphere = self.atmosphere
        return {
            "schema_version": SCHEMA_VERSION,
            "date": self.date.isoformat(),
            "drop_reason": self.drop_reason,
            "drop_detail": self.drop_detail,
            "atmosphere": None
            if atmosphere is None
            else {
                "aod_total": atmosphere.aod_total,
                "aod_dust": atmosphere.aod_dust,
                "aod_smoke": atmosphere.aod_smoke,
                "ozone_du": atmosphere.ozone_du,
                "pwat_kg_m2": atmosphere.pwat_kg_m2,
                "lcdc_pct": atmosphere.lcdc_pct,
            },
            "contrast": {scale: array.tolist() for scale, array in self.contrast.items()},
            "floor_channels": dict(self.floor_channels),
            "curve_lab": None if self.curve_lab is None else self.curve_lab.tolist(),
            "inputs": self.inputs,
        }

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> DateRecord:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise RecordError(
                f"unsupported record schema {document.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION}"
            )
        try:
            raw = document["atmosphere"]
            return cls(
                date=dt.date.fromisoformat(document["date"]),
                drop_reason=document["drop_reason"],
                drop_detail=document["drop_detail"],
                atmosphere=None if raw is None else AtmosphereRecord(**raw),
                contrast={
                    s: np.array(document["contrast"][s], dtype=float)
                    for s in RENDER_SCALES
                    if s in document["contrast"]
                },
                floor_channels={
                    s: int(document["floor_channels"][s])
                    for s in RENDER_SCALES
                    if s in document["floor_channels"]
                },
                curve_lab=(
                    None
                    if document["curve_lab"] is None
                    else np.array(document["curve_lab"], dtype=float)
                ),
                inputs=dict(document["inputs"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, RecordError):
                raise
            raise RecordError(f"the record does not follow the schema: {error!r}") from error


def _check_array(day: dt.date, name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise RecordError(f"{day}: {name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise RecordError(f"{day}: {name} holds values that are not finite")


class RecordStore:
    """A directory of records, one `YYYY-MM-DD.json` per date, written atomically."""

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

    def load(self, day: dt.date) -> DateRecord | None:
        """The record of `day`, or None if there is none. A damaged file is an error."""
        path = self.path(day)
        if not path.is_file():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            record = DateRecord.from_json(document)
        except (json.JSONDecodeError, RecordError) as error:
            raise RecordError(f"{path.name}: {error}") from error
        if record.date != day:
            raise RecordError(
                f"{path.name} holds the record of {record.date}, not of {day}; was a file renamed?"
            )
        return record

    def save(self, record: DateRecord) -> None:
        """Write the record, replacing any earlier one for the date, all or nothing."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(record.date)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(
            json.dumps(record.to_json(), allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, path)
