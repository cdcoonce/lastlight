"""Collecting each considered evening's inputs, once, politely, into the check's own store.

Criterion definitions 4 and 7. For one evening:

1. **The atmosphere** is exactly G0's: `lastlight.g0.collect.Collector` builds the evening's
   `DateRecord` from the HRRR analysis and GEFS-Aerosols, and the check takes `aod_550` from its
   `AtmosphereRecord.aod_total` and `ozone_du` from `AtmosphereRecord.ozone_du`. G0's field list
   (`HRRR_FIELD_NAMES`) and behaviour are not changed. If G0 would drop the evening, so does the
   check, with G0's reason.
2. **The cloud** is HRRR's total cloud cover (`TCDC`, entire atmosphere) at forecast hour 0 of
   the cycle at the target hour (the whole UTC hour nearest the instant the sun reaches elevation
   0 at the public reference point), sampled along G0's 17-point transect and averaged. It is
   the analysis or nothing: until the target-hour cycle is two hours old the evening is pending
   and gets no record (it is tried again on the next run), and a forecast is never substituted.
   A transect point with no value drops the evening as `tcdc_missing`.

Requests go through whatever transport the caller gives (the command passes G0's polite one).
A network fault is raised as `TransientError`, nothing is written, and the next run tries again.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from lastlight import geometry
from lastlight.g0.collect import Collector, TransientError
from lastlight.g0.record import DateRecord
from lastlight.g0.transport import is_not_found
from lastlight.inputs import hrrr
from lastlight.inputs.http import FetchError, Transport
from lastlight.renderer_check import rules
from lastlight.renderer_check.store import EveningInputs, InputStore
from lastlight.site import REFERENCE_POINT

ANALYSIS_READY_AFTER: Final = dt.timedelta(hours=2)
"""How old the target-hour cycle must be before its analysis is read: `hrrr.plan_run` waits the
same two hours before it takes the target-hour cycle instead of the newest complete run."""

TRANSECT_POINTS: Final = len(hrrr.TRANSECT_DISTANCES_KM)

RETRIED_REASONS: Final = frozenset({"hrrr_run_missing", "aerosol_run_missing"})
RETRY_WINDOW_DAYS: Final = 2
"""A run missing from an archive is a fact about an old evening but may only be a publication lag
for a recent one (GEFS-Aerosols' cycle at the sunset can be published hours after the evening), so
such a record is tried again until the evening is two calendar days old, and kept after that."""

_TCDC: Final = "TCDC"
_FIELD_MISSING: Final = "expected exactly one idx message"
_URL: Final = re.compile(r"https?://\S+")
_RUN_FILE: Final = re.compile(r"\b(?:hrrr|gefs)\.[A-Za-z0-9_.\-/]*")
_DETAIL_LIMIT: Final = 240


class _Dropped(Exception):
    """Internal: the evening cannot be built, for the reason its record will carry."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _clean(text: str) -> str:
    """An error message made safe to keep and print: no URL, no archive file name, short."""
    text = _RUN_FILE.sub("the archive file", _URL.sub("the archive", text))
    return " ".join(text.split())[:_DETAIL_LIMIT]


def analysis_plan(evening: dt.date, now_utc: dt.datetime) -> hrrr.RunPlan | None:
    """The target-hour analysis of `evening` if it is two hours old at `now_utc`, else None.

    Makes no request: asking `hrrr.plan_run` for forecast hour 0 names the target-hour cycle
    without looking at any archive.
    """
    lat, lon, _ = REFERENCE_POINT
    plan = hrrr.plan_run(
        evening, lat, lon, cycle_hour=None, forecast_hour=0, now_utc=now_utc, transport=None
    )
    cycle = dt.datetime.combine(plan.cycle_date, dt.time(plan.cycle_hour), tzinfo=dt.UTC)
    return plan if cycle + ANALYSIS_READY_AFTER <= now_utc else None


def cloud_percent(samples: Sequence[hrrr.TransectSample]) -> float | None:
    """The mean `TCDC` over the sampled points, or None if any point has no value."""
    values = [sample.value for sample in samples if sample.field == _TCDC]
    if not values or any(value is None for value in values):
        return None
    return rules.cloud_mean([value for value in values if value is not None])


class _Collects(Protocol):
    def collect(self, evening: dt.date) -> DateRecord: ...


class InputFetcher:
    """Builds the inputs of one evening from the archives for the check."""

    def __init__(
        self,
        transport: Transport,
        cache_dir: Path | str,
        now_utc: dt.datetime,
        collector: _Collects | None = None,
    ) -> None:
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.now_utc = now_utc
        self.collector = collector or Collector(transport, self.cache_dir, now_utc)

    def _cloud(self, plan: hrrr.RunPlan, evening: dt.date) -> tuple[float | None, dict[str, Any]]:
        lat, lon, _ = REFERENCE_POINT
        try:
            fields = hrrr.fetch_fields(
                plan.cycle_date,
                plan.cycle_hour,
                0,
                (hrrr.field_spec(_TCDC),),
                self.cache_dir / "hrrr",
                transport=self.transport,
            )
            distances = [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
            samples = hrrr.sample_transect(fields, geometry.transect(evening, lat, lon, distances))
        except FetchError as error:
            if is_not_found(error):
                raise _Dropped(
                    "hrrr_run_missing", "HRRR: the file is not in the archive"
                ) from error
            raise TransientError(f"HRRR: {_clean(str(error))}") from error
        except hrrr.HrrrError as error:
            reason = "hrrr_field_missing" if _FIELD_MISSING in str(error) else "hrrr_unreadable"
            raise _Dropped(reason, f"HRRR: {_clean(str(error))}") from error
        return cloud_percent(samples), {"tcdc_digest": fields.digests.get(_TCDC)}

    def fetch(self, evening: dt.date) -> EveningInputs | None:
        """The evening's inputs or drop reason, or None while its analysis is not published.

        Raises `TransientError` for a network fault; nothing is recorded then.
        """
        plan = analysis_plan(evening, self.now_utc)
        if plan is None:
            return None
        record = self.collector.collect(evening)
        if record.drop_reason is not None:
            return EveningInputs.dropped(
                evening, record.drop_reason, record.drop_detail, record.inputs
            )
        atmosphere = record.atmosphere
        assert atmosphere is not None  # a record without a drop reason holds its atmosphere
        try:
            cloud, provenance = self._cloud(plan, evening)
        except _Dropped as dropped:
            return EveningInputs.dropped(evening, dropped.reason, dropped.detail, record.inputs)
        if cloud is None:
            return EveningInputs.dropped(
                evening,
                "tcdc_missing",
                "HRRR: a point of the sightline has no total cloud cover value",
                record.inputs,
            )
        return EveningInputs.rendered(
            evening,
            aod_550=atmosphere.aod_total,
            ozone_du=atmosphere.ozone_du,
            cloud_mean_pct=cloud,
            cloud_points=TRANSECT_POINTS,
            inputs={**record.inputs, **provenance},
        )


class _Fetches(Protocol):
    def fetch(self, evening: dt.date) -> EveningInputs | None: ...


def needs_fetch(store: InputStore, evening: dt.date, now_utc: dt.datetime) -> bool:
    """True when the evening has no record, or a missing-run one still inside its retry window."""
    record = store.load(evening)
    if record is None:
        return True
    recent = (now_utc.date() - evening).days < RETRY_WINDOW_DAYS
    return recent and record.drop_reason in RETRIED_REASONS


@dataclass(frozen=True)
class FetchSummary:
    """What a run did. `pending` evenings are not published yet, `faults` had a network fault."""

    recorded: int
    dropped: int
    skipped: int
    pending: tuple[dt.date, ...]
    faults: tuple[dt.date, ...]


def fetch_range(
    evenings: Iterable[dt.date],
    store: InputStore,
    fetcher: _Fetches,
    *,
    now_utc: dt.datetime,
    progress: Callable[[str], None] | None = None,
) -> FetchSummary:
    """Fetch each evening that needs it, saving every record as soon as it exists."""
    report = progress or (lambda line: None)
    recorded = dropped = skipped = 0
    pending: list[dt.date] = []
    faults: list[dt.date] = []
    for day in evenings:
        if not needs_fetch(store, day, now_utc):
            skipped += 1
            report(f"{day.isoformat()} skipped (already collected)")
            continue
        try:
            record = fetcher.fetch(day)
        except TransientError as error:
            faults.append(day)
            report(f"{day.isoformat()} fault, will be retried: {error}")
            continue
        if record is None:
            pending.append(day)
            report(f"{day.isoformat()} pending: the analysis is not published yet")
            continue
        store.save(record)
        if record.ok:
            recorded += 1
            assert record.cloud_mean_pct is not None
            report(f"{day.isoformat()} collected (cloud {record.cloud_mean_pct:.1f} percent)")
        else:
            dropped += 1
            report(f"{day.isoformat()} dropped: {record.drop_reason}")
    return FetchSummary(recorded, dropped, skipped, tuple(pending), tuple(faults))


__all__ = [
    "ANALYSIS_READY_AFTER",
    "RETRIED_REASONS",
    "RETRY_WINDOW_DAYS",
    "TRANSECT_POINTS",
    "FetchSummary",
    "InputFetcher",
    "analysis_plan",
    "cloud_percent",
    "fetch_range",
    "needs_fetch",
]
