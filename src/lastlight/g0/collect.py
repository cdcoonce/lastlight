"""One evening from archived inputs to a G0 record, and the resumable run over a range of dates.

For each evening (`Collector.collect`):

1. **HRRR.** The analysis (forecast hour 0) of the whole UTC hour nearest the instant the sun
   reaches the horizon at the public reference point, exactly the past-evening path of
   `hrrr.plan_run`. Only three fields are downloaded, by byte range: `LCDC` (the reduction
   requires it), `AOTK` (smoke optical depth) and `PWAT`. They are sampled along the sightline
   toward the sunset azimuth.
2. **GEFS-Aerosols.** Total and dust optical depth at 550 nm at the reference point
   (`lastlight.inputs.gefs`), the analysis of the latest six-hourly cycle at or before that
   instant.
3. **Reduce** with `lastlight.atmosphere.reduce_transect(..., aerosol=...)`: the archive's total
   is the aerosol optical depth rendered; ozone is the month's climatology.
4. **Render** WHTM1's contrast bands at the relative azimuth of every sweep elevation, at the
   fit's nominal pixel scale and at each end of its range (`lastlight.g0.contrast`), and the
   forecast's own absolute curve for tier B.

An evening that cannot be built is a record with a reason from `record.DROP_REASONS`. A network
fault is not a reason: it is raised as `TransientError`, nothing is written, and the next run
tries the date again. `collect_range` writes each record as soon as it exists, so a run can stop
and resume (or be split in chunks, or stop itself at a time budget) without losing or repeating
work.

The v1 re-run (`render_v1`, `rerender_range`) does not touch an archive: a record the v0 run
wrote already holds the atmosphere its date was rendered from, so the date is re-rendered from
that with renderer v1, through the same seams, into a sibling store.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from lastlight import geometry
from lastlight.atmosphere import AtmosphereV0, ReductionError, reduce_transect
from lastlight.cameras import load_registry
from lastlight.camgeom.fitfile import GEOMETRY_DIR, load_fit
from lastlight.g0.contrast import day_contrasts
from lastlight.g0.record import (
    RENDER_SCALES,
    AtmosphereRecord,
    DateRecord,
    RecordStore,
)
from lastlight.g0.transport import is_not_found
from lastlight.g0.view import SkyView, pixel_scales, relative_azimuth, sky_view, sun_azimuths
from lastlight.inputs import gefs, hrrr
from lastlight.inputs.http import FetchError, Transport
from lastlight.observe.bands import validate_contrast
from lastlight.render import Atmosphere, render_curve
from lastlight.render import v1 as v1_engine
from lastlight.site import REFERENCE_POINT

HRRR_FIELD_NAMES: Final = ("LCDC", "AOTK", "PWAT")
"""The HRRR fields downloaded: the two the reduction reads and precipitable water."""

_FIELD_MISSING: Final = "expected exactly one idx message"
_URL: Final = re.compile(r"https?://\S+")
_RUN_FILE: Final = re.compile(r"\b(?:hrrr|gefs)\.[A-Za-z0-9_.\-/]*")
_DETAIL_LIMIT: Final = 240


class TransientError(RuntimeError):
    """A fault of the network or an archive server, not a fact about the date: try again later."""


class _Dropped(Exception):
    """Internal: the date cannot be built, for the reason a record will carry."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _clean(text: str) -> str:
    """An error message made safe to keep and print: no URL, no archive file name, short."""
    text = _URL.sub("the archive", text)
    text = _RUN_FILE.sub("the archive file", text)
    return " ".join(text.split())[:_DETAIL_LIMIT]


def _classify_fetch(error: FetchError, missing_reason: str, source: str) -> _Dropped:
    if is_not_found(error):
        return _Dropped(missing_reason, f"{source}: the file is not in the archive")
    raise TransientError(f"{source}: {_clean(str(error))}") from error


def build_views(camera_id: str = "WHTM1", geometry_dir: Path = GEOMETRY_DIR) -> dict[str, SkyView]:
    """The camera's view at each of `RENDER_SCALES`, from its frozen fit and registry table.

    Checks first that the bands lie inside the fit's sky mask, as `lastlight observe` does.
    """
    fit = load_fit(camera_id, geometry_dir)
    camera = load_registry()[camera_id]
    if camera.contrast is None:
        raise ValueError(f"{camera_id} has no contrast table in the registry")
    validate_contrast(fit, camera.contrast)
    scales = pixel_scales(fit)
    return {
        name: sky_view(fit, camera.contrast, px_per_deg=None if name == "nominal" else scales[name])
        for name in RENDER_SCALES
    }


class Collector:
    """Builds the record of one evening from the archives, for one camera's frozen geometry."""

    def __init__(
        self,
        transport: Transport,
        cache_dir: Path | str,
        now_utc: dt.datetime,
        *,
        camera_id: str = "WHTM1",
        geometry_dir: Path = GEOMETRY_DIR,
    ) -> None:
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.now_utc = now_utc
        self.views: dict[str, SkyView] = build_views(camera_id, geometry_dir)

    # --- the inputs ------------------------------------------------------------------------

    def _hrrr_samples(self, evening: dt.date) -> tuple[list[hrrr.TransectSample], dict]:
        lat, lon, _ = REFERENCE_POINT
        specs = tuple(hrrr.field_spec(name) for name in HRRR_FIELD_NAMES)
        try:
            plan = hrrr.plan_run(
                evening,
                lat,
                lon,
                cycle_hour=None,
                forecast_hour=None,
                now_utc=self.now_utc,
                transport=self.transport,
            )
            fields = hrrr.fetch_fields(
                plan.cycle_date,
                plan.cycle_hour,
                plan.forecast_hour,
                specs,
                self.cache_dir / "hrrr",
                transport=self.transport,
            )
            distances = [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
            samples = hrrr.sample_transect(fields, geometry.transect(evening, lat, lon, distances))
        except FetchError as error:
            raise _classify_fetch(error, "hrrr_run_missing", "HRRR") from error
        except hrrr.HrrrError as error:
            reason = "hrrr_field_missing" if _FIELD_MISSING in str(error) else "hrrr_unreadable"
            raise _Dropped(reason, f"HRRR: {_clean(str(error))}") from error
        provenance = {
            "hrrr_cycle_date": plan.cycle_date.isoformat(),
            "hrrr_cycle_hour": plan.cycle_hour,
            "hrrr_forecast_hour": plan.forecast_hour,
            "hrrr_digests": dict(fields.digests),
        }
        return samples, provenance

    def _aerosol(self, evening: dt.date) -> tuple[gefs.AerosolSample, dict]:
        lat, lon, _ = REFERENCE_POINT
        try:
            sample = gefs.fetch_aerosol(
                evening, lat, lon, self.cache_dir / "gefs", transport=self.transport
            )
        except FetchError as error:
            raise _classify_fetch(error, "aerosol_run_missing", "GEFS-Aerosols") from error
        except gefs.AerosolArchiveError as error:
            raise _Dropped("aerosol_unreadable", f"GEFS-Aerosols: {_clean(str(error))}") from error
        provenance = {
            "aerosol_cycle_date": sample.cycle_date.isoformat(),
            "aerosol_cycle_hour": sample.cycle_hour,
            "aerosol_forecast_hour": sample.forecast_hour,
            "aerosol_digests": dict(sample.digests),
            "aerosol_cell": list(sample.cell),
            "aerosol_cell_distance_km": sample.cell_distance_km,
        }
        return sample, provenance

    # --- one evening -----------------------------------------------------------------------

    def collect(self, evening: dt.date) -> DateRecord:
        """The record of `evening`; raises `TransientError` for a fault that should be retried."""
        try:
            samples, hrrr_provenance = self._hrrr_samples(evening)
            aerosol, aerosol_provenance = self._aerosol(evening)
            try:
                reduction = reduce_transect(samples, evening, aerosol=aerosol.column())
            except ReductionError as error:
                raise _Dropped("hrrr_field_empty", _clean(str(error))) from error
        except _Dropped as dropped:
            return DateRecord(
                date=evening,
                drop_reason=dropped.reason,
                drop_detail=dropped.detail,
                atmosphere=None,
                contrast={},
                floor_channels={},
                curve_lab=None,
                inputs={},
            )
        return self._render(evening, reduction, {**hrrr_provenance, **aerosol_provenance})

    def _render(self, evening: dt.date, reduction: AtmosphereV0, inputs: dict) -> DateRecord:
        lat, lon, _ = REFERENCE_POINT
        atmosphere = reduction.atmosphere
        azimuths = tuple(
            relative_azimuth(self.views["nominal"].azimuth_deg, sun)
            for sun in sun_azimuths(evening, lat, lon)
        )
        contrast: dict[str, np.ndarray] = {}
        floored: dict[str, int] = {}
        for name in RENDER_SCALES:
            contrast[name], floored[name] = day_contrasts(atmosphere, self.views[name], azimuths)
        lab = np.array([[row["L"], row["a"], row["b"]] for row in render_curve(atmosphere)])
        assert reduction.aod_dust is not None
        return DateRecord(
            date=evening,
            drop_reason=None,
            drop_detail="",
            atmosphere=AtmosphereRecord(
                aod_total=atmosphere.aod_550,
                aod_dust=reduction.aod_dust,
                aod_smoke=reduction.aod_smoke,
                ozone_du=reduction.ozone_du,
                pwat_kg_m2=reduction.pwat_kg_m2,
                lcdc_pct=reduction.clouds.low_pct,
            ),
            contrast=contrast,
            floor_channels=floored,
            curve_lab=lab.reshape(len(geometry.ELEVATION_SWEEP), -1, 3),
            inputs=inputs,
        )


# --- the run over a range ---------------------------------------------------------------------


@dataclass(frozen=True)
class CollectSummary:
    """What a run did. `unrecorded` are dates with a fault, `remaining` dates not yet tried."""

    collected: int
    dropped: int
    skipped: int
    faults: int
    unrecorded: tuple[dt.date, ...]
    remaining: tuple[dt.date, ...]
    stopped_early: bool


def collect_range(
    days: Iterable[dt.date],
    store: RecordStore,
    collector: Collector,
    *,
    progress: Callable[[str], None] | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    refresh: bool = False,
) -> CollectSummary:
    """Collect each date that has no record yet, saving every record as it is made.

    A date already in the store is skipped without a request (unless `refresh`). A fault
    (`TransientError`) leaves the date unrecorded and the run goes on. With `budget_seconds`, the
    run stops between dates once that much time has passed, and says which dates are left.
    """
    report = progress or (lambda line: None)
    todo = list(days)
    started = clock()
    collected = dropped = skipped = 0
    unrecorded: list[dt.date] = []
    for position, day in enumerate(todo):
        if budget_seconds is not None and clock() - started >= budget_seconds:
            return CollectSummary(
                collected,
                dropped,
                skipped,
                len(unrecorded),
                tuple(unrecorded),
                tuple(d for d in todo[position:] if d not in store.dates()),
                True,
            )
        if not refresh and store.path(day).is_file():
            skipped += 1
            report(f"{day.isoformat()} skipped (already collected)")
            continue
        try:
            record = collector.collect(day)
        except TransientError as error:
            unrecorded.append(day)
            report(f"{day.isoformat()} fault, will be retried: {error}")
            continue
        store.save(record)
        if record.ok:
            collected += 1
            assert record.atmosphere is not None
            report(
                f"{day.isoformat()} rendered (aerosol optical depth "
                f"{record.atmosphere.aod_total:.3f}, dust {record.atmosphere.aod_dust:.3f})"
            )
        else:
            dropped += 1
            report(f"{day.isoformat()} dropped: {record.drop_reason}")
    return CollectSummary(
        collected, dropped, skipped, len(unrecorded), tuple(unrecorded), (), False
    )


# --- the re-run with renderer v1 --------------------------------------------------------------


def render_v1(record: DateRecord, views: Mapping[str, SkyView]) -> DateRecord:
    """The record of the same date rendered with renderer v1 from the v0 record's atmosphere.

    The atmosphere is rebuilt as `reduce_transect` builds it: `aod_550` is the record's
    `aod_total`, `ozone_du` its ozone, every other field of `Atmosphere` at its default. A dropped
    v0 record stays dropped with the same reason and detail. An atmosphere outside the table v1
    reads is dropped with the reason `v1_outside_table`, never clamped.
    """
    if record.atmosphere is None:
        return DateRecord(
            date=record.date,
            drop_reason=record.drop_reason,
            drop_detail=record.drop_detail,
            atmosphere=None,
            contrast={},
            floor_channels={},
            curve_lab=None,
            inputs=dict(record.inputs),
        )
    atmosphere = Atmosphere(
        aod_550=record.atmosphere.aod_total, ozone_du=record.atmosphere.ozone_du
    )
    try:
        for view in views.values():
            v1_engine.check_domain(atmosphere, geometry.ELEVATION_SWEEP[0], view.elevations_deg)
    except ValueError as error:
        return DateRecord(
            date=record.date,
            drop_reason="v1_outside_table",
            drop_detail=_clean(str(error)),
            atmosphere=None,
            contrast={},
            floor_channels={},
            curve_lab=None,
            inputs=dict(record.inputs),
        )
    lat, lon, _ = REFERENCE_POINT
    azimuths = tuple(
        relative_azimuth(views["nominal"].azimuth_deg, sun)
        for sun in sun_azimuths(record.date, lat, lon)
    )
    contrast: dict[str, np.ndarray] = {}
    floored: dict[str, int] = {}
    for name in RENDER_SCALES:
        contrast[name], floored[name] = day_contrasts(
            atmosphere, views[name], azimuths, engine="v1"
        )
    lab = np.array(
        [[row["L"], row["a"], row["b"]] for row in render_curve(atmosphere, engine="v1")]
    )
    return DateRecord(
        date=record.date,
        drop_reason=None,
        drop_detail="",
        atmosphere=record.atmosphere,
        contrast=contrast,
        floor_channels=floored,
        curve_lab=lab.reshape(len(geometry.ELEVATION_SWEEP), -1, 3),
        inputs={**record.inputs, "engine": "v1"},
    )


def rerender_range(
    days: Iterable[dt.date],
    source: RecordStore,
    target: RecordStore,
    views: Mapping[str, SkyView],
    *,
    progress: Callable[[str], None] | None = None,
    budget_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    refresh: bool = False,
) -> CollectSummary:
    """Re-render each date of `source` (the v0 records) with v1 into `target`, resumably.

    A date already in `target` is skipped (unless `refresh`); a date with no record in `source`
    cannot be re-rendered and is listed in `unrecorded`. `collected` counts dates rendered,
    `dropped` dropped records carried over (or newly outside the table). No archive is read.
    With `budget_seconds`, the run stops between dates once that much time has passed.
    """
    report = progress or (lambda line: None)
    todo = list(days)
    started = clock()
    rendered = dropped = skipped = 0
    unrecorded: list[dt.date] = []
    for position, day in enumerate(todo):
        if budget_seconds is not None and clock() - started >= budget_seconds:
            return CollectSummary(
                rendered,
                dropped,
                skipped,
                len(unrecorded),
                tuple(unrecorded),
                tuple(d for d in todo[position:] if d not in target.dates()),
                True,
            )
        if not refresh and target.path(day).is_file():
            skipped += 1
            report(f"{day.isoformat()} skipped (already re-rendered)")
            continue
        original = source.load(day)
        if original is None:
            unrecorded.append(day)
            report(f"{day.isoformat()} has no v0 record, so there is nothing to re-render")
            continue
        record = render_v1(original, views)
        target.save(record)
        if record.ok:
            rendered += 1
            report(f"{day.isoformat()} re-rendered with v1")
        else:
            dropped += 1
            report(f"{day.isoformat()} dropped: {record.drop_reason}")
    return CollectSummary(rendered, dropped, skipped, len(unrecorded), tuple(unrecorded), (), False)
