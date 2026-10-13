"""Shared support for the renderer-check tests (this file holds helpers and no tests).

* `write_observed` writes a small `observed.json` the way `lastlight observe` does (schema
  version 1), one file per evening, from (sun elevation, log(B/G), log(R/G)) triples.
* `save_inputs` writes per-evening input records into the check's own store, so a test of the
  report stage never needs the network or the archives.
* `expected_rendered` is an independent route to a frame's rendered lowest-band contrast: the
  sweep elevations have known sun azimuths in `lastlight.g0.view`, so a frame at a sweep
  elevation is rendered here without the check's own sun-azimuth search.
* `served_run` serves the recorded HRRR and GEFS crops at the URLs the planner picks for an
  evening at a given clock, optionally with the `TCDC` message replaced by a constant field.
* `NoNetwork` is a transport that fails the test on any request.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import eccodes
import gefs_support
import hrrr_support
import numpy as np
from g0_support import gefs_files
from hrrr_support import FakeTransport

from lastlight.g0.collect import build_views
from lastlight.g0.contrast import log_chromaticity
from lastlight.g0.view import relative_azimuth, sun_azimuths
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.inputs import hrrr
from lastlight.render import Atmosphere, sky_xyz
from lastlight.renderer_check.store import EveningInputs, InputStore
from lastlight.site import REFERENCE_POINT

LAT, LON, _ = REFERENCE_POINT

NOW = dt.datetime(2026, 10, 20, 12, tzinfo=dt.UTC)
"""A clock far enough after the first evenings that every HRRR analysis is published."""

SUNS = (3.0, 1.5, 0.0, -1.5)
"""Four sweep elevations: frames at these have a sun azimuth known from the G0 view module."""

CRITERION = Path(__file__).resolve().parents[1] / "docs" / "gates" / "renderer-v1" / "criterion.md"

TABLE_SHA256 = "c59ef9240ad6843ebc0136de64fbd90c4d5eef0aa9ac3b5e0ec05e03ad3fc368"


class NoNetwork:
    """Any request is a bug: the report stage must read only the cache and the observed files."""

    def exists(self, url: str) -> bool:
        raise AssertionError(f"the check asked for {url}")

    def get(self, url, byte_range=None):
        raise AssertionError(f"the check asked for {url}")


# --- observed files ------------------------------------------------------------------------------


def frame_json(sun: float, log_bg: float | None, log_rg: float | None = 0.01, **extra) -> dict:
    """One frame as `lastlight observe` writes it (the keys the check reads, and others)."""
    band = {
        "rows": [8, 28],
        "log_rg": log_rg,
        "log_bg": log_bg,
        "n_pixels": 16000,
        "n_excluded": 0,
        "missing": None if log_bg is not None else "too_few_pixels",
    }
    other = {"rows": [28, 48], "log_rg": 0.5, "log_bg": 0.5, "missing": None}
    return {
        "sha256_prefix": "0123456789ab",
        "sun_elevation_deg": sun,
        "light_dome_excluded": False,
        "reference": {"rows": [118, 138], "n_pixels": 16000, "n_excluded": 0, "missing": None},
        "bands": [band, other],
        **extra,
    }


def observed_document(day: dt.date, frames: list[dict], **overrides) -> dict:
    document = {
        "schema_version": 1,
        "camera_id": "WHTM1",
        "date": day.isoformat(),
        "frames_found": len(frames),
        "frames_used": len(frames),
        "frames": frames,
    }
    document.update(overrides)
    return document


def write_observed(
    directory: Path, day: dt.date, triples, *, name: str | None = None, **overrides
) -> Path:
    """Write `<day>.json` from (sun, log_bg, log_rg) triples (log_rg optional)."""
    frames = [frame_json(*triple) for triple in triples]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{day.isoformat()}.json")
    path.write_text(json.dumps(observed_document(day, frames, **overrides)), encoding="utf-8")
    return path


def write_raw(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --- input records -------------------------------------------------------------------------------


def rendered_inputs(
    day: dt.date, *, cloud: float = 0.0, aod: float = 0.05, ozone: float = 280.0
) -> EveningInputs:
    return EveningInputs.rendered(
        day,
        aod_550=aod,
        ozone_du=ozone,
        cloud_mean_pct=cloud,
        inputs={"hrrr_cycle_date": day.isoformat()},
    )


def save_inputs(cache_dir: Path, *records: EveningInputs) -> InputStore:
    store = InputStore(cache_dir / "renderer-v1")
    for record in records:
        store.save(record)
    return store


# --- an independent route to the rendered contrast -----------------------------------------------


def expected_rendered(
    engine: str, day: dt.date, aod: float, ozone: float, suns, scale: str = "nominal"
) -> np.ndarray:
    """(frames, 2) of the lowest band's [log(R/G), log(B/G)] minus the reference's.

    `suns` must be sweep elevations; the relative azimuth is the fit's centre azimuth (nominal
    view) minus the sun's azimuth at that sweep elevation on `day`.
    """
    views = build_views()
    azimuths = dict(zip(ELEVATION_SWEEP, sun_azimuths(day, LAT, LON), strict=True))
    rows = []
    for sun in suns:
        relative = relative_azimuth(views["nominal"].azimuth_deg, azimuths[sun])
        xyz = sky_xyz(
            Atmosphere(aod_550=aod, ozone_du=ozone),
            sun,
            views[scale].elevations_deg,
            relative,
            engine=engine,
        )
        logs, _ = log_chromaticity(xyz)
        rows.append(logs[0] - logs[-1])
    return np.array(rows)


def observed_for_ratio(
    day: dt.date, ratio: float, *, aod: float = 0.05, ozone: float = 280.0, suns=SUNS
) -> list[tuple[float, float, float]]:
    """Observed (sun, log_bg, log_rg) triples whose v1 r_e for log(B/G) is `ratio`.

    Every observed log(B/G) is the v1 rendered one divided by `ratio` (so the medians of the
    absolute values are in that ratio), keeping the rendered sign.
    """
    rendered = expected_rendered("v1", day, aod, ozone, suns)
    return [
        (sun, float(row[1] / ratio), float(row[0] / ratio))
        for sun, row in zip(suns, rendered, strict=True)
    ]


# --- served archives -----------------------------------------------------------------------------


def _with_tcdc(value: float, cycle: tuple[dt.date, int]) -> tuple[bytes, str]:
    """The HRRR crop re-dated to `cycle`, its TCDC message a constant field; the idx to match."""
    parts = []
    for name in hrrr_support.CROP_MESSAGES:
        data = hrrr_support.crop_message(name)
        data = gefs_support.edited(data, cycle=cycle)
        if name == "TCDC" and value is not None:
            handle = eccodes.codes_new_from_message(data)
            try:
                size = eccodes.codes_get_size(handle, "values")
                eccodes.codes_set_values(handle, np.full(size, float(value)))
                data = bytes(eccodes.codes_get_message(handle))
            finally:
                eccodes.codes_release(handle)
        parts.append(data)
    offsets = [0]
    for part in parts[:-1]:
        offsets.append(offsets[-1] + len(part))
    lines = []
    for line, offset in zip(hrrr_support.CROP_IDX_TEXT.splitlines(), offsets, strict=True):
        fields = line.split(":")
        fields[1] = str(offset)
        lines.append(":".join(fields))
    return b"".join(parts), "\n".join(lines) + "\n"


def served_run(evening: dt.date, now: dt.datetime = NOW, *, tcdc: float | None = None):
    """A transport serving the recorded HRRR and GEFS runs for `evening`, as planned at `now`.

    With `tcdc`, the HRRR file's total cloud cover is that constant everywhere.
    """
    lat, lon, _ = REFERENCE_POINT
    plan = hrrr.plan_run(
        evening, lat, lon, cycle_hour=None, forecast_hour=None, now_utc=now, transport=None
    )
    data, idx = _with_tcdc(tcdc, (plan.cycle_date, plan.cycle_hour))
    args = (plan.cycle_date, plan.cycle_hour, plan.forecast_hour)
    files = {hrrr.grib_url(*args): data, hrrr.idx_url(*args): idx.encode()}
    files.update(gefs_files(evening))
    return FakeTransport(files)


def merged(*transports: FakeTransport) -> FakeTransport:
    files: dict[str, bytes] = {}
    for transport in transports:
        files.update(transport.files)
    return FakeTransport(files)
