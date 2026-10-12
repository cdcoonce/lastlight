"""Shared support for the G0 tests: served runs for any evening, and synthetic per-date records.

The recorded HRRR and GEFS crops are for one run each, but their messages are checked against the
run they were asked for, so `combined_transport` re-dates them to the runs the code will plan for
an evening and serves both at the real URLs. `synthetic_record` builds a rendered date without
rendering anything, with contrast arrays whose distance from the median is known by construction.
"""

from __future__ import annotations

import datetime as dt

import gefs_support
import hrrr_support
import numpy as np
from hrrr_support import FakeTransport

from lastlight.g0.record import (
    RENDER_SCALES,
    AtmosphereRecord,
    DateRecord,
)
from lastlight.g0.stats import band_members
from lastlight.inputs import gefs, hrrr
from lastlight.site import REFERENCE_POINT

LAT, LON, _ = REFERENCE_POINT
NOW = dt.datetime(2026, 10, 11, tzinfo=dt.UTC)
EVENING = dt.date(2025, 9, 30)


def hrrr_files(evening: dt.date) -> dict[str, bytes]:
    """The HRRR crop and its idx at the URLs `plan_run` picks for `evening`, re-dated to match."""
    plan = hrrr.plan_run(
        evening, LAT, LON, cycle_hour=None, forecast_hour=None, now_utc=NOW, transport=None
    )
    parts = []
    for byte_range in hrrr_support.CROP_MESSAGES.values():
        assert byte_range.end is not None
        data = hrrr_support.CROP_GRIB[byte_range.start : byte_range.end + 1]
        parts.append(gefs_support.edited(data, cycle=(plan.cycle_date, plan.cycle_hour)))
    args = (plan.cycle_date, plan.cycle_hour, plan.forecast_hour)
    return {
        hrrr.grib_url(*args): b"".join(parts),
        hrrr.idx_url(*args): hrrr_support.CROP_IDX_TEXT.encode(),
    }


def gefs_files(evening: dt.date) -> dict[str, bytes]:
    plan = gefs.aerosol_run(evening, LAT, LON)
    return dict(gefs_support.crop_transport(plan.cycle_date, plan.cycle_hour).files)


def combined_transport(
    evening: dt.date = EVENING, *, with_hrrr: bool = True, with_gefs: bool = True
) -> FakeTransport:
    files: dict[str, bytes] = {}
    if with_hrrr:
        files.update(hrrr_files(evening))
    if with_gefs:
        files.update(gefs_files(evening))
    return FakeTransport(files)


def shifted_contrast(shifts: dict[str, float]) -> np.ndarray:
    """A (29, 5, 2) contrast array, zero except `shifts[band]` added over each band's elevations."""
    out = np.zeros((29, 5, 2))
    for band, amount in shifts.items():
        out[list(band_members(band))] += amount
    return out


def synthetic_record(
    day: dt.date,
    shifts: dict[str, float] | None = None,
    *,
    low: dict[str, float] | None = None,
    high: dict[str, float] | None = None,
    aod: float = 0.08,
    dust: float = 0.02,
    smoke: float = 0.0,
    lab_shift: float = 0.0,
) -> DateRecord:
    """A rendered date: `shifts` for the nominal scale, optionally different ones at the ends."""
    shifts = shifts or {}
    contrast = {
        "nominal": shifted_contrast(shifts),
        "low": shifted_contrast(low if low is not None else shifts),
        "high": shifted_contrast(high if high is not None else shifts),
    }
    assert tuple(contrast) == RENDER_SCALES
    lab = np.zeros((29, 7, 3))
    lab[:, :, 0] = 50.0 + lab_shift
    return DateRecord(
        date=day,
        drop_reason=None,
        drop_detail="",
        atmosphere=AtmosphereRecord(
            aod_total=aod,
            aod_dust=dust,
            aod_smoke=smoke,
            ozone_du=280.0,
            pwat_kg_m2=15.0,
            lcdc_pct=10.0,
        ),
        contrast=contrast,
        floor_channels={"nominal": 3, "low": 5, "high": 0},
        curve_lab=lab,
        inputs={"hrrr_cycle_date": day.isoformat()},
    )


def dropped_record(day: dt.date, reason: str, detail: str = "") -> DateRecord:
    return DateRecord(
        date=day,
        drop_reason=reason,
        drop_detail=detail,
        atmosphere=None,
        contrast={},
        floor_channels={},
        curve_lab=None,
        inputs={},
    )


def dates(start: dt.date, n: int) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(n)]


def whtm1_views():
    """WHTM1's views at the three pixel scales, from the frozen fit and the registry."""
    from lastlight.cameras import load_registry
    from lastlight.camgeom.fitfile import load_fit
    from lastlight.g0.view import pixel_scales, sky_view

    fit = load_fit("WHTM1")
    contrast = load_registry()["WHTM1"].contrast
    scales = pixel_scales(fit)
    return {
        name: sky_view(fit, contrast, px_per_deg=None if name == "nominal" else scales[name])
        for name in RENDER_SCALES
    }
