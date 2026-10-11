"""One evening's rendered record from archived inputs, and the resumable run over a range.

The pipeline for a date: plan the HRRR analysis for the hour nearest the sun reaching the horizon,
download only LCDC, AOTK and PWAT, sample them along the sightline, read total and dust aerosol
from GEFS-Aerosols, reduce with `reduce_transect(..., aerosol=...)`, render WHTM1's rows at the
relative azimuth of every sweep elevation at three pixel scales, and render the forecast's own
curve for tier B. A date that cannot be built is a record with its reason, never a gap; a network
fault is not a reason (it is raised, nothing is written, and the date is tried again next run).
"""

from __future__ import annotations

import datetime as dt
import itertools

import gefs_support
import hrrr_support
import numpy as np
import pytest
from g0_support import EVENING, LAT, LON, NOW, combined_transport, dates

from lastlight import atmosphere as atmosphere_module
from lastlight import geometry
from lastlight.atmosphere import ReductionError, ozone_du
from lastlight.cameras import load_registry
from lastlight.camgeom.fitfile import load_fit
from lastlight.g0 import collect as collect_module
from lastlight.g0.collect import (
    HRRR_FIELD_NAMES,
    Collector,
    TransientError,
    collect_range,
)
from lastlight.g0.contrast import day_contrasts
from lastlight.g0.record import RENDER_SCALES, RecordStore
from lastlight.g0.view import pixel_scales, relative_azimuth, sky_view, sun_azimuths
from lastlight.inputs import gefs, hrrr
from lastlight.inputs.http import FetchError
from lastlight.render import Atmosphere, render_curve


def collector(tmp_path, transport, **kwargs) -> Collector:
    return Collector(transport=transport, cache_dir=tmp_path / "cache", now_utc=NOW, **kwargs)


# --- the pipeline -------------------------------------------------------------------------------


def test_the_hrrr_fields_are_the_three_the_reduction_and_the_study_use() -> None:
    assert HRRR_FIELD_NAMES == ("LCDC", "AOTK", "PWAT")


def test_an_evening_becomes_a_record_with_three_scales_and_an_absolute_curve(tmp_path) -> None:
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    assert record.ok and record.drop_reason is None
    assert record.date == EVENING
    assert set(record.contrast) == set(RENDER_SCALES)
    for scale in RENDER_SCALES:
        assert record.contrast[scale].shape == (29, 5, 2)
        assert np.isfinite(record.contrast[scale]).all()
    assert record.curve_lab is not None and record.curve_lab.shape == (29, 7, 3)
    assert not np.array_equal(record.contrast["low"], record.contrast["nominal"])
    assert not np.array_equal(record.contrast["high"], record.contrast["nominal"])


def test_the_atmosphere_is_the_archives_aerosol_the_hrrr_smoke_and_the_months_ozone(
    tmp_path,
) -> None:
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    assert record.atmosphere is not None
    assert record.atmosphere.aod_total == pytest.approx(gefs_support.TOTAL_AT_REFERENCE, abs=1e-6)
    assert record.atmosphere.aod_dust == pytest.approx(gefs_support.DUST_AT_REFERENCE, abs=1e-6)
    assert record.atmosphere.ozone_du == ozone_du(EVENING.month)
    assert record.atmosphere.aod_smoke >= 0.0
    assert record.atmosphere.pwat_kg_m2 is not None and record.atmosphere.pwat_kg_m2 > 0.0
    assert 0.0 <= record.atmosphere.lcdc_pct <= 100.0


def test_the_nominal_contrast_is_the_renderer_at_the_relative_azimuth_of_each_elevation(
    tmp_path,
) -> None:
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    fit = load_fit("WHTM1")
    contrast = load_registry()["WHTM1"].contrast
    assert record.atmosphere is not None and contrast is not None
    view = sky_view(fit, contrast)
    azimuths = tuple(
        relative_azimuth(view.azimuth_deg, sun) for sun in sun_azimuths(EVENING, LAT, LON)
    )
    atmosphere = Atmosphere(
        aod_550=record.atmosphere.aod_total, ozone_du=record.atmosphere.ozone_du
    )
    expected, floored = day_contrasts(atmosphere, view, azimuths)
    np.testing.assert_allclose(record.contrast["nominal"], expected, rtol=0, atol=1e-12)
    assert record.floor_channels["nominal"] == floored
    low = sky_view(fit, contrast, px_per_deg=pixel_scales(fit)["low"])
    np.testing.assert_allclose(
        record.contrast["low"], day_contrasts(atmosphere, low, azimuths)[0], rtol=0, atol=1e-12
    )


def test_the_absolute_curve_is_the_forecasts_own_curve_for_the_same_atmosphere(tmp_path) -> None:
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    assert record.atmosphere is not None
    atmosphere = Atmosphere(
        aod_550=record.atmosphere.aod_total, ozone_du=record.atmosphere.ozone_du
    )
    curve = render_curve(atmosphere)
    lab = np.array([[row["L"], row["a"], row["b"]] for row in curve]).reshape(29, 7, 3)
    np.testing.assert_allclose(record.curve_lab, lab, rtol=0, atol=1e-12)


def test_only_three_hrrr_messages_and_two_gefs_messages_are_downloaded(tmp_path) -> None:
    transport = combined_transport()
    collector(tmp_path, transport).collect(EVENING)
    ranged = transport.ranged_gets()
    assert len(ranged) == 3 + 2
    plain_gets = [c for c in transport.calls if c.method == "GET" and c.byte_range is None]
    assert len(plain_gets) == 2  # the two idx files
    assert not [c for c in transport.calls if c.method == "HEAD"]  # the analysis path needs none


def test_the_hrrr_run_is_the_analysis_of_the_hour_nearest_the_sun_reaching_the_horizon(
    tmp_path,
) -> None:
    transport = combined_transport()
    record = collector(tmp_path, transport).collect(EVENING)
    plan = hrrr.plan_run(
        EVENING, LAT, LON, cycle_hour=None, forecast_hour=None, now_utc=NOW, transport=None
    )
    assert plan.forecast_hour == 0
    instant = geometry.elevation_crossings(EVENING, LAT, LON)[0.0]
    target = (instant + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    assert (plan.cycle_date, plan.cycle_hour) == (target.date(), target.hour)
    assert any(hrrr.idx_url(plan.cycle_date, plan.cycle_hour, 0) == c.url for c in transport.calls)
    assert record.inputs["hrrr_forecast_hour"] == 0
    assert record.inputs["hrrr_cycle_date"] == plan.cycle_date.isoformat()


def test_the_record_remembers_which_runs_it_read_for_the_cache_only(tmp_path) -> None:
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    assert record.inputs["aerosol_cycle_date"] == (EVENING + dt.timedelta(days=1)).isoformat()
    assert record.inputs["aerosol_forecast_hour"] == 0
    assert len(record.inputs["hrrr_digests"]) == 3 and len(record.inputs["aerosol_digests"]) == 2


def test_the_byte_range_cache_is_shared_by_a_second_collection(tmp_path) -> None:
    transport = combined_transport()
    first = collector(tmp_path, transport).collect(EVENING)
    transport2 = combined_transport()
    second = collector(tmp_path, transport2).collect(EVENING)
    assert len(transport2.ranged_gets()) == 0
    np.testing.assert_array_equal(second.contrast["nominal"], first.contrast["nominal"])


# --- dates that cannot be built are records with a reason --------------------------------------


def test_a_missing_hrrr_run_is_a_dropped_date(tmp_path) -> None:
    record = collector(tmp_path, combined_transport(with_hrrr=False)).collect(EVENING)
    assert not record.ok and record.drop_reason == "hrrr_run_missing"
    assert record.atmosphere is None and record.curve_lab is None


def test_a_missing_aerosol_run_is_a_dropped_date(tmp_path) -> None:
    record = collector(tmp_path, combined_transport(with_gefs=False)).collect(EVENING)
    assert record.drop_reason == "aerosol_run_missing"


def test_an_idx_without_the_aerosol_field_is_a_dropped_date(tmp_path) -> None:
    transport = combined_transport()
    plan = hrrr.plan_run(
        EVENING, LAT, LON, cycle_hour=None, forecast_hour=None, now_utc=NOW, transport=None
    )
    url = hrrr.idx_url(plan.cycle_date, plan.cycle_hour, 0)
    lines = [ln for ln in hrrr_support.CROP_IDX_TEXT.splitlines() if ":AOTK:" not in ln]
    transport.files[url] = ("\n".join(lines) + "\n").encode()
    record = collector(tmp_path, transport).collect(EVENING)
    assert record.drop_reason == "hrrr_field_missing"
    assert "AOTK" in record.drop_detail


def test_a_hrrr_message_from_another_run_is_a_dropped_date(tmp_path) -> None:
    transport = combined_transport()
    plan = hrrr.plan_run(
        EVENING, LAT, LON, cycle_hour=None, forecast_hour=None, now_utc=NOW, transport=None
    )
    url = hrrr.grib_url(plan.cycle_date, plan.cycle_hour, 0)
    transport.files[url] = hrrr_support.CROP_GRIB  # the unedited crop: cycle 2026-10-10
    record = collector(tmp_path, transport).collect(EVENING)
    assert record.drop_reason == "hrrr_unreadable"


def test_an_aerosol_message_from_another_run_is_a_dropped_date(tmp_path) -> None:
    transport = combined_transport()
    plan = gefs.aerosol_run(EVENING, LAT, LON)
    transport.files[gefs.chem_url(plan.cycle_date, plan.cycle_hour, 0)] = gefs_support.redated_crop(
        dt.date(2025, 9, 29), 0
    )
    record = collector(tmp_path, transport).collect(EVENING)
    assert record.drop_reason == "aerosol_unreadable"


def test_a_transect_with_no_aotk_value_is_a_dropped_date(tmp_path, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise ReductionError("no AOTK value anywhere on the transect")

    monkeypatch.setattr(collect_module, "reduce_transect", refuse)
    record = collector(tmp_path, combined_transport()).collect(EVENING)
    assert record.drop_reason == "hrrr_field_empty"
    assert "AOTK" in record.drop_detail


def test_a_dropped_records_detail_never_carries_a_clock_time_or_a_url(tmp_path) -> None:
    record = collector(tmp_path, combined_transport(with_hrrr=False)).collect(EVENING)
    assert "http" not in record.drop_detail
    assert ".t" not in record.drop_detail.replace(".txt", "")


# --- the detail of a dropped date is made safe to keep and print ------------------------------


def test_clean_replaces_a_url_wherever_it_sits() -> None:
    # A URL that is not an archive file name, so only the URL scrub can remove it.
    cleaned = collect_module._clean("failed to read https://example.org/a/b.bin: refused")
    assert cleaned == "failed to read the archive refused"


def test_clean_replaces_an_archive_file_name_that_is_not_in_a_url() -> None:
    cleaned = collect_module._clean("cannot read hrrr.t06z.wrfsfcf00.grib2 as asked")
    assert cleaned == "cannot read the archive file as asked"


def test_clean_replaces_a_gefs_file_name_that_is_not_in_a_url() -> None:
    cleaned = collect_module._clean("cannot read gefs.chem.t12z.a2d_0p25.f000.grib2 as asked")
    assert cleaned == "cannot read the archive file as asked"


def test_clean_cuts_a_long_message_to_the_limit() -> None:
    cleaned = collect_module._clean("word " * 500)
    assert len(cleaned) == collect_module._DETAIL_LIMIT


def test_clean_joins_lines_and_runs_of_spaces() -> None:
    assert collect_module._clean("one\n  two\t\tthree") == "one two three"


def test_an_unreadable_aerosol_messages_detail_has_no_url_and_no_file_name(tmp_path) -> None:
    # A run whose bytes are not one whole message: the archive error names the URL (with its cycle
    # hour) and the range, and the record must keep neither.
    transport = combined_transport()
    plan = gefs.aerosol_run(EVENING, LAT, LON)
    url = gefs.chem_url(plan.cycle_date, plan.cycle_hour, 0)
    transport.files[url] = transport.files[url][:-20]  # the open-ended dust range loses its trailer
    record = collector(tmp_path, transport).collect(EVENING)
    assert record.drop_reason == "aerosol_unreadable"
    assert "whole GRIB message" in record.drop_detail  # the detail is still the archive's finding
    assert "http" not in record.drop_detail and "grib2" not in record.drop_detail
    assert ".t" not in record.drop_detail


# --- a network fault is not a reason -----------------------------------------------------------


class Broken:
    """A transport whose every request times out."""

    def exists(self, url: str) -> bool:
        raise FetchError(f"{url}: timed out")

    def get(self, url, byte_range=None):
        raise FetchError(f"{url}: timed out")


def test_a_network_fault_is_raised_and_not_recorded(tmp_path) -> None:
    with pytest.raises(TransientError, match="timed out"):
        collector(tmp_path, Broken()).collect(EVENING)


def test_a_server_error_in_the_aerosol_archive_is_a_fault_too(tmp_path) -> None:
    transport = combined_transport()

    class Flaky503:
        def exists(self, url):
            return transport.exists(url)

        def get(self, url, byte_range=None):
            if "gefs" in url:
                raise FetchError(f"{url}: HTTP 503 Slow Down")
            return transport.get(url, byte_range)

    with pytest.raises(TransientError, match="503"):
        collector(tmp_path, Flaky503()).collect(EVENING)


# --- the resumable run -------------------------------------------------------------------------


def multi_transport(days: list[dt.date], skip: tuple[dt.date, ...] = ()):
    files: dict[str, bytes] = {}
    for day in days:
        if day not in skip:
            files.update(combined_transport(day).files)
    return hrrr_support.FakeTransport(files)


def test_a_range_is_collected_saved_and_summarised(tmp_path) -> None:
    days = dates(EVENING, 3)
    store = RecordStore(tmp_path / "records")
    transport = multi_transport(days, skip=(days[1],))
    summary = collect_range(days, store, collector(tmp_path, transport))
    assert (summary.collected, summary.dropped, summary.skipped, summary.faults) == (2, 1, 0, 0)
    assert store.dates() == set(days)
    assert store.load(days[1]).drop_reason == "hrrr_run_missing"  # type: ignore[union-attr]
    assert store.load(days[0]).ok  # type: ignore[union-attr]


def test_a_rerun_skips_dates_already_in_the_store_without_a_request(tmp_path) -> None:
    days = dates(EVENING, 2)
    store = RecordStore(tmp_path / "records")
    collect_range(days, store, collector(tmp_path, multi_transport(days)))
    again = multi_transport(days)
    summary = collect_range(days, store, collector(tmp_path, again))
    assert (summary.collected, summary.skipped) == (0, 2)
    assert again.calls == []


def test_a_fault_leaves_the_date_unrecorded_and_the_run_goes_on(tmp_path) -> None:
    days = dates(EVENING, 2)
    store = RecordStore(tmp_path / "records")
    good = multi_transport(days)
    sick_day = days[0]

    class Sick:
        def exists(self, url):
            return good.exists(url)

        def get(self, url, byte_range=None):
            plan = hrrr.plan_run(
                sick_day, LAT, LON, cycle_hour=None, forecast_hour=None, now_utc=NOW, transport=None
            )
            if f"hrrr.{plan.cycle_date:%Y%m%d}" in url and f"t{plan.cycle_hour:02d}z" in url:
                raise FetchError(f"{url}: timed out")
            return good.get(url, byte_range)

    summary = collect_range(days, store, collector(tmp_path, Sick()))
    assert (summary.collected, summary.faults) == (1, 1)
    assert store.dates() == {days[1]}
    assert summary.unrecorded == (sick_day,)


def test_progress_is_reported_per_date_without_a_clock_time(tmp_path) -> None:
    days = dates(EVENING, 2)
    lines: list[str] = []
    collect_range(
        days,
        RecordStore(tmp_path / "records"),
        collector(tmp_path, multi_transport(days)),
        progress=lines.append,
    )
    assert len(lines) == 2
    assert lines[0].startswith(days[0].isoformat()) and "rendered" in lines[0]


def test_a_time_budget_stops_the_run_between_dates(tmp_path) -> None:
    days = dates(EVENING, 4)
    store = RecordStore(tmp_path / "records")
    ticks = iter([0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    summary = collect_range(
        days,
        store,
        collector(tmp_path, multi_transport(days)),
        budget_seconds=50.0,
        clock=lambda: next(ticks),
    )
    assert summary.stopped_early
    assert summary.collected < 4
    assert len(store.dates()) == summary.collected
    assert summary.remaining == tuple(d for d in days if d not in store.dates())


def test_a_run_stops_when_the_budget_is_exactly_used_up(tmp_path) -> None:
    days = dates(EVENING, 3)
    ticks = itertools.chain([0.0], itertools.repeat(50.0))  # the first check finds 50 s gone
    summary = collect_range(
        days,
        RecordStore(tmp_path / "records"),
        collector(tmp_path, multi_transport(days)),
        budget_seconds=50.0,
        clock=lambda: next(ticks),
    )
    assert summary.stopped_early
    assert (summary.collected, summary.remaining) == (0, tuple(days))


def test_a_5xx_for_a_run_of_4_april_is_a_fault_not_a_missing_archive_file(tmp_path) -> None:
    # The aerosol run for the evening of 3 April is the cycle of 4 April, so 404 is in its URL.
    evening = dt.date(2026, 4, 3)
    transport = combined_transport(evening)
    assert "404" in gefs.chem_url(gefs.aerosol_run(evening, LAT, LON).cycle_date, 0, 0)

    class Flaky503:
        def exists(self, url):
            return transport.exists(url)

        def get(self, url, byte_range=None):
            if "gefs" in url:
                raise FetchError(f"{url}: HTTP 503 Slow Down")
            return transport.get(url, byte_range)

    with pytest.raises(TransientError, match="503"):
        collector(tmp_path, Flaky503()).collect(evening)


def test_atmosphere_module_is_the_one_the_collector_reduces_with() -> None:
    assert collect_module.reduce_transect is atmosphere_module.reduce_transect
